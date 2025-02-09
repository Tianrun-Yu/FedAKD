#!/usr/bin/env python3
import os
import time  # <=== 新增：导入 time 模块
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

############################################################
#                  CFFL 配置超参数
############################################################
THETA_U  = 0.5     # 部分上传比例(0,1]，示例中取 0.5，表示上传 50% 最重要的梯度分量
CLIP_NORM= 5.0     # 梯度裁剪阈值
C_TH     = 0.05    # 信誉度阈值(低于此值的客户端将被剔除)
ALPHA    = 1.0     # 信誉度更新时, sinh( α * normalized_acc )
LOCAL_EPOCHS = 1   # 每轮本地训练的轮数
NUM_GLOBAL_ROUNDS = 20  # 全局迭代轮数
BATCH_SIZE   = 32
NUM_CLIENTS  = 10

############################################################
#                 路径等常量
############################################################
TRAIN_PATH = '/home/tvy5242/EHR_fl/A_Experiment/DATA/train'
RESULT_DIR = '/home/tvy5242/EHR_fl/A_Experiment/METHOD/time_cost'

POW_PATH   = os.path.join(TRAIN_PATH, 'POW')
ECS_PATH   = os.path.join(TRAIN_PATH, 'ECS')
PECS_PATH  = os.path.join(TRAIN_PATH, 'PECS')

# 定义不同数据集使用的学习率 (可自行调整)
learning_rates = {
    "FashionMNIST": 0.001,
    "CIFAR10": 0.005
}

############################################################
#                 模型定义：SimpleCNN
############################################################
class SimpleCNN(nn.Module):
    def __init__(self, in_channels, img_size, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2, 2)
        feature_h = img_size[0] // 4
        feature_w = img_size[1] // 4
        self.fc    = nn.Linear(64 * feature_h * feature_w, num_classes)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def get_model(in_channels, img_size, num_classes=10):
    return SimpleCNN(in_channels, img_size, num_classes)

############################################################
#            数据加载 & 预处理
############################################################
def load_client_data_labels(base_dir, client_id):
    """
    读取指定客户端 (data.npy, labels.npy)
    """
    client_dir  = os.path.join(base_dir, f'client_{client_id}')
    data_path   = os.path.join(client_dir, 'data.npy')
    labels_path = os.path.join(client_dir, 'labels.npy')
    if os.path.exists(data_path) and os.path.exists(labels_path):
        data   = np.load(data_path)
        labels = np.load(labels_path)
        return data, labels
    else:
        return None, None

def preprocess_data(data):
    """
    将 (N,H,W) => (N,1,H,W) 或 (N,H,W,3) => (N,3,H,W)
    """
    if len(data.shape) == 3:
        data = data[:, None, :, :]
    elif len(data.shape) == 4 and data.shape[-1] == 3:
        data = np.transpose(data, (0, 3, 1, 2))
    return data.astype(np.float32)

def create_data_loaders(data, labels, batch_size):
    """
    这里不分本地 train/test。CFFL中我们假设已经由外部切好。
    实际上为了简化，可对client数据也做80/20划分，这里只做一个DataLoader即可。
    """
    tensor_x = torch.tensor(data, dtype=torch.float32)
    tensor_y = torch.tensor(labels, dtype=torch.long)
    dataset  = TensorDataset(tensor_x, tensor_y)
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

############################################################
#   CFFL 核心函数： partial_upload, clip, aggregator, ...
############################################################
def get_local_gradient(model, loader, device, local_epochs, lr):
    """
    在本地数据上训练 local_epochs 次，返回： 
      grads: model梯度 (最后一次反向传播产生)
      model_state: 训练完成后的模型参数(用于做 '本地更新')
    """
    # 复制一份初始权重
    old_params = {}
    for name, param in model.named_parameters():
        old_params[name] = param.clone().detach()

    # 常规训练
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(local_epochs):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss    = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # 计算 "delta w" = new_params - old_params
    grads = {}
    for name, param in model.named_parameters():
        grads[name] = (param.detach() - old_params[name]).clone().detach()

    # 返回更新后的模型参数 & grads
    return grads, dict(model.state_dict())

def clip_gradients(grads, clip_value=CLIP_NORM):
    """
    对梯度字典进行裁剪，使其 L2 norm 不超过 clip_value
    """
    # 先拼成一个大向量
    vec = []
    names = []
    for name in grads:
        vec.append(grads[name].view(-1))
        names.append(name)
    full_vec = torch.cat(vec)
    norm = torch.norm(full_vec)
    if norm > clip_value:
        scale = clip_value / norm
        full_vec = full_vec * scale
    
    # 拆回 grads
    idx = 0
    new_grads = {}
    for name in names:
        shape_ = grads[name].shape
        size_  = grads[name].numel()
        chunk  = full_vec[idx : idx+size_]
        idx   += size_
        new_grads[name] = chunk.view(shape_)
    return new_grads

def select_top_fraction(grads, fraction=THETA_U):
    """
    从 grads 中选出绝对值最大的 fraction 比例的元素，其他置为 0。
    """
    if fraction >= 1.0:
        return grads  # 不做截取
    # 拼成向量并排序
    vec = []
    names_shapes = []
    for name in grads:
        v_ = grads[name].view(-1)
        vec.append(v_)
        names_shapes.append((name, grads[name].shape))
    full_vec = torch.cat(vec)
    abs_vec  = torch.abs(full_vec)
    k = int(len(abs_vec) * fraction)
    if k <= 0:
        # fraction 非常小 => 全部置0
        new_grads = {}
        for name, shape_ in names_shapes:
            new_grads[name] = torch.zeros(shape_, dtype=torch.float32)
        return new_grads
    
    # 找到第 k 大的阈值
    threshold = torch.topk(abs_vec, k)[0][-1]
    # mask
    mask = (abs_vec >= threshold)
    masked_vec = full_vec.clone()
    masked_vec[~mask] = 0.0

    # 拆回 grads
    idx = 0
    new_grads = {}
    for (name, shape_) in names_shapes:
        size_ = np.prod(shape_)
        chunk = masked_vec[idx : idx+size_]
        idx   += size_
        new_grads[name] = chunk.view(shape_)
    return new_grads

def combine_gradients(grad_list, weights=None):
    """
    服务器端聚合：加权平均 (data大小,类别数,或信誉度等)
    grad_list: list of grads dict, each = {layer_name: tensor}
    weights: list of scalar for each grad
    """
    if weights is None:
        weights = [1.0]*len(grad_list)
    combined = {}
    all_keys = set()
    for gdict in grad_list:
        all_keys.update(gdict.keys())

    total_w = sum(weights)
    # 初始化
    for k in all_keys:
        combined[k] = 0.0

    # 加权求和
    for i, gdict in enumerate(grad_list):
        w = weights[i]
        for k in gdict:
            combined[k] = combined[k] + gdict[k].float() * w

    # 再做除以 total_w
    if total_w>0:
        for k in combined:
            combined[k] = combined[k] / total_w
    return combined

def subtract_part(original, to_sub, scale=1.0):
    """
    original, to_sub: 都是 dict[name, tensor]
    计算: original - scale * to_sub
    """
    result = {}
    for name in original:
        if name in to_sub:
            result[name] = original[name] - scale * to_sub[name]
        else:
            result[name] = original[name]  # 若 to_sub无此键,直接保持
    return result

def apply_grad_to_model(model_state, grad):
    """
    给定模型参数 dict, 以及更新(梯度) dict，返回新的参数 dict
    w' = w + grad
    """
    new_dict = {}
    for k in model_state:
        if k in grad:
            new_dict[k] = model_state[k] + grad[k]
        else:
            # 若 grad无此key(例如 running_mean), 保持原值
            new_dict[k] = model_state[k]
    return new_dict

############################################################
#  信誉度管理
############################################################
def init_reputation(num_clients):
    # 初始均匀
    return np.ones(num_clients) / num_clients

def update_reputation(reputation, val_accs):
    """
    传入每个客户端的验证准确率 val_accs ( shape=(num_clients,) )。
    先做归一化 => sum(val_accs)，再进 sinh( alpha * ratio )，
    然后跟旧 reputation 0.5:0.5 融合 => 重新归一化 => 剔除低于 C_TH => 再归一化
    """
    sum_acc = np.sum(val_accs)
    new_rep = reputation.copy()
    if sum_acc>0:
        ratio = val_accs / sum_acc
        new_c = np.sinh(ALPHA * ratio)
        new_c = new_c / (np.sum(new_c) + 1e-9)
        new_rep = 0.5 * reputation + 0.5 * new_c
        new_rep = new_rep / (np.sum(new_rep)+1e-9)

    # 剔除低于门槛
    to_remove = (new_rep < C_TH)
    if np.any(to_remove):
        new_rep[to_remove] = 0.0
        s_ = np.sum(new_rep)
        if s_>0:
            new_rep = new_rep / s_
    return new_rep

############################################################
#   验证集评估
############################################################
def evaluate_on_public(model_state, public_loader, device, in_channels, img_size):
    """
    在公共验证集上评估给定模型参数 model_state 的准确率
    """
    model = get_model(in_channels, img_size).to(device)
    model.load_state_dict(model_state, strict=False)
    model.eval()

    correct=0
    total=0
    with torch.no_grad():
        for bx, by in public_loader:
            bx, by = bx.to(device), by.to(device)
            outputs = model(bx)
            _, pred = torch.max(outputs, 1)
            total += by.size(0)
            correct += (pred==by).sum().item()
    return correct/total if total>0 else 0.0

############################################################
#  构建公共验证集
############################################################
def build_public_validation(distribution_path, device):
    """
    简单地从每个客户端数据中抽取10% (或者一个固定数量) 作为公共验证集。
    """
    all_data = []
    all_labels = []
    for cid in range(NUM_CLIENTS):
        d_, l_ = load_client_data_labels(distribution_path, cid)
        if d_ is not None:
            n_ = int(len(d_)*0.1)  # 10%
            if n_>0:
                data_part   = d_[:n_]
                labels_part = l_[:n_]
                all_data.append(data_part)
                all_labels.append(labels_part)
    if len(all_data)>0:
        data_big   = np.concatenate(all_data, axis=0)
        labels_big = np.concatenate(all_labels, axis=0)
    else:
        data_big   = np.zeros((1,28,28), dtype=np.float32)
        labels_big = np.zeros((1,), dtype=np.int64)
    
    data_big = preprocess_data(data_big)
    dataset  = TensorDataset(torch.tensor(data_big), torch.tensor(labels_big))
    loader   = DataLoader(dataset, batch_size=64, shuffle=False)
    return loader

############################################################
#   本地评估
############################################################
def local_evaluate_for_cffl(model, data_loader, device):
    """
    在本例中，CFFL 只保存了(全部) loader。这里简化地把 loader 当测试集。
    """
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for bx, by in data_loader:
            bx,by = bx.to(device), by.to(device)
            pred = model(bx)
            _, predicted = torch.max(pred, 1)
            correct += (predicted==by).sum().item()
            total   += by.size(0)
    return correct/total if total>0 else 0

############################################################
#   CFFL 主循环 (新增计时逻辑)
############################################################
def cffl_training(base_dir, lr, device):
    """
    返回 shape=(NUM_GLOBAL_ROUNDS, NUM_CLIENTS) 的测试准确率矩阵
    """
    # (1) 加载所有客户端的数据 => 用于构建 DataLoader
    client_loaders = []
    valid_clients  = [False]*NUM_CLIENTS
    input_shapes   = [None]*NUM_CLIENTS

    for cid in range(NUM_CLIENTS):
        data, labels = load_client_data_labels(base_dir, cid)
        if data is None or labels is None:
            client_loaders.append(None)
            continue
        data_ = preprocess_data(data)
        loader_ = create_data_loaders(data_, labels, BATCH_SIZE)
        client_loaders.append(loader_)
        valid_clients[cid] = True
        if data_.ndim==4:
            _, C, H, W = data_.shape
            input_shapes[cid] = (C,H,W)

    if not any(valid_clients):
        return np.zeros((NUM_GLOBAL_ROUNDS, NUM_CLIENTS))
    
    first_valid = valid_clients.index(True)
    c,h,w = input_shapes[first_valid]
    
    # (A) 构建公共验证集
    public_loader = build_public_validation(base_dir, device)
    
    # (B) 初始化全局模型
    global_model = get_model(c, (h,w)).to(device)
    global_model_state = dict(global_model.state_dict())

    # (C) 初始化信誉
    reputation = init_reputation(NUM_CLIENTS)
    
    # (D) shape=(NUM_GLOBAL_ROUNDS, NUM_CLIENTS)
    acc_matrix = np.zeros((NUM_GLOBAL_ROUNDS, NUM_CLIENTS))

    # 记录 data_size 以进行加权 (示例)
    data_sizes = np.zeros(NUM_CLIENTS, dtype=np.float32)
    for cid in range(NUM_CLIENTS):
        d_, l_ = load_client_data_labels(base_dir, cid)
        if d_ is not None:
            data_sizes[cid] = len(d_)

    # 这里用于保存每个客户端的本地模型参数
    local_models = {}
    for cid in range(NUM_CLIENTS):
        if valid_clients[cid] and data_sizes[cid] > 0:
            local_models[cid] = global_model_state

    for round_idx in range(NUM_GLOBAL_ROUNDS):
        # ============== 新增：开始计时 ==============
        start_time = time.time()

        # 收集上传的梯度 (partial)
        grads_list = []
        client_ids = []
        
        # 参与者集 R：信誉>0 的参与者
        R = [i for i in range(NUM_CLIENTS) if valid_clients[i] and reputation[i]>0]
        if len(R) == 0:
            # 全部被踢出
            break
        
        # (1) 各客户端本地训练 -> 得到 "完整梯度" + 本地模型
        for cid in R:
            # 用 global_model_state 初始化模型
            local_model = get_model(c,(h,w)).to(device)
            local_model.load_state_dict(global_model_state, strict=False)
            
            # local train => grads
            grads, local_state = get_local_gradient(local_model, client_loaders[cid], device, LOCAL_EPOCHS, lr)
            # clip
            grads_clipped = clip_gradients(grads, CLIP_NORM)
            # partial upload
            grads_partial = select_top_fraction(grads_clipped, THETA_U)
            
            grads_list.append(grads_partial)
            client_ids.append(cid)
            # 暂存“本地训练后”的模型
            local_models[cid] = local_state

        # (2) 服务器端聚合 (data_size加权)
        sum_data = np.sum(data_sizes[client_ids])
        weights_ = []
        for cid in client_ids:
            if sum_data > 0:
                w_ = data_sizes[cid] / sum_data
            else:
                w_ = 1.0
            weights_.append(w_)
        combined_grad = combine_gradients(grads_list, weights_)

        # (3) 服务器更新 "临时全局模型" => wg' = wg + combined_grad
        temp_global_model = get_model(c,(h,w)).to(device)
        temp_global_model.load_state_dict(global_model_state, strict=False)
        new_global_state = apply_grad_to_model(dict(temp_global_model.state_dict()), combined_grad)

        # (4) 服务器计算每个客户端在公共验证集上的贡献 => val_acc
        val_accs = np.zeros(NUM_CLIENTS, dtype=np.float32)
        for cid in R:
            # 这里演示: w_j' = wg' + partial_grad_j
            partial_g = grads_list[client_ids.index(cid)]
            model_state_j = apply_grad_to_model(new_global_state, partial_g)
            val_accs[cid] = evaluate_on_public(model_state_j, public_loader, device, c, (h, w))

        # (5) 更新信誉
        old_rep = reputation.copy()
        reputation = update_reputation(old_rep, val_accs)

        # (6) 服务器对 each client 分配更新(简化)
        max_data = np.max(data_sizes[client_ids]) if len(client_ids) > 0 else 1
        final_grads_to_client = []
        for i, cid in enumerate(client_ids):
            sub_ = data_sizes[cid] / (max_data + 1e-9)
            final_grads_j = subtract_part(combined_grad, grads_list[i], scale=sub_)
            final_grads_to_client.append(final_grads_j)

        # (7) 客户端本地融合: w_j' = local_models[cid] + grads_j + final_grads_j
        for i, cid in enumerate(client_ids):
            g_j = grads_list[i]
            tmp = apply_grad_to_model(local_models[cid], g_j)
            new_ = apply_grad_to_model(tmp, final_grads_to_client[i])
            local_models[cid] = new_

        # (8) 服务器更新 "真正全局模型" = new_global_state
        global_model_state = new_global_state

        # (8.1) 确保所有 valid 且有数据的客户端，至少有一个本地模型
        for cid_ in range(NUM_CLIENTS):
            if valid_clients[cid_] and data_sizes[cid_] > 0:
                if cid_ not in local_models:
                    local_models[cid_] = global_model_state
        
        # (9) 在每个客户端的本地测试集上评估准确率
        for cid in range(NUM_CLIENTS):
            if not valid_clients[cid] or data_sizes[cid] == 0:
                acc_matrix[round_idx, cid] = 0.0
            else:
                m_ = get_model(c,(h,w)).to(device)
                m_.load_state_dict(local_models[cid], strict=False)
                test_acc = local_evaluate_for_cffl(m_, client_loaders[cid], device)
                acc_matrix[round_idx, cid] = test_acc

        # ============== 新增：结束计时并打印 ==============
        end_time = time.time()
        round_time = end_time - start_time
        print(f"Round {round_idx+1}/{NUM_GLOBAL_ROUNDS} took {round_time:.4f} seconds.")

    return acc_matrix

############################################################
#   与“独自训练”对比 => 计算相关系数 => 输出
############################################################
def load_standalone_accuracies(dataset_name, dist_name, num_epochs=NUM_GLOBAL_ROUNDS):
    """
    用随机数演示独自训练的准确率。可根据实际情况替换成真实数据。
    """
    rng = np.random.RandomState(hash(dataset_name + dist_name) & 0xffff)
    return rng.rand(num_epochs, NUM_CLIENTS)

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    distributions = {
        'POW':      lambda ds: os.path.join(POW_PATH, ds),
        'ECS_C5':   lambda ds: os.path.join(ECS_PATH, 'C5', ds),
        'PECS_C2':  lambda ds: os.path.join(PECS_PATH, 'C2', ds),
        'PECS_C5':  lambda ds: os.path.join(PECS_PATH, 'C5', ds),
        'PECS_C10': lambda ds: os.path.join(PECS_PATH, 'C10', ds)
    }
    
    datasets_list = ['FashionMNIST','CIFAR10']

    results = {}

    for dataset_name in datasets_list:
        lr = learning_rates.get(dataset_name, 0.001)
        print(f"\n====== CFFL on {dataset_name} (lr={lr}) ======")
        
        for dist_name, base_dir_fn in distributions.items():
            base_dir = base_dir_fn(dataset_name)
            print(f"--- Distribution: {dist_name} ---")

            # (1) CFFL 训练 => acc_matrix = (NUM_GLOBAL_ROUNDS, NUM_CLIENTS)
            acc_matrix = cffl_training(base_dir, lr, device)

            # (2) 统计最后3轮
            mean_acc_each_round = np.mean(acc_matrix, axis=1) # (R,)
            max_acc_each_round  = np.max(acc_matrix, axis=1)  # (R,)

            last3_mean_acc = mean_acc_each_round[-3:]
            avg3_mean_acc  = np.mean(last3_mean_acc)
            std3_mean_acc  = np.std(last3_mean_acc)

            last3_max_acc = max_acc_each_round[-3:]
            avg3_max_acc  = np.mean(last3_max_acc)
            std3_max_acc  = np.std(last3_max_acc)

            # (3) 计算与独自训练的相关系数 (演示)
            standalone_matrix = load_standalone_accuracies(dataset_name, dist_name, NUM_GLOBAL_ROUNDS)
            corr_vals = []
            for r_ in range(NUM_GLOBAL_ROUNDS-3, NUM_GLOBAL_ROUNDS):
                cffl_accs = acc_matrix[r_, :]
                stand_accs= standalone_matrix[r_, :]
                if np.all(cffl_accs==0) or np.all(stand_accs==0):
                    corr=0.0
                else:
                    corr = np.corrcoef(cffl_accs, stand_accs)[0,1]
                corr_vals.append(corr*100)
            corr_vals = np.array(corr_vals)
            avg_corr = corr_vals.mean()
            std_corr = corr_vals.std()

            results[(dataset_name, dist_name)] = (
                (avg3_mean_acc, std3_mean_acc),
                (avg3_max_acc,  std3_max_acc),
                (avg_corr,      std_corr)
            )

    # (4) 输出到 final_results.txt
    final_result_file = os.path.join(RESULT_DIR, "final_results.txt")
    with open(final_result_file, "w") as f:
        f.write("dataset, distribution, last3_AvgAcc(mean±std), last3_MaxAcc(mean±std), last3_Corr(mean±std)\n")
        for dataset_name in datasets_list:
            for dist_name in distributions.keys():
                val = results.get((dataset_name, dist_name), None)
                if val is None:
                    f.write(f"{dataset_name}, {dist_name}, 0±0, 0±0, 0±0\n")
                else:
                    (m1, s1), (m2, s2), (mc, sc) = val
                    line = (f"{dataset_name}, {dist_name}, "
                            f"{m1:.4f}±{s1:.4f}, "
                            f"{m2:.4f}±{s2:.4f}, "
                            f"{mc:.4f}±{sc:.4f}\n")
                    f.write(line)
    print(f"\n>>> CFFL 结果已保存至: {final_result_file}")

if __name__ == '__main__':
    main()
