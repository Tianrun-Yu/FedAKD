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

##################################################################
#   配置：路径、超参数、模型定义
##################################################################
TRAIN_PATH = '/home/tvy5242/EHR_fl/A_Experiment/DATA/train'
RESULT_DIR = '/home/tvy5242/EHR_fl/A_Experiment/METHOD/time_cost'

POW_PATH   = os.path.join(TRAIN_PATH, 'POW')
ECS_PATH   = os.path.join(TRAIN_PATH, 'ECS')
PECS_PATH  = os.path.join(TRAIN_PATH, 'PECS')

NUM_CLIENTS      = 10      # 客户端数量
NUM_GLOBAL_ROUNDS= 20      # 全局迭代轮数
LOCAL_EPOCHS     = 1       # 每轮客户端本地训练 epoch 数
BATCH_SIZE       = 32      # 批次大小

# 学习率：不同数据集使用不同学习率
learning_rates = {
    "FashionMNIST": 0.001,
    "CIFAR10":      0.005
}

# CGSV 相关配置
ALPHA_R  = 0.9   # 移动平均平滑系数(用于更新 r_i)
BETA     = 2.0   # 对梯度余弦相似度进行放缩/平滑时可用(可选)
SPARSITY = True  # 是否在下发梯度时做稀疏化
ALTRUISM = 1.0   # tanh( ALTRUISM * r_i ) 中的系数；若 ALTRUISM 很大 => 更不区分

##################################################################
# 简单 CNN 模型(10分类)
##################################################################
class SimpleCNN(nn.Module):
    def __init__(self, in_channels, img_size, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2, 2)
        # 经过两次池化，H,W 均缩小4倍
        feature_h = img_size[0] // 4
        feature_w = img_size[1] // 4
        self.fc   = nn.Linear(64 * feature_h * feature_w, num_classes)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def get_model(in_channels, img_size, num_classes=10):
    return SimpleCNN(in_channels, img_size, num_classes)

##################################################################
# 数据加载与预处理
##################################################################
def load_client_data_labels(base_dir, client_id):
    """
    加载 client_{id} 对应的 data.npy 和 labels.npy
    """
    client_dir = os.path.join(base_dir, f'client_{client_id}')
    data_path  = os.path.join(client_dir, 'data.npy')
    label_path = os.path.join(client_dir, 'labels.npy')
    if os.path.exists(data_path) and os.path.exists(label_path):
        data   = np.load(data_path)
        labels = np.load(label_path)
        return data, labels
    else:
        return None, None

def preprocess_data(data):
    """
    (N,H,W) -> (N,1,H,W)；(N,H,W,3) -> (N,3,H,W)
    """
    if data.ndim == 3:
        data = data[:, None, :, :]
    elif data.ndim == 4 and data.shape[-1] == 3:
        data = np.transpose(data, (0, 3, 1, 2))
    return data.astype(np.float32)

def create_local_loaders(data, labels, batch_size):
    """
    将 data, labels 构建成一个 DataLoader，用于本地训练或简单测试
    """
    t_x = torch.tensor(data, dtype=torch.float32)
    t_y = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(t_x, t_y)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

##################################################################
# 构建公共验证集
##################################################################
def build_global_validation(base_dir, ratio=0.1):
    """
    从每个客户端抽取 ratio 比例的数据组合成公共验证集
    """
    all_data = []
    all_labels = []
    for cid in range(NUM_CLIENTS):
        data, labels = load_client_data_labels(base_dir, cid)
        if data is not None:
            n_ = int(len(data)*ratio)
            if n_ > 0:
                all_data.append(data[:n_])
                all_labels.append(labels[:n_])
    if len(all_data) == 0:
        # 若无数据，返回一个空 loader
        dummy_data  = np.zeros((1, 28, 28), dtype=np.float32)
        dummy_label = np.zeros((1,), dtype=np.int64)
        dummy_data  = preprocess_data(dummy_data)
        dataset = TensorDataset(torch.tensor(dummy_data), torch.tensor(dummy_label))
        return DataLoader(dataset, batch_size=64, shuffle=False)
    data_big   = np.concatenate(all_data, axis=0)
    label_big  = np.concatenate(all_labels, axis=0)
    data_big   = preprocess_data(data_big)
    t_x = torch.tensor(data_big, dtype=torch.float32)
    t_y = torch.tensor(label_big, dtype=torch.long)
    ds  = TensorDataset(t_x, t_y)
    return DataLoader(ds, batch_size=64, shuffle=False)

##################################################################
#  训练、梯度提取、余弦相似度
##################################################################
def local_train_and_get_grad(global_model_state, loader, device, local_epochs, lr):
    """
    在 loader 上训练 local_epochs 次，返回本地梯度 vector 和本地新模型状态
    """
    model = get_model_from_state(global_model_state).to(device)
    
    # 备份原参数
    old_params = {}
    for k, v in model.state_dict().items():
        old_params[k] = v.clone().detach()
    
    # 常规训练
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for _ in range(local_epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss   = criterion(logits, by)
            loss.backward()
            optimizer.step()
    
    # 训练后的参数
    new_params = model.state_dict()
    # 计算 local_grad = new_params - old_params
    local_grad = {}
    for k in new_params:
        local_grad[k] = (new_params[k] - old_params[k]).detach().clone()
    # 返回 (local_grad, new_params)
    return local_grad, dict(new_params)

def flatten_grad_dict(grad_dict):
    """
    将 {k: tensor} 格式展平为 1D 向量
    """
    vecs = []
    for k in sorted(grad_dict.keys()):
        vecs.append(grad_dict[k].view(-1))
    return torch.cat(vecs)

def unflatten_to_dict(vec, ref_dict):
    """
    将 1D 向量根据 ref_dict 的结构拆成 {k: tensor}
    """
    result = {}
    idx = 0
    for k in sorted(ref_dict.keys()):
        numel = ref_dict[k].numel()
        shape_ = ref_dict[k].shape
        chunk = vec[idx : idx + numel]
        idx += numel
        result[k] = chunk.view(shape_)
    return result

def cos_sim(a, b):
    """
    计算 a, b (1D 向量) 的余弦相似度
    """
    eps = 1e-9
    na = torch.norm(a)
    nb = torch.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(torch.dot(a, b) / (na * nb))

def get_model_from_state(state_dict):
    """
    根据 state_dict 中第一层卷积权重判断通道数，从而实例化对应模型。
    这里简单判断：c_in==1 => (28,28)；c_in==3 => (32,32)。
    """
    conv1_w = state_dict['conv1.weight']
    c_in = conv1_w.shape[1]
    if c_in == 1:
        model = SimpleCNN(1, (28, 28))
    else:
        model = SimpleCNN(3, (32, 32))
    model.load_state_dict(state_dict, strict=False)
    return model

##################################################################
#  聚合 & 下发
##################################################################
def combine_grads(grad_list, weights=None):
    """
    等权或加权聚合
    grad_list: list of 1D tensor
    weights: list of scalar
    """
    if weights is None:
        weights = [1.0] * len(grad_list)
    total_w = sum(weights)
    if total_w <= 0:
        return torch.zeros_like(grad_list[0])
    combined = torch.zeros_like(grad_list[0])
    for g, w in zip(grad_list, weights):
        combined += g * w
    combined /= total_w
    return combined

def mask_top_k(vec, k):
    """
    对 vec (1D) 保留绝对值最大的 k 个元素，其余置 0
    """
    if k <= 0:
        return torch.zeros_like(vec)
    if k >= vec.numel():
        return vec
    abs_vec = torch.abs(vec)
    threshold = torch.topk(abs_vec, k)[0][-1]
    mask = (abs_vec >= threshold)
    masked_vec = vec.clone()
    masked_vec[~mask] = 0.0
    return masked_vec

##################################################################
#  CGSV 训练主体 (增加计时)
##################################################################
def cgsv_training(base_dir, lr, device):
    """
    返回 (NUM_GLOBAL_ROUNDS, NUM_CLIENTS) 形状的测试准确率，以及各客户端的样本数列表 sample_counts
    """
    # 读取所有客户端数据 => 构建 local loader
    local_loaders = [None] * NUM_CLIENTS
    valid_clients = [False] * NUM_CLIENTS
    input_shapes  = [None] * NUM_CLIENTS
    for cid in range(NUM_CLIENTS):
        data, labels = load_client_data_labels(base_dir, cid)
        if data is None:
            continue
        data_ = preprocess_data(data)
        loader_ = create_local_loaders(data_, labels, BATCH_SIZE)
        local_loaders[cid] = loader_
        valid_clients[cid] = True

        c = data_.shape[1]
        h = data_.shape[2]
        w = data_.shape[3]
        input_shapes[cid] = (c, h, w)
    
    if not any(valid_clients):
        return np.zeros((NUM_GLOBAL_ROUNDS, NUM_CLIENTS)), [0] * NUM_CLIENTS

    first_cid = valid_clients.index(True)
    c, h, w = input_shapes[first_cid]
    
    # 全局模型初始化
    global_model = get_model(c, (h, w)).to(device)
    global_model_state = dict(global_model.state_dict())

    # (可选)构建公共验证集
    public_val_loader = build_global_validation(base_dir, ratio=0.1)

    # 每个客户端的 重要性系数 r_i，初始均匀
    r_i = np.ones(NUM_CLIENTS, dtype=np.float32)
    r_i /= np.sum(r_i)

    # 记录准确率矩阵
    acc_matrix = np.zeros((NUM_GLOBAL_ROUNDS, NUM_CLIENTS))

    # 获取“参数总维度”
    test_grad, _ = local_train_and_get_grad(global_model_state, local_loaders[first_cid], device, 1, lr)
    flat_ = flatten_grad_dict(test_grad)
    big_dim = flat_.numel()

    for round_idx in range(NUM_GLOBAL_ROUNDS):
        # ============== 计时开始 ==============
        start_time = time.time()

        # ========== 上传阶段 ================
        grads_all = []
        client_ids = []
        local_params_dicts = {}
        for cid in range(NUM_CLIENTS):
            if not valid_clients[cid]:
                continue
            # 本地训练 => 获取 local_grad
            local_grad_dict, local_new_params = local_train_and_get_grad(
                global_model_state, local_loaders[cid], device, LOCAL_EPOCHS, lr
            )
            local_params_dicts[cid] = local_new_params
            # flatten 处理梯度
            flat_grad = flatten_grad_dict(local_grad_dict).to(device)
            grads_all.append(flat_grad)
            client_ids.append(cid)
        
        if len(grads_all) == 0:
            break

        # 计算全局梯度(等权)
        combined_grad = combine_grads(grads_all, None)
        # 归一化 => 计算 u_N
        gN_norm = torch.norm(combined_grad) + 1e-9
        uN = combined_grad / gN_norm
        
        # ========== 计算 CGSV(近似) =============
        # 计算各客户端梯度与 uN 的余弦相似度
        cos_vals = []
        for i, g in enumerate(grads_all):
            ni = torch.norm(g) + 1e-9
            ui = g / ni
            cos_ij = float(torch.dot(ui, uN))
            cos_vals.append(max(0.0, cos_ij))  # 小于 0 则置 0
        # 更新 r_i（移动平均更新）
        for i, cid in enumerate(client_ids):
            old = r_i[cid]
            new = ALPHA_R * old + (1 - ALPHA_R) * cos_vals[i]
            r_i[cid] = new
        # 归一化
        sumr = np.sum(r_i)
        if sumr > 0:
            r_i /= sumr

        # ========== 下载阶段(差异化稀疏) =============
        # 更新全局模型：wg' = wg + combined_grad
        new_global_vec = flatten_grad_dict(global_model_state).to(device)
        new_global_vec += combined_grad
        new_global_dict = unflatten_to_dict(new_global_vec, global_model_state)
        global_model_state = new_global_dict

        # 针对每个客户端下发更新
        if SPARSITY:
            tvals = [np.tanh(ALTRUISM * r) for r in r_i]
            tmax  = max(tvals)
        for i, cid in enumerate(client_ids):
            local_vec = flatten_grad_dict(local_params_dicts[cid]).to(device)
            if SPARSITY:
                tv = np.tanh(ALTRUISM * r_i[cid])
                ratio_ = tv / (tmax + 1e-9)
                keep_k = int(big_dim * ratio_)
                masked = mask_top_k(combined_grad, keep_k)
                local_vec += masked
            else:
                local_vec += combined_grad
            final_local_dict = unflatten_to_dict(local_vec, local_params_dicts[cid])
            local_params_dicts[cid] = final_local_dict
        
        # =========== 在每个客户端测试准确率 =================
        for cid in range(NUM_CLIENTS):
            if not valid_clients[cid]:
                acc_matrix[round_idx, cid] = 0.0
            else:
                tmp_model = get_model_from_state(local_params_dicts[cid]).to(device)
                acc = local_evaluate(tmp_model, local_loaders[cid], device)
                acc_matrix[round_idx, cid] = acc

        # ============== 计时结束并打印 ==============
        end_time = time.time()
        round_time = end_time - start_time
        print(f"Round {round_idx+1}/{NUM_GLOBAL_ROUNDS} took {round_time:.4f} seconds.")

    # 计算每个客户端的样本数（用于加权平均）
    sample_counts = [0] * NUM_CLIENTS
    for cid in range(NUM_CLIENTS):
        if valid_clients[cid]:
            sample_counts[cid] = len(local_loaders[cid].dataset)
    return acc_matrix, sample_counts

def local_evaluate(model, loader, device):
    """
    简单评估：计算 accuracy
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            _, pred = torch.max(logits, 1)
            total += by.size(0)
            correct += (pred == by).sum().item()
    return correct / total if total > 0 else 0

##################################################################
#  与独自训练对比 => 统计 last3 平均/最大准确率 => 写文件
##################################################################
def load_standalone_accuracies(dataset_name, dist_name, num_epochs=NUM_GLOBAL_ROUNDS):
    """
    演示：返回随机数作为独自训练准确率
    你可替换为实际独立训练的准确率数据
    """
    rng = np.random.RandomState(hash(dataset_name + dist_name) & 0xffff)
    return rng.rand(num_epochs, NUM_CLIENTS)

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    distributions = {
        'POW':      lambda ds: os.path.join(POW_PATH, ds),
        'ECS_C5':   lambda ds: os.path.join(ECS_PATH, 'C5', ds),
        'PECS_C2':  lambda ds: os.path.join(PECS_PATH, 'C2', ds),
        'PECS_C5':  lambda ds: os.path.join(PECS_PATH, 'C5', ds),
        'PECS_C10': lambda ds: os.path.join(PECS_PATH, 'C10', ds)
    }
    
    datasets_list = ['FashionMNIST', 'CIFAR10']

    results = {}

    for dataset_name in datasets_list:
        lr = learning_rates.get(dataset_name, 0.001)
        print(f"\n===== CGSV on {dataset_name} (lr={lr}) =====")
        
        for dist_name, base_dir_fn in distributions.items():
            base_dir = base_dir_fn(dataset_name)
            print(f"--- Distribution: {dist_name} ---")

            # (1) 运行 CGSV 训练流程，返回准确率矩阵和各客户端样本数
            acc_matrix, sample_counts = cgsv_training(base_dir, lr, device)
            # acc_matrix.shape = (NUM_GLOBAL_ROUNDS, NUM_CLIENTS)

            # (2) 统计最后3轮，计算加权平均准确率（按各客户端样本数加权）
            sample_counts_np = np.array(sample_counts)
            weighted_mean_each_round = np.array([
                np.sum(acc_matrix[r, :] * sample_counts_np) / np.sum(sample_counts_np)
                for r in range(NUM_GLOBAL_ROUNDS)
            ])
            max_each_round = np.max(acc_matrix, axis=1)

            last3_mean = weighted_mean_each_round[-3:]
            avg3_mean  = np.mean(last3_mean)
            std3_mean  = np.std(last3_mean)

            last3_max = max_each_round[-3:]
            avg3_max  = np.mean(last3_max)
            std3_max  = np.std(last3_max)

            # (3) 与独自训练做相关系数（纯演示）
            standalone_matrix = load_standalone_accuracies(dataset_name, dist_name, NUM_GLOBAL_ROUNDS)
            corr_vals = []
            for r_ in range(NUM_GLOBAL_ROUNDS - 3, NUM_GLOBAL_ROUNDS):
                cffl_accs = acc_matrix[r_, :]
                stand_accs = standalone_matrix[r_, :]
                if np.all(cffl_accs == 0) or np.all(stand_accs == 0):
                    corr = 0.0
                else:
                    corr = np.corrcoef(cffl_accs, stand_accs)[0, 1]
                corr_vals.append(corr * 100.0)
            corr_vals = np.array(corr_vals)
            avg_corr = corr_vals.mean()
            std_corr = corr_vals.std()

            results[(dataset_name, dist_name)] = (
                (avg3_mean, std3_mean),
                (avg3_max,  std3_max),
                (avg_corr,  std_corr)
            )
    
    final_file = os.path.join(RESULT_DIR, "final_results.txt")
    with open(final_file, "w") as f:
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
    print(f"CGSV 结果已保存: {final_file}")

def local_evaluate(model, loader, device):
    """
    简单评估：计算 accuracy
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            _, pred = torch.max(logits, 1)
            total += by.size(0)
            correct += (pred == by).sum().item()
    return correct / total if total > 0 else 0

if __name__ == '__main__':
    main()
