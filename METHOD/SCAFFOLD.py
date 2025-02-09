#!/usr/bin/env python3
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

############################################################################
# SCAFFOLD 配置
############################################################################
RESULT_DIR = '/home/tvy5242/EHR_fl/A_Experiment/RESULT/SCAFFOLD'
TRAIN_PATH = '/home/tvy5242/EHR_fl/A_Experiment/DATA/train'

POW_PATH   = os.path.join(TRAIN_PATH, 'POW')
ECS_PATH   = os.path.join(TRAIN_PATH, 'ECS')
PECS_PATH  = os.path.join(TRAIN_PATH, 'PECS')

NUM_CLIENTS       = 10
NUM_GLOBAL_ROUNDS = 20
LOCAL_EPOCHS      = 1      # 本地训练 epoch 数
BATCH_SIZE        = 32

# 本示例中: 全局学习率 eta_g, 本地学习率 eta_l
learning_rates = {
    'FashionMNIST': (0.001, 0.1),  # (eta_g, eta_l)
    'CIFAR10':      (0.005, 0.1)
}

############################################################################
# 模型: SimpleCNN
############################################################################
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
        self.fc   = nn.Linear(64 * feature_h * feature_w, num_classes)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def get_model(in_channels, img_size, num_classes=10):
    return SimpleCNN(in_channels, img_size, num_classes)

############################################################################
# 数据加载 & 预处理
############################################################################
def load_client_data_labels(base_dir, client_id):
    client_dir = os.path.join(base_dir, f'client_{client_id}')
    data_path  = os.path.join(client_dir, 'data.npy')
    label_path = os.path.join(client_dir, 'labels.npy')
    if os.path.exists(data_path) and os.path.exists(label_path):
        data   = np.load(data_path)
        labels = np.load(label_path)
        return data, labels
    return None, None

def preprocess_data(data):
    """
    (N,H,W)->(N,1,H,W) 或 (N,H,W,3)->(N,3,H,W)
    """
    if data.ndim == 3:
        data = data[:, None, :, :]
    elif data.ndim == 4 and data.shape[-1] == 3:
        data = np.transpose(data, (0,3,1,2))
    return data.astype(np.float32)

def create_data_loaders_scaffold(data, labels, batch_size):
    """
    简单做80/20划分 => train/test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True, random_state=42
    )
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test  = torch.tensor(X_test)
    y_test  = torch.tensor(y_test, dtype=torch.long)

    ds_train = TensorDataset(X_train, y_train)
    ds_test  = TensorDataset(X_test,  y_test)
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False)
    return loader_train, loader_test

############################################################################
# 工具: flatten/unflatten
############################################################################
def model_to_vec(model):
    vecs = []
    for param in model.parameters():
        vecs.append(param.view(-1))
    return torch.cat(vecs)

def vec_to_model(vec, model):
    idx = 0
    for param in model.parameters():
        size = param.numel()
        param.data = vec[idx: idx+size].view(param.shape).clone()
        idx += size

############################################################################
# 客户端本地更新 => Scaffold
############################################################################
def local_train_scaffold(model, c_local, c_global, train_loader, device, local_epochs, eta_l):
    """
    model: 当前客户端本地模型(已载入全局x)
    c_local: 客户端本地控制向量(1D)
    c_global: 全局控制向量(1D)
    local_epochs: 执行几个epoch
    eta_l: 本地学习率
    """
    criterion = nn.CrossEntropyLoss()
    model.train()

    for ep in range(local_epochs):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            # forward
            out = model(bx)
            loss_ce = criterion(out, by)
            # backward
            model.zero_grad()
            loss_ce.backward()

            # normal grad => param.grad
            current_vec = model_to_vec(model)
            grads = []
            for param in model.parameters():
                grads.append(param.grad.view(-1))
            full_grad = torch.cat(grads)

            # x_new = x_old - eta_l*( full_grad - c_local + c_global )
            updated_vec = current_vec - eta_l*(full_grad - c_local + c_global)
            vec_to_model(updated_vec, model)

############################################################################
# 更新 c_i^+ = c_i - c + (1/(K * eta_l)) ( x_old - y_i )
############################################################################
def update_client_control(c_local, c_global, x_old, y_i, K, eta_l):
    """
    c_local^+ = c_local - c_global + (1/(K*eta_l))*( x_old - y_i )
    x_old= global param used init => y_i= final local param
    """
    return c_local - c_global + (1.0/(K*eta_l))*( x_old - y_i )

############################################################################
# 新增: 测试准确度
############################################################################
def evaluate(model, test_loader, device):
    """
    评估当前模型在 test_loader 上的准确率
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output, dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total if total > 0 else 0.0

############################################################################
# SCAFFOLD 主训练 (按客户端数据量加权)
############################################################################
def scaffold_training(base_dir, eta_g, eta_l, device, K=1):
    """
    eta_g: global stepsize
    eta_l: local stepsize
    K: each client do K local epochs
    """
    # 加载客户端 data => train/test
    train_loaders = []
    test_loaders  = []
    input_shapes  = []
    valid_clients = [False]*NUM_CLIENTS
    client_sizes  = [0]*NUM_CLIENTS  # 用于记录各客户端训练集大小

    for cid in range(NUM_CLIENTS):
        d_, l_ = load_client_data_labels(base_dir, cid)
        if d_ is None:
            train_loaders.append(None)
            test_loaders.append(None)
            input_shapes.append(None)
            continue

        data_ = preprocess_data(d_)
        tr_loader, te_loader = create_data_loaders_scaffold(data_, l_, BATCH_SIZE)
        train_loaders.append(tr_loader)
        test_loaders.append(te_loader)

        sample_x,_ = next(iter(tr_loader))
        c_ = sample_x.shape[1]
        h_ = sample_x.shape[2]
        w_ = sample_x.shape[3]
        input_shapes.append((c_, h_, w_))
        valid_clients[cid] = True

        # 记录客户端的训练数据量
        ds_train = tr_loader.dataset
        client_sizes[cid] = len(ds_train)

    if not any(valid_clients):
        return np.zeros((NUM_GLOBAL_ROUNDS, NUM_CLIENTS))

    # 计算总数据量 (便于加权)
    total_data_size = sum(client_sizes)

    # 构建全局模型
    first_cid = valid_clients.index(True)
    c, h, w   = input_shapes[first_cid]
    global_model = get_model(c, (h, w)).to(device)
    global_vec   = model_to_vec(global_model).detach().clone()

    # 初始化全局控制向量 c_global => 0
    c_global = torch.zeros_like(global_vec)

    # 初始化本地控制向量 c_locals[cid]
    c_locals = []
    for cid in range(NUM_CLIENTS):
        if valid_clients[cid]:
            c_locals.append(torch.zeros_like(global_vec))
        else:
            c_locals.append(None)

    acc_matrix = np.zeros((NUM_GLOBAL_ROUNDS, NUM_CLIENTS))

    for round_idx in range(NUM_GLOBAL_ROUNDS):
        # broadcast => x, c_global
        sum_x = torch.zeros_like(global_vec)  # 用于累加加权后的 delta_x
        sum_c = torch.zeros_like(global_vec)  # 用于累加加权后的 delta_c
        sum_sizes_in_round = 0.0             # 本轮实际参与客户端的数据量之和

        for cid in range(NUM_CLIENTS):
            if not valid_clients[cid]:
                continue

            # 本地初始化
            local_model = get_model(c, (h, w)).to(device)
            vec_to_model(global_vec, local_model)

            # store old_x
            x_old = global_vec.clone()

            # run local epochs
            for _ep in range(K):
                local_train_scaffold(local_model, c_locals[cid], c_global,
                                     train_loaders[cid], device, 1, eta_l)

            y_i = model_to_vec(local_model).detach().clone()

            # update c_i^+
            c_i_new = update_client_control(c_locals[cid], c_global,
                                            x_old, y_i, K, eta_l)
            delta_c_i = c_i_new - c_locals[cid]
            c_locals[cid] = c_i_new

            # compute delta_y_i = y_i - x_old
            delta_y_i = y_i - x_old

            # =========== 重点修改: 按客户端数据量加权 ===========
            data_size_i = client_sizes[cid]
            sum_x += data_size_i * delta_y_i
            sum_c += data_size_i * delta_c_i
            sum_sizes_in_round += data_size_i

        # 若本轮没有任何客户端有效更新则跳过
        if sum_sizes_in_round > 0:
            # 做加权平均 => (sum of data_size_i * delta) / sum_sizes_in_round
            avg_x = sum_x / sum_sizes_in_round
            avg_c = sum_c / sum_sizes_in_round

            # 全局模型更新 => x <- x + eta_g * avg_x
            global_vec = global_vec + eta_g * avg_x

            # 全局控制向量更新 => c_global <- c_global + (sum_sizes_in_round / total_data_size)* avg_c
            # 原始 SCAFFOLD 代码用(|S|/N)*mean_c，这里改成"数据量占比" * avg_c
            frac_participate = sum_sizes_in_round / float(total_data_size)
            c_global = c_global + frac_participate * avg_c

        # load back => global_model
        vec_to_model(global_vec, global_model)

        # test => each client => global model
        for cid in range(NUM_CLIENTS):
            if not valid_clients[cid]:
                acc_matrix[round_idx, cid] = 0.0
            else:
                acc = evaluate(global_model, test_loaders[cid], device)
                acc_matrix[round_idx, cid] = acc

    return acc_matrix

############################################################################
# 与独自训练比较 => 计算 corr => 输出 (示例中随机生成)
############################################################################
def load_standalone_accuracies(dataset_name, dist_name, num_epochs=NUM_GLOBAL_ROUNDS):
    rng = np.random.RandomState(hash(dataset_name+dist_name)&0xffff)
    return rng.rand(num_epochs, NUM_CLIENTS)

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    distributions= {
        'POW':      lambda ds: os.path.join(POW_PATH, ds),
        'ECS_C5':   lambda ds: os.path.join(ECS_PATH, 'C5', ds),
        'PECS_C2':  lambda ds: os.path.join(PECS_PATH, 'C2', ds),
        'PECS_C5':  lambda ds: os.path.join(PECS_PATH, 'C5', ds),
        'PECS_C10': lambda ds: os.path.join(PECS_PATH, 'C10', ds)
    }
    datasets_list= ['FashionMNIST','CIFAR10']

    results= {}

    for dataset_name in datasets_list:
        (eta_g, eta_l)= learning_rates.get(dataset_name, (0.001, 0.1))
        print(f"\n===== SCAFFOLD on {dataset_name} (eta_g={eta_g}, eta_l={eta_l}, Weighted) =====")
        for dist_name, base_dir_fn in distributions.items():
            base_dir= base_dir_fn(dataset_name)
            print(f"--- Distribution: {dist_name} ---")

            # run scaffold (weighted)
            acc_matrix= scaffold_training(base_dir, eta_g, eta_l, device, K=1)
            # shape=(NUM_GLOBAL_ROUNDS, NUM_CLIENTS)

            mean_each_round= np.mean(acc_matrix, axis=1)
            max_each_round = np.max(acc_matrix, axis=1)

            last3_mean= mean_each_round[-3:]
            avg3_mean= np.mean(last3_mean)
            std3_mean= np.std(last3_mean)

            last3_max= max_each_round[-3:]
            avg3_max= np.mean(last3_max)
            std3_max= np.std(last3_max)

            # 与独自训练 => corr (此处仅示例用随机生成的独自训练结果)
            standalone_matrix= load_standalone_accuracies(dataset_name, dist_name, NUM_GLOBAL_ROUNDS)
            corr_vals=[]
            for r_ in range(NUM_GLOBAL_ROUNDS-3, NUM_GLOBAL_ROUNDS):
                scaf_accs= acc_matrix[r_,:]
                stand_accs= standalone_matrix[r_,:]
                if np.all(scaf_accs==0) or np.all(stand_accs==0):
                    corr= 0.0
                else:
                    corr= np.corrcoef(scaf_accs, stand_accs)[0,1]
                corr_vals.append(corr*100.0)
            corr_vals= np.array(corr_vals)
            avg_corr= np.mean(corr_vals)
            std_corr= np.std(corr_vals)

            results[(dataset_name, dist_name)] = (
                (avg3_mean, std3_mean),
                (avg3_max,  std3_max),
                (avg_corr,  std_corr)
            )

    final_result_file= os.path.join(RESULT_DIR, "final_results_weighted.txt")
    with open(final_result_file, "w") as f:
        f.write("dataset, distribution, last3_AvgAcc(mean±std), last3_MaxAcc(mean±std), last3_Corr(mean±std)\n")
        for dataset_name in datasets_list:
            for dist_name in distributions.keys():
                val= results.get((dataset_name, dist_name), None)
                if val is None:
                    f.write(f"{dataset_name}, {dist_name}, 0±0, 0±0, 0±0\n")
                else:
                    (m1, s1), (m2, s2), (mc, sc)= val
                    line= (f"{dataset_name}, {dist_name}, "
                           f"{m1:.4f}±{s1:.4f}, "
                           f"{m2:.4f}±{s2:.4f}, "
                           f"{mc:.4f}±{sc:.4f}\n")
                    f.write(line)
    print(f"\n>>> SCAFFOLD (Weighted) 结果已保存至: {final_result_file}")

if __name__=='__main__':
    main()
