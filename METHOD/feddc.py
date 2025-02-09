#!/usr/bin/env python3
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 路径设置
TRAIN_PATH = '/home/tvy5242/EHR_fl/A_Experiment/DATA/train'
RESULT_DIR = '/home/tvy5242/EHR_fl/A_Experiment/RESULT/FedDC'

POW_PATH   = os.path.join(TRAIN_PATH, 'POW')
ECS_PATH   = os.path.join(TRAIN_PATH, 'ECS')
PECS_PATH  = os.path.join(TRAIN_PATH, 'PECS')

NUM_CLIENTS       = 10       # 每个分布下客户端数量
NUM_GLOBAL_ROUNDS = 20       # 全局轮次数
LOCAL_EPOCHS      = 1        # 每轮本地训练的 epoch 数
BATCH_SIZE        = 32       # 批次大小

# 学习率设置（可根据数据集调整）
learning_rates = {
    "FashionMNIST": 0.001,
    "CIFAR10": 0.005
}

# FedDC 中的惩罚项超参数 α（可根据需求调整）
ALPHA = 1.0

########################################
# 简单 CNN 模型（适用于 10 类分类）
########################################
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

########################################
# 数据加载与预处理
########################################
def load_client_data_labels(base_dir, client_id):
    """
    加载指定客户端的数据和标签 (data.npy, labels.npy)。
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
    将数据转为 CNN 可用的格式：
      - (N, H, W) => (N, 1, H, W)
      - (N, H, W, 3) => (N, 3, H, W)
    """
    if len(data.shape) == 3:
        data = data[:, None, :, :]
    elif len(data.shape) == 4 and data.shape[-1] == 3:
        data = np.transpose(data, (0, 3, 1, 2))
    return data.astype(np.float32)

########################################
# 创建 DataLoader (train, test)
########################################
def create_data_loaders_for_fedavg(data, labels, batch_size):
    """
    将客户端数据划分为训练集、测试集 (80%/20%)，再分别构造 DataLoader。
    """
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42, shuffle=True
    )
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test  = torch.tensor(X_test)
    y_test  = torch.tensor(y_test, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset  = TensorDataset(X_test,  y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

########################################
# 本地训练：FedDC 版本（仅包含惩罚项，不计算梯度修正项）
########################################
def local_train_feddc(model, train_loader, device, local_epochs, lr, global_state, drift, alpha):
    """
    本地训练：最小化目标函数
      Empirical Loss + (α/2)*||θ + h - w||^2
    其中：
      - θ: 本地模型参数（通过 model.named_parameters() 获取）
      - h: 客户端的局部漂移变量（dict，初始为 0）
      - w: 本轮开始时的全局模型参数（global_state，dict）
    注意：漂移变量 h 在本地训练期间保持不变，训练结束后更新为 h + (θ_final - θ_initial)
    最后返回校正后的模型参数（θ_final + h_new）以及更新后的漂移变量。
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 保存初始参数（仅对需要训练的参数）
    initial_state = {name: param.clone() for name, param in model.named_parameters()}
    
    for _ in range(local_epochs):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss_empirical = criterion(outputs, batch_y)
            penalty = 0.0
            # 仅对模型中可训练参数计算惩罚项
            for name, param in model.named_parameters():
                # 注意：global_state 和 drift 可能存放在 CPU 上，需转移到 device
                global_param = global_state[name].to(device)
                drift_param  = drift[name].to(device)
                penalty += torch.sum((param + drift_param - global_param) ** 2)
            loss = loss_empirical + (alpha / 2.0) * penalty
            loss.backward()
            optimizer.step()
    
    # 训练结束，获取最终参数
    final_state = {name: param.clone() for name, param in model.named_parameters()}
    # 更新漂移变量： h_new = h + (θ_final - θ_initial)
    new_drift = {}
    for name in drift.keys():
        new_drift[name] = drift[name] + (final_state[name] - initial_state[name])
    # 计算校正后的模型参数： θ_final + h_new
    corrected_state = {}
    for name in final_state.keys():
        corrected_state[name] = final_state[name] + new_drift[name]
    
    # 为了聚合时保持 BN 等缓冲变量一致，获取完整 state_dict，并用校正后的参数覆盖对应项
    full_state = model.state_dict()
    for name in corrected_state.keys():
        full_state[name] = corrected_state[name]
    return full_state, new_drift

########################################
# 全局训练：FedDC
########################################
def feddc_training(base_dir, lr, device, alpha):
    """
    在给定 base_dir（某分布下的数据集目录）上进行 FedDC 训练。
    返回 shape = (NUM_GLOBAL_ROUNDS, NUM_CLIENTS) 的准确率矩阵。
    """
    # 加载所有客户端数据
    train_loaders = []
    test_loaders  = []
    input_shapes  = []
    valid_clients = [False] * NUM_CLIENTS
    
    for cid in range(NUM_CLIENTS):
        data, labels = load_client_data_labels(base_dir, cid)
        if data is None or labels is None:
            train_loaders.append(None)
            test_loaders.append(None)
            input_shapes.append(None)
            continue
        
        data = preprocess_data(data)
        tr_loader, te_loader = create_data_loaders_for_fedavg(data, labels, BATCH_SIZE)
        train_loaders.append(tr_loader)
        test_loaders.append(te_loader)
        
        # 记录输入形状 (C, H, W)
        _, c, h, w = next(iter(tr_loader))[0].shape
        input_shapes.append((c, h, w))
        
        valid_clients[cid] = True
    
    if not any(valid_clients):
        # 如果没有任何客户端可用，则返回全0
        return np.zeros((NUM_GLOBAL_ROUNDS, NUM_CLIENTS))
    
    first_valid_cid = valid_clients.index(True)
    c, h, w = input_shapes[first_valid_cid]
    global_model = get_model(c, (h, w)).to(device)
    
    # 为每个有效客户端初始化漂移变量（仅针对可训练参数）
    drift_vars = [None] * NUM_CLIENTS
    for cid in range(NUM_CLIENTS):
        if valid_clients[cid]:
            drift_vars[cid] = {name: torch.zeros_like(param.data) for name, param in global_model.named_parameters()}
    
    # 记录每轮测试准确率，形状为 (NUM_GLOBAL_ROUNDS, NUM_CLIENTS)
    acc_matrix = np.zeros((NUM_GLOBAL_ROUNDS, NUM_CLIENTS))
    
    for round_idx in range(NUM_GLOBAL_ROUNDS):
        local_corrected_states = []
        # 保存本轮开始时的全局模型参数（仅取可训练参数）
        global_state = {name: param.clone() for name, param in global_model.named_parameters()}
        
        # 各客户端本地训练
        for cid in range(NUM_CLIENTS):
            if not valid_clients[cid]:
                continue
            # 拷贝全局模型到本地
            local_model = get_model(c, (h, w)).to(device)
            local_model.load_state_dict(global_model.state_dict())
            
            # 使用客户端的漂移变量
            drift = drift_vars[cid]
            
            # 本地训练：返回校正后的模型参数和更新后的漂移变量
            corrected_state, new_drift = local_train_feddc(local_model, train_loaders[cid], device, LOCAL_EPOCHS, lr, global_state, drift, alpha)
            drift_vars[cid] = new_drift
            local_corrected_states.append(corrected_state)
        
        # (b) 服务器端聚合：对所有客户端上传的校正参数求平均
        if len(local_corrected_states) > 0:
            new_global_state = {}
            for key in local_corrected_states[0].keys():
                stacked = torch.stack([client_state[key].float() for client_state in local_corrected_states], dim=0)
                new_global_state[key] = stacked.mean(dim=0)
            # 更新全局模型：仅覆盖模型中参数部分，对于 BN 等缓冲变量可保持不变
            global_model_state = global_model.state_dict()
            for key in new_global_state.keys():
                global_model_state[key] = new_global_state[key]
            global_model.load_state_dict(global_model_state)
        
        # (c) 测试：使用全局模型在各客户端测试集上评估准确率
        for cid in range(NUM_CLIENTS):
            if not valid_clients[cid]:
                acc_matrix[round_idx, cid] = 0.0
            else:
                acc_matrix[round_idx, cid] = evaluate(global_model, test_loaders[cid], device)
    
    return acc_matrix

########################################
# 测试函数：在单个客户端的测试集上测试准确率
########################################
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            total   += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    return correct / total if total > 0 else 0.0

########################################
# 加载独自训练（standalone）的结果，用于计算相关系数（示例代码：随机生成）
########################################
def load_standalone_accuracies(dataset_name, dist_name, num_epochs=NUM_GLOBAL_ROUNDS):
    """
    请根据实际情况替换此函数，本示例仅生成 shape=(num_epochs, NUM_CLIENTS) 的随机数。
    """
    rng = np.random.RandomState(hash(dataset_name + dist_name) & 0xffff)
    return rng.rand(num_epochs, NUM_CLIENTS)

########################################
# 主流程
########################################
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
    
    datasets_list = ['FashionMNIST', 'CIFAR10']
    
    # 记录结果： (avgAcc, maxAcc, corr) 三大指标
    results = {}
    
    for dataset_name in datasets_list:
        lr = learning_rates.get(dataset_name, 0.001)
        print(f"\n====== FedDC on {dataset_name} (lr={lr}, α={ALPHA}) ======")
        
        for dist_name, base_dir_fn in distributions.items():
            base_dir = base_dir_fn(dataset_name)
            print(f"--- Distribution: {dist_name} ---")
            
            # FedDC 训练
            acc_matrix = feddc_training(base_dir, lr, device, ALPHA)
            # acc_matrix 形状 (NUM_GLOBAL_ROUNDS, NUM_CLIENTS)
            
            # 统计最后 3 轮
            mean_acc_each_round = np.mean(acc_matrix, axis=1)
            max_acc_each_round  = np.max(acc_matrix, axis=1)
            
            last3_mean_acc = mean_acc_each_round[-3:]
            avg3_mean_acc  = np.mean(last3_mean_acc)
            std3_mean_acc  = np.std(last3_mean_acc)
            
            last3_max_acc = max_acc_each_round[-3:]
            avg3_max_acc  = np.mean(last3_max_acc)
            std3_max_acc  = np.std(last3_max_acc)
            
            # 与独自训练相关系数
            standalone_matrix = load_standalone_accuracies(dataset_name, dist_name, NUM_GLOBAL_ROUNDS)
            corr_vals = []
            for r in range(NUM_GLOBAL_ROUNDS-3, NUM_GLOBAL_ROUNDS):
                feddc_accs = acc_matrix[r, :]       # shape=(NUM_CLIENTS,)
                stand_accs  = standalone_matrix[r, :]
                if np.all(feddc_accs == 0) or np.all(stand_accs == 0):
                    corr = 0.0
                else:
                    corr = np.corrcoef(feddc_accs, stand_accs)[0, 1]
                corr_vals.append(corr * 100.0)
            corr_vals = np.array(corr_vals)
            avg_corr = np.mean(corr_vals)
            std_corr = np.std(corr_vals)
            
            results[(dataset_name, dist_name)] = (
                (avg3_mean_acc, std3_mean_acc),
                (avg3_max_acc,  std3_max_acc),
                (avg_corr,      std_corr)
            )
    
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
    print(f"\n>>> FedDC 结果已保存至: {final_result_file}")

if __name__ == '__main__':
    main()
