#!/usr/bin/env python3
import os
import time  # <=== 新增：导入time模块
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 路径设置
TRAIN_PATH = '/home/tvy5242/EHR_fl/A_Experiment/DATA/train'
RESULT_DIR = '/home/tvy5242/EHR_fl/A_Experiment/RESULT/fedavg'

POW_PATH   = os.path.join(TRAIN_PATH, 'POW')
ECS_PATH   = os.path.join(TRAIN_PATH, 'ECS')
PECS_PATH  = os.path.join(TRAIN_PATH, 'PECS')

NUM_CLIENTS   = 10       # 每个分布下客户端数量
NUM_GLOBAL_ROUNDS = 20   # FedAvg 全局轮次数
LOCAL_EPOCHS  = 1        # 每轮本地训练的 epoch 数（可根据需求修改）
BATCH_SIZE    = 32       # 批次大小

# 定义不同数据集使用的学习率
learning_rates = {
    "FashionMNIST": 0.001,
    "CIFAR10": 0.005
}

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
# 创建 DataLoader (train, test) 用于本地训练/测试
########################################
def create_data_loaders_for_fedavg(data, labels, batch_size):
    """
    先将客户端数据划分为训练集、测试集 (80%/20%)，再分别构造 DataLoader。
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

def local_train(model, train_loader, device, local_epochs, lr):
    """
    单个客户端在其数据上进行本地训练 (local_epochs) 次。
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for _ in range(local_epochs):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

def evaluate(model, test_loader, device):
    """
    在单个客户端的测试集上测试准确率。
    """
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

def fedavg_training(base_dir, lr, device):
    """
    在给定 base_dir (某分布下的数据集目录) 上进行 FedAvg 训练。
    返回 shape = (NUM_GLOBAL_ROUNDS, NUM_CLIENTS) 的准确率矩阵。
    """
    # 1) 加载所有客户端的数据
    train_loaders = []
    test_loaders  = []
    input_shapes  = []
    valid_clients = [False]*NUM_CLIENTS
    
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
    
    # 全局轮次
    acc_matrix = np.zeros((NUM_GLOBAL_ROUNDS, NUM_CLIENTS))
    
    for round_idx in range(NUM_GLOBAL_ROUNDS):
        # ============== 新增：开始计时 ==============
        start_time = time.time()
        
        local_params_list = []
        
        # (a) 各客户端本地训练
        for cid in range(NUM_CLIENTS):
            if not valid_clients[cid]:
                continue
            # 拷贝全局模型到本地
            local_model = get_model(c, (h, w)).to(device)
            local_model.load_state_dict(global_model.state_dict())
            
            # 本地训练
            local_train(local_model, train_loaders[cid], device, LOCAL_EPOCHS, lr)
            
            # 记录训练后参数
            local_params_list.append(local_model.state_dict())
        
        # (b) 服务器做平均聚合
        if len(local_params_list) > 0:
            new_state_dict = {}
            for key in local_params_list[0].keys():
                # 将所有客户端对应参数转为 float，再堆叠取 mean
                stacked = torch.stack([lp[key].float() for lp in local_params_list], dim=0)
                avg = stacked.mean(dim=0)
                new_state_dict[key] = avg
            
            global_model.load_state_dict(new_state_dict)
        
        # (c) 测试：将当前全局模型用到各客户端的测试集
        for cid in range(NUM_CLIENTS):
            if not valid_clients[cid]:
                acc_matrix[round_idx, cid] = 0.0
            else:
                acc_matrix[round_idx, cid] = evaluate(global_model, test_loaders[cid], device)
        
        # ============== 新增：结束计时并打印 ==============
        end_time = time.time()
        round_time = end_time - start_time
        print(f"Round {round_idx+1}/{NUM_GLOBAL_ROUNDS} took {round_time:.4f} seconds.")
    
    return acc_matrix

########################################
# 假设已有独自训练(standalone)的结果，用于计算相关系数
# 这里演示：随机生成一些准确率来示范
########################################
def load_standalone_accuracies(dataset_name, dist_name, num_epochs=NUM_GLOBAL_ROUNDS):
    """
    你需要自行将此函数替换为真实的加载逻辑（例如从txt或npy读取）。
    这里仅用随机数演示返回 shape=(num_epochs, NUM_CLIENTS)。
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
        print(f"\n====== FedAvg on {dataset_name} (lr={lr}) ======")
        
        for dist_name, base_dir_fn in distributions.items():
            base_dir = base_dir_fn(dataset_name)
            print(f"--- Distribution: {dist_name} ---")
            
            # FedAvg
            acc_matrix = fedavg_training(base_dir, lr, device)
            # 形状 (NUM_GLOBAL_ROUNDS, NUM_CLIENTS)
            
            # 统计最后 3 轮
            mean_acc_each_round = np.mean(acc_matrix, axis=1)
            max_acc_each_round  = np.max(acc_matrix,  axis=1)
            
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
                fedavg_accs = acc_matrix[r, :]       # shape=(num_clients,)
                stand_accs  = standalone_matrix[r, :]
                if np.all(fedavg_accs == 0) or np.all(stand_accs == 0):
                    corr = 0.0
                else:
                    corr = np.corrcoef(fedavg_accs, stand_accs)[0, 1]
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
    print(f"\n>>> FedAvg 结果已保存至: {final_result_file}")

if __name__ == '__main__':
    main()
