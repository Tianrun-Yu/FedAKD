#!/usr/bin/env python3
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

###################################################
# FedAvg 配置
###################################################
RESULT_DIR   = '/home/tvy5242/EHR_fl/A_Experiment/RESULT/FedAvg'
TRAIN_PATH   = '/home/tvy5242/EHR_fl/A_Experiment/DATA/train'

# 独自训练结果文件目录（请根据自己实际路径来修改）
STANDALONE_DIR = '/home/tvy5242/EHR_fl/A_Experiment/RESULT/standalone'

POW_PATH   = os.path.join(TRAIN_PATH, 'POW')
ECS_PATH   = os.path.join(TRAIN_PATH, 'ECS')
PECS_PATH  = os.path.join(TRAIN_PATH, 'PECS')

NUM_CLIENTS       = 10
NUM_GLOBAL_ROUNDS = 20
LOCAL_EPOCHS      = 1
BATCH_SIZE        = 32

learning_rates = {
    "FashionMNIST": 0.001,
    "CIFAR10":      0.005
}

###################################################
# 模型定义: SimpleCNN
###################################################
class SimpleCNN(nn.Module):
    def __init__(self, in_channels, img_size, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.in_channels = in_channels
        self.img_size    = img_size
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2,2)
        # 经过两次 pool => h,w都是原来的1/4
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

###################################################
# 数据加载与预处理
###################################################
def load_client_data_labels(base_dir, cid):
    """
    从 base_dir/client_{cid}/data.npy 和 labels.npy 加载本地数据
    """
    client_dir = os.path.join(base_dir, f'client_{cid}')
    data_path  = os.path.join(client_dir, 'data.npy')
    label_path = os.path.join(client_dir, 'labels.npy')
    if os.path.exists(data_path) and os.path.exists(label_path):
        data   = np.load(data_path)
        labels = np.load(label_path)
        return data, labels
    return None, None

def preprocess_data(data):
    """
    根据数据维度做适当的 reshape/转置:
      - [N,H,W]      => [N,1,H,W]
      - [N,H,W,3]    => [N,3,H,W]
    """
    if data.ndim == 3:
        data = data[:, None, :, :]
    elif data.ndim == 4 and data.shape[-1] == 3:
        data = np.transpose(data, (0, 3, 1, 2))
    return data.astype(np.float32)

def create_data_loaders_fedavg(data, labels, batch_size):
    """
    先拆分 80% 训练，20% 测试，然后构造 DataLoader
    """
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True, random_state=42
    )
    # 这里可以再次调用 preprocess_data (如果还需要)
    X_train = preprocess_data(X_train)
    X_test  = preprocess_data(X_test)

    ds_train = TensorDataset(torch.tensor(X_train), torch.tensor(y_train,dtype=torch.long))
    ds_test  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test,dtype=torch.long))
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False)
    return loader_train, loader_test

###################################################
# 从独自训练文件中读取准确率
###################################################
def load_standalone_accuracies(dataset_name, dist_name, num_epochs=NUM_GLOBAL_ROUNDS):
    """
    假设在 STANDALONE_DIR 下存在名为 "{dataset_name}_{dist_name}.txt" 的文件，
    内容类似:
      Client 0: 0.5560
      Client 1: 0.5400
      ...
      Client 9: 0.4920
    我们读取这 10 行准确率后，构造 shape=(num_epochs, NUM_CLIENTS) 的矩阵，
    每轮都使用同一份准确率(如只记录最终)。若找不到该文件，则返回零矩阵。
    """
    file_name = f"{dataset_name}_{dist_name}.txt"
    file_path = os.path.join(STANDALONE_DIR, file_name)

    if not os.path.exists(file_path):
        # 文件不存在 => 用全0矩阵填充
        return np.zeros((num_epochs, NUM_CLIENTS), dtype=np.float32)
    
    single_acc = np.zeros(NUM_CLIENTS, dtype=np.float32)
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Client"):
                # e.g. "Client 0: 0.5560"
                parts = line.split(":")
                left  = parts[0].strip()  # "Client 0"
                right = parts[1].strip()  # "0.5560"
                cidx_str = left.split()[1]  # "0"
                cidx = int(cidx_str)
                val  = float(right)
                single_acc[cidx] = val
    # 将这一行复制成 (num_epochs, NUM_CLIENTS)
    expanded = np.tile(single_acc, (num_epochs,1))
    return expanded

###################################################
# 客户端本地训练
###################################################
def local_train(model, train_loader, device, local_epochs, lr):
    """
    给定模型，训练数据，做一次本地训练。
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for _ in range(local_epochs):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

def evaluate(model, test_loader, device):
    """
    评估模型在本地测试集上的准确率
    """
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for bx,by in test_loader:
            bx,by= bx.to(device), by.to(device)
            out= model(bx)
            _, pred= torch.max(out,1)
            correct+= (pred==by).sum().item()
            total  += by.size(0)
    return correct/(total+1e-9)

###################################################
# 聚合 (FedAvg)
###################################################
def fedavg_aggregate(global_model, local_models, data_sizes):
    """
    local_models: 包含若干客户端本地训练后模型，长度=有效客户端数
    data_sizes:   每个客户端的数据集大小（用于加权平均）
    """
    # 各客户端的权重
    total_size = sum(data_sizes)
    # 全局模型参数初始化
    global_dict = dict(global_model.state_dict())
    for key in global_dict.keys():
        global_dict[key] = 0.0

    for i, local_m in enumerate(local_models):
        local_dict = local_m.state_dict()
        frac = data_sizes[i] / total_size
        for key in global_dict.keys():
            global_dict[key] += local_dict[key] * frac

    global_model.load_state_dict(global_dict)

###################################################
# FedAvg 主过程
###################################################
def fedavg_training(base_dir, lr, device):
    """
    读取客户端数据，然后执行 NUM_GLOBAL_ROUNDS 的FedAvg
    返回 (NUM_GLOBAL_ROUNDS, NUM_CLIENTS) 的准确率矩阵
    """
    train_loaders= [None]*NUM_CLIENTS
    test_loaders = [None]*NUM_CLIENTS
    data_sizes   = [0]*NUM_CLIENTS
    valid_clients= [False]*NUM_CLIENTS
    input_shapes = [None]*NUM_CLIENTS

    # 1) 读取各客户端数据
    for cid in range(NUM_CLIENTS):
        data, labels = load_client_data_labels(base_dir, cid)
        if data is None:
            continue
        data_sizes[cid] = len(data)
        tr_loader, te_loader = create_data_loaders_fedavg(data, labels, BATCH_SIZE)
        train_loaders[cid]   = tr_loader
        test_loaders[cid]    = te_loader

        # 获取 shape
        sample_x, _ = next(iter(tr_loader))
        c = sample_x.shape[1]
        h = sample_x.shape[2]
        w = sample_x.shape[3]
        input_shapes[cid] = (c,h,w)

        valid_clients[cid] = True

    if not any(valid_clients):
        return np.zeros((NUM_GLOBAL_ROUNDS, NUM_CLIENTS))

    # 2) 建立初始全局模型
    first_cid = valid_clients.index(True)
    c,h,w = input_shapes[first_cid]
    global_model = get_model(c,(h,w)).to(device)

    # 记录各轮准确率
    acc_matrix = np.zeros((NUM_GLOBAL_ROUNDS, NUM_CLIENTS))

    # 3) 开始联邦训练
    for round_idx in range(NUM_GLOBAL_ROUNDS):
        local_models = []
        local_data_sizes = []
        
        # (a) 每个客户端：拷贝 global -> local, 在本地训练
        for cid in range(NUM_CLIENTS):
            if not valid_clients[cid]:
                continue
            # clone
            local_m = get_model(c,(h,w)).to(device)
            local_m.load_state_dict(global_model.state_dict())

            # local train
            local_train(local_m, train_loaders[cid], device, LOCAL_EPOCHS, lr)

            local_models.append(local_m)
            local_data_sizes.append(data_sizes[cid])
        
        # (b) 聚合
        fedavg_aggregate(global_model, local_models, local_data_sizes)
        
        # (c) 评估
        for cid in range(NUM_CLIENTS):
            if not valid_clients[cid]:
                acc_matrix[round_idx, cid] = 0.0
            else:
                acc = evaluate(global_model, test_loaders[cid], device)
                acc_matrix[round_idx, cid] = acc
    
    return acc_matrix

###################################################
# 主函数
###################################################
def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 同之前：给定不同分布目录
    distributions = {
        'POW':      lambda ds: os.path.join(POW_PATH, ds),
        'ECS_C5':   lambda ds: os.path.join(ECS_PATH, 'C5', ds),
        'PECS_C2':  lambda ds: os.path.join(PECS_PATH, 'C2', ds),
        'PECS_C5':  lambda ds: os.path.join(PECS_PATH, 'C5', ds),
        'PECS_C10': lambda ds: os.path.join(PECS_PATH, 'C10', ds)
    }

    datasets_list= ['FashionMNIST','CIFAR10']
    results= {}

    for dataset_name in datasets_list:
        lr= learning_rates.get(dataset_name, 0.001)
        print(f"\n===== FedAvg on {dataset_name} (lr={lr}) =====")
        for dist_name, base_dir_fn in distributions.items():
            base_dir = base_dir_fn(dataset_name)
            print(f"--- Distribution: {dist_name} ---")
            acc_matrix = fedavg_training(base_dir, lr, device)

            mean_acc_each_round = np.mean(acc_matrix, axis=1)
            max_acc_each_round  = np.max(acc_matrix,  axis=1)

            last3_mean = mean_acc_each_round[-3:]
            avg3_mean  = np.mean(last3_mean)
            std3_mean  = np.std(last3_mean)

            last3_max  = max_acc_each_round[-3:]
            avg3_max   = np.mean(last3_max)
            std3_max   = np.std(last3_max)

            # ====== 计算与独自训练的相关系数 ======
            standalone_matrix = load_standalone_accuracies(dataset_name, dist_name, NUM_GLOBAL_ROUNDS)
            corr_vals = []
            # 仅取最后3轮 => 做个均值
            for r_ in range(NUM_GLOBAL_ROUNDS-3, NUM_GLOBAL_ROUNDS):
                fedavg_accs = acc_matrix[r_, :]
                stand_accs  = standalone_matrix[r_, :]
                if np.all(fedavg_accs == 0) or np.all(stand_accs == 0):
                    corr = 0.0
                else:
                    corr = np.corrcoef(fedavg_accs, stand_accs)[0, 1]
                corr_vals.append(corr * 100.0)
            
            corr_vals = np.array(corr_vals)
            avg_corr   = np.mean(corr_vals)
            std_corr   = np.std(corr_vals)

            # 记录结果
            results[(dataset_name, dist_name)] = (
                (avg3_mean, std3_mean),
                (avg3_max,  std3_max),
                (avg_corr,  std_corr)
            )
    
    # ====== 写结果到文件 ======
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
    print(f"\n>>> FedAvg 结果已保存至: {final_file}")

if __name__ == '__main__':
    main()
