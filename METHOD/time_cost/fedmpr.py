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

# 路径设置
TRAIN_PATH = '/home/tvy5242/EHR_fl/A_Experiment/DATA/train'
RESULT_DIR = '/home/tvy5242/EHR_fl/A_Experiment/RESULT/FedMPR'

POW_PATH   = os.path.join(TRAIN_PATH, 'POW')
ECS_PATH   = os.path.join(TRAIN_PATH, 'ECS')
PECS_PATH  = os.path.join(TRAIN_PATH, 'PECS')

NUM_CLIENTS       = 10       # 客户端数量
NUM_GLOBAL_ROUNDS = 20       # 总共进行的联邦轮数
LOCAL_EPOCHS      = 1        # 客户端本地训练 epoch 数
BATCH_SIZE        = 32       # mini-batch 大小

# 定义剪枝比例 p (例如 10% => 0.1)
PRUNE_PERCENT = 0.1

# 定义不同数据集使用的学习率 (lr) 和 weight_decay (正则化)
learning_configs = {
    "FashionMNIST": (0.001, 1e-4),  # (lr, weight_decay)
    "CIFAR10":      (0.005, 1e-4)
}

############################################################################
# 模型定义: SimpleCNN (与 FedAvg 相同)
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
        self.fc    = nn.Linear(64 * feature_h * feature_w, num_classes)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def get_model(in_channels, img_size, num_classes=10):
    return SimpleCNN(in_channels, img_size, num_classes)

############################################################################
# 数据加载与预处理
############################################################################
def load_client_data_labels(base_dir, client_id):
    """
    读取指定客户端的数据 (data.npy) 与标签 (labels.npy).
    """
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
    (N, H, W)->(N,1,H,W) 或 (N,H,W,3)->(N,3,H,W)
    """
    if data.ndim==3:
        data= data[:,None,:,:]
    elif data.ndim==4 and data.shape[-1]==3:
        data= np.transpose(data, (0,3,1,2))
    return data.astype(np.float32)

def create_data_loaders_mpr(data, labels, batch_size):
    """
    80/20 => train/test
    """
    X_train, X_test, y_train, y_test= train_test_split(
        data, labels, test_size=0.2, shuffle=True, random_state=42
    )
    X_train= torch.tensor(X_train)
    y_train= torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test)
    y_test = torch.tensor(y_test,  dtype=torch.long)

    ds_train= TensorDataset(X_train, y_train)
    ds_test = TensorDataset(X_test, y_test)
    loader_train= DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    return loader_train, loader_test

############################################################################
# 本地训练 (带正则化), 训练后返回 local_model
############################################################################
def local_train_with_reg(model, train_loader, device, local_epochs, lr, weight_decay):
    """
    在客户端进行本地训练: 
    - 使用 weight_decay 做 L2 正则
    - epoch 数 = local_epochs
    """
    model.train()
    criterion= nn.CrossEntropyLoss()
    optimizer= optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for ep in range(local_epochs):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y= batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs= model(batch_x)
            loss= criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    return model

############################################################################
# 迭代式幅度剪枝: 对 weights 做 global threshold => set small ones to zero
############################################################################
def magnitude_prune(model, prune_percent=0.1):
    """
    在 model 的所有参数中，找出绝对值排名在百分位 prune_percent 的阈值 t，
    将小于 t 的权重置 0
    """
    # 1) 将全部参数 flatten
    all_weights=[]
    for param in model.parameters():
        all_weights.append(param.data.view(-1))
    all_weights= torch.cat(all_weights)
    # 2) 找绝对值分位数 prune_percent => t
    threshold_value= torch.quantile(all_weights.abs(), prune_percent)
    # 3) 遍历所有 param，置 0
    with torch.no_grad():
        for param in model.parameters():
            mask= param.data.abs() >= threshold_value
            param.data= param.data * mask  # small ones => 0

############################################################################
# 评估模型的辅助函数 evaluate
############################################################################
def evaluate(model, test_loader, device):
    """
    在给定的 test_loader 上测试模型的准确率
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, dim=1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
    return correct / total if total > 0 else 0.0

############################################################################
# 在给定 base_dir 进行 FedMPR 训练, 每轮后进行 magnitude pruning
############################################################################
def fedmpr_training(base_dir, lr, weight_decay, device):
    """
    返回 shape=(NUM_GLOBAL_ROUNDS, NUM_CLIENTS) 的准确率矩阵
    """
    # 1) 加载客户端数据
    train_loaders= []
    test_loaders= []
    input_shapes= []
    valid_clients= [False]*NUM_CLIENTS

    for cid in range(NUM_CLIENTS):
        data, labels= load_client_data_labels(base_dir, cid)
        if data is None:
            train_loaders.append(None)
            test_loaders.append(None)
            input_shapes.append(None)
            continue
        data_= preprocess_data(data)
        tr_loader, te_loader= create_data_loaders_mpr(data_, labels, BATCH_SIZE)
        train_loaders.append(tr_loader)
        test_loaders.append(te_loader)

        sample_x,_= next(iter(tr_loader))
        c_= sample_x.shape[1]
        h_= sample_x.shape[2]
        w_= sample_x.shape[3]
        input_shapes.append((c_,h_,w_))
        valid_clients[cid]= True

    if not any(valid_clients):
        return np.zeros((NUM_GLOBAL_ROUNDS, NUM_CLIENTS))

    first_valid_cid= valid_clients.index(True)
    c,h,w= input_shapes[first_valid_cid]
    # 初始化全局模型
    global_model= get_model(c, (h,w)).to(device)

    # 记录准确率
    acc_matrix= np.zeros((NUM_GLOBAL_ROUNDS, NUM_CLIENTS))

    for round_idx in range(NUM_GLOBAL_ROUNDS):
        # ============== 新增：开始计时 ==============
        start_time = time.time()

        local_states=[]
        used_clients= []
        # (a) 每个客户端 => 复制 global_model => 本地训练 => 剪枝 => 返回
        for cid in range(NUM_CLIENTS):
            if not valid_clients[cid]:
                continue
            local_model= get_model(c,(h,w)).to(device)
            local_model.load_state_dict(global_model.state_dict())

            # 本地训练(带正则)
            local_model= local_train_with_reg(
                local_model, train_loaders[cid], device, LOCAL_EPOCHS, lr, weight_decay
            )

            # 剪枝
            magnitude_prune(local_model, prune_percent=PRUNE_PERCENT)

            local_states.append(local_model.state_dict())
            used_clients.append(cid)

        # (b) 服务器聚合 (FedAvg)
        if len(local_states) > 0:
            new_state= {}
            # param key => average
            for key in local_states[0].keys():
                stacked= torch.stack([ls[key].float() for ls in local_states], dim=0)
                avg_val= stacked.mean(dim=0)
                new_state[key]= avg_val
            global_model.load_state_dict(new_state)

        # (c) 测试 => 用 global_model 对各客户端测试集
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

############################################################################
# 假设已有独自训练 => 相关系数 => 输出 (示例中用随机数模拟)
############################################################################
def load_standalone_accuracies(dataset_name, dist_name, num_epochs=NUM_GLOBAL_ROUNDS):
    rng= np.random.RandomState(hash(dataset_name+dist_name)&0xffff)
    return rng.rand(num_epochs, NUM_CLIENTS)

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 几种分布
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
        (lr, wd)= learning_configs.get(dataset_name, (0.001, 1e-4))
        print(f"\n===== FedMPR on {dataset_name} (lr={lr}, weight_decay={wd}, prune={PRUNE_PERCENT}) =====")
        
        for dist_name, base_dir_fn in distributions.items():
            base_dir= base_dir_fn(dataset_name)
            print(f"--- Distribution: {dist_name} ---")

            acc_matrix= fedmpr_training(base_dir, lr, wd, device)
            # (NUM_GLOBAL_ROUNDS, NUM_CLIENTS)

            mean_each_round= np.mean(acc_matrix, axis=1)
            max_each_round = np.max(acc_matrix, axis=1)

            last3_mean= mean_each_round[-3:]
            avg3_mean= np.mean(last3_mean)
            std3_mean= np.std(last3_mean)

            last3_max= max_each_round[-3:]
            avg3_max= np.mean(last3_max)
            std3_max= np.std(last3_max)

            # 与独自训练 => corr (此处仅用随机数模拟)
            standalone_matrix= load_standalone_accuracies(dataset_name, dist_name, NUM_GLOBAL_ROUNDS)
            corr_vals=[]
            for r_ in range(NUM_GLOBAL_ROUNDS-3, NUM_GLOBAL_ROUNDS):
                fedmpr_accs= acc_matrix[r_,:]
                stand_accs= standalone_matrix[r_,:]
                if np.all(fedmpr_accs==0) or np.all(stand_accs==0):
                    corr=0.0
                else:
                    corr= np.corrcoef(fedmpr_accs, stand_accs)[0,1]
                corr_vals.append(corr*100.0)
            corr_vals= np.array(corr_vals)
            avg_corr= np.mean(corr_vals)
            std_corr= np.std(corr_vals)

            results[(dataset_name, dist_name)] = (
                (avg3_mean, std3_mean),
                (avg3_max,  std3_max),
                (avg_corr,  std_corr)
            )

    final_result_file= os.path.join(RESULT_DIR, "final_results.txt")
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
    print(f"\n>>> FedMPR 结果已保存至: {final_result_file}")

if __name__=='__main__':
    main()
