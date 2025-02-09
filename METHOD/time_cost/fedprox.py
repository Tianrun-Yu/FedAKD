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
#   FedProx 配置: 路径, 超参数等
##################################################################
RESULT_DIR = '/home/tvy5242/EHR_fl/A_Experiment/RESULT/FedProx'
TRAIN_PATH = '/home/tvy5242/EHR_fl/A_Experiment/DATA/train'

POW_PATH   = os.path.join(TRAIN_PATH, 'POW')
ECS_PATH   = os.path.join(TRAIN_PATH, 'ECS')
PECS_PATH  = os.path.join(TRAIN_PATH, 'PECS')

NUM_CLIENTS       = 10
NUM_GLOBAL_ROUNDS = 20
LOCAL_EPOCHS      = 1         # 每个客户端本地训练轮数
BATCH_SIZE        = 32

# 学习率: 不同数据集区分
learning_rates = {
    "FashionMNIST": 0.001,
    "CIFAR10":      0.005
}

# FedProx 超参数
MU = 0.01   # proximal 正则系数 mu

##################################################################
# 模型定义: SimpleCNN
##################################################################
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

##################################################################
# 数据加载与预处理
##################################################################
def load_client_data_labels(base_dir, client_id):
    """
    加载 client_{id} 下 data.npy 与 labels.npy
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
    (N,H,W)->(N,1,H,W) 或 (N,H,W,3)->(N,3,H,W)
    """
    if data.ndim==3:
        data= data[:,None,:,:]
    elif data.ndim==4 and data.shape[-1]==3:
        data= np.transpose(data, (0,3,1,2))
    return data.astype(np.float32)

def create_data_loaders_for_fedprox(data, labels, batch_size):
    """
    80/20拆分 => train_loader, test_loader
    """
    X_train, X_test, y_train, y_test= train_test_split(
        data, labels, test_size=0.2, shuffle=True, random_state=42
    )
    X_train= torch.tensor(X_train)
    y_train= torch.tensor(y_train, dtype=torch.long)
    X_test= torch.tensor(X_test)
    y_test= torch.tensor(y_test, dtype=torch.long)

    ds_train= TensorDataset(X_train,y_train)
    ds_test = TensorDataset(X_test, y_test)
    loader_train= DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    return loader_train, loader_test

##################################################################
# 本地训练 => FedProx
##################################################################
def local_train_prox(model, global_params, train_loader, device, local_epochs, lr, mu):
    """
    FedProx: 在本地优化  F_k(w) + mu/2 * ||w - w_global||^2
    model: 当前客户端的本地模型
    global_params: 全局模型参数 (w^t)
    mu: proximal 系数
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)  # 用SGD以便添加prox grad, Adam也可以

    # 先将 global_params 存入 CPU tensor，便于后面做 proximal
    global_params_buffer = {}
    for name, param in global_params.items():
        global_params_buffer[name] = param.detach().clone()

    for epoch in range(local_epochs):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # FedProx proximal term
            prox_term = 0.0
            # sum( mu/2 * || param - global_param ||^2 )
            for name, param in model.named_parameters():
                g_param= global_params_buffer[name].to(device)
                prox_term += (param - g_param).pow(2).sum()
            prox_term= (mu/2.0)* prox_term

            total_loss= loss + prox_term
            total_loss.backward()

            # 另一种方法: 在 backward 后手动加 grad += mu*(w - w^t)
            #  => 这里直接加到 loss 也可以

            optimizer.step()

def evaluate(model, test_loader, device):
    model.eval()
    correct= 0
    total= 0
    with torch.no_grad():
        for bx,by in test_loader:
            bx,by= bx.to(device), by.to(device)
            out= model(bx)
            _,pred= torch.max(out,1)
            total+= by.size(0)
            correct+= (pred==by).sum().item()
    return correct/total if total>0 else 0.0

##################################################################
# FedProx 训练逻辑
##################################################################
def fedprox_training(base_dir, lr, device, mu):
    # 加载客户端 data => train/test
    train_loaders= []
    test_loaders = []
    input_shapes = []
    valid_clients= [False]*NUM_CLIENTS

    for cid in range(NUM_CLIENTS):
        data, labels= load_client_data_labels(base_dir, cid)
        if data is None:
            train_loaders.append(None)
            test_loaders.append(None)
            input_shapes.append(None)
            continue
        data= preprocess_data(data)
        tr_loader, te_loader= create_data_loaders_for_fedprox(data, labels, BATCH_SIZE)
        train_loaders.append(tr_loader)
        test_loaders.append(te_loader)
        # shape
        sample_x, _= next(iter(tr_loader))
        c_= sample_x.shape[1]
        h_= sample_x.shape[2]
        w_= sample_x.shape[3]
        input_shapes.append((c_,h_,w_))
        valid_clients[cid]= True

    if not any(valid_clients):
        return np.zeros((NUM_GLOBAL_ROUNDS, NUM_CLIENTS))

    first_cid= valid_clients.index(True)
    c,h,w= input_shapes[first_cid]
    global_model= get_model(c,(h,w)).to(device)

    acc_matrix= np.zeros((NUM_GLOBAL_ROUNDS, NUM_CLIENTS))

    for round_idx in range(NUM_GLOBAL_ROUNDS):
        # ============== 新增：开始计时 ==============
        start_time = time.time()

        local_params_list= []
        # broadcast global => each client local
        global_params= {k: v.detach().clone() for k,v in global_model.state_dict().items()}

        # each client
        for cid in range(NUM_CLIENTS):
            if not valid_clients[cid]:
                continue
            # 复制 global model
            local_model= get_model(c,(h,w)).to(device)
            local_model.load_state_dict(global_params, strict=False)

            # local train (FedProx)
            local_train_prox(local_model, global_params, train_loaders[cid], device, LOCAL_EPOCHS, lr, mu)

            # collect
            local_params_list.append(local_model.state_dict())

        # aggregate => simple average
        if len(local_params_list)>0:
            new_state_dict= {}
            for key in local_params_list[0].keys():
                stacked= torch.stack([lp[key].float() for lp in local_params_list], dim=0)
                avg= stacked.mean(dim=0)
                new_state_dict[key]= avg
            global_model.load_state_dict(new_state_dict)

        # test => each client
        for cid in range(NUM_CLIENTS):
            if not valid_clients[cid]:
                acc_matrix[round_idx,cid]=0.0
            else:
                acc= evaluate(global_model, test_loaders[cid], device)
                acc_matrix[round_idx,cid]= acc

        # ============== 新增：结束计时并打印 ==============
        end_time = time.time()
        round_time = end_time - start_time
        print(f"Round {round_idx+1}/{NUM_GLOBAL_ROUNDS} took {round_time:.4f} seconds.")

    return acc_matrix

##################################################################
# 与独自训练比较 => 计算Corr => 输出
##################################################################
def load_standalone_accuracies(dataset_name, dist_name, num_epochs=NUM_GLOBAL_ROUNDS):
    """
    演示：返回随机数
    """
    rng= np.random.RandomState(hash(dataset_name+dist_name) & 0xffff)
    return rng.rand(num_epochs, NUM_CLIENTS)

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 定义分布目录
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
        lr= learning_rates.get(dataset_name, 0.001)
        mu= MU   # 你可以根据需要修改
        print(f"\n===== FedProx on {dataset_name} (lr={lr}, mu={mu}) =====")
        for dist_name, base_dir_fn in distributions.items():
            base_dir= base_dir_fn(dataset_name)
            print(f"--- Distribution: {dist_name} ---")

            acc_matrix= fedprox_training(base_dir, lr, device, mu)
            # shape=(NUM_GLOBAL_ROUNDS, NUM_CLIENTS)

            mean_each_round= np.mean(acc_matrix, axis=1)
            max_each_round = np.max(acc_matrix, axis=1)

            last3_mean= mean_each_round[-3:]
            avg3_mean= np.mean(last3_mean)
            std3_mean= np.std(last3_mean)

            last3_max= max_each_round[-3:]
            avg3_max= np.mean(last3_max)
            std3_max= np.std(last3_max)

            # 与独自训练 => corr
            standalone_matrix= load_standalone_accuracies(dataset_name, dist_name, NUM_GLOBAL_ROUNDS)
            corr_vals= []
            for r_ in range(NUM_GLOBAL_ROUNDS-3, NUM_GLOBAL_ROUNDS):
                fedprox_accs= acc_matrix[r_, :]
                stand_accs= standalone_matrix[r_, :]
                if np.all(fedprox_accs==0) or np.all(stand_accs==0):
                    corr=0.0
                else:
                    corr= np.corrcoef(fedprox_accs, stand_accs)[0,1]
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
    print(f"\n>>> FedProx 结果已保存至: {final_result_file}")

if __name__=='__main__':
    main()
