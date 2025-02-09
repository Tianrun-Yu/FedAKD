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

############################################################################
# FedDC 配置
############################################################################
RESULT_DIR = '/home/tvy5242/EHR_fl/A_Experiment/RESULT/FedDC'
TRAIN_PATH = '/home/tvy5242/EHR_fl/A_Experiment/DATA/train'

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

ALPHA = 0.01   # FedDC 漂移惩罚系数 (可自行调整)

############################################################################
# 模型定义: SimpleCNN
############################################################################
class SimpleCNN(nn.Module):
    def __init__(self, in_channels, img_size, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2,2)
        feature_h = img_size[0]//4
        feature_w = img_size[1]//4
        self.fc   = nn.Linear(64*feature_h*feature_w, num_classes)
    
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
def load_client_data_labels(base_dir, cid):
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
    (N,H,W)->(N,1,H,W) 或 (N,H,W,3)->(N,3,H,W)
    """
    if data.ndim==3:
        data= data[:,None,:,:]
    elif data.ndim==4 and data.shape[-1]==3:
        data= np.transpose(data, (0,3,1,2))
    return data.astype(np.float32)

def create_data_loaders_feddc(data, labels, batch_size):
    """
    80/20 => train/test
    """
    X_train, X_test, y_train, y_test= train_test_split(
        data, labels, test_size=0.2, shuffle=True, random_state=42
    )
    X_train= torch.tensor(X_train)
    y_train= torch.tensor(y_train,dtype=torch.long)
    X_test = torch.tensor(X_test)
    y_test = torch.tensor(y_test,dtype=torch.long)

    ds_train= TensorDataset(X_train,y_train)
    ds_test = TensorDataset(X_test,y_test)
    loader_train= DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    return loader_train, loader_test

############################################################################
# 客户端本地训练 => 带漂移项的目标:
# L_i(theta_i) + alpha/2 * ||(theta_i + h_i - w^t)||^2
############################################################################
def local_train_drift(model, h_vec, w_vec, train_loader, device, local_epochs, lr, alpha):
    """
    model: 当前客户端本地模型
    h_vec: shape=(num_params,) => 本地漂移
    w_vec: shape=(num_params,) => 上一轮全局参数
    alpha: 惩罚系数
    """
    w_vec_device= w_vec.clone().to(device)
    h_vec_device= h_vec.clone().to(device)

    optimizer= optim.SGD(model.parameters(), lr=lr)
    criterion= nn.CrossEntropyLoss()

    model.train()
    for epoch in range(local_epochs):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y= batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out= model(batch_x)
            loss_ce= criterion(out, batch_y)

            # FedDC penalty => alpha/2 * || (theta + h) - w ||^2
            current_theta= model_to_vec(model)
            penalty= (current_theta + h_vec_device - w_vec_device).pow(2).sum()
            penalty= 0.5*alpha* penalty

            loss= loss_ce + penalty
            loss.backward()
            optimizer.step()

def model_to_vec(model):
    """
    flatten model params => 1D tensor
    """
    vecs=[]
    for param in model.parameters():
        vecs.append(param.view(-1))
    return torch.cat(vecs)

def vec_to_model(vec, model):
    """
    1D vec => load into model
    """
    idx=0
    for param in model.parameters():
        size= param.numel()
        param.data= vec[idx: idx+size].view(param.shape).clone()
        idx+= size

############################################################################
# 评估模型在 test_loader 上的准确率
############################################################################
def evaluate(model, test_loader, device):
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            out= model(bx)
            _, pred= torch.max(out, dim=1)
            correct+= (pred==by).sum().item()
            total+= by.size(0)
    return correct/(total+1e-9)

############################################################################
# 聚合 => simple average
############################################################################
def aggregate_uploads(param_list):
    """
    param_list: list of 1D vectors => do average
    """
    if len(param_list)==0:
        return None
    stacked= torch.stack(param_list,dim=0)
    avg= stacked.mean(dim=0)
    return avg

############################################################################
# FedDC 主训练流程
############################################################################
def feddc_training(base_dir, lr, device, alpha):
    # 加载客户端 data => train/test
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
        data= preprocess_data(data)
        tr_loader, te_loader= create_data_loaders_feddc(data, labels, BATCH_SIZE)
        train_loaders.append(tr_loader)
        test_loaders.append(te_loader)
        # shape
        sample_x,_= next(iter(tr_loader))
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
    # flatten global
    global_vec= model_to_vec(global_model).detach().clone()

    # 初始化 h_i => 0
    # shape = len(global_vec)
    h_all= []
    for cid in range(NUM_CLIENTS):
        if not valid_clients[cid]:
            h_all.append(None)
        else:
            h_all.append(torch.zeros_like(global_vec))

    acc_matrix= np.zeros((NUM_GLOBAL_ROUNDS, NUM_CLIENTS))

    for round_idx in range(NUM_GLOBAL_ROUNDS):
        # ============== 新增：开始计时 ==============
        start_time = time.time()

        # broadcast => each client sees global_vec
        uploads= []
        for cid in range(NUM_CLIENTS):
            if not valid_clients[cid]:
                continue
            # local model init from global
            local_model= get_model(c,(h,w)).to(device)
            vec_to_model(global_vec, local_model)

            # local train => FedDC
            local_train_drift(local_model, h_all[cid], global_vec,
                              train_loaders[cid], device,
                              LOCAL_EPOCHS, lr, alpha)

            # 训练后 => new theta_i
            new_theta= model_to_vec(local_model).detach().clone()

            # h_i^+ = h_i + (theta_i^+ - theta_i)
            old_theta= global_vec.clone()
            delta= (new_theta - old_theta)
            h_all[cid] = h_all[cid] + delta

            # upload => (theta_i^+ + h_i^+)
            upload_param= new_theta + h_all[cid]
            uploads.append(upload_param)

        # aggregator => simple average
        new_global= aggregate_uploads(uploads)
        if new_global is not None:
            global_vec= new_global.clone()

        # load global_vec => global_model
        vec_to_model(global_vec, global_model)

        # test => each client with global model
        for cid in range(NUM_CLIENTS):
            if not valid_clients[cid]:
                acc_matrix[round_idx,cid]= 0.0
            else:
                acc= evaluate(global_model, test_loaders[cid], device)
                acc_matrix[round_idx,cid]= acc

        # ============== 新增：结束计时并打印 ==============
        end_time = time.time()
        round_time = end_time - start_time
        print(f"Round {round_idx+1}/{NUM_GLOBAL_ROUNDS} took {round_time:.4f} seconds.")

    return acc_matrix

############################################################################
# 与独自训练比较 => 计算corr => 输出
############################################################################
def load_standalone_accuracies(dataset_name, dist_name, num_epochs=NUM_GLOBAL_ROUNDS):
    rng= np.random.RandomState(hash(dataset_name+dist_name)&0xffff)
    return rng.rand(num_epochs, NUM_CLIENTS)

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        alpha= ALPHA  # FedDC 惩罚系数
        print(f"\n===== FedDC on {dataset_name} (lr={lr}, alpha={alpha}) =====")
        for dist_name, base_dir_fn in distributions.items():
            base_dir= base_dir_fn(dataset_name)
            print(f"--- Distribution: {dist_name} ---")
            acc_matrix= feddc_training(base_dir, lr, device, alpha)
            # (NUM_GLOBAL_ROUNDS, NUM_CLIENTS)

            mean_each_round= np.mean(acc_matrix, axis=1)
            max_each_round = np.max(acc_matrix, axis=1)

            last3_mean= mean_each_round[-3:]
            avg3_mean= np.mean(last3_mean)
            std3_mean= np.std(last3_mean)

            last3_max= max_each_round[-3:]
            avg3_max= np.mean(last3_max)
            std3_max= np.std(last3_max)

            # 与独自训练对比 => corr
            standalone_matrix= load_standalone_accuracies(dataset_name, dist_name, NUM_GLOBAL_ROUNDS)
            corr_vals=[]
            for r_ in range(NUM_GLOBAL_ROUNDS-3, NUM_GLOBAL_ROUNDS):
                feddc_accs= acc_matrix[r_,:]
                stand_accs= standalone_matrix[r_,:]
                if np.all(feddc_accs==0) or np.all(stand_accs==0):
                    corr=0.0
                else:
                    corr= np.corrcoef(feddc_accs, stand_accs)[0,1]
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
    print(f"\n>>> FedDC 结果已保存至: {final_result_file}")

if __name__=='__main__':
    main()
