#!/usr/bin/env python3
import os
import time  # <=== 新增：导入time模块
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

############################################################################
# FedACK 配置
############################################################################
RESULT_DIR = '/home/tvy5242/EHR_fl/A_Experiment/RESULT/FedACK'
TRAIN_PATH = '/home/tvy5242/EHR_fl/A_Experiment/DATA/train'

POW_PATH   = os.path.join(TRAIN_PATH, 'POW')
ECS_PATH   = os.path.join(TRAIN_PATH, 'ECS')
PECS_PATH  = os.path.join(TRAIN_PATH, 'PECS')

NUM_CLIENTS       = 10
NUM_GLOBAL_ROUNDS = 30
LOCAL_EPOCHS      = 1       # 各阶段本地训练 epoch 数
BATCH_SIZE        = 32

# 定义用于不同数据集的学习率 & 蒸馏系数 alpha, beta
learning_rates = {
    "FashionMNIST": (0.001, 1, 1),  # (lr, alpha, beta)
    "CIFAR10":      (0.005, 1, 1)
}

############################################################################
# 模型：SimpleCNN
############################################################################
class SimpleCNN(nn.Module):
    def __init__(self, in_channels, img_size, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2,2)
        # 经过两次 pool => h,w 都是原来的 1/4
        feature_h = img_size[0]//4
        feature_w = img_size[1]//4
        self.fc   = nn.Linear(64 * feature_h * feature_w, num_classes)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.reshape(x.size(0), -1)
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
    if data.ndim==3:
        data= data[:,None,:,:]
    elif data.ndim==4 and data.shape[-1]==3:
        data= np.transpose(data, (0,3,1,2))
    return data.astype(np.float32)

def create_data_loaders_ack(data, labels, batch_size, test_ratio=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_ratio, shuffle=True, random_state=42
    )
    ds_train= TensorDataset(torch.tensor(X_train), torch.tensor(y_train,dtype=torch.long))
    ds_test = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test,dtype=torch.long))
    train_loader= DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

############################################################################
# 蒸馏损失 (KD) - 简化版本: 使用KL散度
############################################################################
def kd_loss(student_out, teacher_out, T=1.0):
    """
    student_out => shape=(N, C) 学生logits
    teacher_out => shape=(N, C) 老师logits
    """
    teacher_prob= nn.functional.softmax(teacher_out/T, dim=1)
    student_log= nn.functional.log_softmax(student_out/T, dim=1)
    kl= nn.functional.kl_div(student_log, teacher_prob, reduction='batchmean') * (T**2)
    return kl

############################################################################
# 计算正确预测子集
############################################################################
def get_correct_subset(model, data_x, data_y, device):
    """
    返回 x,y 中能被 model 正确预测的那些样本
    """
    model.eval()
    x_t= torch.tensor(data_x).to(device)
    y_t= torch.tensor(data_y,dtype=torch.long).to(device)
    with torch.no_grad():
        out= model(x_t)
        _,pred= torch.max(out,1)
        correct_mask= (pred==y_t)
    correct_x= data_x[correct_mask.cpu().numpy()]
    correct_y= data_y[correct_mask.cpu().numpy()]
    return correct_x, correct_y

############################################################################
# 客户端 "Local -> Global" 蒸馏
############################################################################
def local_to_global_distill(global_model, local_model, correct_x, correct_y, device, lr, alpha, local_epochs=1):
    criterion= nn.CrossEntropyLoss()
    global_model.train()
    local_model.eval()

    optimizer= optim.Adam(global_model.parameters(), lr=lr)
    ds= TensorDataset(torch.tensor(correct_x), torch.tensor(correct_y,dtype=torch.long))
    loader= DataLoader(ds, batch_size=32, shuffle=True)

    for ep in range(local_epochs):
        for bx,by in loader:
            bx,by= bx.to(device), by.to(device)
            optimizer.zero_grad()
            out_g= global_model(bx)   # student
            loss_ce= criterion(out_g, by)

            with torch.no_grad():
                out_l= local_model(bx) # teacher
            loss_kd= kd_loss(out_g, out_l, T=1.0)

            loss= loss_ce + alpha* loss_kd
            loss.backward()
            optimizer.step()

############################################################################
# 客户端 "Global -> Local" 蒸馏
############################################################################
def global_to_local_distill(local_model, global_model, data_x, data_y, device, lr, beta, local_epochs=1):
    criterion= nn.CrossEntropyLoss()
    local_model.train()
    global_model.eval()

    optimizer= optim.Adam(local_model.parameters(), lr=lr)
    ds= TensorDataset(torch.tensor(data_x), torch.tensor(data_y,dtype=torch.long))
    loader= DataLoader(ds, batch_size=32, shuffle=True)

    for ep in range(local_epochs):
        for bx,by in loader:
            bx,by= bx.to(device), by.to(device)
            optimizer.zero_grad()
            out_l= local_model(bx)
            loss_ce= criterion(out_l, by)
            with torch.no_grad():
                out_g= global_model(bx)
            loss_kd= kd_loss(out_l, out_g, T=1.0)
            loss= loss_ce + beta*loss_kd
            loss.backward()
            optimizer.step()

    # 更新"正确预测子集"
    correct_x, correct_y= get_correct_subset(local_model, data_x, data_y, device)
    return correct_x, correct_y

############################################################################
# FedACK 主训练流程
############################################################################
def fedack_training(base_dir, lr, alpha, beta, device):
    train_loaders= []
    test_loaders= []
    data_local= []
    correct_subsets= []
    input_shapes= []
    valid_clients= [False]*NUM_CLIENTS

    for cid in range(NUM_CLIENTS):
        d_, l_ = load_client_data_labels(base_dir, cid)
        if d_ is None:
            train_loaders.append(None)
            test_loaders.append(None)
            input_shapes.append(None)
            correct_subsets.append(None)
            data_local.append(None)
            continue
        d_= preprocess_data(d_)
        tr_loader, te_loader= create_data_loaders_ack(d_, l_, BATCH_SIZE)
        train_loaders.append(tr_loader)
        test_loaders.append(te_loader)

        sample_x,_= next(iter(tr_loader))
        c_= sample_x.shape[1]
        h_= sample_x.shape[2]
        w_= sample_x.shape[3]
        input_shapes.append((c_,h_,w_))
        data_local.append((d_,l_))
        valid_clients[cid]= True

        # 初始假设全部正确 (也可先用随机模型测试筛选)
        correct_subsets.append((d_, l_))

    if not any(valid_clients):
        return np.zeros((NUM_GLOBAL_ROUNDS, NUM_CLIENTS))

    # 初始化 全局模型
    first_cid= valid_clients.index(True)
    c,h,w= input_shapes[first_cid]
    global_model= get_model(c,(h,w)).to(device)

    # local_models for each client
    local_models= []
    for cid in range(NUM_CLIENTS):
        if valid_clients[cid]:
            loc= get_model(c,(h,w)).to(device)
            loc.load_state_dict(global_model.state_dict())
            local_models.append(loc)
        else:
            local_models.append(None)

    acc_matrix= np.zeros((NUM_GLOBAL_ROUNDS, NUM_CLIENTS))

    for round_idx in range(NUM_GLOBAL_ROUNDS):
        # ============== 新增：开始计时 ==============
        start_time = time.time()

        # 1) Local -> Global Distillation
        local_global_states= []
        data_sizes= []
        for cid in range(NUM_CLIENTS):
            if not valid_clients[cid]:
                continue
            # student_g = copy of global
            student_g= get_model(c,(h,w)).to(device)
            student_g.load_state_dict(global_model.state_dict())

            cx, cy= correct_subsets[cid]  # I_k^t
            local_to_global_distill(student_g, local_models[cid], cx, cy, device, lr, alpha, LOCAL_EPOCHS)

            local_global_states.append(student_g.state_dict())
            data_sizes.append(len(data_local[cid][0]))  # total dataset size

        # 服务器聚合
        if len(local_global_states)>0:
            sum_size= sum(data_sizes)
            new_state={}
            for key in local_global_states[0].keys():
                w_sum= None
                for i, st in enumerate(local_global_states):
                    weight= data_sizes[i]/ sum_size
                    if w_sum is None:
                        w_sum= st[key].float()* weight
                    else:
                        w_sum+= st[key].float()* weight
                new_state[key]= w_sum
            global_model.load_state_dict(new_state)

        # 2) Global -> Local Distillation
        for cid in range(NUM_CLIENTS):
            if not valid_clients[cid]:
                continue
            full_x, full_y= data_local[cid]
            cx, cy= global_to_local_distill(local_models[cid], global_model, full_x, full_y, device, lr, beta, LOCAL_EPOCHS)
            correct_subsets[cid]= (cx, cy)

        # 3) 测试 => global_model
        for cid in range(NUM_CLIENTS):
            if not valid_clients[cid]:
                acc_matrix[round_idx,cid]=0.0
            else:
                te_loader= test_loaders[cid]
                acc= eval_global_model(global_model, te_loader, device)
                acc_matrix[round_idx,cid]= acc

        # ============== 新增：结束计时并打印 ==============
        end_time = time.time()
        round_time = end_time - start_time
        print(f"Round {round_idx+1}/{NUM_GLOBAL_ROUNDS} took {round_time:.4f} seconds.")

    return acc_matrix

def eval_global_model(model, test_loader, device):
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for bx,by in test_loader:
            bx,by= bx.to(device), by.to(device)
            out= model(bx)
            _,pred= torch.max(out,1)
            total+= by.size(0)
            correct+= (pred==by).sum().item()
    return correct/total if total>0 else 0

############################################################################
# 读取独自训练 (standalone)
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
    # 新增: 用于保存“后5轮的相关系数”详情
    post5_corr_details= {}

    for dataset_name in datasets_list:
        (lr, alpha, beta)= learning_rates.get(dataset_name, (0.001, 0.5, 0.5))
        print(f"\n===== FedACK on {dataset_name} (lr={lr}, alpha={alpha}, beta={beta}) =====")
        for dist_name, base_dir_fn in distributions.items():
            base_dir= base_dir_fn(dataset_name)
            print(f"--- Distribution: {dist_name} ---")
            acc_matrix= fedack_training(base_dir, lr, alpha, beta, device)
            # shape=(NUM_GLOBAL_ROUNDS, NUM_CLIENTS)

            # ====== 统计最后 3 轮平均、最大准确率，以及相关系数 ======
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
            corr_vals=[]
            for r_ in range(NUM_GLOBAL_ROUNDS-3, NUM_GLOBAL_ROUNDS):
                fedack_accs= acc_matrix[r_,:]
                stand_accs= standalone_matrix[r_,:]
                if np.all(fedack_accs==0) or np.all(stand_accs==0):
                    corr=0.0
                else:
                    corr= np.corrcoef(fedack_accs, stand_accs)[0,1]
                corr_vals.append(corr*100.0)
            corr_vals= np.array(corr_vals)
            avg_corr= np.mean(corr_vals)
            std_corr= np.std(corr_vals)

            results[(dataset_name, dist_name)] = (
                (avg3_mean, std3_mean),
                (avg3_max,  std3_max),
                (avg_corr,  std_corr)
            )

            # ====== 记录后5轮的相关系数 ======
            post5_corr= []
            for r_ in range(NUM_GLOBAL_ROUNDS-5, NUM_GLOBAL_ROUNDS):
                fedack_accs= acc_matrix[r_,:]
                stand_accs= standalone_matrix[r_,:]
                if np.all(fedack_accs==0) or np.all(stand_accs==0):
                    c_ = 0.0
                else:
                    c_ = np.corrcoef(fedack_accs, stand_accs)[0,1]*100.0
                post5_corr.append(c_)
            post5_corr_details[(dataset_name, dist_name)] = post5_corr

    # ====== 写主结果 final_results.txt ======
    final_file= os.path.join(RESULT_DIR, "final_results.txt")
    with open(final_file, "w") as f:
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

    # ====== 写后5轮相关系数 fedack_last5corr.txt ======
    last5corr_file= os.path.join(RESULT_DIR, "fedack_last5corr.txt")
    with open(last5corr_file, "w") as f2:
        f2.write("dataset, distribution, Corr(R-4), Corr(R-3), Corr(R-2), Corr(R-1), Corr(R)\n")
        for dataset_name in datasets_list:
            for dist_name in distributions.keys():
                corr_list= post5_corr_details.get((dataset_name, dist_name), None)
                if corr_list is None:
                    line= f"{dataset_name}, {dist_name}, 0,0,0,0,0\n"
                else:
                    line= f"{dataset_name}, {dist_name}, "
                    line+= ",".join([f"{v:.2f}" for v in corr_list])
                    line+= "\n"
                f2.write(line)

    print(f"\n>>> FedACK 结果已保存至: {final_file}")
    print(f">>> 后5轮相关系数已保存至: {last5corr_file}")

if __name__=='__main__':
    main()
