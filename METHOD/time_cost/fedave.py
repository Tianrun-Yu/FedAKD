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
#  FedAVE 配置超参数
##################################################################
RESULT_DIR = '/home/tvy5242/EHR_fl/A_Experiment/RESULT/FedAVE'
TRAIN_PATH = '/home/tvy5242/EHR_fl/A_Experiment/DATA/train'

POW_PATH   = os.path.join(TRAIN_PATH, 'POW')
ECS_PATH   = os.path.join(TRAIN_PATH, 'ECS')
PECS_PATH  = os.path.join(TRAIN_PATH, 'PECS')

# 客户端、轮次、批大小等
NUM_CLIENTS       = 10
NUM_GLOBAL_ROUNDS = 20
LOCAL_EPOCHS      = 1
BATCH_SIZE        = 32

# 学习率（按数据集区分）
learning_rates = {
    'FashionMNIST': 0.001,
    'CIFAR10':      0.005
}

# 部分梯度上传/下载比例
UPLOAD_FRACTION = 0.5   # 只上传一半的最重要梯度分量
DOWNLOAD_FRACTION_BASE = 0.3  # 一个基准下发比例
# 信誉参数
ALPHA = 0.9  # 移动平均平滑
BETA  = 1.0  # 计算下发比例时, tanh(BETA * r_i)

##################################################################
#  模型定义：SimpleCNN
##################################################################
class SimpleCNN(nn.Module):
    def __init__(self, in_channels, img_size, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2, 2)
        # 经过两次池化, h,w各缩小4倍
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
# 数据加载 & 预处理
##################################################################
def load_client_data_labels(base_dir, client_id):
    """
    加载 client_{id} 下的 data.npy 与 labels.npy
    """
    client_dir = os.path.join(base_dir, f'client_{client_id}')
    data_path  = os.path.join(client_dir, 'data.npy')
    labels_path= os.path.join(client_dir, 'labels.npy')
    if os.path.exists(data_path) and os.path.exists(labels_path):
        data   = np.load(data_path)
        labels = np.load(labels_path)
        return data, labels
    else:
        return None, None

def preprocess_data(data):
    """
    (N,H,W) => (N,1,H,W), (N,H,W,3) => (N,3,H,W)
    """
    if data.ndim == 3:
        data = data[:,None,:,:]
    elif data.ndim == 4 and data.shape[-1] == 3:
        data = np.transpose(data, (0,3,1,2))
    return data.astype(np.float32)

##################################################################
#  构建公共验证集
##################################################################
def build_public_val_loader(base_dir, ratio=0.1):
    """
    简单地从每个客户端抽取 ratio 比例的数据组成公共验证集
    """
    all_data = []
    all_labels = []
    for cid in range(NUM_CLIENTS):
        d_, l_ = load_client_data_labels(base_dir, cid)
        if d_ is not None:
            n_ = int(len(d_)*ratio)
            if n_>0:
                all_data.append(d_[:n_])
                all_labels.append(l_[:n_])
    if len(all_data)==0:
        # dummy
        arr_x = np.zeros((1,28,28), dtype=np.float32)
        arr_y = np.zeros((1,), dtype=np.int64)
        arr_x = preprocess_data(arr_x)
        ds = TensorDataset(torch.tensor(arr_x), torch.tensor(arr_y))
        return DataLoader(ds, batch_size=64, shuffle=False)
    data_big   = np.concatenate(all_data, axis=0)
    label_big  = np.concatenate(all_labels, axis=0)
    data_big   = preprocess_data(data_big)
    tx = torch.tensor(data_big, dtype=torch.float32)
    ty = torch.tensor(label_big, dtype=torch.long)
    ds = TensorDataset(tx, ty)
    return DataLoader(ds, batch_size=64, shuffle=False)

##################################################################
#  创建 DataLoader (train/test) - 简化
##################################################################
def create_data_loaders_for_fedave(data, labels, batch_size):
    """
    拆分 80%/20% => train/test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42, shuffle=True
    )
    X_train = torch.tensor(preprocess_data(X_train))
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test  = torch.tensor(preprocess_data(X_test))
    y_test  = torch.tensor(y_test, dtype=torch.long)

    ds_train= TensorDataset(X_train, y_train)
    ds_test = TensorDataset(X_test,  y_test)

    train_loader= DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ds_test,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

##################################################################
#  客户端本地训练
##################################################################
def client_local_train(model, train_loader, device, local_epochs, lr):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for _ in range(local_epochs):
        for bx,by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss= criterion(out,by)
            loss.backward()
            optimizer.step()

def evaluate(model, loader, device):
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for bx,by in loader:
            bx,by = bx.to(device), by.to(device)
            logits = model(bx)
            _, pred = torch.max(logits,1)
            total += by.size(0)
            correct+=(pred==by).sum().item()
    return correct/total if total>0 else 0.0

##################################################################
#  部分梯度上传/下载
##################################################################
def get_grad_from_models(old_model, new_model):
    """
    计算梯度近似: grad = new_params - old_params
    """
    grad_dict={}
    for (k1,v1),(k2,v2) in zip(old_model.state_dict().items(), new_model.state_dict().items()):
        grad_dict[k1] = (v2 - v1).clone().detach()
    return grad_dict

def flatten_grad(grad_dict):
    vecs=[]
    for k in sorted(grad_dict.keys()):
        vecs.append(grad_dict[k].view(-1))
    return torch.cat(vecs)

def unflatten_grad(vec, ref_dict):
    """
    将 1D vec 按照 ref_dict 的形状拆分
    """
    grad_dict={}
    idx=0
    for k in sorted(ref_dict.keys()):
        shape_ = ref_dict[k].shape
        numel  = ref_dict[k].numel()
        chunk  = vec[idx: idx+numel]
        idx+= numel
        grad_dict[k] = chunk.view(shape_)
    return grad_dict

def mask_top_k(vec, fraction):
    """
    对 1D 向量 vec保留 abs最大 fraction比例
    fraction in (0,1]
    """
    if fraction>=1.0:
        return vec
    length= vec.numel()
    k=int(length*fraction)
    if k<=0:
        return torch.zeros_like(vec)
    # 找到 abs最大 k
    abs_vec= vec.abs()
    # topk
    threshold = torch.topk(abs_vec, k)[0][-1]
    mask= (abs_vec>= threshold)
    vec_new= vec.clone()
    vec_new[~mask]=0.0
    return vec_new

##################################################################
#  FedAVE 主训练
##################################################################
def fedave_training(base_dir, lr, device):
    """
    返回 shape=(NUM_GLOBAL_ROUNDS, NUM_CLIENTS) 的测试准确率矩阵
    """
    train_loaders= [None]*NUM_CLIENTS
    test_loaders = [None]*NUM_CLIENTS
    valid_clients= [False]*NUM_CLIENTS
    input_shapes = [None]*NUM_CLIENTS

    # 读取客户端数据
    for cid in range(NUM_CLIENTS):
        data, labels= load_client_data_labels(base_dir,cid)
        if data is None:
            continue
        train_loader, test_loader= create_data_loaders_for_fedave(data, labels, BATCH_SIZE)
        train_loaders[cid]= train_loader
        test_loaders[cid] = test_loader
        # 检查 shape
        sample_x, _= next(iter(train_loader))
        c= sample_x.shape[1]
        h= sample_x.shape[2]
        w= sample_x.shape[3]
        input_shapes[cid]=(c,h,w)
        valid_clients[cid]= True

    # 若无客户端可用
    if not any(valid_clients):
        return np.zeros((NUM_GLOBAL_ROUNDS, NUM_CLIENTS))

    # 找到第一个可用客户端 => 用其 shape 创建global模型
    first_cid= valid_clients.index(True)
    c,h,w= input_shapes[first_cid]
    global_model= get_model(c,(h,w)).to(device)
    global_dict= dict(global_model.state_dict())

    # 构建公共验证集(由各客户端贡献若干数据合并)
    public_loader= build_public_val_loader(base_dir, ratio=0.1)

    # 初始化信誉度 => 均匀
    reputation= np.ones(NUM_CLIENTS,dtype=np.float32)
    rep_sum= np.sum(reputation)
    if rep_sum>0:
        reputation/= rep_sum

    # 记录准确率
    acc_matrix= np.zeros((NUM_GLOBAL_ROUNDS, NUM_CLIENTS))

    for round_idx in range(NUM_GLOBAL_ROUNDS):
        # ============== 新增：开始计时 ==============
        start_time = time.time()

        # 1) 客户端本地训练 => 上传部分梯度
        partial_grads= []
        client_ids   = []
        data_sizes   = []
        new_models   = {}
        for cid in range(NUM_CLIENTS):
            if not valid_clients[cid]:
                continue
            # 构建本地模型
            local_model= get_model(c,(h,w)).to(device)
            local_model.load_state_dict(global_dict, strict=False)

            # 备份 old
            old_model= get_model(c,(h,w)).to(device)
            old_model.load_state_dict(global_dict, strict=False)

            # 本地训练
            client_local_train(local_model, train_loaders[cid], device, LOCAL_EPOCHS, lr)

            # 计算梯度
            grad_dict= get_grad_from_models(old_model, local_model)
            grad_vec= flatten_grad(grad_dict)

            # 部分上传 => 只保留 fraction=UPLOAD_FRACTION
            partial= mask_top_k(grad_vec, UPLOAD_FRACTION)

            partial_grads.append(partial)
            client_ids.append(cid)
            data_sizes.append(len(train_loaders[cid].dataset))  # 本示例以样本数近似

            new_models[cid]= local_model.state_dict() # 保留本地新模型(未加“奖励”)

        if len(client_ids)==0:
            # 无上传
            break

        # 2) 服务器加权聚合 => 全局梯度
        sum_data= np.sum(data_sizes)
        combined= torch.zeros_like(partial_grads[0])
        for i,pv in enumerate(partial_grads):
            w_ = data_sizes[i]/(sum_data+1e-9)
            combined += pv* w_
        # 3) 服务器更新全局模型
        global_vec= flatten_grad(global_dict).to(device)
        global_vec+= combined
        new_global_dict= unflatten_grad(global_vec, global_dict)
        global_dict= new_global_dict

        # 4) 服务器根据“公共验证集”评估客户端表现 => 更新信誉
        rep_scores= np.zeros(NUM_CLIENTS,dtype=np.float32)
        for i,cid in enumerate(client_ids):
            local_model2= get_model(c,(h,w)).to(device)
            local_model2.load_state_dict(new_models[cid], strict=False)
            acc= evaluate(local_model2, public_loader, device)
            rep_scores[cid]= acc

        # 移动平均更新
        for cid in range(NUM_CLIENTS):
            old= reputation[cid]
            new= rep_scores[cid]
            updated= ALPHA*old + (1-ALPHA)*new
            reputation[cid]= updated
        # 归一化
        sum_rep= np.sum(reputation)
        if sum_rep>0:
            reputation/= sum_rep

        # 5) 服务器下发“奖励梯度” => fraction_i
        base= DOWNLOAD_FRACTION_BASE
        for i,cid in enumerate(client_ids):
            r_ = reputation[cid]
            fraction_i= base + (1-base)* np.tanh(BETA*r_)
            fraction_i= float(min(1.0,max(0.0, fraction_i)))
            reward_vec= mask_top_k(combined, fraction_i)

            local_vec = flatten_grad(new_models[cid]).to(device)
            local_vec += reward_vec
            final_dict= unflatten_grad(local_vec, new_models[cid])
            new_models[cid]= final_dict

        # 6) 客户端本地测试 => 记录准确率
        for cid in range(NUM_CLIENTS):
            if not valid_clients[cid]:
                acc_matrix[round_idx,cid]= 0.0
            else:
                m_ = get_model(c,(h,w)).to(device)
                if cid in new_models:
                    m_.load_state_dict(new_models[cid], strict=False)
                else:
                    m_.load_state_dict(global_dict, strict=False)
                te_acc= evaluate(m_, test_loaders[cid], device)
                acc_matrix[round_idx, cid]= te_acc

        # ============== 新增：结束计时并打印 ==============
        end_time = time.time()
        round_time = end_time - start_time
        print(f"Round {round_idx+1}/{NUM_GLOBAL_ROUNDS} took {round_time:.4f} seconds.")

    return acc_matrix

##################################################################
# 与独自训练对比 => 统计最后3轮 & 输出
##################################################################
def load_standalone_accuracies(dataset_name, dist_name, num_epochs=NUM_GLOBAL_ROUNDS):
    # 这里仍然用随机数演示
    rng= np.random.RandomState(hash(dataset_name+dist_name) & 0xffff)
    return rng.rand(num_epochs, NUM_CLIENTS)

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据集及对应目录
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
        lr = learning_rates.get(dataset_name, 0.001)
        print(f"\n===== FedAVE on {dataset_name} (lr={lr}) =====")
        for dist_name, base_dir_fn in distributions.items():
            base_dir= base_dir_fn(dataset_name)
            print(f"--- Distribution: {dist_name} ---")

            acc_matrix = fedave_training(base_dir, lr, device)
            # 形状 (NUM_GLOBAL_ROUNDS, NUM_CLIENTS)

            # 统计最后 3 轮
            mean_acc_each_round= np.mean(acc_matrix, axis=1)
            max_acc_each_round = np.max(acc_matrix,  axis=1)

            last3_mean_acc= mean_acc_each_round[-3:]
            avg3_mean_acc= np.mean(last3_mean_acc)
            std3_mean_acc= np.std(last3_mean_acc)

            last3_max_acc= max_acc_each_round[-3:]
            avg3_max_acc= np.mean(last3_max_acc)
            std3_max_acc= np.std(last3_max_acc)

            # 与独自训练做相关系数
            standalone_matrix= load_standalone_accuracies(dataset_name, dist_name, NUM_GLOBAL_ROUNDS)
            corr_vals=[]
            for r_ in range(NUM_GLOBAL_ROUNDS-3, NUM_GLOBAL_ROUNDS):
                fedave_accs= acc_matrix[r_,:]
                stand_accs= standalone_matrix[r_,:]
                if np.all(fedave_accs==0) or np.all(stand_accs==0):
                    corr=0.0
                else:
                    corr= np.corrcoef(fedave_accs, stand_accs)[0,1]
                corr_vals.append(corr*100.0)
            corr_vals= np.array(corr_vals)
            avg_corr= np.mean(corr_vals)
            std_corr= np.std(corr_vals)

            results[(dataset_name, dist_name)] = (
                (avg3_mean_acc, std3_mean_acc),
                (avg3_max_acc,  std3_max_acc),
                (avg_corr,      std_corr)
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

    print(f"\n>>> FedAVE 结果已保存至: {final_result_file}")

if __name__=='__main__':
    main()
