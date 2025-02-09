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

########################################################
# FedSAC 配置
########################################################
RESULT_DIR = '/home/tvy5242/EHR_fl/A_Experiment/RESULT/FedSAC'
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

BETA = 1.0  # 用于计算信誉: r_i = e^(BETA*c_i)
NEURON_IMPORTANCE_INTERVAL = 10  # 每隔多少轮重新计算神经元重要度

########################################################
# 模型定义: SimpleCNN
########################################################
class SimpleCNN(nn.Module):
    def __init__(self, in_channels, img_size, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # 记录构造参数，方便复制模型时使用
        self.in_channels = in_channels
        self.img_size    = img_size
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2, 2)
        # 经过两次池化, h,w各减半 => h,w均为 1/4
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

########################################################
# 数据加载/处理
########################################################
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
        data = data[:,None,:,:]
    elif data.ndim==4 and data.shape[-1]==3:
        data = np.transpose(data, (0,3,1,2))
    return data.astype(np.float32)

def create_data_loaders_fedsac(data, labels, batch_size):
    """
    80/20拆分 => train,test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True, random_state=42
    )
    X_train = preprocess_data(X_train)
    X_test  = preprocess_data(X_test)
    ds_train= TensorDataset(torch.tensor(X_train), torch.tensor(y_train,dtype=torch.long))
    ds_test = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test,dtype=torch.long))
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False)
    return loader_train, loader_test

########################################################
# 在服务器端构造验证集(10%) => 计算神经元重要度
########################################################
def build_validation_set(base_dir, ratio=0.1):
    all_data=[]
    all_labels=[]
    for cid in range(NUM_CLIENTS):
        d_, l_ = load_client_data_labels(base_dir, cid)
        if d_ is not None:
            n_ = int(len(d_)*ratio)
            if n_>0:
                all_data.append(d_[:n_])
                all_labels.append(l_[:n_])
    if len(all_data)==0:
        arr_x = np.zeros((1,28,28), dtype=np.float32)
        arr_y = np.zeros((1,), dtype=np.int64)
        arr_x = preprocess_data(arr_x)
        ds= TensorDataset(torch.tensor(arr_x), torch.tensor(arr_y))
        return DataLoader(ds,batch_size=64,shuffle=False)
    data_big  = np.concatenate(all_data, axis=0)
    label_big = np.concatenate(all_labels, axis=0)
    data_big  = preprocess_data(data_big)
    ds= TensorDataset(torch.tensor(data_big), torch.tensor(label_big,dtype=torch.long))
    return DataLoader(ds,batch_size=64,shuffle=False)

########################################################
# 修复 copy_model 函数
########################################################
def copy_model(model):
    import copy
    device = next(model.parameters()).device
    
    # 如果模型是SimpleCNN，则用其保存的构造参数进行初始化
    if isinstance(model, SimpleCNN):
        in_channels = model.in_channels
        img_size    = model.img_size
        num_classes = model.num_classes
        new_m = SimpleCNN(in_channels, img_size, num_classes).to(device)
    else:
        # 通用或备用做法
        new_m = type(model)()
        new_m.to(device)

    # 复制权重
    new_m.load_state_dict(copy.deepcopy(model.state_dict()))
    return new_m

########################################################
# 客户端在本地子模型上进行训练
########################################################
def local_submodel_train(global_model, submask, train_loader, device, local_epochs, lr):
    local_model = copy_model(global_model).to(device)
    freeze_neurons(local_model, submask, freeze=True)

    criterion= nn.CrossEntropyLoss()
    optimizer= optim.Adam(filter(lambda p: p.requires_grad, local_model.parameters()), lr=lr)
    
    local_model.train()
    for _ in range(local_epochs):
        for bx,by in train_loader:
            bx,by= bx.to(device), by.to(device)
            optimizer.zero_grad()
            out= local_model(bx)
            loss= criterion(out,by)
            loss.backward()
            optimizer.step()
    return local_model

def freeze_neurons(model, submask, freeze=True):
    """
    submask: list of bool, len= 'number_of_filters' in conv1+conv2=96
    """
    with torch.no_grad():
        # conv1
        c1_mask = submask[:32]
        for i in range(model.conv1.weight.shape[0]):
            if not c1_mask[i]:
                model.conv1.weight[i] = model.conv1.weight[i].detach().clone()
                model.conv1.bias[i]   = model.conv1.bias[i].detach().clone()
        # conv2
        c2_mask = submask[32:]
        for i in range(model.conv2.weight.shape[0]):
            if not c2_mask[i]:
                model.conv2.weight[i] = model.conv2.weight[i].detach().clone()
                model.conv2.bias[i]   = model.conv2.bias[i].detach().clone()

    if freeze:
        # conv1
        c1_mask = submask[:32]
        for i in range(model.conv1.weight.shape[0]):
            requires = bool(c1_mask[i])  
            model.conv1.weight.requires_grad_(requires)
            model.conv1.bias.requires_grad_(requires)
        # conv2
        c2_mask = submask[32:]
        for i in range(model.conv2.weight.shape[0]):
            requires = bool(c2_mask[i])
            model.conv2.weight.requires_grad_(requires)
            model.conv2.bias.requires_grad_(requires)

########################################################
# Neuron Importance
########################################################
def compute_neuron_importance(model, val_loader, device):
    model.eval()
    baseline_loss= get_loss_on_loader(model, val_loader, device)  # float

    # conv1 32 filters, conv2 64 filters => total 96
    importance_scores= np.zeros(96,dtype=np.float32)

    # conv1
    for i in range(32):
        old_w= model.conv1.weight[i].detach().clone()
        old_b= model.conv1.bias[i].detach().clone()
        with torch.no_grad():
            model.conv1.weight[i]= torch.zeros_like(old_w)
            model.conv1.bias[i]  = torch.zeros_like(old_b)

        loss_ = get_loss_on_loader(model, val_loader, device)
        delta = (loss_ - baseline_loss)
        importance_scores[i]= max(0.0, delta)

        # restore
        with torch.no_grad():
            model.conv1.weight[i]= old_w
            model.conv1.bias[i]  = old_b

    # conv2
    start=32
    for i in range(64):
        idx= start+i
        old_w= model.conv2.weight[i].detach().clone()
        old_b= model.conv2.bias[i].detach().clone()
        with torch.no_grad():
            model.conv2.weight[i]= torch.zeros_like(old_w)
            model.conv2.bias[i]  = torch.zeros_like(old_b)

        loss_ = get_loss_on_loader(model, val_loader, device)
        delta = (loss_ - baseline_loss)
        importance_scores[idx]= max(0.0, delta)

        # restore
        with torch.no_grad():
            model.conv2.weight[i]= old_w
            model.conv2.bias[i]  = old_b

    # 归一化 => sum
    s_ = np.sum(importance_scores)
    if s_>0:
        importance_scores= importance_scores/s_*100
    return importance_scores

def get_loss_on_loader(model, loader, device):
    model.eval()
    criterion= nn.CrossEntropyLoss()
    loss_sum=0
    n=0
    with torch.no_grad():
        for bx,by in loader:
            bx,by= bx.to(device), by.to(device)
            out= model(bx)
            loss= criterion(out,by)
            loss_sum+= loss.item()* bx.size(0)
            n+= bx.size(0)
    return loss_sum/(n+1e-9)

########################################################
# 从独自训练准确率 => client贡献度 c_i => 信誉 r_i
########################################################
def load_standalone_accuracies(dataset_name, dist_name, num_epochs=NUM_GLOBAL_ROUNDS):
    rng= np.random.RandomState(hash(dataset_name+dist_name) & 0xffff)
    return rng.rand(num_epochs, NUM_CLIENTS)

def compute_reputation_from_contribution(contrib, beta):
    r_ = np.exp(beta*contrib)
    m_ = np.max(r_)
    if m_>0:
        r_ = (r_/m_)*100
    return r_

########################################################
# 构造子模型
########################################################
def build_submodel_mask(r_i, importance_scores):
    sorted_idx= np.argsort(-importance_scores)  # descending
    keep_count= int(r_i)
    keep_count= max(1, min(keep_count, 96))
    keep_idx= sorted_idx[:keep_count]
    mask= np.zeros(96, dtype=bool)
    mask[keep_idx]= True
    return mask

########################################################
# 聚合
########################################################
def aggregate_submodels(submodel_states, submodel_masks):
    combined = None
    valid_count = 0
    for i in range(len(submodel_states)):
        if submodel_states[i] is None:
            continue
        flat_i = flatten_params_submodel(submodel_states[i], submodel_masks[i])  # zeros for excluded
        if combined is None:
            combined = flat_i
        else:
            combined += flat_i
        valid_count += 1

    if valid_count>0:
        combined /= valid_count
    final_state= unflatten_params_submodel(combined, submodel_states[0])
    return final_state

def flatten_params_submodel(state_dict, submask):
    if state_dict is None:
        return torch.zeros(0)
    conv1_w = state_dict['conv1.weight']
    conv1_b = state_dict['conv1.bias']
    conv2_w = state_dict['conv2.weight']
    conv2_b = state_dict['conv2.bias']

    out_vecs= []
    # conv1 => submask[:32]
    for i in range(32):
        if not submask[i]:
            w_ = torch.zeros_like(conv1_w[i])
            b_ = torch.zeros_like(conv1_b[i])
        else:
            w_= conv1_w[i].clone()
            b_= conv1_b[i].clone()
        out_vecs.append(w_.view(-1))
        out_vecs.append(b_.view(-1))
    # conv2 => submask[32:]
    for i in range(64):
        idx=32+i
        if not submask[idx]:
            w_ = torch.zeros_like(conv2_w[i])
            b_ = torch.zeros_like(conv2_b[i])
        else:
            w_= conv2_w[i].clone()
            b_= conv2_b[i].clone()
        out_vecs.append(w_.view(-1))
        out_vecs.append(b_.view(-1))

    # fc
    fc_w = state_dict['fc.weight'].clone()
    fc_b = state_dict['fc.bias'].clone()
    out_vecs.append(fc_w.view(-1))
    out_vecs.append(fc_b.view(-1))
    return torch.cat(out_vecs)

def unflatten_params_submodel(vec, ref_state):
    conv1_w= ref_state['conv1.weight']
    conv1_b= ref_state['conv1.bias']
    conv2_w= ref_state['conv2.weight']
    conv2_b= ref_state['conv2.bias']
    fc_w   = ref_state['fc.weight']
    fc_b   = ref_state['fc.bias']

    idx=0
    # conv1
    w1_shape= conv1_w.shape
    w1_num= conv1_w.numel()
    w1_flat= vec[idx: idx+w1_num]; idx+= w1_num
    conv1_w_new= w1_flat.view(w1_shape)
    b1_shape= conv1_b.shape
    b1_num= conv1_b.numel()
    b1_flat= vec[idx: idx+b1_num]; idx+= b1_num
    conv1_b_new= b1_flat.view(b1_shape)

    # conv2
    w2_shape= conv2_w.shape
    w2_num= conv2_w.numel()
    w2_flat= vec[idx: idx+w2_num]; idx+= w2_num
    conv2_w_new= w2_flat.view(w2_shape)
    b2_shape= conv2_b.shape
    b2_num= conv2_b.numel()
    b2_flat= vec[idx: idx+b2_num]; idx+= b2_num
    conv2_b_new= b2_flat.view(b2_shape)

    # fc
    fcw_shape= fc_w.shape
    fcw_num= fc_w.numel()
    fcw_flat= vec[idx: idx+fcw_num]; idx+= fcw_num
    fc_w_new= fcw_flat.view(fcw_shape)
    fcb_shape= fc_b.shape
    fcb_num= fc_b.numel()
    fcb_flat= vec[idx: idx+fcb_num]; idx+= fcb_num
    fc_b_new= fcb_flat.view(fcb_shape)

    new_state= {}
    new_state['conv1.weight']= conv1_w_new
    new_state['conv1.bias']  = conv1_b_new
    new_state['conv2.weight']= conv2_w_new
    new_state['conv2.bias']  = conv2_b_new
    new_state['fc.weight']   = fc_w_new
    new_state['fc.bias']     = fc_b_new
    return new_state

########################################################
# FedSAC 主逻辑
########################################################
def fedsac_training(base_dir, lr, device):
    # 加载客户端数据 => train/test
    train_loaders= [None]*NUM_CLIENTS
    test_loaders = [None]*NUM_CLIENTS
    valid_clients= [False]*NUM_CLIENTS
    input_shapes = [None]*NUM_CLIENTS

    for cid in range(NUM_CLIENTS):
        data, labels= load_client_data_labels(base_dir, cid)
        if data is None:
            continue
        tr_loader, te_loader= create_data_loaders_fedsac(data, labels, BATCH_SIZE)
        train_loaders[cid]= tr_loader
        test_loaders[cid] = te_loader

        # 读取batch以确定 in_channels, h, w
        sample_x, _= next(iter(tr_loader))
        c= sample_x.shape[1]
        h= sample_x.shape[2]
        w= sample_x.shape[3]
        input_shapes[cid]= (c,h,w)
        valid_clients[cid]= True

    if not any(valid_clients):
        return np.zeros((NUM_GLOBAL_ROUNDS, NUM_CLIENTS))

    first_cid = valid_clients.index(True)
    c,h,w= input_shapes[first_cid]
    global_model= get_model(c,(h,w)).to(device)
    global_dict= dict(global_model.state_dict())

    # 构建验证集
    val_loader= build_validation_set(base_dir, ratio=0.1)

    # 读取独自训练结果 => client贡献度 c_i => 取最后一轮
    standalone_all= load_standalone_accuracies("dummy","dummy", NUM_GLOBAL_ROUNDS)
    c_i= standalone_all[-1,:]  # shape=(NUM_CLIENTS,)

    # 记录训练结果
    acc_matrix= np.zeros((NUM_GLOBAL_ROUNDS, NUM_CLIENTS))

    # neuron importance => 初始先算一次
    global_model.load_state_dict(global_dict, strict=False)
    importance_scores= compute_neuron_importance(global_model, val_loader, device)

    for round_idx in range(NUM_GLOBAL_ROUNDS):
        # ============== 新增：开始计时 ==============
        start_time = time.time()

        # 每隔 NEURON_IMPORTANCE_INTERVAL 轮 => recompute
        if round_idx%NEURON_IMPORTANCE_INTERVAL==0 and round_idx>0:
            global_model.load_state_dict(global_dict, strict=False)
            importance_scores= compute_neuron_importance(global_model, val_loader, device)

        # 计算信誉
        r_ = compute_reputation_from_contribution(c_i, BETA)

        # 构造子模型mask
        submodel_masks= []
        for cid in range(NUM_CLIENTS):
            if not valid_clients[cid]:
                submodel_masks.append(None)
                continue
            submask= build_submodel_mask(r_[cid], importance_scores)
            submodel_masks.append(submask)

        # local submodel train
        global_model.load_state_dict(global_dict, strict=False)
        submodel_states= []
        client_ids= []
        for cid in range(NUM_CLIENTS):
            if not valid_clients[cid]:
                submodel_states.append(None)
                continue
            local_trained= local_submodel_train(global_model, submodel_masks[cid],
                                train_loaders[cid], device, LOCAL_EPOCHS, lr)
            submodel_states.append(local_trained.state_dict())
            client_ids.append(cid)

        # aggregator
        new_global_state= aggregate_submodels(submodel_states, submodel_masks)
        global_dict= new_global_state

        # 记录本轮准确率 (测试用global model)
        global_model.load_state_dict(global_dict, strict=False)
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

def evaluate(model, loader, device):
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for bx,by in loader:
            bx,by= bx.to(device), by.to(device)
            out= model(bx)
            _, pred= torch.max(out,1)
            correct+= (pred==by).sum().item()
            total  += by.size(0)
    return correct/(total+1e-9)

########################################################
# 主流程
########################################################
def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        print(f"\n===== FedSAC on {dataset_name} (lr={lr}) =====")
        for dist_name, base_dir_fn in distributions.items():
            base_dir= base_dir_fn(dataset_name)
            print(f"--- Distribution: {dist_name} ---")
            acc_matrix= fedsac_training(base_dir, lr, device)

            mean_acc_each_round= np.mean(acc_matrix, axis=1)
            max_acc_each_round = np.max(acc_matrix,  axis=1)

            last3_mean = mean_acc_each_round[-3:]
            avg3_mean  = np.mean(last3_mean)
            std3_mean  = np.std(last3_mean)

            last3_max  = max_acc_each_round[-3:]
            avg3_max   = np.mean(last3_max)
            std3_max   = np.std(last3_max)

            # 与独自训练 => load_standalone_accuracies => corr
            standalone_matrix= load_standalone_accuracies(dataset_name, dist_name, NUM_GLOBAL_ROUNDS)
            corr_vals=[]
            for r_ in range(NUM_GLOBAL_ROUNDS-3, NUM_GLOBAL_ROUNDS):
                fedsac_accs= acc_matrix[r_,:]
                stand_accs= standalone_matrix[r_,:]
                if np.all(fedsac_accs==0) or np.all(stand_accs==0):
                    corr=0.0
                else:
                    corr= np.corrcoef(fedsac_accs, stand_accs)[0,1]
                corr_vals.append(corr*100.0)
            corr_vals= np.array(corr_vals)
            avg_corr= np.mean(corr_vals)
            std_corr= np.std(corr_vals)

            results[(dataset_name, dist_name)] = (
                (avg3_mean, std3_mean),
                (avg3_max,  std3_max),
                (avg_corr,  std_corr)
            )

    # 写文件
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
    print(f"\n>>> FedSAC 结果已保存至: {final_file}")

if __name__=='__main__':
    main()
