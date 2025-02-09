#!/usr/bin/env python3
import os
import numpy as np
from torchvision import datasets, transforms
from sklearn.decomposition import PCA

# 路径设置
RAW_DATA_PATH = '/home/tvy5242/EHR_fl/A_Experiment/DATA/raw'
TRAIN_PATH = '/home/tvy5242/EHR_fl/A_Experiment/DATA/train'

POW_PATH  = os.path.join(TRAIN_PATH, 'POW')
ECS_PATH  = os.path.join(TRAIN_PATH, 'ECS')
PECS_PATH = os.path.join(TRAIN_PATH, 'PECS')

NUM_CLIENTS = 10   # 客户端数量
DIM = 100          # PCA 降维后的维数

########################################
# 下载数据集
########################################
def download_datasets():
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    transform = transforms.ToTensor()
    # 下载 FashionMNIST
    _ = datasets.FashionMNIST(root=RAW_DATA_PATH, download=True, transform=transform)
    # 下载 CIFAR10
    _ = datasets.CIFAR10(root=RAW_DATA_PATH, download=True, transform=transform)
    print("数据集已下载到：", RAW_DATA_PATH)

########################################
# 加载数据
########################################
def get_fashionmnist_data():
    """加载 FashionMNIST，返回 (data, labels)。"""
    dataset = datasets.FashionMNIST(root=RAW_DATA_PATH, download=False, transform=transforms.ToTensor())
    data = dataset.data.numpy().astype(np.float32) / 255.0   # (N, 28, 28)
    labels = dataset.targets.numpy()                        # (N,)
    return data, labels

def get_cifar10_data():
    """加载 CIFAR10，返回 (data, labels)。"""
    dataset = datasets.CIFAR10(root=RAW_DATA_PATH, download=False, transform=transforms.ToTensor())
    data = dataset.data.astype(np.float32) / 255.0  # (N, 32, 32, 3)
    labels = np.array(dataset.targets)              # (N,)
    return data, labels

########################################
# PCA 降维
########################################
def apply_pca(data, n_components=DIM):
    """
    将原始图像数据展平后做 PCA 降维
    返回降维后的 data_pca, 以及 PCA 模型 pca
    """
    N = data.shape[0]
    flat = data.reshape(N, -1)     # 展平
    pca = PCA(n_components=n_components, random_state=42)
    data_pca = pca.fit_transform(flat)
    return data_pca, pca

########################################
# 计算全局分布参数
########################################
def compute_global_gaussian(data_pca):
    """返回 PCA 后数据的全局均值和协方差。"""
    global_mean = np.mean(data_pca, axis=0)
    global_cov  = np.cov(data_pca, rowvar=False)
    return global_mean, global_cov

########################################
# 划分函数：POW, ECS, PECS
########################################

def partition_data_pow(global_mean, global_cov, dataset_name,
                       data, labels, num_clients=NUM_CLIENTS,
                       alpha=1.0, n_half=30000):
    """
    POW 方法：在原始数据中随机抽取 n_half 个样本，
    然后按幂律分配到各客户端。
    """
    N = data.shape[0]
    # 先随机抽取 n_half 个样本索引
    indices = np.random.permutation(N)[:n_half]
    # 幂律
    weights = np.array([1.0 / ((i+1)**alpha) for i in range(num_clients)])
    weights /= np.sum(weights)
    client_counts = np.floor(weights * n_half).astype(int)
    # 若有分配不均的剩余
    diff = n_half - np.sum(client_counts)
    for i in range(diff):
        client_counts[i % num_clients] += 1
    # 将抽取到的索引再随机洗牌
    indices = np.random.permutation(indices)
    base_dir = os.path.join(POW_PATH, dataset_name)
    os.makedirs(base_dir, exist_ok=True)
    start = 0
    for cid in range(num_clients):
        count = client_counts[cid]
        client_inds = indices[start : start+count]
        start += count
        client_data   = data[client_inds]
        client_labels = labels[client_inds]
        client_dir = os.path.join(base_dir, f'client_{cid}')
        os.makedirs(client_dir, exist_ok=True)
        np.save(os.path.join(client_dir, 'data.npy'),   client_data)
        np.save(os.path.join(client_dir, 'labels.npy'), client_labels)
        with open(os.path.join(client_dir, 'kl.txt'), 'w') as f:
            f.write("NA")
    print(f"POW ({dataset_name}) done. client counts: {client_counts}")

def partition_data_ecs(global_mean, global_cov, dataset_name,
                       data, labels, data_pca, C,
                       num_clients=NUM_CLIENTS, n_half=30000):
    """
    ECS 方法：对 PCA 第一维排序后，取 n_half 个样本，
    然后均分给 num_clients 个客户端。
    """
    n_client = n_half // num_clients
    # 对 data_pca[:, 0] 从小到大排序
    sorted_indices = np.argsort(data_pca[:, 0])
    selected = sorted_indices[:n_half]
    base_dir = os.path.join(ECS_PATH, f'C{C}', dataset_name)
    os.makedirs(base_dir, exist_ok=True)
    for cid in range(num_clients):
        client_inds = selected[cid*n_client:(cid+1)*n_client]
        client_data   = data[client_inds]
        client_labels = labels[client_inds]
        client_dir = os.path.join(base_dir, f'client_{cid}')
        os.makedirs(client_dir, exist_ok=True)
        np.save(os.path.join(client_dir, 'data.npy'),   client_data)
        np.save(os.path.join(client_dir, 'labels.npy'), client_labels)
        with open(os.path.join(client_dir, 'kl.txt'), 'w') as f:
            f.write("NA")
    print(f"ECS (C={C}, {dataset_name}) done. each client has {n_client} samples.")

def partition_data_pecs(global_mean, global_cov, dataset_name,
                        data, labels, data_pca, C,
                        num_clients=NUM_CLIENTS, alpha=1.0, n_half=30000):
    """
    PECS 方法：对 PCA 第一维排序后，取 n_half 个样本，
    然后再按照幂律分配给客户端。
    """
    sorted_indices = np.argsort(data_pca[:, 0])
    selected = sorted_indices[:n_half]
    weights = np.array([1.0 / ((i+1)**alpha) for i in range(num_clients)])
    weights /= np.sum(weights)
    client_counts = np.floor(weights * n_half).astype(int)
    diff = n_half - np.sum(client_counts)
    for i in range(diff):
        client_counts[i % num_clients] += 1
    
    base_dir = os.path.join(PECS_PATH, f'C{C}', dataset_name)
    os.makedirs(base_dir, exist_ok=True)
    start = 0
    for cid in range(num_clients):
        count = client_counts[cid]
        client_inds = selected[start : start+count]
        start += count
        client_data   = data[client_inds]
        client_labels = labels[client_inds]
        client_dir = os.path.join(base_dir, f'client_{cid}')
        os.makedirs(client_dir, exist_ok=True)
        np.save(os.path.join(client_dir, 'data.npy'),   client_data)
        np.save(os.path.join(client_dir, 'labels.npy'), client_labels)
        with open(os.path.join(client_dir, 'kl.txt'), 'w') as f:
            f.write("NA")
    print(f"PECS (C={C}, {dataset_name}) done. client counts: {client_counts}")

########################################
# 主函数：下载数据，加载，PCA，划分
########################################
def main():
    download_datasets()  # 1. 下载数据
    
    # 定义要处理的数据集
    dataset_map = {
        'FashionMNIST': get_fashionmnist_data,
        'CIFAR10':      get_cifar10_data
    }
    
    for dataset_name, loader_fn in dataset_map.items():
        print(f"\n=== Processing {dataset_name} ===")
        data, labels = loader_fn()
        n_total = data.shape[0]
        n_half_dataset = n_total // 2
        print(f"{dataset_name}: total={n_total}, use half={n_half_dataset}")
        
        # 2. 对原始数据进行 PCA （仅用于 ECS/PECS 划分时的排序）
        data_pca, _ = apply_pca(data, n_components=DIM)
        global_mean, global_cov = compute_global_gaussian(data_pca)
        
        # 3. 调用划分函数
        partition_data_pow(global_mean, global_cov, dataset_name,
                           data, labels, num_clients=NUM_CLIENTS,
                           alpha=1.0, n_half=n_half_dataset)
        
        partition_data_ecs(global_mean, global_cov, dataset_name,
                           data, labels, data_pca, C=5,
                           num_clients=NUM_CLIENTS, n_half=n_half_dataset)
        
        partition_data_pecs(global_mean, global_cov, dataset_name,
                            data, labels, data_pca, C=2,
                            num_clients=NUM_CLIENTS, alpha=1.0, n_half=n_half_dataset)
        partition_data_pecs(global_mean, global_cov, dataset_name,
                            data, labels, data_pca, C=5,
                            num_clients=NUM_CLIENTS, alpha=1.0, n_half=n_half_dataset)
        partition_data_pecs(global_mean, global_cov, dataset_name,
                            data, labels, data_pca, C=10,
                            num_clients=NUM_CLIENTS, alpha=1.0, n_half=n_half_dataset)

if __name__ == '__main__':
    main()
