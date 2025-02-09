#!/usr/bin/env python3
import os
import numpy as np
from torchvision import datasets, transforms
from sklearn.decomposition import PCA

RAW_DATA_PATH = ''
TRAIN_PATH = ''

POW_PATH  = os.path.join(TRAIN_PATH, 'POW')
ECS_PATH  = os.path.join(TRAIN_PATH, 'ECS')
PECS_PATH = os.path.join(TRAIN_PATH, 'PECS')

NUM_CLIENTS = 10
DIM = 100  # Target dimension after PCA


def download_datasets():
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    transform = transforms.ToTensor()
    # Download FashionMNIST
    _ = datasets.FashionMNIST(root=RAW_DATA_PATH, download=True, transform=transform)
    # Download CIFAR10
    _ = datasets.CIFAR10(root=RAW_DATA_PATH, download=True, transform=transform)
    print("Datasets have been downloaded to:", RAW_DATA_PATH)


########################################
# Load data
########################################
def get_fashionmnist_data():
    """Return (data, labels) for FashionMNIST."""
    dataset = datasets.FashionMNIST(root=RAW_DATA_PATH, download=False, transform=transforms.ToTensor())
    data = dataset.data.numpy().astype(np.float32) / 255.0
    labels = dataset.targets.numpy()
    return data, labels

def get_cifar10_data():
    """Return (data, labels) for CIFAR10."""
    dataset = datasets.CIFAR10(root=RAW_DATA_PATH, download=False, transform=transforms.ToTensor())
    data = dataset.data.astype(np.float32) / 255.0
    labels = np.array(dataset.targets)
    return data, labels


########################################
# PCA
########################################
def apply_pca(data, n_components=DIM):
    """
    Flatten images and apply PCA.
    Return reduced data_pca and the PCA model.
    """
    N = data.shape[0]
    flat = data.reshape(N, -1)
    pca = PCA(n_components=n_components, random_state=42)
    data_pca = pca.fit_transform(flat)
    return data_pca, pca

def compute_global_gaussian(data_pca):
    """
    Compute global mean and covariance of PCA-transformed data.
    """
    global_mean = np.mean(data_pca, axis=0)
    global_cov = np.cov(data_pca, rowvar=False)
    return global_mean, global_cov

########################################
# Partition functions: POW, ECS, PECS
########################################

def partition_data_pow(global_mean, global_cov, dataset_name,
                       data, labels, num_clients=NUM_CLIENTS,
                       alpha=1.0, n_half=30000):
    """
    POW: Randomly select n_half samples from the original data,
    then assign them to clients according to a power-law distribution.
    """
    N = data.shape[0]
    # Randomly select n_half sample indices
    indices = np.random.permutation(N)[:n_half]
    # Power law
    weights = np.array([1.0 / ((i+1)**alpha) for i in range(num_clients)])
    weights /= np.sum(weights)
    client_counts = np.floor(weights * n_half).astype(int)
    remainder = n_half - np.sum(client_counts)
    for i in range(remainder):
        client_counts[i % num_clients] += 1

    # Shuffle selected indices again
    indices = np.random.permutation(indices)
    base_dir = os.path.join(POW_PATH, dataset_name)
    os.makedirs(base_dir, exist_ok=True)
    start = 0
    for cid in range(num_clients):
        count = client_counts[cid]
        client_inds = indices[start : start+count]
        start += count
        client_data = data[client_inds]
        client_labels = labels[client_inds]
        client_dir = os.path.join(base_dir, f'client_{cid}')
        os.makedirs(client_dir, exist_ok=True)
        np.save(os.path.join(client_dir, 'data.npy'), client_data)
        np.save(os.path.join(client_dir, 'labels.npy'), client_labels)
        with open(os.path.join(client_dir, 'kl.txt'), 'w') as f:
            f.write("NA")
    print(f"POW ({dataset_name}) completed. Client counts: {client_counts}")

def partition_data_ecs(global_mean, global_cov, dataset_name,
                       data, labels, data_pca, C,
                       num_clients=NUM_CLIENTS, n_half=30000):
    """
    ECS: Sort data by the first principal component in ascending order,
    select n_half samples, and split them equally among num_clients.
    """
    n_client = n_half // num_clients
    sorted_indices = np.argsort(data_pca[:, 0])
    selected = sorted_indices[:n_half]
    base_dir = os.path.join(ECS_PATH, f'C{C}', dataset_name)
    os.makedirs(base_dir, exist_ok=True)
    for cid in range(num_clients):
        client_inds = selected[cid*n_client : (cid+1)*n_client]
        client_data = data[client_inds]
        client_labels = labels[client_inds]
        client_dir = os.path.join(base_dir, f'client_{cid}')
        os.makedirs(client_dir, exist_ok=True)
        np.save(os.path.join(client_dir, 'data.npy'), client_data)
        np.save(os.path.join(client_dir, 'labels.npy'), client_labels)
        with open(os.path.join(client_dir, 'kl.txt'), 'w') as f:
            f.write("NA")
    print(f"ECS (C={C}, {dataset_name}) completed. Each client has {n_client} samples.")

def partition_data_pecs(global_mean, global_cov, dataset_name,
                        data, labels, data_pca, C,
                        num_clients=NUM_CLIENTS, alpha=1.0, n_half=30000):
    """
    PECS: Sort data by the first principal component in ascending order,
    select n_half samples, then assign them to clients according to a power-law distribution.
    """
    sorted_indices = np.argsort(data_pca[:, 0])
    selected = sorted_indices[:n_half]
    weights = np.array([1.0 / ((i+1)**alpha) for i in range(num_clients)])
    weights /= np.sum(weights)
    client_counts = np.floor(weights * n_half).astype(int)
    remainder = n_half - np.sum(client_counts)
    for i in range(remainder):
        client_counts[i % num_clients] += 1

    base_dir = os.path.join(PECS_PATH, f'C{C}', dataset_name)
    os.makedirs(base_dir, exist_ok=True)
    start = 0
    for cid in range(num_clients):
        count = client_counts[cid]
        client_inds = selected[start : start+count]
        start += count
        client_data = data[client_inds]
        client_labels = labels[client_inds]
        client_dir = os.path.join(base_dir, f'client_{cid}')
        os.makedirs(client_dir, exist_ok=True)
        np.save(os.path.join(client_dir, 'data.npy'), client_data)
        np.save(os.path.join(client_dir, 'labels.npy'), client_labels)
        with open(os.path.join(client_dir, 'kl.txt'), 'w') as f:
            f.write("NA")
    print(f"PECS (C={C}, {dataset_name}) completed. Client counts: {client_counts}")


def main():
    # Step 1: Download datasets
    download_datasets()

    # Define datasets to process
    dataset_map = {
        'FashionMNIST': get_fashionmnist_data,
        'CIFAR10':      get_cifar10_data
    }
    
    for dataset_name, loader_fn in dataset_map.items():
        print(f"\n=== Processing {dataset_name} ===")
        data, labels = loader_fn()
        n_total = data.shape[0]
        n_half_dataset = n_total // 2
        print(f"{dataset_name}: total={n_total}, half={n_half_dataset}")
        
        # Step 2: PCA (used for ECS/PECS sorting)
        data_pca, _ = apply_pca(data, n_components=DIM)
        global_mean, global_cov = compute_global_gaussian(data_pca)
        
        # Step 3: Partition
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
