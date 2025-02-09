#!/usr/bin/env python3
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Directory paths
TRAIN_PATH = ''
RESULT_DIR = ''

# Root directories for different data distributions (must match the structure used in data partitioning)
POW_PATH   = os.path.join(TRAIN_PATH, 'POW')
ECS_PATH   = os.path.join(TRAIN_PATH, 'BCS')
PECS_PATH  = os.path.join(TRAIN_PATH, 'ICS')

NUM_CLIENTS = 10       # Number of clients for each distribution
NUM_EPOCHS  = 20       # Number of epochs per client
BATCH_SIZE  = 32       # Batch size

# Learning rates for different datasets
learning_rates = {
    "FashionMNIST": 0.0001,
    "CIFAR10":      0.0005
}

########################################
# Define a simple CNN model (for 10-class classification)
########################################
class SimpleCNN(nn.Module):
    def __init__(self, in_channels, img_size, num_classes=10):
        """
        :param in_channels: number of input channels (1 or 3)
        :param img_size: (height, width)
        :param num_classes: number of classes, default 10
        """
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2, 2)
        # After two 2×2 poolings, height and width are each reduced by a factor of 4
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

########################################
# Data loading and preprocessing
########################################
def load_client_data_labels(base_dir, client_id):
    """
    Load data and labels for the specified client.
    It is assumed each client directory has data.npy and labels.npy.
    """
    client_dir = os.path.join(base_dir, f'client_{client_id}')
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
    Convert data to a format usable by a CNN:
      - If the shape is (N, H, W), add a channel dimension -> (N, 1, H, W).
      - If the shape is (N, H, W, 3), transpose to (N, 3, H, W).
    """
    if len(data.shape) == 3:
        # (N, H, W)
        data = data[:, None, :, :]
    elif len(data.shape) == 4 and data.shape[-1] == 3:
        # (N, H, W, 3) => (N, 3, H, W)
        data = np.transpose(data, (0, 3, 1, 2))
    return data.astype(np.float32)

########################################
# Client training and returning per-epoch test accuracy
########################################
def train_and_evaluate_client(data, labels, device,
                              num_epochs=NUM_EPOCHS,
                              batch_size=BATCH_SIZE,
                              lr=0.001):
    """
    Train on a single client's data, return a list of test accuracies of length num_epochs.
    The test set is evaluated after each epoch.
    """
    # Split train/test  (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42, shuffle=True
    )
    # Convert to torch tensors
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test  = torch.tensor(X_test)
    y_test  = torch.tensor(y_test, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset  = TensorDataset(X_test,  y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    
    # Obtain input shape
    _, C, H, W = X_train.shape
    model = get_model(C, (H, W), num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    epoch_acc_list = []  # record test accuracy for each epoch
    
    for epoch in range(num_epochs):
        # ---- Training phase ----
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # ---- Testing phase ----
        model.eval()
        correct = 0
        total   = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs, 1)
                total   += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        acc = correct / total if total > 0 else 0.0
        epoch_acc_list.append(acc)
    
    return epoch_acc_list

########################################
# Main training flow: iterate over datasets, distributions, clients
# and record all results to one txt file
########################################
def standalone_cnn_training():
    os.makedirs(RESULT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Distribution root directories (matching data partition structure)
    distributions = {
        'POW':      lambda ds: os.path.join(POW_PATH, ds),
        'ECS_C5':   lambda ds: os.path.join(ECS_PATH, 'C5', ds),
        'PECS_C2':  lambda ds: os.path.join(PECS_PATH, 'C2', ds),
        'PECS_C5':  lambda ds: os.path.join(PECS_PATH, 'C5', ds),
        'PECS_C10': lambda ds: os.path.join(PECS_PATH, 'C10', ds)
    }
    
    # Dataset names (matching folder names used in partitioning)
    datasets_list = ['FashionMNIST', 'CIFAR10']
    
    # Used to store final statistics. Structure: results[(dataset_name, dist_name)] = (val1, val2, val3, val4)
    # where val1, val2 are the mean and std of the final 3-epoch average accuracy,
    # and val3, val4 are the mean and std of the final 3-epoch max accuracy
    results = {}
    
    for dataset_name in datasets_list:
        # Get learning rate for this dataset
        lr = learning_rates.get(dataset_name, 0.001)
        print(f"\n===== Processing dataset: {dataset_name} (lr={lr}) =====")
        
        for dist_name, base_dir_fn in distributions.items():
            base_dir = base_dir_fn(dataset_name)
            print(f"  >> Distribution: {dist_name}")
            
            # Store the training curve for all clients (num_clients, num_epochs)
            # Each row: client_id, each column: epoch
            all_clients_acc = []
            
            for client_id in range(NUM_CLIENTS):
                data, labels = load_client_data_labels(base_dir, client_id)
                if data is None or labels is None:
                    print(f"    Warning: client_{client_id} is missing data under {base_dir}")
                    # If missing, fill with zeros
                    all_clients_acc.append([0.0]*NUM_EPOCHS)
                    continue
                # Preprocess data
                data = preprocess_data(data)
                
                # Get per-epoch accuracy
                epoch_acc_list = train_and_evaluate_client(data, labels, device, lr=lr)
                all_clients_acc.append(epoch_acc_list)
            
            # Convert to numpy array, shape (NUM_CLIENTS, NUM_EPOCHS)
            all_clients_acc = np.array(all_clients_acc)
            
            # For each epoch, compute average accuracy (over 10 clients) and max accuracy (among 10 clients)
            # Shape (NUM_EPOCHS,)
            mean_acc_each_epoch = np.mean(all_clients_acc, axis=0)
            max_acc_each_epoch  = np.max(all_clients_acc,  axis=0)
            
            # Take the final 3 epochs of average accuracy => mean_acc_each_epoch[-3:]
            last3_mean_acc = mean_acc_each_epoch[-3:]
            avg3_mean_acc  = np.mean(last3_mean_acc)
            std3_mean_acc  = np.std(last3_mean_acc)
            
            # Take the final 3 epochs of max accuracy => max_acc_each_epoch[-3:]
            last3_max_acc = max_acc_each_epoch[-3:]
            avg3_max_acc  = np.mean(last3_max_acc)
            std3_max_acc  = np.std(last3_max_acc)
            
            # Store in results dictionary
            results[(dataset_name, dist_name)] = (
                avg3_mean_acc, std3_mean_acc,  # final 3-epoch average accuracy (mean, std)
                avg3_max_acc,  std3_max_acc    # final 3-epoch max accuracy (mean, std)
            )
    
    # === Write results to a single txt file ===
    final_result_file = os.path.join(RESULT_DIR, "final_results.txt")
    with open(final_result_file, "w") as f:
        f.write("Dataset, Distribution, last3_AvgAcc(mean±std), last3_MaxAcc(mean±std)\n")
        for dataset_name in datasets_list:
            for dist_name in distributions.keys():
                (avg3_mean_acc, std3_mean_acc, avg3_max_acc, std3_max_acc) = results.get((dataset_name, dist_name), (0,0,0,0))
                line = (f"{dataset_name}, {dist_name}, "
                        f"{avg3_mean_acc:.4f}±{std3_mean_acc:.4f}, "
                        f"{avg3_max_acc:.4f}±{std3_max_acc:.4f}\n")
                f.write(line)
    print(f"\n>>> All final results have been saved to: {final_result_file}")


if __name__ == '__main__':
    standalone_cnn_training()
