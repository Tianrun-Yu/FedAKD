#!/usr/bin/env python3
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

RESULT_DIR = ''
TRAIN_PATH = ''

STANDALONE_DIR = ''

POW_PATH   = os.path.join(TRAIN_PATH, 'POW')
ECS_PATH   = os.path.join(TRAIN_PATH, 'ECS')
PECS_PATH  = os.path.join(TRAIN_PATH, 'PECS')

NUM_CLIENTS       = 10
NUM_GLOBAL_ROUNDS = 30
LOCAL_EPOCHS      = 1
BATCH_SIZE        = 32

learning_rates = {
    "FashionMNIST": (0.001, 1, 1),  # (lr, alpha, beta)
    "CIFAR10":      (0.005, 1, 1)
}

class SimpleCNN(nn.Module):
    def __init__(self, in_channels, img_size, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2,2)
        feature_h = img_size[0] // 4
        feature_w = img_size[1] // 4
        self.fc   = nn.Linear(64 * feature_h * feature_w, num_classes)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

def get_model(in_channels, img_size, num_classes=10):
    return SimpleCNN(in_channels, img_size, num_classes)

def load_client_data_labels(base_dir, cid):
    """
    Load data and labels for client `cid`.
    Return (data, labels) if files exist, otherwise (None, None).
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
    Transpose the data to the shape [N,C,H,W].
    If data shape is [N,H,W], we reshape it to [N,1,H,W].
    If data shape is [N,H,W,3], we transpose to [N,3,H,W].
    Convert to float32.
    """
    if data.ndim == 3:
        data = data[:, None, :, :]
    elif data.ndim == 4 and data.shape[-1] == 3:
        data = np.transpose(data, (0, 3, 1, 2))
    return data.astype(np.float32)

def create_data_loaders_ack(data, labels, batch_size, test_ratio=0.2):
    """
    Split data into train/test sets, then create DataLoader objects.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_ratio, shuffle=True, random_state=42
    )
    ds_train = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    ds_test  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test,  dtype=torch.long))
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def kd_loss(student_out, teacher_out, T=1.0):
    """
    Knowledge Distillation loss function.
    """
    teacher_prob = nn.functional.softmax(teacher_out / T, dim=1)
    student_log  = nn.functional.log_softmax(student_out / T, dim=1)
    kl = nn.functional.kl_div(student_log, teacher_prob, reduction='batchmean') * (T**2)
    return kl

def get_correct_subset(model, data_x, data_y, device):
    """
    Evaluate model predictions on the full dataset, select correctly classified samples.
    Return (correct_x, correct_y).
    """
    model.eval()
    x_t = torch.tensor(data_x).to(device)
    y_t = torch.tensor(data_y, dtype=torch.long).to(device)
    with torch.no_grad():
        out = model(x_t)
        _, pred = torch.max(out, 1)
        correct_mask = (pred == y_t)
    correct_x = data_x[correct_mask.cpu().numpy()]
    correct_y = data_y[correct_mask.cpu().numpy()]
    return correct_x, correct_y

def local_to_global_distill(global_model, local_model, correct_x, correct_y, device, lr, alpha, local_epochs=1):
    """
    Distill knowledge from a local model to the global model using correct subsets.
    """
    criterion = nn.CrossEntropyLoss()
    global_model.train()
    local_model.eval()

    optimizer = optim.Adam(global_model.parameters(), lr=lr)
    ds = TensorDataset(torch.tensor(correct_x), torch.tensor(correct_y, dtype=torch.long))
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    for ep in range(local_epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out_g = global_model(bx)   # Student
            loss_ce = criterion(out_g, by)

            with torch.no_grad():
                out_l = local_model(bx) # Teacher
            loss_kd = kd_loss(out_g, out_l, T=1.0)

            loss = loss_ce + alpha * loss_kd
            loss.backward()
            optimizer.step()

def global_to_local_distill(local_model, global_model, data_x, data_y, device, lr, beta, local_epochs=1):
    """
    Distill knowledge from global model to the local model.
    Return the correct subset of local data after distillation.
    """
    criterion = nn.CrossEntropyLoss()
    local_model.train()
    global_model.eval()

    optimizer = optim.Adam(local_model.parameters(), lr=lr)
    ds = TensorDataset(torch.tensor(data_x), torch.tensor(data_y, dtype=torch.long))
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    for ep in range(local_epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out_l = local_model(bx)
            loss_ce = criterion(out_l, by)

            with torch.no_grad():
                out_g = global_model(bx)
            loss_kd = kd_loss(out_l, out_g, T=1.0)
            loss = loss_ce + beta*loss_kd
            loss.backward()
            optimizer.step()

    correct_x, correct_y = get_correct_subset(local_model, data_x, data_y, device)
    return correct_x, correct_y

def eval_global_model(model, test_loader, device):
    """
    Evaluate global model accuracy on the given test_loader.
    Return accuracy in [0,1].
    """
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            out = model(bx)
            _, pred = torch.max(out, 1)
            total   += by.size(0)
            correct += (pred == by).sum().item()
    return correct / total if total > 0 else 0

def load_standalone_accuracies(dataset_name, dist_name, num_epochs=NUM_GLOBAL_ROUNDS):
    """
    This function is reserved if standalone comparison is needed.
    Otherwise, you can ignore it or remove it if not used.
    """
    filename = f"{dataset_name}_{dist_name}.txt"
    file_path = os.path.join(STANDALONE_DIR, filename)

    if not os.path.exists(file_path):
        return np.zeros((num_epochs, NUM_CLIENTS), dtype=np.float32)

    standalone_acc = np.zeros(NUM_CLIENTS, dtype=np.float32)
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Client"):
                parts = line.split(":")
                left  = parts[0].strip()  # "Client 0"
                right = parts[1].strip()  # "0.5560"
                cidx_str = left.split()[1]  # "0"
                cidx = int(cidx_str)
                acc = float(right)
                standalone_acc[cidx] = acc

    expanded = np.tile(standalone_acc, (num_epochs, 1))
    return expanded

def fedack_training(base_dir, lr, alpha, beta, device):
    """
    The main training logic for FedACK. Returns a matrix of shape
    (NUM_GLOBAL_ROUNDS, NUM_CLIENTS), where entry (r,c) is the global model
    accuracy on client c's test set after round r.
    """
    train_loaders = []
    test_loaders  = []
    data_local    = []
    correct_subsets = []
    input_shapes  = []
    valid_clients = [False] * NUM_CLIENTS

    # 1) Load data for each client
    for cid in range(NUM_CLIENTS):
        d_, l_ = load_client_data_labels(base_dir, cid)
        if d_ is None:
            train_loaders.append(None)
            test_loaders.append(None)
            input_shapes.append(None)
            correct_subsets.append(None)
            data_local.append(None)
            continue
        d_ = preprocess_data(d_)
        tr_loader, te_loader = create_data_loaders_ack(d_, l_, BATCH_SIZE)
        train_loaders.append(tr_loader)
        test_loaders.append(te_loader)

        sample_x, _ = next(iter(tr_loader))
        c_ = sample_x.shape[1]
        h_ = sample_x.shape[2]
        w_ = sample_x.shape[3]
        input_shapes.append((c_, h_, w_))
        data_local.append((d_, l_))
        valid_clients[cid] = True

        # At initialization, assume all data is in the "correct subset"
        correct_subsets.append((d_, l_))

    if not any(valid_clients):
        # If no valid clients exist, return a zero matrix
        return np.zeros((NUM_GLOBAL_ROUNDS, NUM_CLIENTS))

    # 2) Create global model
    first_cid = valid_clients.index(True)
    c, h, w = input_shapes[first_cid]
    global_model = get_model(c, (h, w)).to(device)

    # 3) Initialize local models
    local_models = []
    for cid in range(NUM_CLIENTS):
        if valid_clients[cid]:
            loc = get_model(c, (h, w)).to(device)
            loc.load_state_dict(global_model.state_dict())
            local_models.append(loc)
        else:
            local_models.append(None)

    # This matrix records global model accuracy on each client's test set per round
    acc_matrix = np.zeros((NUM_GLOBAL_ROUNDS, NUM_CLIENTS))

    # 4) Main iterative training
    for round_idx in range(NUM_GLOBAL_ROUNDS):
        # (a) Local -> Global Distillation
        local_global_states = []
        data_sizes = []
        for cid in range(NUM_CLIENTS):
            if not valid_clients[cid]:
                continue
            # student_g is a copy of global
            student_g = get_model(c, (h, w)).to(device)
            student_g.load_state_dict(global_model.state_dict())

            cx, cy = correct_subsets[cid]
            local_to_global_distill(student_g, local_models[cid], cx, cy, device, lr, alpha, LOCAL_EPOCHS)

            local_global_states.append(student_g.state_dict())
            data_sizes.append(len(data_local[cid][0]))  # total dataset size for weighting

        # (b) Aggregate and update global model
        if len(local_global_states) > 0:
            sum_size = sum(data_sizes)
            new_state = {}
            for key in local_global_states[0].keys():
                w_sum = None
                for i, st in enumerate(local_global_states):
                    weight = data_sizes[i] / sum_size
                    if w_sum is None:
                        w_sum = st[key].float() * weight
                    else:
                        w_sum += st[key].float() * weight
                new_state[key] = w_sum
            global_model.load_state_dict(new_state)

        # (c) Global -> Local Distillation
        for cid in range(NUM_CLIENTS):
            if not valid_clients[cid]:
                continue
            full_x, full_y = data_local[cid]
            cx, cy = global_to_local_distill(local_models[cid], global_model, full_x, full_y, device, lr, beta, LOCAL_EPOCHS)
            # Update the correct subset
            correct_subsets[cid] = (cx, cy)

        # (d) Evaluate current global model on each client's test set
        for cid in range(NUM_CLIENTS):
            if not valid_clients[cid]:
                acc_matrix[round_idx, cid] = 0.0
            else:
                te_loader = test_loaders[cid]
                acc = eval_global_model(global_model, te_loader, device)
                acc_matrix[round_idx, cid] = acc

    return acc_matrix

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    distributions = {
        'POW':      lambda ds: os.path.join(POW_PATH, ds),
        'BCS_C5':   lambda ds: os.path.join(ECS_PATH, 'C5', ds),
        'ICS_C2':   lambda ds: os.path.join(PECS_PATH, 'C2', ds),
        'ICS_C5':   lambda ds: os.path.join(PECS_PATH, 'C5', ds),
        'ICS_C10':  lambda ds: os.path.join(PECS_PATH, 'C10', ds)
    }
    datasets_list = ['FashionMNIST','CIFAR10']

    # Prepare a single file to store all rounds' results (fairness, max acc, avg acc)
    # e.g., "fedack_all_rounds_results.txt"
    all_rounds_file = os.path.join(RESULT_DIR, "fedack_all_rounds_results.txt")
    with open(all_rounds_file, "w") as f_all:
        # Header: Dataset, Distribution, Round, Fairness, MaxAcc, AvgAcc
        f_all.write("Dataset,Distribution,Round,Fairness,MaxAcc,AvgAcc\n")

        for dataset_name in datasets_list:
            (lr, alpha, beta) = learning_rates.get(dataset_name, (0.001, 0.5, 0.5))
            print(f"\n===== FedACK on {dataset_name} (lr={lr}, alpha={alpha}, beta={beta}) =====")
            
            for dist_name, base_dir_fn in distributions.items():
                base_dir = base_dir_fn(dataset_name)
                print(f"--- Distribution: {dist_name} ---")

                # FedACK training, returns accuracy matrix per round
                acc_matrix = fedack_training(base_dir, lr, alpha, beta, device)
                # acc_matrix.shape = (NUM_GLOBAL_ROUNDS, NUM_CLIENTS)

                # Calculate fairness, max, and average accuracy for each round
                for rd in range(NUM_GLOBAL_ROUNDS):
                    round_accuracies = acc_matrix[rd, :]
                    avg_acc = np.mean(round_accuracies)
                    max_acc = np.max(round_accuracies)
                    # Use standard deviation as the fairness metric (you can change it)
                    fairness = np.std(round_accuracies)

                    # Write to file: Dataset,Distribution,R, Fairness, MaxAcc, AvgAcc
                    line = f"{dataset_name},{dist_name},{rd+1},{fairness:.4f},{max_acc:.4f},{avg_acc:.4f}\n"
                    f_all.write(line)

    print(f"\n[Info] All rounds results have been saved to: {all_rounds_file}")


if __name__ == '__main__':
    main()

