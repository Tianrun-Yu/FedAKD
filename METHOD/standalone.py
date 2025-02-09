#!/usr/bin/env python3
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 路径设置
TRAIN_PATH = '/home/tvy5242/EHR_fl/A_Experiment/DATA/train'
RESULT_DIR = '/home/tvy5242/EHR_fl/A_Experiment/RESULT/standalone'

# 各种数据分布存放的根目录（需与数据划分时的目录结构一致）
POW_PATH   = os.path.join(TRAIN_PATH, 'POW')
ECS_PATH   = os.path.join(TRAIN_PATH, 'ECS')
PECS_PATH  = os.path.join(TRAIN_PATH, 'PECS')

NUM_CLIENTS = 10  # 每个分布下客户端数量

############################################################################
# 为不同数据集配置不同 (学习率, 训练轮数)
############################################################################
dataset_configs = {
    "FashionMNIST": {
        "lr": 0.15,
        "num_epochs": 30
    },
    "CIFAR10": {
        "lr": 0.015,
        "num_epochs": 100
    }
}

############################################################################
# 两种专用模型
############################################################################
class TwoLayerCNN_MNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(TwoLayerCNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc    = nn.Linear(32 * 28 * 28, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ThreeLayerCNN_CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(ThreeLayerCNN_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64,128,3, padding=1)
        self.fc    = nn.Linear(128 * 32 * 32, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

############################################################################
# 根据 dataset_name 返回合适的模型
############################################################################
def get_model_for_dataset(dataset_name):
    if dataset_name == "FashionMNIST":
        return TwoLayerCNN_MNIST(num_classes=10)
    elif dataset_name == "CIFAR10":
        return ThreeLayerCNN_CIFAR(num_classes=10)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name} (expected FashionMNIST or CIFAR10)")

############################################################################
# 数据加载与预处理函数
############################################################################
def load_client_data_labels(base_dir, client_id):
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
    if data.ndim==3:
        data = data[:, None, :, :]
    elif data.ndim==4 and data.shape[-1] == 3:
        data = np.transpose(data, (0, 3, 1, 2))
    return data.astype(np.float32)

############################################################################
# 客户端模型训练并返回每个 epoch 的测试准确率
############################################################################
def train_and_evaluate_client(dataset_name, data, labels, device):
    """
    根据 dataset_name => 取其对应的 lr, num_epochs
    划分 (80/20) => (train/test)
    训练 => 返回 (num_epochs,) 的准确率曲线
    """
    config = dataset_configs[dataset_name]
    lr = config["lr"]
    num_epochs = config["num_epochs"]

    # 划分 train/test
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42, shuffle=True
    )
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test  = torch.tensor(X_test)
    y_test  = torch.tensor(y_test, dtype=torch.long)

    train_dataset= TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test,  y_test)
    train_loader= DataLoader(train_dataset,batch_size=32,shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32,shuffle=False)

    model= get_model_for_dataset(dataset_name).to(device)
    criterion= nn.CrossEntropyLoss()
    optimizer= optim.Adam(model.parameters(), lr=lr)

    epoch_acc_list=[]
    for ep in range(num_epochs):
        model.train()
        for bx,by in train_loader:
            bx,by= bx.to(device), by.to(device)
            optimizer.zero_grad()
            out= model(bx)
            loss= criterion(out, by)
            loss.backward()
            optimizer.step()

        # test
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
        acc= correct/total if total>0 else 0.0
        epoch_acc_list.append(acc)

    return epoch_acc_list

############################################################################
# 主流程
############################################################################
def standalone_cnn_training():
    os.makedirs(RESULT_DIR, exist_ok=True)
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义分布根目录
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
        print(f"\n===== Processing dataset: {dataset_name} =====")

        # 读取 dataset_configs => 获取 num_epochs
        num_epochs= dataset_configs[dataset_name]["num_epochs"]

        for dist_name, base_dir_fn in distributions.items():
            base_dir= base_dir_fn(dataset_name)
            print(f"  >> Distribution: {dist_name}")

            all_clients_acc= []  # shape = (num_clients, num_epochs)
            for client_id in range(NUM_CLIENTS):
                data, labels= load_client_data_labels(base_dir, client_id)
                if data is None or labels is None:
                    print(f"    Warning: client_{client_id} 数据缺失于 {base_dir}")
                    # 如果缺失,返回全 0 curve
                    all_clients_acc.append([0.0]*num_epochs)
                    continue
                data= preprocess_data(data)
                # 训练 & 记录
                epoch_acc_list= train_and_evaluate_client(dataset_name, data, labels, device)
                all_clients_acc.append(epoch_acc_list)

            all_clients_acc= np.array(all_clients_acc)  # shape=(num_clients, num_epochs)

            # === 同时保存每个client的最终(最后一个epoch)准确率到一个文件 ===
            # 例如 => "FashionMNIST_ECS_C5.txt"
            final_accs= all_clients_acc[:,-1]  # shape=(num_clients,) =>最后一个epoch的值
            dist_file_path= os.path.join(RESULT_DIR, f"{dataset_name}_{dist_name}.txt")
            with open(dist_file_path, "w") as df:
                for cid,acc_val in enumerate(final_accs):
                    df.write(f"Client {cid}: {acc_val:.4f}\n")

            # 下面计算最后3轮平均准确率 & 最大准确率
            mean_acc_each_epoch= np.mean(all_clients_acc, axis=0)
            max_acc_each_epoch = np.max(all_clients_acc, axis=0)

            last3_mean_acc= mean_acc_each_epoch[-3:]
            avg3_mean_acc= np.mean(last3_mean_acc)
            std3_mean_acc= np.std(last3_mean_acc)

            last3_max_acc= max_acc_each_epoch[-3:]
            avg3_max_acc= np.mean(last3_max_acc)
            std3_max_acc= np.std(last3_max_acc)

            results[(dataset_name, dist_name)] = (
                avg3_mean_acc, std3_mean_acc,
                avg3_max_acc,  std3_max_acc
            )

    # 输出到 final_results.txt
    final_result_file= os.path.join(RESULT_DIR, "final_results.txt")
    with open(final_result_file, "w") as f:
        f.write("Dataset, Distribution, last3_AvgAcc(mean±std), last3_MaxAcc(mean±std)\n")
        for dataset_name in datasets_list:
            # 先 distinct
            for dist_name in distributions.keys():
                val= results.get((dataset_name, dist_name), (0,0,0,0))
                (m1,s1,m2,s2)= val
                line= (f"{dataset_name}, {dist_name}, "
                       f"{m1:.4f}±{s1:.4f}, "
                       f"{m2:.4f}±{s2:.4f}\n")
                f.write(line)
    print(f"\n>>> 所有分布的最终结果已保存到：{final_result_file}")
    print(">>> 同时生成了各分布的客户端最终准确率文件 (dataset_dist.txt).")

if __name__=='__main__':
    standalone_cnn_training()
