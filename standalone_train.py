#!/usr/bin/env python3
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

NUM_CLIENTS = 10       # 每个分布下客户端数量
NUM_EPOCHS  = 20       # 每个客户端训练的轮数
BATCH_SIZE  = 32       # 批次大小

# 定义不同数据集使用的学习率
learning_rates = {
    "FashionMNIST": 0.0003,
    "CIFAR10": 0.0003
}

########################################
# 定义简单的 CNN 模型（适用于 10 类分类）
########################################
class SimpleCNN(nn.Module):
    def __init__(self, in_channels, img_size, num_classes=10):
        """
        :param in_channels: 输入通道数（1 或 3）
        :param img_size: (高度, 宽度)
        :param num_classes: 类别数，默认为 10
        """
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2, 2)
        # 经过两次 2×2 池化后，高宽分别缩小为原来的 1/4
        feature_h = img_size[0] // 4
        feature_w = img_size[1] // 4
        self.fc    = nn.Linear(64 * feature_h * feature_w, num_classes)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def get_model(in_channels, img_size, num_classes=10):
    return SimpleCNN(in_channels, img_size, num_classes)

########################################
# 数据加载与预处理函数
########################################
def load_client_data_labels(base_dir, client_id):
    """
    加载指定客户端的数据和标签。
    假定每个客户端文件夹下有 data.npy 和 labels.npy
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
    将数据转为 CNN 可用的格式：
      - 若数据形状为 (N, H, W)，则添加通道维度，变为 (N, 1, H, W)；
      - 若数据形状为 (N, H, W, 3)，则转换为 (N, 3, H, W)。
    """
    if len(data.shape) == 3:
        # (N, H, W)
        data = data[:, None, :, :]
    elif len(data.shape) == 4 and data.shape[-1] == 3:
        # (N, H, W, 3) => (N, 3, H, W)
        data = np.transpose(data, (0, 3, 1, 2))
    return data.astype(np.float32)

########################################
# 客户端模型训练并返回每个 epoch 的测试准确率
########################################
def train_and_evaluate_client(data, labels, device,
                              num_epochs=NUM_EPOCHS,
                              batch_size=BATCH_SIZE,
                              lr=0.001):
    """
    在单个客户端数据上进行训练，返回长度为 num_epochs 的测试准确率列表。
    每个 epoch 结束后在测试集上做评估。
    """
    # 划分训练/测试集 (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42, shuffle=True
    )
    # 转为 torch 张量
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test  = torch.tensor(X_test)
    y_test  = torch.tensor(y_test, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset  = TensorDataset(X_test,  y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    
    # 获取输入图像形状
    _, C, H, W = X_train.shape
    model = get_model(C, (H, W), num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    epoch_acc_list = []  # 用于记录每个 epoch 的测试准确率
    
    for epoch in range(num_epochs):
        # ---- 训练阶段 ----
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # ---- 测试阶段 ----
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
    
    return epoch_acc_list  # 返回该客户端所有 epoch 的测试准确率

########################################
# 主训练流程：遍历数据集、分布、客户端
# 并将所有结果记录到一个 txt 文件中
########################################
def standalone_cnn_training():
    os.makedirs(RESULT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 定义各分布的根目录生成方式（与数据划分时的目录结构对应）
    distributions = {
        'POW':      lambda ds: os.path.join(POW_PATH, ds),
        'ECS_C5':   lambda ds: os.path.join(ECS_PATH, 'C5', ds),
        'PECS_C2':  lambda ds: os.path.join(PECS_PATH, 'C2', ds),
        'PECS_C5':  lambda ds: os.path.join(PECS_PATH, 'C5', ds),
        'PECS_C10': lambda ds: os.path.join(PECS_PATH, 'C10', ds)
    }
    
    # 待实验的数据集名称（对应划分时文件夹名称）
    datasets_list = ['FashionMNIST', 'CIFAR10']
    
    # 用于保存最终统计结果，结构： results[ (dataset_name, dist_name) ] = (val1, val2, val3, val4)
    # 其中 val1, val2 对应 "最后3轮平均准确率" 的均值和方差
    #      val3, val4 对应 "最后3轮最大准确率" 的均值和方差
    results = {}
    
    for dataset_name in datasets_list:
        # 根据数据集获取对应学习率
        lr = learning_rates.get(dataset_name, 0.001)
        print(f"\n===== Processing dataset: {dataset_name} (lr={lr}) =====")
        
        for dist_name, base_dir_fn in distributions.items():
            base_dir = base_dir_fn(dataset_name)
            print(f"  >> Distribution: {dist_name}")
            
            # 用于保存当前分布下所有客户端的训练曲线 (num_clients, num_epochs)
            # 每行：client_id, 每列：epoch
            all_clients_acc = []
            
            for client_id in range(NUM_CLIENTS):
                data, labels = load_client_data_labels(base_dir, client_id)
                if data is None or labels is None:
                    print(f"    Warning: client_{client_id} 数据缺失于 {base_dir}")
                    # 如果缺失，则给出一个全零的曲线
                    all_clients_acc.append([0.0]*NUM_EPOCHS)
                    continue
                # 预处理数据格式，保证适用于 CNN
                data = preprocess_data(data)
                
                # 获取该客户端全部 epoch 的准确率列表
                epoch_acc_list = train_and_evaluate_client(data, labels, device, lr=lr)
                all_clients_acc.append(epoch_acc_list)
            
            # 转为 numpy 数组，形状 (NUM_CLIENTS, NUM_EPOCHS)
            all_clients_acc = np.array(all_clients_acc)
            
            # 对每个 epoch，先求"平均准确率" (10个client平均)，再求"最大准确率" (10个client里最大)
            # 形状 (NUM_EPOCHS,)
            mean_acc_each_epoch = np.mean(all_clients_acc, axis=0)
            max_acc_each_epoch  = np.max(all_clients_acc,  axis=0)
            
            # 取最后 3 个 epoch 的平均准确率 => mean_acc_each_epoch[-3:]
            last3_mean_acc = mean_acc_each_epoch[-3:]  # 形如 [a, b, c]
            # 计算 3 个值的均值与标准差
            avg3_mean_acc  = np.mean(last3_mean_acc)
            std3_mean_acc  = np.std(last3_mean_acc)
            
            # 取最后 3 个 epoch 的最大准确率 => max_acc_each_epoch[-3:]
            last3_max_acc = max_acc_each_epoch[-3:]    # 形如 [a, b, c]
            avg3_max_acc  = np.mean(last3_max_acc)
            std3_max_acc  = np.std(last3_max_acc)
            
            # 记录到 results 字典
            results[(dataset_name, dist_name)] = (
                avg3_mean_acc, std3_mean_acc,  # 最后3轮平均准确率(均值,方差)
                avg3_max_acc,  std3_max_acc    # 最后3轮最大准确率(均值,方差)
            )
    
    # === 所有结果写入一个 txt 文件 ===
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
    print(f"\n>>> 所有分布的最终结果已保存到：{final_result_file}")

if __name__ == '__main__':
    standalone_cnn_training()
