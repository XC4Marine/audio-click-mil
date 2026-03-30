import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim.lr_scheduler import LambdaLR, StepLR

# --- 1. 固定随机种子 ---
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# --- 2. 路径配置 ---
FOLD_PATH = r"D:\Project_Github\audio_click_mil\processed_data\folds\fold_0.json"
FEATURE_BASE = r"D:\Project_Github\audio_click_mil\processed_data\features"
LABEL_CSV = r"D:\Project_Github\audio_click_mil\processed_data\balanced_bag_labels.csv"
GT_CSVS = [
    r"D:\Project_Github\audio_click_mil\data\ClickTrains.csv",
    r"D:\Project_Github\audio_click_mil\data\BurstPulseTrains.csv",
    r"D:\Project_Github\audio_click_mil\data\BuzzTrains.csv"
]
RESULT_DIR = r"D:\Project_Github\audio_click_mil\results"
MODEL_DIR = r"D:\Project_Github\audio_click_mil\models"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 3. 模型定义 (Gated Attention MIL) ---
class BagDataset(Dataset):
    def __init__(self, bag_list, feature_method):
        self.bag_paths = []
        self.labels = []
        feat_dir = os.path.join(FEATURE_BASE, feature_method)
        for bag_info in bag_list:
            fname = f"file_{bag_info['file_num']:02d}_bag_{bag_info['bag_idx']:03d}_label_{bag_info['bag_label']}_feat.npy"
            fpath = os.path.join(feat_dir, fname)
            if os.path.exists(fpath):
                self.bag_paths.append(fpath)
                self.labels.append(bag_info['bag_label'])
    
    def __len__(self): return len(self.labels)
    
    def __getitem__(self, idx):
        feat = np.load(self.bag_paths[idx])
        return torch.from_numpy(feat).float(), torch.tensor(self.labels[idx]).float()

class GatedAttentionMIL(nn.Module):
    def __init__(self, feat_dim=128):
        super(GatedAttentionMIL, self).__init__()
        # 降维投影层 (Bottleneck)
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Gated Attention 支路
        self.attention_V = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(64, 32),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(32, 1)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: (60, feat_dim)
        h = self.proj(x)  # (60, 64)

        # 计算门控权重
        A_V = self.attention_V(h)  # (60, 32)
        A_U = self.attention_U(h)  # (60, 32)
        # 元素相乘实现门控过滤
        A = self.attention_weights(A_V * A_U).squeeze(-1) # (60,)
        
        weights = torch.softmax(A, dim=0) # 归一化权重
        
        # 聚合 Bag 特征
        bag_rep = torch.sum(weights.unsqueeze(-1) * h, dim=0) # (64,)
        logit = self.classifier(bag_rep)
        
        return logit, weights

# --- 4. 辅助函数 ---
def load_bag_list(fold_data, split, label_df):
    file_list = fold_data[split + "_files"]
    if isinstance(file_list[0], str): file_list = [int(f) for f in file_list]
    bag_list = []
    for _, row in label_df.iterrows():
        if int(row['file_num']) in file_list:
            bag_list.append({'file_num': int(row['file_num']), 'bag_idx': int(row['bag_idx']), 'bag_label': int(row['bag_label'])})
    return bag_list

def compute_instance_label(file_num, bag_start_sec, instance_idx, gt_dfs):
    instance_start = bag_start_sec + instance_idx
    instance_end = instance_start + 1.0
    for df in gt_dfs:
        gt = df[df['Ori_file_num(No.)'] == file_num]
        for _, row in gt.iterrows():
            if max(instance_start, row['Train_start(s)']) < min(instance_end, row['Train_end(s)']): return 1
    return 0

# --- 5. 主程序 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_method', type=str, default='logmel')
    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=60)
    
    import sys
    if 'ipykernel' in sys.modules:
        # Jupyter 环境下，parse_known_args 返回 (Namespace, list)
        args, _ = parser.parse_known_args()
    else:
        # 命令行环境下，parse_args 只返回 Namespace，不能用 args, _ 接收
        args = parser.parse_args()
    
    # 数据加载
    with open(FOLD_PATH, 'r') as f: fold = json.load(f)
    label_df = pd.read_csv(LABEL_CSV)
    gt_dfs = [pd.read_csv(p) for p in GT_CSVS]
    
    train_bags = load_bag_list(fold, 'train', label_df)
    test_bags = load_bag_list(fold, 'test', label_df)
    train_loader = DataLoader(BagDataset(train_bags, args.feature_method), batch_size=1, shuffle=True)
    test_loader = DataLoader(BagDataset(test_bags, args.feature_method), batch_size=1, shuffle=False)
    
    # 初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GatedAttentionMIL(feat_dim=args.feat_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    
    # 策略：增加 Weight Decay 进一步防止 0.0几 Loss 时的过拟合
    base_lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=5e-4)
    
    # Warmup + Scheduler
    warmup_epochs = 10
    warmup_lambda = lambda epoch: (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    main_scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    best_loss = float('inf')
    best_model_path = os.path.join(MODEL_DIR, f"best_mil_gated_{args.feature_method}.pth")

    print(f"Using device: {device} | Gated Attention MIL | Seed: 42")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for bag, label in pbar:
            bag, label = bag.to(device), label.to(device)
            optimizer.zero_grad()
            logit, _ = model(bag.squeeze(0))
            loss = criterion(logit.view(-1), label.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{optimizer.param_groups[0]['lr']:.6f}"})

        if epoch < warmup_epochs: warmup_scheduler.step()
        else: main_scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)

    # 推理评估
    print("\nEvaluating with Gated Attention Best Model...")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()
    
    sheet0_data, sheet1_data, all_true, all_pred = [], [], [], []

    with torch.no_grad():
        for i, (bag, label) in enumerate(tqdm(test_loader)):
            bag, label = bag.to(device), label.to(device)
            logit, attn = model(bag.squeeze(0))
            prob = torch.sigmoid(logit.view(-1)).item()
            pred = 1 if prob > 0.5 else 0
            all_true.append(int(label.item()))
            all_pred.append(pred)
            
            bag_info = test_bags[i]
            bag_id = f"file_{bag_info['file_num']:02d}_bag_{bag_info['bag_idx']:03d}"
            sheet0_data.append({'bag_id': bag_id, 'true_label': int(label.item()), 'pred_label': pred, 'prob': round(prob, 4)})
            
            bag_start_sec = label_df[(label_df['file_num'] == bag_info['file_num']) & (label_df['bag_idx'] == bag_info['bag_idx'])]['bag_start_sec'].values[0]
            attn_np = attn.cpu().numpy()
            for inst_idx in range(60):
                gt_inst = compute_instance_label(bag_info['file_num'], bag_start_sec, inst_idx, gt_dfs)
                sheet1_data.append({'bag_id': bag_id, 'instance_idx': inst_idx, 'gt_label': gt_inst, 'attn_weight': round(float(attn_np[inst_idx]), 4)})

    # 保存结果
    report_path = os.path.join(RESULT_DIR, f"report_{args.feature_method}.txt")
    bag_csv = os.path.join(RESULT_DIR, f"mil_bag_results_{args.feature_method}.csv")
    inst_csv = os.path.join(RESULT_DIR, f"mil_instance_results_{args.feature_method}.csv")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(classification_report(all_true, all_pred))
        df_inst = pd.DataFrame(sheet1_data)
        high_attn = df_inst[df_inst['attn_weight'] > (1/60 * 2)]
        hit_rate = high_attn['gt_label'].mean() if not high_attn.empty else 0
        f.write(f"\nHit Rate (Attn > 2/N): {hit_rate:.4f}")

    pd.DataFrame(sheet0_data).to_csv(bag_csv, index=False, encoding='utf-8-sig')
    pd.DataFrame(sheet1_data).to_csv(inst_csv, index=False, encoding='utf-8-sig')

    print(f"Done! Report: {report_path}")