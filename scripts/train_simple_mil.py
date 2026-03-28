import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
from sklearn.metrics import classification_report, confusion_matrix

# --- 路径配置保持不变 ---
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

# --- 模型与 Dataset 定义保持不变 ---
class BagDataset(Dataset):
    def __init__(self, bag_list, feature_method):
        self.bag_paths = [] # 改为存储路径
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
        # 每次读取时转换，确保它是全新的叶子节点，不带任何梯度历史
        feat = np.load(self.bag_paths[idx])
        return torch.from_numpy(feat).float(), torch.tensor(self.labels[idx]).float()
    
class SimpleMIL(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.BatchNorm1d(128), # fix:加入Batchnorm稳定特征分布
            nn.ReLU()
        )
        self.attn = nn.Linear(128, 1)
        self.classifier = nn.Linear(128, 1)
    
    def forward(self, bag): # bag: (60, 128)
        h = self.proj(bag)  # (60, 128)
        logits = self.attn(h).squeeze(-1) # (60,)
        attn_weights = torch.softmax(logits, dim=0)
        
        # 这里的注意力加权融合
        bag_rep = torch.sum(attn_weights.unsqueeze(-1) * h, dim=0) # (128,)
        logit = self.classifier(bag_rep)
        return logit, attn_weights

# --- 辅助函数保持不变 ---
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_method', type=str, default='mfcc', choices=['teager', 'wavelet', 'mfcc', 'logmel','cqt','gammatone'])
    # 适配 Jupyter
    import sys
    if 'ipykernel' in sys.modules:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()
    
    # 1. 数据准备
    with open(FOLD_PATH, 'r') as f: fold = json.load(f)
    label_df = pd.read_csv(LABEL_CSV)
    gt_dfs = [pd.read_csv(p) for p in GT_CSVS]
    
    train_bags = load_bag_list(fold, 'train', label_df)
    test_bags = load_bag_list(fold, 'test', label_df)
    train_ds = BagDataset(train_bags, args.feature_method)
    test_ds = BagDataset(test_bags, args.feature_method)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    # 2. 初始化模型与训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleMIL().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"Using device: {device}")
    print(f"开始训练 {args.feature_method} 特征模型...")
    for epoch in tqdm(range(50), desc="Epoch进度"):
        model.train()
        for bag, label in train_loader:
            bag, label = bag.to(device).detach(), label.to(device) # 增加 .detach()
            optimizer.zero_grad()
            
            # 这里的 bag.squeeze(0) 是为了去掉 DataLoader 自动加的 batch 维度
            logit, _ = model(bag.squeeze(0))
            loss = criterion(logit.view(-1), label.view(-1))
            
            loss.backward()
            optimizer.step()

            print(f"Loss: {loss.item():.4f}") # 看看 Loss 是否在变化如果 Loss 永远不变：说明梯度断了。如果 Loss 在变但结果不变：说明学习率太低或特征完全没有区分度。
            break

    # 3. 保存模型
    model_path = os.path.join(MODEL_DIR, f"simple_mil_{args.feature_method}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"模型权重已保存至: {model_path}")

    # 4. 推理与评估
    model.eval()
    sheet0_data, sheet1_data = [], []
    all_true_labels, all_pred_labels = [], []

    with torch.no_grad():
        for i, (bag, label) in enumerate(tqdm(test_loader, desc="测试集推理进度")):
            bag = bag.to(device)
            logit, attn = model(bag.squeeze(0))
            prob = torch.sigmoid(logit.view(-1)).item()
            pred_label = 1 if prob > 0.5 else 0
            true_label = int(label.item())
            
            all_true_labels.append(true_label)
            all_pred_labels.append(pred_label)
            
            bag_info = test_bags[i]
            sheet0_data.append({
                'bag_id': f"file_{bag_info['file_num']}_bag_{bag_info['bag_idx']}",
                'true_label': true_label, 'pred_label': pred_label, 'prob': round(prob, 4)
            })
            
            # Instance-level 
            bag_start_sec = label_df[(label_df['file_num'] == bag_info['file_num']) & 
                                     (label_df['bag_idx'] == bag_info['bag_idx'])]['bag_start_sec'].values[0]
            for inst_idx in range(60):
                orig_label = compute_instance_label(bag_info['file_num'], bag_start_sec, inst_idx, gt_dfs)
                attn_val = attn[inst_idx].item()
                # 设定注意力阈值，通常均值为 1/60=0.016，这里可设稍高
                sheet1_data.append({
                    'bag_id': f"file_{bag_info['file_num']}_bag_{bag_info['bag_idx']}",
                    'instance_idx': inst_idx,
                    'gt_label': orig_label,
                    'attn_weight': round(attn_val, 4)
                })

    # 5. 生成并保存文本报告
    report_path = os.path.join(RESULT_DIR, f"report_{args.feature_method}.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"=== MIL 检测报告 ({args.feature_method}) ===\n\n")
        f.write("1. Bag-level Classification Report:\n")
        f.write(classification_report(all_true_labels, all_pred_labels))
        f.write("\n2. Confusion Matrix:\n")
        f.write(str(confusion_matrix(all_true_labels, all_pred_labels)))
        f.write("\n\n3. Instance-level Localization Summary:\n")
        
        # 统计高注意力机制下，instance的召回情况
        df_inst = pd.DataFrame(sheet1_data)
        high_attn = df_inst[df_inst['attn_weight'] > (1/60 * 2)] # 2倍均值以上的定义为活跃
        hit_rate = high_attn['gt_label'].mean() if not high_attn.empty else 0
        f.write(f"高注意力实例(Attn > 2/N)中的真实目标占比: {hit_rate:.4f}\n")
        f.write(f"注：该指标反映了模型注意力机制定位目标的准确度。\n")

    # 6. 保存为 CSV 文件
    # 保存 Bag 级别结果 (原 sheet0)
    bag_result_path = os.path.join(RESULT_DIR, f"mil_bag_results_{args.feature_method}.csv")
    pd.DataFrame(sheet0_data).to_excel = False # 确保不调用excel引擎
    pd.DataFrame(sheet0_data).to_csv(bag_result_path, index=False, encoding='utf-8-sig')
    
    # 保存 Instance 级别结果 (原 sheet1)
    inst_result_path = os.path.join(RESULT_DIR, f"mil_instance_results_{args.feature_method}.csv")
    pd.DataFrame(sheet1_data).to_csv(inst_result_path, index=False, encoding='utf-8-sig')

    print(f"任务完成！")
    print(f"- 报告: {report_path}")
    print(f"- Bag级结果: {bag_result_path}")
    print(f"- Instance级结果: {inst_result_path}")
    print(f"- 模型权重: {model_path}")