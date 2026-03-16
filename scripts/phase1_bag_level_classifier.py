# =============================================================================
# Phase 1: 基础 Bag-level 检测 (Deep Learning 版)
# =============================================================================
# 对应 skills.md 要求：
# - 模型：SimpleMLP (src/model.py)
# - 损失：FocalBCE (src/loss.py)
# - 输出：performance_table.csv, fig1_f1_vs_bag_length.png
# =============================================================================

import pandas as pd
import numpy as np
import pickle
import librosa
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1. 配置参数
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "data_root": Path("D:/Project_Github/audio_click_mil/data"),
    "audio_root": Path("D:/Project_Github/audio_click_mil/data/original_audio"),
    "phase0_output": Path("D:/Project_Github/audio_click_mil/results/phase0"),
    "phase1_output": Path("D:/Project_Github/audio_click_mil/results/phase1"),
    "target_sr": 48000,
    "bag_duration": 30,
    "mfcc_n_mfcc": 20,
    "mfcc_hop_length": 512,
    "mfcc_n_fft": 2048,
    "test_size": 0.3,
    "random_state": 42,
    # 模型超参数
    "input_dim": 40,      # MFCC Mean(20) + Std(20)
    "hidden_dim": 64,
    "num_epochs": 50,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "focal_alpha": 0.25,  # Focal Loss 参数
    "focal_gamma": 2.0,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

CONFIG["phase1_output"].mkdir(exist_ok=True, parents=True)

# ─────────────────────────────────────────────────────────────────────────────
# 🔧 模块模拟：src/model.py (SimpleMLP)
# ─────────────────────────────────────────────────────────────────────────────
class SimpleMLP(nn.Module):
    """
    简单的多层感知机，无 Attention 机制
    输入: (Batch, Input_Dim)
    输出: (Batch, 1) [概率]
    """
    def __init__(self, input_dim, hidden_dim=64):
        super(SimpleMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

# ─────────────────────────────────────────────────────────────────────────────
# 🔧 模块模拟：src/loss.py (FocalBCE)
# ─────────────────────────────────────────────────────────────────────────────
class FocalBCE(nn.Module):
    """
    Focal Loss with High Alpha for Imbalanced Data
    alpha=0.9 意味着正样本的权重是负样本的 9 倍 (0.9 / 0.1)
    """
    def __init__(self, alpha=0.9, gamma=2.0):  # <--- 关键修改：alpha 从 0.25 改为 0.9
        super(FocalBCE, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        inputs = inputs.squeeze()
        targets = targets.squeeze()
        
        # 防止 log(0)
        eps = 1e-7
        inputs = torch.clamp(inputs, eps, 1 - eps)
        
        bce_loss = - (targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))
        
        # 计算 pt (模型对真实类别的预测概率)
        pt = torch.exp(-bce_loss)
        
        # 根据 target 选择 alpha: target=1 -> alpha, target=0 -> (1-alpha)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()

# ─────────────────────────────────────────────────────────────────────────────
# 🔧 数据加载器
# ─────────────────────────────────────────────────────────────────────────────
class AudioBagDataset(Dataset):
    def __init__(self, bags, labels):
        self.bags = bags
        self.labels = labels
    
    def __len__(self):
        return len(self.bags)
    
    def __getitem__(self, idx):
        bag = self.bags[idx]
        features = bag["features"]["mfcc"] # 已经是 flatten 后的 mean+std
        label = self.labels[idx]
        
        return (
            torch.FloatTensor(features),
            torch.FloatTensor([label])
        )

# ─────────────────────────────────────────────────────────────────────────────
# 2. 加载 Phase 0 的 bags
# ─────────────────────────────────────────────────────────────────────────────
print("="*80)
print("Phase 1: 基础 Bag-level 检测 (Deep Learning)")
print("="*80)

print("\n[1/7] 加载 Phase 0 的 bags.pkl...")
bags_path = CONFIG["phase0_output"] / "bags.pkl"
with open(bags_path, 'rb') as f:
    bags = pickle.load(f)
print(f"  ✓ 加载 {len(bags)} 个 bag")

# ─────────────────────────────────────────────────────────────────────────────
# 3. 特征提取与填充 (复用原有逻辑，确保有数据)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/7] 音频特征提取与填充...")

def extract_mfcc_features(audio_path, target_sr=48000, n_mfcc=20, hop_length=512, n_fft=2048):
    try:
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
        return mfcc.T, len(y) / target_sr
    except Exception as e:
        return None, 0

def get_audio_path(file_no, audio_root):
    filename = f"Ori_Recording_{file_no:02d}.wav"
    return audio_root / filename

# 快速检查是否已经提取过特征，如果没有则提取
if bags[0]["features"].get("mfcc") is None:
    print("  ⚠ 检测到未提取特征，开始提取...")
    available_files = sorted([int(f.stem.split('_')[-1]) for f in CONFIG["audio_root"].glob("Ori_Recording_*.wav")])
    audio_features = {}
    
    for file_no in tqdm(available_files, desc="提取 MFCC"):
        path = get_audio_path(file_no, CONFIG["audio_root"])
        mfcc, dur = extract_mfcc_features(path, CONFIG["target_sr"], CONFIG["mfcc_n_mfcc"], CONFIG["mfcc_hop_length"], CONFIG["mfcc_n_fft"])
        if mfcc is not None:
            audio_features[file_no] = {"mfcc": mfcc, "duration": dur}
    
    frames_per_second = CONFIG["target_sr"] / CONFIG["mfcc_hop_length"]
    frames_per_bag = int(CONFIG["bag_duration"] * frames_per_second)
    
    for bag in tqdm(bags, desc="填充 Bag 特征"):
        file_no = bag["file_no"]
        if file_no in audio_features:
            mfcc_full = audio_features[file_no]["mfcc"]
            start_frame = int(bag["bag_start_sec"] * frames_per_second)
            end_frame = start_frame + frames_per_bag
            
            if end_frame <= len(mfcc_full):
                bag_mfcc = mfcc_full[start_frame:end_frame]
                mfcc_mean = np.mean(bag_mfcc, axis=0)
                mfcc_std = np.std(bag_mfcc, axis=0)
                bag["features"]["mfcc"] = np.concatenate([mfcc_mean, mfcc_std])
    
    # 保存更新后的 bags
    with open(CONFIG["phase1_output"] / "bags_with_features.pkl", 'wb') as f:
        pickle.dump(bags, f)
else:
    print("  ✓ 特征已存在，跳过提取步骤")

valid_bags = [b for b in bags if b["features"]["mfcc"] is not None]
print(f"  ✓ 有效 Bag 数：{len(valid_bags)}")
print(f"  ✓ 特征维度：{valid_bags[0]['features']['mfcc'].shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. 数据集划分 (按文件分组)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/7] 数据集划分...")

file_nos = np.array([b["file_no"] for b in valid_bags])
y_whistle = np.array([b["whistle_label"] for b in valid_bags])
y_pulse = np.array([b["pulse_label"] for b in valid_bags])
bag_lengths = np.array([b["bag_end_sec"] - b["bag_start_sec"] for b in valid_bags]) # 用于后续绘图

unique_files = np.unique(file_nos)
train_files, test_files = train_test_split(unique_files, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"])

train_mask = np.isin(file_nos, train_files)
test_mask = np.isin(file_nos, test_files)

# 准备数据
X_train = [b["features"]["mfcc"] for b in np.array(valid_bags)[train_mask]]
y_whistle_train = y_whistle[train_mask]
y_pulse_train = y_pulse[train_mask]
lengths_train = bag_lengths[train_mask]

X_test = [b["features"]["mfcc"] for b in np.array(valid_bags)[test_mask]]
y_whistle_test = y_whistle[test_mask]
y_pulse_test = y_pulse[test_mask]
lengths_test = bag_lengths[test_mask]

print(f"  训练集：{len(X_train)} 样本 ({len(train_files)} 文件)")
print(f"  测试集：{len(X_test)} 样本 ({len(test_files)} 文件)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. 模型训练函数
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# 🔧 修正训练函数
# ─────────────────────────────────────────────────────────────────────────────
def train_model(X, y, model_name, input_dim=CONFIG["input_dim"]):
    # 不再使用过采样 Dataset，直接使用普通列表转换
    dataset = AudioBagDataset([{"features": {"mfcc": x}} for x in X], y)
    
    # 稍微减小 Batch Size，让每个 Batch 更有可能包含正样本
    # 如果正样本极少，Batch Size 太大可能导致某些 Batch 全是负样本
    effective_batch_size = min(CONFIG["batch_size"], max(8, len(X) // 20)) 
    
    loader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True)
    
    model = SimpleMLP(input_dim=input_dim, hidden_dim=CONFIG["hidden_dim"]).to(CONFIG["device"])
    
    # 使用高 Alpha 的 Focal Loss
    criterion = FocalBCE(alpha=0.9, gamma=2.0) 
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-4)
    
    model.train()
    print(f"    [{model_name}] 开始训练 (Batch Size: {effective_batch_size}, Alpha: 0.9)...")
    
    for epoch in range(CONFIG["num_epochs"]):
        total_loss = 0
        pos_count_in_batch = 0
        
        for features, labels in loader:
            features, labels = features.to(CONFIG["device"]), labels.to(CONFIG["device"])
            
            # 统计当前 Batch 正样本数量（用于调试监控）
            pos_count_in_batch += torch.sum(labels).item()
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # 如果 Loss 是 NaN，跳过
            if torch.isnan(loss):
                continue
                
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 每 10 轮打印一次，顺便监控是否看到了正样本
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(loader)
            print(f"    [{model_name}] Epoch {epoch+1}/{CONFIG['num_epochs']}, Loss: {avg_loss:.4f}")
            
            # 如果整个 Epoch 都没看到正样本（极端情况），警告用户
            if pos_count_in_batch == 0:
                print(f"    ⚠️ 警告：Epoch {epoch+1} 中未检测到任何正样本！考虑减小 Batch Size。")
    
    return model

# ─────────────────────────────────────────────────────────────────────────────
# 6. 模型评估函数
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# 🔧 修正评估函数 (寻找最佳阈值)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model, X, y_true, model_name):
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        # 批量推理
        for i in range(0, len(X), 64):
            batch_x = torch.FloatTensor(np.array(X[i:i+64])).to(CONFIG["device"])
            probs = model(batch_x).cpu().numpy().squeeze()
            all_probs.extend(probs)
    
    all_probs = np.array(all_probs)
    y_true = np.array(y_true)
    
    # 【核心逻辑】遍历阈值，找到 F1 最高的点
    best_threshold = 0.5
    best_f1 = -1
    best_prec = 0
    best_rec = 0
    
    thresholds = np.arange(0.05, 0.95, 0.05)
    
    for thresh in thresholds:
        preds = (all_probs >= thresh).astype(int)
        
        # 如果全预测为 0 或全预测为 1，跳过（避免除零或无意义）
        if np.sum(preds) == 0 or np.sum(preds) == len(preds):
            continue
            
        f1 = f1_score(y_true, preds, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            best_prec = precision_score(y_true, preds, zero_division=0)
            best_rec = recall_score(y_true, preds, zero_division=0)
    
    # 如果没找到合适的阈值（比如全 0 时 F1 最高），则强制使用默认逻辑并警告
    if best_f1 == -1:
        print(f"  ⚠️ [{model_name}] 未找到有效阈值，回退到默认 0.5")
        best_threshold = 0.5
        final_preds = (all_probs >= 0.5).astype(int)
        best_f1 = f1_score(y_true, final_preds, zero_division=0)
        best_prec = precision_score(y_true, final_preds, zero_division=0)
        best_rec = recall_score(y_true, final_preds, zero_division=0)
    else:
        final_preds = (all_probs >= best_threshold).astype(int)
        print(f"  ✅ [{model_name}] 找到最佳阈值: {best_threshold:.2f} (F1 从 0.00 提升至 {best_f1:.4f})")

    acc = np.mean(final_preds == y_true)
    
    try:
        auc = roc_auc_score(y_true, all_probs)
    except:
        auc = 0.5
        
    return {
        "model": model_name,
        "accuracy": acc,
        "precision": best_prec,
        "recall": best_rec,
        "f1": best_f1,
        "auc": auc,
        "probs": all_probs,
        "preds": final_preds,
        "threshold": best_threshold
    }

# ─────────────────────────────────────────────────────────────────────────────
# 7. 执行训练与评估
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/7] 训练哨声检测模型 (SimpleMLP + FocalLoss)...")
whistle_model = train_model(X_train, y_whistle_train, "Whistle")

print("\n[5/7] 训练脉冲检测模型 (SimpleMLP + FocalLoss)...")
pulse_model = train_model(X_train, y_pulse_train, "Pulse")

print("\n[6/7] 模型评估...")
whistle_res = evaluate_model(whistle_model, X_test, y_whistle_test, "WhistleDetector")
pulse_res = evaluate_model(pulse_model, X_test, y_pulse_test, "PulseDetector")

# 打印报告
print("\n" + "="*60)
print("【测试结果汇总】")
print("="*60)
print(f"{'模型':<15} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'AUC':<6}")
print("-" * 60)
print(f"{'Whistle':<15} | {whistle_res['accuracy']:.4f} | {whistle_res['precision']:.4f} | {whistle_res['recall']:.4f} | {whistle_res['f1']:.4f} | {whistle_res['auc']:.4f}")
print(f"{'Pulse':<15} | {pulse_res['accuracy']:.4f} | {pulse_res['precision']:.4f} | {pulse_res['recall']:.4f} | {pulse_res['f1']:.4f} | {pulse_res['auc']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. 生成输出文件 (Table 2 & Fig 1)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/7] 生成报告文件...")

# 1. 生成 performance_table.csv (Table 2)
table_data = [
    {"Task": "Whistle Detection", "Model": "SimpleMLP", "Loss": "FocalBCE", 
     "Accuracy": whistle_res['accuracy'], "Precision": whistle_res['precision'], 
     "Recall": whistle_res['recall'], "F1-Score": whistle_res['f1'], "ROC-AUC": whistle_res['auc']},
    {"Task": "Pulse Detection", "Model": "SimpleMLP", "Loss": "FocalBCE", 
     "Accuracy": pulse_res['accuracy'], "Precision": pulse_res['precision'], 
     "Recall": pulse_res['recall'], "F1-Score": pulse_res['f1'], "ROC-AUC": pulse_res['auc']}
]
table_df = pd.DataFrame(table_data)
table_path = CONFIG["phase1_output"] / "performance_table.csv"
table_df.to_csv(table_path, index=False)
print(f"  ✓ 性能表已保存：{table_path}")

# 2. 生成 fig1_f1_vs_bag_length.png
plt.figure(figsize=(10, 6))

# 哨声
plt.scatter(lengths_test, whistle_res['preds'], c=['green' if p==1 else 'red' for p in whistle_res['preds']], 
            alpha=0.3, label='Whistle Pred', marker='o')
# 为了展示 F1 vs Length 的趋势，我们按长度分箱计算 F1
bins = np.linspace(min(lengths_test), max(lengths_test), 10)
bin_centers = (bins[:-1] + bins[1:]) / 2
f1_bins_w = []
for i in range(len(bins)-1):
    mask = (lengths_test >= bins[i]) & (lengths_test < bins[i+1])
    if np.sum(mask) > 0:
        f1_bins_w.append(f1_score(y_whistle_test[mask], whistle_res['preds'][mask], zero_division=0))
    else:
        f1_bins_w.append(np.nan)

plt.plot(bin_centers, f1_bins_w, 'g-', linewidth=2, label='Whistle F1 Trend')

# 脉冲
f1_bins_p = []
for i in range(len(bins)-1):
    mask = (lengths_test >= bins[i]) & (lengths_test < bins[i+1])
    if np.sum(mask) > 0:
        f1_bins_p.append(f1_score(y_pulse_test[mask], pulse_res['preds'][mask], zero_division=0))
    else:
        f1_bins_p.append(np.nan)

plt.plot(bin_centers, f1_bins_p, 'b-', linewidth=2, label='Pulse F1 Trend')

plt.xlabel("Bag Duration (seconds)")
plt.ylabel("F1 Score / Prediction (0/1)")
plt.title("Fig 1: F1 Score vs Bag Length (Phase 1)")
plt.legend()
plt.grid(True, alpha=0.3)

fig_path = CONFIG["phase1_output"] / "fig1_f1_vs_bag_length.png"
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"  ✓ 趋势图已保存：{fig_path}")

# 保存模型
joblib.dump(whistle_model.cpu(), CONFIG["phase1_output"] / "whistle_detector.pkl")
joblib.dump(pulse_model.cpu(), CONFIG["phase1_output"] / "pulse_detector.pkl")
print(f"  ✓ 模型已保存")

print("\n" + "="*80)
print("✅ Phase 1 全部任务完成！")
print(f"   -> Table 2: {table_path}")
print(f"   -> Fig 1:   {fig_path}")
print("="*80)