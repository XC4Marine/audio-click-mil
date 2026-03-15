# =============================================================================
# Phase 2: 弱监督MIL模型训练
# =============================================================================
# 功能：
#   - 构建MIL数据集（PyTorch DataLoader）
#   - 实现Attention-MIL模型
#   - 弱监督训练（bag-level标签）
#   - 模型评估与保存
# =============================================================================

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# 1. 配置参数
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "data_root": Path("D:/Project_Github/audio_click_mil/data"),
    "phase1_output": Path("D:/Project_Github/audio_click_mil/results/phase1"),
    "output_dir": Path("D:/Project_Github/audio_click_mil/results/phase2"),
    "bags_path": Path("D:/Project_Github/audio_click_mil/results/phase1/bags_with_features.pkl"),
    
    # 模型参数
    "input_dim": 40,          # MFCC维度
    "hidden_dim": 128,        # 隐藏层维度
    "attention_dim": 64,      # 注意力维度
    "num_classes": 2,         # 二分类（哨声/非哨声）或改为1（sigmoid）
    
    # 训练参数
    "batch_size": 16,         # batch大小
    "num_epochs": 50,         # 训练轮数
    "learning_rate": 0.001,   # 学习率
    "weight_decay": 1e-5,     # 权重衰减
    "dropout": 0.3,           # Dropout比例
    
    # 数据划分
    "train_ratio": 0.7,       # 训练集比例
    "val_ratio": 0.15,        # 验证集比例
    "test_ratio": 0.15,       # 测试集比例
    
    # 任务选择
    "task": "whistle",        # 'whistle' 或 'pulse'
}

# 创建输出目录
CONFIG["output_dir"].mkdir(exist_ok=True, parents=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. 加载带特征的bags
# ─────────────────────────────────────────────────────────────────────────────
print("[1/8] 加载带特征的bags...")

with open(CONFIG["bags_path"], 'rb') as f:
    bags = pickle.load(f)

# 过滤出有特征的bag
valid_bags = [b for b in bags if b["features"]["mfcc"] is not None]
print(f"  总Bag数: {len(bags)}")
print(f"  有效Bag数（有特征）: {len(valid_bags)}")

if len(valid_bags) < 50:
    print("\n  ⚠️  警告：有效Bag数量过少，建议补充音频文件后再训练！")

# ─────────────────────────────────────────────────────────────────────────────
# 3. 构建MIL数据集
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/8] 构建MIL数据集...")

class MILDataset(Dataset):
    """
    多实例学习数据集
    每个样本是一个bag，包含多个instance（MFCC帧）
    """
    def __init__(self, bags, task='whistle', max_frames=3000):
        self.bags = bags
        self.task = task
        self.max_frames = max_frames
    
    def __len__(self):
        return len(self.bags)
    
    def __getitem__(self, idx):
        bag = self.bags[idx]
        
        # 获取MFCC特征 (40, n_frames)
        mfcc = bag["features"]["mfcc"]
        
        # 转置为 (n_frames, 40)
        mfcc = mfcc.T
        
        # 截断或填充到固定长度
        n_frames = mfcc.shape[0]
        if n_frames > self.max_frames:
            mfcc = mfcc[:self.max_frames, :]
        elif n_frames < self.max_frames:
            pad = np.zeros((self.max_frames - n_frames, mfcc.shape[1]))
            mfcc = np.vstack([mfcc, pad])
        
        # 获取标签
        if self.task == 'whistle':
            label = bag["whistle_label"]
        elif self.task == 'pulse':
            label = bag["pulse_label"]
        else:
            raise ValueError(f"Unknown task: {self.task}")
        
        return {
            'bag': torch.FloatTensor(mfcc),
            'label': torch.FloatTensor([label]),
            'file_no': bag["file_no"],
            'bag_id': bag["bag_id"],
        }

# 创建数据集
dataset = MILDataset(valid_bags, task=CONFIG["task"])
print(f"  数据集大小: {len(dataset)}")

# 统计标签分布
labels = [b[CONFIG["task"] + "_label"] for b in valid_bags]
label_counts = np.bincount(labels)
print(f"  标签分布: 0类={label_counts[0]}, 1类={label_counts[1]}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. 数据划分
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/8] 划分训练/验证/测试集...")

# 按file_no分组，确保同一录音的bag在同一集合中
file_nos = list(set(b["file_no"] for b in valid_bags))
train_files, temp_files = train_test_split(
    file_nos, 
    test_size=CONFIG["val_ratio"]+CONFIG["test_ratio"],
    random_state=42,
    stratify=[sum(1 for b in valid_bags if b["file_no"]==f and b[CONFIG["task"]+"_label"]==1) for f in file_nos]
)
val_files, test_files = train_test_split(
    temp_files,
    test_size=CONFIG["test_ratio"]/(CONFIG["val_ratio"]+CONFIG["test_ratio"]),
    random_state=42
)

print(f"  训练文件: {len(train_files)} 个")
print(f"  验证文件: {len(val_files)} 个")
print(f"  测试文件: {len(test_files)} 个")

# 按文件划分bag
train_bags = [b for b in valid_bags if b["file_no"] in train_files]
val_bags = [b for b in valid_bags if b["file_no"] in val_files]
test_bags = [b for b in valid_bags if b["file_no"] in test_files]

print(f"  训练Bag: {len(train_bags)}")
print(f"  验证Bag: {len(val_bags)}")
print(f"  测试Bag: {len(test_bags)}")

# 创建DataLoader
train_dataset = MILDataset(train_bags, task=CONFIG["task"])
val_dataset = MILDataset(val_bags, task=CONFIG["task"])
test_dataset = MILDataset(test_bags, task=CONFIG["task"])

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

# ─────────────────────────────────────────────────────────────────────────────
# 5. 构建Attention-MIL模型
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/8] 构建Attention-MIL模型...")

class AttentionMIL(nn.Module):
    """
    Attention-based Multiple Instance Learning
    参考：Ilse et al., "Attention-based Deep Multiple Instance Learning", ICML 2018
    """
    def __init__(self, input_dim, hidden_dim, attention_dim, num_classes=2, dropout=0.3):
        super(AttentionMIL, self).__init__()
        
        # 实例编码器
        self.instance_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        
        # 分类器
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x, return_attention=False):
        """
        x: (batch, n_instances, input_dim)
        """
        # 实例编码
        h = self.instance_encoder(x)  # (batch, n_instances, hidden_dim)
        
        # 计算注意力权重
        a = self.attention(h)  # (batch, n_instances, 1)
        a = torch.softmax(a, dim=1)  # softmax over instances
        
        # 加权聚合
        z = torch.sum(a * h, dim=1)  # (batch, hidden_dim)
        
        # 分类
        logits = self.classifier(z)  # (batch, num_classes)
        
        if return_attention:
            return logits, a
        return logits

# 初始化模型
model = AttentionMIL(
    input_dim=CONFIG["input_dim"],
    hidden_dim=CONFIG["hidden_dim"],
    attention_dim=CONFIG["attention_dim"],
    num_classes=CONFIG["num_classes"],
    dropout=CONFIG["dropout"]
)

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  总参数量: {total_params:,}")
print(f"  可训练参数量: {trainable_params:,}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. 训练配置
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/8] 配置训练优化器...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  使用设备: {device}")

model = model.to(device)

# 损失函数（二分类用BCE，多分类用CrossEntropy）
if CONFIG["num_classes"] == 2:
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()

# 优化器
optimizer = optim.Adam(
    model.parameters(),
    lr=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"]
)

# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# ─────────────────────────────────────────────────────────────────────────────
# 7. 训练循环
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/8] 开始训练...")

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(loader, desc="Training", leave=False):
        x = batch['bag'].to(device)  # (batch, n_instances, input_dim)
        y = batch['label'].to(device)  # (batch, 1)
        
        optimizer.zero_grad()
        
        # 前向传播
        if CONFIG["num_classes"] == 2:
            logits = model(x)[:, 0]  # 取第一个logit
            loss = criterion(logits, y.squeeze())
            preds = torch.sigmoid(logits) > 0.5
        else:
            logits = model(x)
            loss = criterion(logits, y.squeeze().long())
            preds = torch.argmax(logits, dim=1)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.squeeze().cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    return avg_loss, accuracy

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            x = batch['bag'].to(device)
            y = batch['label'].to(device)
            
            if CONFIG["num_classes"] == 2:
                logits = model(x)[:, 0]
                loss = criterion(logits, y.squeeze())
                probs = torch.sigmoid(logits)
                preds = probs > 0.5
            else:
                logits = model(x)
                loss = criterion(logits, y.squeeze().long())
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)
            
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.squeeze().cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    return avg_loss, accuracy, all_probs, all_labels

# 训练记录
train_history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
}

best_val_loss = float('inf')
best_model_state = None
patience_counter = 0
early_stop_patience = 10

for epoch in range(CONFIG["num_epochs"]):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
    print(f"{'='*60}")
    
    # 训练
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    print(f"  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
    
    # 验证
    val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
    print(f"  验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
    
    # 记录历史
    train_history['train_loss'].append(train_loss)
    train_history['train_acc'].append(train_acc)
    train_history['val_loss'].append(val_loss)
    train_history['val_acc'].append(val_acc)
    
    # 学习率调度
    scheduler.step(val_loss)
    
    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict().copy()
        patience_counter = 0
        print(f"  ✓ 保存最佳模型 (val_loss={val_loss:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"  ⚠️  早停触发 (patience={early_stop_patience})")
            break

# ─────────────────────────────────────────────────────────────────────────────
# 8. 测试评估
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/8] 测试集评估...")

# 加载最佳模型
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("  ✓ 加载最佳模型")

# 测试
test_loss, test_acc, test_probs, test_labels = evaluate(model, test_loader, criterion, device)
print(f"  测试损失: {test_loss:.4f}")
print(f"  测试准确率: {test_acc:.4f}")

# 计算AUC
if len(np.unique(test_labels)) > 1:
    test_auc = roc_auc_score(test_labels, test_probs)
    print(f"  测试AUC: {test_auc:.4f}")
else:
    test_auc = None
    print("  测试AUC: N/A (单类别)")

# 分类报告
print("\n  分类报告:")
print(classification_report(test_labels, np.array(test_probs) > 0.5, 
                            target_names=['Negative', 'Positive']))

# 混淆矩阵
cm = confusion_matrix(test_labels, np.array(test_probs) > 0.5)
print(f"\n  混淆矩阵:\n{cm}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. 保存结果
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8/8] 保存训练结果...")

# 保存模型
model_path = CONFIG["output_dir"] / "mil_model.pth"
torch.save({
    'epoch': epoch,
    'model_state_dict': best_model_state if best_model_state else model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_history': train_history,
    'config': CONFIG,
}, model_path)
print(f"  ✓ 模型保存至: {model_path}")

# 保存训练历史
history_path = CONFIG["output_dir"] / "training_history.json"
with open(history_path, 'w', encoding='utf-8') as f:
    json.dump(train_history, f, indent=2)
print(f"  ✓ 训练历史保存至: {history_path}")

# 绘制训练曲线
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(train_history['train_loss'], label='Train Loss')
axes[0].plot(train_history['val_loss'], label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss Curve')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(train_history['train_acc'], label='Train Acc')
axes[1].plot(train_history['val_acc'], label='Val Acc')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy Curve')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
curve_path = CONFIG["output_dir"] / "training_curves.png"
plt.savefig(curve_path, dpi=150)
plt.close()
print(f"  ✓ 训练曲线保存至: {curve_path}")

# 生成统计报告
stats_report = f"""
================================================================================
Phase 2 - 模型训练统计
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

【基础配置】
  任务类型: {CONFIG['task']}
  输入维度: {CONFIG['input_dim']} (MFCC)
  隐藏层维度: {CONFIG['hidden_dim']}
  注意力维度: {CONFIG['attention_dim']}
  Batch大小: {CONFIG['batch_size']}
  训练轮数: {CONFIG['num_epochs']}
  学习率: {CONFIG['learning_rate']}

【数据规模】
  总有效Bag: {len(valid_bags)}
  训练集: {len(train_bags)} bags ({len(train_files)} files)
  验证集: {len(val_bags)} bags ({len(val_files)} files)
  测试集: {len(test_bags)} bags ({len(test_files)} files)

【训练结果】
  最终Epoch: {epoch+1}
  最佳验证损失: {best_val_loss:.4f}
  测试准确率: {test_acc:.4f}
  测试AUC: {test_auc if test_auc else 'N/A'}

【输出文件】
  mil_model.pth: {model_path}
  training_history.json: {history_path}
  training_curves.png: {curve_path}
  phase2_statistics.txt: {CONFIG['output_dir'] / 'phase2_statistics.txt'}
================================================================================
"""

stats_path = CONFIG["output_dir"] / "phase2_statistics.txt"
stats_path.write_text(stats_report, encoding='utf-8')
print(stats_report)

print("\n" + "="*80)
print("✅ Phase 2 全部任务完成！")
print("="*80)