# =============================================================================
# Phase 1: 基础Bag-level检测
# =============================================================================
# 任务：
#   - 子任务1: 音频特征提取（MFCC，48kHz）
#   - 子任务2: 填充bags.pkl中的features字段
#   - 子任务3: 训练bag-level分类器（哨声 + 脉冲）
#   - 子任务4: 评估并保存模型
# =============================================================================

import pandas as pd
import numpy as np
import pickle
import librosa
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. 配置参数
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "data_root": Path("D:/Project_Github/audio_click_mil/data"),
    "audio_root": Path("D:/Project_Github/audio_click_mil/data/original_audio"),
    "phase0_output": Path("D:/Project_Github/audio_click_mil/results/phase0"),
    "phase1_output": Path("D:/Project_Github/audio_click_mil/results/phase1"),
    "target_sr": 48000,           # 目标采样率
    "bag_duration": 30,           # bag时长（秒）
    "mfcc_n_mfcc": 20,            # MFCC维度
    "mfcc_hop_length": 512,       # 跳帧长度
    "mfcc_n_fft": 2048,           # FFT窗口
    "test_size": 0.3,             # 测试集比例
    "random_state": 42,
}

# 创建输出目录
CONFIG["phase1_output"].mkdir(exist_ok=True, parents=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. 加载Phase 0的bags
# ─────────────────────────────────────────────────────────────────────────────
print("="*80)
print("Phase 1: 基础Bag-level检测")
print("="*80)

print("\n[1/6] 加载Phase 0的bags.pkl...")
bags_path = CONFIG["phase0_output"] / "bags.pkl"
with open(bags_path, 'rb') as f:
    bags = pickle.load(f)

print(f"  ✓ 加载 {len(bags)} 个bag")

# ─────────────────────────────────────────────────────────────────────────────
# 3. 子任务1: 音频特征提取（MFCC）
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/6] 音频特征提取（MFCC, 48kHz）...")

def extract_mfcc_features(audio_path, target_sr=48000, n_mfcc=20, 
                          hop_length=512, n_fft=2048):
    """
    提取音频的MFCC特征
    返回: (n_frames, n_mfcc) 的numpy数组
    """
    try:
        # 加载音频并降采样
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        
        # 提取MFCC
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=n_mfcc, 
            hop_length=hop_length, n_fft=n_fft
        )
        
        # 转置为 (n_frames, n_mfcc)
        mfcc = mfcc.T
        
        return mfcc, len(y) / target_sr  # 返回特征和时长
        
    except Exception as e:
        print(f"  ⚠ 音频加载失败 {audio_path}: {e}")
        return None, 0

def get_audio_path(file_no, audio_root):
    """根据file_no生成音频文件路径"""
    # 文件名格式：Ori_Recording_01.wav, Ori_Recording_02.wav, ...
    filename = f"Ori_Recording_{file_no:02d}.wav"
    return audio_root / filename

# 统计可用的音频文件
available_files = []
for f in CONFIG["audio_root"].glob("Ori_Recording_*.wav"):
    try:
        file_no = int(f.stem.split('_')[-1])
        available_files.append(file_no)
    except:
        pass

available_files = sorted(available_files)
print(f"  可用音频文件: {available_files}")
print(f"  共 {len(available_files)} 个音频文件")

# 为每个录音提取MFCC特征
audio_features = {}  # {file_no: mfcc_array}

for file_no in tqdm(available_files, desc="提取MFCC特征"):
    audio_path = get_audio_path(file_no, CONFIG["audio_root"])
    mfcc, duration = extract_mfcc_features(
        audio_path,
        target_sr=CONFIG["target_sr"],
        n_mfcc=CONFIG["mfcc_n_mfcc"],
        hop_length=CONFIG["mfcc_hop_length"],
        n_fft=CONFIG["mfcc_n_fft"]
    )
    
    if mfcc is not None:
        audio_features[file_no] = {
            "mfcc": mfcc,
            "duration": duration,
            "path": str(audio_path),
        }

print(f"  ✓ 成功提取 {len(audio_features)} 个音频的特征")

# ─────────────────────────────────────────────────────────────────────────────
# 4. 子任务2: 填充bags中的features字段
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/6] 填充Bag特征...")

# 计算每个bag对应的MFCC帧范围
frames_per_second = CONFIG["target_sr"] / CONFIG["mfcc_hop_length"]
frames_per_bag = int(CONFIG["bag_duration"] * frames_per_second)

print(f"  每秒帧数: {frames_per_second:.1f}")
print(f"  每Bag帧数: {frames_per_bag}")

filled_count = 0
for bag in tqdm(bags, desc="填充特征"):
    file_no = bag["file_no"]
    bag_id = bag["bag_id"]
    
    if file_no in audio_features:
        mfcc_full = audio_features[file_no]["mfcc"]
        
        # 计算该bag对应的帧范围
        start_frame = int(bag["bag_start_sec"] * frames_per_second)
        end_frame = start_frame + frames_per_bag
        
        # 确保不超出范围
        if end_frame <= len(mfcc_full):
            bag_mfcc = mfcc_full[start_frame:end_frame]
            
            # 聚合为bag-level特征（均值+标准差）
            mfcc_mean = np.mean(bag_mfcc, axis=0)
            mfcc_std = np.std(bag_mfcc, axis=0)
            bag_features = np.concatenate([mfcc_mean, mfcc_std])
            
            bag["features"]["mfcc"] = bag_features
            bag["features"]["mfcc_mean"] = mfcc_mean
            bag["features"]["mfcc_std"] = mfcc_std
            bag["audio_path"] = audio_features[file_no]["path"]
            filled_count += 1
        else:
            bag["features"]["mfcc"] = None
    else:
        bag["features"]["mfcc"] = None

print(f"  ✓ 成功填充 {filled_count}/{len(bags)} 个bag的特征")
print(f"  ⚠ 剩余 {len(bags) - filled_count} 个bag无音频特征（对应录音文件缺失）")

# 保存更新后的bags
updated_bags_path = CONFIG["phase1_output"] / "bags_with_features.pkl"
with open(updated_bags_path, 'wb') as f:
    pickle.dump(bags, f)
print(f"  ✓ 已保存: {updated_bags_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. 子任务3: 准备训练数据
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/6] 准备训练数据...")

# 筛选有特征的bag
valid_bags = [b for b in bags if b["features"]["mfcc"] is not None]
print(f"  有效Bag数: {len(valid_bags)}")

# 提取特征矩阵和标签
X = np.array([b["features"]["mfcc"] for b in valid_bags])
y_whistle = np.array([b["whistle_label"] for b in valid_bags])
y_pulse = np.array([b["pulse_label"] for b in valid_bags])

print(f"  特征矩阵形状: {X.shape}")
print(f"  哨声标签分布: {np.sum(y_whistle)} 正 / {len(y_whistle) - np.sum(y_whistle)} 负")
print(f"  脉冲标签分布: {np.sum(y_pulse)} 正 / {len(y_pulse) - np.sum(y_pulse)} 负")

# 划分训练集和测试集（按file_no分组，避免数据泄漏）
file_nos = np.array([b["file_no"] for b in valid_bags])
unique_files = np.unique(file_nos)

# 确保训练集和测试集的文件不重叠
train_files, test_files = train_test_split(
    unique_files, test_size=CONFIG["test_size"], 
    random_state=CONFIG["random_state"]
)

train_mask = np.isin(file_nos, train_files)
test_mask = np.isin(file_nos, test_files)

X_train, X_test = X[train_mask], X[test_mask]
y_whistle_train, y_whistle_test = y_whistle[train_mask], y_whistle[test_mask]
y_pulse_train, y_pulse_test = y_pulse[train_mask], y_pulse[test_mask]

print(f"  训练集: {len(X_train)} 个bag ({len(train_files)} 个文件)")
print(f"  测试集: {len(X_test)} 个bag ({len(test_files)} 个文件)")

# ─────────────────────────────────────────────────────────────────────────────
# 6. 子任务4: 训练Bag-level分类器
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/6] 训练Bag-level分类器...")

# 训练哨声检测模型
print("  训练哨声检测模型...")
whistle_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=CONFIG["random_state"],
    n_jobs=-1
)
whistle_model.fit(X_train, y_whistle_train)

# 训练脉冲检测模型
print("  训练脉冲检测模型...")
pulse_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=CONFIG["random_state"],
    n_jobs=-1
)
pulse_model.fit(X_train, y_pulse_train)

# ─────────────────────────────────────────────────────────────────────────────
# 7. 评估模型
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/6] 模型评估...")

# 哨声模型评估
whistle_pred = whistle_model.predict(X_test)
whistle_proba = whistle_model.predict_proba(X_test)[:, 1]

print("\n" + "="*60)
print("【哨声检测模型 - 测试集评估】")
print("="*60)
print(classification_report(y_whistle_test, whistle_pred, 
                            target_names=['Negative', 'Positive']))

if len(np.unique(y_whistle_test)) > 1:
    whistle_auc = roc_auc_score(y_whistle_test, whistle_proba)
    print(f"  ROC-AUC: {whistle_auc:.4f}")

# 脉冲模型评估
pulse_pred = pulse_model.predict(X_test)
pulse_proba = pulse_model.predict_proba(X_test)[:, 1]

print("\n" + "="*60)
print("【脉冲检测模型 - 测试集评估】")
print("="*60)
print(classification_report(y_pulse_test, pulse_pred, 
                            target_names=['Negative', 'Positive']))

if len(np.unique(y_pulse_test)) > 1:
    pulse_auc = roc_auc_score(y_pulse_test, pulse_proba)
    print(f"  ROC-AUC: {pulse_auc:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. 保存模型和结果
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("保存模型和结果...")
print("="*80)

# 保存模型
whistle_model_path = CONFIG["phase1_output"] / "whistle_detector.pkl"
pulse_model_path = CONFIG["phase1_output"] / "pulse_detector.pkl"

joblib.dump(whistle_model, whistle_model_path)
joblib.dump(pulse_model, pulse_model_path)

print(f"  ✓ 哨声模型: {whistle_model_path}")
print(f"  ✓ 脉冲模型: {pulse_model_path}")

# 生成评估报告
eval_report = f"""
================================================================================
Phase 1 - Bag-level检测模型评估报告
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

【数据配置】
  音频文件数: {len(available_files)}
  有效Bag数: {len(valid_bags)}
  训练集Bag数: {len(X_train)} (文件: {list(train_files)})
  测试集Bag数: {len(X_test)} (文件: {list(test_files)})

【特征配置】
  特征类型: MFCC
  MFCC维度: {CONFIG['mfcc_n_mfcc']}
  采样率: {CONFIG['target_sr']} Hz
  每Bag特征维度: {X.shape[1]}

【哨声检测模型】
  模型类型: RandomForestClassifier
  测试集正样本数: {np.sum(y_whistle_test)}
  测试集负样本数: {len(y_whistle_test) - np.sum(y_whistle_test)}
"""

if len(np.unique(y_whistle_test)) > 1:
    eval_report += f"  ROC-AUC: {whistle_auc:.4f}\n"

eval_report += f"""
【脉冲检测模型】
  模型类型: RandomForestClassifier
  测试集正样本数: {np.sum(y_pulse_test)}
  测试集负样本数: {len(y_pulse_test) - np.sum(y_pulse_test)}
"""

if len(np.unique(y_pulse_test)) > 1:
    eval_report += f"  ROC-AUC: {pulse_auc:.4f}\n"

eval_report += f"""
【输出文件】
  哨声模型: {whistle_model_path}
  脉冲模型: {pulse_model_path}
  带特征bags: {updated_bags_path}

【下一步建议】
  → Phase 2: 实例级定位（从bag预测反推具体时间位置）
  → 建议收集更多音频文件以提升模型泛化能力

================================================================================
"""

eval_report_path = CONFIG["phase1_output"] / "01_evaluation_report.txt"
eval_report_path.write_text(eval_report, encoding='utf-8')
print(f"  ✓ 评估报告: {eval_report_path}")

print("\n" + "="*80)
print("✅ Phase 1 全部任务完成！")
print("="*80)
print(eval_report)