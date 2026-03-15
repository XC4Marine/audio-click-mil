# =============================================================================
# Phase 3: 分层类型细分 (Hierarchical Type Classification)
# =============================================================================

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. 配置参数
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "data_root": Path("D:/Project_Github/audio_click_mil/data"),
    "phase1_output": Path("D:/Project_Github/audio_click_mil/results/phase1"),
    "phase2_output": Path("D:/Project_Github/audio_click_mil/results/phase2"),
    "phase3_output": Path("D:/Project_Github/audio_click_mil/results/phase3"),
    "target_sr": 48000,
    "bag_duration": 30,
}


CONFIG["phase3_output"].mkdir(exist_ok=True, parents=True)

print("="*80)
print("Phase 3: 分层类型细分")
print("="*80)

# ─────────────────────────────────────────────────────────────────────────────
# 2. 加载标注数据
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/5] 加载标注数据...")

whistle_df = pd.read_csv(CONFIG["data_root"] / "WhistleParameters-clean.csv")
click_df = pd.read_csv(CONFIG["data_root"] / "ClickTrains.csv")
burst_df = pd.read_csv(CONFIG["data_root"] / "BurstPulseTrains.csv")
buzz_df = pd.read_csv(CONFIG["data_root"] / "BuzzTrains.csv")

# 统一列名
whistle_df.columns = whistle_df.columns.str.strip()
click_df.columns = click_df.columns.str.strip()
burst_df.columns = burst_df.columns.str.strip()
buzz_df.columns = buzz_df.columns.str.strip()

print(f"  ✓ 哨声标注: {len(whistle_df)} 条")
print(f"  ✓ Click标注: {len(click_df)} 条")
print(f"  ✓ Burst标注: {len(burst_df)} 条")
print(f"  ✓ Buzz标注: {len(buzz_df)} 条")
print("  哨声列名:", whistle_df.columns.tolist())
print("  脉冲列名:", click_df.columns.tolist())
# ─────────────────────────────────────────────────────────────────────────────
# 3. 任务1: 哨声类型细分
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/5] 哨声类型细分...")

def classify_whistle_type(row):
    # 注意：实际列名单位是 kHz 和 ms
    freq_start = row.get("Start Frequency (kHz)", 0) * 1000  # kHz→Hz
    freq_end = row.get("End Frequency (kHz)", 0) * 1000      # kHz→Hz
    freq_mean = (freq_start + freq_end) / 2
    duration = row.get("Duration (ms)", 0) / 1000            # ms→s
    
    # 频率分类
    if freq_mean < 5000:
        freq_type = "low"
    elif freq_mean < 15000:
        freq_type = "mid"
    else:
        freq_type = "high"
    
    # 调频类型
    freq_diff = abs(freq_end - freq_start)
    if freq_diff < 1000:
        mod_type = "constant"
    elif freq_end > freq_start:
        mod_type = "upsweep"
    else:
        mod_type = "downsweep"
    
    # 持续时间分类
    if duration < 0.1:
        dur_type = "short"
    elif duration < 0.5:
        dur_type = "medium"
    else:
        dur_type = "long"
    
    return f"{freq_type}_{mod_type}_{dur_type}"

whistle_df["whistle_type"] = whistle_df.apply(classify_whistle_type, axis=1)

# 统计哨声类型分布
whistle_type_dist = whistle_df["whistle_type"].value_counts()
print("\n  【哨声类型分布】")
for wh_type, count in whistle_type_dist.items():
    print(f"    {wh_type}: {count} ({count/len(whistle_df)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 4. 任务2: 脉冲子类型区分
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] 脉冲子类型区分...")

# 添加子类型标签
click_df["pulse_subtype"] = "click"
burst_df["pulse_subtype"] = "burst"
buzz_df["pulse_subtype"] = "buzz"

# 合并脉冲数据
pulse_df = pd.concat([
    click_df[["Ori_file_num(No.)", "Train_start(s)", "Train_end(s)", "pulse_subtype"]],
    burst_df[["Ori_file_num(No.)", "Train_start(s)", "Train_end(s)", "pulse_subtype"]],
    buzz_df[["Ori_file_num(No.)", "Train_start(s)", "Train_end(s)", "pulse_subtype"]],
], ignore_index=True)

# 脉冲子类型分布
pulse_subtype_dist = pulse_df["pulse_subtype"].value_counts()
print("\n  【脉冲子类型分布】")
for subtype, count in pulse_subtype_dist.items():
    print(f"    {subtype}: {count} ({count/len(pulse_df)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. 任务3: 多标签分类模型
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] 训练多标签分类模型...")

# 加载 Phase 1 的 bags 和模型
bags_path = CONFIG["phase1_output"] / "bags_with_features.pkl"
with open(bags_path, 'rb') as f:
    bags = pickle.load(f)

valid_bags = [b for b in bags if b["features"].get("mfcc") is not None]
print(f"  ✓ 有效Bag数: {len(valid_bags)}")

# 构建训练数据（基于标注匹配）
available_files = [1, 2, 3, 4, 5,6,7,8,9,10]

# 哨声特征提取
whistle_features = []
whistle_labels = []

for bag in valid_bags:
    file_no = bag["file_no"]
    bag_start = bag["bag_start_sec"]
    bag_end = bag_start + CONFIG["bag_duration"]
    mfcc = bag["features"]["mfcc"]
    
    if mfcc is None:
        continue
    
    # 先打印调试信息
    if bag["file_no"] in [1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15]:
        print(f"    Bag {bag['bag_id']}: file={file_no}, time={bag_start}-{bag_end}s")
        print(f"    哨声文件列名: {whistle_df.columns.tolist()}")

    # 尝试多种列名匹配
    file_col = None
    for col in whistle_df.columns:
        if "file" in col.lower() or "no" in col.lower():
            file_col = col
            break

    time_col = None
    for col in whistle_df.columns:
        if "begin" in col.lower() or "start" in col.lower():
            time_col = col
            break

    if file_col and time_col:
        bag_whistles = whistle_df[
            (whistle_df[file_col].astype(str) == str(file_no)) &
            (whistle_df[time_col].astype(float) >= bag_start) &
            (whistle_df[time_col].astype(float) <= bag_end)
        ]
    else:
        bag_whistles = pd.DataFrame()  # 空
    
        if len(bag_whistles) > 0:
            # 提取主要哨声类型
            main_type = bag_whistles["whistle_type"].mode().values[0]
            whistle_features.append(mfcc)
            whistle_labels.append(main_type)

print(f"  ✓ 哨声训练样本: {len(whistle_features)}")

# 脉冲特征提取
pulse_features = []
pulse_labels = []

for bag in valid_bags:
    file_no = bag["file_no"]
    bag_start = bag["bag_start_sec"]
    bag_end = bag_start + CONFIG["bag_duration"]
    mfcc = bag["features"]["mfcc"]
    
    if mfcc is None:
        continue
    
    # 匹配该bag时间范围内的脉冲
    bag_pulses = pulse_df[
        (pulse_df["Ori_file_num(No.)"].astype(int) == file_no) &
        (pulse_df["Train_start(s)"] >= bag_start) &
        (pulse_df["Train_start(s)"] <= bag_end)
    ]
    
    if len(bag_pulses) > 0:
        main_subtype = bag_pulses["pulse_subtype"].mode().values[0]
        pulse_features.append(mfcc)
        pulse_labels.append(main_subtype)

print(f"  ✓ 脉冲训练样本: {len(pulse_features)}")

# 训练哨声分类器
if len(whistle_features) >= 5:
    print("\n  训练哨声类型分类器...")
    whistle_le = LabelEncoder()
    whistle_y = whistle_le.fit_transform(whistle_labels)
    
    whistle_clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    whistle_clf.fit(whistle_features, whistle_y)
    
    # 交叉验证
    cv_scores = cross_val_score(whistle_clf, whistle_features, whistle_y, cv=3)
    print(f"    哨声分类器 CV 准确率: {cv_scores.mean():.4f if whistle_clf is not None else 'N/A'} (+/- {cv_scores.std()*2:.4f})")
    
    joblib.dump(whistle_clf, CONFIG["phase3_output"] / "whistle_type_classifier.pkl")
    joblib.dump(whistle_le, CONFIG["phase3_output"] / "whistle_type_encoder.pkl")
    print("    ✓ 哨声分类器已保存")
else:
    print("    ⚠ 哨声样本不足，跳过训练")
    whistle_clf = None
    whistle_le = None
whistle_cv_mean = cv_scores.mean() if whistle_clf is not None else None

# 训练脉冲分类器
if len(pulse_features) >= 5:
    print("\n  训练脉冲子类型分类器...")
    pulse_le = LabelEncoder()
    pulse_y = pulse_le.fit_transform(pulse_labels)
    
    pulse_clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    pulse_clf.fit(pulse_features, pulse_y)
    
    cv_scores = cross_val_score(pulse_clf, pulse_features, pulse_y, cv=3)
    print(f"    脉冲分类器 CV 准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    joblib.dump(pulse_clf, CONFIG["phase3_output"] / "pulse_subtype_classifier.pkl")
    joblib.dump(pulse_le, CONFIG["phase3_output"] / "pulse_subtype_encoder.pkl")
    print("    ✓ 脉冲分类器已保存")
else:
    print("    ⚠ 脉冲样本不足，跳过训练")
    pulse_clf = None
    pulse_le = None
pulse_cv_mean = cv_scores.mean() if pulse_clf is not None else None
# ─────────────────────────────────────────────────────────────────────────────
# 6. 任务4 & 5: 类型分布统计与报告
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] 生成评估报告...")

# 哨声详细统计
whistle_stats = whistle_df.groupby("whistle_type").agg({
    "Start Frequency (kHz)": ["mean", "std"],
    "Duration (ms)": ["mean", "std"],
    "Original Audio File (No.)": "count"
}).round(3)

# 脉冲详细统计
pulse_stats = pulse_df.groupby("pulse_subtype").agg({
    "Ori_file_num(No.)": "count"
}).round(3)

# 生成报告
report = f"""
================================================================================
Phase 3 - 分层类型细分评估报告
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

【数据概览】
  总标注文件: 35 个
  已处理文件: {available_files}
  有效Bag数: {len(valid_bags)}

================================================================================
【哨声类型细分】
================================================================================

  总哨声数量: {len(whistle_df)}
  可分类哨声: {len(whistle_df[whistle_df['Original Audio File (No.)'].astype(int).isin(available_files)])}

  类型分布:
"""

for wh_type, count in whistle_type_dist.items():
    pct = count / len(whistle_df) * 100
    report += f"    {wh_type}: {count} ({pct:.1f}%)\n"

report += f"""
  哨声分类器训练样本: {len(whistle_features)}
  哨声分类器类别: {len(whistle_le.classes_) if whistle_le else 0}
  哨声分类器 CV 准确率: {whistle_cv_mean:.4f if whistle_clf is not None else 'N/A'}

================================================================================
【脉冲子类型区分】
================================================================================

  总脉冲数量: {len(pulse_df)}
  
  子类型分布:
"""

for subtype, count in pulse_subtype_dist.items():
    pct = count / len(pulse_df) * 100
    report += f"    {subtype}: {count} ({pct:.1f}%)\n"

report += f"""
  脉冲分类器训练样本: {len(pulse_features)}
  脉冲分类器类别: {len(pulse_le.classes_) if pulse_le else 0}
  脉冲分类器 CV 准确率: {pulse_cv_mean:.4f if pulse_clf is not None else 'N/A'}

================================================================================
【输出文件】
================================================================================

  哨声分类器: {CONFIG['phase3_output'] / 'whistle_type_classifier.pkl'}
  哨声编码器: {CONFIG['phase3_output'] / 'whistle_type_encoder.pkl'}
  脉冲分类器: {CONFIG['phase3_output'] / 'pulse_subtype_classifier.pkl'}
  脉冲编码器: {CONFIG['phase3_output'] / 'pulse_subtype_encoder.pkl'}
  哨声类型分布: {CONFIG['phase3_output'] / 'whistle_type_distribution.csv'}
  脉冲子类型分布: {CONFIG['phase3_output'] / 'pulse_subtype_distribution.csv'}
  评估报告: {CONFIG['phase3_output'] / '03_classification_report.txt'}

================================================================================
【下一步建议】
================================================================================

  1. 收集剩余 30 个音频文件以扩充训练数据
  2. 增加更多声学特征（频谱质心、过零率、频谱滚降点）
  3. 尝试深度学习模型（CNN/LSTM）进行端到端分类
  4. 考虑数据增强技术处理类别不平衡

================================================================================
"""

# 保存报告
report_path = CONFIG["phase3_output"] / "03_classification_report.txt"
report_path.write_text(report, encoding='utf-8')

# 保存分布统计
whistle_type_dist.to_csv(CONFIG["phase3_output"] / "whistle_type_distribution.csv", encoding='utf-8-sig')
pulse_subtype_dist.to_csv(CONFIG["phase3_output"] / "pulse_subtype_distribution.csv", encoding='utf-8-sig')

print(f"  ✓ 评估报告: {report_path}")
print(f"  ✓ 哨声类型分布: {CONFIG['phase3_output'] / 'whistle_type_distribution.csv'}")
print(f"  ✓ 脉冲子类型分布: {CONFIG['phase3_output'] / 'pulse_subtype_distribution.csv'}")

print("\n" + "="*80)
print("✅ Phase 3 全部任务完成！")
print("="*80)
print(report)