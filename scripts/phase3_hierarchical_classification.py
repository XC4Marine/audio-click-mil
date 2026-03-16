# =============================================================================
# Phase 3: 分层类型细分 (Hierarchical Type Classification) - 修复版
# =============================================================================

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
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
print("Phase 3: 分层类型细分 (修复版)")
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

# ─────────────────────────────────────────────────────────────────────────────
# 3. 任务1: 哨声类型细分
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# 3. 任务1: 哨声类型细分 (修改版：仅基于形状)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/5] 哨声类型细分 (仅形状分类)...")

def classify_whistle_type(row):
    """
    仅根据频率变化轨迹将哨声分为 6 种形状：
    1. constant (平坦)
    2. upsweep (上升)
    3. downsweep (下降)
    4. concave (凹型/先降后升)
    5. convex (凸型/先升后降)
    6. complex (复杂/多起伏)
    """
    # 获取关键参数 (单位转换)
    freq_start = row.get("Start Frequency (kHz)", 0) * 1000  # Hz
    freq_end = row.get("End Frequency (kHz)", 0) * 1000      # Hz
    freq_min = row.get("Minimum Frequency (kHz)", 0) * 1000  # Hz (如果有此列)
    freq_max = row.get("Maximum Frequency (kHz)", 0) * 1000  # Hz (如果有此列)
    
    # 如果没有 Min/Max 列，用 Start/End 近似，或者尝试计算极值逻辑
    # 注意：如果您的 CSV 没有 Min/Max 列，我们需要用更简单的逻辑或假设 Start/End 代表趋势
    # 这里假设 CSV 中有 'Minimum Frequency (kHz)' 和 'Maximum Frequency (kHz)'
    # 如果没有，代码会自动降级使用 Start/End 逻辑
    
    has_min_max = ("Minimum Frequency (kHz)" in row and "Maximum Frequency (kHz)" in row) and \
                  (row.get("Minimum Frequency (kHz)", 0) > 0) and (row.get("Maximum Frequency (kHz)", 0) > 0)

    if has_min_max:
        # 使用极值点判断形状
        diff_start_end = freq_end - freq_start
        threshold = 1000 # 1kHz 容差
        
        # 1. Constant: 起止频率接近，且最大最小值差异不大
        if abs(diff_start_end) < threshold and (freq_max - freq_min) < 2000:
            return "constant"
        
        # 2. Concave (U型): 最小值明显小于起止频率 (先降后升)
        # 条件：Min 比 Start 和 End 都小很多
        if freq_min < min(freq_start, freq_end) - 2000:
            return "concave"
        
        # 3. Convex (∩型): 最大值明显大于起止频率 (先升后降)
        # 条件：Max 比 Start 和 End 都大很多
        if freq_max > max(freq_start, freq_end) + 2000:
            return "convex"
        
        # 4. Upsweep: 整体趋势向上，且没有明显的先升后降
        if freq_end > freq_start + threshold:
            return "upsweep"
            
        # 5. Downsweep: 整体趋势向下
        if freq_end < freq_start - threshold:
            return "downsweep"
            
        # 6. Complex: 如果既有极大值又有极小值，或者起伏很大
        if (freq_max - freq_min) > 5000: 
            return "complex"
            
        # 默认 fallback
        if freq_end > freq_start: return "upsweep"
        elif freq_end < freq_start: return "downsweep"
        else: return "constant"

    else:
        # --- 降级方案：如果 CSV 中没有 Min/Max 列，只能根据 Start/End 简单判断 ---
        # 这种情况下很难区分 Concave/Convex/Complex，只能分为 3 类或强行模拟
        # 为了凑齐 6 类，我们可以结合 Duration 或 Frequency Change 列辅助判断
        freq_change = row.get("Frequcncy Change (kHz)", 0) * 1000 # 注意列名拼写 Frequcncy
        duration = row.get("Duration (ms)", 0) / 1000
        
        diff = freq_end - freq_start
        threshold = 1000
        
        if abs(diff) < threshold:
            # 可能是 constant，也可能是复杂的往复运动抵消了
            # 如果有 "No. Local Extrema" (极值点数量)，可以用它
            extrema = row.get("No. Local Extrema", 0)
            if extrema > 2:
                return "complex"
            elif extrema == 1: 
                # 只有一个极值点，可能是 concave 或 convex，无法确定方向，暂归为 complex 或 constant
                return "complex" 
            else:
                return "constant"
        elif diff > 0:
            # 上升趋势
            extrema = row.get("No. Local Extrema", 0)
            if extrema > 1: return "complex" # 有波动
            # 检查是否有 "No. Inflection Point" (拐点) 辅助判断 convex/concave
            # 如果没有更多数据，暂时归为 upsweep
            return "upsweep"
        else:
            # 下降趋势
            extrema = row.get("No. Local Extrema", 0)
            if extrema > 1: return "complex"
            return "downsweep"

# 应用新函数
whistle_df["whistle_type"] = whistle_df.apply(classify_whistle_type, axis=1)

# 统计并打印
whistle_type_dist = whistle_df["whistle_type"].value_counts()
print("\n  【新的哨声形状分布 (6类)】")
for wh_type, count in whistle_type_dist.items():
    print(f"    {wh_type}: {count} ({count/len(whistle_df)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 4. 任务2: 脉冲子类型区分
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] 脉冲子类型区分...")

click_df["pulse_subtype"] = "click"
burst_df["pulse_subtype"] = "burst"
buzz_df["pulse_subtype"] = "buzz"

pulse_df = pd.concat([
    click_df[["Ori_file_num(No.)", "Train_start(s)", "Train_end(s)", "pulse_subtype"]],
    burst_df[["Ori_file_num(No.)", "Train_start(s)", "Train_end(s)", "pulse_subtype"]],
    buzz_df[["Ori_file_num(No.)", "Train_start(s)", "Train_end(s)", "pulse_subtype"]],
], ignore_index=True)

pulse_subtype_dist = pulse_df["pulse_subtype"].value_counts()
print("\n  【脉冲子类型分布】")
for subtype, count in pulse_subtype_dist.items():
    print(f"    {subtype}: {count} ({count/len(pulse_df)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. 任务3: 多标签分类模型
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] 训练多标签分类模型...")

bags_path = CONFIG["phase1_output"] / "bags_with_features.pkl"
with open(bags_path, 'rb') as f:
    bags = pickle.load(f)

valid_bags = [b for b in bags if b["features"].get("mfcc") is not None]
print(f"  ✓ 有效Bag数: {len(valid_bags)}")

available_files = sorted(list(set([b["file_no"] for b in valid_bags])))
print(f"  正在评估的文件列表: {available_files}")

# 🔧 修复：明确指定列名，避免动态匹配出错
# 根据日志：'Original Audio File (No.)', 'Whistle Begins (s)'
WHISTLE_FILE_COL = "Original Audio File (No.)"
WHISTLE_TIME_COL = "Whistle Begins (s)"

# 哨声特征提取
whistle_features = []
whistle_labels = []

print("\n  开始匹配哨声样本...")
for bag in tqdm(valid_bags, desc="匹配哨声"):
    file_no = bag["file_no"]
    bag_start = bag["bag_start_sec"]
    bag_end = bag_start + CONFIG["bag_duration"]
    mfcc = bag["features"]["mfcc"]
    
    if mfcc is None:
        continue
    
    # 🔧 修复：正确的筛选逻辑
    # 确保列存在且类型正确
    if WHISTLE_FILE_COL in whistle_df.columns and WHISTLE_TIME_COL in whistle_df.columns:
        # 转换类型以匹配
        mask_file = whistle_df[WHISTLE_FILE_COL].astype(int) == int(file_no)
        mask_time = (whistle_df[WHISTLE_TIME_COL].astype(float) >= bag_start) & \
                    (whistle_df[WHISTLE_TIME_COL].astype(float) <= bag_end)
        
        bag_whistles = whistle_df[mask_file & mask_time]
    else:
        bag_whistles = pd.DataFrame()
    
    # 🔧 修复：缩进修正，独立判断是否有数据
    if len(bag_whistles) > 0:
        main_type = bag_whistles["whistle_type"].mode().values[0]
        whistle_features.append(mfcc)
        whistle_labels.append(main_type)

print(f"  ✓ 哨声训练样本: {len(whistle_features)}")

# 脉冲特征提取
pulse_features = []
pulse_labels = []

print("\n  开始匹配脉冲样本...")
for bag in tqdm(valid_bags, desc="匹配脉冲"):
    file_no = bag["file_no"]
    bag_start = bag["bag_start_sec"]
    bag_end = bag_start + CONFIG["bag_duration"]
    mfcc = bag["features"]["mfcc"]
    
    if mfcc is None:
        continue
    
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

# 初始化变量以防止未定义错误
whistle_clf = None
whistle_le = None
whistle_cv_mean = None

pulse_clf = None
pulse_le = None
pulse_cv_mean = None

# 训练哨声分类器
if len(whistle_features) >= 5:
    print("\n  训练哨声类型分类器...")
    whistle_le = LabelEncoder()
    whistle_y = whistle_le.fit_transform(whistle_labels)
    
    whistle_clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    whistle_clf.fit(whistle_features, whistle_y)
    
    cv_scores = cross_val_score(whistle_clf, whistle_features, whistle_y, cv=3)
    whistle_cv_mean = cv_scores.mean()
    print(f"    哨声分类器 CV 准确率: {whistle_cv_mean:.4f} (+/- {cv_scores.std()*2:.4f})")
    
    joblib.dump(whistle_clf, CONFIG["phase3_output"] / "whistle_type_classifier.pkl")
    joblib.dump(whistle_le, CONFIG["phase3_output"] / "whistle_type_encoder.pkl")
    print("    ✓ 哨声分类器已保存")
else:
    print(f"    ⚠ 哨声样本不足 ({len(whistle_features)} < 5)，跳过训练")

# 训练脉冲分类器
if len(pulse_features) >= 5:
    print("\n  训练脉冲子类型分类器...")
    pulse_le = LabelEncoder()
    pulse_y = pulse_le.fit_transform(pulse_labels)
    
    pulse_clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    pulse_clf.fit(pulse_features, pulse_y)
    
    cv_scores = cross_val_score(pulse_clf, pulse_features, pulse_y, cv=3)
    pulse_cv_mean = cv_scores.mean()
    print(f"    脉冲分类器 CV 准确率: {pulse_cv_mean:.4f} (+/- {cv_scores.std()*2:.4f})")
    
    joblib.dump(pulse_clf, CONFIG["phase3_output"] / "pulse_subtype_classifier.pkl")
    joblib.dump(pulse_le, CONFIG["phase3_output"] / "pulse_subtype_encoder.pkl")
    print("    ✓ 脉冲分类器已保存")
else:
    print(f"    ⚠ 脉冲样本不足 ({len(pulse_features)} < 5)，跳过训练")

# ─────────────────────────────────────────────────────────────────────────────
# 6. 任务4 & 5: 类型分布统计与报告
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] 生成评估报告...")

# 🔧 修复：安全的格式化字符串，处理 None 值
whistle_cv_str = f"{whistle_cv_mean:.4f}" if whistle_cv_mean is not None else "N/A (样本不足)"
pulse_cv_str = f"{pulse_cv_mean:.4f}" if pulse_cv_mean is not None else "N/A (样本不足)"
whistle_cls_count = len(whistle_le.classes_) if whistle_le else 0
pulse_cls_count = len(pulse_le.classes_) if pulse_le else 0

report = f"""
================================================================================
Phase 3 - 分层类型细分评估报告
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

【数据概览】
  已处理文件: {available_files}
  有效Bag数: {len(valid_bags)}

================================================================================
【哨声类型细分】
================================================================================

  总哨声数量: {len(whistle_df)}
  可分类哨声 (在当前文件中): {len(whistle_df[whistle_df[WHISTLE_FILE_COL].astype(int).isin(available_files)])}

  类型分布:
"""

for wh_type, count in whistle_type_dist.items():
    pct = count / len(whistle_df) * 100
    report += f"    {wh_type}: {count} ({pct:.1f}%)\n"

report += f"""
  哨声分类器训练样本: {len(whistle_features)}
  哨声分类器类别数: {whistle_cls_count}
  哨声分类器 CV 准确率: {whistle_cv_str}

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
  脉冲分类器类别数: {pulse_cls_count}
  脉冲分类器 CV 准确率: {pulse_cv_str}

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
【诊断信息】
================================================================================

  如果哨声样本为 0，请检查：
  1. 文件编号是否匹配 (当前使用列: '{WHISTLE_FILE_COL}')
  2. 时间戳是否重叠 (当前使用列: '{WHISTLE_TIME_COL}')
  3. Bag 的时间范围是否覆盖了标注时间
  
  当前 Bag 时间范围示例:
"""

# 添加几个示例以便调试
for i, bag in enumerate(valid_bags[:3]):
    report += f"    Bag {i}: File {bag['file_no']}, Time {bag['bag_start_sec']:.1f}-{bag['bag_start_sec']+30:.1f}s\n"

report += """
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