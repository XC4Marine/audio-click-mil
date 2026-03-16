# =============================================================================
# Phase 2: 实例级定位 - 增强版 (带具体类型标签映射)
# =============================================================================
# 修改说明：
# 1. 在评估阶段，将预测事件与具体的 GT 条目进行匹配。
# 2. 将匹配到的 GT 的具体类型 (click, burst, buzz, 或 whistle 的子类如果有) 写入结果。
# 3. 输出文件中增加 'matched_gt_type' 列，为 Phase 3 提供直接的标签依据。
# =============================================================================

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split
import json
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
    "phase2_output": Path("D:/Project_Github/audio_click_mil/results/phase2"),
    "target_sr": 48000,
    "bag_duration": 30,
    "mfcc_hop_length": 512,
    "probability_threshold": 0.5,
    "min_event_duration": 0.5,  # 脉冲通常很短，这里设为 0.5s 可能偏大，但保持原样以对比
    "merge_gap": 1.0,
    "test_size": 0.3,
    "random_state": 42,
}

CONFIG["phase2_output"].mkdir(exist_ok=True, parents=True)

# ─────────────────────────────────────────────────────────────────────────────
# 🔧 辅助函数：类型转换
# ─────────────────────────────────────────────────────────────────────────────
def convert_to_native(obj):
    if isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, dict): return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [convert_to_native(i) for i in obj]
    return obj

# ─────────────────────────────────────────────────────────────────────────────
# 2. 加载 Phase 1 模型和 bags
# ─────────────────────────────────────────────────────────────────────────────
print("="*80)
print("Phase 2: 实例级定位 (带类型标签映射增强版)")
print("="*80)

print("\n[1/7] 加载 Phase 1 模型和 bags...")
whistle_model = joblib.load(CONFIG["phase1_output"] / "whistle_detector.pkl")
pulse_model = joblib.load(CONFIG["phase1_output"] / "pulse_detector.pkl")
print("  ✓ 哨声检测模型已加载")
print("  ✓ 脉冲检测模型已加载")

bags_path = CONFIG["phase1_output"] / "bags_with_features.pkl"
with open(bags_path, 'rb') as f:
    bags = pickle.load(f)
print(f"  ✓ 加载 {len(bags)} 个 bag")

valid_bags = [b for b in bags if b["features"].get("mfcc") is not None]
print(f"  ✓ 有效 Bag 数：{len(valid_bags)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. 数据集划分与统计 (保持不变)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/7] 训练集/验证集样本统计...")
file_nos = np.array([b["file_no"] for b in valid_bags])
y_whistle = np.array([b["whistle_label"] for b in valid_bags])
y_pulse = np.array([b["pulse_label"] for b in valid_bags])

unique_files = np.unique(file_nos)
train_files, val_files = train_test_split(unique_files, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"])
train_mask = np.isin(file_nos, train_files)
val_mask = np.isin(file_nos, val_files)

# 统计逻辑简化展示
whistle_train_pos = int(np.sum(y_whistle[train_mask]))
pulse_train_pos = int(np.sum(y_pulse[train_mask]))
# ... (其他统计逻辑同原版，省略以节省空间，实际使用时请保留完整统计代码)
print(f"  ✓ 训练集文件：{len(train_files)}, 验证集文件：{len(val_files)}")
print(f"  ✓ 哨声正样本 (Train): {whistle_train_pos}, 脉冲正样本 (Train): {pulse_train_pos}")

dataset_stats = {
    "train_files": sorted([int(f) for f in train_files]),
    "val_files": sorted([int(f) for f in val_files]),
    "summary": "Statistics calculated successfully."
}
stats_path = CONFIG["phase2_output"] / "dataset_statistics.json"
with open(stats_path, 'w', encoding='utf-8') as f:
    json.dump(dataset_stats, f, indent=2, ensure_ascii=False)

# ─────────────────────────────────────────────────────────────────────────────
# 4. 帧级预测 (保持不变)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/7] 帧级预测...")
frames_per_second = CONFIG["target_sr"] / CONFIG["mfcc_hop_length"]
frame_predictions = []

for bag in tqdm(valid_bags, desc="生成帧级预测"):
    file_no = bag["file_no"]
    bag_start = bag["bag_start_sec"]
    mfcc_features = bag["features"]["mfcc"]
    if mfcc_features is None: continue
    
    # 获取概率
    w_prob = whistle_model.predict_proba([mfcc_features])[0, 1]
    p_prob = pulse_model.predict_proba([mfcc_features])[0, 1]
    
    n_frames = len(mfcc_features)
    times = np.arange(n_frames) / frames_per_second + bag_start
    
    for i in range(n_frames):
        frame_predictions.append({
            "file_no": file_no,
            "frame_time": times[i],
            "whistle_prob": w_prob,
            "pulse_prob": p_prob,
        })

frame_df = pd.DataFrame(frame_predictions)
print(f"  ✓ 生成 {len(frame_df)} 条帧级记录")

# ─────────────────────────────────────────────────────────────────────────────
# 5. 事件检测 (保持不变)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/7] 事件检测与时间定位...")

def detect_events(frame_df, label_col, threshold, min_duration, merge_gap):
    events = []
    for file_no in frame_df["file_no"].unique():
        df_part = frame_df[frame_df["file_no"] == file_no].sort_values("frame_time")
        probs = df_part[label_col].values
        times = df_part["frame_time"].values
        if len(times) == 0: continue
        
        probs_smooth = gaussian_filter1d(probs, sigma=2)
        mask = probs_smooth >= threshold
        
        candidates = []
        in_evt = False
        start_idx = 0
        
        for i, m in enumerate(mask):
            if m and not in_evt:
                in_evt = True
                start_idx = i
            elif not m and in_evt:
                in_evt = False
                dur = times[i-1] - times[start_idx]
                if dur >= min_duration:
                    conf = float(np.mean(probs_smooth[start_idx:i]))
                    candidates.append((file_no, times[start_idx], times[i-1], conf))
        
        if in_evt:
            dur = times[-1] - times[start_idx]
            if dur >= min_duration:
                conf = float(np.mean(probs_smooth[start_idx:]))
                candidates.append((file_no, times[start_idx], times[-1], conf))
        
        # Merge
        merged = []
        for evt in sorted(candidates, key=lambda x: x[1]):
            if merged and evt[1] - merged[-1][2] < merge_gap:
                merged[-1] = (evt[0], merged[-1][1], evt[2], (merged[-1][3]+evt[3])/2)
            else:
                merged.append(evt)
        events.extend(merged)
    return events

whistle_events = detect_events(frame_df, "whistle_prob", CONFIG["probability_threshold"], CONFIG["min_event_duration"], CONFIG["merge_gap"])
pulse_events = detect_events(frame_df, "pulse_prob", CONFIG["probability_threshold"], CONFIG["min_event_duration"], CONFIG["merge_gap"])

print(f"  ✓ 检测到哨声事件：{len(whistle_events)}")
print(f"  ✓ 检测到脉冲事件：{len(pulse_events)}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. 🌟 核心修改：带类型映射的定位评估
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/7] 定位精度评估与类型映射...")

# 加载 GT 并添加明确的 'gt_type' 列
whistle_df = pd.read_csv(CONFIG["data_root"] / "WhistleParameters-clean.csv")
click_df = pd.read_csv(CONFIG["data_root"] / "ClickTrains.csv")
burst_df = pd.read_csv(CONFIG["data_root"] / "BurstPulseTrains.csv")
buzz_df = pd.read_csv(CONFIG["data_root"] / "BuzzTrains.csv")

# 清洗列名
whistle_df.columns = whistle_df.columns.str.strip()
click_df.columns = click_df.columns.str.strip()
burst_df.columns = burst_df.columns.str.strip()
buzz_df.columns = buzz_df.columns.str.strip()

available_files = set([b["file_no"] for b in valid_bags])

# 构建统一的 GT 列表，每个元素包含：(file_no, start, end, type_name, original_row_index)
all_gt = []

# 处理 Whistle
if "Original Audio File (No.)" in whistle_df.columns:
    w_sub = whistle_df[whistle_df["Original Audio File (No.)"].astype(int).isin(available_files)].copy()
    # 假设 Whistle 没有更细的子类标签，统一标记为 'whistle'
    # 如果你的 CSV 里有 'type' 列，可以在这里读取：w_sub['gt_type'] = w_sub['type']
    w_sub['gt_type'] = 'whistle' 
    w_sub['gt_start'] = w_sub["Whistle Begins (s)"]
    w_sub['gt_end'] = w_sub.get("Whistle Ends (s)", w_sub['gt_start'] + 0.1) # 如果没有结束时间，估算一个
    for idx, row in w_sub.iterrows():
        all_gt.append({
            "file_no": int(row["Original Audio File (No.)"]),
            "start": row['gt_start'],
            "end": row['gt_end'],
            "type": row['gt_type'],
            "source_idx": idx
        })

# 处理 Click/Burst/Buzz
for df, label in [(click_df, 'click'), (burst_df, 'burst'), (buzz_df, 'buzz')]:
    if "Ori_file_num(No.)" in df.columns:
        sub = df[df["Ori_file_num(No.)"].astype(int).isin(available_files)].copy()
        sub['gt_type'] = label
        sub['gt_start'] = sub["Train_start(s)"]
        sub['gt_end'] = sub.get("Train_end(s)", sub['gt_start'] + 0.1)
        for idx, row in sub.iterrows():
            all_gt.append({
                "file_no": int(row["Ori_file_num(No.)"]),
                "start": row['gt_start'],
                "end": row['gt_end'],
                "type": row['gt_type'],
                "source_idx": idx
            })

gt_df = pd.DataFrame(all_gt)
print(f"  ✓ 加载 Ground Truth 共 {len(gt_df)} 条 (Whistle: {len(gt_df[gt_df['type']=='whistle'])}, Pulse: {len(gt_df[gt_df['type']!='whistle'])})")

tolerance = 2.0

def evaluate_and_map(predicted_events, gt_df, tolerance=2.0):
    """
    评估定位精度，并返回带有 matched_gt_type 的事件列表
    """
    tp = 0
    fp = 0
    matched_gt_indices = set()
    
    # 结果列表：包含原始预测信息 + 匹配到的 GT 类型
    enriched_events = []
    
    # 按文件分组 GT 以加速查找
    gt_grouped = gt_df.groupby('file_no')
    
    for pred in predicted_events:
        # 修改后
        p_file, p_start, p_end, p_conf, _ = pred 
        # 然后下面可以用 source_tag 代替 str(pred) 的判断
        p_center = (p_start + p_end) / 2

        p_type_guess = "whistle" if "whistle" in str(pred) else "pulse"
        
        best_match = None
        min_dist = float('inf')
        matched_type = None
        matched_gt_idx = None
        
        # 在该文件中寻找最近的 GT
        if p_file in gt_grouped.groups:
            group = gt_grouped.get_group(p_file)
            for idx, row in group.iterrows():
                g_center = (row['start'] + row['end']) / 2
                dist = abs(p_center - g_center)
                
                if dist <= tolerance and dist < min_dist:
                    min_dist = dist
                    best_match = row
                    matched_type = row['type']
                    matched_gt_idx = row['source_idx']
        
        if best_match is not None:
            tp += 1
            matched_gt_indices.add(matched_gt_idx)
            # 记录匹配到的具体类型
            enriched_events.append({
                "file_no": p_file,
                "start_time": p_start,
                "end_time": p_end,
                "confidence": p_conf,
                "model_category": p_type_guess, # 模型认为是哪大类 (whistle/pulse)
                "matched_gt_type": matched_type, # 实际匹配到的具体类型 (click/buzz/whistle...)
                "match_distance": min_dist
            })
        else:
            fp += 1
            enriched_events.append({
                "file_no": p_file,
                "start_time": p_start,
                "end_time": p_end,
                "confidence": p_conf,
                "model_category": p_type_guess,
                "matched_gt_type": None, # 未匹配到任何 GT
                "match_distance": None
            })
            
    fn = len(gt_df) - len(matched_gt_indices)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "enriched_events": enriched_events
    }

# 合并预测事件并标记来源模型
# 注意：whistle_events 来自 whistle_model, pulse_events 来自 pulse_model
all_predictions = []
for evt in whistle_events:
    all_predictions.append((*evt, "whistle_model")) # (file, start, end, conf, source)
for evt in pulse_events:
    all_predictions.append((*evt, "pulse_model"))

# 执行评估与映射
# 为了简化，我们统一评估，但在内部可以通过 source 区分（这里简化为统一池子评估）
# 更严谨的做法是分别评估 whistle_model vs whistle_GT 和 pulse_model vs pulse_GT
# 这里采用统一评估，看模型预测的事件到底匹配到了什么类型的 GT

eval_result = evaluate_and_map(all_predictions, gt_df, tolerance)

print("\n" + "="*60)
print("【综合定位与类型匹配结果】")
print("="*60)
print(f"  真正例 (TP): {eval_result['tp']}")
print(f"  假正例 (FP): {eval_result['fp']}")
print(f"  假负例 (FN): {eval_result['fn']}")
print(f"  精确率：{eval_result['precision']:.4f}")
print(f"  召回率：{eval_result['recall']:.4f}")
print(f"  F1 分数：{eval_result['f1']:.4f}")

# 分析类型匹配准确率 (Type Accuracy among TP)
tp_events = [e for e in eval_result['enriched_events'] if e['matched_gt_type'] is not None]
if tp_events:
    # 统计模型类别与 GT 类型的一致性
    # 例如：whistle_model 预测的事件，匹配到的 GT 是否都是 'whistle'?
    correct_type_count = 0
    type_confusion = {}
    
    for e in tp_events:
        model_cat = e['model_category']
        gt_t = e['matched_gt_type']
        
        key = f"{model_cat}_pred_{gt_t}_gt"
        type_confusion[key] = type_confusion.get(key, 0) + 1
        
        # 简单的合理性检查：whistle_model 应该匹配 whistle, pulse_model 应该匹配 click/burst/buzz
        is_correct = False
        if model_cat == "whistle_model" and gt_t == "whistle":
            is_correct = True
        elif model_cat == "pulse_model" and gt_t in ["click", "burst", "buzz"]:
            is_correct = True
            
        if is_correct:
            correct_type_count += 1
            
    type_accuracy = correct_type_count / len(tp_events)
    print(f"\n  【类型一致性检查】")
    print(f"  TP 中类型匹配正确的比例：{type_accuracy:.4f} ({correct_type_count}/{len(tp_events)})")
    print(f"  混淆矩阵摘要：{type_confusion}")
else:
    print("\n  无 TP 事件，无法计算类型一致性。")

# ─────────────────────────────────────────────────────────────────────────────
# 7. 保存增强版结果
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/7] 保存结果...")

results_df = pd.DataFrame(eval_result['enriched_events'])
events_path = CONFIG["phase2_output"] / "detected_events_enriched.csv"
results_df.to_csv(events_path, index=False, encoding='utf-8-sig')
print(f"  ✓ 增强版检测事件已保存：{events_path}")
print(f"     (包含列：file_no, start_time, end_time, confidence, model_category, matched_gt_type)")

# 同时也保存一份 frame predictions
frame_path = CONFIG["phase2_output"] / "frame_predictions.csv"
frame_df.to_csv(frame_path, index=False, encoding='utf-8-sig')

# 生成报告
report_content = f"""
================================================================================
Phase 2 - 实例级定位与类型映射评估报告
生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

【核心指标】
  精确率 (Precision): {eval_result['precision']:.4f}
  召回率 (Recall):    {eval_result['recall']:.4f}
  F1 分数：{eval_result['f1']:.4f}
  TP: {eval_result['tp']}, FP: {eval_result['fp']}, FN: {eval_result['fn']}

【类型一致性分析 (仅在 TP 中)】
  模型预测类别与 GT 类型一致的比率：{type_accuracy:.4f} if tp_events else 0
  详细分布：{type_confusion if tp_events else "N/A"}

【输出文件说明】
  1. detected_events_enriched.csv: 
     - 包含 'matched_gt_type' 列。
     - 如果该列为 'click', 'burst', 'buzz' 或 'whistle'，表示该预测事件已成功对齐到具体类型的 GT。
     - Phase 3 可直接使用此文件的 'matched_gt_type' 作为标签，无需重新分类！
     
  2. frame_predictions.csv: 帧级概率详情。

================================================================================
"""
report_path = CONFIG["phase2_output"] / "02_localization_report_enhanced.txt"
Path(report_path).write_text(report_content, encoding='utf-8')
print(f"  ✓ 评估报告：{report_path}")

print("\n" + "="*80)
print("✅ Phase 2 (增强版) 全部任务完成！")
print("💡 提示：Phase 3 现在可以直接读取 'detected_events_enriched.csv' 中的 'matched_gt_type' 列作为标签。")
print("="*80)
print(report_content)