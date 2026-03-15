# =============================================================================
# Phase 2: 实例级定位 (Instance-level Localization) - 修复版
# =============================================================================

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
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
    "min_event_duration": 0.5,
    "merge_gap": 1.0,
}

CONFIG["phase2_output"].mkdir(exist_ok=True, parents=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. 加载Phase 1模型和bags
# ─────────────────────────────────────────────────────────────────────────────
print("="*80)
print("Phase 2: 实例级定位 (修复版)")
print("="*80)

print("\n[1/5] 加载Phase 1模型和bags...")

whistle_model = joblib.load(CONFIG["phase1_output"] / "whistle_detector.pkl")
pulse_model = joblib.load(CONFIG["phase1_output"] / "pulse_detector.pkl")
print("  ✓ 哨声检测模型已加载")
print("  ✓ 脉冲检测模型已加载")

bags_path = CONFIG["phase1_output"] / "bags_with_features.pkl"
with open(bags_path, 'rb') as f:
    bags = pickle.load(f)
print(f"  ✓ 加载 {len(bags)} 个bag")

valid_bags = [b for b in bags if b["features"].get("mfcc") is not None]
print(f"  ✓ 有效Bag数: {len(valid_bags)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. 帧级预测
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/5] 帧级预测（从Bag到帧）...")

frames_per_second = CONFIG["target_sr"] / CONFIG["mfcc_hop_length"]
print(f"  帧率: {frames_per_second:.1f} frames/sec")

frame_predictions = []

for bag in tqdm(valid_bags, desc="生成帧级预测"):
    file_no = bag["file_no"]
    bag_id = bag["bag_id"]
    bag_start = bag["bag_start_sec"]
    mfcc_features = bag["features"]["mfcc"]
    
    if mfcc_features is None:
        continue
    
    whistle_prob = whistle_model.predict_proba([mfcc_features])[0, 1]
    pulse_prob = pulse_model.predict_proba([mfcc_features])[0, 1]
    
    n_frames = len(mfcc_features)
    frame_time = np.arange(n_frames) / frames_per_second + bag_start
    
    for i in range(n_frames):
        frame_predictions.append({
            "file_no": file_no,
            "bag_id": bag_id,
            "frame_idx": i,
            "frame_time": frame_time[i],
            "whistle_prob": whistle_prob,
            "pulse_prob": pulse_prob,
        })

frame_df = pd.DataFrame(frame_predictions)
print(f"  ✓ 生成 {len(frame_df)} 条帧级预测")

# ─────────────────────────────────────────────────────────────────────────────
# 4. 事件检测与时间定位 (修复IndexError)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] 事件检测与时间定位...")

def detect_events(frame_df, label_col, threshold, min_duration, merge_gap, 
                  frames_per_second):
    """
    从帧级概率检测事件 (已修复IndexError)
    """
    events = []
    
    for file_no in frame_df["file_no"].unique():
        file_frames = frame_df[frame_df["file_no"] == file_no].sort_values("frame_time")
        probs = file_frames[label_col].values
        times = file_frames["frame_time"].values
        
        if len(times) == 0:
            continue
        
        # 平滑概率曲线
        probs_smooth = gaussian_filter1d(probs, sigma=2)
        
        # 阈值检测
        above_threshold = probs_smooth >= threshold
        
        event_candidates = []
        in_event = False
        start_idx = 0
        
        for i, is_above in enumerate(above_threshold):
            if is_above and not in_event:
                in_event = True
                start_idx = i
            elif not is_above and in_event:
                in_event = False
                end_idx = i  # 事件结束于当前帧之前
                
                start_time = times[start_idx]
                # 🔧 修复：使用 end_idx - 1 确保不越界
                end_time = times[min(end_idx - 1, len(times) - 1)]
                duration = end_time - start_time
                
                if duration >= min_duration:
                    confidence = float(np.mean(probs_smooth[start_idx:end_idx]))
                    event_candidates.append((file_no, start_time, end_time, confidence))
        
        # 🔧 修复：处理最后一个事件（如果持续到末尾）
        if in_event:
            start_time = times[start_idx]
            # 🔧 修复：使用 len(times) - 1 作为最后一个有效索引
            end_time = times[len(times) - 1]
            duration = end_time - start_time
            if duration >= min_duration:
                confidence = float(np.mean(probs_smooth[start_idx:]))
                event_candidates.append((file_no, start_time, end_time, confidence))
        
        # 合并相邻事件
        merged_events = []
        for evt in sorted(event_candidates, key=lambda x: x[1]):
            if merged_events and evt[1] - merged_events[-1][2] < merge_gap:
                merged_events[-1] = (
                    evt[0],
                    merged_events[-1][1],
                    evt[2],
                    (merged_events[-1][3] + evt[3]) / 2
                )
            else:
                merged_events.append(evt)
        
        events.extend(merged_events)
    
    return events

# 检测哨声事件
whistle_events = detect_events(
    frame_df, "whistle_prob", 
    CONFIG["probability_threshold"],
    CONFIG["min_event_duration"],
    CONFIG["merge_gap"],
    frames_per_second
)

# 检测脉冲事件
pulse_events = detect_events(
    frame_df, "pulse_prob",
    CONFIG["probability_threshold"],
    CONFIG["min_event_duration"],
    CONFIG["merge_gap"],
    frames_per_second
)

print(f"  ✓ 检测哨声事件: {len(whistle_events)} 个")
print(f"  ✓ 检测脉冲事件: {len(pulse_events)} 个")

# ─────────────────────────────────────────────────────────────────────────────
# 5. 定位精度评估
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] 定位精度评估...")

whistle_df = pd.read_csv(CONFIG["data_root"] / "WhistleParameters-clean.csv")
click_df = pd.read_csv(CONFIG["data_root"] / "ClickTrains.csv")
burst_df = pd.read_csv(CONFIG["data_root"] / "BurstPulseTrains.csv")
buzz_df = pd.read_csv(CONFIG["data_root"] / "BuzzTrains.csv")

whistle_df.columns = whistle_df.columns.str.strip()
click_df.columns = click_df.columns.str.strip()
burst_df.columns = burst_df.columns.str.strip()
buzz_df.columns = buzz_df.columns.str.strip()

available_files = [1, 2, 3, 4, 5]
tolerance = 2.0

# 哨声评估
whistle_gt = whistle_df[
    whistle_df["Original Audio File (No.)"].astype(int).isin(available_files)
].copy()

print(f"\n  哨声标注（GT）: {len(whistle_gt)} 条")
print(f"  哨声预测: {len(whistle_events)} 条")

def evaluate_localization(predicted_events, ground_truth, tolerance=2.0):
    tp = 0
    fp = 0
    matched_gt = set()
    
    for i, pred in enumerate(predicted_events):
        file_no, pred_start, pred_end, conf = pred
        pred_center = (pred_start + pred_end) / 2
        
        matched = False
        for j, gt_row in ground_truth.iterrows():
            if j in matched_gt:
                continue
            
            gt_file = int(gt_row.get("Original Audio File (No.)", -1))
            gt_time = gt_row.get("Whistle Begins (s)", -1)
            
            if gt_file == file_no and abs(pred_center - gt_time) <= tolerance:
                tp += 1
                matched_gt.add(j)
                matched = True
                break
        
        if not matched:
            fp += 1
    
    fn = len(ground_truth) - len(matched_gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "matched_gt": len(matched_gt),
        "total_gt": len(ground_truth),
        "total_pred": len(predicted_events)
    }

whistle_eval = evaluate_localization(whistle_events, whistle_gt, tolerance)

print("\n" + "="*60)
print("【哨声定位精度】")
print("="*60)
print(f"  真正例 (TP): {whistle_eval['tp']}")
print(f"  假正例 (FP): {whistle_eval['fp']}")
print(f"  假负例 (FN): {whistle_eval['fn']}")
print(f"  精确率: {whistle_eval['precision']:.4f}")
print(f"  召回率: {whistle_eval['recall']:.4f}")
print(f"  F1分数: {whistle_eval['f1']:.4f}")

# 脉冲评估
pulse_gt = pd.concat([
    click_df[click_df["Ori_file_num(No.)"].astype(int).isin(available_files)],
    burst_df[burst_df["Ori_file_num(No.)"].astype(int).isin(available_files)],
    buzz_df[buzz_df["Ori_file_num(No.)"].astype(int).isin(available_files)],
])

print(f"\n  脉冲标注（GT）: {len(pulse_gt)} 条")
print(f"  脉冲预测: {len(pulse_events)} 条")

pulse_eval = {
    "tp": 0, "fp": 0, "fn": 0,
    "precision": 0, "recall": 0, "f1": 0,
    "matched_gt": 0, "total_gt": len(pulse_gt),
    "total_pred": len(pulse_events)
}

tp_pulse = 0
matched_gt_pulse = set()

for i, pred in enumerate(pulse_events):
    file_no, pred_start, pred_end, conf = pred
    pred_center = (pred_start + pred_end) / 2
    
    matched = False
    for j, gt_row in pulse_gt.iterrows():
        if j in matched_gt_pulse:
            continue
        
        gt_file = int(gt_row.get("Ori_file_num(No.)", -1))
        gt_start = gt_row.get("Train_start(s)", -1)
        gt_end = gt_row.get("Train_end(s)", gt_start + 1)
        gt_center = (gt_start + gt_end) / 2
        
        if gt_file == file_no and abs(pred_center - gt_center) <= tolerance:
            tp_pulse += 1
            matched_gt_pulse.add(j)
            matched = True
            break
    
    if not matched:
        pulse_eval["fp"] += 1

pulse_eval["tp"] = tp_pulse
pulse_eval["fn"] = len(pulse_gt) - len(matched_gt_pulse)
pulse_eval["matched_gt"] = len(matched_gt_pulse)

if pulse_eval["tp"] + pulse_eval["fp"] > 0:
    pulse_eval["precision"] = pulse_eval["tp"] / (pulse_eval["tp"] + pulse_eval["fp"])
if pulse_eval["tp"] + pulse_eval["fn"] > 0:
    pulse_eval["recall"] = pulse_eval["tp"] / (pulse_eval["tp"] + pulse_eval["fn"])
if pulse_eval["precision"] + pulse_eval["recall"] > 0:
    pulse_eval["f1"] = 2 * pulse_eval["precision"] * pulse_eval["recall"] / (pulse_eval["precision"] + pulse_eval["recall"])

print("\n" + "="*60)
print("【脉冲定位精度】")
print("="*60)
print(f"  真正例 (TP): {pulse_eval['tp']}")
print(f"  假正例 (FP): {pulse_eval['fp']}")
print(f"  假负例 (FN): {pulse_eval['fn']}")
print(f"  精确率: {pulse_eval['precision']:.4f}")
print(f"  召回率: {pulse_eval['recall']:.4f}")
print(f"  F1分数: {pulse_eval['f1']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. 保存结果
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] 保存结果...")

whistle_events_df = pd.DataFrame(whistle_events, 
    columns=["file_no", "start_time", "end_time", "confidence"])
whistle_events_df["type"] = "whistle"

pulse_events_df = pd.DataFrame(pulse_events,
    columns=["file_no", "start_time", "end_time", "confidence"])
pulse_events_df["type"] = "pulse"

all_events_df = pd.concat([whistle_events_df, pulse_events_df], ignore_index=True)
events_path = CONFIG["phase2_output"] / "detected_events.csv"
all_events_df.to_csv(events_path, index=False, encoding='utf-8-sig')
print(f"  ✓ 检测事件: {events_path}")

frame_path = CONFIG["phase2_output"] / "frame_predictions.csv"
frame_df.to_csv(frame_path, index=False, encoding='utf-8-sig')
print(f"  ✓ 帧级预测: {frame_path}")

eval_report = f"""
================================================================================
Phase 2 - 实例级定位评估报告 (修复版)
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

【配置参数】
  概率阈值: {CONFIG['probability_threshold']}
  最小事件时长: {CONFIG['min_event_duration']} 秒
  合并间隙: {CONFIG['merge_gap']} 秒
  定位容差: {tolerance} 秒

【哨声定位结果】
  标注数量: {whistle_eval['total_gt']}
  预测数量: {whistle_eval['total_pred']}
  TP: {whistle_eval['tp']}, FP: {whistle_eval['fp']}, FN: {whistle_eval['fn']}
  精确率: {whistle_eval['precision']:.4f}
  召回率: {whistle_eval['recall']:.4f}
  F1分数: {whistle_eval['f1']:.4f}

【脉冲定位结果】
  标注数量: {pulse_eval['total_gt']}
  预测数量: {pulse_eval['total_pred']}
  TP: {pulse_eval['tp']}, FP: {pulse_eval['fp']}, FN: {pulse_eval['fn']}
  精确率: {pulse_eval['precision']:.4f}
  召回率: {pulse_eval['recall']:.4f}
  F1分数: {pulse_eval['f1']:.4f}

【输出文件】
  检测事件: {events_path}
  帧级预测: {frame_path}

================================================================================
"""

report_path = CONFIG["phase2_output"] / "02_localization_report.txt"
report_path.write_text(eval_report, encoding='utf-8')
print(f"  ✓ 评估报告: {report_path}")

print("\n" + "="*80)
print("✅ Phase 2 全部任务完成！")
print("="*80)
print(eval_report)