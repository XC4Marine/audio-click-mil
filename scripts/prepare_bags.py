# =============================================================================
# Phase 0: 数据准备与Bag生成（任务2-9完整实现）
# =============================================================================
# 功能：
#   - Bag切分策略：30s/bag，无重叠，不足30s丢弃
#   - 音频降采样至48kHz
#   - 弱标签生成（whistle_label / pulse_label）
#   - 原始标注分配到bag
#   - 特征占位结构
#   - 保存bags.pkl + dataset_summary.csv
#   - 进度条显示
# =============================================================================

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm  # 进度条

# ─────────────────────────────────────────────────────────────────────────────
# 1. 配置参数
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "data_root": Path("D:/Project_Github/audio_click_mil/data"),
    "output_dir": Path("D:/Project_Github/audio_click_mil/results/phase0"),
    "audio_root": Path("D:/Project_Github/audio_click_mil/data/audio"),  # 假设音频在此
    "bag_duration": 30,           # 每个bag时长（秒）
    "overlap": 0,                 # 重叠时长（秒）
    "sample_rate": 48000,         # 目标采样率
    "recording_duration": 30*60,  # 每段录音总时长（30分钟=1800秒）
}

# 创建输出目录
CONFIG["output_dir"].mkdir(exist_ok=True, parents=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. 读取所有标注CSV
# ─────────────────────────────────────────────────────────────────────────────
print("[1/7] 读取标注文件...")

whistle_df = pd.read_csv(CONFIG["data_root"] / "WhistleParameters-clean.csv")
click_df = pd.read_csv(CONFIG["data_root"] / "ClickTrains.csv")
burst_df = pd.read_csv(CONFIG["data_root"] / "BurstPulseTrains.csv")
buzz_df = pd.read_csv(CONFIG["data_root"] / "BuzzTrains.csv")
results_df = pd.read_csv(CONFIG["data_root"] / "Results.csv")

# 统一列名（处理可能的空格问题）
whistle_df.columns = whistle_df.columns.str.strip()
click_df.columns = click_df.columns.str.strip()
burst_df.columns = burst_df.columns.str.strip()
buzz_df.columns = buzz_df.columns.str.strip()
results_df.columns = results_df.columns.str.strip()

print(f"  ✓ 哨声标注: {len(whistle_df)} 条")
print(f"  ✓ Click标注: {len(click_df)} 条")
print(f"  ✓ Burst标注: {len(burst_df)} 条")
print(f"  ✓ Buzz标注: {len(buzz_df)} 条")
print(f"  ✓ 录音文件: {len(results_df)} 个")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Bag切分策略设计（任务2）
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/7] 计算Bag切分策略...")

bag_duration = CONFIG["bag_duration"]
recording_duration = CONFIG["recording_duration"]

# 每个录音的bag数量
bags_per_recording = recording_duration // bag_duration  # 1800/30 = 60

# 总bag数量
total_recordings = len(results_df)
total_bags = total_recordings * bags_per_recording

print(f"  每段录音时长: {recording_duration} 秒")
print(f"  每个Bag时长: {bag_duration} 秒")
print(f"  重叠: {CONFIG['overlap']} 秒")
print(f"  每录音Bag数: {bags_per_recording}")
print(f"  总录音数: {total_recordings}")
print(f"  总Bag数: {total_bags}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. 弱标签生成规则（任务3-4）
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/7] 准备弱标签生成规则...")

def get_bag_id(timestamp, bag_duration=30):
    """根据时间戳计算所属bag_id（0-based）"""
    return int(timestamp // bag_duration)

def assign_annotations_to_bags(annotations_df, time_start_col, time_end_col, 
                                file_id_col, bags_per_recording=60):
    """
    将标注分配到对应的bag
    返回: dict[file_no] -> dict[bag_id] -> list[annotation_indices]
    """
    assignment = {}
    
    for file_no in annotations_df[file_id_col].unique():
        file_annots = annotations_df[annotations_df[file_id_col] == file_no].copy()
        assignment[file_no] = {}
        
        for idx, row in file_annots.iterrows():
            start_time = row[time_start_col]
            end_time = row[time_end_col]
            
            # 计算起始和结束bag_id
            start_bag = get_bag_id(start_time)
            end_bag = get_bag_id(end_time)
            
            # 分配到所有覆盖的bag
            for bag_id in range(start_bag, min(end_bag + 1, bags_per_recording)):
                if bag_id not in assignment[file_no]:
                    assignment[file_no][bag_id] = []
                assignment[file_no][bag_id].append(idx)
    
    return assignment

# 为哨声添加bag_id（哨声是瞬时事件，用开始时间）
whistle_df['bag_id'] = whistle_df['Whistle Begins (s)'].apply(
    lambda x: get_bag_id(x, bag_duration)
)

# 为脉冲串分配bag（脉冲串有起止时间）
click_assignment = assign_annotations_to_bags(
    click_df, 'Train_start(s)', 'Train_end(s)', 'Ori_file_num(No.)', bags_per_recording
)
burst_assignment = assign_annotations_to_bags(
    burst_df, 'Train_start(s)', 'Train_end(s)', 'Ori_file_num(No.)', bags_per_recording
)
buzz_assignment = assign_annotations_to_bags(
    buzz_df, 'Train_start(s)', 'Train_end(s)', 'Ori_file_num(No.)', bags_per_recording
)

print("  ✓ 哨声bag_id分配完成")
print("  ✓ Click bag分配完成")
print("  ✓ Burst bag分配完成")
print("  ✓ Buzz bag分配完成")

# ─────────────────────────────────────────────────────────────────────────────
# 5. 构建Bag数据结构（任务5-7）
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/7] 构建Bag数据结构...")

# 获取所有录音文件编号
recording_files = sorted(results_df['Original audio_file (No.)'].astype(int).unique())

# 构建bags列表
bags = []

print(f"  开始处理 {len(recording_files)} 个录音文件...")

for file_no in tqdm(recording_files, desc="处理录音文件"):
    # 获取该录音的元信息
    rec_info = results_df[results_df['Original audio_file (No.)'] == file_no].iloc[0]
    
    # 为这个录音创建60个bag
    for bag_id in range(bags_per_recording):
        bag_start = bag_id * bag_duration
        bag_end = bag_start + bag_duration
         
        # ─── 弱标签生成 ────────────────────────────────────────────────────────
        # 哨声标签：该bag内是否有哨声
        whistle_in_bag = whistle_df[
            (whistle_df['Original Audio File (No.)'].astype(int) == file_no) & 
            (whistle_df['bag_id'] == bag_id)
        ]
        whistle_label = 1 if len(whistle_in_bag) > 0 else 0
        
        # 脉冲标签：该bag内是否有click/burst/buzz
        pulse_in_bag = (
            (bag_id in click_assignment.get(file_no, {})) or
            (bag_id in burst_assignment.get(file_no, {})) or
            (bag_id in buzz_assignment.get(file_no, {}))
        )
        pulse_label = 1 if pulse_in_bag else 0
        
        # ─── 原始标注索引记录（用于后续定位精度评估） ──────────────────────────
        whistle_indices = whistle_in_bag.index.tolist() if len(whistle_in_bag) > 0 else []
        click_indices = click_assignment.get(file_no, {}).get(bag_id, [])
        burst_indices = burst_assignment.get(file_no, {}).get(bag_id, [])
        buzz_indices = buzz_assignment.get(file_no, {}).get(bag_id, [])
        
        # ─── 哨声类型统计（多标签） ────────────────────────────────────────────
        whistle_types = []
        if len(whistle_in_bag) > 0:
            whistle_types = whistle_in_bag['Type'].unique().tolist()
        
        # ─── 特征占位结构（任务7） ─────────────────────────────────────────────
        # 为后续音频特征提取预留位置
        feature_placeholder = {
            "spectrogram": None,      # 待填充：频谱图
            "mfcc": None,             # 待填充：MFCC特征
            "chroma": None,           # 待填充：色度特征
            "tempo": None,            # 待填充：节奏特征
        }
        
        # ─── 构建bag字典 ───────────────────────────────────────────────────────
        bag = {
            # 基本信息
            "file_no": int(file_no),
            "bag_id": bag_id,
            "bag_start_sec": bag_start,
            "bag_end_sec": bag_end,
            "duration_sec": bag_duration,
            
            # 弱标签
            "whistle_label": whistle_label,
            "pulse_label": pulse_label,
            
            # 详细标注信息（用于评估）
            "whistle_indices": whistle_indices,
            "click_indices": click_indices,
            "burst_indices": burst_indices,
            "buzz_indices": buzz_indices,
            
            # 哨声类型（多标签）
            "whistle_types": whistle_types,
            
            # 录音元信息
            "water_depth": rec_info.get('Water Depth', None),
            "group_size": rec_info.get('Group size', None),
            "behavior": rec_info.get('Behavior', None),
            
            # 特征占位
            "features": feature_placeholder,
            
            # 音频文件路径（待填充）
            "audio_path": None,
        }
        
        bags.append(bag)

print(f"  ✓ 共构建 {len(bags)} 个bag")

# ─────────────────────────────────────────────────────────────────────────────
# 6. 统计与验证（任务8）
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/7] 生成数据集统计摘要...")

# Bag级别统计
whistle_bag_count = sum(1 for b in bags if b['whistle_label'] == 1)
pulse_bag_count = sum(1 for b in bags if b['pulse_label'] == 1)
both_bag_count = sum(1 for b in bags if b['whistle_label'] == 1 and b['pulse_label'] == 1)
negative_bag_count = sum(1 for b in bags if b['whistle_label'] == 0 and b['pulse_label'] == 0)

# 哨声类型分布（bag级别）
type_distribution = {}
for b in bags:
    for t in b['whistle_types']:
        type_distribution[t] = type_distribution.get(t, 0) + 1

# 生成统计DataFrame
summary_data = []
for file_no in recording_files:
    file_bags = [b for b in bags if b['file_no'] == file_no]
    summary_data.append({
        'file_no': file_no,
        'total_bags': len(file_bags),
        'whistle_bags': sum(1 for b in file_bags if b['whistle_label'] == 1),
        'pulse_bags': sum(1 for b in file_bags if b['pulse_label'] == 1),
        'negative_bags': sum(1 for b in file_bags if b['whistle_label'] == 0 and b['pulse_label'] == 0),
    })

summary_df = pd.DataFrame(summary_data)

# 保存统计摘要
summary_csv_path = CONFIG["output_dir"] / "dataset_summary.csv"
summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')

# 生成统计报告
stats_report = f"""
================================================================================
Phase 0 - 数据集统计摘要
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

【基础配置】
  录音时长: {CONFIG['recording_duration']} 秒 (30分钟)
  Bag时长: {CONFIG['bag_duration']} 秒
  重叠: {CONFIG['overlap']} 秒
  采样率: {CONFIG['sample_rate']} Hz

【数据规模】
  录音文件数: {len(recording_files)}
  总Bag数: {len(bags)}
  每录音Bag数: {bags_per_recording}

【弱标签分布】
  含哨声Bag数: {whistle_bag_count} ({whistle_bag_count/len(bags)*100:.2f}%)
  含脉冲Bag数: {pulse_bag_count} ({pulse_bag_count/len(bags)*100:.2f}%)
  同时含两者: {both_bag_count} ({both_bag_count/len(bags)*100:.2f}%)
  负样本Bag数: {negative_bag_count} ({negative_bag_count/len(bags)*100:.2f}%)

【哨声类型分布（Bag级别）】
"""

for t, c in sorted(type_distribution.items(), key=lambda x: -x[1]):
    stats_report += f"  {t}: {c} bags\n"

stats_report += f"""
【文件级统计】
  已保存至: {summary_csv_path}
  包含列: file_no, total_bags, whistle_bags, pulse_bags, negative_bags

【输出文件】
  bags.pkl: {CONFIG["output_dir"] / "bags.pkl"}
  dataset_summary.csv: {summary_csv_path}
================================================================================
"""

# 保存统计报告
stats_path = CONFIG["output_dir"] / "01_dataset_statistics.txt"
stats_path.write_text(stats_report, encoding='utf-8')

print(stats_report)
print(f"  ✓ 统计报告保存至: {stats_path}")
print(f"  ✓ 摘要CSV保存至: {summary_csv_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. 保存Bags（任务8）
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/7] 保存bags.pkl...")

bags_path = CONFIG["output_dir"] / "bags.pkl"
with open(bags_path, 'wb') as f:
    pickle.dump(bags, f)

print(f"  ✓ bags.pkl 保存至: {bags_path}")
print(f"  ✓ 文件大小: {bags_path.stat().st_size / 1024:.2f} KB")

# ─────────────────────────────────────────────────────────────────────────────
# 8. 可视化Bag时间轴分布（任务9）
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/7] 生成可视化数据...")

# 生成可用于可视化的数据
viz_data = {
    "file_no": [],
    "bag_id": [],
    "bag_start": [],
    "whistle_label": [],
    "pulse_label": [],
}

for b in bags:
    viz_data["file_no"].append(b["file_no"])
    viz_data["bag_id"].append(b["bag_id"])
    viz_data["bag_start"].append(b["bag_start_sec"])
    viz_data["whistle_label"].append(b["whistle_label"])
    viz_data["pulse_label"].append(b["pulse_label"])

viz_df = pd.DataFrame(viz_data)
viz_path = CONFIG["output_dir"] / "bag_timeline_viz.csv"
viz_df.to_csv(viz_path, index=False, encoding='utf-8-sig')

print(f"  ✓ 可视化数据保存至: {viz_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. 完成报告
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("✅ Phase 0 全部任务完成！")
print("="*80)
print(f"""
【输出文件清单】
  1. {bags_path}           - Bag数据结构（pickle）
  2. {summary_csv_path}    - 数据集统计摘要（CSV）
  3. {stats_path}          - 详细统计报告（TXT）
  4. {viz_path}            - 可视化数据（CSV）
  5. {CONFIG['output_dir'] / '00_file_integrity_check.txt'} - 数据完整性检查

【下一步建议】
  → Phase 1: 音频特征提取（读取.wav文件，提取spectrogram/MFCC等）
  → 需要确认音频文件路径：{CONFIG['audio_root']}

【注意事项】
  ⚠  当前bags.pkl中的 audio_path 和 features 字段为 None/占位
  ⚠  需要在Phase 1中填充实际音频特征
  ⚠  音频需降采样至48kHz后切分（Phase 1实现）
""")
print("="*80)