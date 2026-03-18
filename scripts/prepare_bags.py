# scripts/prepare_bags.py
"""
【模块：Phase 0 Data Preparation】
主入口脚本：执行数据加载、Bag 构建、特征提取、弱标签生成及报告输出。
对应 skill.md Section 3 (Phase 0)
"""

import os
import sys
import pickle
import yaml
import numpy as np
import pandas as pd
import tqdm

# 添加 src 到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.dataset_utils import (
    load_and_resample_audio, 
    extract_522_features, 
    load_annotations, 
    get_weak_labels,
    create_instances
)

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    print("【当前执行：Phase 0 的任务 1-4 - 开始数据预处理】")
    config = load_config()
    data_root = os.path.join(os.path.dirname(__file__), '..', config['data']['root'])
    audio_dir = os.path.join(data_root, config['data']['original_audio_dir'])
    
    # 1. 加载标注
    print("正在加载标注文件...")
    annotations = load_annotations(data_root)
    
    # 获取所有音频文件列表 (假设命名为 Ori_Recording_01.wav 等)
    audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
    
    if not audio_files:
        print(f"错误：在 {audio_dir} 中未找到 .wav 文件。请确认文件已放入。")
        return

    bags = []
    stats = {
        'total_whistles': 0,
        'high_quality_whistles': 0,
        'total_clicks': 0,
        'total_bursts': 0,
        'total_buzzes': 0,
        'total_bags': 0,
        'positive_whistle_bags': 0,
        'positive_pulse_bags': 0,
        'depths': [],
        'behaviors': []
    }
    
    # 用于统计全局标注数量 (从 CSV 直接读)
    stats['total_whistles'] = len(annotations['whistle'])
    if 'Quality Grade' in annotations['whistle'].columns:
        stats['high_quality_whistles'] = len(annotations['whistle'][annotations['whistle']['Quality Grade'] == 1]) # 假设 1 是高质量
    
    stats['total_clicks'] = len(annotations['click'])
    stats['total_bursts'] = len(annotations['burst'])
    stats['total_buzzes'] = len(annotations['buzz'])
    
    # 从 Results.csv 提取元数据统计
    meta_df = annotations['meta']
    if 'Water Depth (m)' in meta_df.columns:
        stats['depths'] = meta_df['Water Depth (m)'].dropna().tolist()
    if 'Dolphin Behavior' in meta_df.columns:
        stats['behaviors'] = meta_df['Dolphin Behavior'].dropna().tolist()

    print(f"发现 {len(audio_files)} 个音频文件，开始处理...")

    for fname in tqdm.tqdm(audio_files, desc="Processing Audio"):
        file_path = os.path.join(audio_dir, fname)
        # 提取文件 ID (假设文件名格式 Ori_Recording_01.wav -> 1)
        try:
            file_id = int(''.join(filter(str.isdigit, fname)))
        except:
            print(f"无法从 {fname} 解析文件 ID，跳过。")
            continue
            
        # 加载音频
        y, sr = load_and_resample_audio(file_path)
        
        # 切分实例 (10s)
        instances_wave = create_instances(y, sr, duration_sec=config['data']['instance_duration_sec'])
        
        # 构建 Bags (每 60s 一个 Bag，即 6 个实例)
        # 注意：skill.md 定义 Bag 为 1min 非重叠，Instance 为 10s
        inst_per_bag = int(config['data']['bag_hop_sec'] / config['data']['instance_duration_sec']) # 60/10 = 6
        
        num_bags = len(instances_wave) // inst_per_bag
        
        for b_idx in range(num_bags):
            start_inst_idx = b_idx * inst_per_bag
            end_inst_idx = start_inst_idx + inst_per_bag
            
            bag_instances_wave = instances_wave[start_inst_idx:end_inst_idx]
            
            # 计算时间范围
            bag_start_time = start_inst_idx * config['data']['instance_duration_sec']
            bag_end_time = end_inst_idx * config['data']['instance_duration_sec']
            
            # 提取特征
            bag_features = []
            for wave_seg in bag_instances_wave:
                feat = extract_522_features(wave_seg, sr)
                bag_features.append(feat)
            
            bag_features_np = np.stack(bag_features, axis=0) # Shape: [6, 522]
            
            # 生成弱标签
            labels = get_weak_labels(bag_start_time, bag_end_time, annotations, file_id)
            
            # 构建 Bag 字典
            bag_dict = {
                'bag_id': f"{file_id}_{b_idx}",
                'file_id': file_id,
                'instances': bag_features_np,
                'label': [labels['whistle'], labels['pulse']], # [whistle_label, pulse_label]
                'start_time': bag_start_time,
                'end_time': bag_end_time,
                'meta': {} # 后续 Phase 4 填充
            }
            
            bags.append(bag_dict)
            
            if labels['whistle'] == 1:
                stats['positive_whistle_bags'] += 1
            if labels['pulse'] == 1:
                stats['positive_pulse_bags'] += 1
                
        stats['total_bags'] += num_bags

    # 保存 bags.pkl
    output_pkl = os.path.join(data_root, 'bags.pkl')
    with open(output_pkl, 'wb') as f:
        pickle.dump(bags, f)
    print(f"✅ 已保存 {len(bags)} 个 Bags 到 {output_pkl}")

    # 生成报告
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'phase0')
    os.makedirs(results_dir, exist_ok=True)
    
    # Summary CSV
    summary_data = {
        'Metric': ['Total Whistles', 'High Quality Whistles', 'Total Clicks', 'Total Bursts', 'Total Buzzes', 
                   'Total Bags', 'Positive Whistle Bags', 'Positive Pulse Bags', 
                   'Avg Depth (m)', 'Behavior Types'],
        'Value': [
            stats['total_whistles'],
            stats['high_quality_whistles'],
            stats['total_clicks'],
            stats['total_bursts'],
            stats['total_buzzes'],
            stats['total_bags'],
            stats['positive_whistle_bags'],
            stats['positive_pulse_bags'],
            np.mean(stats['depths']) if stats['depths'] else 0,
            len(set(stats['behaviors']))
        ]
    }
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join(results_dir, 'dataset_summary.csv'), index=False)
    
    # Text Report
    report_path = os.path.join(results_dir, 'dataset_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== CWD-MIL Phase 0 Dataset Report ===\n\n")
        f.write(f"Total Audio Files: {len(audio_files)}\n")
        f.write(f"Total Bags Generated: {stats['total_bags']}\n")
        f.write(f"Instance Duration: {config['data']['instance_duration_sec']}s\n")
        f.write(f"Bag Duration: {config['data']['bag_hop_sec']}s\n\n")
        f.write("--- Annotation Stats ---\n")
        f.write(f"Whistles: {stats['total_whistles']} (HQ: {stats['high_quality_whistles']})\n")
        f.write(f"Click Trains: {stats['total_clicks']}\n")
        f.write(f"Burst Trains: {stats['total_bursts']}\n")
        f.write(f"Buzz Trains: {stats['total_buzzes']}\n\n")
        f.write("--- Label Distribution ---\n")
        f.write(f"Bags with Whistles: {stats['positive_whistle_bags']} ({stats['positive_whistle_bags']/max(1, stats['total_bags'])*100:.2f}%)\n")
        f.write(f"Bags with Pulses: {stats['positive_pulse_bags']} ({stats['positive_pulse_bags']/max(1, stats['total_bags'])*100:.2f}%)\n")
        
    print(f"✅ 报告已生成：{report_path}")
    print("【Phase 0 完成】请检查结果后，输入 '下一个任务' 或 '进入 Phase 1'。")

if __name__ == "__main__":
    main()