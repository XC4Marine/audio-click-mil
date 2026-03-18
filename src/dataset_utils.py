# src/dataset_utils.py
"""
【模块：Dataset Utils】
负责音频加载、下采样、特征提取 (522维)、时间戳对齐及弱标签生成。
对应 skill.md Section 1 & 4
【版本：v2 - 已修复列名映射问题】
"""

import os
import numpy as np
import pandas as pd
import librosa
from typing import List, Dict, Tuple, Any

# --- 配置常量 ---
TARGET_SR = 48000  # 下采样目标
FFT_SIZE = 2048
HOP_SIZE = 512
N_MELS = 128
FMIN = 20
FMAX = 150000  # 中华白海豚高频特性

def load_and_resample_audio(file_path: str, target_sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    """
    加载音频并下采样到 48kHz
    """
    y, sr = librosa.load(file_path, sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y, target_sr

def extract_522_features(y_segment: np.ndarray, sr: int) -> np.ndarray:
    """
    提取 522 维特征：
    - 512 维：Mel 频谱图统计量
    - 10 维：时域与频域统计特征
    """
    features = []
    
    if len(y_segment) < 10: 
        return np.zeros(522, dtype=np.float32)
        
    mel_spec = librosa.feature.melspectrogram(y=y_segment, sr=sr, n_mels=N_MELS, 
                                              fmin=FMIN, fmax=FMAX, n_fft=FFT_SIZE, hop_length=HOP_SIZE)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80)
    
    # 512 维 Mel 统计量
    features.extend(np.mean(mel_db, axis=1))
    features.extend(np.std(mel_db, axis=1))
    features.extend(np.max(mel_db, axis=1))
    features.extend(np.min(mel_db, axis=1))
    
    # 10 维时域/频域特征
    rms = np.mean(librosa.feature.rms(y=y_segment))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y_segment))
    peak_amp = np.max(np.abs(y_segment))
    y_env = np.abs(librosa.onset.onset_strength(y=y_segment, sr=sr))
    spec_cent = librosa.feature.spectral_centroid(y=y_segment, sr=sr)
    spec_cent_mean = np.mean(spec_cent)
    
    mean_val = np.mean(y_segment)
    std_val = np.std(y_segment) + 1e-8
    skewness = np.mean(((y_segment - mean_val) / std_val) ** 3)
    kurtosis = np.mean(((y_segment - mean_val) / std_val) ** 4)
    energy = np.sum(y_segment ** 2)
    
    time_feats = [rms, zcr, peak_amp, np.mean(y_env), np.std(y_env), np.max(y_env), spec_cent_mean, skewness, kurtosis, energy]
    features.extend(time_feats)
    
    return np.array(features, dtype=np.float32)

def load_annotations(data_root: str) -> Dict[str, pd.DataFrame]:
    """
    加载所有 CSV 标注文件
    """
    dfs = {}
    # 精确文件名
    dfs['whistle'] = pd.read_csv(os.path.join(data_root, 'WhistleParameters-clean.csv'))
    dfs['click'] = pd.read_csv(os.path.join(data_root, 'ClickTrains.csv'))
    dfs['burst'] = pd.read_csv(os.path.join(data_root, 'BurstPulseTrains.csv'))
    dfs['buzz'] = pd.read_csv(os.path.join(data_root, 'BuzzTrains.csv'))
    dfs['meta'] = pd.read_csv(os.path.join(data_root, 'Results.csv'))
    
    # 【关键修复】清理列名中的首尾空格，防止匹配失败
    for key in dfs:
        dfs[key].columns = dfs[key].columns.str.strip()
        
    return dfs

def get_weak_labels(bag_start: float, bag_end: float, annotations: Dict[str, pd.DataFrame], file_id: int) -> Dict[str, int]:
    """
    根据 Bag 的时间范围 [start, end] 和文件 ID，判断是否存在哨声或脉冲串。
    【精确列名映射版】
    
    【当前执行：Phase 0 的任务 3 - 弱标签生成】
    """
    whistle_label = 0
    pulse_label = 0
    
    # --- 1. 检查哨声 (Whistle) ---
    w_df = annotations['whistle']
    if not w_df.empty:
        # 列名：'Original Audio File (No.)' (已 strip 空格)
        file_col = 'Original Audio File (No.)'
        time_col = 'Whistle Begins (s)'
        
        if file_col in w_df.columns and time_col in w_df.columns:
            # 转换为字符串比较，防止类型不匹配
            mask = (w_df[file_col].astype(str) == str(file_id)) & \
                   (w_df[time_col] >= bag_start) & \
                   (w_df[time_col] < bag_end)
            if mask.any():
                whistle_label = 1
        else:
            print(f"警告：Whistle CSV 中缺少列 {file_col} 或 {time_col}。实际列名：{list(w_df.columns)}")

    # --- 2. 检查脉冲串 (Click, Burst, Buzz) ---
    for p_type in ['click', 'burst', 'buzz']:
        p_df = annotations[p_type]
        if not p_df.empty:
            # 列名：'Ori_file_num(No.)' 和 'Train_start(s)'
            file_col = 'Ori_file_num(No.)'
            time_col = 'Train_start(s)'
            
            if file_col in p_df.columns and time_col in p_df.columns:
                mask = (p_df[file_col].astype(str) == str(file_id)) & \
                       (p_df[time_col] >= bag_start) & \
                       (p_df[time_col] < bag_end)
                if mask.any():
                    pulse_label = 1
                    break # 只要有一种脉冲存在即可
            else:
                if p_type == 'click':
                    print(f"警告：{p_type} CSV 中缺少列 {file_col} 或 {time_col}。实际列名：{list(p_df.columns)}")

    return {'whistle': whistle_label, 'pulse': pulse_label}

def create_instances(y: np.ndarray, sr: int, duration_sec: int = 10) -> List[np.ndarray]:
    """
    将长音频切分为固定时长的实例
    """
    samples_per_inst = int(sr * duration_sec)
    instances = []
    for i in range(0, len(y), samples_per_inst):
        segment = y[i:i+samples_per_inst]
        if len(segment) < samples_per_inst:
            segment = np.pad(segment, (0, samples_per_inst - len(segment)), mode='constant')
        instances.append(segment)
    return instances