import os
import numpy as np
from tqdm import tqdm
import argparse
import librosa

# 路径配置
INSTANCE_DIR = r"D:\Project_Github\audio_click_mil\processed_data\balanced_bags"
FEATURE_BASE = r"D:\Project_Github\audio_click_mil\processed_data\features"

def max_pooling_only(matrix, target_n=128):
    """
    纯 Max-Pooling 池化：将频率/特征维度压缩并只提取最大值
    """
    # 1. 首先在时间轴（列）上取最大值，保留每个频带最突出的信号
    # matrix 形状: (n_bins, n_frames) -> 结果形状: (n_bins,)
    if matrix.ndim > 1:
        max_over_time = np.max(matrix, axis=1)
    else:
        max_over_time = matrix # 对于已经是 1D 的 TKEO

    # 2. 如果频带数量 n_bins 不等于目标维度 target_n，进行切分并取最大值
    # np.array_split 会尽可能均匀地切分数组
    parts = np.array_split(max_over_time, target_n)
    
    # 3. 每一段只取 max，不取 mean
    res_max = np.array([np.max(part) if len(part) > 0 else 0 for part in parts])
    
    return res_max

def teager_kaiser_feature(wave, n=128):
    """
    Teager-Kaiser 算子：提取能量瞬时变化
    """
    tkeo = np.zeros_like(wave)
    # 计算公式: ψ(x[n]) = x[n]^2 - x[n-1] * x[n+1]
    tkeo[1:-1] = wave[1:-1]**2 - wave[:-2] * wave[2:]
    tkeo = np.abs(tkeo)
    
    # 直接对 1D 信号进行 128 点 Max-Pooling
    return max_pooling_only(tkeo, target_n=n)

def logmel_feature(wave, sr, n=128):
    """
    Log-Mel 特征：仅保留 Max-Pooling
    """
    # 使用更高频率分辨率（n_mels=128 或更高），然后池化到 128
    mel = librosa.feature.melspectrogram(
        y=wave, 
        sr=sr, 
        n_mels=max(n, 128), # 至少提取 128 阶
        n_fft=2048, 
        hop_length=512, # 减小步长以获得更多时间帧
        fmin=5000, 
        fmax=sr//2
    )
    logmel = librosa.power_to_db(mel)
    
    # 执行 Max-Pooling
    return max_pooling_only(logmel, target_n=n)

def extract_features(method: str):
    out_dir = os.path.join(FEATURE_BASE, method)
    os.makedirs(out_dir, exist_ok=True)
    
    npy_files = [f for f in os.listdir(INSTANCE_DIR) if f.endswith('.npy')]
    print(f"模式: {method} | 池化策略: Max-Pooling Only (128维)")
    
    for fname in tqdm(npy_files, desc=f"{method} 提取进度"):
        path = os.path.join(INSTANCE_DIR, fname)
        data = np.load(path)  # 形状 (60, samples_per_second)
        
        current_sr = data.shape[1] 
        feats = []
        
        for i in range(data.shape[0]):
            wave = data[i]
            if method == 'teager':
                feat = teager_kaiser_feature(wave, n=128)
            elif method == 'logmel':
                feat = logmel_feature(wave, sr=current_sr, n=128)
            else:
                raise ValueError("仅支持 'teager' 或 'logmel'")
            feats.append(feat)
        
        feat_bag = np.stack(feats)  # (60, 128)
        out_name = fname.replace('.npy', '_feat.npy')
        np.save(os.path.join(out_dir, out_name), feat_bag)
    
    print(f"\n{method} 提取完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True, choices=['teager', 'logmel'])
    
    import sys
    # 修复之前的 Namespace 解包报错                          
    if 'ipykernel' in sys.modules:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()
        
    extract_features(args.method)