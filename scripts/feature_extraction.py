import os
import numpy as np
from tqdm import tqdm
import argparse
import librosa
import pywt

# 绝对路径
INSTANCE_DIR = r"D:\Project_Github\audio_click_mil\processed_data\instances"
FEATURE_BASE = r"D:\Project_Github\audio_click_mil\processed_data\features"

def robust_pool(matrix, target_n=128):
    """
    鲁棒的池化：处理 Bins 数量过少的情况
    """
    half_n = target_n // 2
    
    # 1. 压缩时间轴：取每个频带的时间平均和时间最大
    # matrix 形状通常是 (n_bins, n_frames)
    mean_over_time = np.mean(matrix, axis=1) # (n_bins,)
    max_over_time = np.max(matrix, axis=1)   # (n_bins,)

    # 2. 如果 n_bins 小于 half_n (64)，先线性插值扩充到 half_n
    n_bins = len(mean_over_time)
    if n_bins < half_n:
        x_old = np.linspace(0, 1, n_bins)
        x_new = np.linspace(0, 1, half_n)
        mean_over_time = np.interp(x_new, x_old, mean_over_time)
        max_over_time = np.interp(x_new, x_old, max_over_time)

    # 3. 再进行切分（现在保证了长度至少为 half_n）
    res_mean = np.array([np.mean(part) for part in np.array_split(mean_over_time, half_n)])
    res_max = np.array([np.max(part) for part in np.array_split(max_over_time, half_n)])
    
    return np.concatenate([res_mean, res_max])

def cqt_feature(wave, sr, n=128):
    """CQT 特征：增加最低 bins 限制"""
    # 限制最高频率
    fmax_limit = (sr // 2) * 0.9
    
    # 计算 max_bins，至少保证有 12 个 bins（一个八度）
    max_bins = int(12 * np.log2(fmax_limit / 5000))
    safe_bins = max(12, min(64, max_bins)) # 保证在 12 到 64 之间
    
    try:
        cqt = np.abs(librosa.cqt(wave, 
                                 sr=sr, 
                                 fmin=5000, 
                                 n_bins=safe_bins, 
                                 bins_per_octave=12, 
                                 hop_length=512))
        return robust_pool(cqt, target_n=n)
    except Exception as e:
        # 如果 CQT 依然失败（比如音频太短），返回全零向量防止崩溃
        print(f"CQT 提取失败警告: {e}")
        return np.zeros(n)

def gammatone_feature(wave, sr, n=128):
    """
    Gammatone 滤波器组：增加兼容性检查
    """
    S = np.abs(librosa.stft(wave, n_fft=2048, hop_length=512))
    fmax = min(sr // 2 - 1000, 250000) # 针对高采样率做上限保护
    
    try:
        # 尝试调用（较新版本 librosa）
        if hasattr(librosa.filters, 'gammatone'):
            weights = librosa.filters.gammatone(sr=sr, n_filters=64, fmin=5000, fmax=fmax)
        else:
            # 如果没有 gammatone，回退到高分辨率的 Mel 滤波器组
            # Mel 在高频虽然是对数的，但在你的高采样率下依然能很好地模拟生物听觉
            weights = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=64, fmin=5000, fmax=fmax)
            
        g_feat = np.dot(weights, S)
        # 转换为分贝
        g_db = librosa.amplitude_to_db(g_feat)
        return robust_pool(g_db, target_n=n)
        
    except Exception as e:
        print(f"滤波器组提取失败: {e}")
        return np.zeros(n)

def teager_kaiser_feature(wave, n=128):
    tkeo = np.zeros_like(wave)
    tkeo[1:-1] = wave[1:-1]**2 - wave[:-2] * wave[2:]
    tkeo = np.abs(tkeo)
    # 对于一维信号，直接切分取 Mean 和 Max
    half_n = n // 2
    parts = np.array_split(tkeo, half_n)
    res_mean = np.array([np.mean(p) for p in parts])
    res_max = np.array([np.max(p) for p in parts])
    return np.concatenate([res_mean, res_max])

def wavelet_feature(wave, n=128):
    scales = np.arange(1, 65) # 64 个尺度
    coefs, freqs = pywt.cwt(wave, scales, 'morl')
    return robust_pool(np.abs(coefs), target_n=n)

def mfcc_feature(wave, sr, n=128):
    # 提取 64 维 MFCC
    mfcc = librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=64, n_fft=2048, hop_length=512)
    return robust_pool(mfcc, target_n=n)

def logmel_cnn_feature(wave, sr, n=128):
    mel = librosa.feature.melspectrogram(y=wave, sr=sr, n_mels=64, n_fft=2048, 
                                         hop_length=512, fmin=5000, fmax=sr//2)
    logmel = librosa.power_to_db(mel)
    return robust_pool(logmel, target_n=n)

def extract_features(method: str):
    out_dir = os.path.join(FEATURE_BASE, method)
    os.makedirs(out_dir, exist_ok=True)
    
    npy_files = [f for f in os.listdir(INSTANCE_DIR) if f.endswith('.npy')]
    print(f"共发现 {len(npy_files)} 个 bag 文件，开始 {method} (128维) 特征提取...")
    
    for fname in tqdm(npy_files, desc=f"{method} 提取进度"):
        path = os.path.join(INSTANCE_DIR, fname)
        data = np.load(path)  # 形状 (60, SR)
        
        # 动态获取当前文件的采样率 (SR)
        current_sr = data.shape[1] 
        
        feats = []
        for i in range(data.shape[0]):
            wave = data[i]
            if method == 'teager':
                feat = teager_kaiser_feature(wave, n=128)
            elif method == 'wavelet':
                feat = wavelet_feature(wave, n=128)
            elif method == 'mfcc':
                feat = mfcc_feature(wave, sr=current_sr, n=128)
            elif method == 'logmel':
                feat = logmel_cnn_feature(wave, sr=current_sr, n=128)
            elif method == 'cqt':
                feat = cqt_feature(wave, sr=current_sr, n=128)
            elif method == 'gammatone':
                feat = gammatone_feature(wave, sr=current_sr, n=128)
            else:
                raise ValueError("未知 method")
            feats.append(feat)
        
        feat_bag = np.stack(feats)  # (60, 128)
        out_name = fname.replace('.npy', '_feat.npy')
        np.save(os.path.join(out_dir, out_name), feat_bag)
    
    print(f"\n{method} 特征提取完成！保存至: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True, 
                        choices=['teager', 'wavelet', 'mfcc', 'logmel', 'cqt', 'gammatone'])
    # 适配 Jupyter 环境运行
    import sys
    if 'ipykernel' in sys.modules:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()
        
    extract_features(args.method)