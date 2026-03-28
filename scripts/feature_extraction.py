import os
import numpy as np
from tqdm import tqdm
import argparse
import librosa
import pywt  # 替换 scipy.signal 为 pywt

# 绝对路径保持不变
INSTANCE_DIR = r"D:\Project_Github\audio_click_mil\processed_data\instances"
FEATURE_BASE = r"D:\Project_Github\audio_click_mil\processed_data\features"

def teager_kaiser_feature(wave, n=32):
    tkeo = np.zeros_like(wave)
    tkeo[1:-1] = wave[1:-1]**2 - wave[:-2] * wave[2:]
    # 这里使用 split 而不是 reshape，防止 wave 长度不能被 n 整除
    return np.array([np.mean(part) for part in np.array_split(tkeo, n)])

def wavelet_feature(wave, n=32):
    """
    使用 PyWavelets 实现连续小波变换 (CWT)
    wave: 原始音频信号
    n: 尺度数量（特征维度）
    """
    # 1. 定义尺度 (Scales)
    scales = np.arange(1, n + 1)
    
    # 2. 调用 pywt.cwt
    # 'cmor1.5-1.0' 是复数 Morlet 小波，类似于 scipy 的 morlet2
    # 你也可以简单使用 'mexh' (墨西哥帽) 或 'morl'
    coefs, freqs = pywt.cwt(wave, scales, 'morl')
    
    # 3. 取每个尺度绝对值的均值，得到 n 维特征
    # coefs 的形状是 (n, len(wave))
    return np.mean(np.abs(coefs), axis=1)

def mfcc_feature(wave, sr=200000, n=32):
    mfcc = librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=n, n_fft=2048, hop_length=512)
    return np.mean(mfcc, axis=1)

def logmel_cnn_feature(wave, sr=200000, n=32):
    mel = librosa.feature.melspectrogram(y=wave, sr=sr, n_mels=64, n_fft=2048, hop_length=512, fmin=5000, fmax=200000)
    logmel = librosa.power_to_db(mel)
    pooled = np.mean(logmel, axis=1)
    return np.array([np.mean(part) for part in np.array_split(pooled, n)])

def extract_features(method: str):
    out_dir = os.path.join(FEATURE_BASE, method)
    os.makedirs(out_dir, exist_ok=True)
    
    npy_files = [f for f in os.listdir(INSTANCE_DIR) if f.endswith('.npy')]
    print(f"共发现 {len(npy_files)} 个 bag 文件，开始 {method} 特征提取...")
    
    for fname in tqdm(npy_files, desc=f"{method} 提取进度"):
        path = os.path.join(INSTANCE_DIR, fname)
        try:
            data = np.load(path)  # 预期形状 (60, 200000)
        except Exception as e:
            print(f"读取文件 {fname} 失败: {e}")
            continue
        
        feats = []
        for i in range(data.shape[0]):
            wave = data[i]
            if method == 'teager':
                feat = teager_kaiser_feature(wave)
            elif method == 'wavelet':
                feat = wavelet_feature(wave)
            elif method == 'mfcc':
                feat = mfcc_feature(wave, n=32)
            elif method == 'logmel':
                feat = logmel_cnn_feature(wave, n=32)
            else:
                raise ValueError("未知method")
            feats.append(feat)
        
        feat_bag = np.stack(feats)  # (60, 32)
        out_name = fname.replace('.npy', '_feat.npy')
        np.save(os.path.join(out_dir, out_name), feat_bag)
    
    print(f"\n{method} 特征提取完成！")
    print(f"保存路径: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True, choices=['teager', 'wavelet', 'mfcc', 'logmel'])
    args = parser.parse_args()
    extract_features(args.method)