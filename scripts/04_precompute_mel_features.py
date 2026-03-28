import os
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
import json

# ==================== 配置 ====================
ROOT_DIR = r"D:\Project_Github\audio_click_mil"
INSTANCES_DIR = os.path.join(ROOT_DIR, "processed_data", "instances")
PRECOMPUTED_DIR = os.path.join(ROOT_DIR, "processed_data", "mel_features")  # 新文件夹

os.makedirs(PRECOMPUTED_DIR, exist_ok=True)

sample_rate = 200000
n_mels = 128
max_time_frames = 100   # 和你原来代码保持一致

# Mel 变换（只创建一次）
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=4096,
    win_length=4096,
    hop_length=1024,
    n_mels=n_mels,
    f_min=5000,
    f_max=200000,
    norm="slaney"
).to('cuda' if torch.cuda.is_available() else 'cpu')  # 可选放到 GPU 加速

to_db = torchaudio.transforms.AmplitudeToDB()

print("开始预计算所有 bag 的 log-mel 特征...")

# 遍历所有 .npy 文件
bag_files = [f for f in os.listdir(INSTANCES_DIR) if f.endswith(".npy")]

for fname in tqdm(bag_files):
    # 解析文件名获取信息（和原来一样）
    parts = fname.split("_")
    file_num = int(parts[1])
    bag_idx = int(parts[3])
    label = int(parts[5].split(".")[0])
    
    input_path = os.path.join(INSTANCES_DIR, fname)
    output_path = os.path.join(PRECOMPUTED_DIR, fname.replace(".npy", ".pt"))
    
    # 如果已经存在就跳过（支持断点续算）
    if os.path.exists(output_path):
        continue
    
    # 加载 waveform
    waveform = np.load(input_path)  # (60, 200000)
    waveform = torch.from_numpy(waveform).float()
    
    if torch.cuda.is_available():
        waveform = waveform.to('cuda')
    
    mel_features = []
    for i in range(60):
        seg = waveform[i:i+1, :]  # (1, 200000)
        mel = mel_transform(seg)
        mel_db = to_db(mel)
        mel_db = mel_db.squeeze(0)  # (64, time_frames)
        
        # 固定时间维度（和原来完全一致）
        if mel_db.shape[1] > max_time_frames:
            mel_db = mel_db[:, :max_time_frames]
        elif mel_db.shape[1] < max_time_frames:
            pad = torch.zeros((n_mels, max_time_frames - mel_db.shape[1]), 
                             device=mel_db.device)
            mel_db = torch.cat([mel_db, pad], dim=1)
        
        mel_features.append(mel_db.cpu())  # 保存到 CPU
    
    mel_features = torch.stack(mel_features)  # (60, 64, 100)
    
    # 保存为 .pt 文件，同时保存 label 和 bag_info
    torch.save({
        "mel_features": mel_features,
        "label": torch.tensor(label, dtype=torch.float32),
        "file_num": file_num,
        "bag_idx": bag_idx,
        "original_filename": fname
    }, output_path)

print(f"预计算完成！所有特征已保存到: {PRECOMPUTED_DIR}")
print(f"总共处理了 {len(bag_files)} 个 bag。")