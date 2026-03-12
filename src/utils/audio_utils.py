# src/utils/audio_utils.py
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path


def load_and_resample(
    audio_path: Path | str,
    target_sr: int,
) -> tuple[np.ndarray, int]:
    """
    加载音频文件并重采样到目标采样率（单声道）。
    返回 (y, sr) ，y 是 float32 numpy array。
    """
    audio_path = Path(audio_path)
    y, sr_orig = librosa.load(audio_path, sr=None, mono=True)
    
    if sr_orig != target_sr:
        y = librosa.resample(y, orig_sr=sr_orig, target_sr=target_sr)
        sr = target_sr
    else:
        sr = sr_orig
        
    return y.astype(np.float32), sr


def intervals_overlap(
    seg_start: float,
    seg_end: float,
    click_start: float,
    click_end: float
) -> bool:
    """判断两个区间是否有重叠"""
    return not (click_end <= seg_start or click_start >= seg_end)


def make_clip_filename(
    stem: str,
    index: int,
    has_click: bool,
    pad_zero: int = 3,
    label_in_name: bool = True
) -> str:
    """生成切片文件名，例如 Ori_Recording_04_000_1.wav"""
    idx_str = f"{index:0{pad_zero}d}"
    suffix = f"_{int(has_click)}" if label_in_name else ""
    return f"{stem}_{idx_str}{suffix}.wav"


def save_audio_clip(
    segment: np.ndarray,
    sr: int,
    output_path: Path,
    subtype: str = "FLOAT"
) -> None:
    """保存音频片段"""
    sf.write(output_path, segment, sr, subtype=subtype)



def compute_mfcc_with_deltas(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 40,
    hop_length: int = 512,
    n_fft: int = 2048,
    fmax: int = 8000,
    include_delta: bool = True,
    include_delta_delta: bool = True
) -> np.ndarray:
    """计算 MFCC + delta + delta-delta"""
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc,
        hop_length=hop_length, n_fft=n_fft, fmax=fmax
    )
    if include_delta:
        delta = librosa.feature.delta(mfcc)
        mfcc = np.vstack([mfcc, delta])
    if include_delta_delta:
        dd = librosa.feature.delta(mfcc, order=2)
        mfcc = np.vstack([mfcc, dd])
    return mfcc.T  # (n_frames, n_features)


def normalize_per_instance(mfcc: np.ndarray) -> np.ndarray:
    """每个实例独立做 CMVN"""
    mean = np.mean(mfcc, axis=0, keepdims=True)
    std = np.std(mfcc, axis=0, keepdims=True) + 1e-8
    return (mfcc - mean) / std


def plot_and_save_mfcc(
    mfcc: np.ndarray,
    sr: int,
    hop_length: int,
    title: str,
    save_path: Path,
    dpi: int = 150
):
    """保存 MFCC 可视化图"""
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mfcc.T,
        x_axis='time',
        hop_length=hop_length,
        sr=sr,
        cmap='viridis'
    )
    plt.colorbar(format='%+2.0f')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()