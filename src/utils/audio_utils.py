# src/utils/audio_utils.py
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf


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