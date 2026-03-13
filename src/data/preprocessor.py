# src/data/preprocessor.py
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.signal as signal

from src.utils.audio_utils import (
    load_and_resample,
    intervals_overlap,
    make_clip_filename,
    save_audio_clip,
)


def highpass_filter(y: np.ndarray, sr: int, cutoff: float = 5000.0, order: int = 4) -> np.ndarray:
    sos = signal.butter(order, cutoff, btype='high', fs=sr, output='sos')
    return signal.sosfiltfilt(sos, y)


def load_intervals_for_type(
    anno_config: dict,
    file_num: int,
    enabled: bool
) -> list[tuple[float, float]]:
    """根据配置加载某种信号类型的区间（如果未启用则返回空列表）"""
    if not enabled:
        return []

    path = Path(anno_config["path"])
    if not path.exists():
        print(f"警告：标注文件不存在，跳过 → {path}")
        return []

    df = pd.read_csv(path)

    # 统一文件编号列（根据实际情况可能需要调整列名）
    file_col_candidates = [
        'Ori_file_num(No.)', 'Original Audio File (No.)  ',
        'Ori_file_num(No.)',  # 兼容不同写法
    ]
    file_col = next((c for c in file_col_candidates if c in df.columns), None)
    if file_col is None:
        raise ValueError(f"无法找到文件编号列 in {path.name}")

    df_this = df[df[file_col].astype(str).str.strip() == str(file_num)]

    intervals = []

    start_col = anno_config["start_column"]
    end_col   = anno_config.get("end_column")
    dur_col   = anno_config.get("duration_column")

    for _, row in df_this.iterrows():
        try:
            start = float(row[start_col])
            if end_col and pd.notna(row[end_col]):
                end = float(row[end_col])
            elif dur_col and pd.notna(row[dur_col]):
                end = start + float(row[dur_col]) / 1000.0
            else:
                continue  # 缺少必要列，跳过

            intervals.append((start, end))
        except (ValueError, TypeError, KeyError):
            continue  # 该行数据有问题，跳过

    return intervals


def process_one_long_audio(
    audio_path: Path,
    anno_configs: dict,               # 新增：从 cfg 传整个 annotation_files
    positive_criteria: dict,           # 新增：四个 bool
    sr: int,
    clip_duration: float,
    pad_zero: int,
    subtype: str,
    discard_short: bool,
    label_in_name: bool,
    clips_root: Path,
) -> list[dict]:
    stem = audio_path.stem
    file_num_str = stem.split("_")[-1]
    try:
        file_num = int(file_num_str)
    except ValueError:
        print(f"跳过文件名不符合预期的文件: {audio_path.name}")
        return []

    y, sr = load_and_resample(audio_path, sr)
    y = highpass_filter(y, sr)   # 保持原有高通滤波

    total_sec = len(y) / sr

    n_clips = int(total_sec // clip_duration) if discard_short else int(np.ceil(total_sec / clip_duration))

    # 预加载所有启用的信号类型的区间（只读一次）
    all_intervals = []   # list of (start, end)

    for sig_type, enabled in positive_criteria.items():
        if sig_type not in anno_configs:
            continue
        intervals = load_intervals_for_type(
            anno_configs[sig_type],
            file_num,
            enabled
        )
        all_intervals.extend(intervals)

    # 如果没有任何一种信号启用，且没有区间 → 直接报错
    if not positive_criteria or not any(positive_criteria.values()):
        raise ValueError(
            f"文件 {audio_path.name}：没有任何信号类型被启用，无法判定正负样本，请检查配置 positive_criteria"
        )

    records = []

    for i in range(n_clips):
        start_sec = i * clip_duration
        end_sec = min((i + 1) * clip_duration, total_sec)

        if discard_short and (end_sec - start_sec) < clip_duration:
            continue

        # 只要任意一个区间与当前 clip 重叠 → 正样本
        has_positive = any(
            intervals_overlap(start_sec, end_sec, c_start, c_end)
            for c_start, c_end in all_intervals
        )

        label = 1 if has_positive else 0

        clip_name = make_clip_filename(stem, i, has_positive, pad_zero, label_in_name)
        out_path = clips_root / clip_name

        start_idx = int(start_sec * sr)
        end_idx = int(end_sec * sr)
        segment = y[start_idx:end_idx]
        
        # 在循环里面，每次保存前都确保目录存在（非常保险）
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_audio_clip(segment, sr, out_path, subtype)

        records.append({
            'clip_filename': clip_name,
            'has_positive': label,          # 建议改名，更通用（原来是 has_click）
            'original_audio': audio_path.name,
            'start_sec': round(start_sec, 3),
            'end_sec': round(end_sec, 3)
        })

    return records