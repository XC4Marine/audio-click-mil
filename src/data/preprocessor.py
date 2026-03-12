# src/data/preprocessor.py
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.utils.audio_utils import (
    load_and_resample,
    intervals_overlap,
    make_clip_filename,
    save_audio_clip,
)


def process_one_long_audio(
    audio_path: Path,
    anno_df: pd.DataFrame,
    target_sr: int,
    clip_duration: float,
    pad_zero: int,
    subtype: str,
    discard_short: bool,
    label_in_name: bool,
    clips_root: Path,
) -> list[dict]:
    """
    处理单个长音频，返回该音频生成的 clips 元数据列表
    """
    stem = audio_path.stem
    file_num_str = stem.split("_")[-1]
    try:
        file_num = int(file_num_str)
    except ValueError:
        print(f"跳过文件名不符合预期的文件: {audio_path.name}")
        return []

    # 加载 & 重采样
    y, sr = load_and_resample(audio_path, target_sr)

    total_sec = len(y) / sr

    if discard_short:
        n_clips = int(total_sec // clip_duration)
    else:
        n_clips = int(np.ceil(total_sec / clip_duration))

    # 获取该文件的 click 区间
    df_this = anno_df[anno_df['Ori_file_num(No.)'] == file_num]
    click_intervals = list(zip(df_this['Train_start(s)'], df_this['Train_end(s)']))

    records = []

    for i in range(n_clips):
        start_sec = i * clip_duration
        end_sec = min((i + 1) * clip_duration, total_sec)

        if discard_short and (end_sec - start_sec) < clip_duration:
            continue

        # 判断是否有 click 重叠
        has_click = any(
            intervals_overlap(start_sec, end_sec, c_start, c_end)
            for c_start, c_end in click_intervals
        )

        label = 1 if has_click else 0

        clip_name = make_clip_filename(stem, i, has_click, pad_zero, label_in_name)
        out_path = clips_root / clip_name

        # 切片 & 保存
        start_idx = int(start_sec * sr)
        end_idx = int(end_sec * sr)
        segment = y[start_idx:end_idx]

        save_audio_clip(segment, sr, out_path, subtype)

        records.append({
            'clip_filename': clip_name,
            'has_click': label,
            'original_audio': audio_path.name,
            'start_sec': round(start_sec, 3),
            'end_sec': round(end_sec, 3)
        })

    return records