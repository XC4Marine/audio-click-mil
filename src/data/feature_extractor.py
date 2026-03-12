# src/data/feature_extractor.py
from pathlib import Path
import librosa
import numpy as np
import pandas as pd

from src.utils.audio_utils import (
    compute_mfcc_with_deltas,
    normalize_per_instance,
    plot_and_save_mfcc
)


# src/data/feature_extractor.py
def process_one_clip(
    clip_row: dict,
    anno_df: pd.DataFrame,
    config: dict   # ← 改回 config
) -> list[dict]:
    clip_path = Path(config["clips_dir"]) / clip_row["clip_filename"]
    if not clip_path.exists():
        print(f"文件不存在，跳过: {clip_path}")
        return []

    y, sr = librosa.load(clip_path, sr=None, mono=True)
    if sr != config["sr"]:
        y = librosa.resample(y, orig_sr=sr, target_sr=config["sr"])
        sr = config["sr"]

    duration = len(y) / sr
    n_instances = int(duration // config["instance_duration_sec"])

    # 取得 click 區間
    orig_audio = clip_row["original_audio"]
    file_num = int(orig_audio.replace("Ori_Recording_", "").replace(".wav", ""))
    clicks_this_file = anno_df[anno_df["Ori_file_num(No.)"] == file_num]
    click_intervals = list(zip(clicks_this_file["Train_start(s)"], clicks_this_file["Train_end(s)"]))

    clip_start_sec = clip_row["start_sec"]
    records = []

    for i in range(n_instances):
        inst_start = i * config["instance_duration_sec"]
        inst_end = (i + 1) * config["instance_duration_sec"]

        abs_start = clip_start_sec + inst_start
        abs_end = clip_start_sec + inst_end

        has_click = any(
            not (c_end <= abs_start or c_start >= abs_end)
            for c_start, c_end in click_intervals
        )
        label = 1 if has_click else 0

        start_sample = int(inst_start * sr)
        end_sample = int(inst_end * sr)
        segment = y[start_sample:end_sample]

    mfcc = compute_mfcc_with_deltas(
            segment, sr=sr,
            n_mfcc=config["n_mfcc"],
            hop_length=config["hop_length"],
            n_fft=config["n_fft"],
            fmax=config["fmax"],
            include_delta=config["include_delta"],
            include_delta_delta=config["include_delta_delta"]
        )

    if config["cmvn"]:
        mfcc = normalize_per_instance(mfcc)

        stem = clip_row["clip_filename"].replace(".wav", "")
        inst_name = f"{stem}_{i:03d}"
        # 保存 npy
        npy_path = Path(config["output_root"]) / config["npy_dir"] / f"{inst_name}.npy"
        np.save(npy_path, mfcc)

        # 保存 png
        img_path = Path(config["output_root"]) / config["img_dir"] / f"{inst_name}.png"
        plot_and_save_mfcc(
            mfcc=mfcc,
            sr=sr,
            hop_length=config["hop_length"],
            title=f"{stem} | inst {i:03d} | label:{label}",
            save_path=img_path,
            dpi=config["dpi"]
        )

        records.append({
            "instance_filename": f"{inst_name}.npy",
            "clip_filename": clip_row["clip_filename"],
            "instance_idx": i,
            "start_sec_in_clip": round(inst_start, 3),
            "end_sec_in_clip": round(inst_end, 3),
            "start_sec_abs": round(abs_start, 3),
            "end_sec_abs": round(abs_end, 3),
            "label": label,
            "original_audio": orig_audio
        })

    return records