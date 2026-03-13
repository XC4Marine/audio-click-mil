# scripts/01_preprocess_audio.py
import yaml
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.data.preprocessor import process_one_long_audio   # ← 保持导入


def main():
    config_path = Path("configs/default.yaml")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    p = cfg["preprocess"]
    paths = cfg["paths"]

    raw_audio_dir = Path(paths["raw_audio_dir"])
    clips_root = Path(p["clips_root"])
    metadata_dir = Path(p["metadata"]["dir"])
    clip_csv_path = metadata_dir / p["metadata"]["clip_label_csv"]

    clips_root.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(raw_audio_dir.glob("Ori_Recording_*.wav"))

    all_records = []

    # 提取需要的配置项传给处理函数
    anno_configs = p.get("annotation_files", {})
    pos_criteria  = p.get("positive_criteria", {})

    for audio_path in tqdm(audio_files, desc="Processing long audios"):
        records = process_one_long_audio(
            audio_path=audio_path,
            anno_configs=anno_configs,           # 新增
            positive_criteria=pos_criteria,        # 新增
            sr=p["sr"],
            clip_duration=p["clip_duration_sec"],
            pad_zero=p["filename_pad_zero"],
            subtype=p["audio_subtype"],
            discard_short=p["discard_short_tail"],
            label_in_name=p["label_in_filename"],
            clips_root=clips_root,
        )
        all_records.extend(records)

    if all_records:
        df = pd.DataFrame(all_records)
        df.to_csv(clip_csv_path, index=False)
        print(f"完成。生成 {len(all_records)} 个片段")
        print(f"正样本数量: {df['has_positive'].sum()}")               # ← 改这里
        print(f"正样本比例: {df['has_positive'].mean():.2%}")         # 可选加这一行，更直观
        print(f"保存路径: {clips_root}")
        print(f"元数据: {clip_csv_path}")
    else:
        print("没有生成任何片段，请检查文件和标注")


if __name__ == "__main__":
    main()