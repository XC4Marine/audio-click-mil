# scripts/03_extract_instance_features.py
"""
根据 splits 生成 instance-level 特征和标签
"""

import yaml
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import sys

from src.data.feature_extractor import process_one_clip


def load_config(config_path: str = "configs/default.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        print(f"配置文件不存在：{path}", file=sys.stderr)
        sys.exit(1)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_needed_stems(splits_dir: Path) -> set:
    needed = set()
    for name in ["train", "val", "test"]:
        p = splits_dir / f"{name}.txt"
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                needed.update(line.strip() for line in f if line.strip())
        else:
            print(f"警告：split 文件不存在，已跳过：{p}")
    return needed


def main():
    cfg = load_config()
    paths = {k: Path(v) for k, v in cfg["paths"].items()}
    preprocess = cfg["preprocess"]
    extract_cfg = cfg.get("extract", {})

    # 建立输出目录
    for d in [paths["features_root"] / paths["npy_subdir"],
              paths["features_root"] / paths["img_subdir"],
              paths["instance_label_csv"].parent]:
        d.mkdir(parents=True, exist_ok=True)

    # 读取数据
    clip_df = pd.read_csv(paths["clip_label_csv"])
    anno_df = pd.read_csv(paths["anno_csv"])
    anno_df["Ori_file_num(No.)"] = anno_df["Ori_file_num(No.)"].astype(int)

    needed_stems = get_needed_stems(paths["splits_dir"])
    if not needed_stems:
        print("没有找到任何 split 文件，退出")
        return

    process_df = clip_df[
        clip_df["clip_filename"].str.replace(".wav", "", regex=False).isin(needed_stems)
    ]

    if process_df.empty:
        print("过滤后没有 clip 需要处理，退出")
        return

    print(f"将处理 {len(process_df)} / {len(clip_df)} 个 clip")

    full_config = {
                **preprocess,                          # sr, n_mfcc, hop_length 等
                "clips_dir": str(paths["clips_dir"]),  # 转 str 避免 Path 对象问题
                "output_root": str(paths["features_root"]),
                "npy_dir": str(paths["npy_subdir"]),
                "img_dir": str(paths["img_subdir"]),
                "dpi": preprocess.get("dpi", 150),     # 如果 dpi 不在 preprocess 里
            }
    
    all_records = []

    for row in tqdm(
        process_df.itertuples(index=False),
        total=len(process_df),
        desc=extract_cfg.get("tqdm_desc", "提取 instance-level 特征")
    ):
        try:
            
            records = process_one_clip(row._asdict(), anno_df, full_config)
            all_records.extend(records)
        except Exception as e:
            print(f"处理 clip {row.clip_filename} 时出错：{e}")
            if extract_cfg.get("error_on_missing_clip", True):
                raise

    instance_df = pd.DataFrame(all_records)
    instance_df.to_csv(paths["instance_label_csv"], index=False)

    print(f"\n完成！共产生 {len(instance_df)} 条记录")
    print(f"保存至：{paths['instance_label_csv']}")
    if not instance_df.empty:
        pos_ratio = instance_df["label"].mean()
        print(f"实例级正样本比例：{pos_ratio:.4f} ({instance_df['label'].sum()} / {len(instance_df)})")


if __name__ == "__main__":
    main()