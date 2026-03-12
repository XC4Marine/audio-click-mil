# scripts/02_create_dataset_splits.py
"""
第二步：数据集划分脚本
1. 读取 clip_labels.csv（包级标签）
2. 按每个原始长音频内部进行正负平衡（可选，TARGET_RATIO 可调）
3. 按原始长音频分组进行 train/val/test 划分（8:1:1）
4. 保存 train.txt / val.txt / test.txt（每个文件列出 stem，不带 .wav）
"""

import yaml
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """读取配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def balance_per_audio(
    df: pd.DataFrame,
    target_ratio: float = 1.5,
    max_neg_full_neg_audio: int = 10
) -> pd.DataFrame:
    """
    按每个原始长音频内部平衡正负样本
    - 保留所有正样本
    - 负样本最多保留到 正样本数量 × target_ratio
    - 全负音频最多保留 max_neg_full_neg_audio 个负样本
    """
    grouped = df.groupby("original_audio")
    balanced_records = []

    for orig_audio, group in grouped:
        pos = group[group["has_click"] == 1]
        neg = group[group["has_click"] == 0]

        n_pos = len(pos)
        n_neg = len(neg)

        if n_pos == 0:
            # 全负音频
            if n_neg > 0:
                kept_neg = neg.sample(min(max_neg_full_neg_audio, n_neg))
                balanced_records.extend(kept_neg.to_dict("records"))
            continue

        # 有正样本的音频
        n_keep_neg = min(n_neg, int(n_pos * target_ratio))
        kept_neg = neg.sample(n_keep_neg) if n_keep_neg < n_neg else neg

        balanced_records.extend(pos.to_dict("records"))
        balanced_records.extend(kept_neg.to_dict("records"))

    return pd.DataFrame(balanced_records)


def create_splits(
    balanced_df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按原始长音频分组进行 train/val/test 划分"""
    unique_audios = balanced_df["original_audio"].unique()

    # 先分 test
    audio_train_val, audio_test = train_test_split(
        unique_audios,
        test_size=1 - train_ratio,
        random_state=random_state,
    )

    # 再从剩余分 val
    audio_train, audio_val = train_test_split(
        audio_train_val,
        test_size=val_ratio / (train_ratio + val_ratio),
        random_state=random_state,
    )

    train_df = balanced_df[balanced_df["original_audio"].isin(audio_train)]
    val_df   = balanced_df[balanced_df["original_audio"].isin(audio_val)]
    test_df  = balanced_df[balanced_df["original_audio"].isin(audio_test)]

    return train_df, val_df, test_df


def save_split_txt(df: pd.DataFrame, filepath: Path):
    """保存 stem 列表到 txt"""
    stems = df["clip_filename"].str.replace(".wav", "", regex=False).tolist()
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(stems))
    print(f"保存 {len(stems)} 条到 {filepath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--balance", action="store_true", help="是否进行正负平衡")
    parser.add_argument("--target-ratio", type=float, default=1.5, help="负:正 最大比例")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # 路径配置
    clip_csv_path = Path(cfg["paths"]["clip_label_csv"])
    splits_dir = Path(cfg["paths"]["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    # 读取包级标签
    clip_df = pd.read_csv(clip_csv_path)
    print(f"原始 clip 数量: {len(clip_df)}")
    print("原始正样本比例:", clip_df["has_click"].mean())

    # 是否平衡
    balance_cfg = cfg.get("balance", {"enabled": False, "target_ratio": 1.5, "max_neg_full_neg_audio": 10})

    if balance_cfg["enabled"]:
        print("執行正負平衡...")
        balanced_df = balance_per_audio(
            clip_df,
            target_ratio=balance_cfg["target_ratio"],
            max_neg_full_neg_audio=balance_cfg["max_neg_full_neg_audio"]
        )
    else:
        balanced_df = clip_df.copy()
        print("跳過平衡，使用原始數據")

    # 划分
    print("开始数据集划分...")
    train_df, val_df, test_df = create_splits(
        balanced_df,
        train_ratio=cfg.get("splits", {}).get("train_ratio", 0.8),
        val_ratio=cfg.get("splits", {}).get("val_ratio", 0.1),
        random_state=cfg.get("splits", {}).get("random_state", 42)
    )

    print("\n划分结果：")
    print(f"Train: {len(train_df)} clips ({train_df['has_click'].mean():.4f} 正)")
    print(f"Val:   {len(val_df)} clips ({val_df['has_click'].mean():.4f} 正)")
    print(f"Test:  {len(test_df)} clips ({test_df['has_click'].mean():.4f} 正)")

    # 保存 txt
    save_split_txt(train_df, splits_dir / "train.txt")
    save_split_txt(val_df,   splits_dir / "val.txt")
    save_split_txt(test_df,  splits_dir / "test.txt")

    print("\n划分完成")


if __name__ == "__main__":
    main()