# Dataset
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MILBagDataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir: D:\Project_Github\audio_click_mil\processed_data
        """
        self.root_dir = root_dir

        self.mfcc_dir = os.path.join(root_dir, "mfcc")
        self.bag_csv = pd.read_csv(os.path.join(root_dir, "all_bags.csv"))
        self.prior_csv = pd.read_csv(os.path.join(root_dir, "instance_prior.csv"))

        # ===== 构建索引 =====
        self.samples = []

        for _, row in self.bag_csv.iterrows():
            self.samples.append({
                "audio_file": row["audio_file"].replace(".wav", ""),
                "bag_idx": int(row["bag_idx"]),
                "label": int(row["label"]),
                "file_num": int(row["file_num"])
            })

        # ===== 构建 prior 索引（加速）=====
        self.prior_group = self.prior_csv.groupby(["file_num", "bag_idx"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        audio_file = sample["audio_file"]
        bag_idx = sample["bag_idx"]
        label = sample["label"]
        file_num = sample["file_num"]

        # ===== 1. 读取 MFCC =====
        npy_path = os.path.join(
            self.mfcc_dir,
            audio_file,
            f"bag_{bag_idx:03d}.npy"
        )

        x = np.load(npy_path)  # (60,128,128)

        # ===== 安全检查 =====
        if x.shape != (60, 128, 128):
            raise ValueError(f"Shape error: {npy_path} -> {x.shape}")

        # ===== 加channel维度 =====
        x = np.expand_dims(x, axis=1)  # (60,1,128,128)

        # ===== 2. 读取 prior =====
        prior_rows = self.prior_group.get_group((file_num, bag_idx))
        prior_rows = prior_rows.sort_values("instance_idx")

        s = prior_rows["prior_score"].values.astype(np.float32)  # (60,)

        if len(s) != 60:
            raise ValueError(f"Prior length error: file {file_num}, bag {bag_idx}")

        s = np.expand_dims(s, axis=1)  # (60,1)

        # ===== 3. label =====
        y = np.array([label], dtype=np.float32)

        # ===== 转tensor =====
        x = torch.from_numpy(x).float()
        s = torch.from_numpy(s).float()
        y = torch.from_numpy(y).float()

        return x, s, y