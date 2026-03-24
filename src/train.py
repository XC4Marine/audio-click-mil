import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm, trange
from src.dataset import MILBagDataset
from src.model import MILModel
from src.loss import mil_loss
from torch.utils.tensorboard import SummaryWriter

# =========================
# 5-fold划分（按文件）
# =========================
def build_folds(csv_path, n_splits=5, seed=42):
    df = pd.read_csv(csv_path)

    files = df["audio_file"].unique().tolist()
    np.random.seed(seed)
    np.random.shuffle(files)

    folds = np.array_split(files, n_splits)

    return folds


# =========================
# 根据文件筛选索引
# =========================
def get_indices(dataset, file_list):
    indices = []
    for i, s in enumerate(dataset.samples):
        name = s["audio_file"] + ".wav"
        if name in file_list:
            indices.append(i)
    return indices


# =========================
# 单个epoch训练
# =========================
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc="Training", leave=False)

    for x, s, y in pbar:
        x, s, y = x.to(device), s.to(device), y.to(device)

        optimizer.zero_grad()

        y_pred, A, H = model(x, s)

        loss = mil_loss(y_pred, y, A, H)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 👉 动态显示loss
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}"
        })

    return total_loss / len(loader)


# =========================
# 验证
# =========================
def evaluate(model, loader, device):
    model.eval()

    y_true, y_pred = [], []

    pbar = tqdm(loader, desc="Validation", leave=False)

    with torch.no_grad():
        for x, s, y in pbar:
            x, s = x.to(device), s.to(device)

            pred, _, _ = model(x, s)

            y_true.extend(y.numpy())
            y_pred.extend(pred.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ===== 指标 =====
    pred_label = (y_pred > 0.5).astype(int)

    TP = np.sum((pred_label == 1) & (y_true == 1))
    FP = np.sum((pred_label == 1) & (y_true == 0))
    FN = np.sum((pred_label == 0) & (y_true == 1))
    TN = np.sum((pred_label == 0) & (y_true == 0))

    recall = TP / (TP + FN + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    fdr = 1 - precision
    acc = (TP + TN) / (TP + TN + FP + FN + 1e-6)

    return recall, fdr, acc


# =========================
# 主训练（5-fold）
# =========================
def run_training(root_dir):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MILBagDataset(root_dir)

    folds = build_folds(os.path.join(root_dir, "all_bags.csv"))

    
    for fold_idx in range(5):
        print(f"\n===== Fold {fold_idx} =====")
        writer = SummaryWriter(log_dir=f"logs/fold_{fold_idx}")

        val_files = folds[fold_idx]
        train_files = np.concatenate([f for i, f in enumerate(folds) if i != fold_idx])

        train_idx = get_indices(dataset, train_files)
        val_idx = get_indices(dataset, val_files)

        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=4,
            shuffle=True,
            num_workers=4
        )

        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=4,
            shuffle=False,
            num_workers=4
        )

        model = MILModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        for epoch in trange(30, desc=f"Fold {fold_idx}"):

            train_loss = train_one_epoch(model, train_loader, optimizer, device)

            recall, fdr, acc = evaluate(model, val_loader, device)

            
            # ===== TensorBoard记录 =====
            global_step = fold_idx * 1000 + epoch

            writer.add_scalar("Loss/train", train_loss, global_step)
            writer.add_scalar("Metrics/Recall", recall, global_step)
            writer.add_scalar("Metrics/FDR", fdr, global_step)
            writer.add_scalar("Metrics/Accuracy", acc, global_step)

            # ⭐ 关键：alpha
            writer.add_scalar("Attention/alpha", model.attn.alpha.item(), global_step)

            print(f"Epoch {epoch} | Loss {train_loss:.4f} | Recall {recall:.4f} | FDR {fdr:.4f}")


        writer.close()
            