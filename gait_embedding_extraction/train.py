import argparse
import os
from pathlib import Path
from typing import Dict

import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from data.dataset import GaitEmbeddingExtractionDataset
from data.augmentations import default_augmentation
from lightning.module import GaitLitModule
from utils.seed import set_global_seed

# -----------------------------------------------------------------------------
#  Utility functions
# -----------------------------------------------------------------------------

def load_yaml(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg: Dict, fold_idx: int = 0):
    """Very simple subject-wise split: 80/20 using modulo on labels.
    In production you'd load predefined folds.
    """
    data_np = torch.load(cfg["dataset"]["path"])  # expects dict with sequences & labels

    seqs = data_np["keypoints_sequences"]
    labels = data_np["labels"]

    # Basic split using modulo for reproducibility
    train_idx = [i for i, lb in enumerate(labels) if lb % 5 != fold_idx]
    val_idx = [i for i, lb in enumerate(labels) if lb % 5 == fold_idx]

    train_set = GaitEmbeddingExtractionDataset(
        {"keypoints_sequences": seqs[train_idx], "labels": labels[train_idx]},
        transform=default_augmentation(cfg["model"]["input_len"]),
    )
    val_set = GaitEmbeddingExtractionDataset(
        {"keypoints_sequences": seqs[val_idx], "labels": labels[val_idx]},
        transform=None,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg["optim"].get("batch_size", 32),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        drop_last=True,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=cfg["optim"].get("batch_size", 32),
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )
    return train_loader, val_loader, len(set(labels))


# -----------------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GaitPT model")
    parser.add_argument("--cfg", type=str, default="config/default.yaml", help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0, help="Cross-val fold index [0-4]")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.cfg))

    set_global_seed(args.seed)

    # ---------------- Dataloaders ----------------
    train_loader, val_loader, num_classes = build_dataloaders(cfg, args.fold)

    # ---------------- Model ---------------------
    lit_model = GaitLitModule(num_classes=num_classes, cfg=cfg)

    # ---------------- Callbacks -----------------
    ckpt_cb = ModelCheckpoint(
        monitor="val/acc",
        mode="max",
        save_top_k=1,
        filename="epoch={epoch}-val_acc={val/acc:.4f}",
    )
    early_cb = EarlyStopping(monitor="val/acc", mode="max", patience=15)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # ---------------- Logger --------------------
    tb_logger = TensorBoardLogger(save_dir="logs", name="gaitpt")

    # ---------------- Trainer -------------------
    trainer = pl.Trainer(
        max_epochs=cfg["trainer"].get("epochs", 120),
        precision=16 if torch.cuda.is_available() or torch.backends.mps.is_available() else 32,
        devices=1,
        accelerator="auto",
        logger=tb_logger,
        callbacks=[ckpt_cb, early_cb, lr_monitor],
        deterministic=True,
    )

    trainer.fit(lit_model, train_loader, val_loader)
