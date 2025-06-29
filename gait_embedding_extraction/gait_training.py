"""
gait_training.py
----------------
Vanilla‑PyTorch training scaffolding for the GaitPT backbone + ArcFace head.

Riproduce la logica dei file Lightning (ottimizzatore, scheduler, metrica EER)
ma con un loop manuale in stile `gait_embedding_extraction_train.py`.
"""

import math
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# --- project imports ----------------------------------------------------- #
from gait_embedding_extraction.metrics import compute_eer_far

# EarlyStopping & Checkpoint custom (stessi che usi altrove)
from gait_embedding_extraction.model_class.gait_model import GaitModel
from gait_embedding_extraction.checkpoint_classes.early_stopping import EarlyStopping
from gait_embedding_extraction.checkpoint_classes.model_checkpoint import ModelCheckpoint

# ------------------------------------------------------------------------ #
#  Factory
# ------------------------------------------------------------------------ #

def build_model(cfg: Dict, device: torch.device) -> nn.Module:
    model = GaitModel(cfg)
    model.to(device)
    return model

# ------------------------------------------------------------------------ #
#  Optimizer & scheduler
# ------------------------------------------------------------------------ #

def prepare_training_process(cfg: Dict, model: nn.Module, train_loader: DataLoader):
    optim_cfg = cfg["optim"]
    sched_cfg = cfg["scheduler"]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optim_cfg.get("lr", 3e-4),
        weight_decay=optim_cfg.get("weight_decay", 1e-4),
    )

    # Cosine with warm‑up implemented via LambdaLR
    if sched_cfg.get("name", "cosine") == "cosine":
        total_steps = len(train_loader) * cfg["trainer"]["epochs"]
        warm_steps = int(total_steps * sched_cfg.get("warmup_epochs", 10) / cfg["trainer"]["epochs"])

        def lr_lambda(step):
            if step < warm_steps:
                return step / float(max(1, warm_steps))
            progress = (step - warm_steps) / float(max(1, total_steps - warm_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = None

    return optimizer, scheduler

# ------------------------------------------------------------------------ #
#  Epoch loops
# ------------------------------------------------------------------------ #

def train_one_epoch(cfg: Dict, model: nn.Module, loader: DataLoader, optimizer, scheduler,
                    device: torch.device, epoch: int, comet_exp=None) -> Dict:
    model.train()
    run_loss = run_ce = run_center = 0.0
    correct = total = 0

    for step, batch in enumerate(loader):
        seq = batch["keypoints_sequence"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        emb, logits, c_loss = model(seq, labels)
        ce = model.ce_loss(logits, labels)
        loss = ce + c_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        run_loss += loss.item()
        run_ce += ce.item()
        run_center += c_loss.item()
        pred = logits.argmax(dim=1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    metrics = {
        "loss": run_loss / len(loader),
        "ce_loss": run_ce / len(loader),
        "center_loss": run_center / len(loader),
        "accuracy": 100.0 * correct / total,
    }
    if comet_exp:
        comet_exp.log_metrics({f"train_{k}": v for k, v in metrics.items()}, epoch=epoch)
    return metrics


@torch.no_grad()
def evaluate(cfg: Dict, model: nn.Module, loader: DataLoader, device: torch.device,
             epoch: int, comet_exp=None) -> Dict:
    model.eval()
    val_loss = 0.0
    emb_list: List[torch.Tensor] = []
    lab_list: List[torch.Tensor] = []
    correct = total = 0

    for batch in loader:
        seq = batch["keypoints_sequence"].to(device)
        labels = batch["label"].to(device)
        emb, logits, c_loss = model(seq, labels)
        ce = model.ce_loss(logits, labels)
        loss = ce + c_loss
        val_loss += loss.item()
        emb_list.append(F.normalize(emb, dim=1).cpu())
        lab_list.append(labels.cpu())
        pred = logits.argmax(dim=1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    embeddings = torch.cat(emb_list, dim=0)
    labels = torch.cat(lab_list, dim=0)
    eer, far1 = compute_eer_far(embeddings, labels, far_target=0.01)

    metrics = {
        "loss": val_loss / len(loader),
        "accuracy": 100.0 * correct / total,
        "EER": eer,
        "FAR@1%": far1,
    }
    if comet_exp:
        comet_exp.log_metrics({f"val_{k}": v for k, v in metrics.items()}, epoch=epoch)
    return metrics

# ------------------------------------------------------------------------ #
#  Training loop
# ------------------------------------------------------------------------ #

def training_loop(cfg: Dict, model: nn.Module, device: torch.device,
                  train_loader: DataLoader, val_loader: DataLoader, comet_exp=None):
    optimizer, scheduler = prepare_training_process(cfg, model, train_loader)

    early_stop = EarlyStopping(
        monitor="EER", patience=cfg["trainer"].get("patience", 15),
        lower_is_better=True, verbose=True,
    )
    if cfg["trainer"].get("save_best", True):
        ckpt = ModelCheckpoint(monitor="EER", lower_is_better=True, verbose=True)

    best_metrics = None
    start = time.time()

    for epoch in range(cfg["trainer"]["epochs"]):
        train_met = train_one_epoch(cfg, model, train_loader, optimizer, scheduler, device, epoch, comet_exp)
        val_met = evaluate(cfg, model, val_loader, device, epoch, comet_exp)

        if best_metrics is None or val_met["EER"] < best_metrics["EER"]:
            best_metrics = val_met
        early_stop(val_met, epoch)
        if cfg["trainer"].get("save_best", True):
            ckpt(val_met, model)

        elapsed = time.time() - start
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        print(f"Epoch {epoch+1}/{cfg['trainer']['epochs']} | "
              f"Train Acc {train_met['accuracy']:.1f}% | Val Acc {val_met['accuracy']:.1f}% | "
              f"EER {val_met['EER']:.4f} | FAR1 {val_met['FAR@1%']:.4f} | "
              f"Time {h}:{m:02d}:{s:02d}")

        if early_stop.early_stop:
            break

    if cfg["trainer"].get("save_best", True):
        model.load_state_dict(ckpt.best_model_state)
    return model, best_metrics
