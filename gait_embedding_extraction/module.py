import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy

from models.gaitpt import GaitPTS
from models.embedding_head import EmbeddingHead
from utils.metrics import compute_eer_far


class GaitLitModule(pl.LightningModule):
    """LightningModule that wraps GaitPT backbone + ArcFace head and logs
    classification accuracy *and* biometric metrics (EER, FAR@1%).
    """

    def __init__(self, num_classes: int, cfg: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()

        # -------------------------- Backbone --------------------------- #
        model_cfg = cfg.get("model", {})
        self.backbone = GaitPTS(
            num_joints=model_cfg.get("num_joints", 12),
            embed_dim=model_cfg.get("embed_dim", 256),
            tiny=model_cfg.get("tiny", False),
        )
        embed_dim = self.backbone.proj.out_features

        # -------------------------- Head -------------------------------- #
        head_cfg = model_cfg.get("head", {})
        self.head = EmbeddingHead(
            in_dim=embed_dim,
            num_classes=num_classes,
            s=head_cfg.get("scale", 30.0),
            m=head_cfg.get("margin", 0.50),
            lambda_center=head_cfg.get("lambda_center", 0.10),
        )

        # -------------------------- Losses & Metrics ------------------- #
        self.ce_loss = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.optim_cfg = cfg.get("optim", {})
        self.sched_cfg = cfg.get("scheduler", {})

        # Buffers to collect embeddings during validation for EER/FAR
        self._reset_val_buffers()

    # ------------------------------------------------------------------
    #  Utility
    # ------------------------------------------------------------------
    def _reset_val_buffers(self):
        self.val_embeddings: List[torch.Tensor] = []
        self.val_labels: List[torch.Tensor] = []

    # ------------------------------------------------------------------
    #  Forward (inference)
    # ------------------------------------------------------------------
    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        emb = self.backbone(seq)
        logits, _ = self.head(emb)
        return logits

    # ------------------------------------------------------------------
    #  Training
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        seq, y = batch["keypoints_sequence"], batch["label"]
        emb = self.backbone(seq)
        logits, c_loss = self.head(emb, y)
        ce = self.ce_loss(logits, y)
        loss = ce + c_loss

        # logs
        self.log_dict({"train/loss": loss, "train/ce": ce, "train/center": c_loss},
                      prog_bar=True, on_step=True, logger=True)
        self.train_acc(logits.softmax(dim=-1), y)
        self.log("train/acc", self.train_acc, on_epoch=True, prog_bar=True)
        return loss

    # ------------------------------------------------------------------
    #  Validation
    # ------------------------------------------------------------------
    def on_validation_epoch_start(self):
        self._reset_val_buffers()

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        seq, y = batch["keypoints_sequence"], batch["label"]
        emb = F.normalize(self.backbone(seq), dim=1)  # normalized embeddings
        logits, _ = self.head(emb, y)
        self.val_acc(logits.softmax(dim=-1), y)

        # collect for EER/FAR
        self.val_embeddings.append(emb.detach())
        self.val_labels.append(y.detach())

    def on_validation_epoch_end(self):
        # accuracy
        self.log("val/acc", self.val_acc, prog_bar=True)

        # concatenate
        embeddings = torch.cat(self.val_embeddings, dim=0)
        labels = torch.cat(self.val_labels, dim=0)

        eer, far1 = compute_eer_far(embeddings, labels, far_target=0.01)
        self.log("val/EER", eer, prog_bar=True)
        self.log("val/FAR@1%", far1, prog_bar=False)

    # ------------------------------------------------------------------
    #  Optimizers / Scheduler
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        lr = self.optim_cfg.get("lr", 3e-4)
        wd = self.optim_cfg.get("weight_decay", 1e-4)
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)

        if self.sched_cfg.get("name", "cosine") == "cosine":
            warm = self.sched_cfg.get("warmup_epochs", 10)
            max_ep = self.trainer.max_epochs if self.trainer else 120
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep - warm)
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
            }
        return opt

    # ------------------------------------------------------------------
    #  Prediction — returns normalized embeddings
    # ------------------------------------------------------------------
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0):
        seq = batch["keypoints_sequence"]
        return F.normalize(self.backbone(seq), dim=1)
