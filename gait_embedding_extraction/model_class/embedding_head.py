import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
#  ArcMarginProduct: implementation of ArcFace head
# -----------------------------------------------------------------------------
class ArcMarginProduct(nn.Module):
    """Implements the multiplicative angular margin (ArcFace).

    Parameters
    ----------
    in_features : int
        Size of the input embedding.
    out_features : int
        Number of classes.
    s : float, default 30.0
        Scale factor. Paper uses 64, but 30 is common for smaller datasets.
    m : float, default 0.50
        Angular margin [rad].
    easy_margin : bool, default False
        Whether to use the easy_margin trick.
    """

    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.50,
                 easy_margin: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # cos(m), sin(m) pre-computed
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute logits with arc margin.

        Parameters
        ----------
        embeddings : Tensor (B, C)
        labels : Tensor (B,) long
        """
        # L2-normalize features and weights
        embeddings = F.normalize(embeddings, dim=1)
        weight = F.normalize(self.weight, dim=1)

        # Cosine similarity between features and class centers
        cos_theta = torch.matmul(embeddings, weight.t())  # (B, out_features)
        # Clamp for numerical stability
        cos_theta = cos_theta.clamp(-1.0, 1.0)

        # Compute phi = cos(theta + m)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        phi = cos_theta * self.cos_m - sin_theta * self.sin_m  # cos(theta+m)

        if self.easy_margin:
            # If easy margin, use phi when cos>0 else keep cos
            phi = torch.where(cos_theta > 0, phi, cos_theta)
        else:
            # Traditional margin
            phi = torch.where(cos_theta > self.th, phi, cos_theta - self.mm)

        # One-hot encode labels
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # Combine: for target class use phi, else cos
        logits = (one_hot * phi) + ((1.0 - one_hot) * cos_theta)
        logits = logits * self.s
        return logits


# -----------------------------------------------------------------------------
#  Center Loss (Wen et al., 2016)
# -----------------------------------------------------------------------------
class CenterLoss(nn.Module):
    """ Center Loss: encourages features of the same class to be close.

    Parameters
    ----------
    num_classes : int
    feat_dim : int  (embedding size)
    lr : float, learning rate multiplier for the centers (optional)
    """

    def __init__(self, num_classes: int, feat_dim: int, lr: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lr = lr  # not used here; kept for compatibility
        # centers: (num_classes, feat_dim)
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # embeddings: (B, feat_dim)
        batch_size = embeddings.size(0)
        # Get centers for each label
        centers_batch = self.centers.index_select(0, labels)
        # Compute distances
        loss = (embeddings - centers_batch).pow(2).sum(dim=1).mean()
        return loss


# -----------------------------------------------------------------------------
#  Wrapper: returns logits + center loss
# -----------------------------------------------------------------------------
class EmbeddingHead(nn.Module):
    """Wrapper that encapsulates ArcFace and optional CenterLoss.

    During training returns (logits, center_loss). During inference returns only logits.
    """

    def __init__(self, in_dim: int, num_classes: int, s: float = 30.0, m: float = 0.50, lambda_center: float = 0.1):
        super().__init__()
        self.arcface = ArcMarginProduct(in_dim, num_classes, s=float(s), m=float(m), easy_margin=True)
        self.center_loss_fn = CenterLoss(num_classes, in_dim) if lambda_center > 0 else None
        self.lambda_center = lambda_center

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return logits and (optional) center loss."""
        if labels is None:
            # Inference mode: just return logits computed with cosine (no margin)
            weight = F.normalize(self.arcface.weight, dim=1)
            logits = torch.matmul(F.normalize(embeddings, dim=1), weight.t()) * self.arcface.s
            return logits, torch.zeros(1, device=embeddings.device, dtype=embeddings.dtype)

        logits = self.arcface(embeddings, labels)

        # ---- CenterLoss ----------------------------------------------------
        # closs = (
        #     self.center_loss_fn(embeddings, labels) * self.lambda_center if self.center_loss_fn is not None else 0.0
        # )
        if self.lambda_center == 0.0 or self.center_loss_fn is None:
            closs = torch.zeros(1, device=embeddings.device, dtype=embeddings.dtype)
        else:
            closs = self.center_loss_fn(embeddings, labels) * self.lambda_center

        return logits, closs