import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
#  Helpers
# -----------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (1‑D)."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C)"""
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0).to(x.dtype)


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, drop: float = 0.0):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Standard Transformer encoder block with LayerNorm‑Pre."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=drop)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.drop_path(attn_out)
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PyramidStage(nn.Module):
    """Group of Transformer blocks + optional temporal down‑sampling."""

    def __init__(self, dim: int, depth: int, num_heads: int, downsample: bool):
        super().__init__()
        self.blocks = nn.Sequential(*[TransformerBlock(dim, num_heads) for _ in range(depth)])
        self.downsample = nn.AvgPool1d(kernel_size=2, stride=2) if downsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x = self.blocks(x)
        if self.downsample is not None:
            # permute for pooling: (B,C,T)
            x = x.permute(0, 2, 1)
            x = self.downsample(x)
            x = x.permute(0, 2, 1)
        return x


# -----------------------------------------------------------------------------
#  GaitPT‑S implementation (Tiny/Small)
# -----------------------------------------------------------------------------
class GaitPTS(nn.Module):
    """Pyramid Transformer backbone for gait recognition on 2‑D key‑points.

    Expected input: (B, T, J*2) after flatten.  J = num joints (12‑17).
    Returns: (B, C) sequence‑level embedding (average pooled).
    """

    def __init__(
        self,
        num_joints: int = 12,
        embed_dim: int = 256,
        depth_per_stage: List[int] = [2, 2, 2, 2],
        num_heads: int = 4,
        tiny: bool = False,
    ):
        super().__init__()
        if tiny:
            embed_dim = embed_dim // 2  # 128‑D for Tiny
            num_heads = max(1, num_heads // 2)

        self.proj = nn.Linear(num_joints * 2, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim, max_len=64)

        stages = []
        for s, depth in enumerate(depth_per_stage):
            stages.append(
                PyramidStage(
                    dim=embed_dim,
                    depth=depth,
                    num_heads=num_heads,
                    downsample=(s < len(depth_per_stage) - 1),
                )
            )
        self.stages = nn.ModuleList(stages)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape  (B, T, J*2)"""
        x = self.proj(x)                  # (B, T, C)
        x = self.pos_enc(x)
        for stage in self.stages:
            x = stage(x)                  # (B, T', C)
        x = self.norm(x)
        # Aggregate sequence → embedding
        emb = x.mean(dim=1)              # (B, C)
        return emb