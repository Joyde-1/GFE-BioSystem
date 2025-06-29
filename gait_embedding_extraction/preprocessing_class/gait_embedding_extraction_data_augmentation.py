import random
import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d


class Jitter(nn.Module):
    """Gaussian noise on (x, y) coordinates."""

    def __init__(self, sigma: float = 0.005):
        super().__init__()
        self.sigma = sigma

    def forward(self, seq: torch.Tensor) -> torch.Tensor:  # (L, J, 2)
        seq = seq.float()
        return seq + torch.randn_like(seq) * self.sigma

    __call__ = forward


class TimeWarp(nn.Module):
    """Stretch/compress the sequence length, then resample to *target_len*."""

    def __init__(self, min_scale: float = 0.9, max_scale: float = 1.1, target_len: int = 32):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.target_len = target_len

    def forward(self, seq: torch.Tensor) -> torch.Tensor:  # (L, J, 2)
        L, J, _ = seq.shape
        factor = random.uniform(self.min_scale, self.max_scale)
        L_new = max(2, int(L * factor))

        idx_old = np.linspace(0, 1, L, dtype=np.float32)
        idx_new = np.linspace(0, 1, L_new, dtype=np.float32)

        flat = seq.cpu().numpy().reshape(L, -1)  # (L, J*2)
        flat_new = interp1d(idx_old, flat, axis=0)(idx_new)  # (L_new, J*2)

        idx_final = np.linspace(0, 1, self.target_len, dtype=np.float32)
        flat_final = interp1d(idx_new, flat_new, axis=0)(idx_final)
        return torch.from_numpy(flat_final.reshape(self.target_len, J, 2)).float()

    __call__ = forward


class PhaseShift(nn.Module):
    """Circular shift of the start frame (±*max_shift*)."""

    def __init__(self, max_shift: int = 2):
        super().__init__()
        self.max_shift = max_shift

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        seq = seq.float()
        if self.max_shift == 0:
            return seq
        k = random.randint(-self.max_shift, self.max_shift)
        if k == 0:
            return seq
        return torch.roll(seq, shifts=k, dims=0)

    __call__ = forward


def default_augmentation(gait_embedding_extraction_config):
    """Componi la pipeline usata in training."""
    data_augmentation_params = gait_embedding_extraction_config.data.data_augmentation_params
    return torch.nn.Sequential(
        Jitter(data_augmentation_params.sigma),
        # TimeWarp(data_augmentation_params.min_scale, data_augmentation_params.max_scale, gait_embedding_extraction_config.data.fixed_length),
        PhaseShift(data_augmentation_params.shift_max)
    )