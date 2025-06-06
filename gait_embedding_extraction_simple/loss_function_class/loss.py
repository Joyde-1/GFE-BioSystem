import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """
    Implementazione di ArcFace Loss.
    - embeddings: tensor (B, emb_dim), L2-normalizzati
    - labels:     tensor (B,) con etichette zero-based [0..num_classes-1]
    - weight:     parametro del classificatore, tensor (num_classes, emb_dim)
    Restituisce la CrossEntropyLoss applicata sui logit angolari con margine.
    """
    def __init__(self, scale: float = 30.0, margin: float = 0.50):
        super().__init__()
        self.s = scale
        self.m = margin
        # Pre-calcoliamo cos(m) e sin(m) come tensori
        self.cos_m = torch.cos(torch.tensor(margin))
        self.sin_m = torch.sin(torch.tensor(margin))
        # soglia per stabilità numerica: cos(pi - m)
        self.th = torch.cos(torch.pi - torch.tensor(margin))
        # mm = sin(pi - m) * m, usato in casi borderline
        self.mm = torch.sin(torch.pi - torch.tensor(margin)) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor, weight: nn.Parameter) -> torch.Tensor:
        """
        Args:
            embeddings: tensor di shape (B, emb_dim), L2-normalizzati
            labels:     tensor di shape (B,), dtype long, zero-based
            weight:     parametro del classificatore, tensor di shape (num_classes, emb_dim)
                        (non normalizzato)
        Returns:
            loss: scalar tensor
        """
        device = embeddings.device
        # Normalizziamo i pesi della testa: (num_classes, emb_dim)
        W = F.normalize(weight, p=2, dim=1).to(device)

        # Calcoliamo cosθ (B, num_classes): ogni elemento = <x_i, W_j>
        cos_theta = F.linear(embeddings, W)  # (B, num_classes)
        cos_theta = cos_theta.clamp(-1.0, 1.0)  # stabilità numerica

        # Estraiamo cosθ per la classe corretta y: (B,)
        idx = torch.arange(0, embeddings.size(0), device=device)
        cos_theta_y = cos_theta[idx, labels]  # (B,)

        # Calcoliamo sinθ_y = sqrt(1 - cosθ_y^2)
        sin_theta_y = torch.sqrt(1.0 - cos_theta_y.pow(2) + 1e-9)

        # Calcoliamo cos(θ_y + m) = cosθ_y * cos(m) - sinθ_y * sin(m)
        cos_m = self.cos_m.to(device)
        sin_m = self.sin_m.to(device)
        cos_theta_y_m = cos_theta_y * cos_m - sin_theta_y * sin_m

        # In casi borderline, se cosθ_y < th, usiamo cosθ_y - mm
        th = self.th.to(device)
        mm = self.mm.to(device)
        cond = cos_theta_y < th
        cos_theta_y_m = torch.where(cond, cos_theta_y - mm, cos_theta_y_m)

        # Costruiamo i logit finali: per tutte le classi cosθ, per y usiamo cosθ_y_m
        logits = cos_theta.clone()  # (B, num_classes)
        logits[idx, labels] = cos_theta_y_m

        # Applichiamo lo scale
        logits = logits * self.s

        # Calcoliamo la CrossEntropyLoss
        loss = F.cross_entropy(logits, labels)
        return loss


class BatchHardTripletLoss(nn.Module):
    """
    Batch-hard triplet loss:
    - Per ogni ancora 'i', seleziona il positive 'j' con distanza massima (hardest positive)
      e il negative 'k' con distanza minima (hardest negative) all'interno del batch.
    - Loss = mean_i max(0, margin + d(i, hardest_pos) - d(i, hardest_neg))
    """
    def __init__(self, margin: float = 0.3, squared: bool = False):
        super().__init__()
        self.margin = margin
        self.squared = squared  # se True, usa distanza al quadrato

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: tensor di shape (B, emb_dim)
            labels:     tensor di shape (B,), dtype long, zero-based
        Returns:
            loss: scalar tensor
        """
        device = embeddings.device
        B = embeddings.size(0)

        # # Calcoliamo la matrice di distanza pairwise (B, B)
        # if self.squared:
        #     dist_matrix = torch.cdist(embeddings, embeddings, p=2.0).pow(2)
        # else:
        #     dist_matrix = torch.cdist(embeddings, embeddings, p=2.0)

        # ======================
        # Calcolo della matrice di distanza pairwise, con fallback a CPU se siamo su MPS
        # ======================
        if embeddings.device.type == 'mps':
            # Sposto gli embedding su CPU per il cdist
            emb_cpu = embeddings.cpu()
            if self.squared:
                dist_matrix = torch.cdist(emb_cpu, emb_cpu, p=2.0).pow(2).to(embeddings.device)
            else:
                dist_matrix = torch.cdist(emb_cpu, emb_cpu, p=2.0).to(embeddings.device)
        else:
            # Caso normale (CUDA o CPU)
            if self.squared:
                dist_matrix = torch.cdist(embeddings, embeddings, p=2.0).pow(2)
            else:
                dist_matrix = torch.cdist(embeddings, embeddings, p=2.0)

        # Creiamo maschera per pos e neg: (B, B)
        labels = labels.unsqueeze(1)  # (B, 1)
        mask_positive = labels.eq(labels.t())  # True se stessa classe
        mask_negative = ~mask_positive         # True se classe diversa

        # Hardest positive per ogni i: massima distanza tra i e j con stessa label, j != i
        inf = torch.tensor(float('inf'), device=device)
        dist_pos = dist_matrix.clone()
        # Invalidate negativi e diagonale
        dist_pos[~mask_positive] = -inf
        diag_idx = torch.arange(B, device=device)
        dist_pos[diag_idx, diag_idx] = -inf
        hardest_pos, _ = dist_pos.max(dim=1)  # (B,)

        # Hardest negative per ogni i: minima distanza tra i e k con label diversa
        dist_neg = dist_matrix.clone()
        dist_neg[~mask_negative] = inf
        hardest_neg, _ = dist_neg.min(dim=1)  # (B,)

        # Calcoliamo la triplet loss: max(0, margin + d_pos - d_neg)
        loss_vals = F.relu(self.margin + hardest_pos - hardest_neg)
        loss = loss_vals.mean()
        return loss