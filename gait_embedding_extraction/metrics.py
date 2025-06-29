# import torch
# from typing import Tuple


# def _pairwise_cosine(emb: torch.Tensor) -> torch.Tensor:
#     """Compute pairwise cosine similarity matrix (N, N)."""
#     emb = torch.nn.functional.normalize(emb, dim=1)
#     return emb @ emb.t()


# def compute_eer_far(embeddings: torch.Tensor, labels: torch.Tensor, far_target: float = 0.01) -> Tuple[float, float]:
#     """Compute EER and FAR@far_target (e.g. 0.01 -> 1%).

#     Parameters
#     ----------
#     embeddings : Tensor (N, C) normalized
#     labels : Tensor (N,)
#     far_target : float
#         False Accept Rate target for which FAR@target is returned.

#     Returns
#     -------
#     eer : float
#     far_at_target : float
#     """
#     device = embeddings.device
#     with torch.no_grad():
#         sims = _pairwise_cosine(embeddings)  # (N,N)
#         N = sims.size(0)
#         # Create masks
#         labels = labels.view(-1, 1)
#         same = labels.eq(labels.t())
#         mask_triu = torch.triu(torch.ones_like(same, dtype=torch.bool), diagonal=1)
#         same = same & mask_triu
#         diff = (~same) & mask_triu

#         genuine_scores = sims[same].view(-1)
#         imposter_scores = sims[diff].view(-1)

#         # Concatenate all scores and labels (1 genuine, 0 imposter)
#         scores = torch.cat([genuine_scores, imposter_scores])
#         y = torch.cat([torch.ones_like(genuine_scores), torch.zeros_like(imposter_scores)])

#         # Sort scores descending
#         sorted_scores, idx = torch.sort(scores, descending=True)
#         y_sorted = y[idx]

#         # Cumulative sums for FAR/FRR calculations
#         cum_true = torch.cumsum(y_sorted, dim=0)
#         cum_false = torch.cumsum(1 - y_sorted, dim=0)
#         total_true = genuine_scores.numel()
#         total_false = imposter_scores.numel()

#         frr = (total_true - cum_true).float() / total_true  # False Rejection Rate
#         far = cum_false.float() / total_false              # False Accept Rate

#         # EER: point where |FAR - FRR| minimal
#         diff = torch.abs(far - frr)
#         min_idx = torch.argmin(diff)
#         eer = (far[min_idx] + frr[min_idx]) / 2.0

#         # FAR at given threshold (find first far <= target)
#         try:
#             thr_idx = torch.where(far <= far_target)[0][0]
#             far_at_target = far[thr_idx].item()
#         except IndexError:
#             far_at_target = far[-1].item()

#         return eer.item(), far_at_target





import torch
from typing import Tuple


def _pairwise_cosine(x: torch.Tensor) -> torch.Tensor:
    x = torch.nn.functional.normalize(x, dim=1)
    return x @ x.t()


def compute_eer_far(emb: torch.Tensor, labels: torch.Tensor, far_target: float = 0.01) -> Tuple[float, float]:
    """Compute EER and FAR@target for embeddings.

    Args
    ----
    emb : (N, C) tensor normalized or not (will be normalized here)
    labels : (N,) int tensor
    far_target : e.g. 0.01 → FAR at 1 %
    """
    emb = torch.nn.functional.normalize(emb, dim=1)
    sim = _pairwise_cosine(emb)

    N = sim.size(0)
    triu_mask = torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)
    label_eq = labels.unsqueeze(1).eq(labels.unsqueeze(0)) & triu_mask

    genuine = sim[label_eq]
    imposter = sim[~label_eq & triu_mask]

    scores = torch.cat([genuine, imposter])
    y_true = torch.cat([torch.ones_like(genuine), torch.zeros_like(imposter)])

    sorted_scores, idx = torch.sort(scores, descending=True)
    y_sorted = y_true[idx]

    # cumulative
    cum_true = torch.cumsum(y_sorted, 0)
    cum_false = torch.cumsum(1 - y_sorted, 0)
    total_true = genuine.numel()
    total_false = imposter.numel()

    frr = (total_true - cum_true).float() / total_true
    far = cum_false.float() / total_false

    # EER
    diff = torch.abs(far - frr)
    eer_idx = torch.argmin(diff)
    eer = ((far[eer_idx] + frr[eer_idx]) / 2).item()

    # FAR @ target
    try:
        thr_idx = (far <= far_target).nonzero(as_tuple=True)[0][0]
        far_at = far[thr_idx].item()
    except IndexError:
        far_at = far[-1].item()

    return eer, far_at
