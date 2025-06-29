import torch
import torch.nn as nn

from gait_embedding_extraction.model_class.gaitpt import GaitPTS
from gait_embedding_extraction.model_class.embedding_head import EmbeddingHead


# ------------------------------------------------------------------------ #
#  Model wrapper
# ------------------------------------------------------------------------ #
class GaitModel(nn.Module):
    """Backbone (GaitPTS) + EmbeddingHead → embeddings & logits."""

    def __init__(self, gait_embedding_extraction_config):
        super().__init__()
        self.backbone = GaitPTS(
            num_joints=gait_embedding_extraction_config.model.backbone.num_joints,
            embed_dim=gait_embedding_extraction_config.model.backbone.embed_dim,
            num_heads=gait_embedding_extraction_config.model.backbone.num_heads,
            tiny=gait_embedding_extraction_config.model.backbone.tiny
        )
        embed_dim = self.backbone.proj.out_features
        self.head = EmbeddingHead(
            in_dim=embed_dim,
            num_classes=gait_embedding_extraction_config.data.num_classes,
            s=gait_embedding_extraction_config.model.head.scale,
            m=gait_embedding_extraction_config.model.head.margin,
            lambda_center=gait_embedding_extraction_config.model.head.lambda_center,
        )
        self.ce_loss = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    def forward(self, seq: torch.Tensor, labels: torch.Tensor = None):
        """If `labels` is None → inference (no center loss)."""
        emb = self.backbone(seq)
        # If we're debugging with a plain nn.Linear head, it only expects
        # the embeddings tensor, not labels.
        if isinstance(self.head, nn.Linear):
            logits = self.head(emb)
            return emb, logits, torch.zeros(1, device=seq.device, dtype=emb.dtype)
        if labels is None:
            logits, _ = self.head(emb)
            return emb, logits, torch.zeros(1, device=seq.device)
        logits, c_loss = self.head(emb, labels)
        return emb, logits, c_loss