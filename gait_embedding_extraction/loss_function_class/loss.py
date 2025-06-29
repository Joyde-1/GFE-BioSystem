import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HardTripletLoss(nn.Module):
    """Hard Triplet Loss con mining automatico"""
    def __init__(self, margin=0.5):
        super(HardTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        # Calcola matrice di distanze
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
        
        # Maschera per positive pairs (stessa classe)
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_not_equal = ~labels_equal
        
        # Per ogni anchor, trova hardest positive e hardest negative
        losses = []
        
        for i in range(embeddings.size(0)):
            # Positive distances (escludi se stesso)
            pos_mask = labels_equal[i].clone()
            pos_mask[i] = False
            
            if pos_mask.sum() == 0:
                continue
                
            pos_dists = pairwise_dist[i][pos_mask]
            hardest_pos_dist = pos_dists.max()
            
            # Negative distances
            neg_mask = labels_not_equal[i]
            if neg_mask.sum() == 0:
                continue
                
            neg_dists = pairwise_dist[i][neg_mask]
            hardest_neg_dist = neg_dists.min()
            
            # Triplet loss
            loss = F.relu(hardest_pos_dist - hardest_neg_dist + self.margin)
            losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            
        return torch.stack(losses).mean()


class CenterLoss(nn.Module):
    """Center Loss per migliorare la compattezza intra-classe"""
    def __init__(self, num_classes, embedding_dim, alpha=0.5):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        
        # Centri delle classi (learnable parameters)
        self.centers = nn.Parameter(torch.randn(num_classes, embedding_dim))
        
    def forward(self, embeddings, labels):
        batch_size = embeddings.size(0)
        
        # Seleziona i centri corrispondenti alle labels
        centers_batch = self.centers[labels]  # (N, embedding_dim)
        
        # Calcola la distanza dai centri
        center_loss = F.mse_loss(embeddings, centers_batch)
        
        return center_loss


class CombinedLoss(nn.Module):
    """Loss combinata migliorata"""
    def __init__(self, num_classes, embedding_dim, margin=0.5, alpha=0.7, beta=0.1):
        super(CombinedLoss, self).__init__()
        self.hard_triplet_loss = HardTripletLoss(margin)
        self.center_loss = CenterLoss(num_classes, embedding_dim)
        self.ce_loss = nn.CrossEntropyLoss()
        self.alpha = alpha  # peso triplet loss
        self.beta = beta    # peso center loss

    def forward(self, embeddings, logits, labels, triplet_data=None):
        # CrossEntropy Loss
        ce_loss = self.ce_loss(logits, labels)
        
        # Hard Triplet Loss
        triplet_loss = self.hard_triplet_loss(embeddings, labels)
        
        # Center Loss
        center_loss = self.center_loss(embeddings, labels)
        
        # Loss combinata
        total_loss = (1 - self.alpha - self.beta) * ce_loss + \
                     self.alpha * triplet_loss + \
                     self.beta * center_loss

        return total_loss, ce_loss, triplet_loss