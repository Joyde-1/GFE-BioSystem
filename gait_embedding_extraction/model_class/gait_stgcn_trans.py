import torch
import torch.nn as nn
import torch.nn.functional as F


def get_coco_adjacency_matrix(norm=True, include_self=True):
    """
    Returns normalized adjacency matrix for COCO-17 skeleton graph.
    """
    # COCO-17 keypoints and skeleton connections (0-based)
    edges = [
        (0,1),(0,2),(1,3),(2,4),
        (5,6),(5,11),(6,12),(11,12),
        (5,7),(7,9),(6,8),(8,10),
        (11,13),(13,15),(12,14),(14,16)
    ]
    num_nodes = 17
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for i,j in edges:
        A[i,j] = 1
        A[j,i] = 1
    if include_self:
        A += torch.eye(num_nodes)
    if norm:
        D = torch.diag(torch.pow(A.sum(dim=1), -0.5))
        A = D @ A @ D
    return A


class SpatialGCN(nn.Module):
    """
    Spatial graph convolution: aggregates neighbor features via adjacency A.
    Input shape: B x C x T x N
    Output shape: B x C_out x T x N
    """
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.A = A  # Expecting tensor [N,N]
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # x: B x C x T x N
        B, C, T, N = x.shape
        # spatial aggregation: x * A
        x = torch.einsum('bctn,nm->bctm', x, self.A.to(x.device))
        x = self.conv(x)
        return self.bn(x)


class TemporalConv(nn.Module):
    """
    Temporal convolution along the time dimension.
    Input/output: B x C x T x N
    """
    def __init__(self, channels, kernel_size=9, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(channels, channels, (kernel_size, 1),
                              padding=(padding,0), stride=(stride,1))
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        x = self.conv(x)
        return self.bn(x)


class STGCNBlock(nn.Module):
    """
    One block of ST-GCN: SpatialGCN -> ReLU -> TemporalConv -> ReLU, with residual.
    """
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super().__init__()
        self.gcn = SpatialGCN(in_channels, out_channels, A)
        self.tcn = TemporalConv(out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride,1)),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.relu(x)
        x = self.tcn(x)
        x = x + res
        return self.relu(x)


class GaitSTGCNTrans(nn.Module):
    """
    ST-GCN backbone with a self-attention module and embedding head.
    Input: B x T x N x D  (D=2 or 3)
    Output: B x emb_dim (normalized)
    """
    def __init__(self, A, in_channels=2, base_channels=64,
                 num_layers=3, emb_dim=128, num_heads=4):
        super().__init__()
        self.A = A
        # initial projection from D->base_channels
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(0))
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=1)
        # ST-GCN blocks
        self.stgcn_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.stgcn_blocks.append(
                STGCNBlock(
                    in_channels=base_channels if i>0 else base_channels,
                    out_channels=base_channels,
                    A=A,
                    stride=1,
                    residual=True
                )
            )
        # multi-head self-attention on time dimension
        self.attn = nn.MultiheadAttention(embed_dim=base_channels, num_heads=num_heads)
        # embedding projection
        self.fc = nn.Linear(base_channels, emb_dim, bias=False)
        self.bn = nn.BatchNorm1d(emb_dim)

    def forward(self, x):
        # x: B x T x N x D
        B, T, N, D = x.shape
        # reshape for BatchNorm1d
        x = x.view(B, T * N * D).unsqueeze(-1)
        x = self.data_bn(x).squeeze(-1)
        x = x.view(B, T, N, D)
        # permute to B x D x T x N
        x = x.permute(0,3,1,2).contiguous()
        # init conv D->C
        x = self.init_conv(x)
        # ST-GCN backbone
        for blk in self.stgcn_blocks:
            x = blk(x)
        # x: B x C x T x N
        C = x.size(1)
        # prepare for attention: (T, B*N, C)
        xt = x.permute(2,0,3,1).reshape(T, B*N, C)
        xt2, _ = self.attn(xt, xt, xt)
        # reshape back to B x C x T x N
        x2 = xt2.reshape(T, B, N, C).permute(1,3,0,2)
        # global average pooling over time and nodes
        feat = x2.mean(dim=[2,3])  # B x C
        # projection + norm
        emb = self.fc(feat)
        emb = self.bn(emb)
        emb = F.normalize(emb, dim=1)
        return emb


# Example usage:
# A = get_coco_adjacency_matrix()
# model = GaitSTGCNTrans(A)
# inp = torch.randn(8, 60, 17, 2)  # batch of 8 sequences, 60 frames, 17 keypoints, 2 coords
# out = model(inp)
# print(out.shape)  # should be [8, 128]
