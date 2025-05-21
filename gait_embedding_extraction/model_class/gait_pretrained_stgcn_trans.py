import torch
import torch.nn as nn

from gait_stgcn_trans import get_coco_adjacency_matrix, GaitSTGCNTrans


def build_pretrained_gait_model(pretrained_ckpt: str,
                                in_channels: int = 2,
                                base_channels: int = 64,
                                num_layers: int = 3,
                                emb_dim: int = 128,
                                num_heads: int = 4,
                                device: str = 'cpu') -> GaitSTGCNTrans:
    """
    Creates a GaitSTGCNTrans model, loads pretrained ST-GCN weights,
    and freezes the ST-GCN backbone parameters. Only the attention +
    embedding head layers remain trainable.

    Args:
        pretrained_ckpt: Path to the pretrained ST-GCN state_dict (.pth or .pt).
        in_channels: Number of input channels (2 for x,y or 3 with confidence).
        base_channels: Feature dimension in ST-GCN blocks.
        num_layers: Number of ST-GCN blocks.
        emb_dim: Output embedding dimension.
        num_heads: Number of heads in the self-attention module.
        device: Compute device ('cpu' or 'cuda' or 'mps').

    Returns:
        model: Instance of GaitSTGCNTrans with pretrained backbone frozen.
    """
    # 1) Build new model
    A = get_coco_adjacency_matrix()
    model = GaitSTGCNTrans(A,
                             in_channels=in_channels,
                             base_channels=base_channels,
                             num_layers=num_layers,
                             emb_dim=emb_dim,
                             num_heads=num_heads)
    model.to(device)

    # 2) Load pretrained weights
    ckpt = torch.load(pretrained_ckpt, map_location=device)
    # Extract only ST-GCN backbone params: init_conv + stgcn_blocks
    pretrained_dict = {k: v for k, v in ckpt.items()
                       if k.startswith('init_conv.') or k.startswith('stgcn_blocks.')}
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # 3) Freeze backbone layers
    for name, param in model.named_parameters():
        if name.startswith('init_conv') or name.startswith('stgcn_blocks'):
            param.requires_grad = False

    return model


# Example usage
if __name__ == '__main__':
    PRETRAINED_PATH = 'path/to/stgcn_pretrained.pth'
    model = build_pretrained_gait_model(
        pretrained_ckpt=PRETRAINED_PATH,
        in_channels=2,
        base_channels=64,
        num_layers=3,
        emb_dim=128,
        num_heads=4,
        device='cpu'
    )
    # Verify trainable params
    trainable = [n for n,p in model.named_parameters() if p.requires_grad]
    print(f"Trainable parameters ({len(trainable)}):")
    for n in trainable:
        print('  -', n)
