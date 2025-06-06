import torch
import torch.nn as nn
import torch.nn.functional as F

# Import del backbone ST-GCN da MMAction2
# (assicurati di avere mmaction2 installato: pip install mmaction2)
from mmaction.models.backbones.stgcn import STGCN


class SpatialAttention(nn.Module):
    """
    Spatial Attention: dato un feature-map (B, C, T, V),
    calcola un peso su ciascun joint (V) per enfatizzare i nodi più discriminativi.
    """
    def __init__(self, in_channels, num_joints=17):
        super().__init__()
        # Un 1x1 conv per mappare da in_channels -> num_joints score (senza bias)
        self.conv1x1 = nn.Conv2d(in_channels, num_joints, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=-1)  # softmax sui V joint

    def forward(self, x):
        # x: (B, C, T, V)
        # applichiamo conv1x1 sui canali C per ottenere (B, V, T, V)
        # poi facciamo softmax lungo l'ultima dim (quella dei joints)
        B, C, T, V = x.size()
        # Score: (B, V, T, V)
        score = self.conv1x1(x)           
        # spostiamo le dimensioni per fare softmax su dim “joint”
        # score_perm: (B, T, V, V)
        score_perm = score.permute(0, 2, 1, 3).contiguous()
        # Softmax sui joint (ultima dim)
        attn = self.softmax(score_perm)   # (B, T, V, V)

        # Ora moltiplichiamo x per la maschera di attenzione
        # Vogliamo un peso per ogni joint “target” basato su tutti i joint “source”
        # In pratica, calcoliamo: x' = attn * x_source (moltiplicazione matmul su dim joint)
        # risolviamo con tensor matmul:
        # x (B, C, T, V)   -> la trattiamo come (B, C, T, V, 1)
        x_ = x.unsqueeze(-1)             # (B, C, T, V, 1)
        # attn: (B, T, V_source, V_target)
        # Permutiamo attn in (B, 1, T, V_source, V_target) per broadcast
        attn_ = attn.unsqueeze(1)        # (B, 1, T, V, V)
        # Poi facciamo matmul su dim V_source:
        # Risultato: (B, C, T, V_target, 1) -> poi squeeze → (B, C, T, V)
        out = torch.matmul(x_, attn_)    # (B, C, T, V, 1)
        out = out.squeeze(-1)            # (B, C, T, V)
        return out
    

class TemporalAttention(nn.Module):
    """
    Temporal Attention: dato un feature-map (B, C, T, V),
    calcola un peso su ciascun istante temporale (T), per enfatizzare i frame più discriminativi.
    """
    def __init__(self, in_channels, num_frames):
        super().__init__()
        # Mappiamo i canali C -> T score, usando conv1x1 sui joints
        # Per semplificare, useremo un 1x1 conv su (C, V) per ottenere (T)
        self.conv1x1 = nn.Conv2d(in_channels, num_frames, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=2)  # softmax lungo la dimensione T

    def forward(self, x):
        # x: (B, C, T, V)
        B, C, T, V = x.size()
        # Applichiamo conv1x1 lungo (C → T) mantenendo la dim V
        # Otteniamo score: (B, T, T, V)
        score = self.conv1x1(x)          # (B, T, T, V)
        # Permutiamo per fare softmax su “T” (le righe corrispondono ai frame)
        score_perm = score.permute(0, 3, 1, 2).contiguous()  # (B, V, T, T)
        attn = self.softmax(score_perm)  # (B, V, T, T)

        # Ora la moltiplicazione: x `(B, C, T, V)`  → espandiamo a `(B, C, T, V, 1)`
        x_ = x.unsqueeze(-1)             # (B, C, T, V, 1)
        # attn: (B, V, T_source, T_target) → vogliamo (B, 1, V, T_source, T_target)
        attn_ = attn.unsqueeze(1)        # (B, 1, V, T, T)
        # Matmul su dim T_source:
        # Risultato: (B, C, T_source, V, 1) → poi squeeze → (B, C, T, V)
        out = torch.matmul(attn_, x_.permute(0, 1, 3, 2, 4))  
        # spieghiamo: 
        # - x_.permute(0,1,3,2,4) = (B, C, V, T, 1)
        # - attn_:              = (B, 1, V, T, T)
        # matmul sul terzo indice (T_source), otteniamo (B, C, V, T, 1)
        out = out.squeeze(-1).permute(0, 1, 3, 2).contiguous()  # torna a (B, C, T, V)
        return out
    

class GaitSTGCNModel(nn.Module):
    """
    Modello principale per l’estrazione degli embedding di gait:
      - Backbone ST-GCN (pre-addestrato su NTU, se richiesto)
      - Blocchi di Spatial + Temporal Attention
      - Transformer Encoder sul tempo
      - Testa lineare finale per ArcFace (49 classi)
    """

    def __init__(self,
                 num_classes: int = 49,
                 pretrained_backbone: bool = True,
                 checkpoint_path: str = None,
                 stgcn_cfg: dict = None,
                 emb_dim: int = 256,
                 transformer_layers: int = 2,
                 transformer_heads: int = 4,
                 transformer_ffn_dim: int = 512,
                 dropout: float = 0.3):
        """
        Args:
            num_classes (int): numero di identità nel tuo dataset (49).
            pretrained_backbone (bool): se True, carica i pesi pre-addestrati su NTU.
            checkpoint_path (str): path al file .pth di un checkpoint ST-GCN pre-addestrato.
                                    Se None, usa i pesi di default (NTU joint modality).
            stgcn_cfg (dict): se vuoi passare una config personalizzata per ST-GCN;
                               se None, usa i valori di default (COCO skeleton, 64 base channels, 10 stage, ecc.).
            emb_dim (int): dimensione finale dell’embedding.
            transformer_layers (int): numero di layer del Transformer Encoder.
            transformer_heads (int): numero di teste di attenzione per il Transformer.
            transformer_ffn_dim (int): dimensione dell’FFN interno al Transformer.
            dropout (float): dropout nel Transformer/FC layer.
        """
        super().__init__()

        # 1. Configurazione del backbone ST-GCN
        # ----------------------------------------------------
        # Se l’utente non fornisce stgcn_cfg, definiamo uno di base per COCO skeleton a 17 joint:
        if stgcn_cfg is None:
            stgcn_cfg = dict(
                layout='coco',         # 17 keypoint COCO
                strategy='spatial',    # convenzione di connettività
            )

        # STGCN: in_channels=3 (x,y,confidence), base_channels=64, 10 stage, 
        #        inflate e down sampling nelle stesse epoche del paper originale.
        self.backbone = STGCN(
            graph_cfg=stgcn_cfg,
            in_channels=3,
            base_channels=64,
            data_bn_type='VC',
            ch_ratio=2,
            num_person=1,        # dato che c’è un solo soggetto
            num_stages=10,
            inflate_stages=[5, 8],
            down_stages=[5, 8],
            stage_cfgs=dict()
        )

        # Caricamento dei pesi pre-addestrati (NTU joint modality) da MMAction2
        if pretrained_backbone:
            if checkpoint_path is None:
                # Se non specificato, proviamo a prendere il checkpoint NTU joint da MMAction2
                # Attenzione: qui devi sostituire il link con quello effettivo che scarichi in locale.
                # Per es. potresti aver fatto il download di:
                #   stgcn_joint_ntu60_xsub.pth 
                # e lo salvi in "/path/to/stgcn_joint_ntu60_xsub.pth"
                # Noi assumiamo che il file sia in "./pretrained/stgcn_joint_ntu60_xsub.pth"
                checkpoint_path = "./pretrained/stgcn_joint_ntu60_xsub.pth"

            # Carichiamo lo stato
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            # In MMAction2 di solito il dict è { 'state_dict': {...} }
            if 'state_dict' in ckpt:
                self.backbone.load_state_dict(ckpt['state_dict'], strict=False)
            else:
                self.backbone.load_state_dict(ckpt, strict=False)
            print(f"[INFO] ST-GCN backbone pre-addestrato caricato da: {checkpoint_path}")

        # 2. Blocchi di Attention
        # ----------------------------------------------------
        # Dopo che la backbone restituisce (B, C_backbone, T, V=17),
        # vogliamo applicare un blocco di Spatial + Temporal Attention.
        # Immaginiamo che C_backbone = 256 * ch_ratio^(stage_inflation), 
        # ma qui non importa esattamente: recuperiamo dinamicamente il numero di canali.
        dummy_input = torch.zeros(1, 3, 10, 17)  # solo per test dimensione; non passa in forward!
        with torch.no_grad():
            out_bk = self.backbone(dummy_input)    # (1, C_bk, 10, 17)
        C_bk = out_bk.shape[1]
        T_dummy = out_bk.shape[2]  # numero di frame in output (dovrebbe essere uguale a T_in, a meno di downsampling)

        # Spatial e Temporal Attention
        self.spatial_attention = SpatialAttention(in_channels=C_bk, num_joints=17)
        self.temporal_attention = TemporalAttention(in_channels=C_bk, num_frames=T_dummy)

        # 3. Transformer Encoder per la dimensione temporale
        # ----------------------------------------------------
        # Dopo l’attenzione, facciamo un pooling sui 17 joints → (B, C_bk, T)
        # e lo passiamo a un Transformer Encoder dimensione D_key = C_bk
        self.transformer_dim = C_bk
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_ffn_dim,
            dropout=dropout,
            activation='relu',
            batch_first=False  # PyTorch Transformer usa (T, B, C) per default
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers
        )

        # Dropout finale e fully-connected per ottenere embedding finale di dimensione emb_dim
        self.fc_embed = nn.Sequential(
            nn.Linear(self.transformer_dim, emb_dim, bias=False),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )

        # 4. Testa ArcFace (Layer per classificazione)
        # ----------------------------------------------------
        # Per ArcFace serve normalizzare i pesi; qui creiamo un Linear normale
        # (senza bias) e faremo l’operazione di normalizzazione esternamente nella loss.
        self.classifier = nn.Linear(emb_dim, num_classes, bias=False)

        # Salviamo dimensioni utili
        self.emb_dim = emb_dim
        self.num_classes = num_classes


    def forward(self, x, labels=None):
        """
        Args:
            x: tensor (B, 3, T, 17) con (x,y,conf) normalizzati e z-score normalized
            labels: (B,) int64 con etichette zero-based [0..num_classes-1]; 
                    serve solo per ArcFace (calcolare i logit con margin).
                    Se None, torniamo solo l'embedding.
        Returns:
            Se labels is None:
                emb: (B, emb_dim) – embeddings normalizzati L2
            Altrimenti:
                emb: (B, emb_dim)
                logits: (B, num_classes) – da usare dentro ArcFaceLoss
        """
        # --- 1) Backbone ST-GCN ---
        # Input: (B, 3, T, 17) → ST-GCN → (B, C_bk, T, V=17)
        feat = self.backbone(x)  # (B, C_bk, T, 17)

        # --- 2) Spatial Attention ---
        feat = self.spatial_attention(feat)  # (B, C_bk, T, 17)

        # --- 3) Temporal Attention ---
        feat = self.temporal_attention(feat) # (B, C_bk, T, 17)

        # --- 4) Pooling sui joints → (B, C_bk, T) ---
        # Facciamo la media su V=17 joints
        feat = feat.mean(dim=-1)  # (B, C_bk, T)

        # --- 5) Transformer Encoder ---
        # Transformer di PyTorch si aspetta (T, B, C)
        feat = feat.permute(2, 0, 1).contiguous()  # (T, B, C_bk)
        # Attenzione: per batch_first=False, feed i dati come (T, B, C)
        out_tr = self.transformer_encoder(feat)     # (T, B, C_bk)
        out_tr = out_tr.permute(1, 0, 2).contiguous()  # (B, T, C_bk)

        # --- 6) Pooling sui frame → (B, C_bk) ---
        emb_pool = out_tr.mean(dim=1)  # (B, C_bk)

        # --- 7) Fully Connected → emb_dim ---
        emb = self.fc_embed(emb_pool)  # (B, emb_dim)

        # --- 8) Normalizzazione L2 dell'embedding ---
        emb_norm = F.normalize(emb, p=2, dim=1)  # (B, emb_dim)

        if labels is None:
            return emb_norm

        # --- 9) Calcolo dei logit (per ArcFace) ---
        # Supponiamo di normalizzare i pesi della classifier all’esterno nella loss:
        #   W_norm = normalize(W, dim=0)
        #   cos_theta = emb_norm @ W_norm
        #   poi in ArcFaceLoss si aggiunge il margin angolare
        # Qui calcoliamo semplicemente i logit non normalizzati:
        logits = self.classifier(emb_norm)  # (B, num_classes)

        return emb_norm, logits
    
# ================================
# Esempio di istanziazione e caricamento pesi
# ================================
if __name__ == "__main__":
    """
    Esempio di come creare il modello e caricare i pesi pre-addestrati.
    Salva questo file come model.py, poi da un altro script puoi fare:

        from model import GaitSTGCNModel
        model = GaitSTGCNModel(
            num_classes=49,
            pretrained_backbone=True,
            checkpoint_path="./pretrained/stgcn_joint_ntu60_xsub.pth",
            emb_dim=256
        )
        model = model.to(device)
    """

    # Esempio: istanziamo il modello su CPU (per test)
    device = torch.device("cpu")
    model = GaitSTGCNModel(
        num_classes=49,
        pretrained_backbone=True,
        checkpoint_path="./pretrained/stgcn_joint_ntu60_xsub.pth",
        emb_dim=256,
        transformer_layers=2,
        transformer_heads=4,
        transformer_ffn_dim=512,
        dropout=0.3
    )
    model.to(device)

    # Test dimensionale rapido con input fittizio (B=4, T=25)
    x_dummy = torch.randn(4, 3, 25, 17, device=device)
    emb, logits = model(x_dummy, torch.tensor([0,1,2,3], dtype=torch.long, device=device))
    print("Output embedding:", emb.shape)  # (4, 256)
    print("Logit ArcFace:", logits.shape)  # (4, 49)