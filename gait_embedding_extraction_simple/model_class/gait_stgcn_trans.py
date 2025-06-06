import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Graph:
    """
    Definisce il grafo COCO a 17 joint e calcola la matrice di adiacenza normalizzata.
    """

    def __init__(self, layout='coco'):
        # Definizione degli archi nel formato COCO (17 joint)
        # Ogni tupla (i, j) rappresenta un collegamento anatomico tra joint i e joint j
        if layout != 'coco':
            raise ValueError(f"Layout non supportato: {layout}. Serve 'coco'.")
        self.num_nodes = 17
        self.self_link = [(i, i) for i in range(self.num_nodes)]
        self.neighbor_link = [
            (15, 13), (13, 11),  # left ankle–knee, knee–hip
            (16, 14), (14, 12),  # right ankle–knee, knee–hip
            (11, 12),            # hip–hip
            (5, 11),  (6, 12),   # shoulder–hip
            (5, 6),              # shoulder–shoulder
            (5, 7),  (7, 9),     # left shoulder–elbow, elbow–wrist
            (6, 8),  (8, 10),    # right shoulder–elbow, elbow–wrist
            (1, 2),              # left_eye–right_eye
            (0, 1),  (0, 2),     # nose–eyes
            (1, 3),  # left_eye–left_ear  
            (2, 4),  # right_eye–right_ear  
            (3, 5),  # left_ear–left_shoulder  
            (4, 6)  # right_ear–right_shoulder  
        ]
        self.edge_list = self.self_link + self.neighbor_link + [(j, i) for (i, j) in self.neighbor_link]
        self.A = self._get_adjacency_matrix()
        self.A_norm = self._normalize_adjacency_matrix(self.A)

    def _get_adjacency_matrix(self):
        """Costruisce la matrice di adiacenza non normalizzata A (shape: 17×17)."""
        A = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.float32)
        for i, j in self.edge_list:
            A[i, j] = 1.0
        return A

    @staticmethod
    def _normalize_adjacency_matrix(A):
        """Normalizza A usando D^{-1/2} A D^{-1/2}."""
        Dl = A.sum(dim=1)               # grado di ogni nodo
        # Evitiamo divisione per zero
        Dn = torch.diag(torch.pow(Dl + 1e-6, -0.5))
        A_norm = Dn @ A @ Dn
        return A_norm              # shape: (17, 17)


class STGCNBlock(nn.Module):
    """
    Un singolo blocco ST-GCN:
      1) Convoluzione spaziale su grafo: aggiorna feature dei joint aggregando dai vicini.
      2) BatchNorm + ReLU
      3) Convoluzione temporale (Conv2d con kernel su dimensione tempo)
      4) BatchNorm + ReLU
      5) Collegamento residuo se input e output hanno stesse dimensioni; altrimenti Proiezione lineare del residuo.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 A_norm: torch.Tensor,
                 kernel_size: int = 9,
                 stride: int = 1):
        """
        Args:
            in_channels (int): numero di canali in input (feature per joint).
            out_channels (int): canali in output (feature per joint).
            A_norm (torch.Tensor): matrice di adiacenza normalizzata (17×17).
            kernel_size (int): dimensione del filtro temporale (numero frame).
            stride (int): stride temporale.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.A_norm = A_norm           # (17, 17)
        # Registriamo A_norm come buffer, così viene spostato su GPU/CPU insieme al modello
        self.register_buffer('A_norm', A_norm)
        
        self.num_nodes = A_norm.size(0)

        # 1) Peso per la Graph Convolution (proiezione lineare su canali)
        self.gc_weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        nn.init.kaiming_uniform_(self.gc_weight, a=math.sqrt(5))

        # 2) BatchNorm dopo GCN
        self.bn_gc = nn.BatchNorm2d(out_channels)

        # 3) Convoluzione temporale: Conv2d con kernel (kernel_size, 1) su (T, 1)
        pad = (kernel_size - 1) // 2
        self.conv_t = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(pad, 0),
            bias=False
        )
        self.bn_tc = nn.BatchNorm2d(out_channels)

        # 4) Residual/shortcut
        if in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            # se cambia dimensione canali o stride, usiamo una conv 1x1 sul tempo per adattare
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, 1),
                    stride=(stride, 1),
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        x: tensor di shape (B, C_in, T, V)
        Ritorna: tensor di shape (B, C_out, T_out, V)
        """
        # --- 1) Graph Convolution ---
        # Input: x_gc (B, C_in, T, V)
        B, C_in, T, V = x.size()
        # 1a) Permutiamo per avere shape (B, T, V, C_in) e moltiplichiamo per i pesi gc_weight (C_in→C_out)
        x_perm = x.permute(0, 2, 3, 1).contiguous()         # (B, T, V, C_in)
        x_lin = torch.matmul(x_perm, self.gc_weight)       # (B, T, V, C_out)

        # 1b) Aggregazione spaziale: sommiamo le feature dai vicini secondo A_norm
        # Permutiamo in (B, T, C_out, V) per matmul su A
        x_lin = x_lin.permute(0, 1, 3, 2).contiguous()     # (B, T, C_out, V)
        # Matmul: (B, T, C_out, V) × (V, V) → (B, T, C_out, V)
        x_gc = torch.matmul(x_lin, self.A_norm)             # (B, T, C_out, V)

        # 1c) Permutiamo indietro in (B, C_out, T, V)
        x_gc = x_gc.permute(0, 2, 1, 3).contiguous()        # (B, C_out, T, V)
        x_gc = self.bn_gc(x_gc)                             # BatchNorm
        x_gc = self.relu(x_gc)

        # --- 2) Convoluzione Temporale ---
        x_tc = self.conv_t(x_gc)   # (B, C_out, T_out, V)
        x_tc = self.bn_tc(x_tc)
        # Nota: ReLU sarà applicata dopo aver sommato il residuo

        # --- 3) Residual Connection ---
        res = self.residual(x)     # (B, C_out, T_out, V)
        out = x_tc + res
        out = self.relu(out)

        return out


class SpatialAttention(nn.Module):
    """
    Spatial Attention: dato un feature-map (B, C, T, V),
    calcola un peso su ciascun joint (V) per enfatizzare i nodi più discriminativi.
    """

    def __init__(self, in_channels, num_joints=17):
        super().__init__()
        # Conv1x1 per mappare da in_channels -> num_joints score (senza bias)
        self.conv1x1 = nn.Conv2d(in_channels, num_joints, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=-1)  # softmax sui V joint

    def forward(self, x):
        # x: (B, C, T, V)
        B, C, T, V = x.size()
        # # Score: (B, V, T, V)
        # score = self.conv1x1(x)           
        # # Permutiamo per avere (B, T, V, V), così softmax sarà sull'ultimo V
        # score_perm = score.permute(0, 2, 1, 3).contiguous()  # (B, T, V, V)
        # # Softmax sull'ultima dimensione (joint “target”)
        # attn = self.softmax(score_perm)  # (B, T, V, V)
        # Otteniamo i punteggi: (B, V, T, V)
        score = self.conv1x1(x)  # (B, V, T, V)
        # Permutiamo per avere (B, T, V, V)
        score = score.permute(0, 2, 1, 3).contiguous()  # (B, T, V, V)
        # Applichiamo softmax sui joint di origine (dim=-1)
        attn = self.softmax(score)  # (B, T, V, V)

        # # Matmul: vogliamo pesare i joint “source” per ottenere nuovi feature
        # # Espandiamo x: (B, C, T, V, 1)
        # x_ = x.unsqueeze(-1)               # (B, C, T, V, 1)
        # # Espandiamo attn: (B, 1, T, V, V) per broadcast
        # attn_ = attn.unsqueeze(1)          # (B, 1, T, V, V)
        # # Matmul su dim V “source”: (B, C, T, V, 1) x (B, 1, T, V, V) → (B, C, T, V, 1)
        # out = torch.matmul(x_, attn_)      # (B, C, T, V, 1)
        # out = out.squeeze(-1)              # (B, C, T, V)
        # Prepariamo per batch matmul: x (B, C, T, V) -> (B*T, C, V)
        x_perm = x.permute(0, 2, 1, 3).contiguous()    # (B, T, C, V)
        x_reshaped = x_perm.view(B * T, C, V)           # (B*T, C, V)

        # attn: (B, T, V, V) -> (B*T, V, V)
        attn_reshaped = attn.view(B * T, V, V)          # (B*T, V, V)

        # out = x_reshaped (B*T, C, V) @ attn_reshaped (B*T, V, V) -> (B*T, C, V)
        out = torch.matmul(x_reshaped, attn_reshaped)   # (B*T, C, V)

        # Rimettiamo le dimensioni a (B, C, T, V)
        out = out.view(B, T, C, V).permute(0, 2, 1, 3).contiguous()  # (B, C, T, V)
        return out


class TemporalAttention(nn.Module):
    """
    Temporal Attention: dato un feature-map (B, C, T, V),
    calcola un peso su ciascun istante temporale (T), per enfatizzare i frame più discriminativi.
    """

    # def __init__(self, in_channels, num_frames):
    def __init__(self, in_channels):
        super().__init__()
        # # Conv1x1 per mappare da in_channels -> num_frames score, mantenendo V
        # self.conv1x1 = nn.Conv2d(in_channels, num_frames, kernel_size=1, bias=False)
        # self.softmax = nn.Softmax(dim=2)  # softmax lungo la dimensione T
        # Conv1x1 per ridurre su canale singolo
        self.conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=2)  # softmax sulla dimensione T

    # def forward(self, x):
    #     # x: (B, C, T, V)
    #     B, C, T, V = x.size()
    #     # # Score: (B, T, T, V)
    #     # score = self.conv1x1(x)           # (B, T, T, V)
    #     # Otteniamo i punteggi: (B, T, T, V)
    #     score = self.conv1x1(x)  # (B, T, T, V)
    #     # # Permutiamo per avere (B, V, T, T)
    #     # score_perm = score.permute(0, 3, 1, 2).contiguous()  # (B, V, T, T)
    #     # attn = self.softmax(score_perm)   # (B, V, T, T)
    #     # Permutiamo per avere (B, V, T, T)
    #     score = score.permute(0, 3, 1, 2).contiguous()  # (B, V, T, T)
    #     # Applichiamo softmax sui frame (dim=-1)
    #     attn = self.softmax(score)  # (B, V, T, T)

    #     # Ora matmul per pesare i frame
    #     # # Espandiamo x: (B, C, T, V, 1)
    #     # x_ = x.unsqueeze(-1)              # (B, C, T, V, 1)
    #     # # Permutiamo x_ in (B, C, V, T, 1)
    #     # x_perm = x_.permute(0, 1, 3, 2, 4).contiguous()  # (B, C, V, T, 1)
    #     # # Espandiamo attn: (B, 1, V, T, T)
    #     # attn_ = attn.unsqueeze(1)         # (B, 1, V, T, T)
    #     # # Matmul su dim T “source”: (B, 1, V, T, T) × (B, C, V, T, 1) → (B, C, V, T, 1)
    #     # out = torch.matmul(attn_, x_perm)  # (B, C, V, T, 1)
    #     # out = out.squeeze(-1)              # (B, C, V, T)
    #     # # Rimettiamo nell’ordine (B, C, T, V)
    #     # out = out.permute(0, 1, 3, 2).contiguous()  # (B, C, T, V)
    #     # Prepariamo per batch matmul: x (B, C, T, V) -> (B*V, C, T)
    #     x_perm = x.permute(0, 3, 1, 2).contiguous()    # (B, V, C, T)
    #     x_reshaped = x_perm.view(B * V, C, T)           # (B*V, C, T)

    #     # attn: (B, V, T, T) -> (B*V, T, T)
    #     attn_reshaped = attn.view(B * V, T, T)          # (B*V, T, T)

    #     # out = attn_reshaped (B*V, T, T) @ x_reshaped (B*V, C, T).transpose -> (B*V, T, C),
    #     # poi trasponiamo il risultato
    #     out = torch.matmul(attn_reshaped, x_reshaped.transpose(1, 2))  # (B*V, T, C)
    #     out = out.transpose(1, 2)  # (B*V, C, T)

    #     # Rimettiamo le dimensioni a (B, C, T, V)
    #     out = out.view(B, V, C, T).permute(0, 2, 3, 1).contiguous()  # (B, C, T, V)
    #     return out
    
    def forward(self, x):
        # x: (B, C, T, V)
        # 1) Otteniamo score: (B, 1, T, V)
        score = self.conv1x1(x)  # (B, 1, T, V)
        # 2) Facciamo media sui joint (V) per ottenere (B, 1, T)
        score = score.mean(dim=-1, keepdim=False)  # (B, 1, T)
        # 3) Softmax sui frame
        attn = self.softmax(score)  # (B, 1, T)
        # 4) Applichiamo attenzione: (B, C, T, V) * (B, 1, T, 1)
        attn = attn.unsqueeze(-1)  # (B, 1, T, 1)
        out = x * attn  # broadcasting su C e V
        return out
    

class GaitSTGCNModel(nn.Module):
    """
    Modello completo per estrarre embedding di gait:
      - Encoder ST-GCN personalizzato (senza dipendenze esterne)
      - Blocchi di Spatial + Temporal Attention
      - Transformer Encoder sulla dimensione temporale
      - FC finale per ridurre a emb_dim e normalizzazione L2
      - Linear finale (senza bias) per ArcFace (num_classes)
    """

    def __init__(self,
                 num_classes: int = 49,
                 pretrained_stgcn_path: str = None,
                 device: str = 'cpu',
                 emb_dim: int = 256,
                 transformer_layers: int = 2,
                 transformer_heads: int = 4,
                 transformer_ffn_dim: int = 512,
                 dropout: float = 0.3):
        """
        Args:
            num_classes (int): numero di identità (49).
            pretrained_stgcn_path (str): path al file .pth con checkpoint del solo backbone ST-GCN
                                         (con chiavi corrispondenti a "stgcn_encoder.*"). Se None, inizializza da capo.
            emb_dim (int): dimensione del vettore embedding finale.
            transformer_layers (int): numero di layer del Transformer Encoder.
            transformer_heads (int): numero di teste di attenzione per il Transformer.
            transformer_ffn_dim (int): dimensione interna del FFN del Transformer.
            dropout (float): dropout negli strati FC/Transformer.
        """
        super().__init__()

        # 1) Costruiamo il grafo COCO e otteniamo A_norm
        graph = Graph(layout='coco')
        A_norm = graph.A_norm  # (17, 17)

        # 2) Definiamo la sequenza di blocchi ST-GCN
        # -------------------------------------------
        #   - Primo blocco: in_channels=3 (x, y, confidence) → out_channels=64
        #   - Secondo:                64 → 128
        #   - Terzo:                  128 → 256
        #   (puoi eventualmente aggiungere altri blocchi o cambiare canali)
        self.stgcn1 = STGCNBlock(in_channels=3, out_channels=64, A_norm=A_norm, kernel_size=9, stride=1)
        self.stgcn2 = STGCNBlock(in_channels=64, out_channels=128, A_norm=A_norm, kernel_size=9, stride=1)
        self.stgcn3 = STGCNBlock(in_channels=128, out_channels=256, A_norm=A_norm, kernel_size=9, stride=1)

        # 3) Carica pesi pre-addestrati per i blocchi ST-GCN (facoltativo)
        #    Si assume che il checkpoint contenga chiavi come "stgcn1.gc_weight", "stgcn1.bn_gc.weight", ecc.
        if pretrained_stgcn_path is not None:
            ckpt = torch.load(pretrained_stgcn_path, map_location=device)
            # Esempio: il checkpoint potrebbe essere un dict con chiavi “state_dict”
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            # Carica ciò che combacia, usando strict=False
            self.load_state_dict(ckpt, strict=False)
            print(f"[INFO] Pesi ST-GCN caricati da: {pretrained_stgcn_path}")

        # 4) Spatial e Temporal Attention
        # --------------------------------
        # Determiniamo quanti canali esce dallo ST-GCN. Per farlo, facciamo un passaggio dummy:
        with torch.no_grad():
            x_dummy = torch.zeros(1, 3, 10, 17)  # B=1, canali=3, T=10, 17 joint
            out1 = self.stgcn1(x_dummy)  # (1, 64, 10, 17)
            out2 = self.stgcn2(out1)     # (1, 128, 10, 17)
            out3 = self.stgcn3(out2)     # (1, 256, 10, 17)
        C_bk = out3.shape[1]  # 256
        T_dummy = out3.shape[2]  # 10 (uguale a input, dato stride=1)

        self.spatial_attention = SpatialAttention(in_channels=C_bk, num_joints=17)
        # self.temporal_attention = TemporalAttention(in_channels=C_bk, num_frames=T_dummy)
        self.temporal_attention = TemporalAttention(in_channels=C_bk)

        # 5) Transformer Encoder
        # ----------------------
        # Dopo attention, facciamo pooling sui 17 joint → (B, C_bk, T)
        # Poi Transformer su (T, B, C_bk)
        self.transformer_dim = C_bk
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_ffn_dim,
            dropout=dropout,
            activation='relu',
            batch_first=False  # Transformer usa (T, B, C)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers
        )

        # 6) Riduzione finale a emb_dim
        # -----------------------------
        # Dopo pooling temporale, otteniamo (B, C_bk); lo riduciamo a (B, emb_dim)
        self.fc_embed = nn.Sequential(
            nn.Linear(self.transformer_dim, emb_dim, bias=False),
            # nn.BatchNorm1d(emb_dim),
            nn.LayerNorm(emb_dim),  
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )

        # 7) Classifier per ArcFace (senza bias; normalizzeremo i pesi esternamente nella loss)
        # -------------------------------------------------------------------------------------
        self.classifier = nn.Linear(emb_dim, num_classes, bias=False)

        # Salviamo alcuni attributi
        self.emb_dim = emb_dim
        self.num_classes = num_classes

    def forward(self, x, labels=None):
        """
        Args:
            x: tensor (B, 3, T, 17) con (x, y, confidence) normalizzati e z-score normalized
            labels: (B,) int64 con etichette zero-based [0..num_classes-1];
                    serve solo per ArcFace (calcolare i logit con margin).
                    Se None, restituiamo solo l'embedding.

        Returns:
            se labels è None:
                emb_norm: (B, emb_dim) – embedding normalizzati L2
            altrimenti:
                emb_norm: (B, emb_dim)
                logits: (B, num_classes) – da passare a ArcFaceLoss
        """
        # --- 1) ST-GCN Layers ---
        feat = self.stgcn1(x)   # (B,  64, T, 17)
        feat = self.stgcn2(feat)  # (B, 128, T, 17)
        feat = self.stgcn3(feat)  # (B, 256, T, 17)

        # --- 2) Spatial Attention ---
        feat = self.spatial_attention(feat)  # (B, 256, T, 17)

        # --- 3) Temporal Attention ---
        feat = self.temporal_attention(feat)  # (B, 256, T, 17)

        # --- 4) Pooling sui joints → (B, 256, T) ---
        feat = feat.mean(dim=-1)  # media su V=17 → (B, 256, T)

        # --- 5) Transformer Encoder ---
        # Cambiamo shape in (T, B, C=256)
        feat = feat.permute(2, 0, 1).contiguous()  # (T, B, 256)
        out_tr = self.transformer_encoder(feat)    # (T, B, 256)
        out_tr = out_tr.permute(1, 0, 2).contiguous()  # (B, T, 256)

        # --- 6) Pooling sui frame → (B, 256) ---
        emb_pool = out_tr.mean(dim=1)  # (B, 256)

        # --- 7) FC inferiore per embedding → (B, emb_dim) ---
        emb = self.fc_embed(emb_pool)  # (B, emb_dim)

        # --- 8) Normalizzazione L2 ---
        emb_norm = F.normalize(emb, p=2, dim=1)  # (B, emb_dim)

        if labels is None:
            return emb_norm

        # --- 9) Calcolo dei logit per ArcFace ---
        logits = self.classifier(emb_norm)  # (B, num_classes)
        return emb_norm, logits


# ================================
# Blocco per testare l’istanziazione e caricamento pesi
# ================================
# if __name__ == "__main__":
#     import math

#     # Esempio di istanziazione del modello (su CPU)
#     device = torch.device("cpu")
#     model = GaitSTGCNModel(
#         num_classes=49,
#         pretrained_stgcn_path='/Users/giovanni/Desktop/Tesi di Laurea/model_checkpoints/gait_embedding_extraction/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth',
#         emb_dim=256,
#         transformer_layers=2,
#         transformer_heads=4,
#         transformer_ffn_dim=512,
#         dropout=0.3
#     )
#     model.to(device)

#     # Verifica forward con input fittizio (B=4, T=25)
#     x_dummy = torch.randn(4, 3, 25, 17, device=device)
#     emb, logits = model(x_dummy, torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device))
#     print("Shape embedding:", emb.shape)  # (4, 256)
#     print("Shape logits:", logits.shape)  # (4, 49)

    # Se vogliamo caricare pesi ST-GCN pre-addestrati:
    # Assumiamo di avere un checkpoint con chiavi come 
    # "stgcn1.gc_weight", "stgcn1.bn_gc.weight", ..., "stgcn3.conv_t.weight", ecc.
    # basta chiamare:
    #   model.load_state_dict(torch.load("./pretrained/stgcn_backbone.pth"), strict=False)
    # e il resto dei parametri verranno inizializzati casualmente.