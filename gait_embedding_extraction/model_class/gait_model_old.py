import torch
import torch.nn as nn
import torch.nn.functional as F


class KeypointAttention(nn.Module):
    """Attention sui keypoints individuali"""
    def __init__(self, keypoint_dim=3):
        super(KeypointAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(keypoint_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (batch, seq_len, 17, 3)
        batch_size, seq_len, num_keypoints, keypoint_dim = x.shape
        
        # Reshape per applicare attention
        x_reshaped = x.view(-1, keypoint_dim)  # (batch*seq*17, 3)
        weights = self.attention(x_reshaped)   # (batch*seq*17, 1)
        weights = weights.view(batch_size, seq_len, num_keypoints, 1)  # (batch, seq, 17, 1)
        
        # Applica attention
        attended = x * weights  # (batch, seq, 17, 3)
        return attended


class TemporalAttention(nn.Module):
    """Attention mechanism per frame temporali"""
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden_dim)
        attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)

        # Weighted sum
        context = torch.sum(lstm_out * attention_weights, dim=1)  # (batch, hidden_dim)
        return context, attention_weights


class GaitModel(nn.Module):
    """Modello migliorato per riconoscimento biometrico basato su andatura"""
    def __init__(self, 
                 input_dim=51,
                 hidden_dim=256,
                 lstm_layers=2,
                 embedding_dim=256,  # CAMBIATO da 128 a 256
                 num_classes=49,
                 dropout=0.2):  # CAMBIATO da 0.3 a 0.2
        super(GaitModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # Keypoint attention (sui singoli keypoints)
        self.keypoint_attention = KeypointAttention(keypoint_dim=3)
        
        # Feature extraction per keypoints (CNN 1D temporale)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(51, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(256, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # LSTM per dinamiche temporali
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # Temporal attention
        self.temporal_attention = TemporalAttention(hidden_dim * 2)

        # Embedding layers con più profondità
        self.embedding_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, embedding_dim)
        )

        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)

        # Inizializzazione pesi
        self._initialize_weights()

    def _initialize_weights(self):
        """Inizializzazione migliorata dei pesi"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_attention=False):
        """Forward pass migliorato"""
        batch_size, seq_len, _ = x.shape

        # Reshape per keypoint attention: (batch, seq, 17, 3)
        x_keypoints = x.view(batch_size, seq_len, 17, 3)
        
        # Applica keypoint attention
        x_attended = self.keypoint_attention(x_keypoints)
        
        # Torna al formato flat: (batch, seq, 51)
        x_flat = x_attended.view(batch_size, seq_len, 51)

        # Trasposizione per CNN 1D: (batch, features, seq_len)
        x_conv = x_flat.transpose(1, 2)

        # Feature extraction temporale
        features = self.temporal_conv(x_conv)  # (batch, hidden_dim, seq_len)

        # Ritorna a formato per LSTM: (batch, seq_len, hidden_dim)
        features = features.transpose(1, 2)

        # LSTM
        lstm_out, _ = self.lstm(features)  # (batch, seq_len, hidden_dim*2)

        # Temporal attention
        context, attention_weights = self.temporal_attention(lstm_out)

        # Embedding biometrico
        embedding = self.embedding_layers(context)

        # L2 normalization per embedding
        embedding = F.normalize(embedding, p=2, dim=1)

        # Classification logits
        logits = self.classifier(embedding)

        if return_attention:
            return embedding, logits, attention_weights
        else:
            return embedding, logits

    def get_embedding(self, x):
        """Estrae solo l'embedding biometrico"""
        with torch.no_grad():
            embedding, _ = self.forward(x)
            return embedding