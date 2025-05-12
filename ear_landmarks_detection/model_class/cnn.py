import torch.nn as nn
import torchvision.models as models


class MobileNetV2EarLandmarks(nn.Module):
    # def __init__(self, output_dim=4):
    #     super(MobileNetV2EarLandmarks, self).__init__()
    #     self.base_model = models.mobilenet_v2(pretrained=True).features

    #     self.regressor = nn.Sequential(
    #         nn.Linear(1280, 512),
    #         nn.ReLU(),
    #         nn.Dropout(0.3),
    #         nn.Linear(512, output_dim),
    #         nn.Sigmoid()
    #     )

    # def forward(self, x):
    #     x = self.base_model(x)
    #     x = x.mean([2, 3])
    #     bbox_out = self.regressor(x)
    #     return bbox_out

    def __init__(self, output_dim=8):
        """
        Inizializza il modello per il rilevamento dei landmark dell'orecchio.
        
        Args:
            output_dim (int): Dimensione dell'output, ad es. 8 per 4 keypoint (x, y) ciascuno.
        """
        super(MobileNetV2EarLandmarks, self).__init__()
        # Usa MobileNetV2 pre-addestrata come feature extractor.
        self.base_model = models.mobilenet_v2(pretrained=True).features
        
        # Il backbone di MobileNetV2 produce in uscita un feature map con 1280 canali.
        # Applichiamo un pooling adattivo per ottenere un vettore di dimensione fissa.
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Head per la regressione dei landmark
        self.regressor = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        # Estrae le feature tramite MobileNetV2
        x = self.base_model(x)
        # Applica pooling per ottenere un vettore per immagine
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        # Predice i landmark (coordinate x e y)
        landmarks = self.regressor(x)
        return landmarks
    

class ResNet50EarLandmarks(nn.Module):
    def __init__(self, output_dim=8):  # output_dim=8 per 4 keypoint (x, y) ciascuno
        super(ResNet50EarLandmarks, self).__init__()
        # Carica ResNet50 pre-addestrata
        self.backbone = models.resnet50(pretrained=True)
        # Rimuovi il classificatore finale
        self.backbone.fc = nn.Identity()
        
        # Definisci il regressore per i landmark
        self.regressor = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        # Estrai le feature dalla ResNet50
        features = self.backbone(x)  # [B, 2048]
        # Predici i landmark
        landmarks = self.regressor(features)  # [B, output_dim]
        return landmarks
    

class ResNet18EarLandmarks(nn.Module):
    def __init__(self, output_dim=8):
        super(ResNet18EarLandmarks, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        self.regressor = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        bbox_out = self.regressor(x)
        return bbox_out

    # def __init__(self, output_dim=8):
    #     """
    #     Modello per il rilevamento dei landmark dell'orecchio basato su ResNet18.
    #     output_dim=8 corrisponde a 4 keypoint (x, y) ciascuno.
    #     """
    #     super(ResNet18EarLandmarks, self).__init__()
    #     # Carica ResNet18 pre-addestrata
    #     resnet = models.resnet18(pretrained=True)
    #     # Rimuovi il classificatore finale
    #     self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
    #     # Head per la regressione dei landmark
    #     self.regressor = nn.Sequential(
    #         nn.Linear(resnet.fc.in_features, 512),
    #         nn.ReLU(),
    #         nn.Dropout(0.3),
    #         nn.Linear(512, output_dim)
    #         # Se le coordinate sono normalizzate, potresti aggiungere nn.Sigmoid()
    #         # nn.Sigmoid()
    #     )

    # def forward(self, x):
    #     # Estrae le feature dalla ResNet18
    #     x = self.feature_extractor(x)  # Output shape: [B, features, 1, 1]
    #     x = x.view(x.size(0), -1)        # Output shape: [B, features]
    #     # Predice i landmark
    #     landmarks = self.regressor(x)    # Output shape: [B, output_dim]
    #     return landmarks