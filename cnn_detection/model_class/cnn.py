import torch.nn as nn
import torchvision.models as models

class CNN(nn.Module):
    def __init__(self, input_channels=3, num_filters=[16, 32], kernel_size=3, stride=1, padding=1, hidden_dim=128, output_dim=4, image_size=224):
        """
        Parametri:
        - input_channels: Numero di canali in input (ad esempio 3 per immagini RGB).
        - num_filters: Lista con il numero di filtri per ciascun livello di convoluzione.
        - kernel_size: Dimensione del kernel di convoluzione.
        - stride: Stride della convoluzione.
        - padding: Padding della convoluzione.
        - hidden_dim: Dimensione del layer completamente connesso nascosto.
        - output_dim: Numero di output (ad esempio, 4 per bounding box: [x_center, y_center, width, height]).
        - input_image_size: Dimensione dell'immagine in input (assunto quadrato, es. 224x224).
        """
        super(CNN, self).__init__()
        
        # Costruzione del feature extractor
        layers = []
        in_channels = input_channels
        for filters in num_filters:
            layers.append(nn.Conv2d(in_channels, filters, kernel_size, stride, padding))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2, 2))  # Riduce le dimensioni della feature map
            in_channels = filters
        self.feature_extractor = nn.Sequential(*layers)
        
        # Calcola la dimensione della feature map dopo il feature extractor
        num_pools = len(num_filters)  # Un MaxPool2d per ogni livello
        feature_map_size = image_size // (2 ** num_pools)  # Riduzione dimensionale dopo ogni pooling
        flattened_size = num_filters[-1] * feature_map_size * feature_map_size
        
        # Costruzione del regressor
        self.regressor = nn.Sequential(
            nn.Linear(flattened_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # Feature extraction
        x = self.feature_extractor(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Bounding box regression
        bbox_out = self.regressor(x)
        return bbox_out
    

class ImprovedCNN(nn.Module):
    def __init__(self, input_channels=3, num_filters=[32, 64, 128], kernel_size=3, stride=1, padding=1, hidden_dim=256, output_dim=4, image_size=224):
        super(ImprovedCNN, self).__init__()

        layers = []
        in_channels = input_channels

        for filters in num_filters:
            layers.append(nn.Conv2d(in_channels, filters, kernel_size, stride, padding))
            layers.append(nn.BatchNorm2d(filters))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2, 2))
            layers.append(nn.Dropout(0.3))
            in_channels = filters

        self.feature_extractor = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  
        
        self.regressor = nn.Sequential(
            nn.Linear(num_filters[-1], hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        bbox_out = self.regressor(x)
        return bbox_out
    

class MobileNetV2_BBox(nn.Module):
    def __init__(self, output_dim=4):
        super(MobileNetV2_BBox, self).__init__()
        self.base_model = models.mobilenet_v2(pretrained=True).features

        self.regressor = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.base_model(x)
        x = x.mean([2, 3])
        bbox_out = self.regressor(x)
        return bbox_out
    

class ResNet_BBox(nn.Module):
    def __init__(self, output_dim=4):
        super(ResNet_BBox, self).__init__()
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