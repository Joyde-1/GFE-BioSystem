import torch.nn as nn
import torchvision.models as models


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