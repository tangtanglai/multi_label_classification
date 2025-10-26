import torch
from torch import nn
from torchvision import models

class MultiLabelResNet50(nn.Module):
    def __init__(self, num_classes: int, dropout_prob: float = 0.5):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in self.backbone.parameters():
            param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = self._create_head(in_features, num_classes, dropout_prob)

    def _create_head(self, in_features, num_classes, dropout_prob):
        return nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.BatchNorm1d(in_features // 2),
            nn.Dropout(dropout_prob),
            nn.Linear(in_features // 2, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
