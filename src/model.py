"""
PyTorch Convolutional Neural Network (CNN) Architectures
for Plant Disease Classification.
"""

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Convolutional Block with 2 Conv layers, Batch Normalization, ReLU, and Max Pooling."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.conv(x)


class PlantDiseaseCNN(nn.Module):
    """
    Custom Deep Convolutional Neural Network Architecture for Plant Leaf Disease Classification.
    
    Architecture:
    - Block 1: Conv(3->32) -> Conv(32->32) -> MaxPool (64x64)
    - Block 2: Conv(32->64) -> Conv(64->64) -> MaxPool (32x32)
    - Block 3: Conv(64->128) -> Conv(128->128) -> MaxPool (16x16)
    - Block 4: Conv(128->256) -> Conv(256->256) -> MaxPool (8x8)
    - Global Adaptive Average Pooling (1x1)
    - Classifier: FC(256->512) -> BatchNorm -> ReLU -> Dropout -> FC(512->15)
    """
    def __init__(self, num_classes=15, dropout_rate=0.4):
        super().__init__()
        
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2.0),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
