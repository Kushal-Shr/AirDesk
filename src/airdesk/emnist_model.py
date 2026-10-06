"""Small convolutional network shared by EMNIST training and inference."""

from __future__ import annotations

import torch.nn as nn


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class EmnistLetterCNN(nn.Module):
    """Classify a 28×28 grayscale image using a configurable label count."""

    def __init__(self, number_of_classes: int = len(LETTERS)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 3 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(128, number_of_classes),
        )

    def forward(self, image):
        return self.classifier(self.features(image))
