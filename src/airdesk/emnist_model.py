"""Small convolutional network shared by EMNIST training and inference."""

from __future__ import annotations

import torch.nn as nn


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class EmnistLetterCNN(nn.Module):
    """Classify a centered 28×28 grayscale image as one of A–Z."""

    def __init__(self) -> None:
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
            nn.Linear(128, len(LETTERS)),
        )

    def forward(self, image):
        return self.classifier(self.features(image))
