import torch
import torch.nn as nn

class CommonCNN(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2),   # Input: 4 stacked frames
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(64 * 7 * 7, output_dim)

    def forward(self, x):
        x = x / 255.0
        x = self.encoder(x)
        return self.fc(x)
