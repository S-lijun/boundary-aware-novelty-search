import torch
import torch.nn as nn

class CommonCNN(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()

        # 12 input channels = 4 stacked frames × 3 RGB channels
        in_channels = 12

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),           # 64 → 32

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),           # 32 → 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)            # 16 → 8
        )

        # final feature map size = 128 × 8 × 8 = 8192
        self.fc = nn.Linear(128 * 8 * 8, output_dim)
        self.output_dim = output_dim

    def forward(self, x):
     
        if x.dim() == 6 and x.size(1) == 1:
            x = x.squeeze(1)

        # x: [B, S, H, W, C]
        if x.dim() == 5:
            B, S, H, W, C = x.shape
            x = x.permute(0, 1, 4, 2, 3).contiguous()  # [B, S, C, H, W]
            x = x.view(B, S * C, H, W)                 # [B, 12, H, W]

        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
