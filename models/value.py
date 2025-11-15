import torch
import torch.nn as nn
from models.common_cnn import CommonCNN


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CommonCNN()
        self.v_head = nn.Linear(256, 1)

    def forward(self, obs):
        x = self.backbone(obs)
        return self.v_head(x)
