import torch
import torch.nn as nn
from common_cnn import CommonCNN

class PolicyNet(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.backbone = CommonCNN()
        self.policy_head = nn.Linear(256, action_dim)

    def forward(self, obs):
        x = self.backbone(obs)
        logits = self.policy_head(x)
        return logits
