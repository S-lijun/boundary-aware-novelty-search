import torch
import torch.nn as nn
from .common_cnn import CommonCNN

class ICM(nn.Module):
    def __init__(self, action_dim, feature_dim=256):
        super().__init__()
        self.encoder = CommonCNN(output_dim=feature_dim)

        # inverse model
        self.inverse = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

        # forward model
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

    def forward(self, obs, next_obs, action):
        phi_s = self.encoder(obs)
        phi_s_next = self.encoder(next_obs)

        # inverse prediction
        inv_in = torch.cat([phi_s, phi_s_next], dim=1)
        pred_action_logits = self.inverse(inv_in)

        # forward prediction
        onehot_action = torch.nn.functional.one_hot(action, num_classes=pred_action_logits.size(-1)).float()
        fwd_in = torch.cat([phi_s, onehot_action], dim=1)
        pred_phi_next = self.forward_model(fwd_in)

        return phi_s, phi_s_next, pred_action_logits, pred_phi_next
