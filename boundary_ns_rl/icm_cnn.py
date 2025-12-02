import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.common_cnn import CommonCNN


class ICMCNN(nn.Module):
    """
    ICM (Intrinsic Curiosity Module) model using CNN for image observations.
    Uses CommonCNN from models.common_cnn as the encoder.
    """
    
    def __init__(self, action_dim: int, feature_dim: int = 256, in_channels: int = 3):
        """
        Args:
            action_dim: number of actions
            feature_dim: dimension of output features
            in_channels: number of input channels (3 for RGB)
        """
        super().__init__()
        
        self.action_dim = action_dim
        self.in_channels = in_channels
        
        # Encoder network (shared for current and next state)
        self.encoder = CommonCNN(output_dim=feature_dim)
        
        # Inverse model: predicts action from (phi(s), phi(s'))
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        # Forward model: predicts phi(s') from (phi(s), action)
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
    
    def _adapt_input(self, obs):
        """
        Adapt input to match CommonCNN's expected format (5D: [B, S, H, W, C] with S*C=12).
        CommonCNN expects 12 channels (4 stacked frames × 3 RGB).
        For single frame (3 channels), we replicate it 4 times to get 12 channels.
        """
        # Handle single observation (no batch dimension)
        if obs.dim() == 3:
            # (H, W, C) or (C, H, W)
            if obs.shape[-1] == self.in_channels:
                # (H, W, C) format
                obs = obs.unsqueeze(0)  # Add batch dimension -> (1, H, W, C)
            else:
                # (C, H, W) format
                obs = obs.unsqueeze(0)  # Add batch dimension -> (1, C, H, W)
                obs = obs.permute(0, 2, 3, 1)  # -> (1, H, W, C)
        
        # Handle 4D input: (B, H, W, C) or (B, C, H, W)
        elif obs.dim() == 4:
            if obs.shape[1] == self.in_channels:
                # (B, C, H, W) format, convert to (B, H, W, C)
                obs = obs.permute(0, 2, 3, 1)
            # else: already (B, H, W, C) format
        
        # Now obs is (B, H, W, C) with C = in_channels
        B, H, W, C = obs.shape
        
        # CommonCNN expects (B, S, H, W, C) where S*C = 12
        # If we have 3 channels, replicate 4 times to get 12 channels
        if C == 3:
            # Replicate the frame 4 times: (B, H, W, 3) -> (B, 4, H, W, 3)
            obs = obs.unsqueeze(1).repeat(1, 4, 1, 1, 1)  # (B, 4, H, W, 3)
        elif C == 12:
            # Already 12 channels, reshape to (B, 4, H, W, 3)
            obs = obs.view(B, 4, H, W, 3)
        else:
            raise ValueError(f"Unsupported input channels: {C}. Expected 3 or 12.")
        
        return obs
    
    def forward(self, obs, next_obs, action):
        """
        Forward pass through ICM.
        
        Args:
            obs: current observation tensor
            next_obs: next observation tensor
            action: action tensor (can be indices or one-hot)
        
        Returns:
            phi_s: encoded current state
            phi_s_next: encoded next state
            pred_action_logits: predicted action logits (from inverse model)
            pred_phi_next: predicted next state features (from forward model)
        """
        # Adapt inputs to CommonCNN's expected format
        obs_adapted = self._adapt_input(obs)
        next_obs_adapted = self._adapt_input(next_obs)
        
        # Encode states
        phi_s = self.encoder(obs_adapted)
        phi_s_next = self.encoder(next_obs_adapted)
        
        # Inverse model: predict action from (phi_s, phi_s_next)
        inv_input = torch.cat([phi_s, phi_s_next], dim=1)
        pred_action_logits = self.inverse_model(inv_input)
        
        # Forward model: predict phi_s_next from (phi_s, action)
        # Convert action to one-hot if needed
        if action.dim() == 0:
            action = action.unsqueeze(0)
        if action.dtype == torch.long:
            onehot_action = F.one_hot(action, num_classes=self.action_dim).float()
        else:
            onehot_action = action
        
        # Ensure batch dimension matches
        if onehot_action.dim() == 1 and phi_s.dim() == 2:
            onehot_action = onehot_action.unsqueeze(0)
        
        fwd_input = torch.cat([phi_s, onehot_action], dim=1)
        pred_phi_next = self.forward_model(fwd_input)
        
        return phi_s, phi_s_next, pred_action_logits, pred_phi_next
    
    def compute_intrinsic_reward(self, obs, next_obs, action):
        """
        Compute intrinsic reward based on forward model prediction error.
        
        Args:
            obs: current observation tensor
            next_obs: next observation tensor
            action: action tensor
        
        Returns:
            intrinsic_reward: scalar reward (forward prediction error)
        """
        _, phi_s_next, _, pred_phi_next = self.forward(obs, next_obs, action)
        
        # Intrinsic reward = forward prediction error (mean squared error)
        prediction_error = torch.mean((phi_s_next - pred_phi_next) ** 2, dim=1)
        
        return prediction_error
    
    def compute_loss(self, obs, next_obs, action, beta: float = 0.2):
        """
        Compute ICM loss (inverse loss + forward loss).
        
        Args:
            obs: current observation tensor
            next_obs: next observation tensor
            action: action tensor (should be indices for cross-entropy)
            beta: weight for forward loss (1-beta for inverse loss)
        
        Returns:
            icm_loss: combined loss for training ICM
        """
        phi_s, phi_s_next, pred_action_logits, pred_phi_next = self.forward(obs, next_obs, action)
        
        # Inverse loss: cross-entropy for action prediction
        if action.dtype != torch.long:
            action = action.long()
        if action.dim() == 0:
            action = action.unsqueeze(0)
        
        inverse_loss = F.cross_entropy(pred_action_logits, action)
        
        # Forward loss: MSE for next state prediction
        forward_loss = 0.5 * F.mse_loss(pred_phi_next, phi_s_next.detach())
        
        # Combined loss
        icm_loss = (1 - beta) * inverse_loss + beta * forward_loss
        
        return icm_loss

