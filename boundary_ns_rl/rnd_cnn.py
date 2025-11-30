import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.common_cnn import CommonCNN


class RNDCNN(nn.Module):
    """
    RND (Random Network Distillation) model using CNN for image observations.
    Uses CommonCNN from models.common_cnn, with an adapter layer to handle single-frame inputs.
    """
    
    def __init__(self, feature_dim: int = 256, in_channels: int = 3):
        """
        Args:
            feature_dim: dimension of output features
            in_channels: number of input channels (3 for RGB, 12 for stacked frames)
        """
        super().__init__()
        
        self.in_channels = in_channels
        
        # Target network (fixed, not trainable) - using CommonCNN
        # CommonCNN expects 12 channels (4 stacked frames), so we'll adapt the input
        self.target_network = CommonCNN(output_dim=feature_dim)
        
        # Freeze target network parameters
        for param in self.target_network.parameters():
            param.requires_grad = False
        
        # Predictor network (trainable) - using CommonCNN
        self.predictor_network = CommonCNN(output_dim=feature_dim)
        
        # Initialize target network with random weights and keep it fixed
        self._initialize_target_network()
    
    def _initialize_target_network(self):
        """Initialize target network with random weights."""
        for param in self.target_network.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param, gain=1.0)
            else:
                nn.init.constant_(param, 0)
    
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
    
    def forward(self, obs):
        """
        Get features from both networks.
        
        Args:
            obs: observation tensor of shape:
                - (batch_size, H, W, C) or (H, W, C) for single image (C=3)
                - (batch_size, C, H, W) or (C, H, W) for CHW format (C=3)
        
        Returns:
            target_features: features from target network
            pred_features: features from predictor network
        """
        # Adapt input to CommonCNN's expected format
        obs_adapted = self._adapt_input(obs)
        
        # CommonCNN expects (B, S, H, W, C) format
        target_features = self.target_network(obs_adapted)
        pred_features = self.predictor_network(obs_adapted)
        
        return target_features, pred_features
    
    def compute_intrinsic_reward(self, obs):
        """
        Compute intrinsic reward based on prediction error.
        
        Args:
            obs: observation tensor (image format)
        
        Returns:
            intrinsic_reward: scalar reward (mean squared prediction error)
        """
        target_features, pred_features = self.forward(obs)
        
        # Intrinsic reward = prediction error (mean squared error)
        prediction_error = torch.mean((target_features - pred_features) ** 2, dim=1)
        
        return prediction_error
    
    def compute_loss(self, obs):
        """
        Compute RND loss for training the predictor network.
        
        Args:
            obs: observation tensor (image format)
        
        Returns:
            rnd_loss: loss for training predictor network
        """
        target_features, pred_features = self.forward(obs)
        
        # Only predictor network is trained, target is fixed
        rnd_loss = torch.mean((target_features.detach() - pred_features) ** 2)
        
        return rnd_loss


# Keep RNDMLP as an alias for backward compatibility, but it's now CNN-based
RNDMLP = RNDCNN

