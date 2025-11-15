import torch
import torch.nn as nn
from models.common_cnn import CommonCNN


class RND(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        
        # Target network (fixed, not trainable)
        self.target_network = CommonCNN(output_dim=feature_dim)
        # Freeze target network parameters
        for param in self.target_network.parameters():
            param.requires_grad = False
        
        # Predictor network (trainable)
        self.predictor_network = CommonCNN(output_dim=feature_dim)
        
        # Initialize target network with random weights and keep it fixed
        self._initialize_target_network()
    
    def _initialize_target_network(self):
        for param in self.target_network.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param, gain=1.0)
            else:
                nn.init.constant_(param, 0)
    
    def forward(self, obs):
        # Get features from both networks
        target_features = self.target_network(obs)
        pred_features = self.predictor_network(obs)
        
        return target_features, pred_features
    
    def compute_intrinsic_reward(self, obs):
        """
        Compute intrinsic reward based on prediction error
        
        Args:
            obs: observation tensor
            
        Returns:
            intrinsic_reward: scalar reward (mean squared prediction error)
        """
        target_features, pred_features = self.forward(obs)
        
        # Intrinsic reward = prediction error (mean squared error)
        prediction_error = torch.mean((target_features - pred_features) ** 2, dim=1)
        
        return prediction_error
    
    def compute_loss(self, obs):
        target_features, pred_features = self.forward(obs)
        
        # Only predictor network is trained, target is fixed
        rnd_loss = torch.mean((target_features.detach() - pred_features) ** 2)
        
        return rnd_loss

