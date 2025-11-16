import os
from typing import Tuple

import numpy as np
import torch

from encoder.trajectory_encoder import TrajectoryEncoder


class BehaviorEmbedder:
    """
    加载已经训练好的 TrajectoryEncoder，
    提供把 (obs_seq, mask) → latent 向量 z 的接口。
    """

    def __init__(
        self,
        obs_dim: int,
        encoder_path: str,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.encoder = TrajectoryEncoder(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        ).to(self.device)

        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"找不到 encoder 权重文件: {encoder_path}")

        state_dict = torch.load(encoder_path, map_location=self.device)
        self.encoder.load_state_dict(state_dict)
        self.encoder.eval()

    @torch.no_grad()
    def embed_torch(
        self, obs: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        """
        obs:   (B, T, obs_dim) float32
        masks: (B, T) float32
        返回:
          z:   (B, latent_dim)
        """
        obs = obs.to(self.device)
        masks = masks.to(self.device)
        z = self.encoder(obs, masks)
        return z.cpu()

    @torch.no_grad()
    def embed_np(
        self, obs_np: np.ndarray, masks_np: np.ndarray, batch_size: int = 64
    ) -> np.ndarray:
        """
        方便直接用 numpy 调用：
        obs_np:   (N, T, obs_dim)
        masks_np: (N, T)
        返回:
          z_np:    (N, latent_dim)
        """
        N, T, obs_dim = obs_np.shape
        assert masks_np.shape == (N, T)

        zs = []
        idx = 0
        while idx < N:
            j = min(idx + batch_size, N)
            obs_batch = torch.from_numpy(obs_np[idx:j]).float()
            mask_batch = torch.from_numpy(masks_np[idx:j]).float()
            z_batch = self.embed_torch(obs_batch, mask_batch)  # (B, latent_dim)
            zs.append(z_batch.numpy())
            idx = j

        z_np = np.concatenate(zs, axis=0)  # (N, latent_dim)
        return z_np
