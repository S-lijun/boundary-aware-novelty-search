import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TrajectoryEncoder(nn.Module):
    """
    输入一条完整轨迹 (obs_seq, mask)，输出一个固定长度的 embedding z。

    obs:  (B, T, obs_dim)
    mask: (B, T)  二值，1=真实步，0=padding
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 128, latent_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # 先把原始 obs 投到 hidden 维度
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        # GRU 读整个序列
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        # 最后一个 hidden → latent z
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, obs, mask):
        """
        obs:  (B, T, obs_dim)
        mask: (B, T)
        返回:
          z: (B, latent_dim)
        """
        # 1) 投影 + 激活
        x = torch.relu(self.obs_proj(obs))  # (B, T, hidden_dim)

        # 2) 由 mask 计算每条轨迹的真实长度
        lengths = mask.sum(dim=1).long()  # (B,)

        # pack_padded_sequence 需要按长度降序排序
        lengths_sorted, perm_idx = lengths.sort(descending=True)
        x_sorted = x[perm_idx]

        packed = pack_padded_sequence(
            x_sorted,
            lengths_sorted.cpu(),
            batch_first=True,
            enforce_sorted=True,
        )
        packed_out, h_n = self.gru(packed)
        # h_n: (num_layers, B, hidden_dim)，这里只有 1 层 → 取第 0 层
        h_last_sorted = h_n[0]  # (B, hidden_dim)

        # 按原顺序还原
        _, unperm_idx = perm_idx.sort()
        h_last = h_last_sorted[unperm_idx]

        z = self.fc(h_last)  # (B, latent_dim)
        return z


class TrajectoryReturnModel(nn.Module):
    """
    一个简单的“encoder + head”，用来预测整条轨迹的 return。
    训练完以后，我们主要用 encoder 部分的参数。
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 128, latent_dim: int = 64):
        super().__init__()
        self.encoder = TrajectoryEncoder(obs_dim, hidden_dim, latent_dim)
        self.head = nn.Linear(latent_dim, 1)

    def forward(self, obs, mask):
        """
        obs:  (B, T, obs_dim)
        mask: (B, T)

        返回:
          pred_return: (B,)
          z:           (B, latent_dim)
        """
        z = self.encoder(obs, mask)
        pred = self.head(z).squeeze(-1)
        return pred, z
