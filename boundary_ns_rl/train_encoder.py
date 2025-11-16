import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from encoder.trajectory_encoder import TrajectoryReturnModel


# ---------- 一些超参数 ----------
LOG_ROOT = os.path.join("logs", "minigrid_ppo_basic")
TRAJ_DIR = os.path.join(LOG_ROOT, "trajectories")

BATCH_SIZE = 32
NUM_EPOCHS = 20
LR = 1e-3
GAMMA = 0.99  # 如果你想算 discounted return，可以用；现在先用 sum of rewards 即可


class TrajectoryDataset(Dataset):
    """
    从 .npz 文件里读出 (obs_seq, mask, return) 的 Dataset。
    """

    def __init__(self, npz_path: str):
        super().__init__()
        data = np.load(npz_path)

        # 原始形状：
        # obs:    (N, T+1, obs_dim)
        # actions:(N, T)
        # rewards:(N, T)
        # masks:  (N, T)
        obs = data["obs"]          # (N, T+1, obs_dim)
        rewards = data["rewards"]  # (N, T)
        masks = data["masks"]      # (N, T)

        self.obs = obs[:, :-1, :]  # 对齐 rewards → (N, T, obs_dim)
        self.masks = masks         # (N, T)

        # 这里先用“总 reward”作为 episode return
        # 你也可以改成 discounted return
        self.returns = (rewards * masks).sum(axis=1)  # (N,)

        # 转成 torch Tensor
        self.obs = torch.from_numpy(self.obs).float()
        self.masks = torch.from_numpy(self.masks).float()
        self.returns = torch.from_numpy(self.returns).float()

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        return self.obs[idx], self.masks[idx], self.returns[idx]


def find_latest_traj_file():
    pattern = os.path.join(TRAJ_DIR, "door_key_ppo_traj_*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"在 {TRAJ_DIR} 下找不到任何 door_key_ppo_traj_*.npz 轨迹文件")
    return files[-1]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[train_encoder] 使用设备:", device)

    traj_path = find_latest_traj_file()
    print("[train_encoder] 使用轨迹文件:", traj_path)

    dataset = TrajectoryDataset(traj_path)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    N, T, obs_dim = dataset.obs.shape
    print(f"[train_encoder] 数据规模: N={N}, T={T}, obs_dim={obs_dim}")

    model = TrajectoryReturnModel(obs_dim=obs_dim, hidden_dim=128, latent_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss = 0.0
        count = 0

        for batch in dataloader:
            obs, masks, returns = batch
            obs = obs.to(device)          # (B, T, obs_dim)
            masks = masks.to(device)      # (B, T)
            returns = returns.to(device)  # (B,)

            optimizer.zero_grad()
            pred_return, z = model(obs, masks)
            loss = criterion(pred_return, returns)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * obs.size(0)
            count += obs.size(0)

        avg_loss = total_loss / max(count, 1)
        print(f"[Epoch {epoch:02d}] train MSE loss = {avg_loss:.4f}")

    # 训练结束，单独保存 encoder 的参数
    encoder_dir = os.path.join(LOG_ROOT, "encoder")
    os.makedirs(encoder_dir, exist_ok=True)
    encoder_path = os.path.join(encoder_dir, "door_key_encoder.pt")
    torch.save(model.encoder.state_dict(), encoder_path)
    print("[train_encoder] 已保存 encoder 权重到:", encoder_path)


if __name__ == "__main__":
    main()
