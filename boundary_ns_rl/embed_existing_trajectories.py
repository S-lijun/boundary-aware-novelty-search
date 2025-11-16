import os
import glob
import numpy as np

import torch

from novelty.behavior_embedding import BehaviorEmbedder


LOG_ROOT = os.path.join("logs", "minigrid_ppo_basic")
TRAJ_DIR = os.path.join(LOG_ROOT, "trajectories")
ENCODER_PATH = os.path.join(LOG_ROOT, "encoder", "door_key_encoder.pt")


def find_latest_traj_file():
    pattern = os.path.join(TRAJ_DIR, "door_key_ppo_traj_*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"在 {TRAJ_DIR} 下找不到任何 door_key_ppo_traj_*.npz 文件")
    return files[-1]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[embed] 使用设备:", device)

    traj_path = find_latest_traj_file()
    print("[embed] 使用轨迹文件:", traj_path)

    data = np.load(traj_path)
    obs = data["obs"][:, :-1, :]   # (N, T, obs_dim)
    masks = data["masks"]          # (N, T)

    N, T, obs_dim = obs.shape
    print(f"[embed] 轨迹数据: N={N}, T={T}, obs_dim={obs_dim}")

    embedder = BehaviorEmbedder(
        obs_dim=obs_dim,
        encoder_path=ENCODER_PATH,
        hidden_dim=128,
        latent_dim=64,
        device=device,
    )

    z = embedder.embed_np(obs, masks, batch_size=64)  # (N, latent_dim)
    print("[embed] 得到 embedding z 形状:", z.shape)

    # 看一下 embedding 的基本统计
    mean_norm = np.linalg.norm(z, axis=1).mean()
    std_norm = np.linalg.norm(z, axis=1).std()
    print(f"[embed] z 的 L2 范数: mean={mean_norm:.4f}, std={std_norm:.4f}")

    # 随机抽一些 pairwise distance 看看范围
    if N >= 2:
        idx = np.random.choice(N, size=min(50, N), replace=False)
        z_sub = z[idx]  # (K, latent_dim)
        # 计算两两距离矩阵
        dists = np.linalg.norm(z_sub[:, None, :] - z_sub[None, :, :], axis=-1)
        # 只看上三角非对角线
        tri = dists[np.triu_indices_from(dists, k=1)]
        print(
            f"[embed] 随机子集 pairwise 距离: "
            f"mean={tri.mean():.4f}, std={tri.std():.4f}, "
            f"min={tri.min():.4f}, max={tri.max():.4f}"
        )

    # 可选：保存一份 embedding 到文件，后面可以用来做可视化
    out_path = os.path.join(LOG_ROOT, "encoder", "door_key_embeddings.npy")
    np.save(out_path, z)
    print("[embed] 已保存所有 embedding 到:", out_path)


if __name__ == "__main__":
    main()
