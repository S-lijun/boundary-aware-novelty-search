import os
import glob
import numpy as np

# 当前脚本所在目录（就是 boundary_ns_rl）
BASE_DIR = os.path.dirname(__file__)

# 轨迹文件所在目录：logs/minigrid_ppo_basic/trajectories
TRAJ_DIR = os.path.join(BASE_DIR, "logs", "minigrid_ppo_basic", "trajectories")

# 找到所有 door_key_ppo_traj_*.npz
pattern = os.path.join(TRAJ_DIR, "door_key_ppo_traj_*.npz")
files = sorted(glob.glob(pattern))

if not files:
    print("没有找到任何轨迹文件，请确认已经跑完 train_ppo_minigrid_basic.py 并成功保存轨迹。")
    print("查找路径：", pattern)
else:
    # 取最新的那个文件
    path = files[-1]
    print("加载轨迹文件:", path)
    data = np.load(path)
    print("obs shape:", data["obs"].shape)
    print("actions shape:", data["actions"].shape)
    print("rewards shape:", data["rewards"].shape)
    print("masks shape:", data["masks"].shape)
