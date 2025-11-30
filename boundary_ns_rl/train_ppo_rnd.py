import os
from typing import Optional

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces, ObservationWrapper, RewardWrapper
from gymnasium.wrappers import ResizeObservation
from minigrid.wrappers import FullyObsWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from rnd_cnn import RNDCNN


class ImageOnlyWrapper(ObservationWrapper):
    """
    只保留 obs["image"]，去掉 mission 字段，
    并把 observation_space 改成对应的 Box。
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(
            env.observation_space, spaces.Dict
        ), f"Expected Dict obs space, got {env.observation_space}"
        # 只保留 image 这个子空间
        self.observation_space = env.observation_space["image"]

    def observation(self, obs):
        # 原始 obs: {"image": ..., "mission": ...}
        return obs["image"]


class RNDRewardWrapper(RewardWrapper):
    """
    包装环境，在每个 step 添加 RND 内在奖励。
    """
    
    def __init__(self, env: gym.Env, rnd_model: RNDCNN, eta: float = 0.01, device: str = "cpu"):
        super().__init__(env)
        self.rnd_model = rnd_model
        self.eta = eta
        self.device = device
        self.last_obs = None
    
    def reward(self, reward):
        """
        在环境奖励基础上添加 RND 内在奖励。
        """
        if self.last_obs is not None:
            obs_tensor = torch.tensor(self.last_obs, dtype=torch.float32, device=self.device)
            intrinsic_reward_tensor = self.rnd_model.compute_intrinsic_reward(obs_tensor)
            intrinsic_reward = self.eta * intrinsic_reward_tensor.item()
            return reward + intrinsic_reward
        return reward
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_obs = obs.copy()
        reward = self.reward(reward)
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs.copy()
        return obs, info


def make_base_env(env_id: str, resize_to: tuple = (64, 64)):
    """
    构建最基础的 Minigrid 环境：
    - FullyObsWrapper: 全局可见
    - ImageOnlyWrapper: 只保留 image
    - ResizeObservation: 调整图像大小以适配 CNN（Minigrid 原始图像太小）
    注意：不使用 FlattenObservation，因为 CNN 需要图像格式
    """
    env = gym.make(env_id)
    env = FullyObsWrapper(env)
    env = ImageOnlyWrapper(env)
    # 调整图像大小：Minigrid 原始图像（如 5x5）太小，需要放大以适配 CNN
    env = ResizeObservation(env, resize_to)
    return env


def make_train_env(env_id: str, rnd_model: Optional[RNDCNN] = None, eta: float = 0.01, device: str = "cpu", resize_to: tuple = (64, 64)):
    """
    训练环境：基础环境包装，如果提供了 RND 模型则添加 RND 奖励包装。
    """
    def _init():
        env = make_base_env(env_id, resize_to=resize_to)
        if rnd_model is not None:
            env = RNDRewardWrapper(env, rnd_model, eta=eta, device=device)
        return env

    return _init


def make_eval_env(env_id: str, resize_to: tuple = (64, 64)):
    """
    eval 环境：不需要记录轨迹，只负责评估。
    """
    def _init():
        env = make_base_env(env_id, resize_to=resize_to)
        return env

    return _init


class RNDTrainingCallback(BaseCallback):
    """
    定期训练 RND 的 predictor network。
    """

    def __init__(
        self,
        rnd_model: RNDCNN,
        rnd_update_freq: int = 4,  # 每 N 个 rollout 更新一次 RND
        rnd_batch_size: int = 256,
        device: str = "cpu",
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.rnd_model = rnd_model
        self.rnd_update_freq = rnd_update_freq
        self.rnd_batch_size = rnd_batch_size
        self.device = device
        
        # RND optimizer
        self.rnd_optimizer = torch.optim.Adam(
            self.rnd_model.predictor_network.parameters(),
            lr=1e-3
        )
        
        # 存储观测用于训练 RND（图像格式）
        self.obs_buffer: list = []
        self.rollout_count = 0

    def _on_step(self) -> bool:
        """
        收集观测用于后续训练。
        """
        # 获取当前观测
        obs = self.locals.get("new_obs")
        
        if obs is not None:
            # 只考虑单环境的情况（DummyVecEnv 里只有一个 env）
            obs_single = obs[0]  # (H, W, C) 图像格式
            self.obs_buffer.append(obs_single.copy())
        
        return True

    def _on_rollout_end(self) -> None:
        """
        在每个 rollout 结束时，训练 RND predictor network。
        """
        self.rollout_count += 1
        
        if len(self.obs_buffer) == 0:
            return
        
        # 每 rnd_update_freq 个 rollout 更新一次 RND
        if self.rollout_count % self.rnd_update_freq == 0:
            # 准备训练数据（图像格式）
            obs_array = np.array(self.obs_buffer)  # (N, H, W, C)
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=self.device)
            
            # 随机采样一批数据
            n_samples = min(len(obs_tensor), self.rnd_batch_size)
            if n_samples > 0:
                indices = np.random.choice(len(obs_tensor), n_samples, replace=False)
                batch_obs = obs_tensor[torch.tensor(indices, dtype=torch.long)]
                
                # 计算 RND loss 并更新
                self.rnd_optimizer.zero_grad()
                rnd_loss = self.rnd_model.compute_loss(batch_obs)
                rnd_loss.backward()
                self.rnd_optimizer.step()
        
        # 清空 buffer（可选：保留一部分历史数据）
        self.obs_buffer = []


def main():
    env_id = "MiniGrid-DoorKey-5x5-v0"
    log_dir = "./logs/minigrid_ppo_rnd"
    os.makedirs(log_dir, exist_ok=True)
    
    # 确定设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 图像大小设置：Minigrid 原始图像太小，需要调整大小以适配 CNN
    resize_to = (64, 64)  # 调整到 64x64 以适配 stable_baselines3 的 CnnPolicy
    
    # 1) 先创建一个临时环境来获取观测格式
    temp_env = make_base_env(env_id, resize_to=resize_to)
    obs_sample, _ = temp_env.reset()
    print(f"[INFO] Observation shape: {obs_sample.shape}")
    
    # 确定输入通道数（Minigrid 图像通常是 (H, W, 3) RGB 格式）
    if len(obs_sample.shape) == 3:
        in_channels = obs_sample.shape[2]  # (H, W, C) 格式
    else:
        in_channels = 3  # 默认 RGB
    print(f"[INFO] Input channels: {in_channels}")

    # 2) 初始化 RND 模型（CNN 版本）
    rnd_model = RNDCNN(feature_dim=256, in_channels=in_channels).to(device)
    
    # 3) 创建训练 & eval 环境（训练环境包含 RND 奖励包装）
    eta = 0.01  # intrinsic reward scale
    train_env = DummyVecEnv([make_train_env(env_id, rnd_model=rnd_model, eta=eta, device=device, resize_to=resize_to)])
    eval_env = DummyVecEnv([make_eval_env(env_id, resize_to=resize_to)])
    
    # 4) 创建 RND 训练 callback
    rnd_callback = RNDTrainingCallback(
        rnd_model=rnd_model,
        rnd_update_freq=4,  # 每 4 个 rollout 更新一次 RND
        rnd_batch_size=256,
        device=device,
        verbose=0,
    )

    # 5) 定义 PPO 模型（使用 CnnPolicy 因为现在是图像输入）
    model = PPO(
        "CnnPolicy",
        train_env,
        verbose=1,
        tensorboard_log=log_dir,
    )

    # 6) 评估回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=10_000,
        deterministic=True,
        render=False,
    )

    # 7) 训练（同时挂两个 callback）
    total_timesteps = 200_000
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, rnd_callback],
        progress_bar=True,  # 需要安装 tqdm 和 rich
    )

    # 8) 保存 PPO 模型
    model.save(os.path.join(log_dir, "ppo_rnd_minigrid_final"))
    
    # 9) 保存 RND 模型
    rnd_save_path = os.path.join(log_dir, "rnd_model.pt")
    torch.save(rnd_model.state_dict(), rnd_save_path)
    print(f"[INFO] Saved RND model to {rnd_save_path}")
    
    print("[INFO] PPO+RND 训练完成")


if __name__ == "__main__":
    main()

