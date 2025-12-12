import os
from typing import Optional

import numpy as np
import torch

import gymnasium as gym
from gymnasium import RewardWrapper
from gymnasium.wrappers import ResizeObservation
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper, ImgObsWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from rnd_cnn import RNDCNN


# ImageOnlyWrapper 不再需要，因为 ImgObsWrapper 已经将观测转换为 Box 类型


class NSRewardWrapper(RewardWrapper):
    """
    包装环境，在episode结束时添加NS内在奖励。
    使用 gymnasium API（reset 返回 obs, info；step 返回 obs, reward, terminated, truncated, info）
    """
    
    def __init__(
        self, 
        env, 
        feature_extractor: RNDCNN, 
        archive_k: int = 10, 
        beta: float = 2.0, 
        device: str = "cpu",
        novelty_clip_min: Optional[float] = -1.0,
        novelty_clip_max: Optional[float] = None,
    ):
        super().__init__(env)
        self.feature_extractor = feature_extractor
        self.beta = beta
        self.device = device
        self.archive_k = archive_k
        self.novelty_clip_min = novelty_clip_min
        self.novelty_clip_max = novelty_clip_max
        
        # NS状态：archive和历史
        self.archive: list[np.ndarray] = []  # list of embeddings (np.array)
        self.novelty_history: list[float] = []  # 存历史新颖度用于 median-IQR 归一化
        
        # Episode跟踪
        self.episode_obs: list[np.ndarray] = []  # 累积当前episode的观测
        self.episode_external_reward = 0.0
        self.episode_intrinsic_reward = 0.0
    
    def _compute_embedding(self, obs_seq: np.ndarray) -> np.ndarray:
        """obs_seq: (T, H, W, C) uint8/float -> embedding vector"""
        with torch.no_grad():
            # 归一化到 [0,1]，转 torch，形状 (T, H, W, C)
            obs_tensor = torch.tensor(obs_seq, dtype=torch.float32, device=self.device) / 255.0
            # 特征提取，使用 target_network（固定随机权重）
            target_features, _ = self.feature_extractor.forward(obs_tensor)
            # target_features: (T, feature_dim)
            feat_np = target_features.cpu().numpy()
            z = feat_np.mean(axis=0)  # 简单平均作为 episode embedding
            # L2 归一化，避免尺度过小
            norm = np.linalg.norm(z) + 1e-8
            z = z / norm
            return z
    
    def _compute_novelty(self, z: np.ndarray) -> float:
        if len(self.archive) == 0:
            return 0.0
        A = np.array(self.archive)  # (N, dim)
        dists = np.linalg.norm(A - z[None, :], axis=1)
        k = min(self.archive_k, len(self.archive))
        topk = np.partition(dists, k - 1)[:k]
        return float(topk.mean())
    
    def step(self, action):
        # gymnasium 新 API：5 个返回值
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 累积观测和外部奖励
        self.episode_obs.append(obs.copy())
        self.episode_external_reward += reward
        
        # 计算总奖励（初始为外部奖励）
        total_reward = reward
        
        # 如果episode结束，计算NS内在奖励
        if terminated or truncated:
            T = len(self.episode_obs)
            if T > 0:
                # 计算embedding和新颖度
                obs_seq = np.array(self.episode_obs)  # (T, H, W, C)
                z = self._compute_embedding(obs_seq)
                novelty = self._compute_novelty(z)
                
                # 第一个episode特殊处理：给予基础新颖度值，避免0值影响统计
                if len(self.archive) == 0:
                    novelty = 1.0  # 给予一个非零的基础值
                
                # 更新历史
                self.novelty_history.append(novelty)
                self.archive.append(z.astype(np.float32))
                
                # median-IQR 归一化（按照论文）
                if len(self.novelty_history) >= 3:
                    med = np.median(self.novelty_history)
                    iqr = np.percentile(self.novelty_history, 75) - np.percentile(self.novelty_history, 25)
                    if iqr > 1e-6:
                        novelty_norm = (novelty - med) / (iqr + 1e-8)
                    else:
                        novelty_norm = (novelty - med) / (med + 1e-8) if med > 1e-8 else 0.0
                elif len(self.novelty_history) == 2:
                    prev_novelty = self.novelty_history[0]
                    if novelty > prev_novelty:
                        novelty_norm = (novelty - prev_novelty) / (prev_novelty + 1e-8)
                    else:
                        novelty_norm = 0.0
                else:
                    novelty_norm = 0.0
                
                # ReLU确保非负
                novelty_hat = max(0.0, novelty_norm)
                
                # 确保 novelty_hat 有一个最小值
                if novelty_hat == 0.0 and novelty > 0:
                    if len(self.novelty_history) > 1:
                        min_novelty = min(self.novelty_history)
                        max_novelty = max(self.novelty_history)
                        if max_novelty > min_novelty:
                            relative_novelty = (novelty - min_novelty) / (max_novelty - min_novelty + 1e-8)
                            novelty_hat = max(0.1, relative_novelty * 0.5)
                        else:
                            novelty_hat = 0.1
                    else:
                        novelty_hat = 0.1
                
                # 限制上界避免极端值
                novelty_hat = min(novelty_hat, 5.0)
                
                # Reward shaping：beta * novelty_hat * T
                intrinsic_reward = self.beta * novelty_hat * T
                total_reward = reward + intrinsic_reward
                self.episode_intrinsic_reward = intrinsic_reward
                
                # 将奖励信息添加到info中（用于日志记录）
                if not isinstance(info, dict):
                    info = {}
                info['episode_external_reward'] = self.episode_external_reward
                info['episode_intrinsic_reward'] = self.episode_intrinsic_reward
                info['episode_total_reward'] = self.episode_external_reward + self.episode_intrinsic_reward
                info['novelty'] = novelty
                info['novelty_hat'] = novelty_hat
                
                # 重置episode状态
                self.episode_obs = []
                self.episode_external_reward = 0.0
                self.episode_intrinsic_reward = 0.0
        
        return obs, total_reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        # gymnasium 新 API：2 个返回值
        obs, info = self.env.reset(**kwargs)
        # 重置episode状态
        self.episode_obs = [obs.copy()]
        self.episode_external_reward = 0.0
        self.episode_intrinsic_reward = 0.0
        return obs, info


def make_base_env(env_id: str):
    """
    构建最基础的 Minigrid 环境：
    - FullyObsWrapper: 全局可见
    - RGBImgObsWrapper: 转换为 RGB 图像
    - ImgObsWrapper: 提取图像观测（已经将 Dict 转换为 Box）
    - ResizeObservation: 调整图像大小以适配 CNN
    """
    env = gym.make(env_id)
    env = FullyObsWrapper(env)
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)  # 这个 wrapper 已经将观测从 Dict 转换为 Box
    # 调整图像大小：Minigrid 原始图像太小，需要放大以适配 CNN
    env = ResizeObservation(env, (64, 64))
    
    # 验证观测空间
    obs, _ = env.reset()
    if not isinstance(obs, np.ndarray):
        raise ValueError(f"Expected numpy array observation, got {type(obs)}")
    if len(obs.shape) != 3 or obs.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) observation shape, got {obs.shape}")
    if obs.shape[0] != 64 or obs.shape[1] != 64:
        raise ValueError(f"Expected (64, 64, 3) observation shape after resize, got {obs.shape}")
    
    return env


def make_train_env(
    env_id: str = "MiniGrid-DoorKey-5x5-v0", 
    feature_extractor: Optional[RNDCNN] = None,
    archive_k: int = 10,
    beta: float = 2.0,
    device: str = "cpu",
    novelty_clip_min: Optional[float] = -1.0,
    novelty_clip_max: Optional[float] = None,
    **env_kwargs
):
    def _init():
        env = make_base_env(env_id)
        if feature_extractor is not None:
            env = NSRewardWrapper(
                env, 
                feature_extractor, 
                archive_k=archive_k, 
                beta=beta, 
                device=device,
                novelty_clip_min=novelty_clip_min,
                novelty_clip_max=novelty_clip_max
            )
        return env
    return _init


def make_eval_env(env_id: str = "MiniGrid-DoorKey-5x5-v0", **env_kwargs):
    def _init():
        env = make_base_env(env_id)
        return env
    return _init


class NSRewardLoggingCallback(BaseCallback):
    """
    记录NS的episode rewards到TensorBoard。
    注意：奖励修改现在在NSRewardWrapper中进行，这个callback只负责日志记录。
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.cumulative_total_reward = 0.0
        self.episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones")
        if dones is None:
            return True
        
        # 单环境
        done = dones[0]
        if not done:
            return True

        # 从info中获取episode奖励信息（由NSRewardWrapper提供）
        if len(infos) > 0 and isinstance(infos[0], dict):
            info = infos[0]
            external_total = info.get('episode_external_reward', 0.0)
            intrinsic_total = info.get('episode_intrinsic_reward', 0.0)
            episode_total = info.get('episode_total_reward', external_total + intrinsic_total)
            novelty = info.get('novelty', 0.0)
            novelty_hat = info.get('novelty_hat', 0.0)
            
            self.cumulative_total_reward += episode_total
            self.episode_count += 1
            
            # 记录到TensorBoard
            self.logger.record("rollout/episode_external_reward", external_total)
            self.logger.record("rollout/episode_intrinsic_reward", intrinsic_total)
            self.logger.record("rollout/total_reward_cumulative", self.cumulative_total_reward)
            self.logger.record("rollout/novelty", novelty)
            self.logger.record("rollout/novelty_hat", novelty_hat)
            
            if self.verbose > 0 and self.episode_count <= 5:
                print(
                    f"[NS] ep={self.episode_count} novelty={novelty:.4f} "
                    f"hat={novelty_hat:.4f} ext={external_total:.2f} intr={intrinsic_total:.2f} "
                    f"total={episode_total:.2f} cum={self.cumulative_total_reward:.2f}"
                )

        return True


def main():
    env_id = "MiniGrid-DoorKey-5x5-v0"
    log_dir = "./logs/minigrid_ppo_ns"

    os.makedirs(log_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 创建临时环境以获取观测形状
    temp_env = make_base_env(env_id)
    obs_sample, _ = temp_env.reset()
    # Minigrid 观测形状: (H, W, C) = (64, 64, 3) after resize
    in_channels = obs_sample.shape[2] if len(obs_sample.shape) == 3 else 3
    print(f"[INFO] Observation shape: {obs_sample.shape}")

    # 特征提取器：使用 RND 的 target 网络（固定随机权重）
    feature_extractor = RNDCNN(feature_dim=256, in_channels=in_channels).to(device)

    # 训练与评估环境
    train_env = DummyVecEnv([
        make_train_env(
            env_id, 
            feature_extractor=feature_extractor,
            archive_k=10,
            beta=0.01,  # NS奖励系数
            device=device,
            novelty_clip_min=None,
            novelty_clip_max=None,
        )
    ])
    eval_env = DummyVecEnv([make_eval_env(env_id)])

    # NS callback（仅用于日志记录，奖励修改在wrapper中进行）
    ns_callback = NSRewardLoggingCallback(verbose=1)

    # PPO 模型
    model = PPO(
        "CnnPolicy",
        train_env,
        verbose=1,
        tensorboard_log=log_dir,
    )

    # 评估回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=10_000,
        deterministic=True,
        render=False,
    )

    # 训练
    total_timesteps = 200_000  # NS 和 BANS 使用更长的训练步数
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, ns_callback],
        progress_bar=True,
    )

    # 保存模型
    model.save(os.path.join(log_dir, "ppo_ns_minigrid_final"))
    print("[INFO] PPO+NS (Minigrid DoorKey-5x5) 训练完成")


if __name__ == "__main__":
    main()

