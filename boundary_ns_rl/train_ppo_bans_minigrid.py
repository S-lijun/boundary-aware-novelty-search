import os
import sys
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

# 添加项目根目录到路径，以便导入 models
script_dir = os.path.dirname(os.path.abspath(__file__))  # boundary_ns_rl/
project_root = os.path.dirname(script_dir)  # 项目根目录
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rnd_cnn import RNDCNN
from models.boundary_aware_ns import BoundaryAwareNS


# ImageOnlyWrapper 不再需要，因为 ImgObsWrapper 已经将观测转换为 Box 类型


class BoundaryAwareNSRewardWrapper(RewardWrapper):
    """
    包装环境，在episode结束时添加BANS内在奖励。
    使用 gymnasium API（reset 返回 obs, info；step 返回 obs, reward, terminated, truncated, info）
    """
    
    def __init__(
        self, 
        env, 
        bans_model: BoundaryAwareNS,
        beta: float = 2.0,  # β(s_t, t)，论文中的权重因子（对应论文的 β）
    ):
        super().__init__(env)
        self.bans_model = bans_model
        self.beta = beta  # 对应论文的 β(s_t, t)
        
        # Episode跟踪
        self.episode_obs: list[np.ndarray] = []  # 累积当前episode的观测（包括初始obs）
        self.episode_actions: list[int] = []  # 累积当前episode的动作
        self.episode_external_reward = 0.0
        self.episode_intrinsic_reward = 0.0
        self.last_obs = None  # 用于保存最后一个观测
        self.episode_count = 0  # 用于调试
    
    def step(self, action):
        # gymnasium 新 API：5 个返回值
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 确保 info 是字典类型
        if not isinstance(info, dict):
            info = {}
        
        # 累积观测、动作和外部奖励
        self.episode_obs.append(obs.copy())
        self.episode_actions.append(action)
        self.episode_external_reward += reward
        
        # 计算总奖励（初始为外部奖励）
        total_reward = reward
        
        # 如果episode结束，计算BANS内在奖励
        if terminated or truncated:
            T = len(self.episode_actions)
            if T > 0:
                # 准备观测序列和动作序列
                # process_episode需要(T+1, H, W, C)的观测序列，包括初始obs和所有后续obs
                # 当前obs是最后一个，需要添加到序列中
                obs_seq_list = self.episode_obs + [obs.copy()]  # (T+1, H, W, C)
                obs_seq = np.array(obs_seq_list)
                actions_seq = np.array(self.episode_actions)  # (T,)
                
                # 使用BANS模型处理episode
                result = self.bans_model.process_episode(obs_seq, actions_seq, add_to_archive=True)
                
                novelty_hat = result["novelty_hat"]
                cumulative_cost = result["cumulative_cost"]
                
                # 论文公式：r''_t = r^ext_t + β(s_t, t) Ñ_b(τ) - λc(s_t, a_t)
                # compute_intrinsic_reward 返回 (intrinsic_reward_total, cost_penalty_total)
                intrinsic_reward_total, cost_penalty_total = self.bans_model.compute_intrinsic_reward(
                    obs_seq, actions_seq, beta=self.beta, episode_length=T
                )
                
                # 在最后一步加上内在奖励（论文中是每步加，这里在episode结束时加总和）
                total_reward = reward + intrinsic_reward_total
                self.episode_intrinsic_reward = intrinsic_reward_total
                
                # 将奖励信息添加到info中（用于日志记录）
                # 确保 info 是字典类型
                if not isinstance(info, dict):
                    info = {}
                
                # 设置 episode 奖励信息（必须在 episode 结束时设置）
                info['episode_external_reward'] = float(self.episode_external_reward)
                info['episode_intrinsic_reward'] = float(self.episode_intrinsic_reward)
                info['episode_total_reward'] = float(self.episode_external_reward + self.episode_intrinsic_reward)
                info['novelty'] = float(result["novelty"])
                info['novelty_hat'] = float(novelty_hat)
                info['n_boundary'] = int(np.sum(result["boundary_mask"]))
                info['cumulative_cost'] = float(cumulative_cost)
                info['cost_penalty'] = float(cost_penalty_total)
                info['lambda'] = float(self.bans_model.get_lambda())
                info['safe_archive_size'] = int(self.bans_model.get_safe_archive_size())
                
                # 更新 episode 计数
                self.episode_count += 1
                
                # 调试：打印前几个 episode 的信息
                if self.episode_count <= 3:
                    print(f"[BANS Wrapper] Episode {self.episode_count} ended: "
                          f"ext={self.episode_external_reward:.2f}, "
                          f"intr={self.episode_intrinsic_reward:.2f}, "
                          f"info_keys={list(info.keys())}")
                
                # 重置episode状态
                self.episode_obs = []
                self.episode_actions = []
                self.episode_external_reward = 0.0
                self.episode_intrinsic_reward = 0.0
        
        return obs, total_reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        # gymnasium 新 API：2 个返回值
        obs, info = self.env.reset(**kwargs)
        # 重置episode状态
        self.episode_obs = [obs.copy()]  # 保存初始obs
        self.episode_actions = []
        self.episode_external_reward = 0.0
        self.episode_intrinsic_reward = 0.0
        self.last_obs = None
        # 注意：不重置 episode_count，保持计数
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
    bans_model: Optional[BoundaryAwareNS] = None,
    beta: float = 2.0,  # β(s_t, t)，论文中的权重因子（对应论文的 β）
    **env_kwargs
):
    def _init():
        env = make_base_env(env_id)
        if bans_model is not None:
            env = BoundaryAwareNSRewardWrapper(env, bans_model, beta=beta)
        return env
    return _init


def make_eval_env(env_id: str = "MiniGrid-DoorKey-5x5-v0", **env_kwargs):
    def _init():
        env = make_base_env(env_id)
        return env
    return _init


class BoundaryAwareNSRewardCallback(BaseCallback):
    """
    记录BANS的episode rewards到TensorBoard。
    注意：奖励修改现在在BoundaryAwareNSRewardWrapper中进行，这个callback只负责日志记录。
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.cumulative_total_reward = 0.0
        self.episode_count = 0

    def _on_step(self) -> bool:
        """
        在每个 step 后检查是否有完成的 episode，并立即记录该 episode 的 rewards。
        """
        # 从 locals 中获取 infos 和 dones（包含 episode 结束信息）
        # 对于 vec_env，infos 和 dones 都是列表
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        
        # 确保 infos 和 dones 是列表格式
        if not isinstance(infos, list):
            infos = [infos] if infos is not None else []
        if not isinstance(dones, list):
            dones = [dones] if dones is not None else []
        
        # 只在 episode 结束时处理（检查 dones）
        # 遍历所有 infos（处理 vec_env 的情况）
        for i, info in enumerate(infos):
            # 只处理 episode 结束时的 info
            if i < len(dones) and dones[i] and isinstance(info, dict):
                # 调试：打印前几个 episode 结束时的 info 内容
                if self.episode_count < 3:
                    print(f"[BANS Callback] Episode {self.episode_count + 1} ended, info keys: {list(info.keys())}")
                
                # 检查是否有从 BoundaryAwareNSRewardWrapper 传递的 episode rewards
                if 'episode_external_reward' in info:
                    # 这些值已经是当前 episode 的总和（不是累积的）
                    external_total = float(info.get('episode_external_reward', 0.0))
                    intrinsic_total = float(info.get('episode_intrinsic_reward', 0.0))
                    episode_total = float(info.get('episode_total_reward', external_total + intrinsic_total))
                    novelty = float(info.get('novelty', 0.0))
                    novelty_hat = float(info.get('novelty_hat', 0.0))
                    n_boundary = int(info.get('n_boundary', 0))
                    cumulative_cost = float(info.get('cumulative_cost', 0.0))
                    cost_penalty = float(info.get('cost_penalty', 0.0))
                    lambda_val = float(info.get('lambda', 0.0))
                    safe_archive_size = int(info.get('safe_archive_size', 0))
                    
                    # 累积 total reward
                    self.cumulative_total_reward += episode_total
                    
                    # 立即记录当前 episode 的值
                    self.episode_count += 1
                    self.logger.record("rollout/episode_external_reward", external_total)
                    self.logger.record("rollout/episode_intrinsic_reward", intrinsic_total)
                    self.logger.record("rollout/total_reward_cumulative", self.cumulative_total_reward)
                    self.logger.record("rollout/novelty", novelty)
                    self.logger.record("rollout/novelty_hat", novelty_hat)
                    self.logger.record("rollout/n_boundary", n_boundary)
                    self.logger.record("rollout/cumulative_cost", cumulative_cost)
                    self.logger.record("rollout/cost_penalty", cost_penalty)
                    self.logger.record("rollout/lambda", lambda_val)
                    self.logger.record("rollout/safe_archive_size", safe_archive_size)
                    
                    # 调试信息（前几个 episode）
                    if self.verbose > 0 and self.episode_count <= 5:
                        print(
                            f"[BANS] ep={self.episode_count} n_boundary={n_boundary} "
                            f"novelty={novelty:.4f} hat={novelty_hat:.4f} "
                            f"cost={cumulative_cost:.4f} lambda={lambda_val:.4f} "
                            f"safe_archive={safe_archive_size} "
                            f"ext={external_total:.2f} intr={intrinsic_total:.2f} "
                            f"total={episode_total:.2f} cum={self.cumulative_total_reward:.2f}"
                        )
        
        return True


def main():
    env_id = "MiniGrid-DoorKey-5x5-v0"
    log_dir = "./logs/minigrid_ppo_bans"

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

    # Boundary-aware NS 模型
    # ===== 参数说明（针对 Minigrid DoorKey-5x5） =====
    # boundary_threshold: 边界检测阈值（归一化的像素变化率 [0,1]）
    #   - Minigrid 是图像环境，agent 移动时会有视觉变化
    #   - 推荐范围: 0.08-0.15
    #   - 建议从 0.12 开始，根据训练日志中的 n_boundary 调整
    #     理想情况：每个 episode 有 3-10 个边界状态
    #
    # archive_k: k-nearest neighbors 的 k 值
    #   - 推荐范围: 5-15
    #   - 建议: 10（默认值）
    #
    # beta: 内在奖励系数
    #   - 控制 boundary-aware novelty 对总奖励的影响
    #   - 推荐范围: 1.0-5.0
    #   - 建议从 2.0-3.0 开始，根据训练曲线调整
    #
    # lambda_cost: 代价系数
    #   - 如果不使用代价函数，设为 0.0
    #   - 如果需要惩罚重复动作等行为，可以设为 0.1-0.5
    bans_model = BoundaryAwareNS(
        feature_extractor=feature_extractor,
        archive_k=10,  # kNN 的 k 值（与 NS 一致）
        boundary_threshold=0.12,  # 边界检测阈值（针对 Minigrid 调整）
        device=device,
        cost_threshold=3.0,  # c̄: cost threshold for safe archive（提高阈值，降低惩罚）
        lambda_init=0.001,  # 初始 lambda 值（降低初始惩罚）
        lambda_lr=1e-5,  # η: lambda 学习率（降低学习率，使惩罚增长更慢）
        lambda_max=0.01,  # lambda 最大值限制（防止过度惩罚）
    )

    # 训练与评估环境
    train_env = DummyVecEnv([
        make_train_env(
            env_id,
            bans_model=bans_model,
            beta=0.01,  # β(s_t, t)，与 NS 的 beta 保持一致
        )
    ])
    eval_env = DummyVecEnv([make_eval_env(env_id)])

    # Boundary-aware NS callback（仅用于日志记录，奖励修改在wrapper中进行）
    bans_callback = BoundaryAwareNSRewardCallback(verbose=1)

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
        callback=[eval_callback, bans_callback],
        progress_bar=True,
    )

    # 保存模型
    model.save(os.path.join(log_dir, "ppo_bans_minigrid_final"))
    print("[INFO] PPO+BANS (Boundary-aware Novelty Search, Minigrid DoorKey-5x5) 训练完成")


if __name__ == "__main__":
    main()

