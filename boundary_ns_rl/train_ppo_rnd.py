import os
from typing import Optional

import numpy as np
import torch

# 旧版 gym（procgen 环境注册在这里）
import gym
import procgen  # 触发 procgen 注册到 gym

# Gymnasium（用于包装器和 Stable-Baselines3）
from gymnasium import RewardWrapper, Env, spaces

# Minigrid 相关导入（已注释，保留用于参考）
# from gymnasium import spaces, ObservationWrapper
# from gymnasium.wrappers import ResizeObservation
# from minigrid.wrappers import FullyObsWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from rnd_cnn import RNDCNN


# ===== Minigrid 相关包装器（已注释，保留用于参考） =====
# class ImageOnlyWrapper(ObservationWrapper):
#     """
#     只保留 obs["image"]，去掉 mission 字段，
#     并把 observation_space 改成对应的 Box。
#     """
# 
#     def __init__(self, env: gym.Env):
#         super().__init__(env)
#         assert isinstance(
#             env.observation_space, spaces.Dict
#         ), f"Expected Dict obs space, got {env.observation_space}"
#         # 只保留 image 这个子空间
#         self.observation_space = env.observation_space["image"]
# 
#     def observation(self, obs):
#         # 原始 obs: {"image": ..., "mission": ...}
#         return obs["image"]


class OldGymEnvCompatibility(Env):
    """
    将旧版 gym.Env 转换为 gymnasium API：
    - reset() 始终返回 (obs, info)
    - step() 始终返回 (obs, reward, terminated, truncated, info)
    """

    @staticmethod
    def _convert_space(space):
        # 将 gym.Space 转换为 gymnasium.spaces，SB3 需要后者
        if isinstance(space, gym.spaces.Discrete):
            return spaces.Discrete(space.n)
        if isinstance(space, gym.spaces.MultiBinary):
            return spaces.MultiBinary(space.n)
        if isinstance(space, gym.spaces.MultiDiscrete):
            return spaces.MultiDiscrete(space.nvec)
        if isinstance(space, gym.spaces.Box):
            return spaces.Box(low=np.array(space.low, dtype=space.dtype),
                              high=np.array(space.high, dtype=space.dtype),
                              shape=space.shape,
                              dtype=space.dtype)
        # 兜底：原样返回（可能 SB3 不支持）
        return space

    def __init__(self, env):
        super().__init__()
        self.env = env
        # 透传空间与元数据，满足 SB3 以及 Gymnasium 的属性访问
        self.observation_space = self._convert_space(env.observation_space)
        self.action_space = self._convert_space(env.action_space)
        self.reward_range = getattr(env, "reward_range", None)
        self.metadata = getattr(env, "metadata", {})
        self.spec = getattr(env, "spec", None)

    def reset(self, **kwargs):
        try:
            result = self.env.reset(**kwargs)
        except TypeError:
            # 旧环境可能不接受 seed/return_info 等参数
            result = self.env.reset()

        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        return obs, info

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            return result

        obs, reward, done, info = result
        truncated = False
        if isinstance(info, dict):
            truncated = bool(
                info.pop("TimeLimit.truncated", False) or info.pop("truncated", False)
            )
        return obs, reward, bool(done), truncated, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()


class RNDRewardWrapper(RewardWrapper):
    """
    包装环境，在每个 step 添加 RND 内在奖励。
    使用 gymnasium API（reset 返回 obs, info；step 返回 obs, reward, terminated, truncated, info）
    """
    
    def __init__(self, env, rnd_model: RNDCNN, eta: float = 0.01, device: str = "cpu"):
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
        # gymnasium 新 API：5 个返回值
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_obs = obs.copy()
        reward = self.reward(reward)
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        # gymnasium 新 API：2 个返回值
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs.copy()
        return obs, info


# ===== Minigrid 环境创建函数（已注释，保留用于参考） =====
# def make_base_env(env_id: str, resize_to: tuple = (64, 64)):
#     """
#     构建最基础的 Minigrid 环境：
#     - FullyObsWrapper: 全局可见
#     - ImageOnlyWrapper: 只保留 image
#     - ResizeObservation: 调整图像大小以适配 CNN（Minigrid 原始图像太小）
#     注意：不使用 FlattenObservation，因为 CNN 需要图像格式
#     """
#     env = gym.make(env_id)
#     env = FullyObsWrapper(env)
#     env = ImageOnlyWrapper(env)
#     # 调整图像大小：Minigrid 原始图像（如 5x5）太小，需要放大以适配 CNN
#     env = ResizeObservation(env, resize_to)
#     return env

# ===== Procgen Maze 环境创建函数 =====
def make_base_env(
    env_id: str = "procgen:procgen-maze-v0",  # 统一使用旧版 gym 格式
    num_levels: int = 0,  # 0 = 无限关卡，>0 = 固定关卡数
    start_level: int = 0,  # 起始关卡种子
    distribution_mode: str = "hard",  # "easy", "hard", "extreme", "memory", "exploration"
    use_backgrounds: bool = True,  # 是否使用背景
    restrict_themes: bool = False,  # 是否限制主题
    use_monochrome_assets: bool = False,  # 是否使用单色资源
    center_agent: bool = True,  # 观察是否以智能体为中心
    use_sequential_levels: bool = False,  # 是否使用连续关卡
    paint_vel_info: bool = False,  # 是否绘制速度信息
    use_generated_assets: bool = False,  # 是否使用生成的资源
    **kwargs  # 其他参数
):
    """
    构建 Procgen Maze 环境。
    Procgen 环境已经是图像格式 (64, 64, 3)，不需要额外的包装器。
    
    参数说明：
    - num_levels: 可生成的独特关卡数量，0 表示无限关卡
    - start_level: 生成关卡时使用的最小种子值
    - distribution_mode: 难度级别 ("easy", "hard", "extreme", "memory", "exploration")
    - use_backgrounds: 是否使用人类设计的背景
    - restrict_themes: 是否限制为单一主题
    - use_monochrome_assets: 是否使用单色矩形代替设计的资产
    - center_agent: 观察是否以智能体为中心
    - use_sequential_levels: 到达关卡末尾是否继续生成新关卡
    - paint_vel_info: 是否在屏幕左上角绘制速度信息
    - use_generated_assets: 是否使用随机生成的资产
    """
    # env_id 兜底处理：如果传入错误格式，自动修正
    if env_id and "procgen" in env_id.lower() and "maze" in env_id.lower():
        # 如果 env_id 不是正确的格式，修正为旧版 gym 格式
        if not env_id.startswith("procgen:procgen-maze"):
            env_id = "procgen:procgen-maze-v0"
    else:
        # 如果没有提供或格式不对，使用默认值
        env_id = "procgen:procgen-maze-v0"
    
    # 1. 旧 gym 里创建 Procgen 环境
    base_env = gym.make(
        env_id,  # 统一使用 "procgen:procgen-maze-v0" 格式
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode=distribution_mode,
        use_backgrounds=use_backgrounds,
        restrict_themes=restrict_themes,
        use_monochrome_assets=use_monochrome_assets,
        center_agent=center_agent,
        use_sequential_levels=use_sequential_levels,
        paint_vel_info=paint_vel_info,
        use_generated_assets=use_generated_assets,
        **kwargs
    )
    
    # 2. 转成 gymnasium.Env（显式转换，确保类型正确）
    env = OldGymEnvCompatibility(base_env)
    return env


# ===== Minigrid 环境创建函数（已注释，保留用于参考） =====
# def make_train_env(env_id: str, rnd_model: Optional[RNDCNN] = None, eta: float = 0.01, device: str = "cpu", resize_to: tuple = (64, 64)):
#     """
#     训练环境：基础环境包装，如果提供了 RND 模型则添加 RND 奖励包装。
#     """
#     def _init():
#         env = make_base_env(env_id, resize_to=resize_to)
#         if rnd_model is not None:
#             env = RNDRewardWrapper(env, rnd_model, eta=eta, device=device)
#         return env
# 
#     return _init
# 
# 
# def make_eval_env(env_id: str, resize_to: tuple = (64, 64)):
#     """
#     eval 环境：不需要记录轨迹，只负责评估。
#     """
#     def _init():
#         env = make_base_env(env_id, resize_to=resize_to)
#         return env
# 
#     return _init

# ===== Procgen Maze 环境创建函数 =====
def make_train_env(
    env_id: str = "procgen:procgen-maze-v0",  # 统一使用旧版 gym 格式
    rnd_model: Optional[RNDCNN] = None,
    eta: float = 0.01,
    device: str = "cpu",
    **env_kwargs  # Procgen 环境参数
):
    """
    训练环境：基础环境包装，如果提供了 RND 模型则添加 RND 奖励包装。
    """
    def _init():
        env = make_base_env(env_id, **env_kwargs)
        if rnd_model is not None:
            env = RNDRewardWrapper(env, rnd_model, eta=eta, device=device)
        return env

    return _init


def make_eval_env(env_id: str = "procgen:procgen-maze-v0", **env_kwargs):  # 统一使用旧版 gym 格式
    """
    eval 环境：不需要记录轨迹，只负责评估。
    """
    def _init():
        env = make_base_env(env_id, **env_kwargs)
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
    # ===== Minigrid 配置（已注释，保留用于参考） =====
    # env_id = "MiniGrid-DoorKey-5x5-v0"
    # log_dir = "./logs/minigrid_ppo_rnd"
    # resize_to = (64, 64)  # 调整到 64x64 以适配 stable_baselines3 的 CnnPolicy
    
    # ===== Procgen Maze 配置 =====
    env_id = "procgen:procgen-maze-v0"  # 统一使用旧版 gym 格式
    log_dir = "./logs/procgen_maze_ppo_rnd"
    
    # Procgen 环境参数配置
    procgen_kwargs = {
        "num_levels": 200,  # 固定 200 个关卡
        "start_level": 0,  # 起始关卡种子
        "distribution_mode": "easy",  # 难度级别: "easy" (先试 easy，可以改成 "hard")
        "render_mode": None,  # 渲染模式
        # 其他可选参数（使用默认值）
        # "use_backgrounds": True,
        # "restrict_themes": False,
        # "use_monochrome_assets": False,
        # "center_agent": True,
        # "use_sequential_levels": False,
        # "paint_vel_info": False,
        # "use_generated_assets": False,
    }
    
    os.makedirs(log_dir, exist_ok=True)
    
    # 确定设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Environment: {env_id}")
    print(f"[INFO] Procgen parameters: {procgen_kwargs}")
    
    # 1) 先创建一个临时环境来获取观测格式
    temp_env = make_base_env(env_id, **procgen_kwargs)
    # gymnasium 新 API：reset() 返回 (obs, info)
    obs_sample, info = temp_env.reset()
    
    # 调试信息：确认环境类型
    print(f"[DEBUG] RNDRewardWrapper MRO: {RNDRewardWrapper.__mro__}")
    print(f"[DEBUG] temp_env type: {type(temp_env)}")
    print(f"[INFO] Observation shape: {obs_sample.shape}")
    print(f"[INFO] Observation space: {temp_env.observation_space}")
    print(f"[INFO] Action space: {temp_env.action_space}")
    
    # 确定输入通道数（Procgen 图像通常是 (H, W, 3) RGB 格式）
    if len(obs_sample.shape) == 3:
        in_channels = obs_sample.shape[2]  # (H, W, C) 格式
    else:
        in_channels = 3  # 默认 RGB
    print(f"[INFO] Input channels: {in_channels}")

    # 2) 初始化 RND 模型（CNN 版本）
    rnd_model = RNDCNN(feature_dim=256, in_channels=in_channels).to(device)
    
    # 3) 创建训练 & eval 环境（训练环境包含 RND 奖励包装）
    eta = 0.01  # intrinsic reward scale
    train_env = DummyVecEnv([make_train_env(env_id, rnd_model=rnd_model, eta=eta, device=device, **procgen_kwargs)])
    eval_env = DummyVecEnv([make_eval_env(env_id, **procgen_kwargs)])
    
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
    # model.save(os.path.join(log_dir, "ppo_rnd_minigrid_final"))  # Minigrid 版本
    model.save(os.path.join(log_dir, "ppo_rnd_procgen_maze_final"))  # Procgen Maze 版本
    
    # 9) 保存 RND 模型
    rnd_save_path = os.path.join(log_dir, "rnd_model.pt")
    torch.save(rnd_model.state_dict(), rnd_save_path)
    print(f"[INFO] Saved RND model to {rnd_save_path}")
    
    print("[INFO] PPO+RND (Procgen Maze) 训练完成")


if __name__ == "__main__":
    main()
