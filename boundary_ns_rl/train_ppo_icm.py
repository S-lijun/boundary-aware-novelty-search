import os
from typing import Optional

import numpy as np
import torch

# 旧版 gym（procgen 环境注册在这里）
import gym
import procgen  # 触发 procgen 注册到 gym

# Gymnasium（用于包装器和 Stable-Baselines3）
from gymnasium import RewardWrapper, Env, spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from icm_cnn import ICMCNN


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


class ICMRewardWrapper(RewardWrapper):
    """
    包装环境，在每个 step 添加 ICM 内在奖励。
    使用 gymnasium API（reset 返回 obs, info；step 返回 obs, reward, terminated, truncated, info）
    """
    
    def __init__(self, env, icm_model: ICMCNN, eta: float = 0.01, device: str = "cpu"):
        super().__init__(env)
        self.icm_model = icm_model
        self.eta = eta
        self.device = device
        self.last_obs = None
        self.last_action = None
        # 用于跟踪 episode 的 rewards
        self.episode_external_reward = 0.0
        self.episode_intrinsic_reward = 0.0
    
    def reward(self, reward, obs, next_obs, action):
        """
        在环境奖励基础上添加 ICM 内在奖励。
        返回总奖励（external + intrinsic），同时计算 intrinsic reward 用于记录。
        """
        if obs is not None and next_obs is not None:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
            action_tensor = torch.tensor(action, dtype=torch.long, device=self.device)
            
            intrinsic_reward_tensor = self.icm_model.compute_intrinsic_reward(
                obs_tensor, next_obs_tensor, action_tensor
            )
            intrinsic_reward = self.eta * intrinsic_reward_tensor.item()
            # 跟踪 episode 的 rewards
            self.episode_external_reward += reward
            self.episode_intrinsic_reward += intrinsic_reward
            return reward + intrinsic_reward
        return reward
    
    def step(self, action):
        # gymnasium 新 API：5 个返回值
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 计算 intrinsic reward 并添加到总奖励（使用 last_obs 和当前 obs）
        if self.last_obs is not None:
            total_reward = self.reward(reward, self.last_obs, obs, action)
        else:
            # 第一次 step，没有 last_obs，只返回外部奖励
            total_reward = reward
            self.episode_external_reward += reward
        
        # 更新 last_obs 和 last_action
        self.last_obs = obs.copy()
        self.last_action = action
        
        # 如果 episode 结束，将 rewards 信息添加到 info 中
        if terminated or truncated:
            if not isinstance(info, dict):
                info = {}
            info['episode_external_reward'] = self.episode_external_reward
            info['episode_intrinsic_reward'] = self.episode_intrinsic_reward
            info['episode_total_reward'] = self.episode_external_reward + self.episode_intrinsic_reward
            # 重置 episode rewards
            self.episode_external_reward = 0.0
            self.episode_intrinsic_reward = 0.0
        
        return obs, total_reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        # gymnasium 新 API：2 个返回值
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs.copy()
        self.last_action = None
        # 重置 episode rewards
        self.episode_external_reward = 0.0
        self.episode_intrinsic_reward = 0.0
        return obs, info


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


def make_train_env(
    env_id: str = "procgen:procgen-maze-v0",  # 统一使用旧版 gym 格式
    icm_model: Optional[ICMCNN] = None,
    eta: float = 0.01,
    device: str = "cpu",
    **env_kwargs  # Procgen 环境参数
):
    """
    训练环境：基础环境包装，如果提供了 ICM 模型则添加 ICM 奖励包装。
    """
    def _init():
        env = make_base_env(env_id, **env_kwargs)
        if icm_model is not None:
            env = ICMRewardWrapper(env, icm_model, eta=eta, device=device)
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


class ICMTrainingCallback(BaseCallback):
    """
    定期训练 ICM 模型。
    """

    def __init__(
        self,
        icm_model: ICMCNN,
        icm_update_freq: int = 4,  # 每 N 个 rollout 更新一次 ICM
        icm_batch_size: int = 256,
        beta: float = 0.2,  # forward loss weight
        device: str = "cpu",
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.icm_model = icm_model
        self.icm_update_freq = icm_update_freq
        self.icm_batch_size = icm_batch_size
        self.beta = beta
        self.device = device
        
        # ICM optimizer
        self.icm_optimizer = torch.optim.Adam(
            self.icm_model.parameters(),
            lr=1e-3
        )
        
        # 存储 (obs, next_obs, action) 用于训练 ICM
        self.obs_buffer: list = []
        self.next_obs_buffer: list = []
        self.action_buffer: list = []
        self.rollout_count = 0

    def _on_step(self) -> bool:
        """
        收集观测和动作用于后续训练。
        """
        # 获取当前观测和动作
        obs = self.locals.get("obs_t")
        new_obs = self.locals.get("new_obs")
        actions = self.locals.get("actions")
        
        if obs is not None and new_obs is not None and actions is not None:
            # 只考虑单环境的情况（DummyVecEnv 里只有一个 env）
            obs_single = obs[0]  # (H, W, C) 图像格式
            next_obs_single = new_obs[0]  # (H, W, C) 图像格式
            action_single = actions[0]  # scalar action
            
            self.obs_buffer.append(obs_single.copy())
            self.next_obs_buffer.append(next_obs_single.copy())
            self.action_buffer.append(action_single)
        
        return True

    def _on_rollout_end(self) -> None:
        """
        在每个 rollout 结束时，训练 ICM 模型。
        """
        self.rollout_count += 1
        
        if len(self.obs_buffer) == 0:
            return
        
        # 每 icm_update_freq 个 rollout 更新一次 ICM
        if self.rollout_count % self.icm_update_freq == 0:
            # 准备训练数据（图像格式）
            obs_array = np.array(self.obs_buffer)  # (N, H, W, C)
            next_obs_array = np.array(self.next_obs_buffer)  # (N, H, W, C)
            action_array = np.array(self.action_buffer)  # (N,)
            
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=self.device)
            next_obs_tensor = torch.tensor(next_obs_array, dtype=torch.float32, device=self.device)
            action_tensor = torch.tensor(action_array, dtype=torch.long, device=self.device)
            
            # 随机采样一批数据
            n_samples = min(len(obs_tensor), self.icm_batch_size)
            if n_samples > 0:
                indices = np.random.choice(len(obs_tensor), n_samples, replace=False)
                batch_obs = obs_tensor[torch.tensor(indices, dtype=torch.long)]
                batch_next_obs = next_obs_tensor[torch.tensor(indices, dtype=torch.long)]
                batch_actions = action_tensor[torch.tensor(indices, dtype=torch.long)]
                
                # 计算 ICM loss 并更新
                self.icm_optimizer.zero_grad()
                icm_loss = self.icm_model.compute_loss(batch_obs, batch_next_obs, batch_actions, beta=self.beta)
                icm_loss.backward()
                self.icm_optimizer.step()
        
        # 清空 buffer（可选：保留一部分历史数据）
        self.obs_buffer = []
        self.next_obs_buffer = []
        self.action_buffer = []


class RewardLoggingCallback(BaseCallback):
    """
    记录 episode intrinsic/external/total rewards 到 TensorBoard。
    - episode_external_reward: 当前 episode 的 external reward（不累积）
    - episode_intrinsic_reward: 当前 episode 的 intrinsic reward（不累积）
    - total_reward_cumulative: 累积的 total reward（所有 episode 的总和）
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_count = 0
        self.cumulative_total_reward = 0.0  # 累积的 total reward
    
    def _on_step(self) -> bool:
        """
        在每个 step 后检查是否有完成的 episode，并立即记录该 episode 的 rewards。
        """
        # 从 locals 中获取 infos（包含 episode 结束信息）
        # 对于 vec_env，infos 是一个列表
        infos = self.locals.get("infos", [])
        
        # 确保 infos 是列表格式
        if not isinstance(infos, list):
            infos = [infos] if infos is not None else []
        
        for info in infos:
            if isinstance(info, dict):
                # 检查是否有从 ICMRewardWrapper 传递的 episode rewards
                if 'episode_external_reward' in info:
                    # 这些值已经是当前 episode 的总和（不是累积的）
                    external = float(info['episode_external_reward'])
                    intrinsic = float(info.get('episode_intrinsic_reward', 0.0))
                    episode_total = external + intrinsic  # 当前 episode 的 total reward
                    
                    # 累积 total reward
                    self.cumulative_total_reward += episode_total
                    
                    # 立即记录当前 episode 的值
                    self.episode_count += 1
                    self.logger.record("rollout/episode_external_reward", external)
                    self.logger.record("rollout/episode_intrinsic_reward", intrinsic)
                    self.logger.record("rollout/total_reward_cumulative", self.cumulative_total_reward)  # 累积值
        
        return True


def main():
    # ===== Procgen Maze 配置 =====
    env_id = "procgen:procgen-maze-v0"  # 统一使用旧版 gym 格式
    log_dir = "./logs/procgen_maze_ppo_icm"
    
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
    
    # 1) 先创建一个临时环境来获取观测格式和动作空间
    temp_env = make_base_env(env_id, **procgen_kwargs)
    # gymnasium 新 API：reset() 返回 (obs, info)
    obs_sample, info = temp_env.reset()
    
    print(f"[INFO] Observation shape: {obs_sample.shape}")
    print(f"[INFO] Observation space: {temp_env.observation_space}")
    print(f"[INFO] Action space: {temp_env.action_space}")
    
    # 确定输入通道数和动作维度
    if len(obs_sample.shape) == 3:
        in_channels = obs_sample.shape[2]  # (H, W, C) 格式
    else:
        in_channels = 3  # 默认 RGB
    
    action_dim = temp_env.action_space.n
    print(f"[INFO] Input channels: {in_channels}")
    print(f"[INFO] Action dimension: {action_dim}")

    # 2) 初始化 ICM 模型（CNN 版本）
    icm_model = ICMCNN(action_dim=action_dim, feature_dim=256, in_channels=in_channels).to(device)
    
    # 3) 创建训练 & eval 环境（训练环境包含 ICM 奖励包装）
    eta = 0.01  # intrinsic reward scale
    train_env = DummyVecEnv([make_train_env(env_id, icm_model=icm_model, eta=eta, device=device, **procgen_kwargs)])
    eval_env = DummyVecEnv([make_eval_env(env_id, **procgen_kwargs)])
    
    # 4) 创建 ICM 训练 callback
    icm_callback = ICMTrainingCallback(
        icm_model=icm_model,
        icm_update_freq=4,  # 每 4 个 rollout 更新一次 ICM
        icm_batch_size=256,
        beta=0.2,  # forward loss weight
        device=device,
        verbose=0,
    )
    
    # 4.5) 创建 reward 记录 callback
    reward_logging_callback = RewardLoggingCallback(verbose=0)

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

    # 7) 训练（同时挂三个 callback）
    total_timesteps = 200_000
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, icm_callback, reward_logging_callback],
        progress_bar=True,  # 需要安装 tqdm 和 rich
    )

    # 8) 保存 PPO 模型
    model.save(os.path.join(log_dir, "ppo_icm_procgen_maze_final"))
    
    # 9) 保存 ICM 模型
    icm_save_path = os.path.join(log_dir, "icm_model.pt")
    torch.save(icm_model.state_dict(), icm_save_path)
    print(f"[INFO] Saved ICM model to {icm_save_path}")
    
    print("[INFO] PPO+ICM (Procgen Maze) 训练完成")


if __name__ == "__main__":
    main()

