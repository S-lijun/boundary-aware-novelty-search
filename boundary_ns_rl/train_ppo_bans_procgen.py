import os
import sys
from typing import Optional

import numpy as np
import torch

# 旧版 gym（procgen 环境注册在这里）
import gym
import procgen  # noqa: F401  # 触发 procgen 注册到 gym

# Gymnasium（用于包装器和 Stable-Baselines3）
from gymnasium import Env, spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from gymnasium import RewardWrapper

# 添加项目根目录到路径，以便导入 models
# 脚本在 boundary_ns_rl/ 目录下，需要添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))  # boundary_ns_rl/
project_root = os.path.dirname(script_dir)  # 项目根目录
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rnd_cnn import RNDCNN
from models.boundary_aware_ns import BoundaryAwareNS


class OldGymEnvCompatibility(Env):
    """
    将旧版 gym.Env 转换为 gymnasium API：
    - reset() 始终返回 (obs, info)
    - step() 始终返回 (obs, reward, terminated, truncated, info)
    """

    @staticmethod
    def _convert_space(space):
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
        return space

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.observation_space = self._convert_space(env.observation_space)
        self.action_space = self._convert_space(env.action_space)
        self.reward_range = getattr(env, "reward_range", None)
        self.metadata = getattr(env, "metadata", {})
        self.spec = getattr(env, "spec", None)

    def reset(self, **kwargs):
        try:
            result = self.env.reset(**kwargs)
        except TypeError:
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
            truncated = bool(info.pop("TimeLimit.truncated", False) or info.pop("truncated", False))
        return obs, reward, bool(done), truncated, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()


# ===== Procgen Maze 环境创建函数 =====
def make_base_env(
    env_id: str = "procgen:procgen-maze-v0",
    num_levels: int = 0,
    start_level: int = 0,
    distribution_mode: str = "hard",
    use_backgrounds: bool = True,
    restrict_themes: bool = False,
    use_monochrome_assets: bool = False,
    center_agent: bool = True,
    use_sequential_levels: bool = False,
    paint_vel_info: bool = False,
    use_generated_assets: bool = False,
    **kwargs
):
    if env_id and "procgen" in env_id.lower() and "maze" in env_id.lower():
        if not env_id.startswith("procgen:procgen-maze"):
            env_id = "procgen:procgen-maze-v0"
    else:
        env_id = "procgen:procgen-maze-v0"

    base_env = gym.make(
        env_id,
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
        **kwargs,
    )
    env = OldGymEnvCompatibility(base_env)
    return env


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


def make_train_env(
    env_id: str = "procgen:procgen-maze-v0",
    bans_model: Optional[BoundaryAwareNS] = None,
    beta: float = 2.0,  # β(s_t, t)，论文中的权重因子（对应论文的 β）
    **env_kwargs
):
    def _init():
        env = make_base_env(env_id, **env_kwargs)
        if bans_model is not None:
            env = BoundaryAwareNSRewardWrapper(env, bans_model, beta=beta)
        return env
    return _init


def make_eval_env(env_id: str = "procgen:procgen-maze-v0", **env_kwargs):
    def _init():
        env = make_base_env(env_id, **env_kwargs)
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
    env_id = "procgen:procgen-maze-v0"
    log_dir = "./logs/procgen_maze_ppo_bans"

    procgen_kwargs = {
        "num_levels": 200,
        "start_level": 0,
        "distribution_mode": "easy",
        "render_mode": None,
    }

    os.makedirs(log_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 创建临时环境以获取观测形状
    temp_env = make_base_env(env_id, **procgen_kwargs)
    obs_sample, _ = temp_env.reset()
    in_channels = obs_sample.shape[2] if len(obs_sample.shape) == 3 else 3
    print(f"[INFO] Observation shape: {obs_sample.shape}")

    # 特征提取器：使用 RND 的 target 网络（固定随机权重）
    feature_extractor = RNDCNN(feature_dim=256, in_channels=in_channels).to(device)

    # Boundary-aware NS 模型
    # ===== 参数说明（针对 Procgen Maze） =====
    # boundary_threshold: 边界检测阈值（归一化的像素变化率 [0,1]）
    #   - Procgen Maze 是图像环境，agent 移动时会有视觉变化
    #   - 推荐范围: 0.08-0.15
    #   - 太低 (0.05): 捕获太多微小变化（噪声），边界状态过多
    #   - 太高 (0.2+): 只捕获大幅变化，可能错过有意义的状态转换
    #   - 建议从 0.12 开始，根据训练日志中的 n_boundary 调整
    #     理想情况：每个 episode 有 3-10 个边界状态
    #
    # archive_k: k-nearest neighbors 的 k 值
    #   - 推荐范围: 5-15
    #   - 太小 (3): 新颖度估计不稳定
    #   - 太大 (20+): 新颖度过于平滑，难以区分不同轨迹
    #   - 建议: 10（默认值）
    #
    # beta: 内在奖励系数
    #   - 控制 boundary-aware novelty 对总奖励的影响
    #   - 推荐范围: 1.0-5.0
    #   - 太小 (0.5): 内在奖励影响太小，探索不足
    #   - 太大 (10+): 内在奖励主导，可能忽略外部奖励
    #   - 建议从 2.0-3.0 开始，根据训练曲线调整
    #
    # lambda_cost: 代价系数
    #   - 如果不使用代价函数，设为 0.0
    #   - 如果需要惩罚重复动作等行为，可以设为 0.1-0.5
    # Boundary-aware NS 模型（按照论文实现）
    # 参数设置：与 NS 相同的参数保持一致，降低惩罚参数
    bans_model = BoundaryAwareNS(
        feature_extractor=feature_extractor,
        archive_k=10,  # kNN 的 k 值（与 NS 一致）
        boundary_threshold=0.08,  # 边界检测阈值
        device=device,
        cost_threshold=1.0,  # c̄: cost threshold for safe archive（提高阈值，降低惩罚）
        lambda_init=0.01,  # 初始 lambda 值（降低初始惩罚）
        lambda_lr=0.001,  # η: lambda 学习率（降低学习率，使惩罚增长更慢）
    )

    # 训练与评估环境
    # 注意：BoundaryAwareNSRewardWrapper在make_train_env中创建，这样奖励修改会在PPO收集数据时生效
    train_env = DummyVecEnv([
        make_train_env(
            env_id,
            bans_model=bans_model,
            beta=20.0,  # β(s_t, t)，与 NS 的 beta 保持一致（进一步增加以匹配RND的奖励强度）
            **procgen_kwargs
        )
    ])
    eval_env = DummyVecEnv([make_eval_env(env_id, **procgen_kwargs)])

    # Boundary-aware NS callback（仅用于日志记录，奖励修改在wrapper中进行）
    bans_callback = BoundaryAwareNSRewardCallback(verbose=1)  # 打印前几集日志便于检查

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
    total_timesteps = 200_000
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, bans_callback],
        progress_bar=True,
    )

    # 保存模型
    model.save(os.path.join(log_dir, "ppo_bans_procgen_maze_final"))
    print("[INFO] PPO+BANS (Boundary-aware Novelty Search, Procgen Maze) 训练完成")


if __name__ == "__main__":
    main()

