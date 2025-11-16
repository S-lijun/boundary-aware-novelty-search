import os
from datetime import datetime

import gymnasium as gym
from gymnasium import spaces, ObservationWrapper
from gymnasium.wrappers import FlattenObservation
from minigrid.wrappers import FullyObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from utils.trajectory_recorder import TrajectoryRecorderWrapper


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


def make_base_env(env_id: str):
    """
    构建最基础的 Minigrid 环境：
    - FullyObsWrapper: 全局可见
    - ImageOnlyWrapper: 只保留 image
    - FlattenObservation: 展平成一维向量，给 MlpPolicy 用
    """
    env = gym.make(env_id)
    env = FullyObsWrapper(env)
    env = ImageOnlyWrapper(env)
    env = FlattenObservation(env)
    return env


def make_train_env(env_id: str, max_record_episodes: int = 200):
    """
    训练环境：在基础 env 外面再套一层 TrajectoryRecorderWrapper
    用来记录一批 episode 作为 encoder 的训练数据。
    """
    def _init():
        env = make_base_env(env_id)
        env = TrajectoryRecorderWrapper(env, max_episodes=max_record_episodes)
        return env

    return _init


def make_eval_env(env_id: str):
    """
    eval 环境：不需要记录轨迹，只负责评估。
    """
    def _init():
        env = make_base_env(env_id)
        return env

    return _init


def main():
    env_id = "MiniGrid-DoorKey-5x5-v0"
    log_dir = "./logs/minigrid_ppo_basic"
    os.makedirs(log_dir, exist_ok=True)

    # 1) 创建训练 & eval 环境
    train_env = DummyVecEnv([make_train_env(env_id, max_record_episodes=200)])
    eval_env = DummyVecEnv([make_eval_env(env_id)])

    # 2) 定义 PPO 模型（和之前一样）
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=log_dir,
    )

    # 3) 评估回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=10_000,
        deterministic=True,
        render=False,
    )

    # 4) 训练
    total_timesteps = 200_000
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    # 5) 保存 PPO 模型
    model.save(os.path.join(log_dir, "ppo_minigrid_final"))

    # 6) 从 DummyVecEnv 里取出我们的 TrajectoryRecorderWrapper，保存轨迹
    recorder_env = train_env.envs[0]
    if hasattr(recorder_env, "save_trajectories"):
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 把轨迹放到当前 log_dir 下面，方便区分不同实验
        traj_dir = os.path.join(log_dir, "trajectories")
        os.makedirs(traj_dir, exist_ok=True)

        traj_path = os.path.join(traj_dir, f"door_key_ppo_traj_{run_tag}.npz")
        recorder_env.save_trajectories(traj_path)
    else:
        print("[Warning] 训练环境中没有 TrajectoryRecorderWrapper，未保存轨迹。")



if __name__ == "__main__":
    main()
