import os
from datetime import datetime

import numpy as np
import gymnasium as gym
from gymnasium import spaces, ObservationWrapper
from gymnasium.wrappers import FlattenObservation
from minigrid.wrappers import FullyObsWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from novelty.behavior_embedding import BehaviorEmbedder
from novelty.archive import BehaviorArchive
from utils.trajectory_recorder import TrajectoryRecorderWrapper

def make_eval_env(env_id: str):
    def _init():
        env = gym.make(env_id)
        env = FullyObsWrapper(env)
        env = ImageOnlyWrapper(env)
        env = FlattenObservation(env)
        return env
    return _init

class ImageOnlyWrapper(ObservationWrapper):
    """
    和 baseline 一样：只保留 obs["image"]，去掉 mission，
    再交给 FlattenObservation 处理。
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(
            env.observation_space, spaces.Dict
        ), f"Expected Dict obs space, got {env.observation_space}"
        self.observation_space = env.observation_space["image"]

    def observation(self, obs):
        return obs["image"]


class NSRewardCallback(BaseCallback):
    """
    每条 episode 结束时：
      1) 从 TrajectoryRecorderWrapper 拿出上一条完整 episode
      2) encoder → z
      3) 计算 novelty，并加到当前 step 的 reward 上（reward shaping）
    """

    def __init__(self, embedder: BehaviorEmbedder, archive: BehaviorArchive, beta: float = 0.1, verbose: int = 0):
        super().__init__(verbose)
        self.embedder = embedder
        self.archive = archive
        self.beta = beta

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        rewards = self.locals.get("rewards")

        if dones is None or rewards is None:
            return True

        # 这里只考虑单环境的情况（DummyVecEnv 里只有一个 env）
        done = dones[0]
        if not done:
            return True

        # 取出 env 和最近完成的一条 episode
        env = self.model.get_env().envs[0]
        if not hasattr(env, "get_last_episode"):
            return True

        ep = env.get_last_episode()
        if ep is None:
            return True

        obs_seq = ep["obs"]       # (T+1, obs_dim)
        rewards_ep = ep["rewards"]  # (T,)
        T = len(rewards_ep)

        # 和训练 encoder 一样，对齐：用前 T 个 obs，对应 T 个 steps
        obs_seq = obs_seq[:-1]        # (T, obs_dim)
        mask = np.ones((T,), dtype=np.float32)

        # 计算 embedding
        z = self.embedder.embed_np(obs_seq[None, :, :], mask[None, :])[0]  # (latent_dim,)

        # 计算 novelty，并更新 archive
        novelty = self.archive.compute_novelty(z)
        self.archive.add(z)

        # 将当前 step 的 reward 做 reward shaping：
        # r' = r_env + beta * novelty
        rewards[0] = rewards[0] + self.beta * novelty

        if self.verbose > 0:
            print(f"[NSRewardCallback] novelty={novelty:.4f}, shaped_reward={rewards[0]:.4f}")

        return True


def make_env(env_id: str):
    def _init():
        env = gym.make(env_id)
        env = FullyObsWrapper(env)
        env = ImageOnlyWrapper(env)
        env = FlattenObservation(env)
        # 对于 NS，不限制 max_episodes（设为 0 表示无限记录）
        env = TrajectoryRecorderWrapper(env, max_episodes=0)
        return env

    return _init


def main():
    env_id = "MiniGrid-DoorKey-5x5-v0"
    log_dir = "./logs/minigrid_ppo_ns"
    os.makedirs(log_dir, exist_ok=True)

    # ---------- 初始化 encoder ----------
    encoder_path = "./logs/minigrid_ppo_basic/encoder/door_key_encoder.pt"
    obs_dim = 75  # 你前面已经验证过 obs_dim=75

    embedder = BehaviorEmbedder(
        obs_dim=obs_dim,
        encoder_path=encoder_path,
        hidden_dim=128,
        latent_dim=64,
        device="cpu",
    )

    archive = BehaviorArchive(k=10)

    # ---------- 构建训练 & eval 环境 ----------
    env = DummyVecEnv([make_env(env_id)])
    eval_env = DummyVecEnv([make_eval_env(env_id)])

    # ---------- PPO model ----------
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
    )

    # ---------- EvalCallback（评估时仍用原始环境 reward） ----------
    from stable_baselines3.common.callbacks import EvalCallback

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=10_000,
        deterministic=True,
        render=False,
    )

    # ---------- NS 奖励 callback ----------
    ns_callback = NSRewardCallback(
        embedder=embedder,
        archive=archive,
        beta=0.1,
        verbose=0,
    )

    # ---------- 开始训练（同时挂两个 callback） ----------
    total_timesteps = 200_000
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, ns_callback],
        progress_bar=True,
    )


    # 保存模型
    model.save(os.path.join(log_dir, "ppo_ns_minigrid_final"))
    print("NS-PPO 训练完成")


if __name__ == "__main__":
    main()
