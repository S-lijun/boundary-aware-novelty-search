import os
from typing import List, Dict, Any

import gymnasium as gym
import numpy as np


class TrajectoryRecorderWrapper(gym.Wrapper):
    """
    一个简单的环境封装器：
    - 在训练过程中记录若干条 episode 的 (obs, action, reward) 序列
    - 训练结束后可以一次性保存到 .npz 文件里
    """

    def __init__(self, env: gym.Env, max_episodes: int = 200):
        super().__init__(env)
        self.max_episodes = max_episodes
        self.recording_enabled = True

        # 当前正在记录的这一条 episode
        self._curr_obs: List[np.ndarray] = []
        self._curr_actions: List[int] = []
        self._curr_rewards: List[float] = []

        # 已经完整记录好的 episodes
        self._episodes: List[Dict[str, Any]] = []
        
        # 新增：在线 NS 使用，保存“最近完成的一条 episode”
        self._last_episode: Dict[str, Any] | None = None

    # Gymnasium: reset 返回 (obs, info)
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # 开始一条新的 episode
        self._curr_obs = [np.array(obs)]
        self._curr_actions = []
        self._curr_rewards = []

        return obs, info

    # Gymnasium: step 返回 (obs, reward, terminated, truncated, info)
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.recording_enabled and self._curr_obs is not None:
            # 记录这一 step
            self._curr_actions.append(int(action))
            self._curr_rewards.append(float(reward))
            self._curr_obs.append(np.array(obs))

            done = terminated or truncated
            if done:
                episode = {
                    "obs": np.stack(self._curr_obs, axis=0),
                    "actions": np.array(self._curr_actions),
                    "rewards": np.array(self._curr_rewards),
                }
                self._episodes.append(episode)
                # 记录“最近完成的一条 episode”，给 NS callback 用
                self._last_episode = episode

                # 只有 max_episodes > 0 时才会自动停记录
                if self.max_episodes > 0 and len(self._episodes) >= self.max_episodes:
                    self.recording_enabled = False

                self._curr_obs = []
                self._curr_actions = []
                self._curr_rewards = []


        return obs, reward, terminated, truncated, info

    # 训练结束后调用，把已记录的 episodes 打包保存
    def save_trajectories(self, file_path: str):
        if not self._episodes:
            print("[TrajectoryRecorder] 没有记录到任何 episode，不保存。")
            return

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # 对所有 episode 做 padding，整理成统一长度的矩阵方便后续训练 encoder
        max_len = max(len(ep["actions"]) for ep in self._episodes)

        obs_list = []
        actions_list = []
        rewards_list = []
        masks_list = []

        for ep in self._episodes:
            obs = ep["obs"]          # (T+1, obs_dim)
            acts = ep["actions"]     # (T,)
            rews = ep["rewards"]     # (T,)
            T = len(acts)

            # pad 到 max_len
            obs_pad = np.zeros((max_len + 1, *obs.shape[1:]), dtype=obs.dtype)
            acts_pad = np.full((max_len,), fill_value=-1, dtype=np.int64)
            rews_pad = np.zeros((max_len,), dtype=np.float32)
            mask = np.zeros((max_len,), dtype=np.float32)

            obs_pad[:T + 1] = obs
            acts_pad[:T] = acts
            rews_pad[:T] = rews
            mask[:T] = 1.0

            obs_list.append(obs_pad)
            actions_list.append(acts_pad)
            rewards_list.append(rews_pad)
            masks_list.append(mask)

        obs_arr = np.stack(obs_list, axis=0)        # (N, max_len+1, obs_dim)
        acts_arr = np.stack(actions_list, axis=0)   # (N, max_len)
        rews_arr = np.stack(rewards_list, axis=0)   # (N, max_len)
        masks_arr = np.stack(masks_list, axis=0)    # (N, max_len)

        np.savez_compressed(
            file_path,
            obs=obs_arr,
            actions=acts_arr,
            rewards=rews_arr,
            masks=masks_arr,
        )
        print(f"[TrajectoryRecorder] 已保存 {len(self._episodes)} 条 episode 到 {file_path}")
    
    def get_current_episode(self):
        if len(self._curr_actions) == 0:
            return None
        return {
            "obs": np.array(self._curr_obs),
            "actions": np.array(self._curr_actions),
            "rewards": np.array(self._curr_rewards),
        }

    def get_last_episode(self):
        """
        返回最近完成的一条 episode（obs/actions/rewards），
        如果还没有完整 episode，则返回 None。
        """
        return self._last_episode

