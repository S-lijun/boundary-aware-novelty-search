import gymnasium as gym
import minigrid
import minigrid.envs       # 你必须要import这个，否则windows不会注册环境

from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from gymnasium.wrappers import FrameStackObservation, ResizeObservation, NormalizeObservation
from procgen import ProcgenEnv

def make_procgen_maze_env(stack=4):

    # --- Create Procgen Maze ---
    env = ProcgenEnv(
        num_envs=1,
        env_name="maze",
        render_mode="rgb_array",
        distribution_mode="easy"
    )

    # Procgen returns vectorized env -> unwrap
    env = gym.wrappers.SyncVectorEnv([lambda: env])[0]

    # --- same preprocessing as MiniGrid ---
    env = ResizeObservation(env, (64, 64))
    env = NormalizeObservation(env)
    env = FrameStackObservation(env, stack)

    return env
