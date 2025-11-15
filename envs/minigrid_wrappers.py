import gymnasium as gym
import minigrid
import minigrid.envs       # 你必须要import这个，否则windows不会注册环境

from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from gymnasium.wrappers import FrameStackObservation, ResizeObservation, NormalizeObservation

def make_minigrid_env(env_id="MiniGrid-Empty-8x8-v0", stack=4):
    env = gym.make(env_id, render_mode="rgb_array")

    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)

    env = ResizeObservation(env, (64, 64))
    env = NormalizeObservation(env)

    env = FrameStackObservation(env, stack)
    return env
