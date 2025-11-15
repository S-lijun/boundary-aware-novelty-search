import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, NormalizeObservation, FrameStackObservation

'''
先用最简单的MiniGrid吧
'''
def make_minigrid_env(env_id="MiniGrid-Empty-8x8-v0", stack=4):
    env = gym.make(env_id, render_mode="rgb_array")
    env = ResizeObservation(env, (64, 64))
    env = NormalizeObservation(env)
    env = FrameStackObservation(env, stack)
    return env
