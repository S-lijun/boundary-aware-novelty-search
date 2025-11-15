import torch
from envs.minigrid_wrappers import make_minigrid_env
from models.policy import PolicyNet
from models.value import ValueNet
from models.icm import ICM
from agents.ppo_agent import PPOAgent
from buffer.rollout_buffer import RolloutBuffer
from trainers.trainer_ppo_icm import PPO_ICM_Trainer

env = make_minigrid_env()

action_dim = env.action_space.n

policy = PolicyNet(action_dim)
value = ValueNet()
icm = ICM(action_dim)

agent = PPOAgent(policy, value)
buffer = RolloutBuffer(size=2048, obs_dim=None, action_dim=action_dim)

trainer = PPO_ICM_Trainer(env, agent, icm, buffer)
trainer.train()
