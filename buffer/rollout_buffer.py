import numpy as np

class RolloutBuffer:
    def __init__(self, size, obs_dim=None, action_dim=None):
        self.size = size
        self.reset_lists()

    def reset_lists(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.dones = []
        self.logprobs = []
        self.values = []

    def add(self, obs, action, reward, next_obs, done, logprob, value):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_observations.append(next_obs)
        self.dones.append(done)
        self.logprobs.append(logprob)
        self.values.append(value)

    def clear(self):
        self.reset_lists()
