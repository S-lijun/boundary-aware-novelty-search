import torch

class RolloutBuffer:
    def __init__(self, size, obs_dim, action_dim):
        self.size = size
        
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
        self.__init__(self.size, None, None)
