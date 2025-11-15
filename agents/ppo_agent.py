import torch
import torch.nn.functional as F

class PPOAgent:
    def __init__(self, policy, value):
        self.policy = policy
        self.value = value

    def get_action(self, obs):
        logits = self.policy(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        value = self.value(obs)
        return action, logprob, value
