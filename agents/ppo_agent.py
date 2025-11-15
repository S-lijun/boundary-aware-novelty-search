import torch
import torch.nn.functional as F

class PPOAgent:
    def __init__(self, policy, value):
        self.policy = policy
        self.value = value

    ########################################
    #  sample action for rollout
    ########################################
    def get_action(self, obs):
        logits = self.policy(obs)
        dist = torch.distributions.Categorical(logits=logits)

        action = dist.sample()
        logprob = dist.log_prob(action)
        value = self.value(obs).squeeze()

        return action, logprob, value

    ########################################
    #  evaluate action for PPO update
    ########################################
    def evaluate_actions(self, obs, actions):
        logits = self.policy(obs)
        dist = torch.distributions.Categorical(logits=logits)

        logprobs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.value(obs).squeeze()

        return logprobs, entropy, values
