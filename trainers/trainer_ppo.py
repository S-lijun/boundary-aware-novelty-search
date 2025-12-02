import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm  # type: ignore[import-untyped]
'''
class PPO_Trainer:
    """
    Pure PPO (Proximal Policy Optimization) Trainer
    
    Standard PPO implementation without any intrinsic motivation.
    Uses only extrinsic rewards from the environment.
    """
    
    def __init__(
        self,
        env,
        agent,
        buffer,
        policy_lr=3e-4,
        value_lr=3e-4,
        gamma=0.99,
        lam=0.95,
        device="cpu"
    ):
        self.env = env
        self.agent = agent
        self.buffer = buffer

        self.gamma = gamma
        self.lam = lam

        self.device = device

        self.policy_opt = torch.optim.Adam(self.agent.policy.parameters(), lr=policy_lr)
        self.value_opt = torch.optim.Adam(self.agent.value.parameters(), lr=value_lr)

    ########################################
    #   GAE Advantage Computation
    ########################################
    def compute_gae(self, rewards, values, dones, last_value):
        advs = []
        gae = 0
        values = values + [last_value]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advs.insert(0, gae)

        returns = [advs[i] + values[i] for i in range(len(advs))]
        return advs, returns

    ########################################
    #   Main Training Loop
    ########################################
    def train(self, total_steps=20000, rollout_steps=2048, update_epochs=4, batch_size=256):

        obs, _ = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        episode_reward = 0
        all_rewards = []

        step_count = 0

        with tqdm(total=total_steps) as pbar:

            while step_count < total_steps:

                # ---------------------------------
                # Rollout Phase (collect data)
                # ---------------------------------
                for _ in range(rollout_steps):

                    action, logprob, value = self.agent.get_action(obs)
                    action_np = action.item()

                    next_obs, ext_reward, terminated, truncated, _ = self.env.step(action_np)
                    done = terminated or truncated

                    next_obs_tensor = torch.tensor(
                        next_obs, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)

                    # Use only extrinsic reward (no intrinsic motivation)
                    total_reward = ext_reward

                    # store to buffer
                    self.buffer.add(
                        obs.cpu().numpy(),
                        action_np,
                        total_reward,
                        next_obs_tensor.cpu().numpy(),
                        done,
                        logprob.item(),
                        value.item()
                    )

                    obs = next_obs_tensor
                    episode_reward += ext_reward
                    step_count += 1

                    if done:
                        all_rewards.append(episode_reward)
                        episode_reward = 0
                        obs, _ = self.env.reset()
                        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

                    pbar.update(1)
                    pbar.set_description(f"Reward: {np.mean(all_rewards[-20:]) if len(all_rewards)>20 else 0:.2f}")

                    if step_count >= total_steps:
                        break

                # ---------------------------------
                # PPO Update Phase
                # ---------------------------------
                self.update(update_epochs, batch_size)

                self.buffer.clear()

        return all_rewards

    ########################################
    #   PPO Update
    ########################################
    def update(self, update_epochs, batch_size):

        obs = torch.tensor(np.array(self.buffer.observations), dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.buffer.actions, device=self.device)
        rewards = self.buffer.rewards
        dones = self.buffer.dones
        old_logprobs = torch.tensor(self.buffer.logprobs, device=self.device)
        values = self.buffer.values

        # ---------------------------------
        # Compute GAE
        # ---------------------------------
        last_value = 0
        advantages, returns = self.compute_gae(rewards, values, dones, last_value)

        advantages = torch.tensor(advantages, device=self.device)
        returns = torch.tensor(returns, device=self.device)

        # ---------------------------------
        # PPO Update Loop
        # ---------------------------------
        n = len(obs)
        inds = np.arange(n)

        for _ in range(update_epochs):
            np.random.shuffle(inds)

            for start in range(0, n, batch_size):
                end = start + batch_size
                batch_idx = inds[start:end]

                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_adv = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                batch_old_logprob = old_logprobs[batch_idx]

                # ---- Policy ----
                logits = self.agent.policy(batch_obs)
                dist = torch.distributions.Categorical(logits=logits)
                logprob = dist.log_prob(batch_actions)

                ratio = torch.exp(logprob - batch_old_logprob)

                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 0.8, 1.2) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # ---- Value ----
                value_pred = self.agent.value(batch_obs).squeeze()
                value_loss = F.mse_loss(value_pred, batch_returns)

                # ---- Total Loss (only policy + value) ----
                total_loss = policy_loss + 0.5 * value_loss

                # ---- Backprop ----
                self.policy_opt.zero_grad()
                self.value_opt.zero_grad()

                total_loss.backward()

                self.policy_opt.step()
                self.value_opt.step()
'''

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os


class PPO_Trainer:
    """
    Pure PPO Trainer
    Fully aligned with the PPO + ICM and PPO + RND logging structure.
    """

    def __init__(
        self,
        env,
        agent,
        buffer,
        policy_lr=3e-4,
        value_lr=3e-4,
        gamma=0.99,
        lam=0.95,
        device="cpu"
    ):
        self.env = env
        self.agent = agent
        self.buffer = buffer

        self.gamma = gamma
        self.lam = lam
        self.device = device

        # ====== LOGGING PATH ======
        this_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(this_dir, ".."))
        log_dir = os.path.join(project_root, "logs", "ppo")
        self.writer = SummaryWriter(log_dir=log_dir)
        # ===========================

        self.policy_opt = torch.optim.Adam(self.agent.policy.parameters(), lr=policy_lr)
        self.value_opt = torch.optim.Adam(self.agent.value.parameters(), lr=value_lr)

    # -------------------------------------------------
    # GAE
    # -------------------------------------------------
    def compute_gae(self, rewards, values, dones, last_value):
        advs = []
        gae = 0
        values = values + [last_value]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advs.insert(0, gae)

        returns = [advs[i] + values[i] for i in range(len(advs))]
        return advs, returns

    # -------------------------------------------------
    # TRAINING LOOP (ALIGNED WITH ICM/RND)
    # -------------------------------------------------
    def train(self, total_steps=20000, rollout_steps=2048, update_epochs=4, batch_size=256):

        obs, _ = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        # ===== aligned stats =====
        episode_rewards = []
        episode_intrinsic = []        # pure PPO → always 0, but kept for compatibility
        episode_lengths = []
        rewarded_steps = []
        all_step_rewards = []

        current_ep_reward = 0
        current_ep_length = 0
        # =========================

        step_count = 0

        with tqdm(total=total_steps) as pbar:
            pbar.set_description("Training (PPO)")

            while step_count < total_steps:

                # -----------------------------------------
                # ROLLOUT
                # -----------------------------------------
                for _ in range(rollout_steps):

                    action, logprob, value = self.agent.get_action(obs)
                    action_np = action.item()

                    next_obs, ext_reward, terminated, truncated, _ = self.env.step(action_np)
                    done = terminated or truncated

                    next_obs_tensor = torch.tensor(
                        next_obs, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)

                    # Pure PPO → no intrinsic reward
                    total_reward = ext_reward

                    # Store in buffer
                    self.buffer.add(
                        obs.cpu().numpy(),
                        action_np,
                        total_reward,
                        next_obs_tensor.cpu().numpy(),
                        done,
                        logprob.item(),
                        value.item()
                    )

                    # ===== Statistics =====
                    all_step_rewards.append(total_reward)

                    if total_reward > 0:
                        rewarded_steps.append(total_reward)

                    current_ep_reward += ext_reward
                    current_ep_length += 1
                    # =======================

                    obs = next_obs_tensor
                    step_count += 1

                    # tqdm display matching ICM/RND
                    pbar.set_postfix({
                        "ep_r": f"{current_ep_reward:.2f}",
                        "ep_int": 0.0,      # PPO has no intrinsic reward
                        "ep_len": current_ep_length
                    })
                    pbar.update(1)

                    if done:
                        episode_rewards.append(current_ep_reward)
                        episode_intrinsic.append(0.0)  # always 0
                        episode_lengths.append(current_ep_length)

                        # TensorBoard logging aligned with ICM & RND
                        ep_id = len(episode_rewards)
                        self.writer.add_scalar("Reward/External", current_ep_reward, ep_id)
                        self.writer.add_scalar("Reward/Intrinsic", 0.0, ep_id)
                        self.writer.add_scalar("Reward/Total", current_ep_reward, ep_id)
                        self.writer.add_scalar("Episode/Length", current_ep_length, ep_id)

                        # Reset episode stats
                        current_ep_reward = 0
                        current_ep_length = 0

                        obs, _ = self.env.reset()
                        obs = torch.tensor(
                            obs, dtype=torch.float32, device=self.device
                        ).unsqueeze(0)

                    if step_count >= total_steps:
                        break

                # PPO Update
                self.update(update_epochs, batch_size)
                self.buffer.clear()

        # ===== return same structure as ICM/RND =====
        Total_Reward = np.array(episode_rewards)  # no intrinsic reward

        return {
            "episode_rewards": np.array(episode_rewards),
            "episode_intrinsic": np.array(episode_intrinsic),
            "episode_lengths": np.array(episode_lengths),
            "Total Rewards": Total_Reward,
            "all_step_rewards": np.array(all_step_rewards),
            "rewarded_steps": np.array(rewarded_steps)
        }

    # -------------------------------------------------
    # UPDATE
    # -------------------------------------------------
    def update(self, update_epochs, batch_size):

        obs = torch.tensor(np.array(self.buffer.observations), dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.buffer.actions, device=self.device)
        rewards = self.buffer.rewards
        dones = self.buffer.dones
        old_logprobs = torch.tensor(self.buffer.logprobs, device=self.device)
        values = self.buffer.values

        # GAE
        last_value = 0
        advantages, returns = self.compute_gae(rewards, values, dones, last_value)
        advantages = torch.tensor(advantages, device=self.device)
        returns = torch.tensor(returns, device=self.device)

        n = len(obs)
        inds = np.arange(n)

        for _ in range(update_epochs):
            np.random.shuffle(inds)

            for start in range(0, n, batch_size):
                end = start + batch_size
                batch_idx = inds[start:end]

                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_adv = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                batch_old_logprob = old_logprobs[batch_idx]

                # ---- PPO policy ----
                logits = self.agent.policy(batch_obs)
                dist = torch.distributions.Categorical(logits=logits)
                logprob = dist.log_prob(batch_actions)

                ratio = torch.exp(logprob - batch_old_logprob)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 0.8, 1.2) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # ---- PPO value ----
                value_pred = self.agent.value(batch_obs).squeeze()
                value_loss = F.mse_loss(value_pred, batch_returns)

                total_loss = policy_loss + 0.5 * value_loss

                self.policy_opt.zero_grad()
                self.value_opt.zero_grad()
                total_loss.backward()
                self.policy_opt.step()
                self.value_opt.step()


