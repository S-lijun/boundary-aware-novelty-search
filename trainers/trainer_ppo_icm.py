import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm  # type: ignore
'''
class PPO_ICM_Trainer:
    
    def __init__(
        self,
        env,
        agent,
        icm,
        buffer,
        policy_lr=3e-4,
        value_lr=3e-4,
        icm_lr=1e-3,
        gamma=0.99,
        lam=0.95,
        beta=0.2,      # ICM forward loss weight
        eta=0.01,      # intrinsic reward scale
        device="cpu"
    ):
        self.env = env
        self.agent = agent
        self.icm = icm
        self.buffer = buffer

        self.gamma = gamma
        self.lam = lam
        self.beta = beta
        self.eta = eta

        self.device = device

        self.policy_opt = torch.optim.Adam(self.agent.policy.parameters(), lr=policy_lr)
        self.value_opt = torch.optim.Adam(self.agent.value.parameters(), lr=value_lr)
        self.icm_opt = torch.optim.Adam(self.icm.parameters(), lr=icm_lr)

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

                    # ---- ICM Intrinsic reward ----
                    _, phi_next, _, pred_phi_next = self.icm(obs, next_obs_tensor, action)

                    intrinsic_reward = self.eta * 0.5 * torch.mean(
                        (phi_next - pred_phi_next) ** 2
                    ).item()

                    total_reward = ext_reward + intrinsic_reward

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
                # PPO + ICM Update Phase
                # ---------------------------------
                self.update(update_epochs, batch_size)

                self.buffer.clear()

        return all_rewards

    ########################################
    #   PPO Update (with intrinsic reward)
    ########################################
    def update(self, update_epochs, batch_size):

        obs = torch.tensor(np.array(self.buffer.observations), dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(np.array(self.buffer.next_observations), dtype=torch.float32, device=self.device)
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
                batch_next_obs = next_obs[batch_idx]
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

                # ---- ICM Loss ----
                phi, phi_next, inv_logits, pred_phi_next = self.icm(batch_obs, batch_next_obs, batch_actions)

                inverse_loss = F.cross_entropy(inv_logits, batch_actions)
                forward_loss = 0.5 * F.mse_loss(pred_phi_next, phi_next)

                icm_loss = (1 - self.beta) * inverse_loss + self.beta * forward_loss

                # ---- Total Loss ----
                total_loss = policy_loss + 0.5 * value_loss + icm_loss

                # ---- Backprop ----
                self.policy_opt.zero_grad()
                self.value_opt.zero_grad()
                self.icm_opt.zero_grad()

                total_loss.backward()

                self.policy_opt.step()
                self.value_opt.step()
                self.icm_opt.step()'''

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os



class PPO_ICM_Trainer:

    def __init__(
        self,
        env,
        agent,
        icm,
        buffer,
        policy_lr=3e-4,
        value_lr=3e-4,
        icm_lr=1e-3,
        gamma=0.99,
        lam=0.95,
        beta=0.2,
        eta=0.01,
        device="cpu"
    ):
        self.env = env
        self.agent = agent
        self.icm = icm
        self.buffer = buffer

        self.gamma = gamma
        self.lam = lam
        self.beta = beta
        self.eta = eta
        self.device = device

        # ========= FIXED DONE RIGHT =========
        # trainer file path → trainers/
        this_dir = os.path.dirname(os.path.abspath(__file__))
        # project root → one level up
        project_root = os.path.abspath(os.path.join(this_dir, ".."))
        # logs/icm/ path
        log_dir = os.path.join(project_root, "logs", "icm")
        self.writer = SummaryWriter(log_dir=log_dir)
        # ====================================

        self.policy_opt = torch.optim.Adam(agent.policy.parameters(), lr=policy_lr)
        self.value_opt = torch.optim.Adam(agent.value.parameters(), lr=value_lr)
        self.icm_opt = torch.optim.Adam(icm.parameters(), lr=icm_lr)

 


    # -----------------------------
    #   GAE Advantage
    # -----------------------------
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

    # -----------------------------
    #   TRAINING LOOP
    # -----------------------------
    def train(self, total_steps=20000, rollout_steps=2048, update_epochs=4, batch_size=256):

        obs, _ = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        # ----------- stats we will record --------------
        episode_rewards = []
        episode_intrinsic = []
        episode_lengths = []
        rewarded_steps = []  # you already used this
        all_step_rewards = []

        current_ep_reward = 0
        current_ep_intrinsic = 0
        current_ep_length = 0

        step_count = 0

        with tqdm(total=total_steps) as pbar:
            pbar.set_description("Training")

            

            while step_count < total_steps:

                for _ in range(rollout_steps):

                    action, logprob, value = self.agent.get_action(obs)
                    action_np = action.item()

                    next_obs, ext_reward, terminated, truncated, _ = self.env.step(action_np)
                    done = terminated or truncated

                    next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)

                    # -------- intrinsic reward --------
                    _, phi_next, _, pred_phi_next = self.icm(obs, next_obs_tensor, action)
                    intrinsic_reward = self.eta * 0.5 * torch.mean((phi_next - pred_phi_next) ** 2).item()

                    total_reward = ext_reward + intrinsic_reward

                    # Buffer
                    self.buffer.add(
                        obs.cpu().numpy(),
                        action_np,
                        total_reward,
                        next_obs_tensor.cpu().numpy(),
                        done,
                        logprob.item(),
                        value.item()
                    )

                    # -------- stats update --------
                    all_step_rewards.append(total_reward)

                    if total_reward > 0:
                        rewarded_steps.append(total_reward)

                    current_ep_reward += ext_reward
                    current_ep_intrinsic += intrinsic_reward
                    current_ep_length += 1

                    obs = next_obs_tensor
                    step_count += 1

                    # -------- update TQDM BAR TEXT --------
                    pbar.set_postfix({
                        "ep_r": f"{current_ep_reward:.2f}",
                        "ep_len": current_ep_length,
                        "ep_int": f"{current_ep_intrinsic:.3f}"
                    })
                    pbar.update(1)

                    if done:
                        episode_rewards.append(current_ep_reward)
                        episode_intrinsic.append(current_ep_intrinsic)
                        episode_lengths.append(current_ep_length)

                            # ==== TensorBoard Logging ====
                        self.writer.add_scalar("Reward/External", current_ep_reward, len(episode_rewards))
                        self.writer.add_scalar("Reward/Intrinsic", current_ep_intrinsic, len(episode_intrinsic))
                        self.writer.add_scalar("Reward/Total", current_ep_reward + current_ep_intrinsic, len(episode_rewards))
                        self.writer.add_scalar("Episode/Length", current_ep_length, len(episode_lengths))

                        # reset episode stats
                        current_ep_reward = 0
                        current_ep_intrinsic = 0
                        current_ep_length = 0

                        obs, _ = self.env.reset()
                        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

                    if step_count >= total_steps:
                        break

                # PPO update
                self.update(update_epochs, batch_size)
                self.buffer.clear()


        # return all collected statistics
        Total_Reward = np.array(episode_rewards) + np.array(episode_intrinsic)
        return {
            "episode_rewards": np.array(episode_rewards), # External Reward
            "episode_intrinsic": np.array(episode_intrinsic), # Intrinsic Reward
            "episode_lengths": np.array(episode_lengths),
            "Total Rewards": Total_Reward,
            "all_step_rewards": np.array(all_step_rewards)
        }

    # -----------------------------
    #     PPO Update
    # -----------------------------
    def update(self, update_epochs, batch_size):

        obs = torch.tensor(np.array(self.buffer.observations), dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(np.array(self.buffer.next_observations), dtype=torch.float32, device=self.device)
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
                batch_next_obs = next_obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_adv = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                batch_old_logprob = old_logprobs[batch_idx]

                # ---------------- policy ----------------
                logits = self.agent.policy(batch_obs)
                dist = torch.distributions.Categorical(logits=logits)
                logprob = dist.log_prob(batch_actions)

                ratio = torch.exp(logprob - batch_old_logprob)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 0.8, 1.2) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # ---------------- value ----------------
                value_pred = self.agent.value(batch_obs).squeeze()
                value_loss = F.mse_loss(value_pred, batch_returns)

                # ---------------- ICM ----------------
                phi, phi_next, inv_logits, pred_phi_next = self.icm(batch_obs, batch_next_obs, batch_actions)

                inverse_loss = F.cross_entropy(inv_logits, batch_actions)
                forward_loss = 0.5 * F.mse_loss(pred_phi_next, phi_next)

                icm_loss = (1 - self.beta) * inverse_loss + self.beta * forward_loss

                total_loss = policy_loss + 0.5 * value_loss + icm_loss

                self.policy_opt.zero_grad()
                self.value_opt.zero_grad()
                self.icm_opt.zero_grad()
                total_loss.backward()
                self.policy_opt.step()
                self.value_opt.step()
                self.icm_opt.step()

