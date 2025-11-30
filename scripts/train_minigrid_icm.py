import torch
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.minigrid_wrappers import make_minigrid_env
from models.policy import PolicyNet
from models.value import ValueNet
from models.icm import ICM
from agents.ppo_agent import PPOAgent
from buffer.rollout_buffer import RolloutBuffer
from trainers.trainer_ppo_icm import PPO_ICM_Trainer

'''

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    # -------- make environment --------
    env = make_minigrid_env()

    action_dim = env.action_space.n

    # -------- create networks --------
    policy = PolicyNet(action_dim).to(device)
    value = ValueNet().to(device)
    icm = ICM(action_dim).to(device)

    # -------- create agent & buffer --------
    agent = PPOAgent(policy, value)
    buffer = RolloutBuffer(size=2048)

    # -------- trainer --------
    trainer = PPO_ICM_Trainer(
        env=env,
        agent=agent,
        icm=icm,
        buffer=buffer,
        device=device
    )

    # -------- train --------
    rewards = trainer.train(total_steps=30000)

    # -------- save results --------
    np.save("results/minigrid_icm_rewards.npy", np.array(rewards))
    print("[INFO] training done. Saved rewards to results/minigrid_icm_rewards.npy")

if __name__ == "__main__":
    main()

'''

import torch
import numpy as np
import sys, os, time

# -------- Path Fix: ensure project root is visible --------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

# -------- Import your modules --------
from envs.minigrid_wrappers import make_minigrid_env
from models.policy import PolicyNet
from models.value import ValueNet
from models.icm import ICM
from agents.ppo_agent import PPOAgent
from buffer.rollout_buffer import RolloutBuffer
from trainers.trainer_ppo_icm import PPO_ICM_Trainer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    # -------- Env --------
    env = make_minigrid_env()
    action_dim = env.action_space.n

    # -------- Networks --------
    policy = PolicyNet(action_dim).to(device)
    value = ValueNet().to(device)
    icm = ICM(action_dim).to(device)

    # -------- Agent & Buffer --------
    agent = PPOAgent(policy, value)
    buffer = RolloutBuffer(size=2048)

    # -------- Trainer --------
    trainer = PPO_ICM_Trainer(
        env=env,
        agent=agent,
        icm=icm,
        buffer=buffer,
        device=device
    )

    # -------- Train --------
    results = trainer.train(total_steps=30000)

    # -------- Build ABSOLUTE SAVE PATH --------
    results_dir = os.path.join(PROJECT_ROOT, "results", "icm")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(results_dir, f"minigrid_icm_stats_{timestamp}.npy")

    # -------- Save --------
    np.save(save_path, results)
    print(f"[INFO] Training done. Saved stats to {save_path}")


if __name__ == "__main__":
    main()
