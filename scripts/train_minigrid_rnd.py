import torch
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.minigrid_wrappers import make_minigrid_env
from models.policy import PolicyNet
from models.value import ValueNet
from models.rnd import RND
from agents.ppo_agent import PPOAgent
from buffer.rollout_buffer import RolloutBuffer
from trainers.trainer_ppo_rnd import PPO_RND_Trainer



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    # -------- make environment --------
    env = make_minigrid_env()

    action_dim = env.action_space.n

    # -------- create networks --------
    policy = PolicyNet(action_dim).to(device)
    value = ValueNet().to(device)
    rnd = RND().to(device)

    # -------- create agent & buffer --------
    agent = PPOAgent(policy, value)
    buffer = RolloutBuffer(size=2048)

    # -------- trainer --------
    trainer = PPO_RND_Trainer(
        env=env,
        agent=agent,
        rnd=rnd,
        buffer=buffer,
        device=device
    )

    # -------- train --------
    rewards = trainer.train(total_steps=30000)

    # -------- save results --------
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(project_root, "results", "minigrid_rnd_rewards.npy")
    np.save(save_path, np.array(rewards))
    print(f"[INFO] training done. Saved rewards to {save_path}")

if __name__ == "__main__":
    main()

