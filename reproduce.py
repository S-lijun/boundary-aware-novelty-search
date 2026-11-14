"""Run one checked-in training script from the repository root.

Examples
--------
python reproduce.py list
python reproduce.py procgen-bans
python reproduce.py minigrid-ns
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPT_DIR = ROOT / "boundary_ns_rl"

# name -> script, environment, method, log directory under boundary_ns_rl/
EXPERIMENTS = [
    ("procgen-ppo", "train_ppo_procgen.py", "Procgen-Maze", "PPO", "logs/procgen_maze_ppo"),
    ("procgen-icm", "train_ppo_icm.py", "Procgen-Maze", "PPO+ICM", "logs/procgen_maze_ppo_icm"),
    ("procgen-rnd", "train_ppo_rnd.py", "Procgen-Maze", "PPO+RND", "logs/procgen_maze_ppo_rnd"),
    ("procgen-ns", "train_ppo_ns_procgen.py", "Procgen-Maze", "PPO+NS", "logs/procgen_maze_ppo_ns"),
    ("procgen-bans", "train_ppo_bans_procgen.py", "Procgen-Maze", "PPO+B-NS", "logs/procgen_maze_ppo_bans"),
    ("minigrid-ppo", "train_ppo_minigrid_basic.py", "MiniGrid-DoorKey-5x5", "PPO", "logs/minigrid_ppo_basic"),
    ("minigrid-ns", "train_ppo_ns_minigrid.py", "MiniGrid-DoorKey-5x5", "PPO+NS", "logs/minigrid_ppo_ns"),
    ("minigrid-bans", "train_ppo_bans_minigrid.py", "MiniGrid-DoorKey-5x5", "PPO+B-NS", "logs/minigrid_ppo_bans"),
]


def _print_list() -> None:
    print(f"{'name':<16} {'environment':<24} {'method':<10} log directory")
    print("-" * 78)
    for name, _script, env, method, log_dir in EXPERIMENTS:
        print(f"{name:<16} {env:<24} {method:<10} boundary_ns_rl/{log_dir}")
    print()
    print("Train one run:  python reproduce.py <name>")
    print("Each run is 200,000 environment steps and uses the script's built-in settings.")
    print("DoorKey has no ICM or RND launcher. procgen-icm and procgen-rnd are the Procgen runs.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce one training run from the repository root.")
    parser.add_argument("name", help="Experiment name, or 'list'.")
    args = parser.parse_args()

    if args.name == "list":
        _print_list()
        return 0

    match = next((row for row in EXPERIMENTS if row[0] == args.name), None)
    if match is None:
        print(f"Unknown experiment: {args.name}\n", file=sys.stderr)
        _print_list()
        return 2

    name, script, env, method, log_dir = match
    script_path = SCRIPT_DIR / script
    if not script_path.is_file():
        print(f"Missing training script: {script_path}", file=sys.stderr)
        return 1

    print(f"Experiment : {name}")
    print(f"Method     : {method} on {env}")
    print(f"Script     : boundary_ns_rl/{script}")
    print(f"Logs       : boundary_ns_rl/{log_dir}")
    print("Steps      : 200,000")
    print()

    completed = subprocess.run([sys.executable, str(script_path)], cwd=SCRIPT_DIR)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
