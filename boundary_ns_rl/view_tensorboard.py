#!/usr/bin/env python3
"""
启动 TensorBoard 来查看所有训练结果（包括 PPO Basic, PPO+NS, PPO+RND）
"""
import os
import subprocess
import sys

# 获取当前脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")

def main():
    print("=" * 60)
    print("TensorBoard 可视化工具")
    print("=" * 60)
    print(f"\n日志目录: {LOGS_DIR}")
    
    # 检查日志目录是否存在
    if not os.path.isdir(LOGS_DIR):
        print(f"\n[错误] 日志目录不存在: {LOGS_DIR}")
        return
    
    # 列出所有可用的实验
    experiments = []
    if os.path.isdir(os.path.join(LOGS_DIR, "minigrid_ppo_basic")):
        experiments.append("minigrid_ppo_basic (PPO Basic)")
    if os.path.isdir(os.path.join(LOGS_DIR, "minigrid_ppo_ns")):
        experiments.append("minigrid_ppo_ns (PPO + Novelty Search)")
    if os.path.isdir(os.path.join(LOGS_DIR, "minigrid_ppo_rnd")):
        experiments.append("minigrid_ppo_rnd (PPO + RND)")
    
    print(f"\n找到 {len(experiments)} 个实验:")
    for exp in experiments:
        print(f"  ✓ {exp}")
    
    print("\n" + "=" * 60)
    print("启动 TensorBoard...")
    print("=" * 60)
    print(f"\nTensorBoard 将在浏览器中打开")
    print(f"如果浏览器没有自动打开，请访问: http://localhost:6006")
    print(f"\n按 Ctrl+C 停止 TensorBoard\n")
    
    # 启动 TensorBoard
    try:
        cmd = [
            sys.executable, "-m", "tensorboard.main",
            "--logdir", LOGS_DIR,
            "--port", "6006",
            "--host", "localhost"
        ]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\nTensorBoard 已停止")
    except Exception as e:
        print(f"\n[错误] 无法启动 TensorBoard: {e}")
        print("\n请确保已安装 TensorBoard:")
        print("  pip install tensorboard")

if __name__ == "__main__":
    main()


