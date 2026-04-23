"""Verify PyTorch MPS and CarRacing environment work."""
import torch
import gymnasium as gym
import numpy as np


def check_torch():
    print(f"PyTorch: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built:     {torch.backends.mps.is_built()}")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    x = torch.randn(1024, 1024, device=device)
    y = x @ x.T
    print(f"Matmul on {device}: ok, result shape {tuple(y.shape)}")


def check_env():
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    obs, info = env.reset(seed=0)
    print(f"CarRacing obs shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"Action space: {env.action_space}")
    total_reward = 0.0
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"50 random steps, total reward: {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    check_torch()
    print("---")
    check_env()
