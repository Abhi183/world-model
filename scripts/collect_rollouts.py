"""Collect random rollouts in CarRacing-v3 and save as .npz files.

Each episode is saved as data/rollouts/episode_XXXX.npz containing:
  obs:     (T, 64, 64, 3) uint8 — preprocessed frames
  actions: (T, 3) float32       — [steer, gas, brake]
  rewards: (T,) float32

Uses action repeat (sticky random actions) for better exploration — pure
random actions in CarRacing just spin the car in place.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image
from tqdm import tqdm


def preprocess(frame: np.ndarray) -> np.ndarray:
    """Resize 96x96x3 -> 64x64x3 uint8."""
    img = Image.fromarray(frame).resize((64, 64), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def sample_action(rng: np.random.Generator) -> np.ndarray:
    """Biased random action for better exploration.

    Mostly accelerates forward, sometimes brakes, steers smoothly.
    """
    steer = rng.uniform(-1.0, 1.0)
    gas = rng.uniform(0.0, 1.0) if rng.random() < 0.85 else 0.0
    brake = rng.uniform(0.0, 0.1) if rng.random() < 0.1 else 0.0
    return np.array([steer, gas, brake], dtype=np.float32)


def collect_episode(
    env: gym.Env,
    rng: np.random.Generator,
    max_steps: int,
    action_repeat: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))

    obs_buf: list[np.ndarray] = []
    act_buf: list[np.ndarray] = []
    rew_buf: list[float] = []

    step = 0
    while step < max_steps:
        action = sample_action(rng)
        for _ in range(action_repeat):
            obs_buf.append(preprocess(obs))
            act_buf.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            rew_buf.append(float(reward))
            step += 1
            if terminated or truncated or step >= max_steps:
                break
        if terminated or truncated:
            break

    return (
        np.stack(obs_buf, axis=0),
        np.stack(act_buf, axis=0),
        np.asarray(rew_buf, dtype=np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=Path("data/rollouts"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    env = gym.make("CarRacing-v3", render_mode="rgb_array")

    total_frames = 0
    start = time.time()
    pbar = tqdm(range(args.num_episodes), desc="Episodes")
    for ep in pbar:
        obs, actions, rewards = collect_episode(env, rng, args.max_steps, args.action_repeat)
        out_path = args.out_dir / f"episode_{ep:04d}.npz"
        np.savez_compressed(out_path, obs=obs, actions=actions, rewards=rewards)
        total_frames += len(obs)
        pbar.set_postfix(frames=total_frames, ret=f"{float(rewards.sum()):.1f}")

    env.close()
    elapsed = time.time() - start
    print(
        f"Done. {args.num_episodes} episodes, {total_frames} frames, "
        f"{elapsed:.1f}s ({total_frames / elapsed:.0f} fps)"
    )


if __name__ == "__main__":
    main()
