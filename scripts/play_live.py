"""Watch the trained world model drive in a live window.

Opens a pygame window (render_mode='human') and runs the agent on random
tracks back-to-back. Press Ctrl+C in the terminal to stop.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from world_model.controller import Controller  # noqa: E402
from world_model.mdn_rnn import MDNRNN, MDNRNNConfig  # noqa: E402
from world_model.vae import VAE, VAEConfig  # noqa: E402


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def preprocess_obs(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    img = Image.fromarray(obs).resize((64, 64), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


@torch.no_grad()
def play_one(env, vae, mdn, controller, device, max_steps: int, action_repeat: int, seed: int) -> float:
    obs, _ = env.reset(seed=seed)
    h = torch.zeros(1, mdn.cfg.hidden_dim, device=device)
    c = torch.zeros(1, mdn.cfg.hidden_dim, device=device)
    total = 0.0
    steps = 0
    while steps < max_steps:
        x = preprocess_obs(obs, device)
        mu, _ = vae.encode(x)
        z = mu
        a_t = controller(z, h)
        action = a_t.cpu().numpy().squeeze(0)

        _, _, _, _, (h, c) = mdn(z.unsqueeze(1), a_t.unsqueeze(1), (h.unsqueeze(0), c.unsqueeze(0)))
        h = h.squeeze(0)
        c = c.squeeze(0)

        for _ in range(action_repeat):
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            steps += 1
            if terminated or truncated or steps >= max_steps:
                break
        if terminated or truncated:
            break
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-ckpt", type=Path, default=Path("checkpoints/vae.pt"))
    parser.add_argument("--mdn-ckpt", type=Path, default=Path("checkpoints/mdn_rnn.pt"))
    parser.add_argument("--ctrl-ckpt", type=Path, default=Path("checkpoints/controller.pt"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=1044)
    args = parser.parse_args()

    device = pick_device()
    print(f"Device: {device}")

    vae_blob = torch.load(args.vae_ckpt, map_location=device, weights_only=False)
    vae = VAE(VAEConfig(**vae_blob["config"])).to(device).eval()
    vae.load_state_dict(vae_blob["model"])

    mdn_blob = torch.load(args.mdn_ckpt, map_location=device, weights_only=False)
    mdn = MDNRNN(MDNRNNConfig(**mdn_blob["config"])).to(device).eval()
    mdn.load_state_dict(mdn_blob["model"])

    controller = Controller().to(device).eval()
    ctrl_blob = torch.load(args.ctrl_ckpt, map_location=device, weights_only=False)
    controller.set_flat_params(ctrl_blob["flat_params"])

    env = gym.make("CarRacing-v3", render_mode="human")
    print(f"Playing {args.episodes} episodes. Close the pygame window or Ctrl+C to stop.")

    returns = []
    try:
        for i in range(args.episodes):
            seed = args.seed_start + i
            ret = play_one(
                env, vae, mdn, controller, device,
                args.max_steps, args.action_repeat, seed,
            )
            returns.append(ret)
            print(f"episode {i + 1}/{args.episodes}  seed={seed}  return={ret:.1f}")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        env.close()

    if returns:
        print(
            f"\nPlayed {len(returns)} episodes. "
            f"Mean: {np.mean(returns):.1f}  Max: {max(returns):.1f}  Min: {min(returns):.1f}"
        )


if __name__ == "__main__":
    main()
