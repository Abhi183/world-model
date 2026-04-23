"""Evaluate a trained controller and record a GIF of real gameplay.

Produces outputs/agent_drives.gif and prints return stats over N seeds.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
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
def run_episode(
    env: gym.Env,
    vae: VAE,
    mdn: MDNRNN,
    controller: Controller,
    device: torch.device,
    max_steps: int,
    action_repeat: int,
    seed: int,
    frames: list | None = None,
) -> float:
    obs, _ = env.reset(seed=seed)
    h = torch.zeros(1, mdn.cfg.hidden_dim, device=device)
    c_state = torch.zeros(1, mdn.cfg.hidden_dim, device=device)
    total = 0.0
    steps = 0
    while steps < max_steps:
        x = preprocess_obs(obs, device)
        mu, _ = vae.encode(x)
        z = mu
        action_t = controller(z, h)
        action = action_t.cpu().numpy().squeeze(0)

        _, _, _, _, (h_new, c_new) = mdn(
            z.unsqueeze(1), action_t.unsqueeze(1), (h.unsqueeze(0), c_state.unsqueeze(0))
        )
        h = h_new.squeeze(0)
        c_state = c_new.squeeze(0)

        for _ in range(action_repeat):
            if frames is not None:
                frames.append(env.render())
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
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--gif-seed", type=int, default=42)
    parser.add_argument("--gif-out", type=Path, default=Path("outputs/agent_drives.gif"))
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    device = pick_device()

    # Load all three models
    vae_blob = torch.load(args.vae_ckpt, map_location=device, weights_only=False)
    vae = VAE(VAEConfig(**vae_blob["config"])).to(device).eval()
    vae.load_state_dict(vae_blob["model"])

    mdn_blob = torch.load(args.mdn_ckpt, map_location=device, weights_only=False)
    mdn = MDNRNN(MDNRNNConfig(**mdn_blob["config"])).to(device).eval()
    mdn.load_state_dict(mdn_blob["model"])

    controller = Controller().to(device).eval()
    ctrl_blob = torch.load(args.ctrl_ckpt, map_location=device, weights_only=False)
    controller.set_flat_params(ctrl_blob["flat_params"])

    env = gym.make("CarRacing-v3", render_mode="rgb_array")

    returns = []
    for i in range(args.n_episodes):
        ret = run_episode(
            env, vae, mdn, controller, device,
            args.max_steps, args.action_repeat,
            seed=args.gif_seed + 1000 + i,
        )
        returns.append(ret)
        print(f"episode {i}: return={ret:.2f}")

    print(
        f"\nMean: {np.mean(returns):.2f}  Std: {np.std(returns):.2f}  "
        f"Max: {max(returns):.2f}  Min: {min(returns):.2f}"
    )

    # Record GIF of the GIF-seed episode
    print(f"\nRecording GIF at seed {args.gif_seed}...")
    frames: list = []
    gif_return = run_episode(
        env, vae, mdn, controller, device,
        args.max_steps, args.action_repeat,
        seed=args.gif_seed, frames=frames,
    )
    env.close()

    args.gif_out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.gif_out, frames, fps=args.fps, loop=0)
    print(f"Saved {args.gif_out}  ({len(frames)} frames, return={gif_return:.2f})")


if __name__ == "__main__":
    main()
