"""Encode all rollout frames with the trained VAE, save latents per episode.

Input:  data/rollouts/episode_XXXX.npz  (obs, actions, rewards)
Output: data/latents/episode_XXXX.npz   (mu, logvar, actions, rewards)

We save mu and logvar so MDN-RNN training can sample z ~ N(mu, sigma) for regularization,
or use mu directly as a deterministic encoding.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from world_model.vae import VAE, VAEConfig  # noqa: E402


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def encode_file(model: VAE, path: Path, out_dir: Path, device: torch.device, bs: int) -> None:
    data = np.load(path)
    obs = data["obs"]  # (T, 64, 64, 3) uint8
    t = len(obs)
    mus = np.empty((t, model.cfg.latent_dim), dtype=np.float32)
    logvars = np.empty((t, model.cfg.latent_dim), dtype=np.float32)
    for i in range(0, t, bs):
        chunk = obs[i : i + bs]
        x = (
            torch.from_numpy(chunk).permute(0, 3, 1, 2).float().to(device) / 255.0
        )
        mu, logvar = model.encode(x)
        mus[i : i + bs] = mu.cpu().numpy()
        logvars[i : i + bs] = logvar.cpu().numpy()
    out_path = out_dir / path.name
    np.savez_compressed(
        out_path,
        mu=mus,
        logvar=logvars,
        actions=data["actions"],
        rewards=data["rewards"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts-dir", type=Path, default=Path("data/rollouts"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/latents"))
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/vae.pt"))
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"Device: {device}")

    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = VAE(VAEConfig(**blob["config"])).to(device).eval()
    model.load_state_dict(blob["model"])

    paths = sorted(args.rollouts_dir.glob("episode_*.npz"))
    for p in tqdm(paths, desc="Encoding"):
        encode_file(model, p, args.out_dir, device, args.batch_size)

    print(f"Wrote {len(paths)} files to {args.out_dir}")


if __name__ == "__main__":
    main()
