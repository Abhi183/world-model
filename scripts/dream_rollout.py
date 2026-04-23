"""Dream rollout: prime MDN-RNN with one real frame, autoregressively predict
the next N latents, decode each through the VAE.

Produces outputs/dream_vs_real.gif — real (left) vs. dreamed (right).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from world_model.mdn_rnn import MDNRNN, MDNRNNConfig  # noqa: E402
from world_model.vae import VAE, VAEConfig  # noqa: E402


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_vae(path: Path, device: torch.device) -> VAE:
    blob = torch.load(path, map_location=device, weights_only=False)
    m = VAE(VAEConfig(**blob["config"])).to(device).eval()
    m.load_state_dict(blob["model"])
    return m


def load_mdn_rnn(path: Path, device: torch.device) -> MDNRNN:
    blob = torch.load(path, map_location=device, weights_only=False)
    m = MDNRNN(MDNRNNConfig(**blob["config"])).to(device).eval()
    m.load_state_dict(blob["model"])
    return m


def tile(left: np.ndarray, right: np.ndarray, label_gap: int = 4) -> np.ndarray:
    """Side-by-side (H, W, 3) frames -> (H, 2W + gap, 3)."""
    h, w, _ = left.shape
    out = np.full((h, 2 * w + label_gap, 3), 255, dtype=np.uint8)
    out[:, :w] = left
    out[:, w + label_gap :] = right
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-ckpt", type=Path, default=Path("checkpoints/vae.pt"))
    parser.add_argument("--mdn-ckpt", type=Path, default=Path("checkpoints/mdn_rnn.pt"))
    parser.add_argument("--latents-dir", type=Path, default=Path("data/latents"))
    parser.add_argument("--episode", type=int, default=3)
    parser.add_argument("--start", type=int, default=20)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=Path("outputs/dream_vs_real.gif"))
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    device = pick_device()
    vae = load_vae(args.vae_ckpt, device)
    mdn = load_mdn_rnn(args.mdn_ckpt, device)

    # Real episode trajectory
    path = sorted(args.latents_dir.glob("episode_*.npz"))[args.episode]
    data = np.load(path)
    mu = data["mu"]            # (T, 32)
    actions = data["actions"]  # (T, 3)

    t_end = min(args.start + args.steps, len(mu) - 1)
    z_real = torch.from_numpy(mu[args.start : t_end]).float().to(device)       # (S, D)
    a_seq = torch.from_numpy(actions[args.start : t_end]).float().to(device)   # (S, 3)
    n_steps = z_real.size(0)

    # Dream: seed with first real z, then roll out autoregressively
    with torch.no_grad():
        # Warm the LSTM with the pre-start context
        ctx = args.start
        if ctx > 0:
            z_ctx = torch.from_numpy(mu[:ctx]).float().unsqueeze(0).to(device)
            a_ctx = torch.from_numpy(actions[:ctx]).float().unsqueeze(0).to(device)
            _, _, _, _, state = mdn(z_ctx, a_ctx)
        else:
            state = None

        dreamed_z = [z_real[0:1]]
        cur_z = z_real[0:1]
        for t in range(n_steps - 1):
            a_t = a_seq[t : t + 1]
            z_next, _, state = mdn.sample_next(cur_z, a_t, state, temperature=args.temperature)
            dreamed_z.append(z_next)
            cur_z = z_next
        dream_seq = torch.cat(dreamed_z, dim=0)  # (S, D)

        # Decode both sequences
        real_img = vae.decode(z_real)              # (S, 3, 64, 64)
        dream_img = vae.decode(dream_seq)          # (S, 3, 64, 64)

    real_np = (real_img.clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    dream_np = (dream_img.clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)

    # Upscale 4x for visibility, then tile side by side
    def upscale(arr: np.ndarray, factor: int = 4) -> np.ndarray:
        return np.repeat(np.repeat(arr, factor, axis=1), factor, axis=2)

    real_np = upscale(real_np)
    dream_np = upscale(dream_np)

    frames = [tile(real_np[i], dream_np[i]) for i in range(n_steps)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, fps=args.fps, loop=0)
    print(f"Saved {args.out}  ({n_steps} frames, {frames[0].shape[1]}x{frames[0].shape[0]})")
    print("Left = real episode (VAE-decoded), Right = dreamed by MDN-RNN")


if __name__ == "__main__":
    main()
