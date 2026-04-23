"""Sample random frames, reconstruct them through the VAE, save a comparison grid.

Also samples from prior (z ~ N(0, I)) to show what the decoder hallucinates.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from world_model.data import FrameDataset  # noqa: E402
from world_model.vae import VAE, VAEConfig  # noqa: E402


def tensor_to_uint8(x: torch.Tensor) -> np.ndarray:
    """(N, 3, H, W) float in [0, 1] -> (N, H, W, 3) uint8."""
    x = x.clamp(0, 1).cpu().numpy()
    return (x.transpose(0, 2, 3, 1) * 255).astype(np.uint8)


def grid(rows: list[np.ndarray], pad: int = 2) -> np.ndarray:
    """List of (N, H, W, 3) -> single (R*H + pad, N*W + pad, 3) grid."""
    h, w = rows[0].shape[1:3]
    n = rows[0].shape[0]
    out = np.full(
        (len(rows) * (h + pad) - pad, n * (w + pad) - pad, 3), 255, dtype=np.uint8
    )
    for r, row in enumerate(rows):
        for c in range(n):
            y0, x0 = r * (h + pad), c * (w + pad)
            out[y0 : y0 + h, x0 : x0 + w] = row[c]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/vae.pt"))
    parser.add_argument("--rollouts-dir", type=Path, default=Path("data/rollouts"))
    parser.add_argument("--out", type=Path, default=Path("outputs/vae_recon.png"))
    parser.add_argument("--n-samples", type=int, default=12)
    args = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = VAE(VAEConfig(**blob["config"])).to(device)
    model.load_state_dict(blob["model"])
    model.eval()

    dataset = FrameDataset(args.rollouts_dir)
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(dataset), size=args.n_samples, replace=False)
    x = torch.stack([dataset[int(i)] for i in idxs]).to(device)

    with torch.no_grad():
        x_hat, mu, logvar = model(x)
        z_prior = torch.randn(args.n_samples, model.cfg.latent_dim, device=device)
        x_sample = model.decode(z_prior)

    row_orig = tensor_to_uint8(x)
    row_recon = tensor_to_uint8(x_hat)
    row_sample = tensor_to_uint8(x_sample)
    img = grid([row_orig, row_recon, row_sample], pad=2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(args.out)
    print(f"Saved {args.out} ({img.shape[1]}x{img.shape[0]})")
    print("Rows: [originals] [reconstructions] [samples from N(0, I)]")


if __name__ == "__main__":
    main()
