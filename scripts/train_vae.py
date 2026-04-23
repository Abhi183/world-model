"""Train the VAE on collected frames.

Usage:
    python scripts/train_vae.py --epochs 10 --batch-size 128
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from world_model.data import FrameDataset  # noqa: E402
from world_model.vae import VAE, VAEConfig  # noqa: E402


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts-dir", type=Path, default=Path("data/rollouts"))
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/vae.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = pick_device()
    print(f"Device: {device}")

    dataset = FrameDataset(args.rollouts_dir)
    print(f"Frames: {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    model = VAE(VAEConfig(latent_dim=args.latent_dim, beta=args.beta)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.ckpt.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        n, total, recon_s, kl_s = 0, 0.0, 0.0, 0.0
        t0 = time.time()
        pbar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")
        for x in pbar:
            x = x.to(device, non_blocking=True)
            x_hat, mu, logvar = model(x)
            loss, recon, kl = model.loss(x, x_hat, mu, logvar)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bs = x.size(0)
            n += bs
            total += loss.item() * bs
            recon_s += recon.item() * bs
            kl_s += kl.item() * bs
            pbar.set_postfix(loss=f"{total/n:.2f}", recon=f"{recon_s/n:.2f}", kl=f"{kl_s/n:.2f}")
        dt = time.time() - t0
        print(
            f"epoch {epoch}: loss={total/n:.3f} recon={recon_s/n:.3f} "
            f"kl={kl_s/n:.3f} ({dt:.1f}s)"
        )
        torch.save(
            {
                "model": model.state_dict(),
                "config": model.cfg.__dict__,
                "epoch": epoch,
            },
            args.ckpt,
        )

    print(f"Saved {args.ckpt}")


if __name__ == "__main__":
    main()
