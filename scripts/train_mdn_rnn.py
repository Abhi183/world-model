"""Train the MDN-RNN on encoded latent sequences.

Usage:
    python scripts/train_mdn_rnn.py --epochs 20 --seq-len 32 --batch-size 64
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from world_model.latent_data import LatentSequenceDataset  # noqa: E402
from world_model.mdn_rnn import MDNRNN, MDNRNNConfig  # noqa: E402


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents-dir", type=Path, default=Path("data/latents"))
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/mdn_rnn.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-mixtures", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--reward-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = pick_device()
    print(f"Device: {device}")

    dataset = LatentSequenceDataset(args.latents_dir, seq_len=args.seq_len)
    print(f"Episodes: {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    cfg = MDNRNNConfig(
        latent_dim=32,
        action_dim=3,
        hidden_dim=args.hidden_dim,
        num_mixtures=args.num_mixtures,
        predict_reward=True,
    )
    model = MDNRNN(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.ckpt.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        n, nll_s, rwd_s = 0, 0.0, 0.0
        t0 = time.time()
        pbar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")
        for z, a, z_tgt, r_tgt in pbar:
            z, a, z_tgt, r_tgt = (
                z.to(device), a.to(device), z_tgt.to(device), r_tgt.to(device)
            )
            pi_logits, mu, log_sigma, reward_pred, _ = model(z, a)
            nll = model.mdn_nll(z_tgt, pi_logits, mu, log_sigma)
            rwd_loss = F.mse_loss(reward_pred, r_tgt)
            loss = nll + args.reward_weight * rwd_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = z.size(0)
            n += bs
            nll_s += nll.item() * bs
            rwd_s += rwd_loss.item() * bs
            pbar.set_postfix(nll=f"{nll_s/n:.3f}", rwd=f"{rwd_s/n:.3f}")
        dt = time.time() - t0
        print(f"epoch {epoch}: nll={nll_s/n:.3f} rwd_mse={rwd_s/n:.3f} ({dt:.1f}s)")
        torch.save(
            {"model": model.state_dict(), "config": cfg.__dict__, "epoch": epoch},
            args.ckpt,
        )

    print(f"Saved {args.ckpt}")


if __name__ == "__main__":
    main()
