"""Dataset yielding (z_seq, a_seq, r_seq, z_next_seq) from encoded rollouts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class LatentSequenceDataset(Dataset):
    """Per-episode sequences of (z, a, r, z_next) for MDN-RNN training.

    If sample_z=True, uses reparameterization: z ~ N(mu, sigma).
    Otherwise uses mu directly (deterministic).
    """

    def __init__(
        self,
        latents_dir: Path | str,
        seq_len: int = 32,
        sample_z: bool = True,
        samples_per_episode: int = 10,
    ) -> None:
        self.paths = sorted(Path(latents_dir).glob("episode_*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"No latents in {latents_dir}")
        self.seq_len = seq_len
        self.sample_z = sample_z
        self.samples_per_episode = samples_per_episode

    def __len__(self) -> int:
        return len(self.paths) * self.samples_per_episode

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        data = np.load(self.paths[idx % len(self.paths)])
        mu = data["mu"]            # (T, D)
        logvar = data["logvar"]    # (T, D)
        actions = data["actions"]  # (T, 3)
        rewards = data["rewards"]  # (T,)

        t = len(mu)
        need = self.seq_len + 1
        if t < need:
            # Pad with last frame (rare; ignore these)
            pad = need - t
            mu = np.concatenate([mu, np.repeat(mu[-1:], pad, axis=0)], axis=0)
            logvar = np.concatenate([logvar, np.repeat(logvar[-1:], pad, axis=0)], axis=0)
            actions = np.concatenate([actions, np.repeat(actions[-1:], pad, axis=0)], axis=0)
            rewards = np.concatenate([rewards, np.repeat(rewards[-1:], pad, axis=0)], axis=0)
            t = need

        start = np.random.randint(0, t - need + 1)
        mu = mu[start : start + need]
        logvar = logvar[start : start + need]
        actions = actions[start : start + need]
        rewards = rewards[start : start + need]

        if self.sample_z:
            eps = np.random.randn(*mu.shape).astype(np.float32)
            z = mu + eps * np.exp(0.5 * logvar)
        else:
            z = mu

        z = torch.from_numpy(z).float()                 # (T+1, D)
        actions_t = torch.from_numpy(actions).float()   # (T+1, 3)
        rewards_t = torch.from_numpy(rewards).float()   # (T+1,)

        # Inputs are t=0..T-1, targets are t=1..T
        z_in = z[:-1]
        a_in = actions_t[:-1]
        z_tgt = z[1:]
        r_tgt = rewards_t[1:]  # reward received after taking a_in at step t
        return z_in, a_in, z_tgt, r_tgt
