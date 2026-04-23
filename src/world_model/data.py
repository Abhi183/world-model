"""Dataset utilities for loading collected rollouts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class FrameDataset(Dataset):
    """Yields individual (C, H, W) float32 frames in [0, 1] from rollout .npz files.

    Frames are concatenated across episodes and memory-mapped for efficiency.
    """

    def __init__(self, rollouts_dir: Path | str) -> None:
        self.paths = sorted(Path(rollouts_dir).glob("episode_*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"No rollouts found in {rollouts_dir}")

        # Load all frames into a single contiguous uint8 array.
        # 95k frames * 64*64*3 bytes = ~1.1 GB — fits in RAM easily.
        frames = [np.load(p)["obs"] for p in self.paths]
        self.frames: np.ndarray = np.concatenate(frames, axis=0)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> torch.Tensor:
        frame = self.frames[idx]  # (H, W, 3) uint8
        # NHWC uint8 -> NCHW float32 in [0, 1]
        return torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0


class SequenceDataset(Dataset):
    """Yields (obs_seq, action_seq) per rollout episode, for MDN-RNN training.

    obs:     (T, 3, 64, 64) float32 in [0, 1]
    actions: (T, 3)         float32
    """

    def __init__(self, rollouts_dir: Path | str, seq_len: int = 32) -> None:
        self.paths = sorted(Path(rollouts_dir).glob("episode_*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"No rollouts found in {rollouts_dir}")
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        data = np.load(self.paths[idx])
        obs = data["obs"]           # (T, 64, 64, 3) uint8
        actions = data["actions"]   # (T, 3) float32
        t = len(obs)
        if t > self.seq_len:
            start = np.random.randint(0, t - self.seq_len)
            obs = obs[start : start + self.seq_len]
            actions = actions[start : start + self.seq_len]
        obs_t = torch.from_numpy(obs).permute(0, 3, 1, 2).float() / 255.0
        actions_t = torch.from_numpy(actions).float()
        return obs_t, actions_t
