"""Tiny linear controller: [z_t; h_t] -> action.

Following Ha & Schmidhuber: 867 parameters total. Small enough that evolution
strategies like CMA-ES converge quickly, large enough to drive a car.

Input:  concat(z (32), h (256))  -> 288
Output: 3-dim action — tanh(steer), sigmoid(gas), sigmoid(brake)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class Controller(nn.Module):
    def __init__(self, latent_dim: int = 32, hidden_dim: int = 256, action_dim: int = 3) -> None:
        super().__init__()
        self.input_dim = latent_dim + hidden_dim
        self.fc = nn.Linear(self.input_dim, action_dim)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, h], dim=-1)
        raw = self.fc(x)
        steer = torch.tanh(raw[..., 0])
        gas = torch.sigmoid(raw[..., 1])
        brake = torch.sigmoid(raw[..., 2])
        return torch.stack([steer, gas, brake], dim=-1)

    def get_flat_params(self) -> np.ndarray:
        return torch.cat([p.data.flatten() for p in self.parameters()]).cpu().numpy()

    def set_flat_params(self, flat: np.ndarray) -> None:
        t = torch.from_numpy(flat).float()
        offset = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(t[offset : offset + n].view_as(p))
            offset += n
