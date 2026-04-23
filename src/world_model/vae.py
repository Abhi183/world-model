"""Convolutional VAE: 64x64x3 -> 32-dim latent z.

Architecture matches Ha & Schmidhuber (2018):
  Encoder:  4 strided Conv2d (3->32->64->128->256), output 2x2x256 = 1024
  Latent:   two Linear heads -> mu, logvar (each 32-dim)
  Decoder:  Linear 32->1024, then 4 strided ConvTranspose2d back to 64x64x3
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VAEConfig:
    latent_dim: int = 32
    beta: float = 1.0  # KL weight (beta-VAE if >1)


class VAE(nn.Module):
    def __init__(self, cfg: VAEConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or VAEConfig()

        # Encoder: 64 -> 31 -> 14 -> 6 -> 2  (kernel=4, stride=2, padding=0)
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2), nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(256 * 2 * 2, self.cfg.latent_dim)
        self.fc_logvar = nn.Linear(256 * 2 * 2, self.cfg.latent_dim)

        # Decoder: 1 -> 5 -> 13 -> 30 -> 64
        self.fc_dec = nn.Linear(self.cfg.latent_dim, 1024)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2), nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(-1, 1024, 1, 1)
        return self.dec(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    def loss(
        self, x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Per-pixel reconstruction, summed over image, averaged over batch.
        recon = F.mse_loss(x_hat, x, reduction="none").flatten(1).sum(dim=1).mean()
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        total = recon + self.cfg.beta * kl
        return total, recon, kl
