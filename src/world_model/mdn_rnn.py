"""MDN-RNN: LSTM + Mixture Density Network over next-latent prediction.

Input  at each step: concat(z_t (32), a_t (3)) -> 35-dim
Hidden:              256
Output:              K Gaussian mixture components over z_{t+1} (32-dim)
                     + predicted reward (scalar)

This is "the dream" — given the current compressed state and action,
predict a distribution over the next compressed state.
"""
from __future__ import annotations

from dataclasses import dataclass

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MDNRNNConfig:
    latent_dim: int = 32
    action_dim: int = 3
    hidden_dim: int = 256
    num_mixtures: int = 5
    predict_reward: bool = True


class MDNRNN(nn.Module):
    def __init__(self, cfg: MDNRNNConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or MDNRNNConfig()
        c = self.cfg

        self.lstm = nn.LSTM(
            input_size=c.latent_dim + c.action_dim,
            hidden_size=c.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        # MDN output: K mixtures, each with (pi, mu[D], log_sigma[D])
        mdn_out = c.num_mixtures * (1 + 2 * c.latent_dim)
        self.mdn = nn.Linear(c.hidden_dim, mdn_out)
        self.reward_head = (
            nn.Linear(c.hidden_dim, 1) if c.predict_reward else None
        )

    def forward(
        self,
        z: torch.Tensor,         # (B, T, latent_dim)
        a: torch.Tensor,         # (B, T, action_dim)
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        c = self.cfg
        x = torch.cat([z, a], dim=-1)
        h, state = self.lstm(x, state)

        mdn = self.mdn(h)  # (B, T, K*(1+2D))
        b, t, _ = mdn.shape
        mdn = mdn.view(b, t, c.num_mixtures, 1 + 2 * c.latent_dim)
        pi_logits = mdn[..., 0]                                     # (B, T, K)
        mu = mdn[..., 1 : 1 + c.latent_dim]                         # (B, T, K, D)
        log_sigma = mdn[..., 1 + c.latent_dim :]                    # (B, T, K, D)
        # Clamp for numerical stability
        log_sigma = log_sigma.clamp(-7.0, 2.0)

        reward = (
            self.reward_head(h).squeeze(-1)
            if self.reward_head is not None
            else torch.zeros(b, t, device=h.device)
        )
        return pi_logits, mu, log_sigma, reward, state

    @staticmethod
    def mdn_nll(
        target: torch.Tensor,     # (B, T, D)
        pi_logits: torch.Tensor,  # (B, T, K)
        mu: torch.Tensor,         # (B, T, K, D)
        log_sigma: torch.Tensor,  # (B, T, K, D)
    ) -> torch.Tensor:
        """Negative log-likelihood under a diagonal GMM."""
        log_pi = F.log_softmax(pi_logits, dim=-1)                            # (B, T, K)
        x = target.unsqueeze(-2)                                              # (B, T, 1, D)
        # log N(x | mu, sigma) for diagonal Gaussian
        log_norm = -0.5 * (
            math.log(2 * math.pi)
            + 2 * log_sigma
            + ((x - mu) / log_sigma.exp()) ** 2
        )
        log_comp = log_norm.sum(dim=-1)                                       # (B, T, K)
        log_mix = torch.logsumexp(log_pi + log_comp, dim=-1)                  # (B, T)
        return -log_mix.mean()

    @torch.no_grad()
    def sample_next(
        self,
        z: torch.Tensor,           # (B, latent_dim)
        a: torch.Tensor,           # (B, action_dim)
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """One-step rollout in latent space. Returns (z_next, reward, new_state)."""
        pi_logits, mu, log_sigma, reward, state = self.forward(
            z.unsqueeze(1), a.unsqueeze(1), state
        )
        pi_logits = pi_logits.squeeze(1) / max(temperature, 1e-3)
        mu = mu.squeeze(1)
        log_sigma = log_sigma.squeeze(1) + math.log(max(temperature, 1e-3))

        pi = F.softmax(pi_logits, dim=-1)                       # (B, K)
        k = torch.multinomial(pi, num_samples=1).squeeze(-1)    # (B,)
        idx = k.view(-1, 1, 1).expand(-1, 1, mu.size(-1))       # (B, 1, D)
        mu_k = mu.gather(1, idx).squeeze(1)                     # (B, D)
        sigma_k = log_sigma.gather(1, idx).squeeze(1).exp()     # (B, D)
        eps = torch.randn_like(mu_k)
        z_next = mu_k + sigma_k * eps
        return z_next, reward.squeeze(1), state
