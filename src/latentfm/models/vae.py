"""Variational autoencoder used for both medical images and segmentation masks.

Architecture:
    - Encoder: stem 3x3 conv -> for each level in `channel_mults` stack
      `num_res_blocks` ResBlocks (with attention at flagged levels), then a
      Downsample (except after the last level). Latent head outputs (mu, logvar)
      via two 1x1 convs.
    - Decoder: mirror of the encoder using Upsamples.

Defaults follow the paper: base channels 64, channel multipliers [1, 2, 4],
two downsamplings (256 -> 64 spatial, factor f=4), latent channels 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import AttnBlock, Downsample, ResBlock, Upsample, _norm


@dataclass
class VAEConfig:
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 64
    channel_mults: Sequence[int] = field(default_factory=lambda: (1, 2, 4))
    num_res_blocks: int = 2
    attn_levels: Sequence[int] = field(default_factory=lambda: (2,))
    latent_channels: int = 3
    final_activation: str = "tanh"   # tanh / sigmoid / none

    def to_dict(self) -> dict:
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "base_channels": self.base_channels,
            "channel_mults": list(self.channel_mults),
            "num_res_blocks": self.num_res_blocks,
            "attn_levels": list(self.attn_levels),
            "latent_channels": self.latent_channels,
            "final_activation": self.final_activation,
        }


class _Encoder(nn.Module):
    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        ch = cfg.base_channels
        self.stem = nn.Conv2d(cfg.in_channels, ch, 3, padding=1)

        layers: list[nn.Module] = []
        in_ch = ch
        for level, mult in enumerate(cfg.channel_mults):
            out_ch = ch * mult
            for _ in range(cfg.num_res_blocks):
                layers.append(ResBlock(in_ch, out_ch))
                in_ch = out_ch
                if level in cfg.attn_levels:
                    layers.append(AttnBlock(in_ch))
            if level != len(cfg.channel_mults) - 1:
                layers.append(Downsample(in_ch))
        self.body = nn.Sequential(*layers)

        self.norm_out = _norm(in_ch)
        self.head = nn.Conv2d(in_ch, 2 * cfg.latent_channels, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        h = self.body(h)
        h = F.silu(self.norm_out(h))
        mu, logvar = self.head(h).chunk(2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        return mu, logvar


class _Decoder(nn.Module):
    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        ch = cfg.base_channels
        mults = list(cfg.channel_mults)
        top_ch = ch * mults[-1]
        self.stem = nn.Conv2d(cfg.latent_channels, top_ch, 3, padding=1)

        layers: list[nn.Module] = []
        in_ch = top_ch
        for level, mult in enumerate(reversed(mults)):
            out_ch = ch * mult
            for _ in range(cfg.num_res_blocks):
                layers.append(ResBlock(in_ch, out_ch))
                in_ch = out_ch
                rev_level = len(mults) - 1 - level
                if rev_level in cfg.attn_levels:
                    layers.append(AttnBlock(in_ch))
            if level != len(mults) - 1:
                layers.append(Upsample(in_ch))
        self.body = nn.Sequential(*layers)
        self.norm_out = _norm(in_ch)
        self.head = nn.Conv2d(in_ch, cfg.out_channels, 3, padding=1)
        self.final_activation = cfg.final_activation

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.stem(z)
        h = self.body(h)
        h = F.silu(self.norm_out(h))
        out = self.head(h)
        if self.final_activation == "tanh":
            out = torch.tanh(out)
        elif self.final_activation == "sigmoid":
            out = torch.sigmoid(out)
        return out


class VAE(nn.Module):
    """VAE with reparameterization, ELBO loss, and convenience encode/decode."""

    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = _Encoder(cfg)
        self.decoder = _Decoder(cfg)

    def encode(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        mu, logvar = self.encoder(x)
        if not sample:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def encode_dist(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.decoder(z)
        return recon, mu, logvar

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(
            1.0 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3]
        ).mean()

    def elbo_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        recon_type: str = "l1",
        kl_weight: float = 1e-6,
        bce_pos_weight: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Eq. 14-15: reconstruction term - KL term, averaged over the batch.

        `recon_type`:
            - 'l1'      image VAE (recon in [-1, 1])
            - 'l2'      alternative image objective
            - 'bce_l1'  mask VAE: BCE on the [0, 1] view + a small L1 term
            - 'dice'    mask VAE: soft-Dice loss on the [0, 1] view (good for sparse
                        positive classes, e.g. DRIVE vessels; insensitive to class imbalance)
            - 'dice_bce' mask VAE: 0.5 * dice + 0.5 * (optionally pos-weighted) BCE,
                         combining region overlap with pixel-level calibration
        """
        if recon_type == "l1":
            l_rec = F.l1_loss(recon, x)
        elif recon_type == "l2":
            l_rec = F.mse_loss(recon, x)
        elif recon_type == "bce_l1":
            recon01, x01 = self._mask_to_unit(recon, x)
            l_rec = self._weighted_bce(recon01, x01, bce_pos_weight) + 0.1 * F.l1_loss(
                recon01, x01
            )
        elif recon_type == "dice":
            recon01, x01 = self._mask_to_unit(recon, x)
            l_rec = self._soft_dice_loss(recon01, x01)
        elif recon_type == "dice_bce":
            recon01, x01 = self._mask_to_unit(recon, x)
            l_dice = self._soft_dice_loss(recon01, x01)
            l_bce = self._weighted_bce(recon01, x01, bce_pos_weight)
            l_rec = 0.5 * l_dice + 0.5 * l_bce
        else:
            raise ValueError(f"Unknown recon_type: {recon_type}")

        l_kl = self.kl_divergence(mu, logvar)
        loss = l_rec + kl_weight * l_kl
        return {"loss": loss, "recon": l_rec, "kl": l_kl}

    @staticmethod
    def _mask_to_unit(recon: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map a mask-VAE prediction (tanh, [-1, 1]) and target to the [0, 1] view."""
        recon01 = (recon + 1.0) / 2.0
        x01 = (x + 1.0) / 2.0 if x.min() < 0 else x
        recon01 = recon01.clamp(1e-6, 1.0 - 1e-6)
        return recon01, x01

    @staticmethod
    def _weighted_bce(
        recon01: torch.Tensor, x01: torch.Tensor, pos_weight: float | None
    ) -> torch.Tensor:
        if pos_weight is None:
            return F.binary_cross_entropy(recon01, x01)
        l_bce = F.binary_cross_entropy(recon01, x01, weight=None, reduction="none")
        pw_map = torch.where(
            x01 > 0.5, torch.full_like(x01, pos_weight), torch.ones_like(x01)
        )
        return (l_bce * pw_map).mean()

    @staticmethod
    def _soft_dice_loss(
        recon01: torch.Tensor, x01: torch.Tensor, smooth: float = 1.0
    ) -> torch.Tensor:
        """Soft-Dice loss: 1 - 2 * sum(p*t) / (sum(p) + sum(t)), with smoothing.

        Reduces over channel/H/W per sample, averages over batch. Insensitive to
        class imbalance, which is the main reason it outperforms BCE on DRIVE.
        """
        dims = tuple(range(1, recon01.dim()))
        inter = (recon01 * x01).sum(dim=dims)
        denom = recon01.sum(dim=dims) + x01.sum(dim=dims)
        dice = (2.0 * inter + smooth) / (denom + smooth)
        return (1.0 - dice).mean()
