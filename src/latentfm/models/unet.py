"""UNet for conditional flow matching in the latent space.

Operates on a [B, C_z, h, w] noisy latent `z_t`, conditioned on:
    - the FM time `t` (sinusoidal embedding -> MLP)
    - the image latent `z_X` (concatenated along channels into the input)

Default config matches the paper: base channels 64, channel multipliers
[1, 2, 2, 4] (-> 4 levels, 3 downsamples => 64 -> 32 -> 16 -> 8 spatial),
attention at the deeper levels.

Construction and runtime use a single skip stack: one skip is pushed after
each (ResBlock + optional AttnBlock) pair on the down path, plus one after
the stem and one after each Downsample. The up path consumes them in LIFO
order, with `num_res_blocks + 1` consumptions per up-level.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import AttnBlock, Downsample, ResBlock, Upsample, _norm, timestep_embedding


@dataclass
class FMUNetConfig:
    latent_channels: int = 3
    cond_channels: int = 3
    base_channels: int = 64
    channel_mults: Sequence[int] = field(default_factory=lambda: (1, 2, 2, 4))
    num_res_blocks: int = 2
    attn_levels: Sequence[int] = field(default_factory=lambda: (2, 3))
    time_embed_dim: int = 256

    def to_dict(self) -> dict:
        return {
            "latent_channels": self.latent_channels,
            "cond_channels": self.cond_channels,
            "base_channels": self.base_channels,
            "channel_mults": list(self.channel_mults),
            "num_res_blocks": self.num_res_blocks,
            "attn_levels": list(self.attn_levels),
            "time_embed_dim": self.time_embed_dim,
        }


class _DownPair(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_ch: int, use_attn: bool) -> None:
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, emb_channels=emb_ch)
        self.attn = AttnBlock(out_ch) if use_attn else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        return self.attn(self.res(x, emb))


class _UpPair(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, emb_ch: int, use_attn: bool) -> None:
        super().__init__()
        self.res = ResBlock(in_ch + skip_ch, out_ch, emb_channels=emb_ch)
        self.attn = AttnBlock(out_ch) if use_attn else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, skip], dim=1)
        return self.attn(self.res(x, emb))


class FMUNet(nn.Module):
    """Conditional flow-matching UNet on the latent grid."""

    def __init__(self, cfg: FMUNetConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_embed_dim, cfg.time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(cfg.time_embed_dim * 4, cfg.time_embed_dim * 4),
        )
        emb_ch = cfg.time_embed_dim * 4

        in_ch = cfg.latent_channels + cfg.cond_channels
        ch = cfg.base_channels
        self.stem = nn.Conv2d(in_ch, ch, 3, padding=1)

        self.down_pairs: nn.ModuleList = nn.ModuleList()
        self.down_samplers: nn.ModuleList = nn.ModuleList()

        skip_chs: list[int] = [ch]
        cur_ch = ch
        for level, mult in enumerate(cfg.channel_mults):
            out_ch = ch * mult
            use_attn = level in cfg.attn_levels
            for _ in range(cfg.num_res_blocks):
                self.down_pairs.append(_DownPair(cur_ch, out_ch, emb_ch, use_attn))
                cur_ch = out_ch
                skip_chs.append(cur_ch)
            is_last = level == len(cfg.channel_mults) - 1
            if not is_last:
                self.down_samplers.append(Downsample(cur_ch))
                skip_chs.append(cur_ch)
            else:
                self.down_samplers.append(nn.Identity())

        self.mid1 = ResBlock(cur_ch, cur_ch, emb_channels=emb_ch)
        self.mid_attn = AttnBlock(cur_ch)
        self.mid2 = ResBlock(cur_ch, cur_ch, emb_channels=emb_ch)

        self.up_levels: nn.ModuleList = nn.ModuleList()   # each entry: ModuleList of UpPairs
        self.up_samplers: nn.ModuleList = nn.ModuleList()
        for level in range(len(cfg.channel_mults) - 1, -1, -1):
            mult = cfg.channel_mults[level]
            out_ch = ch * mult
            use_attn = level in cfg.attn_levels
            pairs = nn.ModuleList()
            for _ in range(cfg.num_res_blocks + 1):
                skip_ch = skip_chs.pop()
                pairs.append(_UpPair(cur_ch, skip_ch, out_ch, emb_ch, use_attn))
                cur_ch = out_ch
            self.up_levels.append(pairs)
            self.up_samplers.append(Upsample(cur_ch) if level != 0 else nn.Identity())

        self.norm_out = _norm(cur_ch)
        self.head = nn.Conv2d(cur_ch, cfg.latent_channels, 3, padding=1)

    def forward(
        self, z_t: torch.Tensor, t: torch.Tensor, z_cond: torch.Tensor
    ) -> torch.Tensor:
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.shape[0] == 1 and z_t.shape[0] > 1:
            t = t.expand(z_t.shape[0])

        emb = timestep_embedding(t, self.cfg.time_embed_dim)
        emb = self.time_mlp(emb)

        if z_cond.shape[-2:] != z_t.shape[-2:]:
            z_cond = F.interpolate(
                z_cond, size=z_t.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([z_t, z_cond], dim=1)
        x = self.stem(x)
        skips: list[torch.Tensor] = [x]

        pair_idx = 0
        for level, _mult in enumerate(self.cfg.channel_mults):
            for _ in range(self.cfg.num_res_blocks):
                pair = self.down_pairs[pair_idx]
                pair_idx += 1
                x = pair(x, emb)
                skips.append(x)
            ds = self.down_samplers[level]
            if not isinstance(ds, nn.Identity):
                x = ds(x)
                skips.append(x)

        x = self.mid1(x, emb)
        x = self.mid_attn(x)
        x = self.mid2(x, emb)

        for pairs, upsampler in zip(self.up_levels, self.up_samplers):
            for pair in pairs:
                skip = skips.pop()
                x = pair(x, skip, emb)
            if not isinstance(upsampler, nn.Identity):
                x = upsampler(x)

        x = F.silu(self.norm_out(x))
        return self.head(x)
