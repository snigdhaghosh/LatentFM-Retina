"""Ensemble multiple FM samples into a final mask + a confidence map.

Given `N` decoded mask predictions (each [B, 1, H, W] in [0, 1]):
    - mean_mask    = average over N
    - final_mask   = (mean_mask > 0.5).float()
    - variance_map = pixel-wise variance over N (low variance = high agreement)
    - confidence   = 1 - normalized(variance)  (in [0, 1])
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EnsembleResult:
    samples: torch.Tensor       # [N, B, 1, H, W]
    mean: torch.Tensor          # [B, 1, H, W]
    final_mask: torch.Tensor    # [B, 1, H, W] in {0, 1}
    variance: torch.Tensor      # [B, 1, H, W]
    confidence: torch.Tensor    # [B, 1, H, W] in [0, 1]


def aggregate(masks: torch.Tensor, threshold: float = 0.5) -> EnsembleResult:
    """`masks`: tensor of shape `[N, B, 1, H, W]`, each entry in [0, 1]."""
    if masks.dim() != 5:
        raise ValueError(f"Expected 5D tensor [N, B, 1, H, W], got {masks.shape}")

    masks = masks.clamp(0.0, 1.0)
    mean = masks.mean(dim=0)
    var = masks.var(dim=0, unbiased=False)
    final = (mean > threshold).float()

    max_var = 0.25
    conf = (1.0 - var / max_var).clamp(0.0, 1.0)

    return EnsembleResult(
        samples=masks, mean=mean, final_mask=final, variance=var, confidence=conf
    )


def latents_to_masks(decoder: torch.nn.Module, latents: torch.Tensor) -> torch.Tensor:
    """Decode `[N, B, C, h, w]` latents to `[N, B, 1, H, W]` masks in [0, 1].

    Assumes the mask VAE decoder outputs values in [-1, 1] (tanh head).
    """
    if latents.dim() != 5:
        raise ValueError(f"Expected 5D latents [N, B, C, h, w], got {latents.shape}")
    N, B, C, h, w = latents.shape
    flat = latents.reshape(N * B, C, h, w)
    # Mask VAE `AttnBlock` builds a (batch, seq, seq) tensor with seq = h*w on
    # the deepest feature map. A single forward with batch N*B explodes memory
    # as O(N*B * (h*w)^2); chunk along the stacked batch (size B) so peak batch
    # matches per-image training geometry.
    decode_chunk = B
    parts: list[torch.Tensor] = []
    for start in range(0, N * B, decode_chunk):
        end = min(start + decode_chunk, N * B)
        parts.append(decoder(flat[start:end]))
    decoded = torch.cat(parts, dim=0)
    decoded01 = (decoded + 1.0) / 2.0
    return decoded01.reshape(N, B, decoded.shape[1], decoded.shape[2], decoded.shape[3]).clamp(
        0.0, 1.0
    )
