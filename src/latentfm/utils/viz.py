"""Visualization helpers for predictions and confidence maps."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _to_uint8(t: torch.Tensor) -> np.ndarray:
    a = t.detach().float().cpu().numpy()
    a = np.clip(a, 0.0, 1.0)
    return (a * 255.0 + 0.5).astype(np.uint8)


def save_mask(mask: torch.Tensor, path: str | Path) -> None:
    """`mask`: [H, W] tensor in [0, 1]."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arr = _to_uint8(mask)
    Image.fromarray(arr, mode="L").save(path)


def save_image(img: torch.Tensor, path: str | Path) -> None:
    """`img`: [C, H, W] tensor in [-1, 1] or [0, 1]."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    a = img.detach().float().cpu()
    if a.min() < 0:
        a = (a + 1.0) / 2.0
    a = torch.clamp(a, 0.0, 1.0).numpy()
    a = np.transpose(a, (1, 2, 0))
    if a.shape[2] == 1:
        a = a[..., 0]
        Image.fromarray((a * 255).astype(np.uint8), mode="L").save(path)
    else:
        Image.fromarray((a * 255).astype(np.uint8), mode="RGB").save(path)


def save_confidence_map(conf: torch.Tensor, path: str | Path, cmap: str = "viridis") -> None:
    """Save a [H, W] confidence map (0..1) as a colored heatmap PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.cm as cm

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arr = conf.detach().float().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    rgba = cm.get_cmap(cmap)(arr)
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def save_grid(images: list[torch.Tensor], path: str | Path, nrow: int = 4) -> None:
    """Save a grid of [C, H, W] images for VAE reconstruction sanity-checks."""
    from torchvision.utils import make_grid

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    stacked = torch.stack([t.detach().float().cpu() for t in images], dim=0)
    if stacked.min() < 0:
        stacked = (stacked + 1.0) / 2.0
    stacked = torch.clamp(stacked, 0.0, 1.0)
    grid = make_grid(stacked, nrow=nrow)
    arr = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    if arr.shape[2] == 1:
        Image.fromarray(arr[..., 0], mode="L").save(path)
    else:
        Image.fromarray(arr, mode="RGB").save(path)
