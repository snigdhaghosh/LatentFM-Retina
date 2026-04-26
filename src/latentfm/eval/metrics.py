"""Evaluation metrics for segmentation and reconstruction.

Segmentation metrics (binary masks in {0, 1}):
    - dice_score
    - iou_score (a.k.a. Jaccard)

Reconstruction metrics (images / masks in any range, batched):
    - ssim_score (uses torchmetrics)
    - psnr_score (uses torchmetrics)
"""
from __future__ import annotations

import torch


def _binarize(t: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (t > threshold).float()


def dice_score(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6, threshold: float = 0.5
) -> torch.Tensor:
    """Per-image Dice; averages over the batch. Inputs `[B, 1, H, W]`."""
    p = _binarize(pred, threshold)
    t = _binarize(target, threshold)
    p = p.flatten(1)
    t = t.flatten(1)
    inter = (p * t).sum(dim=1)
    denom = p.sum(dim=1) + t.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return dice.mean()


def iou_score(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6, threshold: float = 0.5
) -> torch.Tensor:
    p = _binarize(pred, threshold)
    t = _binarize(target, threshold)
    p = p.flatten(1)
    t = t.flatten(1)
    inter = (p * t).sum(dim=1)
    union = p.sum(dim=1) + t.sum(dim=1) - inter
    iou = (inter + eps) / (union + eps)
    return iou.mean()


def _ensure_unit_range(t: torch.Tensor) -> torch.Tensor:
    if t.min() < 0:
        t = (t + 1.0) / 2.0
    return t.clamp(0.0, 1.0)


def ssim_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    from torchmetrics.functional.image import structural_similarity_index_measure

    p = _ensure_unit_range(pred)
    t = _ensure_unit_range(target)
    return structural_similarity_index_measure(p, t, data_range=1.0)


def psnr_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    from torchmetrics.functional.image import peak_signal_noise_ratio

    p = _ensure_unit_range(pred)
    t = _ensure_unit_range(target)
    return peak_signal_noise_ratio(p, t, data_range=1.0)
