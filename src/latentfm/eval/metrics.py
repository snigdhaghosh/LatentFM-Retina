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


def confusion_counts(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return TP/TN/FP/FN counts for binary masks at a threshold."""
    p = _binarize(pred, threshold).flatten()
    t = _binarize(target, threshold).flatten()
    tp = ((p == 1) & (t == 1)).sum()
    tn = ((p == 0) & (t == 0)).sum()
    fp = ((p == 1) & (t == 0)).sum()
    fn = ((p == 0) & (t == 1)).sum()
    return tp, tn, fp, fn


def accuracy_from_confusion(
    tp: torch.Tensor, tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    return (tp + tn) / (tp + tn + fp + fn + eps)


def precision_from_confusion(
    tp: torch.Tensor, fp: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    return tp / (tp + fp + eps)


def specificity_from_confusion(
    tn: torch.Tensor, fp: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    return tn / (tn + fp + eps)


def sensitivity_from_confusion(
    tp: torch.Tensor, fn: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    return tp / (tp + fn + eps)


def f1_from_confusion(
    tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    return (2.0 * tp) / (2.0 * tp + fp + fn + eps)


def auc_roc_score(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """AUROC for binary segmentation from probability map and target mask."""
    from torchmetrics.functional.classification import binary_auroc

    p = _ensure_unit_range(prob).flatten()
    t = _binarize(target).flatten().to(torch.int)
    return binary_auroc(p, t)


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
