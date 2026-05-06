"""Evaluate the full LatentFM pipeline on a test split.

Loads the two VAEs and the FM UNet, runs N-sample ODE inference, ensembles
the masks, computes Dice / IoU per image, averages across the test split,
and writes confidence-map / prediction visualizations to disk.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.isic import ISICDataset
from ..data.drive import DRIVEDataset
from ..eval.metrics import (
    accuracy_from_confusion,
    auc_roc_score,
    confusion_counts,
    dice_score,
    f1_from_confusion,
    iou_score,
    precision_from_confusion,
    sensitivity_from_confusion,
    specificity_from_confusion,
)
from ..flow.matching import sample
from ..flow.ensemble import aggregate, latents_to_masks
from ..models.unet import FMUNet, FMUNetConfig
from ..models.vae import VAE, VAEConfig
from ..utils.device import auto_device
from ..utils.viz import save_confidence_map, save_image, save_mask


# region error log
def _debug_log(hypothesis_id: str, message: str, data: dict, run_id: str = "run1") -> None:
    payload = {
        "sessionId": "fb929b",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": "src/latentfm/eval/evaluate.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        log_path = Path("/Users/snigdhaghoshdastidar/Desktop/latentfm/.cursor/debug-fb929b.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        print("[debug-log] failed to write runtime debug log")
# endregion


@dataclass
class EvalArgs:
    dataset: str
    data_root: str
    out_dir: str
    image_vae_ckpt: str
    mask_vae_ckpt: str
    fm_ckpt: str
    image_size: int = 256
    batch_size: int = 4
    n_samples: int = 5
    n_steps: int = 50
    save_n: int = 16
    num_workers: int = 0
    device: str | None = None
    split: str = "test"
    threshold: float = 0.5
    ode_method: str = "euler"


def _build_eval_dataset(name: str, root: str, image_size: int, split: str):
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be one of train/val/test, got {split!r}")
    if name == "isic":
        return ISICDataset(root=root, split=split, image_size=image_size)
    if name == "drive":
        return DRIVEDataset(root=root, split=split, image_size=image_size)
    raise ValueError(f"Unknown dataset: {name}")


def _load_vae(ckpt_path: str, device: torch.device) -> VAE:
    blob = torch.load(ckpt_path, map_location=device)
    cfg = VAEConfig(**blob["config"])
    vae = VAE(cfg).to(device)
    vae.load_state_dict(blob["model"])
    vae.eval()
    return vae


def _load_unet(ckpt_path: str, device: torch.device) -> FMUNet:
    blob = torch.load(ckpt_path, map_location=device)
    cfg = FMUNetConfig(**blob["config"])
    unet = FMUNet(cfg).to(device)
    unet.load_state_dict(blob["model"])
    unet.eval()
    return unet


@torch.no_grad()
def evaluate(args: EvalArgs) -> dict[str, float]:
    device = auto_device(args.device)
    # region error log
    _debug_log(
        "H1",
        "evaluate entry",
        {
            "device": str(device),
            "threshold": float(args.threshold),
            "n_samples": int(args.n_samples),
            "n_steps": int(args.n_steps),
            "split": args.split,
        },
    )
    # endregion
    image_vae = _load_vae(args.image_vae_ckpt, device)
    mask_vae = _load_vae(args.mask_vae_ckpt, device)
    unet = _load_unet(args.fm_ckpt, device)

    ds = _build_eval_dataset(args.dataset, args.data_root, args.image_size, args.split)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = out_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    dice_total, iou_total, count = 0.0, 0.0, 0
    tp_total = torch.tensor(0.0, device=device)
    tn_total = torch.tensor(0.0, device=device)
    fp_total = torch.tensor(0.0, device=device)
    fn_total = torch.tensor(0.0, device=device)
    # region error log
    _debug_log(
        "H2",
        "accumulator devices initialized",
        {
            "tp_total_device": str(tp_total.device),
            "tn_total_device": str(tn_total.device),
            "fp_total_device": str(fp_total.device),
            "fn_total_device": str(fn_total.device),
        },
    )
    # endregion
    prob_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    saved_idx = 0
    for batch in tqdm(loader, desc="[eval]"):
        img = batch["image"].to(device)
        mask = batch["mask"].to(device)
        zX = image_vae.encode(img, sample=False)
        z_samples = sample(
            unet,
            zX,
            n_samples=args.n_samples,
            n_steps=args.n_steps,
            method=args.ode_method,
        )
        masks = latents_to_masks(mask_vae.decode, z_samples)
        agg = aggregate(masks, threshold=args.threshold)

        d = dice_score(agg.final_mask, mask)
        i = iou_score(agg.final_mask, mask)
        dice_total += float(d) * img.size(0)
        iou_total += float(i) * img.size(0)
        tp, tn, fp, fn = confusion_counts(agg.final_mask, mask, threshold=args.threshold)
        # region error log
        _debug_log(
            "H3",
            "confusion count devices before accumulation",
            {
                "tp_device": str(tp.device),
                "tn_device": str(tn.device),
                "fp_device": str(fp.device),
                "fn_device": str(fn.device),
                "mask_device": str(mask.device),
                "final_mask_device": str(agg.final_mask.device),
            },
        )
        # endregion
        tp_total += tp.float()
        tn_total += tn.float()
        fp_total += fp.float()
        fn_total += fn.float()
        prob_chunks.append(agg.mean.detach().cpu())
        target_chunks.append(mask.detach().cpu())
        count += img.size(0)

        for b in range(img.size(0)):
            if saved_idx >= args.save_n:
                break
            stem = Path(batch["image_path"][b]).stem
            save_image(img[b], viz_dir / f"{stem}_input.png")
            save_mask(mask[b, 0], viz_dir / f"{stem}_gt.png")
            save_mask(agg.final_mask[b, 0], viz_dir / f"{stem}_pred.png")
            save_confidence_map(agg.confidence[b, 0], viz_dir / f"{stem}_conf.png")
            for s in range(min(args.n_samples, 5)):
                save_mask(masks[s, b, 0], viz_dir / f"{stem}_sample{s}.png")
            saved_idx += 1

    probs = torch.cat(prob_chunks, dim=0) if prob_chunks else torch.empty(0)
    targets = torch.cat(target_chunks, dim=0) if target_chunks else torch.empty(0)
    # region agent log
    _debug_log(
        "H4",
        "devices before derived-metric computation",
        {
            "tp_total_device": str(tp_total.device),
            "tn_total_device": str(tn_total.device),
            "fp_total_device": str(fp_total.device),
            "fn_total_device": str(fn_total.device),
            "prob_chunks_len": len(prob_chunks),
            "target_chunks_len": len(target_chunks),
        },
    )
    # endregion
    acc = accuracy_from_confusion(tp_total, tn_total, fp_total, fn_total)
    prec = precision_from_confusion(tp_total, fp_total)
    spe = specificity_from_confusion(tn_total, fp_total)
    sen = sensitivity_from_confusion(tp_total, fn_total)
    f1 = f1_from_confusion(tp_total, fp_total, fn_total)
    auc = auc_roc_score(probs, targets) if probs.numel() > 0 else torch.tensor(0.0)

    metrics = {
        "dice": dice_total / max(count, 1),
        "iou": iou_total / max(count, 1),
        "auc": float(auc),
        "acc": float(acc),
        "precision": float(prec),
        "spe": float(spe),
        "sen": float(sen),
        "f1": float(f1),
        "n": count,
    }
    summary = (
        f"{args.split.capitalize()} set ({args.dataset}, n={count}):\n"
        f"  Dice = {metrics['dice']:.4f}\n"
        f"  IoU  = {metrics['iou']:.4f}\n"
        f"  AUC  = {metrics['auc']:.4f}\n"
        f"  ACC  = {metrics['acc']:.4f}\n"
        f"  Precision = {metrics['precision']:.4f}\n"
        f"  Spe  = {metrics['spe']:.4f}\n"
        f"  Sen  = {metrics['sen']:.4f}\n"
        f"  F1   = {metrics['f1']:.4f}\n"
    )
    print(summary)
    (out_dir / "metrics.txt").write_text(summary)
    return metrics
