"""Evaluate the full LatentFM pipeline on a test split.

Loads the two VAEs and the FM UNet, runs N-sample ODE inference, ensembles
the masks, computes Dice / IoU per image, averages across the test split,
and writes confidence-map / prediction visualizations to disk.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.isic import ISICDataset
from ..data.drive import DRIVEDataset
from ..eval.metrics import dice_score, iou_score
from ..flow.matching import sample
from ..flow.ensemble import aggregate, latents_to_masks
from ..models.unet import FMUNet, FMUNetConfig
from ..models.vae import VAE, VAEConfig
from ..utils.device import auto_device
from ..utils.viz import save_confidence_map, save_image, save_mask


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
    saved_idx = 0
    for batch in tqdm(loader, desc="[eval]"):
        img = batch["image"].to(device)
        mask = batch["mask"].to(device)
        zX = image_vae.encode(img, sample=False)
        z_samples = sample(unet, zX, n_samples=args.n_samples, n_steps=args.n_steps)
        masks = latents_to_masks(mask_vae.decode, z_samples)
        agg = aggregate(masks)

        d = dice_score(agg.final_mask, mask)
        i = iou_score(agg.final_mask, mask)
        dice_total += float(d) * img.size(0)
        iou_total += float(i) * img.size(0)
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

    metrics = {
        "dice": dice_total / max(count, 1),
        "iou": iou_total / max(count, 1),
        "n": count,
    }
    summary = (
        f"{args.split.capitalize()} set ({args.dataset}, n={count}):\n"
        f"  Dice = {metrics['dice']:.4f}\n"
        f"  IoU  = {metrics['iou']:.4f}\n"
    )
    print(summary)
    (out_dir / "metrics.txt").write_text(summary)
    return metrics
