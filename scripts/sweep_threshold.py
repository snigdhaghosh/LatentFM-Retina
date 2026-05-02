"""Threshold sweep on a held-out split.

Runs ODE sampling once at a configurable `n_samples`, caches the decoded
masks, and then evaluates Dice/IoU across a grid of binarization thresholds
reusing the same samples. Avoids redundant ODE integration per threshold.

Usage:
    python scripts/sweep_threshold.py --config configs/drive.yaml \\
        --split val --n-samples 16

After the run, set the best threshold under `eval.threshold` (and optionally
bump `fm.n_inference_samples`) in your config, then re-run
`scripts/evaluate.py` to lock in the result.
"""
from __future__ import annotations

import _bootstrap   # noqa: F401

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from latentfm.data.isic import ISICDataset
from latentfm.data.drive import DRIVEDataset
from latentfm.eval.metrics import dice_score, iou_score
from latentfm.flow.ensemble import latents_to_masks
from latentfm.flow.matching import sample
from latentfm.models.unet import FMUNet, FMUNetConfig
from latentfm.models.vae import VAE, VAEConfig
from latentfm.utils.config import load_config
from latentfm.utils.device import auto_device


@dataclass
class SweepArgs:
    dataset: str
    data_root: str
    out_dir: str
    image_vae_ckpt: str
    mask_vae_ckpt: str
    fm_ckpt: str
    image_size: int
    batch_size: int
    n_samples: int
    n_steps: int
    num_workers: int
    device: str | None
    split: str
    ode_method: str
    thr_min: float
    thr_max: float
    thr_step: float


def _load_vae(ckpt_path: str, device: torch.device) -> VAE:
    blob = torch.load(ckpt_path, map_location=device)
    vae = VAE(VAEConfig(**blob["config"])).to(device)
    vae.load_state_dict(blob["model"])
    vae.eval()
    return vae


def _load_unet(ckpt_path: str, device: torch.device) -> FMUNet:
    blob = torch.load(ckpt_path, map_location=device)
    unet = FMUNet(FMUNetConfig(**blob["config"])).to(device)
    unet.load_state_dict(blob["model"])
    unet.eval()
    return unet


def _build_dataset(name: str, root: str, image_size: int, split: str):
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be one of train/val/test, got {split!r}")
    if name == "isic":
        return ISICDataset(root=root, split=split, image_size=image_size)
    if name == "drive":
        return DRIVEDataset(root=root, split=split, image_size=image_size)
    raise ValueError(f"Unknown dataset: {name}")


@torch.no_grad()
def run_sweep(args: SweepArgs) -> dict:
    device = auto_device(args.device)
    image_vae = _load_vae(args.image_vae_ckpt, device)
    mask_vae = _load_vae(args.mask_vae_ckpt, device)
    unet = _load_unet(args.fm_ckpt, device)

    ds = _build_dataset(args.dataset, args.data_root, args.image_size, args.split)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # Pass 1: sample once per batch and cache decoded masks + ground truths on CPU.
    all_masks: list[torch.Tensor] = []
    all_gts: list[torch.Tensor] = []
    for batch in tqdm(loader, desc=f"[sweep] sampling N={args.n_samples}"):
        img = batch["image"].to(device)
        gt = batch["mask"].to(device)
        zX = image_vae.encode(img, sample=False)
        z_samples = sample(
            unet,
            zX,
            n_samples=args.n_samples,
            n_steps=args.n_steps,
            method=args.ode_method,
        )
        masks = latents_to_masks(mask_vae.decode, z_samples)  # [N, B, 1, H, W] in [0,1]
        all_masks.append(masks.detach().cpu())
        all_gts.append(gt.detach().cpu())

    # Pass 2: threshold sweep on the cached predictions (no model calls).
    n_steps = max(1, int(round((args.thr_max - args.thr_min) / args.thr_step)) + 1)
    thresholds = [round(args.thr_min + i * args.thr_step, 4) for i in range(n_steps)]
    rows = []
    for thr in thresholds:
        d_total, i_total, count = 0.0, 0.0, 0
        for masks, gt in zip(all_masks, all_gts):
            mean_mask = masks.clamp(0.0, 1.0).mean(dim=0)  # [B, 1, H, W]
            final = (mean_mask > thr).float()
            B = final.size(0)
            d_total += float(dice_score(final, gt)) * B
            i_total += float(iou_score(final, gt)) * B
            count += B
        rows.append((thr, d_total / max(count, 1), i_total / max(count, 1)))

    best = max(rows, key=lambda r: r[1])

    print()
    print(
        f"Threshold sweep on {args.split} split "
        f"({args.dataset}, n={count}, N_samples={args.n_samples}, "
        f"n_steps={args.n_steps}, ode={args.ode_method}, image_size={args.image_size}):"
    )
    print(f"  {'thr':>6}  {'Dice':>7}  {'IoU':>7}")
    for thr, d, i in rows:
        flag = "  <-- best" if (thr, d, i) == best else ""
        print(f"  {thr:>6.3f}  {d:>7.4f}  {i:>7.4f}{flag}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"threshold_sweep_{args.split}_N{args.n_samples}.json"
    payload = {
        "split": args.split,
        "dataset": args.dataset,
        "image_size": args.image_size,
        "n_samples": args.n_samples,
        "n_steps": args.n_steps,
        "ode_method": args.ode_method,
        "n_images": count,
        "rows": [{"threshold": t, "dice": d, "iou": i} for t, d, i in rows],
        "best": {"threshold": best[0], "dice": best[1], "iou": best[2]},
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nResults saved to {out_path}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
        help="Split to sweep on (default val).",
    )
    ap.add_argument(
        "--n-samples",
        type=int,
        default=16,
        help="Number of ODE samples per image to ensemble (default 16).",
    )
    ap.add_argument("--n-steps", type=int, default=None,
                    help="Override fm.n_inference_steps from config.")
    ap.add_argument(
        "--ode-method",
        choices=["euler", "heun"],
        default=None,
        help="ODE solver. Defaults to eval.ode_method from config.",
    )
    ap.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override eval input size. Defaults to common.train_resize "
             "if set, else common.image_size.",
    )
    ap.add_argument("--thr-min", type=float, default=0.10)
    ap.add_argument("--thr-max", type=float, default=0.60)
    ap.add_argument("--thr-step", type=float, default=0.025)
    ns = ap.parse_args()

    cfg = load_config(ns.config)
    common = cfg.get("common", {})
    fm = cfg.get("fm", {})
    eval_section = cfg.get("eval", {})
    out_dir = Path(common["out_dir"])
    ckpt_dir = out_dir / "checkpoints"
    eval_subdir = "eval" if ns.split == "test" else f"eval_{ns.split}"
    cfg_image_size = common.get("train_resize") or common.get("image_size", 256)

    args = SweepArgs(
        dataset=common["dataset"],
        data_root=common["data_root"],
        out_dir=str(out_dir / eval_subdir),
        image_vae_ckpt=str(ckpt_dir / "vae_image.pt"),
        mask_vae_ckpt=str(ckpt_dir / "vae_mask.pt"),
        fm_ckpt=str(ckpt_dir / "fm_unet.pt"),
        image_size=ns.image_size or cfg_image_size,
        batch_size=eval_section.get("batch_size", 4),
        n_samples=ns.n_samples,
        n_steps=ns.n_steps or fm.get("n_inference_steps", 50),
        num_workers=common.get("num_workers", 0),
        device=common.get("device"),
        split=ns.split,
        ode_method=ns.ode_method or eval_section.get("ode_method", "euler"),
        thr_min=ns.thr_min,
        thr_max=ns.thr_max,
        thr_step=ns.thr_step,
    )
    run_sweep(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
