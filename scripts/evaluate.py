"""Evaluate the trained pipeline on a held-out split.

Usage:
    python scripts/evaluate.py --config configs/isic.yaml                 # test split (default)
    python scripts/evaluate.py --config configs/drive.yaml --split val    # use val split instead
"""
from __future__ import annotations

import _bootstrap   # noqa: F401

import argparse
from pathlib import Path

from latentfm.eval.evaluate import EvalArgs, evaluate
from latentfm.utils.config import load_config


def _args_from_cfg(cfg: dict, split: str) -> EvalArgs:
    common = cfg.get("common", {})
    fm = cfg.get("fm", {})
    eval_section = cfg.get("eval", {})
    out_dir = Path(common["out_dir"])
    ckpt_dir = out_dir / "checkpoints"
    eval_subdir = "eval" if split == "test" else f"eval_{split}"
    return EvalArgs(
        dataset=common["dataset"],
        data_root=common["data_root"],
        out_dir=str(out_dir / eval_subdir),
        image_vae_ckpt=str(ckpt_dir / "vae_image.pt"),
        mask_vae_ckpt=str(ckpt_dir / "vae_mask.pt"),
        fm_ckpt=str(ckpt_dir / "fm_unet.pt"),
        image_size=common.get("image_size", 256),
        batch_size=eval_section.get("batch_size", 4),
        n_samples=fm.get("n_inference_samples", 5),
        n_steps=fm.get("n_inference_steps", 50),
        save_n=eval_section.get("save_n", 16),
        num_workers=common.get("num_workers", 0),
        device=common.get("device"),
        split=split,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Which split to evaluate on. Use 'val' if test ground truth is unavailable.",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    evaluate(_args_from_cfg(cfg, args.split))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
