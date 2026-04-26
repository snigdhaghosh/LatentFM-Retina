"""Evaluate the trained pipeline on a test split.

Usage:
    python scripts/evaluate.py --config configs/isic.yaml
"""
from __future__ import annotations

import _bootstrap   # noqa: F401

import argparse
from pathlib import Path

from latentfm.eval.evaluate import EvalArgs, evaluate
from latentfm.utils.config import load_config


def _args_from_cfg(cfg: dict) -> EvalArgs:
    common = cfg.get("common", {})
    fm = cfg.get("fm", {})
    eval_section = cfg.get("eval", {})
    out_dir = Path(common["out_dir"])
    ckpt_dir = out_dir / "checkpoints"
    return EvalArgs(
        dataset=common["dataset"],
        data_root=common["data_root"],
        out_dir=str(out_dir / "eval"),
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
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)
    evaluate(_args_from_cfg(cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
