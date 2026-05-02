"""Train the conditional latent flow-matching UNet.

Requires both VAE checkpoints to exist (train them first via
`scripts/train_vae_image.py` and `scripts/train_vae_mask.py`).

Usage:
    python scripts/train_fm.py --config configs/isic.yaml
"""
from __future__ import annotations

import _bootstrap   # noqa: F401

import argparse
from pathlib import Path

from latentfm.train.train_fm import FMTrainArgs, train
from latentfm.utils.config import load_config


def _args_from_cfg(cfg: dict) -> FMTrainArgs:
    common = cfg.get("common", {})
    fm = cfg.get("fm", {})
    vae = cfg.get("vae", {})
    eval_section = cfg.get("eval", {})
    aug = fm.get("augmentation", vae.get("augmentation", {}))
    out_dir = Path(common["out_dir"]) / "checkpoints"
    return FMTrainArgs(
        dataset=common["dataset"],
        data_root=common["data_root"],
        out_dir=str(out_dir),
        image_vae_ckpt=str(out_dir / "vae_image.pt"),
        mask_vae_ckpt=str(out_dir / "vae_mask.pt"),
        image_size=common.get("image_size", 256),
        batch_size=fm.get("batch_size", 4),
        epochs=fm.get("epochs", 250),
        lr=fm.get("lr", 1e-5),
        base_channels=fm.get("base_channels", 64),
        channel_mults=tuple(fm.get("channel_mults", (1, 2, 2, 4))),
        num_res_blocks=fm.get("num_res_blocks", 2),
        attn_levels=tuple(fm.get("attn_levels", (2, 3))),
        time_embed_dim=fm.get("time_embed_dim", 256),
        sigma=fm.get("sigma", 0.0),
        n_inference_samples=fm.get("n_inference_samples", 5),
        n_inference_steps=fm.get("n_inference_steps", 50),
        # Step 3: align validation with the actual eval pipeline so the saved
        # checkpoint is selected by the same metric we report.
        n_validation_samples=fm.get("n_validation_samples"),
        val_threshold=eval_section.get("threshold", 0.5),
        val_ode_method=eval_section.get("ode_method", "euler"),
        ema_decay=fm.get("ema_decay", 0.0),
        flip_prob=aug.get("flip_prob", 0.5),
        rotate_max_deg=aug.get("rotate_max_deg", 15.0),
        train_resize=common.get("train_resize"),
        val_every=fm.get("val_every", 5),
        save_every=fm.get("save_every", 5),
        num_workers=common.get("num_workers", 0),
        seed=common.get("seed", 0),
        device=common.get("device"),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = train(_args_from_cfg(cfg))
    print(f"[done] FM UNet saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
