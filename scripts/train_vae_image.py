"""Train the image VAE.

Usage:
    python scripts/train_vae_image.py --config configs/isic.yaml
"""
from __future__ import annotations

import _bootstrap   # noqa: F401

import argparse
from pathlib import Path

from latentfm.train.train_vae import VAETrainArgs, train
from latentfm.utils.config import load_config


def _args_from_cfg(cfg: dict, mode: str) -> VAETrainArgs:
    common = cfg.get("common", {})
    vae_section = cfg.get("vae", {})
    overrides = vae_section.get(mode, {})
    aug = vae_section.get("augmentation", {})
    return VAETrainArgs(
        mode=mode,
        dataset=common["dataset"],
        data_root=common["data_root"],
        out_dir=str(Path(common["out_dir"]) / "checkpoints"),
        image_size=common.get("image_size", 256),
        batch_size=vae_section.get("batch_size", 4),
        epochs=vae_section.get("epochs", 250),
        lr=vae_section.get("lr", 1e-5),
        base_channels=vae_section.get("base_channels", 64),
        channel_mults=tuple(vae_section.get("channel_mults", (1, 2, 4))),
        num_res_blocks=vae_section.get("num_res_blocks", 2),
        attn_levels=tuple(vae_section.get("attn_levels", (2,))),
        latent_channels=vae_section.get("latent_channels", 3),
        kl_weight=overrides.get("kl_weight", vae_section.get("kl_weight", 1.0e-6)),
        bce_pos_weight=overrides.get("bce_pos_weight"),
        recon_type=overrides.get("recon_type", vae_section.get("recon_type")),
        flip_prob=aug.get("flip_prob", 0.5),
        rotate_max_deg=aug.get("rotate_max_deg", 15.0),
        train_resize=common.get("train_resize"),
        val_every=vae_section.get("val_every", 1),
        save_every=vae_section.get("save_every", 5),
        num_workers=common.get("num_workers", 0),
        seed=common.get("seed", 0),
        device=common.get("device"),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)
    train_args = _args_from_cfg(cfg, mode="image")
    out = train(train_args)
    print(f"[done] image VAE saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
