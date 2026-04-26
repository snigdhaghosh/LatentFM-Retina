"""Train the mask VAE.

Usage:
    python scripts/train_vae_mask.py --config configs/isic.yaml
"""
from __future__ import annotations

import _bootstrap   # noqa: F401

import argparse
from pathlib import Path

from latentfm.train.train_vae import VAETrainArgs, train
from latentfm.utils.config import load_config

from train_vae_image import _args_from_cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)
    train_args = _args_from_cfg(cfg, mode="mask")
    out = train(train_args)
    print(f"[done] mask VAE saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
