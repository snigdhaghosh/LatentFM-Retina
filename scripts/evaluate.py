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


def _args_from_cfg(cfg: dict, ns: argparse.Namespace) -> EvalArgs:
    common = cfg.get("common", {})
    fm = cfg.get("fm", {})
    eval_section = cfg.get("eval", {})
    out_dir = Path(common["out_dir"])
    ckpt_dir = out_dir / "checkpoints"
    eval_subdir = "eval" if ns.split == "test" else f"eval_{ns.split}"
    # When patch training is on (`common.train_resize` set), match the eval
    # input scale to the training resize so the model sees the same vessel
    # detail it was trained on. The model is fully convolutional and handles
    # the larger spatial size. CLI `--image-size` overrides everything.
    cfg_image_size = common.get("train_resize") or common.get("image_size", 256)
    image_size = ns.image_size or cfg_image_size
    return EvalArgs(
        dataset=common["dataset"],
        data_root=common["data_root"],
        out_dir=str(out_dir / eval_subdir),
        image_vae_ckpt=str(ckpt_dir / "vae_image.pt"),
        mask_vae_ckpt=str(ckpt_dir / "vae_mask.pt"),
        fm_ckpt=str(ckpt_dir / "fm_unet.pt"),
        image_size=image_size,
        batch_size=eval_section.get("batch_size", 4),
        n_samples=ns.n_samples or fm.get("n_inference_samples", 5),
        n_steps=ns.n_steps or fm.get("n_inference_steps", 50),
        save_n=eval_section.get("save_n", 16),
        num_workers=common.get("num_workers", 0),
        device=common.get("device"),
        split=ns.split,
        threshold=ns.threshold if ns.threshold is not None else eval_section.get("threshold", 0.5),
        ode_method=ns.ode_method or eval_section.get("ode_method", "euler"),
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
    ap.add_argument("--n-samples", type=int, default=None,
                    help="Override fm.n_inference_samples from config.")
    ap.add_argument("--n-steps", type=int, default=None,
                    help="Override fm.n_inference_steps from config.")
    ap.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override the eval input size. Defaults to `common.train_resize` "
             "when set, else `common.image_size`. Useful for A/B comparing "
             "patch training (eval at train_resize) vs the legacy 256 eval.",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Binarization threshold on the ensemble mean (default 0.5; "
             "lower values recover thin/sparse classes like DRIVE vessels).",
    )
    ap.add_argument(
        "--ode-method",
        choices=["euler", "heun"],
        default=None,
        help="ODE solver for the FM sampler. 'heun' is 2nd-order; default 'euler'.",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    evaluate(_args_from_cfg(cfg, args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
