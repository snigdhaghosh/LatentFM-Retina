"""Run LatentFM inference on every image in a split or a folder.

Common uses:

    # Predict on the configured dataset's test split
    # (works for ISIC; for DRIVE only if you have test/1st_manual/).
    python scripts/predict.py --config configs/isic.yaml

    # Predict on a folder of images (no ground truth needed).
    # Recommended for DRIVE, where test annotations are gated.
    python scripts/predict.py --config configs/drive.yaml \\
        --input-dir data/DRIVE/test/images

    # Use a different split, save extra visualizations.
    python scripts/predict.py --config configs/drive.yaml --split val \\
        --save-confidence --save-overlay

Outputs (one PNG per input image, same stem) land under
`<out_dir>/predictions/<split-or-folder-name>/` by default, e.g.
`outputs/drive/predictions/images/01_test.png`.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

from latentfm.eval.predict import PredictArgs, predict
from latentfm.utils.config import load_config


def _args_from_cfg(cfg: dict, ns: argparse.Namespace) -> PredictArgs:
    common = cfg.get("common", {})
    fm = cfg.get("fm", {})
    eval_section = cfg.get("eval", {})
    out_dir = Path(common["out_dir"])
    ckpt_dir = out_dir / "checkpoints"

    if ns.input_dir is not None:
        sub = Path(ns.input_dir).name or "input"
    else:
        sub = ns.split
    default_pred_dir = out_dir / "predictions" / sub

    return PredictArgs(
        dataset=common["dataset"],
        data_root=common["data_root"],
        out_dir=str(ns.out) if ns.out is not None else str(default_pred_dir),
        image_vae_ckpt=str(ckpt_dir / "vae_image.pt"),
        mask_vae_ckpt=str(ckpt_dir / "vae_mask.pt"),
        fm_ckpt=str(ckpt_dir / "fm_unet.pt"),
        image_size=common.get("image_size", 256),
        batch_size=ns.batch_size or eval_section.get("batch_size", 4),
        n_samples=ns.n_samples or fm.get("n_inference_samples", 5),
        n_steps=ns.n_steps or fm.get("n_inference_steps", 50),
        num_workers=common.get("num_workers", 0),
        device=common.get("device"),
        split=ns.split,
        input_dir=str(ns.input_dir) if ns.input_dir is not None else None,
        save_confidence=ns.save_confidence,
        save_overlay=ns.save_overlay,
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
        help="Dataset split to predict on. Ignored if --input-dir is given.",
    )
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Optional folder of raw images to predict on instead of a dataset split. "
             "Use this when test ground truth is unavailable (e.g. DRIVE).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for predictions. Default: "
             "<common.out_dir>/predictions/<split-or-folder-name>/",
    )
    ap.add_argument("--n-samples", type=int, default=None,
                    help="Override fm.n_inference_samples from config.")
    ap.add_argument("--n-steps", type=int, default=None,
                    help="Override fm.n_inference_steps from config.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Override eval.batch_size from config.")
    ap.add_argument("--save-confidence", action="store_true",
                    help="Also write per-image variance-based confidence heatmaps.")
    ap.add_argument("--save-overlay", action="store_true",
                    help="Also write per-image overlay PNGs (mask blended onto input).")
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
    ns = ap.parse_args()

    cfg = load_config(ns.config)
    predict(_args_from_cfg(cfg, ns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
