"""Run LatentFM inference on a single image.

Usage:
    python scripts/infer.py --config configs/isic.yaml \\
        --image path/to/image.jpg --out outputs/demo
"""
from __future__ import annotations

import _bootstrap   # noqa: F401

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from latentfm.flow.matching import sample
from latentfm.flow.ensemble import aggregate, latents_to_masks
from latentfm.models.unet import FMUNet, FMUNetConfig
from latentfm.models.vae import VAE, VAEConfig
from latentfm.utils.config import load_config
from latentfm.utils.device import auto_device
from latentfm.utils.viz import save_confidence_map, save_image, save_mask


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


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--n-steps", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    common = cfg.get("common", {})
    fm_cfg = cfg.get("fm", {})
    image_size = common.get("image_size", 256)
    n_samples = args.n_samples or fm_cfg.get("n_inference_samples", 5)
    n_steps = args.n_steps or fm_cfg.get("n_inference_steps", 50)

    out_dir = Path(common["out_dir"]) / "checkpoints"
    device = auto_device(common.get("device"))
    image_vae = _load_vae(str(out_dir / "vae_image.pt"), device)
    mask_vae = _load_vae(str(out_dir / "vae_mask.pt"), device)
    unet = _load_unet(str(out_dir / "fm_unet.pt"), device)

    img = Image.open(args.image).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    t = TF.to_tensor(img)
    t = TF.normalize(t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).unsqueeze(0).to(device)

    zX = image_vae.encode(t, sample=False)
    z_samples = sample(unet, zX, n_samples=n_samples, n_steps=n_steps)
    masks = latents_to_masks(mask_vae.decode, z_samples)
    agg = aggregate(masks)

    args.out.mkdir(parents=True, exist_ok=True)
    save_image(t[0], args.out / "input.png")
    save_mask(agg.final_mask[0, 0], args.out / "pred.png")
    save_confidence_map(agg.confidence[0, 0], args.out / "confidence.png")
    for i in range(min(n_samples, 8)):
        save_mask(masks[i, 0, 0], args.out / f"sample_{i}.png")
    print(f"[done] outputs written under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
