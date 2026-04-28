"""Run LatentFM inference over an entire folder/split and dump predictions.

Produces one binary mask PNG per input image (same stem as the input file).
Optionally writes per-image confidence heatmaps and image-mask overlays.

Unlike `evaluate.py`, this does **not** require ground truth — it is the right
entry point for datasets where test annotations are unavailable (e.g. DRIVE
test, where the modern grand-challenge.org distribution withholds the
`1st_manual` GIFs).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm

from ..data.isic import ISICDataset
from ..data.drive import DRIVEDataset
from ..flow.matching import sample
from ..flow.ensemble import aggregate, latents_to_masks
from ..models.unet import FMUNet, FMUNetConfig
from ..models.vae import VAE, VAEConfig
from ..utils.device import auto_device
from ..utils.viz import save_confidence_map, save_mask


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp"}


@dataclass
class PredictArgs:
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
    num_workers: int = 0
    device: str | None = None
    split: str = "test"
    input_dir: str | None = None
    save_confidence: bool = False
    save_overlay: bool = False
    threshold: float = 0.5
    ode_method: str = "euler"


class _FolderDataset(Dataset):
    """Iterate every image in a flat directory, normalized like the eval transform.

    No masks are loaded; this is the dataset to use when ground truth is
    unavailable but predictions are still wanted.
    """

    def __init__(self, folder: str | Path, image_size: int = 256) -> None:
        super().__init__()
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise FileNotFoundError(f"input-dir does not exist: {self.folder}")
        paths = [
            p
            for p in sorted(self.folder.iterdir())
            if p.is_file() and p.suffix.lower() in _IMG_EXTS
        ]
        if not paths:
            raise RuntimeError(
                f"No images with extensions {sorted(_IMG_EXTS)} found under {self.folder}."
            )
        self.paths = paths
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        p = self.paths[idx]
        img = Image.open(p).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        t = TF.to_tensor(img)
        t = TF.normalize(t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return {"image": t, "image_path": str(p)}


def _build_predict_dataset(args: PredictArgs):
    if args.input_dir is not None:
        return _FolderDataset(args.input_dir, args.image_size)
    if args.dataset == "isic":
        return ISICDataset(
            root=args.data_root, split=args.split, image_size=args.image_size
        )
    if args.dataset == "drive":
        return DRIVEDataset(
            root=args.data_root, split=args.split, image_size=args.image_size
        )
    raise ValueError(f"Unknown dataset: {args.dataset}")


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


def _save_overlay(img_norm: torch.Tensor, mask: torch.Tensor, path: Path) -> None:
    """`img_norm`: [3, H, W] in [-1, 1]. `mask`: [H, W] in {0, 1}."""
    a = img_norm.detach().float().cpu()
    if a.min() < 0:
        a = (a + 1.0) / 2.0
    a = torch.clamp(a, 0.0, 1.0).numpy()
    a = np.transpose(a, (1, 2, 0))
    m = mask.detach().float().cpu().numpy()
    overlay = a.copy()
    alpha = 0.45
    overlay[..., 1] = np.clip(
        overlay[..., 1] * (1.0 - alpha * m) + 1.0 * alpha * m, 0.0, 1.0
    )
    overlay[..., 0] = overlay[..., 0] * (1.0 - 0.20 * m)
    overlay[..., 2] = overlay[..., 2] * (1.0 - 0.20 * m)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((overlay * 255).astype(np.uint8), mode="RGB").save(path)


@torch.no_grad()
def predict(args: PredictArgs) -> int:
    device = auto_device(args.device)
    image_vae = _load_vae(args.image_vae_ckpt, device)
    mask_vae = _load_vae(args.mask_vae_ckpt, device)
    unet = _load_unet(args.fm_ckpt, device)

    ds = _build_predict_dataset(args)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_done = 0
    for batch in tqdm(loader, desc="[predict]"):
        img = batch["image"].to(device)
        zX = image_vae.encode(img, sample=False)
        z_samples = sample(
            unet,
            zX,
            n_samples=args.n_samples,
            n_steps=args.n_steps,
            method=args.ode_method,
        )
        masks = latents_to_masks(mask_vae.decode, z_samples)
        agg = aggregate(masks, threshold=args.threshold)

        for b in range(img.size(0)):
            stem = Path(batch["image_path"][b]).stem
            save_mask(agg.final_mask[b, 0], out_dir / f"{stem}.png")
            if args.save_confidence:
                save_confidence_map(agg.confidence[b, 0], out_dir / f"{stem}_conf.png")
            if args.save_overlay:
                _save_overlay(img[b], agg.final_mask[b, 0], out_dir / f"{stem}_overlay.png")
            n_done += 1

    print(f"[done] wrote {n_done} predictions under {out_dir}")
    return n_done
