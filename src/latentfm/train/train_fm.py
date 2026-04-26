"""Trains the conditional flow-matching UNet in the (latent, latent) space.

The two VAEs must already be trained. We:
    1. load both VAEs and freeze them,
    2. encode (image, mask) batches into (z_X, z_S),
    3. minimize the FM loss on `u_theta(z_t, t, z_X) ~= z_S - z_0`.

We also run a held-out validation step that integrates the ODE for a single
sample, decodes through the mask VAE, and reports Dice/IoU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.isic import ISICDataset
from ..data.drive import DRIVEDataset
from ..eval.metrics import dice_score, iou_score
from ..flow.matching import fm_loss, sample
from ..flow.ensemble import aggregate, latents_to_masks
from ..models.unet import FMUNet, FMUNetConfig
from ..models.vae import VAE, VAEConfig
from ..utils.device import auto_device


@dataclass
class FMTrainArgs:
    dataset: str
    data_root: str
    out_dir: str
    image_vae_ckpt: str
    mask_vae_ckpt: str
    image_size: int = 256
    batch_size: int = 4
    epochs: int = 250
    lr: float = 1e-5
    base_channels: int = 64
    channel_mults: tuple[int, ...] = (1, 2, 2, 4)
    num_res_blocks: int = 2
    attn_levels: tuple[int, ...] = (2, 3)
    time_embed_dim: int = 256
    sigma: float = 0.0
    n_inference_samples: int = 5
    n_inference_steps: int = 50
    val_every: int = 5
    save_every: int = 5
    num_workers: int = 0
    seed: int = 0
    device: str | None = None


def _build_dataset(name: str, root: str, split: str, image_size: int):
    if name == "isic":
        return ISICDataset(root=root, split=split, image_size=image_size)
    if name == "drive":
        return DRIVEDataset(root=root, split=split, image_size=image_size)
    raise ValueError(f"Unknown dataset: {name}")


def _load_vae(ckpt_path: str, device: torch.device) -> VAE:
    blob = torch.load(ckpt_path, map_location=device)
    cfg = VAEConfig(**blob["config"])
    vae = VAE(cfg).to(device)
    vae.load_state_dict(blob["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def train(args: FMTrainArgs) -> Path:
    torch.manual_seed(args.seed)
    device = auto_device(args.device)

    image_vae = _load_vae(args.image_vae_ckpt, device)
    mask_vae = _load_vae(args.mask_vae_ckpt, device)

    latent_ch = image_vae.cfg.latent_channels
    cond_ch = image_vae.cfg.latent_channels
    cfg = FMUNetConfig(
        latent_channels=mask_vae.cfg.latent_channels,
        cond_channels=cond_ch,
        base_channels=args.base_channels,
        channel_mults=tuple(args.channel_mults),
        num_res_blocks=args.num_res_blocks,
        attn_levels=tuple(args.attn_levels),
        time_embed_dim=args.time_embed_dim,
    )
    unet = FMUNet(cfg).to(device)
    optimizer = optim.Adam(unet.parameters(), lr=args.lr, betas=(0.9, 0.999))

    train_ds = _build_dataset(args.dataset, args.data_root, "train", args.image_size)
    val_ds = _build_dataset(args.dataset, args.data_root, "val", args.image_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "fm_unet.pt"
    log_path = out_dir / "fm_log.txt"
    log_f = log_path.open("a")

    best_metric = -float("inf")
    for epoch in range(1, args.epochs + 1):
        unet.train()
        running = 0.0
        n = 0
        for batch in tqdm(train_loader, desc=f"[FM] epoch {epoch}", leave=False):
            img = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            mask_pm1 = mask * 2.0 - 1.0
            with torch.no_grad():
                zX = image_vae.encode(img, sample=False)
                zS = mask_vae.encode(mask_pm1, sample=False)
            loss = fm_loss(unet, zX, zS, sigma=args.sigma)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * img.size(0)
            n += img.size(0)
        train_loss = running / max(n, 1)

        if epoch % args.val_every == 0 or epoch == args.epochs:
            dice, iou = _validate(
                unet, image_vae, mask_vae, val_loader, device,
                n_samples=args.n_inference_samples,
                n_steps=args.n_inference_steps,
            )
            line = (
                f"epoch {epoch:04d}  train_loss={train_loss:.4f}  "
                f"val_dice={dice:.4f}  val_iou={iou:.4f}"
            )
            print(line)
            log_f.write(line + "\n")
            log_f.flush()
            if dice > best_metric:
                best_metric = dice
                torch.save(
                    {"model": unet.state_dict(), "config": cfg.to_dict()},
                    ckpt_path,
                )

        if epoch % args.save_every == 0:
            torch.save(
                {"model": unet.state_dict(), "config": cfg.to_dict()},
                out_dir / "fm_unet_last.pt",
            )

    log_f.close()
    return ckpt_path


@torch.no_grad()
def _validate(
    unet: FMUNet,
    image_vae: VAE,
    mask_vae: VAE,
    loader: DataLoader,
    device: torch.device,
    n_samples: int,
    n_steps: int,
) -> tuple[float, float]:
    unet.eval()
    dice_total, iou_total, count = 0.0, 0.0, 0
    for batch in loader:
        img = batch["image"].to(device)
        mask = batch["mask"].to(device)
        zX = image_vae.encode(img, sample=False)
        z_samples = sample(unet, zX, n_samples=n_samples, n_steps=n_steps)
        masks = latents_to_masks(mask_vae.decode, z_samples)
        agg = aggregate(masks)
        d = dice_score(agg.final_mask, mask)
        i = iou_score(agg.final_mask, mask)
        dice_total += float(d) * img.size(0)
        iou_total += float(i) * img.size(0)
        count += img.size(0)
    return dice_total / max(count, 1), iou_total / max(count, 1)
