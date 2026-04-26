"""Trains either the image VAE or the mask VAE.

Both share a single architecture (`models.vae.VAE`); the mode flag selects
input channels, recon-loss type, and (for masks) the [0, 1] -> [-1, 1] mapping
before feeding the network.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.isic import ISICDataset
from ..data.drive import DRIVEDataset
from ..eval.metrics import psnr_score, ssim_score, dice_score, iou_score
from ..models.vae import VAE, VAEConfig
from ..utils.device import auto_device
from ..utils.viz import save_grid


Mode = Literal["image", "mask"]


@dataclass
class VAETrainArgs:
    mode: Mode
    dataset: str               # "isic" | "drive"
    data_root: str
    out_dir: str
    image_size: int = 256
    batch_size: int = 4
    epochs: int = 250
    lr: float = 1e-5
    base_channels: int = 64
    channel_mults: tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    attn_levels: tuple[int, ...] = (2,)
    latent_channels: int = 3
    kl_weight: float = 1.0e-6
    bce_pos_weight: float | None = None
    val_every: int = 1
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


def _input_for_mode(batch: dict, mode: Mode) -> torch.Tensor:
    """Returns the network input in [-1, 1] regardless of mode."""
    if mode == "image":
        return batch["image"]
    mask01 = batch["mask"]
    return mask01 * 2.0 - 1.0


def train(args: VAETrainArgs) -> Path:
    torch.manual_seed(args.seed)
    device = auto_device(args.device)

    in_ch = 3 if args.mode == "image" else 1
    cfg = VAEConfig(
        in_channels=in_ch,
        out_channels=in_ch,
        base_channels=args.base_channels,
        channel_mults=tuple(args.channel_mults),
        num_res_blocks=args.num_res_blocks,
        attn_levels=tuple(args.attn_levels),
        latent_channels=args.latent_channels,
        final_activation="tanh",
    )
    model = VAE(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

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
    ckpt_path = out_dir / f"vae_{args.mode}.pt"
    log_path = out_dir / f"vae_{args.mode}_log.txt"
    log_f = log_path.open("a")

    recon_type = "l1" if args.mode == "image" else "bce_l1"

    best_metric = -float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for batch in tqdm(train_loader, desc=f"[VAE-{args.mode}] epoch {epoch}", leave=False):
            x = _input_for_mode(batch, args.mode).to(device, non_blocking=True)
            recon, mu, logvar = model(x)
            losses = model.elbo_loss(
                x, recon, mu, logvar,
                recon_type=recon_type,
                kl_weight=args.kl_weight,
                bce_pos_weight=args.bce_pos_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += losses["loss"].item() * x.size(0)
            n += x.size(0)
        train_loss = running / max(n, 1)

        if epoch % args.val_every == 0:
            metric = _validate(model, val_loader, device, args.mode, out_dir, epoch)
            line = (
                f"epoch {epoch:04d}  train_loss={train_loss:.4f}  val_metric={metric:.4f}"
            )
            print(line)
            log_f.write(line + "\n")
            log_f.flush()
            if metric > best_metric:
                best_metric = metric
                torch.save(
                    {"model": model.state_dict(), "config": cfg.to_dict(), "mode": args.mode},
                    ckpt_path,
                )

        if epoch % args.save_every == 0:
            torch.save(
                {"model": model.state_dict(), "config": cfg.to_dict(), "mode": args.mode},
                out_dir / f"vae_{args.mode}_last.pt",
            )

    log_f.close()
    return ckpt_path


@torch.no_grad()
def _validate(
    model: VAE,
    loader: DataLoader,
    device: torch.device,
    mode: Mode,
    out_dir: Path,
    epoch: int,
) -> float:
    model.eval()
    total, count = 0.0, 0
    saved = False
    for batch in loader:
        x = _input_for_mode(batch, mode).to(device)
        recon, _, _ = model(x)
        if mode == "image":
            m = ssim_score(recon, x)
        else:
            recon01 = (recon + 1.0) / 2.0
            x01 = (x + 1.0) / 2.0
            m = dice_score(recon01, x01)
        total += float(m) * x.size(0)
        count += x.size(0)
        if not saved:
            grid_dir = out_dir / "vae_recons"
            grid_dir.mkdir(parents=True, exist_ok=True)
            n_show = min(4, x.size(0))
            save_grid(
                [t for t in x[:n_show]],
                grid_dir / f"input_{mode}_e{epoch:04d}.png",
                nrow=n_show,
            )
            save_grid(
                [t for t in recon[:n_show]],
                grid_dir / f"recon_{mode}_e{epoch:04d}.png",
                nrow=n_show,
            )
            saved = True
    return total / max(count, 1)
