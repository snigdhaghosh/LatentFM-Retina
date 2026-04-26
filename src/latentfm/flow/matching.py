"""Flow-matching primitives in latent space.

Direct implementations of the paper's equations:
    Eq. 16 : straight-line interpolation `z_t = (1 - t) * z_0 + t * z_S`
    Eq. 17 : ground-truth velocity `u(t, z_t, z_X) = z_S - z_0`
    Eq. 18 : MSE loss between the predicted and target velocity
    Eq. 11-12 : ODE integration over `t in [0, 1]` to produce a sample (and
                multiple samples for ensembling)
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def interpolate(z0: torch.Tensor, zS: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Eq. 16: `z_t = (1 - t) z_0 + t z_S`. `t` broadcasts over spatial dims."""
    while t.dim() < z0.dim():
        t = t.unsqueeze(-1)
    return (1.0 - t) * z0 + t * zS


def fm_loss(
    unet: nn.Module,
    zX: torch.Tensor,
    zS: torch.Tensor,
    sigma: float = 0.0,
) -> torch.Tensor:
    """Conditional flow-matching loss (Eq. 18).

    Args:
        unet: velocity-field network `u_theta(z_t, t, z_X)`.
        zX  : image latent (the conditioning), shape `[B, C, h, w]`.
        zS  : target mask latent, shape `[B, C, h, w]`.
        sigma: optional standard deviation of the concentrated Gaussian path
               (Eq. 5). Set to 0 to use the exact Dirac path used by the paper.
    """
    B = zS.size(0)
    t = torch.rand(B, device=zS.device)
    z0 = torch.randn_like(zS)
    zt = interpolate(z0, zS, t)
    if sigma > 0.0:
        zt = zt + sigma * torch.randn_like(zt)
    target = zS - z0
    pred = unet(zt, t, zX)
    return F.mse_loss(pred, target)


@torch.no_grad()
def sample(
    unet: nn.Module,
    zX: torch.Tensor,
    n_samples: int = 5,
    n_steps: int = 50,
    method: str = "euler",
) -> torch.Tensor:
    """Eq. 11-12: integrate the ODE `dz/dt = u_theta(z, t, zX)` from t=0 to t=1.

    Returns latent samples of shape `[N, B, C, h, w]`.
    """
    if method not in {"euler", "heun"}:
        raise ValueError(f"Unsupported ODE method: {method}")
    B, C, h, w = zX.shape
    device = zX.device
    z = torch.randn(n_samples, B, C, h, w, device=device)

    dt = 1.0 / n_steps
    for k in range(n_steps):
        t = torch.full((B,), k * dt, device=device)
        for i in range(n_samples):
            v_k = unet(z[i], t, zX)
            if method == "euler":
                z[i] = z[i] + dt * v_k
            else:  # heun
                z_pred = z[i] + dt * v_k
                t_next = torch.full((B,), (k + 1) * dt, device=device)
                v_next = unet(z_pred, t_next, zX)
                z[i] = z[i] + 0.5 * dt * (v_k + v_next)
    return z


@torch.no_grad()
def sample_one(
    unet: nn.Module,
    zX: torch.Tensor,
    z0: torch.Tensor | None = None,
    n_steps: int = 50,
    method: str = "euler",
) -> torch.Tensor:
    """Single-sample ODE integration. Useful for unit tests / single-image demo."""
    B, C, h, w = zX.shape
    device = zX.device
    z = z0 if z0 is not None else torch.randn(B, C, h, w, device=device)

    dt = 1.0 / n_steps
    for k in range(n_steps):
        t = torch.full((B,), k * dt, device=device)
        v_k = unet(z, t, zX)
        if method == "euler":
            z = z + dt * v_k
        else:
            z_pred = z + dt * v_k
            t_next = torch.full((B,), (k + 1) * dt, device=device)
            v_next = unet(z_pred, t_next, zX)
            z = z + 0.5 * dt * (v_k + v_next)
    return z
