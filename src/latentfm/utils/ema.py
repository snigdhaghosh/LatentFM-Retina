"""Exponential moving average of model parameters.

Maintains a shadow copy of `model.state_dict()`, updated after each
optimizer step as `shadow = decay * shadow + (1 - decay) * w`. For
inference and validation, swap in the shadow weights via the
`average_parameters(model)` context manager (which restores the raw
weights on exit). The serialized `state_dict()` is API-compatible with
`model.load_state_dict()` so EMA checkpoints load through the existing
inference pipeline with no further changes.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            shadow = self.shadow[k]
            if v.dtype.is_floating_point:
                shadow.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                # Integer buffers (counters, BN running counts) just track current.
                shadow.copy_(v.detach())

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone() for k, v in sd.items()}

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        """Temporarily swap `model`'s weights for the EMA shadow."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)
        try:
            yield
        finally:
            model.load_state_dict(backup, strict=True)
