"""YAML config loading with simple dict-merge support."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise TypeError(f"Config root must be a mapping, got {type(cfg)}")

    base = cfg.pop("_base_", None)
    if base is not None:
        base_path = (path.parent / base).resolve()
        base_cfg = load_config(base_path)
        cfg = merge_configs(base_cfg, cfg)
    return cfg


def merge_configs(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge `overlay` into `base`. Overlay wins on leaves."""
    out: dict[str, Any] = deepcopy(dict(base))
    for k, v in overlay.items():
        if k in out and isinstance(out[k], Mapping) and isinstance(v, Mapping):
            out[k] = merge_configs(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out
