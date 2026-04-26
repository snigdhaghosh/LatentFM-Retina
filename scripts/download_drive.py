"""Stub downloader for the DRIVE retinal vessel segmentation dataset.

DRIVE is hosted at https://drive.grand-challenge.org and requires accepting
their terms / creating an account. There is no stable, license-clean direct
URL we can hit programmatically.

This script just verifies the expected layout and prints instructions if
the data isn't present.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REQUIRED_DIRS = [
    "training/images",
    "training/1st_manual",
    "test/images",
    "test/1st_manual",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    args = ap.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    missing = [d for d in REQUIRED_DIRS if not (args.root / d).is_dir()]
    if missing:
        print(
            "DRIVE is gated. Please:\n"
            "  1. Register at https://drive.grand-challenge.org/\n"
            "  2. Download `DRIVE.zip` (training + test).\n"
            f"  3. Extract under {args.root} so that the following exist:",
            file=sys.stderr,
        )
        for d in REQUIRED_DIRS:
            print(f"     {args.root / d}", file=sys.stderr)
        return 1

    print(f"[ok] DRIVE layout verified at {args.root}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
