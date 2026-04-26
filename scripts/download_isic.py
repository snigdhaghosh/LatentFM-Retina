"""Downloader for ISIC-2016 Task 1 (lesion boundary segmentation).

The ISIC 2016 Task 1 zips are CC-0 and publicly hosted on the official S3
bucket (no challenge sign-in required), so we can fetch them directly. If a
mirror ever moves, drop the four zips manually under ``--root`` and rerun
with ``--skip-download``.

Usage::

    python scripts/download_isic.py --root data/ISIC2016
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


URLS = {
    "ISBI2016_ISIC_Part1_Training_Data.zip": (
        "https://isic-archive.s3.amazonaws.com/challenges/2016/"
        "ISBI2016_ISIC_Part1_Training_Data.zip"
    ),
    "ISBI2016_ISIC_Part1_Training_GroundTruth.zip": (
        "https://isic-archive.s3.amazonaws.com/challenges/2016/"
        "ISBI2016_ISIC_Part1_Training_GroundTruth.zip"
    ),
    "ISBI2016_ISIC_Part1_Test_Data.zip": (
        "https://isic-archive.s3.amazonaws.com/challenges/2016/"
        "ISBI2016_ISIC_Part1_Test_Data.zip"
    ),
    "ISBI2016_ISIC_Part1_Test_GroundTruth.zip": (
        "https://isic-archive.s3.amazonaws.com/challenges/2016/"
        "ISBI2016_ISIC_Part1_Test_GroundTruth.zip"
    ),
}


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest} already exists.")
        return
    print(f"[get] {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with dest.open("wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 15):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def extract(zip_path: Path, out_dir: Path) -> None:
    print(f"[unzip] {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        for name, url in URLS.items():
            try:
                download(url, args.root / name)
            except Exception as e:
                print(f"[error] failed to download {name}: {e}", file=sys.stderr)
                print(
                    "Place the zip(s) manually under --root and rerun with "
                    "`--skip-download`.",
                    file=sys.stderr,
                )
                return 1

    for name in URLS:
        zp = args.root / name
        if zp.exists():
            extract(zp, args.root)

    print(f"[done] ISIC-2016 ready under {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
