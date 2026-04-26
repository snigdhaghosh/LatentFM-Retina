# LatentFM

A PyTorch implementation of **LatentFM: A Latent Flow Matching Approach for Generative Medical Image Segmentation** (Huynh et al., 2025), targeting:

- **ISIC-2016 Task 1** dermoscopic skin-lesion segmentation (deviation from the paper, which uses ISIC-2018; chosen here because the 2016 zips are CC-0 and ship an official 900/379 train/test split).
- **DRIVE** retinal vessel segmentation (second benchmark, also a deviation from the paper).

The pipeline mirrors the paper:

1. An **image VAE** `E_X / D_X` encodes a 256x256x3 image to a 64x64x3 latent.
2. A **mask VAE** `E_S / D_S` encodes a 256x256x1 mask to a 64x64x3 latent.
3. A **conditional flow-matching UNet** `u_theta(t, z_t, z_X)` learns a velocity field on the latent manifold along the straight-line path `z_t = (1-t)*z_0 + t*z_S` (Eq. 16 of the paper).
4. At inference, multiple source latents are integrated with an Euler ODE solver (Eq. 11-12), decoded through the mask VAE, then averaged for a stable mask and used to compute a pixel-wise variance / confidence map.

## Repository layout

```
latentfm/
  configs/                YAML configs (paper config + MPS preset)
  src/latentfm/
    data/                 ISIC-2016 / DRIVE datasets + paired transforms
    models/               VAE, FM UNet, shared blocks
    flow/                 FM loss + ODE sampler + ensemble aggregation
    train/                training loops for the VAEs and the FM UNet
    eval/                 metrics + evaluation harness
    utils/                config loader, device picker, viz helpers
  scripts/                entry-point CLIs (download / train / eval / infer)
  notebooks/              (placeholder)
```

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Quickstart

### ISIC-2016 (skin lesion)

ISIC-2016 Task 1 is CC-0, so the downloader fetches the four zips (900 train images + masks, 379 test images + masks) directly. If a mirror moves, drop the zips manually under `--root` and rerun with `--skip-download`. If the data is already on disk, just point `common.data_root` in the config at it (default: `data/ISIC2016`) and skip the download step.

The dataset loader auto-detects either the canonical extraction layout (4 `ISBI2016_ISIC_Part1_*` folders directly under root) or a `Train/` + `Test/` wrapper containing those folders.

```bash
python scripts/download_isic.py --root data/ISIC2016   # optional if already downloaded
python scripts/train_vae_image.py --config configs/isic.yaml
python scripts/train_vae_mask.py  --config configs/isic.yaml
python scripts/train_fm.py        --config configs/isic.yaml
python scripts/evaluate.py        --config configs/isic.yaml
python scripts/infer.py           --config configs/isic.yaml \
    --image data/ISIC2016/Test/ISBI2016_ISIC_Part1_Test_Data/ISIC_0000003.jpg \
    --out outputs/isic/infer_demo
```

The 379-image official test set is used as the held-out split; a deterministic 90-image slice of the 900-image training set is reserved for validation, and the remaining 810 are used for training.

Outputs land under `outputs/isic/`:

- `checkpoints/vae_image.pt`, `checkpoints/vae_mask.pt`, `checkpoints/fm_unet.pt`
- `eval/metrics.txt` (Dice / IoU on the test split)
- `eval/viz/` (per-image: input, ground truth, prediction, N samples, confidence map)

### DRIVE (retinal vessels)

DRIVE is gated and cannot be downloaded programmatically. Register at https://drive.grand-challenge.org/, download `DRIVE.zip`, and extract under `data/DRIVE/` so that `data/DRIVE/training/{images,1st_manual}/` and `data/DRIVE/test/{images,1st_manual}/` exist. Then:

```bash
python scripts/download_drive.py --root data/DRIVE   # verifies layout
python scripts/train_vae_image.py --config configs/drive.yaml
python scripts/train_vae_mask.py  --config configs/drive.yaml
python scripts/train_fm.py        --config configs/drive.yaml
python scripts/evaluate.py        --config configs/drive.yaml
```

`configs/drive.yaml` keeps the paper's network configs but trains for more epochs (the dataset is small) and sets a `bce_pos_weight=8.0` for the mask VAE so the thin vessels are not collapsed by the BCE term.

### Apple Silicon (MPS) / dev preset

The full paper config (256x256, batch 4, base channels 64, 250 epochs) is heavy on a Mac. Use the lightweight overlay for development:

```bash
python scripts/train_vae_image.py --config configs/mps_small.yaml
python scripts/train_vae_mask.py  --config configs/mps_small.yaml
python scripts/train_fm.py        --config configs/mps_small.yaml
python scripts/evaluate.py        --config configs/mps_small.yaml
```

`configs/mps_small.yaml` shrinks the resolution to 128 (latent 32x32x3), base channels to 32, batch size to 2, and epoch count to 30. Device selection is automatic: `cuda` -> `mps` -> `cpu`.

A self-contained smoke config is included under `configs/_smoke.yaml` (paired with the recipe in the next section) for verifying the full pipeline runs end-to-end on a tiny synthetic dataset.

## Self-test (no real data needed)

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
from PIL import Image
root = Path('data/mini_isic')
(root / 'images').mkdir(parents=True, exist_ok=True)
(root / 'masks').mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
for i in range(12):
    img = (rng.uniform(0, 255, size=(140, 140, 3))).astype(np.uint8)
    mask = np.zeros((140, 140), dtype=np.uint8)
    cy, cx = rng.integers(40, 100, size=2)
    yy, xx = np.ogrid[:140, :140]
    mask[(yy - cy)**2 + (xx - cx)**2 < 30**2] = 255
    Image.fromarray(img).save(root / 'images' / f'ISIC_{i:07d}.jpg')
    Image.fromarray(mask).save(root / 'masks' / f'ISIC_{i:07d}_segmentation.png')
PY

python scripts/train_vae_image.py --config configs/_smoke.yaml
python scripts/train_vae_mask.py  --config configs/_smoke.yaml
python scripts/train_fm.py        --config configs/_smoke.yaml
python scripts/evaluate.py        --config configs/_smoke.yaml
```

The smoke run trains tiny networks for 2 epochs each on 64x64 inputs; full pipeline completes in well under a minute on CPU.

## Math reference (paper -> code)

| Paper | Code |
|-------|------|
| Eq. 8 / Eq. 16: linear interpolation `z_t = (1-t) z_0 + t z_S` | `latentfm.flow.matching.interpolate` |
| Eq. 9 / Eq. 17: target velocity `z_S - z_0` | `latentfm.flow.matching.fm_loss` |
| Eq. 10 / Eq. 18: MSE flow-matching loss | `latentfm.flow.matching.fm_loss` |
| Eq. 11-12: ODE integration & multi-sample inference | `latentfm.flow.matching.sample` |
| Eq. 14-15: VAE ELBO | `latentfm.models.vae.VAE.elbo_loss` |
| Confidence map = pixel-wise variance | `latentfm.flow.ensemble.aggregate` |

## Notes on retina (DRIVE)

DRIVE vessels are thin and sparse (~10% positive pixels), which biases the binary-mask formulation toward the background class. Two practical knobs:

- **`mask.bce_pos_weight`** in the config: up-weights positive pixels in the mask VAE's reconstruction term.
- If thin-vessel collapse persists, switch the mask VAE recon loss to a Dice term in `train_vae.py` (search for `recon_type`); both `bce_l1` and a Dice variant are easy substitutions.

## Differences from the paper (implementation choices, all flagged)

- The paper does not specify the exact reconstruction term for the mask VAE; we use BCE + 0.1 * L1 on the `[0, 1]` view of the tanh output (`recon_type='bce_l1'`).
- We train the FM UNet with the deterministic Dirac path (`sigma=0.0`, Eq. 7); a positive `sigma` (Eq. 5) is exposed as a config knob.
- The paper uses 5 inference samples; the value lives in `fm.n_inference_samples` (default 5, easily increased).
- For DRIVE we add a positive-class weight (`mask.bce_pos_weight: 8.0`) and longer training, neither of which is in the paper but both are needed because the paper's protocol assumes a more balanced dataset like ISIC.
- The paper trains/evaluates on ISIC-2018 (3194 / 250 / 250). We use ISIC-2016 Task 1 instead, which has a CC-0 license and an official 900 / 379 train / test split (we hold 90 train images out as val). Numerical comparisons to the paper are therefore not apples-to-apples.

## Citation

```bibtex
@article{huynh2025latentfm,
  title={LatentFM: A Latent Flow Matching Approach for Generative Medical Image Segmentation},
  author={Huynh, Trinh Ngoc and Anh, Nguyen Kim Hoang and Toan, Nguyen Hai and Long, Tran Quoc},
  journal={arXiv preprint arXiv:2512.04821},
  year={2025}
}
```
