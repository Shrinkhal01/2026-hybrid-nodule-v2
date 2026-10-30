#!/usr/bin/env python3
"""Phase 7 (step S) numbers for the decisions log, computed from metadata.csv only (no CT needed).

    python scripts/phase7_resampling_stats.py [--meta data/processed/metadata.csv]

For each candidate crop size it reports, over all nodules:
  - padded: patches that would need -1000 HU padding (the 64 x 128 x 128 source is too short)
  - cut:    nodules whose mean diameter is larger than the crop (edges cut off)
and the same two numbers for the legacy 32-voxel crop.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SRC_SHAPE = np.array([64, 128, 128])  # LUNA22 sub-volume (D, H, W)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--meta", type=Path, default=Path("data/processed/metadata.csv"))
    args = p.parse_args()
    df = pd.read_csv(args.meta)
    sp = np.array([[float(v) for v in s.split("x")] for s in df["spacing_dhw_mm"]])  # (n, 3) mm
    diam = df["diameter_mm"].to_numpy()
    n = len(df)

    # legacy: 32 voxels per axis -> physical size differs per nodule and per axis
    legacy_fov = 32 * sp
    print(f"nodules: {n}")
    print(f"legacy 32-voxel crop: depth covers {legacy_fov[:, 0].min():.1f}-{legacy_fov[:, 0].max():.1f} mm, "
          f"in-plane {legacy_fov[:, 1].min():.1f}-{legacy_fov[:, 1].max():.1f} mm")
    print(f"legacy: cut (diameter > smallest crop side) = {int((diam > legacy_fov.min(axis=1)).sum())}, padded = 0")
    print(f"distinct depth spacings: {np.unique(sp[:, 0]).round(3).tolist()}")

    print("\ncrop_mm  voxels@1mm  padded  max_pad_%  cut(diam>crop)")
    for crop in (24, 32, 40, 48, 64):
        # same geometry as resample_crop_iso(): output voxel k sits (k - (N-1)/2) mm from the
        # centre (index (shape-1)/2); it is "inside" if its source index lies in [0, shape-1]
        offsets = np.arange(crop) - (crop - 1) / 2.0
        src = (SRC_SHAPE[None, :, None] - 1) / 2.0 + offsets[None, None, :] / sp[:, :, None]  # (n, 3, N)
        inside = ((src >= 0) & (src <= SRC_SHAPE[None, :, None] - 1)).mean(axis=2)     # (n, 3)
        pad_frac = 1 - inside.prod(axis=1)
        print(f"{crop:>7}  {crop:>9}^3  {int((pad_frac > 0).sum()):>6}  {100 * pad_frac.max():>8.1f}  "
              f"{int((diam > crop).sum()):>13}")


if __name__ == "__main__":
    main()
