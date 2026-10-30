"""
models/xai_metrics.py -- how well does a 3D heat map point at the nodule? (numpy only, CPU)

Phase 7 step R. Every test patch is a 32 mm cube of 1 mm voxels with the nodule at its centre,
so a ball with the nodule's diameter around the centre is a simple "bull's-eye":

    mass_in       share of the heat map's total mass inside the ball
    vol_frac      share of the cube's voxels inside the ball (what a flat map would score)
    concentration mass_in / vol_frac          1.0 = no better than a flat map, >1 = focused on the nodule
    hit           the hottest voxel lies inside the ball (grown by `tol_mm`) -> "pointing game"
    com_dist_mm   distance from the map's centre of mass to the patch centre

Caveat (shown in every report): because nodules sit at the centre, a map that ALWAYS lights the
centre scores well without looking at anything. `center_prior()` builds that trivial map so every
method is compared against it, not only against a flat map.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

PATCH = 32  # voxels per side (1 mm each, Phase 7 step S)


def centre(n: int = PATCH) -> float:
    """Geometric centre index of an n-voxel axis, e.g. 15.5 for 32 (same as resample_crop_iso)."""
    return (n - 1) / 2.0


def ball_mask(diameter_mm: float, n: int = PATCH, spacing_mm: float = 1.0, min_radius_mm: float = 2.0,
              grow_mm: float = 0.0) -> np.ndarray:
    """Boolean (n, n, n) ball centred in the patch with radius = max(diameter/2, min_radius) + grow."""
    r = max(float(diameter_mm) / 2.0, min_radius_mm) + grow_mm
    c = centre(n)
    ax = (np.arange(n) - c) * spacing_mm
    d2 = ax[:, None, None] ** 2 + ax[None, :, None] ** 2 + ax[None, None, :] ** 2
    return d2 <= r * r


def center_prior(sigma_mm: float = 6.0, n: int = PATCH) -> np.ndarray:
    """The trivial 'always look at the centre' map: a Gaussian blob, scaled to [0, 1]."""
    c = centre(n)
    ax = np.arange(n) - c
    d2 = ax[:, None, None] ** 2 + ax[None, :, None] ** 2 + ax[None, None, :] ** 2
    return np.exp(-d2 / (2 * sigma_mm ** 2)).astype(np.float32)


def localisation(cam: np.ndarray, diameter_mm: float, tol_mm: float = 2.0) -> Dict[str, float]:
    """Localisation scores of one heat map (any non-negative (n, n, n) array)."""
    cam = np.clip(np.asarray(cam, dtype=np.float64), 0, None)
    n = cam.shape[0]
    mask = ball_mask(diameter_mm, n)
    vol_frac = float(mask.mean())
    total = float(cam.sum())
    if total <= 1e-12:  # no positive evidence anywhere
        return {"mass_in": np.nan, "vol_frac": vol_frac, "concentration": np.nan, "hit": np.nan,
                "com_dist_mm": np.nan, "all_zero": 1.0}
    mass_in = float((cam * mask).sum() / total)
    peak = np.unravel_index(int(cam.argmax()), cam.shape)
    hit = float(ball_mask(diameter_mm, n, grow_mm=tol_mm)[peak])
    idx = np.indices(cam.shape, dtype=np.float64)
    com = np.array([(idx[i] * cam).sum() / total for i in range(3)])
    return {"mass_in": mass_in, "vol_frac": vol_frac, "concentration": mass_in / vol_frac, "hit": hit,
            "com_dist_mm": float(np.linalg.norm(com - centre(n))), "all_zero": 0.0}


def spearman_maps(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation between two maps (used for the random-weights sanity check).
    Near 1 = the map barely depends on what the network learned."""
    ra = np.argsort(np.argsort(a.ravel())).astype(np.float64)
    rb = np.argsort(np.argsort(b.ravel())).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")
