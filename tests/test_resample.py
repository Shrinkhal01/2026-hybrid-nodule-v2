"""Tests for the Phase 7 resampling core: resample_crop_iso().

Run (CPU only, ~1 s):
    python -m pytest tests/test_resample.py -v

The same tests run against YOUR hand-written version if tests/my_resample.py exists and
defines  resample_crop_iso(vol, spacing_dhw, spacing_mm=1.0, crop_mm=32.0,
pad_value=-1000.0) -> (crop, pad_fraction).
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses look the module up here while it loads
    spec.loader.exec_module(mod)
    return mod


IMPLS = {"pipeline": _load(ROOT / "01_data_pipeline.py", "data_pipeline").resample_crop_iso}
_mine = Path(__file__).with_name("my_resample.py")
if _mine.exists():
    IMPLS["mine"] = _load(_mine, "my_resample").resample_crop_iso

SHAPE = (64, 128, 128)          # LUNA22 sub-volume (D, H, W)
THICK = (2.5, 0.703, 0.703)     # most common thick-slice spacing in the data (288 nodules)
THIN = (0.625, 0.703, 0.703)    # common thin-slice spacing (115 nodules)


def ball(spacing, radius_mm=6.0, shape=SHAPE):
    """1 inside a sphere of radius_mm (physical), 0 outside, centred at (shape-1)/2."""
    idx = np.indices(shape, dtype=np.float64)
    c = (np.asarray(shape, dtype=np.float64) - 1) / 2
    r2 = sum(((idx[i] - c[i]) * spacing[i]) ** 2 for i in range(3))
    return (r2 <= radius_mm ** 2).astype(np.float32)


@pytest.fixture(params=list(IMPLS))
def resample(request):
    return IMPLS[request.param]


@pytest.mark.parametrize("spacing", [THICK, THIN, (0.45, 0.605, 0.605), (1.0, 1.0, 1.0)])
def test_shape_is_32_cubed_for_any_spacing(resample, spacing):
    crop, _ = resample(np.zeros(SHAPE, np.float32), spacing)
    assert crop.shape == (32, 32, 32)
    assert crop.dtype == np.float32


def test_1mm_source_equals_legacy_crop(resample):
    """Already 1 mm isotropic -> nothing to interpolate -> identical to vol[16:48, 48:80, 48:80]."""
    vol = np.random.default_rng(0).normal(-500, 300, SHAPE).astype(np.float32)
    crop, pad = resample(vol, (1.0, 1.0, 1.0))
    np.testing.assert_allclose(crop, vol[16:48, 48:80, 48:80], atol=1e-3)
    assert pad == 0.0


def test_linear_ramp_is_reproduced_exactly(resample):
    """Trilinear interpolation is exact for a linear function of position (in mm)."""
    sd, sh, sw = THICK
    d, h, w = np.indices(SHAPE, dtype=np.float64)
    vol = (3.0 * d * sd + 2.0 * h * sh + 1.0 * w * sw).astype(np.float32)
    crop, _ = resample(vol, THICK)
    c = (np.asarray(SHAPE) - 1) / 2
    off = np.arange(32) - 15.5                          # mm from the centre
    D, H, W = np.meshgrid(c[0] * sd + off, c[1] * sh + off, c[2] * sw + off, indexing="ij")
    np.testing.assert_allclose(crop, 3.0 * D + 2.0 * H + 1.0 * W, atol=0.05)


@pytest.mark.parametrize("spacing", [THICK, THIN])
def test_ball_keeps_its_physical_size(resample, spacing):
    """A 12 mm ball must come out ~12 voxels across in EVERY axis, whatever the slice gap."""
    crop, _ = resample(ball(spacing), spacing, pad_value=0.0)
    inside = crop > 0.5
    true_volume = 4 / 3 * np.pi * 6.0 ** 3              # 904.8 mm^3 = voxels at 1 mm
    assert abs(inside.sum() - true_volume) / true_volume < 0.15
    for axis in range(3):
        other = tuple(a for a in range(3) if a != axis)
        extent = int(inside.any(axis=other).sum())
        assert 10 <= extent <= 14, f"axis {axis}: {extent} voxels across"


def test_ball_stays_centred(resample):
    crop, _ = resample(ball(THICK), THICK, pad_value=0.0)
    com = np.array(np.nonzero(crop > 0.5)).mean(axis=1)
    np.testing.assert_allclose(com, [15.5, 15.5, 15.5], atol=0.5)


def test_padding_only_where_source_ends(resample):
    vol = np.zeros(SHAPE, np.float32)
    # 0.45 mm slices: 64 slices = 28.8 mm < 32 mm -> 2 output slices padded at each end
    crop, pad = resample(vol, (0.45, 0.605, 0.605))
    assert pad == pytest.approx(4 / 32)
    assert crop[0].max() == -1000.0 and crop[-1].max() == -1000.0
    assert crop[2:-2].min() == 0.0
    # 2.5 mm slices: 160 mm of source depth -> no padding
    _, pad = resample(vol, THICK)
    assert pad == 0.0
