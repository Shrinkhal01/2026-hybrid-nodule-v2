"""Phase 7 step R: the heat-map scoring must reward focus on the nodule and nothing else.

    python -m pytest tests/test_xai_metrics.py -v      (numpy only, no torch)
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.xai_metrics import ball_mask, center_prior, localisation, spearman_maps  # noqa: E402


def test_flat_map_scores_exactly_one():
    """A flat map spreads its mass like the voxels -> concentration 1.0 (the 'no information' line)."""
    m = localisation(np.ones((32, 32, 32)), diameter_mm=10)
    assert m["concentration"] == pytest.approx(1.0)
    assert m["mass_in"] == pytest.approx(m["vol_frac"])


def test_ball_volume_matches_geometry():
    """A 10 mm nodule = radius 5 -> 4/3*pi*125 = 523.6 mm^3 -> about 1.6% of the 32^3 cube."""
    frac = ball_mask(10).mean()
    assert frac * 32 ** 3 == pytest.approx(4 / 3 * np.pi * 125, rel=0.06)


def test_map_on_the_nodule_scores_high_and_hits():
    cam = ball_mask(10).astype(float)
    m = localisation(cam, diameter_mm=10)
    assert m["mass_in"] == pytest.approx(1.0)
    assert m["concentration"] > 50          # 1 / 0.016 = ~62
    assert m["hit"] == 1.0 and m["com_dist_mm"] < 0.5


def test_corner_map_scores_below_one_and_misses():
    """What your Part 2 maps did: all heat in one corner -> < 1 and a miss."""
    cam = np.zeros((32, 32, 32))
    cam[20:, 20:, 20:] = 1.0
    m = localisation(cam, diameter_mm=10)
    assert m["concentration"] < 1.0 and m["hit"] == 0.0 and m["com_dist_mm"] > 10


def test_all_zero_map_is_flagged_not_scored():
    m = localisation(np.zeros((32, 32, 32)), diameter_mm=10)
    assert m["all_zero"] == 1.0 and np.isnan(m["concentration"])


def test_center_prior_is_the_trivial_winner():
    """Why we report the centre prior: it 'hits' every centred nodule without looking at anything."""
    m = localisation(center_prior(), diameter_mm=8)
    assert m["hit"] == 1.0 and m["concentration"] > 3


def test_spearman_identical_vs_unrelated():
    rng = np.random.default_rng(0)
    a = rng.random((8, 8, 8))
    assert spearman_maps(a, a) == pytest.approx(1.0)
    assert abs(spearman_maps(a, rng.random((8, 8, 8)))) < 0.15
