"""Phase 7 step A: the 'image' feature set must contain no radiologist rating.

    python -m pytest tests/test_feature_sets.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")  # pipelines.common imports torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipelines.common import (FEATURE_COLS, FEATURE_SETS, IMAGE_FEATURE_COLS,  # noqa: E402
                              RATING_FEATURE_COLS, NoduleDataset, TabularScaler)


def test_image_set_has_no_ratings_and_keeps_order():
    assert len(IMAGE_FEATURE_COLS) == 12
    assert not set(IMAGE_FEATURE_COLS) & set(RATING_FEATURE_COLS)
    assert IMAGE_FEATURE_COLS == [c for c in FEATURE_COLS if c in IMAGE_FEATURE_COLS]
    assert FEATURE_SETS["all"] == FEATURE_COLS and len(FEATURE_COLS) == 15


def test_dataset_uses_only_the_chosen_columns(tmp_path):
    n = 4
    df = pd.DataFrame({c: np.arange(n, dtype=float) + i for i, c in enumerate(FEATURE_COLS)})
    df["diameter_mm"] = 999.0  # a rating: must NOT reach the model in the image set
    for i in range(n):
        np.save(tmp_path / f"p{i}.npy", np.zeros((32, 32, 32), np.float32))
    df["patch_path"] = [str(tmp_path / f"p{i}.npy") for i in range(n)]
    df["label"] = [0, 1, 0, 1]
    scaler = TabularScaler().fit(df[IMAGE_FEATURE_COLS].to_numpy())
    ds = NoduleDataset(df, scaler, feature_cols=IMAGE_FEATURE_COLS)
    assert ds[0]["tabular"].shape == (12,)
    np.testing.assert_allclose(ds.tabular.numpy(), scaler.transform(df[IMAGE_FEATURE_COLS].to_numpy()))
