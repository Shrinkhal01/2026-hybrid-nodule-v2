"""Regression test: load_split(limit=...) must keep the 'label' column on every pandas version.

pandas >= 3.0 changed groupby().apply() so the grouping column was dropped -> KeyError: 'label'
in the --limit smoke test on the GPU server (Phase 7, run 1).
    python -m pytest tests/test_load_split_limit.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")  # pipelines.common imports torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipelines.common import FEATURE_COLS, load_split  # noqa: E402


def test_limit_keeps_label_and_class_balance(tmp_path):
    n = 100  # 60 benign, 40 malignant
    df = pd.DataFrame({c: np.random.default_rng(0).normal(size=n) for c in FEATURE_COLS})
    df["label"] = [0] * 60 + [1] * 40
    df["patch_path"] = [f"p{i}.npy" for i in range(n)]
    df["nodule_id"] = [f"n{i}" for i in range(n)]
    df.to_csv(tmp_path / "train.csv", index=False)

    out = load_split("train", tmp_path, limit=20, seed=0)
    assert "label" in out.columns
    assert len(out) == 20
    assert out["label"].value_counts().to_dict() == {0: 12, 1: 8}  # 60:40 kept
