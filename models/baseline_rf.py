"""
models/baseline_rf.py -- Classical baseline (Model A): class-balanced Random Forest on the 15
tabular radiomic/metadata features.

scikit-learn's ``RandomForestClassifier(class_weight='balanced')`` is used (no imbalanced-learn
dependency): 150 trees, max_depth=6. Trees are scale-invariant, so raw (unstandardised) features
are used.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def build_balanced_rf(seed: int = 42, n_jobs: int = -1) -> RandomForestClassifier:
    """150 trees, depth 6, balanced class weights."""
    return RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        class_weight="balanced",
        random_state=seed,
        n_jobs=n_jobs,
    )


class BaselineRF:
    """Thin wrapper: fit on train features, return P(malignant), persist with joblib."""

    def __init__(self, feature_cols: Sequence[str], seed: int = 42) -> None:
        self.feature_cols = list(feature_cols)
        self.model = build_balanced_rf(seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaselineRF":
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y).astype(int)
        if X.shape[1] != len(self.feature_cols):
            raise ValueError(f"expected {len(self.feature_cols)} features, got {X.shape[1]}")
        if len(np.unique(y)) < 2:
            raise ValueError("training labels contain a single class")
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability of the malignant class (label 1), shape ``(N,)``."""
        proba = self.model.predict_proba(np.asarray(X, dtype=np.float64))
        return proba[:, list(self.model.classes_).index(1)]

    def feature_importances(self) -> dict[str, float]:
        return dict(zip(self.feature_cols, map(float, self.model.feature_importances_)))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "feature_cols": self.feature_cols}, path)

    @classmethod
    def load(cls, path: Path) -> "BaselineRF":
        blob = joblib.load(path)
        obj = cls(blob["feature_cols"])
        obj.model = blob["model"]
        return obj
