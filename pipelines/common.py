"""
pipelines/common.py -- shared utilities for Day 2 (training) and Day 2b (XAI).

Contents: feature list, split loading + validation, train-only feature standardisation, the
patch/tabular ``Dataset`` (with 3D augmentation), device selection, prediction helper and
metric functions (ROC-AUC, Youden threshold, confusion-matrix metrics, bootstrap AUC CI).
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger("day2")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# Must match Day 1 (01_data_pipeline.FEATURE_COLUMNS) -- order matters.
FEATURE_COLS: list[str] = [
    "mean_hu", "std_hu", "min_hu", "max_hu", "median_hu",
    "p25_hu", "p75_hu", "iqr_hu", "skewness", "kurtosis",
    "solid_fraction_pct", "hyperdense_fraction_pct",
    "diameter_mm", "calcification", "texture",
]
# Phase 7 step A (ablation): the last 3 columns are radiologist ratings, made by the same
# readers who gave the malignancy label. "image" = only what is measured from the CT patch.
RATING_FEATURE_COLS: list[str] = ["diameter_mm", "calcification", "texture"]
IMAGE_FEATURE_COLS: list[str] = [c for c in FEATURE_COLS if c not in RATING_FEATURE_COLS]  # 12
FEATURE_SETS: dict[str, list[str]] = {"all": FEATURE_COLS, "image": IMAGE_FEATURE_COLS}
PATCH_SHAPE = (32, 32, 32)


# --------------------------------------------------------------------------- #
# Reproducibility / device
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(require_cuda: bool = False) -> torch.device:
    """CUDA if available (any GPU: H100, H200, A100, ...), else CPU.

    Logs the GPU name and compute capability; warns only if the GPU predates Ampere (capability
    < 8.0), where the bf16 autocast used in training is not natively accelerated.
    """
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        log.info("Using GPU: %s (compute capability %d.%d)", name, major, minor)
        if major < 8:
            log.warning("GPU '%s' has no native bf16 support; consider --no-amp.", name)
        torch.backends.cudnn.benchmark = True  # fixed input size -> faster convs (not bit-reproducible)
        return torch.device("cuda")
    if require_cuda:
        raise RuntimeError("No CUDA device available (torch.cuda.is_available() is False). "
                           "Run on the GPU server, or pass --allow-cpu for debugging.")
    log.warning("CUDA not available -- falling back to CPU (training will be very slow).")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def resolve_patch_path(p: str, root: Path = PROJECT_ROOT) -> Path:
    path = Path(p)
    return path if path.is_absolute() else root / path


def load_split(name: str, processed_dir: Path = PROCESSED_DIR, limit: Optional[int] = None,
               seed: int = 0) -> pd.DataFrame:
    """Read ``{name}.csv`` and validate columns, labels, and finiteness of the 15 features.

    ``limit`` (debug only) keeps a class-stratified random subset of that many rows.
    """
    path = processed_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run 01_data_pipeline.py first.")
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLS + ["label", "patch_path", "nodule_id"] if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name} is missing columns: {missing}")
    if not np.isfinite(df[FEATURE_COLS].to_numpy(dtype=float)).all():
        raise ValueError(f"{path.name} contains NaN/inf in feature columns")
    if not set(df["label"].unique()) <= {0, 1}:
        raise ValueError(f"{path.name}: labels must be in {{0, 1}}")
    if limit is not None and limit < len(df):
        # one sample per class, then concat. Not groupby().apply(): pandas >= 3.0 drops the
        # grouping column ("label") from what apply() returns -> KeyError: 'label' downstream
        parts = [g.sample(max(1, round(limit * len(g) / len(df))), random_state=seed)
                 for _, g in df.groupby("label")]
        df = pd.concat(parts).reset_index(drop=True)
    return df.reset_index(drop=True)


def check_feature_contract(processed_dir: Path = PROCESSED_DIR) -> None:
    """Fail early if Day 1's recorded feature list differs from ``FEATURE_COLS``."""
    summary = processed_dir / "dataset_summary.json"
    if not summary.exists():
        log.warning("%s not found; skipping feature-contract check.", summary)
        return
    recorded = json.loads(summary.read_text()).get("config", {}).get("feature_columns")
    if recorded is not None and recorded != FEATURE_COLS:
        raise ValueError(f"FEATURE_COLS differs from Day 1 summary:\n{recorded}\nvs\n{FEATURE_COLS}")


def assert_group_disjoint(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """Re-assert Day 1's leakage guarantee on the files actually being trained on."""
    col = "group_id" if "group_id" in train.columns else "series_uid"
    sets = {n: set(d[col].astype(str)) for n, d in (("train", train), ("val", val), ("test", test))}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if sets[a] & sets[b]:
            raise AssertionError(f"{col} leakage between {a} and {b}: {len(sets[a] & sets[b])} groups")


class TabularScaler:
    """Per-feature standardisation fitted on the **training** split only."""

    def __init__(self, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None) -> None:
        self.mean, self.std = mean, std

    def fit(self, X: np.ndarray) -> "TabularScaler":
        X = np.asarray(X, dtype=np.float64)
        self.mean = X.mean(axis=0)
        self.std = np.where(X.std(axis=0) < 1e-8, 1.0, X.std(axis=0))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("scaler not fitted")
        return ((np.asarray(X, dtype=np.float64) - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> Dict[str, list]:
        return {"mean": list(map(float, self.mean)), "std": list(map(float, self.std))}

    @classmethod
    def from_dict(cls, d: Dict[str, Sequence[float]]) -> "TabularScaler":
        return cls(np.asarray(d["mean"], dtype=np.float64), np.asarray(d["std"], dtype=np.float64))


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def augment_volume(vol: torch.Tensor) -> torch.Tensor:
    """Random flips on each of axes 0/1/2 (p=0.5) and a random k*90deg rotation in the H-W
    (axial) plane. ``vol`` is ``(1, D, H, W)``. Randomness comes from torch's RNG, which the
    DataLoader seeds per worker."""
    for axis in (1, 2, 3):  # D, H, W (dim 0 is channel)
        if torch.rand(1).item() < 0.5:
            vol = torch.flip(vol, dims=(axis,))
    k = int(torch.randint(0, 4, (1,)).item())
    if k:
        vol = torch.rot90(vol, k, dims=(2, 3))
    return vol.contiguous()


class NoduleDataset(Dataset):
    """32^3 patches (``.npy``, float32 in [0,1], axes D,H,W) + 15 standardised tabular features.

    All patches are read into memory once (1000 x 128 KB ~ 128 MB). ``augment=True`` applies
    :func:`augment_volume` (use for the training split only).
    """

    def __init__(self, df: pd.DataFrame, scaler: TabularScaler, augment: bool = False,
                 root: Path = PROJECT_ROOT, feature_cols: Optional[Sequence[str]] = None) -> None:
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.feature_cols = list(feature_cols) if feature_cols is not None else FEATURE_COLS
        vols = []
        for p in self.df["patch_path"]:
            path = resolve_patch_path(p, root)
            try:
                a = np.load(path)
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"patch missing: {path}") from exc
            if a.shape != PATCH_SHAPE:
                raise ValueError(f"{path.name}: expected {PATCH_SHAPE}, got {a.shape}")
            vols.append(a.astype(np.float32, copy=False))
        self.volumes = torch.from_numpy(np.stack(vols)).unsqueeze(1)  # (N,1,D,H,W)
        self.tabular = torch.from_numpy(scaler.transform(self.df[self.feature_cols].to_numpy()))
        self.labels = torch.from_numpy(self.df["label"].to_numpy(dtype=np.float32))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        vol = self.volumes[i]
        if self.augment:
            vol = augment_volume(vol)
        return {"volume": vol, "tabular": self.tabular[i], "label": self.labels[i], "index": i}


def make_loader(ds: NoduleDataset, batch_size: int, shuffle: bool, num_workers: int = 0,
                pin: bool = False) -> DataLoader:
    """``drop_last`` for shuffled (train) loaders so BatchNorm1d never sees a 1-sample batch."""
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle,
                      num_workers=num_workers, pin_memory=pin,
                      persistent_workers=num_workers > 0)


@torch.no_grad()
def predict_probs(model: torch.nn.Module, loader: DataLoader, device: torch.device,
                  use_amp: bool = False) -> np.ndarray:
    """P(malignant) for every sample, in dataset order (loader must not shuffle)."""
    model.eval()
    out = []
    for batch in loader:
        vol = batch["volume"].to(device, non_blocking=True)
        tab = batch["tabular"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(vol, tab)
        out.append(torch.sigmoid(logits.float()).squeeze(1).cpu())
    return torch.cat(out).numpy().astype(np.float64)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC-AUC, or NaN if only one class is present."""
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximising Youden's J = sensitivity + specificity - 1 (predict 1 if p >= t)."""
    fpr, tpr, thr = roc_curve(y, p)
    t = float(thr[int(np.argmax(tpr - fpr))])
    return t if np.isfinite(t) else 0.5


def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, float]:
    """Accuracy, ROC-AUC (threshold-free), sensitivity, specificity, precision, F1, confusion counts."""
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / len(y),
        "roc_auc": safe_auc(y, p),
        "sensitivity": sens, "specificity": spec, "precision": prec, "f1": f1,
        "false_negatives": fn, "false_positives": fp, "true_positives": tp, "true_negatives": tn,
    }


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 1000, seed: int = 0,
                     alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile bootstrap CI for ROC-AUC (resampling nodules; groups are not resampled)."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], p[idx]))
    if not aucs:
        return float("nan"), float("nan")
    lo, hi = np.percentile(aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
