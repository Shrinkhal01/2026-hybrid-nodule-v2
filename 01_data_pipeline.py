#!/usr/bin/env python3
"""
01_data_pipeline.py  --  Day 1 of 2026-hybrid-xai-lung-nodule

Data acquisition -> volumetric normalisation -> 15 tabular radiomics ->
leakage-free (patient/scan-disjoint) 70/15/15 split -> verification self-check.

Source dataset: LIDC-IDRI 1176-nodule benchmark (LUNA22-ISMI release,
https://zenodo.org/record/6559584): LIDC-IDRI_1176.zip (NIfTI sub-volumes)
+ LIDC-IDRI_1176.npy (per-nodule radiologist metadata).

Typical usage
-------------
    # auto-download into data/raw, then run everything
    python 01_data_pipeline.py

    # use local files
    python 01_data_pipeline.py --raw-dir /path/to/nifti_folder --meta-path /path/to/meta.npy

    # look at the metadata schema first (recommended on first run)
    python 01_data_pipeline.py --raw-dir ... --meta-path ... --inspect

    # re-run only the verification on existing outputs
    python 01_data_pipeline.py --verify-only

Outputs (relative to --data-dir, default ./data)
-------------------------------------------------
    processed/patches/{patch_filename}.npy   float32 (32,32,32) in [0,1]
    processed/{train,val,test}.csv
    processed/metadata.csv                   combined audit trail (+ `split`)
    processed/dataset_summary.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

log = logging.getLogger("pipeline")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent

ZENODO_BASE = "https://zenodo.org/records/6559584/files"
BENCHMARK_FILES = {  # filename -> md5 (from the Zenodo record)
    "LIDC-IDRI_1176.npy": "37f7e77b0d8ad3310e0f7fd0445bc72c",
    "LIDC-IDRI_1176.zip": "4b247b69de74a8df58912a7395944cc1",
}

ROI_SIZE = 32
PAD_HU = -1000.0
HU_MIN, HU_MAX = -1000.0, 400.0
SOLID_HU, HYPERDENSE_HU = -400.0, 100.0
INDETERMINATE_SCORE = 3.0

INTENSITY_FEATURES = [
    "mean_hu", "std_hu", "min_hu", "max_hu", "median_hu",
    "p25_hu", "p75_hu", "iqr_hu", "skewness", "kurtosis",
]
DENSITY_FEATURES = ["solid_fraction_pct", "hyperdense_fraction_pct"]
METADATA_FEATURES = ["diameter_mm", "calcification", "texture"]
FEATURE_COLUMNS = INTENSITY_FEATURES + DENSITY_FEATURES + METADATA_FEATURES  # 15

# Canonical metadata column -> accepted (normalised) aliases, in priority order.
COLUMN_ALIASES: dict[str, list[str]] = {
    "volume_file": ["volumefile", "filename", "file", "filepath", "path", "nifti",
                    "niftipath", "volume", "patch", "image", "imagepath"],
    "nodule_id": ["noduleid", "nodule", "annotationid", "uid", "id", "noduleindex", "index"],
    "series_uid": ["seriesinstanceuid", "seriesuid", "seriesid", "series"],
    "patient_id": ["patientid", "patient", "pid", "subjectid", "subject"],
    "malignancy": ["malignancy", "malignancymean", "meanmalignancy", "malignancyscore",
                   "malignancyconsensus", "consensusmalignancy", "meanmalignancyscore"],
    "diameter_mm": ["diametermm", "diameter", "meandiameter", "meandiametermm", "sizemm", "size"],
    "calcification": ["calcification", "calcificationmean", "meancalcification",
                      "calcificationscore", "calc"],
    "texture": ["texture", "texturemean", "meantexture", "texturescore"],
}


# Exact keys of the official LIDC-IDRI_1176.npy (list of dicts). Applied softly:
# a default only takes effect if that key exists; --column-map is strict.
DEFAULT_COLUMN_MAP: dict[str, str] = {
    "volume_file": "Filename",
    "series_uid": "SeriesInstanceUID",
    "malignancy": "Malignancy",
    "diameter_mm": "Diameter",
    "calcification": "Calcification",
    "texture": "Texture",
}
DEFAULT_RAW_SUBDIR = "LIDC-IDRI"                 # <data-dir>/raw/LIDC-IDRI/*.nii.gz
DEFAULT_META_NAME = "LIDC-IDRI_1176.npy"         # <data-dir>/raw/LIDC-IDRI_1176.npy


@dataclass
class Config:
    data_dir: Path
    raw_dir: Optional[Path]
    meta_path: Optional[Path]
    no_download: bool
    keep_archive: bool
    seed: int
    split_tries: int
    canonical: bool
    allow_index_order: bool
    column_map: dict[str, str]
    column_names: Optional[list[str]]

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def patch_dir(self) -> Path:
        return self.processed_dir / "patches"


# --------------------------------------------------------------------------- #
# 1. Data ingestion
# --------------------------------------------------------------------------- #
def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download_file(url: str, dest: Path, expected_md5: Optional[str]) -> None:
    """Stream `url` to `dest` (via a .part file) and verify the MD5."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    log.info("Downloading %s", url)
    with urllib.request.urlopen(url) as resp, open(part, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        with tqdm(total=total or None, unit="B", unit_scale=True, desc=dest.name) as bar:
            for block in iter(lambda: resp.read(1 << 20), b""):
                out.write(block)
                bar.update(len(block))
    if expected_md5 and md5sum(part) != expected_md5:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"MD5 mismatch for {dest.name}; download corrupted, please retry.")
    part.replace(dest)


def safe_extract_zip(zip_path: Path, dest: Path) -> None:
    """Extract a zip, refusing members that would escape `dest` (zip-slip)."""
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if dest != target and dest not in target.parents:
                raise RuntimeError(f"Unsafe path in archive: {member.filename}")
        zf.extractall(dest)


def nifti_stem(name: str) -> str:
    n = Path(name).name
    for suf in (".nii.gz", ".nii"):
        if n.lower().endswith(suf):
            return n[: -len(suf)]
    return Path(n).stem


def list_nifti(root: Path) -> list[Path]:
    files = [
        p for p in root.rglob("*")
        if p.is_file() and p.name.lower().endswith((".nii", ".nii.gz"))
        and not p.name.startswith("._") and "__MACOSX" not in p.parts
    ]
    return sorted(files)


def find_metadata_file(root: Path) -> Path:
    cands = [p for p in root.rglob("*") if p.suffix.lower() in (".npy", ".csv")
             and not p.name.startswith("._") and "patches" not in p.parts]
    named = [p for p in cands if p.name == "LIDC-IDRI_1176.npy"]
    if named:
        return named[0]
    if len(cands) == 1:
        return cands[0]
    raise FileNotFoundError(
        f"Could not identify the metadata file under {root} "
        f"({len(cands)} candidates). Pass --meta-path explicitly."
    )


def resolve_raw_inputs(cfg: Config) -> tuple[list[Path], Path]:
    """Return (nifti files, metadata path), downloading what is missing."""
    default_vols = cfg.data_dir / "raw" / DEFAULT_RAW_SUBDIR
    raw_dir = cfg.raw_dir or (default_vols if default_vols.is_dir() else cfg.data_dir / "raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    niftis = list_nifti(raw_dir)
    meta_path = cfg.meta_path
    default_meta = cfg.data_dir / "raw" / DEFAULT_META_NAME
    if meta_path is None and default_meta.exists():
        meta_path = default_meta
    if meta_path is None:
        try:
            meta_path = find_metadata_file(raw_dir)
        except FileNotFoundError:
            meta_path = None

    if niftis and meta_path and meta_path.exists():
        log.info("Using local data: %d NIfTI files in %s; metadata %s",
                 len(niftis), raw_dir, meta_path)
        return niftis, meta_path

    if cfg.no_download:
        raise FileNotFoundError(
            f"Raw data incomplete (NIfTI found: {len(niftis)}, metadata: {meta_path}) "
            "and --no-download was set."
        )

    log.info("Raw data not (fully) present in %s -> downloading benchmark.", raw_dir)
    if not (meta_path and meta_path.exists()):
        meta_path = raw_dir / "LIDC-IDRI_1176.npy"
        download_file(f"{ZENODO_BASE}/{meta_path.name}?download=1", meta_path,
                      BENCHMARK_FILES[meta_path.name])
    if not niftis:
        zpath = raw_dir / "LIDC-IDRI_1176.zip"
        if not zpath.exists():
            download_file(f"{ZENODO_BASE}/{zpath.name}?download=1", zpath,
                          BENCHMARK_FILES[zpath.name])
        log.info("Extracting %s", zpath)
        safe_extract_zip(zpath, raw_dir / "LIDC-IDRI_1176")
        if not cfg.keep_archive:
            zpath.unlink(missing_ok=True)
        niftis = list_nifti(raw_dir)
    if not niftis:
        raise RuntimeError(
            "No .nii/.nii.gz files found after extraction. The archive layout differs "
            "from expectations; point --raw-dir at the folder containing the volumes."
        )
    return niftis, meta_path


def _norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def load_metadata_raw(path: Path, column_names: Optional[list[str]]) -> pd.DataFrame:
    """Load the metadata array into a DataFrame, whatever container it uses."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    try:
        arr = np.load(path, allow_pickle=False)
    except ValueError:  # object arrays need pickle; the file is a user-supplied/official one
        log.warning("Metadata requires pickle to load; only use trusted files.")
        arr = np.load(path, allow_pickle=True)

    if arr.dtype == object and arr.ndim == 0:  # 0-d array wrapping a dict/list
        arr = arr.item()
    if isinstance(arr, dict):
        try:
            return pd.DataFrame(arr)
        except ValueError:  # dict keyed by nodule id -> record
            return pd.DataFrame.from_dict(arr, orient="index").reset_index(names="nodule_id")
    if isinstance(arr, (list, tuple)):
        return pd.DataFrame(list(arr))
    if arr.dtype.names:  # structured / record array
        return pd.DataFrame({n: arr[n].tolist() for n in arr.dtype.names})
    if arr.dtype == object and arr.ndim == 1 and len(arr) and isinstance(arr[0], dict):
        return pd.DataFrame(list(arr))
    if arr.ndim == 2:
        if column_names is None:
            raise ValueError(
                f"Metadata is a bare 2-D array {arr.shape} with no column names. "
                "Re-run with --column-names a,b,c,... (use --inspect to see it)."
            )
        if len(column_names) != arr.shape[1]:
            raise ValueError(f"--column-names has {len(column_names)} names, array has {arr.shape[1]} columns")
        return pd.DataFrame(arr, columns=column_names)
    raise ValueError(f"Unsupported metadata layout: dtype={arr.dtype}, shape={arr.shape}")


def _cell_mean(v: Any) -> float:
    """Collapse a per-rater list/array to its mean; pass scalars through."""
    if isinstance(v, (list, tuple, np.ndarray)):
        a = np.asarray(v, dtype=float).ravel()
        return float(np.nanmean(a)) if a.size and not np.all(np.isnan(a)) else np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def canonicalize_metadata(raw: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
    """Map arbitrary column names to canonical ones and reduce per-rater cells to means."""
    by_norm = {_norm(c): c for c in raw.columns}
    resolved: dict[str, str] = {}
    for canon, aliases in COLUMN_ALIASES.items():
        if canon not in column_map and DEFAULT_COLUMN_MAP.get(canon) in raw.columns:
            resolved[canon] = DEFAULT_COLUMN_MAP[canon]
            continue
        if canon in column_map:
            if column_map[canon] not in raw.columns:
                raise KeyError(f"--column-map {canon}->{column_map[canon]!r} not in {list(raw.columns)}")
            resolved[canon] = column_map[canon]
            continue
        for a in aliases:
            if a in by_norm:
                resolved[canon] = by_norm[a]
                break
    log.info("Metadata column mapping: %s", resolved)

    missing = [c for c in ("malignancy", *METADATA_FEATURES) if c not in resolved]
    if missing:
        raise KeyError(
            f"Could not find metadata columns for {missing} in {list(raw.columns)}. "
            "Use --inspect, then --column-map '{\"malignancy\": \"<actual name>\"}'."
        )
    if not any(k in resolved for k in ("series_uid", "patient_id")):
        raise KeyError("Need at least one of SeriesInstanceUID / PatientID columns for grouped splitting.")

    out = pd.DataFrame(index=raw.index)
    for canon, col in resolved.items():
        if canon in ("malignancy", *METADATA_FEATURES):
            out[canon] = raw[col].map(_cell_mean)
        else:
            out[canon] = raw[col].astype(str).str.strip()
    for canon in ("volume_file", "nodule_id", "series_uid", "patient_id"):
        if canon not in out:
            out[canon] = np.nan
    if out["nodule_id"].isna().all():
        out["nodule_id"] = (out["volume_file"].map(lambda s: nifti_stem(s) if isinstance(s, str) else s)
                            if out["volume_file"].notna().any()
                            else pd.Series(raw.index.astype(str), index=raw.index))
    return out.reset_index(drop=True)


def match_volumes(meta: pd.DataFrame, niftis: list[Path], allow_index_order: bool) -> pd.DataFrame:
    """Attach `volume_path` to each metadata row; rows without a volume are dropped (logged)."""
    lookup: dict[str, Path] = {}
    for p in niftis:
        stem = nifti_stem(p.name)
        lookup.setdefault(stem, p)
        if stem.isdigit():
            lookup.setdefault(str(int(stem)), p)

    def find(row: pd.Series) -> Optional[Path]:
        for key in (row["volume_file"], row["nodule_id"]):
            if isinstance(key, str) and key != "nan":
                k = nifti_stem(key)
                if k in lookup:
                    return lookup[k]
                if k.isdigit() and str(int(k)) in lookup:
                    return lookup[str(int(k))]
        return None

    meta = meta.copy()
    meta["volume_path"] = meta.apply(find, axis=1)
    if meta["volume_path"].notna().sum() == 0:
        if allow_index_order and len(meta) == len(niftis):
            log.warning("No filename matches; pairing metadata rows with sorted NIfTI files BY ORDER.")
            meta["volume_path"] = niftis
        else:
            raise RuntimeError(
                "Could not match any metadata row to a NIfTI file by filename/nodule id.\n"
                f"  example metadata keys: {meta[['volume_file', 'nodule_id']].head(3).values.tolist()}\n"
                f"  example NIfTI stems : {[nifti_stem(p.name) for p in niftis[:3]]}\n"
                "Fix with --column-map volume_file=<col>, or (only if you know the order is aligned) "
                "--allow-index-order."
            )
    return meta


def assign_labels(meta: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Label 0: mean<3 (benign), label 1: mean>3 (malignant); drop ==3.0 and missing."""
    dropped: dict[str, int] = {}
    n0 = len(meta)
    meta = meta[meta["malignancy"].notna()]
    dropped["missing_malignancy"] = n0 - len(meta)
    indet = np.isclose(meta["malignancy"], INDETERMINATE_SCORE, atol=1e-6)
    dropped["indeterminate_score_3.0"] = int(indet.sum())
    meta = meta[~indet].copy()
    meta["label"] = (meta["malignancy"] > INDETERMINATE_SCORE).astype(int)
    return meta, dropped


# --------------------------------------------------------------------------- #
# 2. Volumetric preprocessing
# --------------------------------------------------------------------------- #
def load_volume_dhw(path: Path, canonical: bool) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load NIfTI -> float32 array ordered (Depth, Height, Width) + spacing (d,h,w) in mm.

    NIfTI arrays are indexed (x, y, z); z is the craniocaudal (depth) axis, so
    (x,y,z) -> (z,y,x) = (D,H,W), e.g. (128,128,64) -> (64,128,128). Optional
    RAS+ reorientation (`canonical`) may flip/permute axes and is off by default.
    """
    img = nib.load(str(path))
    if canonical:
        img = nib.as_closest_canonical(img)
    data = np.squeeze(img.get_fdata(dtype=np.float32))
    if data.ndim != 3:
        raise ValueError(f"expected 3-D volume, got shape {data.shape}")
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    return np.ascontiguousarray(data.transpose(2, 1, 0)), (zooms[2], zooms[1], zooms[0])


def center_crop_pad(vol: np.ndarray, size: int = ROI_SIZE, pad_value: float = PAD_HU) -> np.ndarray:
    """Center-crop to size^3, padding with `pad_value` where the volume is too small."""
    pads = []
    for n in vol.shape:
        deficit = max(size - n, 0)
        before = deficit // 2
        pads.append((before, deficit - before))
    if any(p != (0, 0) for p in pads):
        vol = np.pad(vol, pads, mode="constant", constant_values=pad_value)
    starts = [(n - size) // 2 for n in vol.shape]
    sl = tuple(slice(s, s + size) for s in starts)
    return vol[sl]


def lung_window(hu: np.ndarray) -> np.ndarray:
    """Clip to [-1000, 400] HU and min-max scale to float32 [0, 1]."""
    clipped = np.clip(hu, HU_MIN, HU_MAX)
    return ((clipped - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


# --------------------------------------------------------------------------- #
# 3. Radiomics (15 features)
# --------------------------------------------------------------------------- #
def extract_radiomics(hu_crop: np.ndarray) -> dict[str, float]:
    """12 image-derived features from the raw-HU 32^3 crop (padding voxels included).

    Skewness / kurtosis: scipy defaults (biased skew, Fisher excess kurtosis);
    defined as 0.0 for a constant crop. Fractions are percentages in [0, 100].
    """
    x = hu_crop.astype(np.float64).ravel()
    p25, med, p75 = np.percentile(x, [25, 50, 75])
    std = x.std()
    constant = std < 1e-9
    return {
        "mean_hu": float(x.mean()),
        "std_hu": float(std),
        "min_hu": float(x.min()),
        "max_hu": float(x.max()),
        "median_hu": float(med),
        "p25_hu": float(p25),
        "p75_hu": float(p75),
        "iqr_hu": float(p75 - p25),
        "skewness": 0.0 if constant else float(sps.skew(x)),
        "kurtosis": 0.0 if constant else float(sps.kurtosis(x)),
        "solid_fraction_pct": float(100.0 * np.mean(x > SOLID_HU)),
        "hyperdense_fraction_pct": float(100.0 * np.mean(x > HYPERDENSE_HU)),
    }


def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))


def build_dataset(cfg: Config) -> tuple[pd.DataFrame, dict[str, int]]:
    niftis, meta_path = resolve_raw_inputs(cfg)
    meta = canonicalize_metadata(load_metadata_raw(meta_path, cfg.column_names), cfg.column_map)
    log.info("Metadata rows: %d", len(meta))

    meta, dropped = assign_labels(meta)
    meta = match_volumes(meta, niftis, cfg.allow_index_order)
    dropped["no_matching_volume"] = int(meta["volume_path"].isna().sum())
    meta = meta[meta["volume_path"].notna()].copy()

    # grouping key: PatientID is the stricter unit (a patient may own several series)
    if meta["patient_id"].notna().any():
        meta["group_id"] = meta["patient_id"]
    elif meta["series_uid"].notna().any():
        meta["group_id"] = meta["series_uid"]
    else:  # last resort: LIDC-IDRI-#### embedded in the id
        found = meta["nodule_id"].str.extract(r"(LIDC-IDRI-\d{4})")[0]
        if found.isna().any():
            raise KeyError("No PatientID/SeriesInstanceUID available for grouping.")
        meta["patient_id"] = found
        meta["group_id"] = found
    log.info("Grouping split by: %s (%d groups)",
             "PatientID" if meta["patient_id"].notna().any() else "SeriesInstanceUID",
             meta["group_id"].nunique())

    meta["patch_name"] = meta["nodule_id"].map(safe_filename)
    if meta["patch_name"].duplicated().any():
        raise ValueError("Duplicate nodule ids -> patch filenames would collide.")

    # features that need all consensus metadata present
    n = len(meta)
    meta = meta.dropna(subset=METADATA_FEATURES)
    dropped["missing_consensus_metadata"] = n - len(meta)

    cfg.patch_dir.mkdir(parents=True, exist_ok=True)
    rows, suspicious_hu = [], 0
    for rec in tqdm(meta.to_dict("records"), desc="Preprocessing", unit="nodule"):
        try:
            vol, spacing = load_volume_dhw(Path(rec["volume_path"]), cfg.canonical)
            if vol.min() > -500:
                suspicious_hu += 1
            hu_crop = center_crop_pad(vol)
            if vol.shape == (64, 128, 128):  # documented LUNA22 geometry: crop == [16:48, 48:80, 48:80]
                assert np.array_equal(hu_crop, vol[16:48, 48:80, 48:80]), "unexpected crop window"
            patch = lung_window(hu_crop)
            patch_file = cfg.patch_dir / f"{rec['patch_name']}.npy"
            np.save(patch_file, patch)
            feats = extract_radiomics(hu_crop)
        except Exception as exc:  # keep going; report at the end
            log.error("Failed on %s: %s", rec["nodule_id"], exc)
            dropped["processing_error"] = dropped.get("processing_error", 0) + 1
            continue
        feats.update({k: float(rec[k]) for k in METADATA_FEATURES})
        rows.append({
            "nodule_id": rec["nodule_id"],
            "patient_id": rec["patient_id"],
            "series_uid": rec["series_uid"],
            "group_id": rec["group_id"],
            "patch_path": patch_file.relative_to(cfg.data_dir.parent).as_posix()
            if cfg.data_dir.parent in patch_file.parents else patch_file.as_posix(),
            "source_volume": Path(rec["volume_path"]).name,
            "orig_shape_dhw": "x".join(map(str, vol.shape)),
            "spacing_dhw_mm": "x".join(f"{s:.3f}" for s in spacing),
            "malignancy_mean": rec["malignancy"],
            "label": int(rec["label"]),
            **feats,
        })
    if suspicious_hu:
        log.warning("%d volume(s) have min > -500: values may not be in Hounsfield units "
                    "(e.g. offset/uint16 storage). Check the NIfTI intensity scale.", suspicious_hu)
    if not rows:
        raise RuntimeError("No nodules survived preprocessing.")
    df = pd.DataFrame(rows)

    bad = ~np.isfinite(df[FEATURE_COLUMNS].to_numpy(dtype=float)).all(axis=1)
    if bad.any():
        dropped["non_finite_features"] = int(bad.sum())
        df = df[~bad].reset_index(drop=True)
    return df, dropped


# --------------------------------------------------------------------------- #
# 4. Leakage-free splitting
# --------------------------------------------------------------------------- #
def _grouped_split_once(df: pd.DataFrame, seed: int, val: float, test: float) -> pd.Series:
    """GroupShuffleSplit twice: train | (val+test), then val | test. Returns split labels."""
    g1 = GroupShuffleSplit(n_splits=1, test_size=val + test, random_state=seed)
    tr_idx, rest_idx = next(g1.split(df, groups=df["group_id"]))
    rest = df.iloc[rest_idx]
    g2 = GroupShuffleSplit(n_splits=1, test_size=test / (val + test), random_state=seed)
    v_rel, t_rel = next(g2.split(rest, groups=rest["group_id"]))
    split = pd.Series("", index=df.index, dtype=object)
    split.iloc[tr_idx] = "train"
    split.iloc[rest_idx[v_rel]] = "val"
    split.iloc[rest_idx[t_rel]] = "test"
    return split


def grouped_split(df: pd.DataFrame, seed: int, tries: int = 1,
                  fractions: tuple[float, float, float] = (0.70, 0.15, 0.15)) -> pd.Series:
    """Patient-disjoint split. GroupShuffleSplit cannot stratify, so with tries>1 several
    seeds are drawn and the one closest to the target sizes and class prevalence is kept
    (still leakage-free: every candidate is group-disjoint)."""
    _, val, test = fractions
    best, best_cost = None, np.inf
    overall = df["label"].mean()
    for s in range(seed, seed + max(tries, 1)):
        split = _grouped_split_once(df, s, val, test)
        cost = 0.0
        for name, frac in zip(("train", "val", "test"), fractions):
            sub = df[split == name]
            cost += abs(len(sub) / len(df) - frac) + abs(sub["label"].mean() - overall)
        if cost < best_cost:
            best, best_cost, used = split, cost, s
    log.info("Split seed used: %d (tries=%d, cost=%.4f)", used, tries, best_cost)
    return best


def class_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in ("train", "val", "test"):
        sub = df[df["split"] == name]
        n = len(sub)
        mal = int(sub["label"].sum())
        rows.append({"split": name, "n_nodules": n, "benign(0)": n - mal, "malignant(1)": mal,
                     "malignant_%": round(100 * mal / n, 2) if n else float("nan"),
                     "n_groups": sub["group_id"].nunique(),
                     "share_%": round(100 * n / len(df), 2)})
    tot = len(df)
    mal = int(df["label"].sum())
    rows.append({"split": "TOTAL", "n_nodules": tot, "benign(0)": tot - mal, "malignant(1)": mal,
                 "malignant_%": round(100 * mal / tot, 2), "n_groups": df["group_id"].nunique(),
                 "share_%": 100.0})
    return pd.DataFrame(rows)


def save_outputs(df: pd.DataFrame, dropped: dict[str, int], cfg: Config) -> None:
    out = cfg.processed_dir
    out.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        df[df["split"] == name].to_csv(out / f"{name}.csv", index=False)
    df.to_csv(out / "metadata.csv", index=False)

    summary = {
        "config": {"seed": cfg.seed, "split_tries": cfg.split_tries, "roi_size": ROI_SIZE,
                   "hu_window": [HU_MIN, HU_MAX], "pad_hu": PAD_HU, "canonical_orientation": cfg.canonical,
                   "grouping_column": "patient_id" if df["patient_id"].notna().any() else "series_uid",
                   "feature_columns": FEATURE_COLUMNS},
        "total_samples": int(len(df)),
        "class_counts": {"benign_0": int((df["label"] == 0).sum()),
                         "malignant_1": int((df["label"] == 1).sum())},
        "dropped": dropped,
        "splits": {},
    }
    for row in class_table(df).to_dict("records"):
        if row["split"] == "TOTAL":
            continue
        summary["splits"][row["split"]] = {
            "n_samples": row["n_nodules"], "benign_0": row["benign(0)"],
            "malignant_1": row["malignant(1)"], "malignant_pct": row["malignant_%"],
            "n_groups": row["n_groups"], "share_pct": row["share_%"]}
    with open(out / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved manifests + summary to %s", out)


# --------------------------------------------------------------------------- #
# 5. Verification self-check (reads back what was written to disk)
# --------------------------------------------------------------------------- #
def run_verification(cfg: Config) -> bool:
    out = cfg.processed_dir
    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)

    parts = {n: pd.read_csv(out / f"{n}.csv") for n in ("train", "val", "test")}
    meta = pd.read_csv(out / "metadata.csv")
    for n, d in parts.items():
        d["split"] = n
    all_df = pd.concat(parts.values(), ignore_index=True)

    # patches: exist, exact shape, float32, finite, in [0,1]
    bad_missing = bad_shape = bad_range = 0
    root = cfg.data_dir.parent
    for p in tqdm(all_df["patch_path"], desc="Verifying patches", unit="patch"):
        path = Path(p) if Path(p).is_absolute() else root / p
        if not path.exists():
            bad_missing += 1
            continue
        a = np.load(path, mmap_mode="r")
        if a.shape != (ROI_SIZE,) * 3 or a.dtype != np.float32:
            bad_shape += 1
        elif not np.isfinite(a).all() or a.min() < 0.0 or a.max() > 1.0:
            bad_range += 1
    check(bad_missing == 0, f"{bad_missing} patch file(s) missing on disk")
    check(bad_shape == 0, f"{bad_shape} patch(es) not float32 with shape (32,32,32)")
    check(bad_range == 0, f"{bad_range} patch(es) contain NaN/inf or values outside [0,1]")

    # radiomics finite
    feats = all_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    check(len(FEATURE_COLUMNS) == 15, "feature vector is not 15-dimensional")
    check(np.isfinite(feats).all(), f"{int((~np.isfinite(feats)).sum())} NaN/inf radiomic value(s)")

    # disjointness (pairwise, which is strictly stronger than the triple intersection)
    for col in ("group_id", "patient_id", "series_uid"):
        if all_df[col].isna().all():
            continue
        s = {n: set(d[col].dropna().astype(str)) for n, d in parts.items()}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            check(not (s[a] & s[b]), f"{col} overlap between {a} and {b}: {len(s[a] & s[b])}")
        check(not (s["train"] & s["val"] & s["test"]), f"{col} triple intersection not empty")

    # bookkeeping
    check(len(all_df) == len(meta), "split CSV rows != metadata.csv rows")
    check(not all_df["nodule_id"].duplicated().any(), "nodule appears in more than one split / twice")
    check(set(all_df["label"].unique()) <= {0, 1}, "labels outside {0,1}")
    check(not np.isclose(all_df["malignancy_mean"], INDETERMINATE_SCORE, atol=1e-6).any(),
          "indeterminate (score == 3.0) nodules present")

    print("\n" + "=" * 78)
    print("CLASS DISTRIBUTION")
    print("=" * 78)
    print(class_table(all_df).to_string(index=False))
    print("=" * 78)
    if failures:
        print("VERIFICATION FAILED:")
        for f in failures:
            print("  [x]", f)
        return False
    print("VERIFICATION PASSED: patches OK, features finite, splits group-disjoint.")
    return True


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data",
                   help="Root for raw/ and processed/ (default: <script dir>/data)")
    p.add_argument("--raw-dir", type=Path, help="Folder with NIfTI volumes (searched recursively)")
    p.add_argument("--meta-path", type=Path, help="Metadata .npy (or .csv)")
    p.add_argument("--no-download", action="store_true", help="Fail instead of downloading")
    p.add_argument("--keep-archive", action="store_true", help="Keep the downloaded zip")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-tries", type=int, default=1,
                   help="Candidate seeds for GroupShuffleSplit; best size/class balance wins (default 1)")
    p.add_argument("--canonical", action="store_true",
                   help="Reorient to RAS+ with nibabel before transposing (default: off; raw "
                        "(x,y,z) array axes are transposed to (z,y,x), matching LUNA22 geometry)")
    p.add_argument("--allow-index-order", action="store_true",
                   help="If filenames don't match, pair metadata rows with sorted NIfTI files by order")
    p.add_argument("--column-map", type=json.loads, default={},
                   help='JSON canonical->actual, e.g. \'{"malignancy": "mal_mean"}\'. '
                        f"Canonical keys: {list(COLUMN_ALIASES)}")
    p.add_argument("--column-names", type=lambda s: s.split(","), default=None,
                   help="Names for a bare 2-D metadata array")
    p.add_argument("--inspect", action="store_true", help="Print metadata schema and exit")
    p.add_argument("--verify-only", action="store_true", help="Only run the self-check on existing outputs")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = Config(
        data_dir=args.data_dir.resolve(), raw_dir=args.raw_dir, meta_path=args.meta_path,
        no_download=args.no_download, keep_archive=args.keep_archive, seed=args.seed,
        split_tries=args.split_tries, canonical=args.canonical,
        allow_index_order=args.allow_index_order, column_map=args.column_map,
        column_names=args.column_names,
    )

    if args.verify_only:
        return 0 if run_verification(cfg) else 1

    if args.inspect:
        niftis, meta_path = resolve_raw_inputs(cfg)
        raw = load_metadata_raw(meta_path, cfg.column_names)
        print(f"NIfTI files: {len(niftis)}  e.g. {[p.name for p in niftis[:3]]}")
        print(f"Metadata: {meta_path}  shape={raw.shape}")
        print(raw.dtypes.to_string())
        print(raw.head(5).to_string())
        return 0

    df, dropped = build_dataset(cfg)
    df["split"] = grouped_split(df, cfg.seed, cfg.split_tries)
    save_outputs(df, dropped, cfg)
    log.info("Dropped: %s", dropped)
    return 0 if run_verification(cfg) else 1


if __name__ == "__main__":
    sys.exit(main())
