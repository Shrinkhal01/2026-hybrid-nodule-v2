"""Tests for Phase 7 step P: patient IDs from pylidc + the patient-level leakage check.

Run (CPU only, no pylidc needed, ~1 s):
    python -m pytest tests/test_patient_ids.py -v
"""
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("data_pipeline", ROOT / "01_data_pipeline.py")
dp = importlib.util.module_from_spec(_spec)
sys.modules["data_pipeline"] = dp
_spec.loader.exec_module(dp)


def make_db(path: Path, rows):
    """Tiny stand-in for pylidc.sqlite: table `scans` with (series_instance_uid, patient_id)."""
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE scans (id INTEGER PRIMARY KEY, series_instance_uid TEXT, patient_id TEXT)")
    con.executemany("INSERT INTO scans (series_instance_uid, patient_id) VALUES (?, ?)", rows)
    con.commit()
    con.close()
    return path


@pytest.fixture
def db(tmp_path):
    # patient 0001 has TWO scans (S1, S2): the case a series-level split cannot see
    return make_db(tmp_path / "pylidc.sqlite",
                   [("S1", "LIDC-IDRI-0001"), ("S2", "LIDC-IDRI-0001"), ("S3", "LIDC-IDRI-0002")])


def test_series_to_patient_reads_the_scans_table(db):
    assert dp.series_to_patient(db) == {"S1": "LIDC-IDRI-0001", "S2": "LIDC-IDRI-0001",
                                        "S3": "LIDC-IDRI-0002"}


def test_attach_fills_patient_id(db):
    meta = pd.DataFrame({"series_uid": ["S1", "S2", "S3"], "patient_id": [None] * 3})
    out = dp.attach_patient_ids(meta, dp.series_to_patient(db))
    assert out["patient_id"].tolist() == ["LIDC-IDRI-0001", "LIDC-IDRI-0001", "LIDC-IDRI-0002"]


def test_unknown_series_is_an_error_not_a_gap(db):
    meta = pd.DataFrame({"series_uid": ["S1", "S9"], "patient_id": [None, None]})
    with pytest.raises(KeyError, match="1 series not in the pylidc database"):
        dp.attach_patient_ids(meta, dp.series_to_patient(db))


def test_conflicting_metadata_patient_id_is_an_error(db):
    meta = pd.DataFrame({"series_uid": ["S1"], "patient_id": ["LIDC-IDRI-0999"]})
    with pytest.raises(ValueError, match="disagrees with pylidc"):
        dp.attach_patient_ids(meta, dp.series_to_patient(db))


# --------------------------------------------------------------------------- #
# The leakage check itself: run_verification() on a tiny processed/ folder
# --------------------------------------------------------------------------- #
def write_processed(data_dir: Path, rows, patient_source="pylidc test"):
    out = data_dir / "processed"
    (out / "patches").mkdir(parents=True)
    records = []
    for i, (series, patient, split) in enumerate(rows):
        patch = out / "patches" / f"n{i}.npy"
        np.save(patch, np.full((32, 32, 32), 0.5, np.float32))
        rec = {"nodule_id": f"n{i}", "patient_id": patient, "series_uid": series,
               "group_id": patient or series, "patch_path": str(patch),
               "malignancy_mean": 4.0 if i % 2 else 2.0, "label": i % 2, "split": split,
               "pad_fraction": 0.0}
        rec.update({c: 1.0 for c in dp.FEATURE_COLUMNS})
        records.append(rec)
    df = pd.DataFrame(records)
    for name in ("train", "val", "test"):
        df[df["split"] == name].to_csv(out / f"{name}.csv", index=False)
    df.to_csv(out / "metadata.csv", index=False)
    (out / "dataset_summary.json").write_text(json.dumps(
        {"config": {"roi_size": 32, "patient_id_source": patient_source}}))


def verify(data_dir: Path) -> bool:
    cfg = SimpleNamespace(data_dir=data_dir, processed_dir=data_dir / "processed", roi_size=32)
    return dp.run_verification(cfg)


def test_clean_patient_split_passes(tmp_path):
    write_processed(tmp_path, [("S1", "P1", "train"), ("S2", "P1", "train"),
                               ("S3", "P2", "val"), ("S4", "P3", "test")])
    assert verify(tmp_path)


def test_same_patient_in_train_and_test_fails(tmp_path):
    """BREAK-IT: two DIFFERENT scans of ONE patient in train and test. Every series is in
    one split only, so a series-level check passes; only the patient check catches it."""
    write_processed(tmp_path, [("S1", "P1", "train"), ("S2", "P1", "test"),
                               ("S3", "P2", "val"), ("S4", "P3", "test")])
    assert not verify(tmp_path)


def test_missing_patient_id_fails_when_source_recorded(tmp_path):
    write_processed(tmp_path, [("S1", "P1", "train"), ("S2", None, "val"), ("S3", "P3", "test")])
    assert not verify(tmp_path)
