#!/usr/bin/env python3
"""
pipelines/02_train.py -- Day 2: train and benchmark three models on the identical splits.

    Model A  Balanced Random Forest            (15 tabular features)
    Model B  Pure 3D ResNet-18                 (32^3 CT patch only)
    Model C  Dual-branch Hybrid Net            (patch + 15 tabular features, gated fusion)

Protocol (no test-set leakage)
  * The tabular scaler (deep models) is fitted on train.csv only.
  * Deep models: BCEWithLogitsLoss(pos_weight = n_benign / n_malignant from train.csv), AdamW
    (lr 1e-4, wd 1e-4), ReduceLROnPlateau(mode='max', factor 0.5, patience 5) on validation ROC-AUC,
    40 epochs. The checkpoint with the best validation ROC-AUC is kept.
  * The decision threshold of every model is chosen by Youden's J on the *validation* set.
  * test.csv is touched once per model, after model selection: ROC-AUC (threshold-free, with a
    bootstrap 95% CI), plus accuracy / sensitivity / specificity / precision / F1 / false negatives at
    the validation-derived threshold.

Outputs (default ``results/``)
  best_hybrid_net.pth, best_pure_resnet.pth, baseline_rf.joblib, model_comparison_metrics.csv,
  test_predictions.csv, training_history_{pure_resnet,hybrid_net}.csv, run_config.json

Checkpoints are dicts: ``{"state_dict", "epoch", "val_auc", "feature_cols", "scaler", "model"}``.

Usage
    python pipelines/02_train.py                      # full run (CUDA if available)
    python pipelines/02_train.py --epochs 2 --limit 64 --results-dir /tmp/smoke   # quick smoke test
    python pipelines/02_train.py --features image --seed 43 --results-dir results/x  # Phase 7 ablation:
        RF + Hybrid get only the 12 image-derived features (no radiologist ratings)
    (CUDA is required by default; add --allow-cpu only for debugging without a GPU)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.baseline_rf import BaselineRF  # noqa: E402
from models.hybrid_net import HybridNoduleNet, Pure3DResNet, count_parameters  # noqa: E402
from pipelines.common import (  # noqa: E402
    FEATURE_COLS, FEATURE_SETS, PROCESSED_DIR, RESULTS_DIR, NoduleDataset, TabularScaler, assert_group_disjoint,
    bootstrap_auc_ci, check_feature_contract, classification_metrics, get_device, load_split,
    make_loader, predict_probs, safe_auc, set_seed, youden_threshold,
)

log = logging.getLogger("day2.train")

MODEL_LABELS = {
    "rf": "A: Balanced Random Forest (tabular)",
    "resnet": "B: Pure 3D ResNet-18 (image only)",
    "hybrid": "C: Hybrid Net (image + radiomics)",
}


# --------------------------------------------------------------------------- #
# Deep-model training
# --------------------------------------------------------------------------- #
def train_deep_model(
    name: str,
    model: nn.Module,
    loaders: Dict[str, Any],
    y_val: np.ndarray,
    pos_weight: float,
    device: torch.device,
    args: argparse.Namespace,
    ckpt_path: Path,
    scaler: TabularScaler,
    results_dir: Path,
    feature_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Train ``model``; save the best-validation-AUC checkpoint to ``ckpt_path``.

    Returns ``{"best_epoch", "best_val_auc"}`` and writes ``training_history_{name}.csv``.
    """
    use_amp = device.type == "cuda" and not args.no_amp
    model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    log.info("[%s] params=%s pos_weight=%.3f amp=%s epochs=%d",
             name, f"{count_parameters(model):,}", pos_weight, use_amp, args.epochs)

    best_auc, best_epoch, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running, seen = 0.0, 0
        for batch in loaders["train"]:
            vol = batch["volume"].to(device, non_blocking=True)
            tab = batch["tabular"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(vol, tab)
            loss = criterion(logits.float().squeeze(1), target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"[{name}] non-finite loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            running += loss.item() * len(target)
            seen += len(target)

        val_probs = predict_probs(model, loaders["val"], device, use_amp)
        val_auc = safe_auc(y_val, val_probs)
        scheduler.step(0.0 if np.isnan(val_auc) else val_auc)
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": running / max(seen, 1), "val_auc": val_auc, "lr": lr})

        improved = val_auc > best_auc
        if improved:
            best_auc, best_epoch = val_auc, epoch
            torch.save({
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "epoch": epoch, "val_auc": val_auc, "feature_cols": feature_cols or FEATURE_COLS,
                "scaler": scaler.to_dict(), "model": name,
            }, ckpt_path)
        log.info("[%s] epoch %02d/%d loss=%.4f val_auc=%.4f lr=%.2e %s (%.1fs)", name, epoch,
                 args.epochs, history[-1]["train_loss"], val_auc, lr, "*saved*" if improved else "",
                 time.time() - t0)

    pd.DataFrame(history).to_csv(results_dir / f"training_history_{name}.csv", index=False)
    return {"best_epoch": best_epoch, "best_val_auc": best_auc}


def load_best(model: nn.Module, ckpt_path: Path, device: torch.device) -> nn.Module:
    """Load the saved best-validation weights into ``model``."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_model(key: str, y_val: np.ndarray, p_val: np.ndarray, y_test: np.ndarray,
                   p_test: np.ndarray, n_boot: int, seed: int) -> Dict[str, Any]:
    """Youden threshold from validation -> metrics on test (+ bootstrap AUC CI)."""
    thr = youden_threshold(y_val, p_val)
    m = classification_metrics(y_test, p_test, thr)
    lo, hi = bootstrap_auc_ci(y_test, p_test, n_boot=n_boot, seed=seed)
    return {"model": MODEL_LABELS[key], "val_auc": safe_auc(y_val, p_val), **m,
            "auc_ci95_low": lo, "auc_ci95_high": hi, "n_test": int(len(y_test))}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--models", nargs="+", choices=list(MODEL_LABELS), default=list(MODEL_LABELS),
                   help="subset of models to run (default: all three)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32, help="32 or 64 (default 32: ~22 steps/epoch on 703 samples)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--features", choices=list(FEATURE_SETS), default="all",
                   help="tabular inputs for RF + Hybrid: 'all' = 15 (incl. 3 radiologist ratings), "
                        "'image' = 12 measured from the CT patch only (Phase 7 ablation)")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (pin_memory is on for CUDA)")
    p.add_argument("--pretrained", action="store_true", help="init the 3D ResNet from Kinetics-400 (downloads)")
    p.add_argument("--no-amp", action="store_true", help="disable bf16 autocast on CUDA")
    p.add_argument("--allow-cpu", action="store_true",
                   help="permit running without CUDA (debug only; by default a CUDA GPU is required)")
    p.add_argument("--limit", type=int, default=None, help="DEBUG: subsample each split to ~N rows")
    p.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples for the test AUC CI")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(require_cuda=not args.allow_cpu)  # asserts CUDA unless --allow-cpu
    set_seed(args.seed)

    # ---- data -------------------------------------------------------------
    check_feature_contract(args.processed_dir)
    train_df = load_split("train", args.processed_dir, args.limit, args.seed)
    val_df = load_split("val", args.processed_dir, args.limit, args.seed)
    test_df = load_split("test", args.processed_dir, args.limit, args.seed)
    assert_group_disjoint(train_df, val_df, test_df)
    log.info("Split sizes: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))
    if args.limit is None and len(test_df) != 151:
        log.warning("test.csv has %d rows (expected 151 from the current Day 1 split).", len(test_df))

    y_train, y_val, y_test = (d["label"].to_numpy(int) for d in (train_df, val_df, test_df))
    n_pos, n_neg = int(y_train.sum()), int((y_train == 0).sum())
    pos_weight = n_neg / max(n_pos, 1)

    cols = FEATURE_SETS[args.features]  # Phase 7 ablation: which tabular inputs RF + Hybrid see
    log.info("Feature set '%s': %d tabular features", args.features, len(cols))
    scaler = TabularScaler().fit(train_df[cols].to_numpy())
    (args.results_dir / "tabular_scaler.json").write_text(json.dumps(scaler.to_dict(), indent=2))

    rows: List[Dict[str, Any]] = []
    test_preds = pd.DataFrame({"nodule_id": test_df["nodule_id"], "label": y_test})
    run_info: Dict[str, Any] = {}

    # ---- Model A: Random Forest ------------------------------------------
    if "rf" in args.models:
        rf = BaselineRF(cols, args.seed).fit(train_df[cols].to_numpy(), y_train)
        p_val = rf.predict_proba(val_df[cols].to_numpy())
        p_test = rf.predict_proba(test_df[cols].to_numpy())
        rf.save(args.results_dir / "baseline_rf.joblib")
        rows.append(evaluate_model("rf", y_val, p_val, y_test, p_test, args.n_boot, args.seed))
        test_preds["prob_rf"] = p_test
        run_info["rf_feature_importances"] = rf.feature_importances()
        log.info("[rf] val_auc=%.4f test_auc=%.4f", rows[-1]["val_auc"], rows[-1]["roc_auc"])

    # ---- Deep models -------------------------------------------------------
    deep = [k for k in ("resnet", "hybrid") if k in args.models]
    if deep:
        pin = device.type == "cuda"
        ds = {
            "train": NoduleDataset(train_df, scaler, augment=True, feature_cols=cols),
            "val": NoduleDataset(val_df, scaler, feature_cols=cols),
            "test": NoduleDataset(test_df, scaler, feature_cols=cols),
        }
        loaders = {
            "train": make_loader(ds["train"], args.batch_size, True, args.num_workers, pin),
            "val": make_loader(ds["val"], args.batch_size, False, args.num_workers, pin),
            "test": make_loader(ds["test"], args.batch_size, False, args.num_workers, pin),
        }
        specs = {
            "resnet": ("pure_resnet", lambda: Pure3DResNet(args.pretrained), "best_pure_resnet.pth"),
            "hybrid": ("hybrid_net", lambda: HybridNoduleNet(n_tabular=len(cols), pretrained=args.pretrained), "best_hybrid_net.pth"),
        }
        for key in deep:
            tag, factory, ckpt_name = specs[key]
            set_seed(args.seed)  # identical RNG state at the start of each deep model
            ckpt_path = args.results_dir / ckpt_name
            info = train_deep_model(tag, factory(), loaders, y_val, pos_weight, device, args,
                                    ckpt_path, scaler, args.results_dir, cols)
            model = load_best(factory(), ckpt_path, device)
            use_amp = device.type == "cuda" and not args.no_amp
            p_val = predict_probs(model, loaders["val"], device, use_amp)
            p_test = predict_probs(model, loaders["test"], device, use_amp)
            rows.append(evaluate_model(key, y_val, p_val, y_test, p_test, args.n_boot, args.seed))
            test_preds[f"prob_{key}"] = p_test
            run_info[tag] = info
            log.info("[%s] best epoch %d val_auc=%.4f | test_auc=%.4f", tag, info["best_epoch"],
                     info["best_val_auc"], rows[-1]["roc_auc"])

    # ---- Reporting ----------------------------------------------------------
    table = pd.DataFrame(rows)
    table.to_csv(args.results_dir / "model_comparison_metrics.csv", index=False)
    test_preds.to_csv(args.results_dir / "test_predictions.csv", index=False)
    run_info.update({
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "device": str(device), "torch": torch.__version__, "pos_weight": pos_weight,
        "feature_set": args.features, "feature_cols": cols,
        "thresholds": {r["model"]: r["threshold"] for r in rows},
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
    })
    (args.results_dir / "run_config.json").write_text(json.dumps(run_info, indent=2, default=str))

    show = ["model", "roc_auc", "auc_ci95_low", "auc_ci95_high", "accuracy", "sensitivity",
            "specificity", "precision", "f1", "false_negatives", "threshold"]
    print("\n" + "=" * 110 + f"\nTEST-SET RESULTS (n={len(test_df)}; thresholds from validation Youden's J)\n" + "=" * 110)
    print(table[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("=" * 110)
    print(f"Saved to {args.results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
