#!/usr/bin/env python3
"""
pipelines/04_report_figures.py -- publication-style figures + paired significance tests.

Reads the artefacts produced by ``02_train.py`` / ``03_evaluate_xai.py`` (no GPU or model weights
needed) and writes to ``results/figures/``:

    fig_roc_curves.png            test ROC for all three models (+ validation-Youden operating points)
    fig_confusion_matrices.png    test confusion matrices at the validation-derived thresholds
    fig_metrics_comparison.png    accuracy / sensitivity / specificity / precision / F1 per model
    fig_auc_differences.png       paired-bootstrap AUC differences with 95% CI (is C really better?)
    fig_training_curves.png       train loss and validation AUC per epoch, best epoch marked
    fig_rf_feature_importance.png Random Forest importances (image-derived vs radiologist-rated features)
    fig_probability_distributions.png  predicted P(malignant) by true class, with thresholds
    fig_dataset_splits.png        class balance of train / val / test
    fig_cam_localization.png      share of Grad-CAM mass inside the central 16^3 cube vs a uniform map
    metrics_table.md              markdown results table (paste into README)
    significance_tests.csv        paired-bootstrap AUC tests + McNemar tests on errors

Usage:  python pipelines/04_report_figures.py [--results-dir results] [--n-boot 5000]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("day3.figures")

# Fixed entity -> colour mapping (reference categorical palette, slots 1-3; colour follows the model).
MODELS: List[Tuple[str, str, str]] = [  # key, short label, colour
    ("rf", "A: Random Forest", "#1baf7a"),
    ("resnet", "B: 3D ResNet-18", "#eb6834"),
    ("hybrid", "C: Hybrid Net", "#2a78d6"),
]
COLOR = {k: c for k, _, c in MODELS}
LABEL = {k: l for k, l, _ in MODELS}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CLASS_COLORS = {0: "#6b7280", 1: "#d1456b"}  # benign, malignant
ANNOTATOR_FEATURES = {"diameter_mm", "calcification", "texture"}


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "legend.frameon": False, "figure.dpi": 100,
    })


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_inputs(results_dir: Path, processed_dir: Path) -> Dict[str, object]:
    need = ["test_predictions.csv", "model_comparison_metrics.csv"]
    missing = [n for n in need if not (results_dir / n).exists()]
    if missing:
        raise FileNotFoundError(f"missing in {results_dir}: {missing} -- run pipelines/02_train.py first")
    preds = pd.read_csv(results_dir / "test_predictions.csv")
    table = pd.read_csv(results_dir / "model_comparison_metrics.csv")
    table["key"] = table["model"].str[0].map({"A": "rf", "B": "resnet", "C": "hybrid"})
    if table["key"].isna().any():
        raise ValueError("could not map model_comparison_metrics.csv rows to A/B/C")
    out: Dict[str, object] = {"preds": preds, "table": table.set_index("key")}
    for k in ("hybrid_net", "pure_resnet"):
        f = results_dir / f"training_history_{k}.csv"
        out[k] = pd.read_csv(f) if f.exists() else None
    cfg = results_dir / "run_config.json"
    out["run_config"] = json.loads(cfg.read_text()) if cfg.exists() else {}
    audit = results_dir / "xai_cases_audit.json"
    out["audit"] = json.loads(audit.read_text()) if audit.exists() else None
    out["splits"] = {n: pd.read_csv(processed_dir / f"{n}.csv", usecols=["label"])
                     for n in ("train", "val", "test") if (processed_dir / f"{n}.csv").exists()}
    return out


def _save(fig: plt.Figure, path: Path, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path.name)
    return path


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_roc(d: Dict, out: Path, dpi: int) -> Path:
    preds, table = d["preds"], d["table"]
    y = preds["label"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.plot([0, 1], [0, 1], ls=":", c=MUTED, lw=1)
    for k, name, c in MODELS:
        p = preds[f"prob_{k}"].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        r = table.loc[k]
        ax.plot(fpr, tpr, c=c, lw=2, label=f"{name}  AUC {r.roc_auc:.3f} [{r.auc_ci95_low:.3f}-{r.auc_ci95_high:.3f}]")
        ax.scatter([r.false_positives / (r.false_positives + r.true_negatives)], [r.sensitivity],
                   s=55, c=c, edgecolor="white", linewidth=1.5, zorder=5)
    ax.set(xlabel="1 - Specificity (false-positive rate)", ylabel="Sensitivity (true-positive rate)",
           xlim=(-0.01, 1.01), ylim=(-0.01, 1.01), aspect="equal",
           title=f"ROC on held-out test set (n={len(y)})")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.text(0.98, 0.02 + 0.30, "dots = validation-Youden thresholds", transform=ax.transAxes,
            ha="right", fontsize=8, color=MUTED)
    return _save(fig, out / "fig_roc_curves.png", dpi)


def confusion_counts(y: np.ndarray, p: np.ndarray, thr: float) -> np.ndarray:
    pred = (p >= thr).astype(int)
    return np.array([[((y == 0) & (pred == 0)).sum(), ((y == 0) & (pred == 1)).sum()],
                     [((y == 1) & (pred == 0)).sum(), ((y == 1) & (pred == 1)).sum()]])


def fig_confusion(d: Dict, out: Path, dpi: int) -> Path:
    preds, table = d["preds"], d["table"]
    y = preds["label"].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), gridspec_kw={"wspace": 0.12})
    for ax, (k, name, c) in zip(axes, MODELS):
        cm = confusion_counts(y, preds[f"prob_{k}"].to_numpy(), table.loc[k, "threshold"])
        rate = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(rate, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({rate[i, j]:.0%})", ha="center", va="center", fontsize=11,
                        color="white" if rate[i, j] > 0.55 else INK, fontweight="bold")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["benign", "malignant"],
               yticklabels=["benign", "malignant"] if k == "rf" else [], xlabel="Predicted",
               ylabel="True" if k == "rf" else "",
               title=f"{name}\nthr={table.loc[k, 'threshold']:.2f}  FN={cm[1, 0]}  FP={cm[0, 1]}")
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Confusion matrices on the test set (percent = row-normalised)", y=1.03, fontweight="bold")
    return _save(fig, out / "fig_confusion_matrices.png", dpi)


def fig_metrics(d: Dict, out: Path, dpi: int) -> Path:
    table = d["table"]
    cols = [("accuracy", "Accuracy"), ("sensitivity", "Sensitivity"), ("specificity", "Specificity"),
            ("precision", "Precision"), ("f1", "F1")]
    x = np.arange(len(cols))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for i, (k, name, c) in enumerate(MODELS):
        vals = [table.loc[k, col] for col, _ in cols]
        bars = ax.bar(x + (i - 1) * (w + 0.02), vals, w, color=c, label=name, linewidth=0)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", fontsize=7.5, color=INK)
    ax.set(xticks=x, xticklabels=[l for _, l in cols], ylim=(0, 1.08), ylabel="Score",
           title="Test-set metrics at validation-derived thresholds")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    return _save(fig, out / "fig_metrics_comparison.png", dpi)


def paired_tests(d: Dict, n_boot: int, seed: int) -> pd.DataFrame:
    """Paired bootstrap on AUC differences + exact McNemar on per-nodule correctness."""
    preds, table = d["preds"], d["table"]
    y = preds["label"].to_numpy()
    P = {k: preds[f"prob_{k}"].to_numpy() for k, _, _ in MODELS}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    aucs = {k: [] for k in P}
    keep = []
    for b in range(n_boot):
        yb = y[idx[b]]
        if yb.min() == yb.max():
            continue
        keep.append(b)
        for k in P:
            aucs[k].append(roc_auc_score(yb, P[k][idx[b]]))
    aucs = {k: np.asarray(v) for k, v in aucs.items()}
    correct = {k: ((P[k] >= table.loc[k, "threshold"]).astype(int) == y) for k in P}
    rows = []
    for a, b in (("hybrid", "rf"), ("hybrid", "resnet"), ("resnet", "rf")):
        diff = aucs[a] - aucs[b]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        p_boot = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))
        only_a = int((correct[a] & ~correct[b]).sum())
        only_b = int((~correct[a] & correct[b]).sum())
        p_mc = binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
        rows.append({
            "comparison": f"{LABEL[a]} vs {LABEL[b]}",
            "auc_a": float(table.loc[a, "roc_auc"]), "auc_b": float(table.loc[b, "roc_auc"]),
            "auc_diff": float(table.loc[a, "roc_auc"] - table.loc[b, "roc_auc"]),
            "diff_ci95_low": float(lo), "diff_ci95_high": float(hi), "p_bootstrap": float(p_boot),
            "correct_only_a": only_a, "correct_only_b": only_b, "p_mcnemar_exact": float(p_mc),
            "n_boot_valid": len(keep),
        })
    return pd.DataFrame(rows)


def fig_auc_diff(sig: pd.DataFrame, out: Path, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(7.6, 2.9 + 0.3 * len(sig)))
    ys = np.arange(len(sig))[::-1]
    for y_, r in zip(ys, sig.itertuples()):
        ax.plot([r.diff_ci95_low, r.diff_ci95_high], [y_, y_], c=COLOR["hybrid"], lw=2.5, solid_capstyle="round")
        ax.scatter([r.auc_diff], [y_], s=70, c=COLOR["hybrid"], edgecolor="white", linewidth=1.5, zorder=5)
        ax.text(0.995, y_ + 0.22, f"{r.auc_diff:+.3f}  [{r.diff_ci95_low:+.3f}, {r.diff_ci95_high:+.3f}]  p={r.p_bootstrap:.2f}",
                transform=ax.get_yaxis_transform(), ha="right", fontsize=8, color=MUTED)
    ax.axvline(0, c=INK, lw=1)
    ax.set(yticks=ys, yticklabels=sig["comparison"], xlabel="AUC difference (paired bootstrap, 95% CI)",
           ylim=(-0.6, len(sig) - 0.2), title="Paired AUC differences on the test set (95% bootstrap CI)")
    ax.grid(axis="y", visible=False)
    return _save(fig, out / "fig_auc_differences.png", dpi)


def fig_training(d: Dict, out: Path, dpi: int) -> Optional[Path]:
    hist = {"resnet": d["pure_resnet"], "hybrid": d["hybrid_net"]}
    if all(v is None for v in hist.values()):
        log.warning("no training-history CSVs; skipping training curves")
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for k in ("resnet", "hybrid"):
        h = hist[k]
        if h is None:
            continue
        best = int(h["val_auc"].idxmax())
        axes[0].plot(h["epoch"], h["train_loss"], c=COLOR[k], lw=2, label=LABEL[k])
        axes[1].plot(h["epoch"], h["val_auc"], c=COLOR[k], lw=2,
                     label=f"{LABEL[k]} (best: epoch {int(h['epoch'][best])}, AUC {h['val_auc'][best]:.3f})")
        axes[1].scatter([h["epoch"][best]], [h["val_auc"][best]], s=70, c=COLOR[k], edgecolor="white",
                        linewidth=1.5, zorder=5)
    axes[0].set(xlabel="Epoch", ylabel="Training loss (BCE, pos_weight)", title="Training loss")
    axes[1].set(xlabel="Epoch", ylabel="Validation ROC-AUC", title="Validation ROC-AUC (model selection)")
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="lower right", fontsize=8.5)
    return _save(fig, out / "fig_training_curves.png", dpi)


def fig_rf_importance(d: Dict, out: Path, dpi: int) -> Optional[Path]:
    imp = d["run_config"].get("rf_feature_importances")
    if not imp:
        log.warning("run_config.json has no rf_feature_importances; skipping")
        return None
    s = pd.Series(imp).sort_values()
    colors = ["#6250d6" if f in ANNOTATOR_FEATURES else "#8a8985" for f in s.index]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.barh(s.index, s.values, color=colors, height=0.68, linewidth=0)
    for yv, v in zip(range(len(s)), s.values):
        ax.text(v + 0.004, yv, f"{v:.3f}", va="center", fontsize=8, color=MUTED)
    ax.set(xlabel="Mean decrease in impurity", title="Random Forest feature importance",
           xlim=(0, s.max() * 1.15))
    ax.grid(axis="y", visible=False)
    ann = s[s.index.isin(ANNOTATOR_FEATURES)].sum()
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#6250d6"), plt.Rectangle((0, 0), 1, 1, color="#8a8985")],
              labels=[f"radiologist-rated ({ann:.0%} of total)", "image-derived"], loc="lower right")
    return _save(fig, out / "fig_rf_feature_importance.png", dpi)


def fig_prob_dist(d: Dict, out: Path, dpi: int) -> Path:
    preds, table = d["preds"], d["table"]
    y = preds["label"].to_numpy()
    bins = np.linspace(0, 1, 21)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    for ax, (k, name, _) in zip(axes, MODELS):
        p = preds[f"prob_{k}"].to_numpy()
        for cls, nm in ((0, "benign"), (1, "malignant")):
            ax.hist(p[y == cls], bins=bins, color=CLASS_COLORS[cls], alpha=0.55, label=nm, linewidth=0)
            ax.hist(p[y == cls], bins=bins, histtype="step", color=CLASS_COLORS[cls], linewidth=1.4)
        thr = table.loc[k, "threshold"]
        ax.axvline(thr, c=INK, lw=1.2, ls="--")
        ax.text(thr + 0.015, ax.get_ylim()[1] * 0.93, f"thr {thr:.2f}", fontsize=8, color=MUTED)
        ax.set(xlabel="Predicted P(malignant)", title=name, xlim=(0, 1))
    axes[0].set_ylabel("Test nodules")
    axes[0].legend(loc="upper left")
    return _save(fig, out / "fig_probability_distributions.png", dpi)


def fig_splits(d: Dict, out: Path, dpi: int) -> Optional[Path]:
    sp = d["splits"]
    if not sp:
        log.warning("split CSVs not found; skipping dataset figure")
        return None
    names = list(sp)
    ben = [int((sp[n]["label"] == 0).sum()) for n in names]
    mal = [int((sp[n]["label"] == 1).sum()) for n in names]
    fig, ax = plt.subplots(figsize=(6, 4.0))
    ax.bar(names, ben, 0.55, color=CLASS_COLORS[0], label="benign (0)", linewidth=0)
    ax.bar(names, mal, 0.55, bottom=[b + 2 for b in ben], color=CLASS_COLORS[1], label="malignant (1)", linewidth=0)
    for i, (b, m) in enumerate(zip(ben, mal)):
        ax.text(i, b / 2, str(b), ha="center", va="center", color="white", fontweight="bold")
        ax.text(i, b + 2 + m / 2, f"{m}", ha="center", va="center", color="white", fontweight="bold")
        ax.text(i, b + m + 14, f"n={b + m}  ({m / (b + m):.0%} malignant)", ha="center", fontsize=8.5, color=MUTED)
    ax.set(ylabel="Nodules", xlim=(-0.65, 2.65), title="Series-disjoint split: class balance", ylim=(0, max(b + m for b, m in zip(ben, mal)) * 1.15))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    return _save(fig, out / "fig_dataset_splits.png", dpi)


def fig_cam_localization(d: Dict, out: Path, dpi: int) -> Optional[Path]:
    audit = d["audit"]
    if not audit:
        return None
    cases = [c for c in audit["cases"] if c.get("available") and "mass_fraction_in_central_16cube" in c]
    if not cases:
        return None
    uniform = 16 ** 3 / 32 ** 3
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    labels = [f"Case {c['case']}\n" + textwrap.fill(c["title"], 16) for c in cases]
    vals = [c["mass_fraction_in_central_16cube"] for c in cases]
    ax.bar(labels, vals, 0.55, color="#2a78d6", linewidth=0)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.006, f"{v:.2f}", ha="center", fontsize=9)
    ax.axhline(uniform, c=INK, ls="--", lw=1.2, label=f"uniform map over the cube = {uniform:.3f}")
    ax.legend(loc="upper right")
    ax.set(ylabel="Grad-CAM mass inside central 16^3 cube", ylim=(0, max(0.5, max(vals) * 1.3)),
           title=f"Does the CAM concentrate on the nodule? ({audit.get('target_layer', 'layer4')})")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", labelsize=8)
    return _save(fig, out / "fig_cam_localization.png", dpi)


def write_tables(d: Dict, sig: pd.DataFrame, out: Path) -> None:
    t = d["table"]
    lines = ["| Model | ROC-AUC (95% CI) | Accuracy | Sensitivity | Specificity | Precision | F1 | FN | FP | Threshold |",
             "|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|"]
    for k, name, _ in MODELS:
        r = t.loc[k]
        lines.append(f"| {name} | {r.roc_auc:.3f} ({r.auc_ci95_low:.3f}-{r.auc_ci95_high:.3f}) | {r.accuracy:.3f} | "
                     f"{r.sensitivity:.3f} | {r.specificity:.3f} | {r.precision:.3f} | {r.f1:.3f} | "
                     f"{int(r.false_negatives)} | {int(r.false_positives)} | {r.threshold:.3f} |")
    (out / "metrics_table.md").write_text("\n".join(lines) + "\n")
    sig.to_csv(out / "significance_tests.csv", index=False)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    p.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    p.add_argument("--out-dir", type=Path, default=None, help="default: <results-dir>/figures")
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    style()
    out = args.out_dir or args.results_dir / "figures"
    d = load_inputs(args.results_dir, args.processed_dir)
    fig_roc(d, out, args.dpi)
    fig_confusion(d, out, args.dpi)
    fig_metrics(d, out, args.dpi)
    sig = paired_tests(d, args.n_boot, args.seed)
    fig_auc_diff(sig, out, args.dpi)
    fig_training(d, out, args.dpi)
    fig_rf_importance(d, out, args.dpi)
    fig_prob_dist(d, out, args.dpi)
    fig_splits(d, out, args.dpi)
    fig_cam_localization(d, out, args.dpi)
    write_tables(d, sig, out)
    print("\nPaired significance tests (test set):")
    print(sig.round(3).to_string(index=False))
    print(f"\nFigures written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
