#!/usr/bin/env python3
"""Phase 7 step A: summarise a multi-seed, two-feature-set run (CPU only, no torch needed).

    python scripts/aggregate_runs.py results/phase7_A

Expects  <run>/<feature_set>/seed_<n>/{model_comparison_metrics.csv, test_predictions.csv}
as written by scripts/train_on_gpu.sh, with feature sets 'all' (15 features) and 'image' (12).

Writes <run>/summary_seeds.csv, <run>/summary_ablation.csv and <run>/SUMMARY.md, and prints them.

What it reports
  1. Per feature set x model: test AUC mean +- SD over seeds (how much one seed can lie),
     plus the AUC of the seed-ENSEMBLE (average probability over seeds) with a bootstrap 95% CI.
  2. Ablation (all - image) for RF and Hybrid: per-seed differences, and a paired bootstrap
     on the ensembles (same 151 test nodules) -> is the drop without ratings real?
  3. Reference models with no training randomness: diameter alone, logistic regression on the
     3 radiologist ratings, logistic regression on the 12 image features.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RATINGS = ["diameter_mm", "calcification", "texture"]
IMAGE = ["mean_hu", "std_hu", "min_hu", "max_hu", "median_hu", "p25_hu", "p75_hu", "iqr_hu",
         "skewness", "kurtosis", "solid_fraction_pct", "hyperdense_fraction_pct"]
MODELS = {"rf": "A: Random Forest", "resnet": "B: 3D ResNet-18", "hybrid": "C: Hybrid Net"}
N_BOOT = 5000


def boot_auc(y, p, rng, n=N_BOOT):
    """Point AUC + bootstrap 95% CI (resampling nodules)."""
    stats = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            stats.append(roc_auc_score(y[i], p[i]))
    return roc_auc_score(y, p), np.percentile(stats, 2.5), np.percentile(stats, 97.5)


def paired_boot(y, a, b, rng, n=N_BOOT):
    """AUC(a) - AUC(b) on the same nodules, bootstrap 95% CI and two-sided p."""
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    d = np.asarray(d)
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return roc_auc_score(y, a) - roc_auc_score(y, b), np.percentile(d, 2.5), np.percentile(d, 97.5), p


def load_run(run: Path) -> tuple[pd.DataFrame, dict]:
    """All per-seed metrics rows + {(feature_set, model): DataFrame nodule x seed of probabilities}."""
    rows, probs = [], {}
    for fdir in sorted(p for p in run.iterdir() if p.is_dir()):
        for sdir in sorted(fdir.glob("seed_*")):
            mfile, pfile = sdir / "model_comparison_metrics.csv", sdir / "test_predictions.csv"
            if not (mfile.exists() and pfile.exists()):
                print(f"skip {sdir}: incomplete", file=sys.stderr)
                continue
            seed = int(sdir.name.split("_")[1])
            m = pd.read_csv(mfile)
            for key, label in MODELS.items():
                hit = m[m["model"].str.startswith(label.split(":")[0] + ":")]
                if len(hit):
                    rows.append({"feature_set": fdir.name, "model": key, "seed": seed,
                                 **hit.iloc[0][["val_auc", "roc_auc", "sensitivity", "specificity",
                                                "false_negatives", "false_positives"]].to_dict()})
            pr = pd.read_csv(pfile).set_index("nodule_id")
            for key in MODELS:
                if f"prob_{key}" in pr:
                    probs.setdefault((fdir.name, key), {})[seed] = pr[f"prob_{key}"]
                    probs.setdefault(("_labels", ""), pr["label"])
    if not rows:
        raise SystemExit(f"no complete seed folders under {run}")
    probs = {k: (pd.DataFrame(v) if k[0] != "_labels" else v) for k, v in probs.items()}
    return pd.DataFrame(rows), probs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, help="e.g. results/phase7_A")
    ap.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    per_seed, probs = load_run(args.run)
    labels = probs.pop(("_labels", ""))
    y = labels.to_numpy(int)
    ids = labels.index

    # 1. seeds -------------------------------------------------------------
    summ = []
    for (fs, key), g in per_seed.groupby(["feature_set", "model"]):
        ens = probs[(fs, key)].reindex(ids).mean(axis=1).to_numpy()
        auc, lo, hi = boot_auc(y, ens, rng)
        summ.append({"feature_set": fs, "model": MODELS[key], "n_seeds": len(g),
                     "test_auc_mean": g["roc_auc"].mean(), "test_auc_sd": g["roc_auc"].std(ddof=1),
                     "test_auc_min": g["roc_auc"].min(), "test_auc_max": g["roc_auc"].max(),
                     "val_auc_mean": g["val_auc"].mean(), "sensitivity_mean": g["sensitivity"].mean(),
                     "false_negatives_mean": g["false_negatives"].mean(),
                     "ensemble_auc": auc, "ensemble_ci_low": lo, "ensemble_ci_high": hi})
    summ = pd.DataFrame(summ).sort_values(["feature_set", "model"]).reset_index(drop=True)

    # 2. ablation: all - image ------------------------------------------------
    abl = []
    if {"all", "image"} <= set(per_seed["feature_set"]):
        for key in ("rf", "hybrid", "resnet"):
            a = per_seed[(per_seed.feature_set == "all") & (per_seed.model == key)].set_index("seed")["roc_auc"]
            b = per_seed[(per_seed.feature_set == "image") & (per_seed.model == key)].set_index("seed")["roc_auc"]
            common = a.index.intersection(b.index)
            if not len(common):
                continue
            d = a[common] - b[common]
            ea = probs[("all", key)].reindex(ids).mean(axis=1).to_numpy()
            eb = probs[("image", key)].reindex(ids).mean(axis=1).to_numpy()
            diff, lo, hi, p = paired_boot(y, ea, eb, rng)
            abl.append({"model": MODELS[key] + (" (control: uses no tabular features)" if key == "resnet" else ""),
                        "seeds": len(common),
                        "per_seed_diff_mean": d.mean(), "per_seed_diff_sd": d.std(ddof=1),
                        "seeds_all_better": int((d > 0).sum()),
                        "ensemble_diff": diff, "ensemble_ci_low": lo, "ensemble_ci_high": hi,
                        "ensemble_p": p})
    abl = pd.DataFrame(abl)

    # 3. reference models (deterministic) ----------------------------------------
    tr = pd.read_csv(args.processed_dir / "train.csv")
    te = pd.read_csv(args.processed_dir / "test.csv").set_index("nodule_id").reindex(ids)
    if te["label"].isna().any():
        raise SystemExit("test.csv does not match the nodules in test_predictions.csv")
    refs = [("diameter alone (no fitting)", te["diameter_mm"].to_numpy())]
    for name, cols in (("logistic regression, 3 radiologist ratings", RATINGS),
                       ("logistic regression, 12 image features", IMAGE)):
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(tr[cols], tr["label"])
        refs.append((name, lr.predict_proba(te[cols])[:, 1]))
    ref = pd.DataFrame([{"reference": n, **dict(zip(("test_auc", "ci_low", "ci_high"), boot_auc(y, p, rng)))}
                        for n, p in refs])

    # write + print ----------------------------------------------------------------
    summ.to_csv(args.run / "summary_seeds.csv", index=False)
    abl.to_csv(args.run / "summary_ablation.csv", index=False)
    ref.to_csv(args.run / "summary_reference.csv", index=False)
    f = lambda v: f"{v:.3f}"  # noqa: E731
    md = [f"# {args.run.name}: seeds x feature sets (test n = {len(y)})", "",
          "## 1. Test AUC over seeds", "",
          "| features | model | seeds | mean ± SD | min–max | ensemble AUC [95% CI] | mean FN |",
          "|:--|:--|:-:|:-:|:-:|:-:|:-:|"]
    for r in summ.itertuples():
        md.append(f"| {r.feature_set} | {r.model} | {r.n_seeds} | {f(r.test_auc_mean)} ± {f(r.test_auc_sd)} | "
                  f"{f(r.test_auc_min)}–{f(r.test_auc_max)} | {f(r.ensemble_auc)} [{f(r.ensemble_ci_low)}, "
                  f"{f(r.ensemble_ci_high)}] | {r.false_negatives_mean:.1f} |")
    if len(abl):
        md += ["", "## 2. Ablation: AUC(all 15) − AUC(image 12)", "",
               "| model | per-seed mean ± SD | seeds where 'all' wins | ensemble diff [95% CI] | p |",
               "|:--|:-:|:-:|:-:|:-:|"]
        for r in abl.itertuples():
            md.append(f"| {r.model} | {r.per_seed_diff_mean:+.3f} ± {f(r.per_seed_diff_sd)} | "
                      f"{r.seeds_all_better}/{r.seeds} | {r.ensemble_diff:+.3f} [{r.ensemble_ci_low:+.3f}, "
                      f"{r.ensemble_ci_high:+.3f}] | {r.ensemble_p:.3f} |")
    md += ["", "## 3. Reference models (no training randomness)", "",
           "| reference | test AUC [95% CI] |", "|:--|:-:|"]
    md += [f"| {r.reference} | {f(r.test_auc)} [{f(r.ci_low)}, {f(r.ci_high)}] |" for r in ref.itertuples()]
    md += ["", "Labels are mean radiologist ratings (not biopsy). Data: LIDC-IDRI, "
           "TCIA DOI 10.7937/K9/TCIA.2015.LO9QL9SX."]
    (args.run / "SUMMARY.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
