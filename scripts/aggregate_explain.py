#!/usr/bin/env python3
"""Phase 7 step R: summarise pipelines/05_explain_eval.py over all seeds of a run (CPU, no torch).

    python scripts/aggregate_explain.py results/phase7_A

Reads <run>/<feature set>/seed_*/explain/{per_nodule.csv, summary.json, dependence.csv, permutation.csv}
and writes <run>/EXPLAIN_SUMMARY.md (+ explain_seeds.csv). Every number is a mean over seeds,
with the SD across seeds, so one lucky run cannot carry the conclusion.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ORDER = ["gradcam_layer4", "gradcam_layer3", "gradcam_layer2", "occlusion_8mm", "random_weights_layer3", "centre_prior"]


def main() -> int:
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/phase7_A")
    seeds = sorted(run.glob("*/seed_*/explain/per_nodule.csv"))
    if not seeds:
        raise SystemExit(f"no explain/per_nodule.csv under {run} -- run scripts/explain_on_gpu.sh first")
    per, dep, perm, san = [], [], [], []
    for f in seeds:
        fs, seed = f.parts[-4], int(f.parts[-3].split("_")[1])
        d = pd.read_csv(f)
        g = (d.groupby(["model", "method"]).agg(conc_median=("concentration", "median"), hit_rate=("hit", "mean"),
                                                 com_dist_mm=("com_dist_mm", "mean"), all_zero=("all_zero", "mean"))
               .reset_index().assign(feature_set=fs, seed=seed))
        per.append(g)
        dep.append(pd.read_csv(f.parent / "dependence.csv").assign(feature_set=fs, seed=seed))
        p = f.parent / "permutation.csv"
        if p.exists() and p.stat().st_size > 1:
            try:
                perm.append(pd.read_csv(p).assign(feature_set=fs, seed=seed))
            except pd.errors.EmptyDataError:
                pass
        s = json.loads((f.parent / "summary.json").read_text())
        for m, v in s["models"].items():
            san.append({"feature_set": fs, "seed": seed, "model": m, "rho": v["sanity_spearman_trained_vs_random_layer3_mean"]})
    per = pd.concat(per)
    per.to_csv(run / "explain_seeds.csv", index=False)
    agg = (per.groupby(["feature_set", "model", "method"])
              .agg(seeds=("seed", "nunique"), conc=("conc_median", "mean"), conc_sd=("conc_median", "std"),
                   hit=("hit_rate", "mean"), hit_sd=("hit_rate", "std"), dist=("com_dist_mm", "mean"),
                   zero=("all_zero", "mean")).reset_index())
    agg["order"] = agg["method"].map({m: i for i, m in enumerate(ORDER)})
    agg = agg.sort_values(["feature_set", "model", "order"])
    f3 = lambda v: f"{v:.2f}"  # noqa: E731
    md = [f"# {run.name}: explanations scored on the test nodules, mean over seeds", "",
          "Concentration = share of the map on the nodule / share of the cube the nodule fills "
          "(1.00 = flat map). Hit = hottest voxel on the nodule (±2 mm). The **centre prior** is the "
          "trivial 'always look at the centre' map; it is the bar a real explanation should approach.", "",
          "| features | model | method | seeds | concentration (median per seed) | hit rate | centre-of-mass distance | all-zero maps |",
          "|:--|:--|:--|:-:|:-:|:-:|:-:|:-:|"]
    for r in agg.itertuples():
        md.append(f"| {r.feature_set} | {r.model} | {r.method} | {r.seeds} | {f3(r.conc)} ± {f3(r.conc_sd)} | "
                  f"{r.hit:.0%} ± {r.hit_sd:.0%} | {r.dist:.1f} mm | {r.zero:.0%} |")
    s = pd.DataFrame(san).groupby(["feature_set", "model"])["rho"].agg(["mean", "std"]).reset_index()
    md += ["", "## Sanity check (Grad-CAM layer3, trained vs random weights)", "",
           "Rank correlation near 1 = the map hardly depends on what the network learned (bad); near 0 = it does.", "",
           "| features | model | correlation, mean ± SD over seeds |", "|:--|:--|:-:|"]
    md += [f"| {r.feature_set} | {r.model} | {r.mean:.2f} ± {r.std:.2f} |" for r in s.itertuples()]
    d = (pd.concat(dep).groupby(["feature_set", "model", "shuffled"])
           .agg(orig=("auc_original", "mean"), shuf=("auc_shuffled_mean", "mean"), drop=("auc_drop", "mean"),
                drop_sd=("auc_drop", "std")).reset_index())
    md += ["", "## Which input does the model rely on? (shuffle it between nodules)", "",
           "| features | model | shuffled | AUC original | AUC shuffled | AUC drop, mean ± SD over seeds |", "|:--|:--|:--|:-:|:-:|:-:|"]
    md += [f"| {r.feature_set} | {r.model} | {r.shuffled} | {f3(r.orig)} | {f3(r.shuf)} | {r.drop:+.3f} ± {r.drop_sd:.3f} |"
           for r in d.itertuples()]
    if perm:
        p = (pd.concat(perm).groupby(["feature_set", "feature"])["auc_drop_mean"].agg(["mean", "std"]).reset_index()
               .sort_values(["feature_set", "mean"], ascending=[True, False]))
        md += ["", "## Hybrid: table features by permutation importance (top 5 per feature set)", "",
               "| features | feature | AUC drop when shuffled, mean ± SD over seeds |", "|:--|:--|:-:|"]
        for fs, g in p.groupby("feature_set"):
            md += [f"| {fs} | {r.feature} | {r.mean:+.3f} ± {r.std:.3f} |" for r in g.head(5).itertuples()]
    md += ["", "The ball uses the radiologists' mean diameter only to draw the target; no model saw it for this test.",
           "Data: LIDC-IDRI, TCIA DOI 10.7937/K9/TCIA.2015.LO9QL9SX."]
    (run / "EXPLAIN_SUMMARY.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
