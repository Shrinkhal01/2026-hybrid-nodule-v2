#!/usr/bin/env python3
"""
pipelines/05_explain_eval.py -- Phase 7 step R: measure the explanations on ALL test nodules.

Part 2 judged Grad-CAM by looking at 4 hand-picked cases. This script scores every heat map
against a "bull's-eye" (a ball with the nodule's diameter at the patch centre; see
models/xai_metrics.py) and checks whether the maps depend on what the network learned.

For each trained model in --results-dir (Hybrid and 3D ResNet), on the 151 test nodules:
  B  ball-mask scoring: concentration (1 = flat map), pointing-game hit rate, centre-of-mass distance
  O  occlusion maps: hide one 8 mm cube at a time (stride 4 mm), map = drop in the explained logit
  A  Grad-CAM at layer2 (16x8x8), layer3 (8x4x4), layer4 (4x2x2), explaining the predicted class
  R  sanity check: Grad-CAM of a RANDOM-weights copy; rank correlation with the trained map
  D  dependence (Hybrid): AUC when the images, or the 12/15 table features, are shuffled between
     nodules (10 repeats), plus per-feature permutation importance
References: a flat map (= 1.0) and the trivial centre prior (a Gaussian at the centre).

Outputs: <results-dir>/explain/{per_nodule.csv, summary.json, dependence.csv, permutation.csv,
         fig_explain_cases.png, fig_explain_scores.png}

Usage
    python pipelines/05_explain_eval.py --results-dir results/phase7_A/all/seed_42
    python pipelines/05_explain_eval.py --results-dir ... --limit 20 --allow-cpu     # quick CPU check
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.gradcam import get_target_block  # noqa: E402
from models.hybrid_net import HybridNoduleNet, Pure3DResNet  # noqa: E402
from models.xai_metrics import center_prior, localisation, spearman_maps  # noqa: E402
from pipelines.common import (  # noqa: E402
    FEATURE_COLS, FEATURE_SETS, PROCESSED_DIR, NoduleDataset, TabularScaler, get_device, load_split,
    resolve_patch_path, set_seed,
)

log = logging.getLogger("explain")
MODELS = {  # key -> (checkpoint, class, run_config threshold label, display name)
    "hybrid": ("best_hybrid_net.pth", HybridNoduleNet, "C: Hybrid Net (image + radiomics)", "C: Hybrid"),
    "resnet": ("best_pure_resnet.pth", Pure3DResNet, "B: Pure 3D ResNet-18 (image only)", "B: 3D ResNet"),
}
LAYERS = ("layer2", "layer3", "layer4")


# --------------------------------------------------------------------------- #
# model loading + batched forward passes
# --------------------------------------------------------------------------- #
def load_model(key: str, results_dir: Path, device: torch.device):
    ckpt_name, cls, _, _ = MODELS[key]
    ckpt = torch.load(results_dir / ckpt_name, map_location="cpu", weights_only=True)
    cols = ckpt.get("feature_cols", FEATURE_COLS)
    if cols not in FEATURE_SETS.values():
        raise ValueError(f"{ckpt_name}: unknown feature set {cols}")
    model = cls(n_tabular=len(cols)) if cls is HybridNoduleNet else cls()
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), cols, TabularScaler.from_dict(ckpt["scaler"]), ckpt


def random_copy(model: torch.nn.Module, n_tab: int, seed: int, device: torch.device) -> torch.nn.Module:
    """Same architecture, fresh random weights (never trained) -> the sanity-check reference."""
    torch.manual_seed(seed)
    m = HybridNoduleNet(n_tabular=n_tab) if isinstance(model, HybridNoduleNet) else Pure3DResNet()
    return m.to(device).eval()


@torch.no_grad()
def logits_of(model, vols: torch.Tensor, tabs: torch.Tensor, device, bs: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(vols), bs):
        out.append(model(vols[i:i + bs].to(device), tabs[i:i + bs].to(device)).float().squeeze(1).cpu())
    return torch.cat(out).numpy()


def gradcam_batch(model, vols, tabs, signs, layer: str, device, bs: int = 32):
    """Batched 3D Grad-CAM (eval mode, so samples don't interact). Returns (N, 32, 32, 32) in [0, 1]
    and the native map shape. Same maths as models/gradcam.GradCAM3D."""
    cams, native = [], None
    block = get_target_block(model, layer)
    for i in range(0, len(vols), bs):
        store: Dict[str, torch.Tensor] = {}
        h = block.register_forward_hook(lambda _m, _i, o: store.__setitem__("a", o))
        try:
            v, t = vols[i:i + bs].to(device), tabs[i:i + bs].to(device)
            s = torch.as_tensor(signs[i:i + bs], dtype=torch.float32, device=device)
            with torch.enable_grad():
                logit = model(v, t).float().squeeze(1)
                g = torch.autograd.grad((s * logit).sum(), store["a"])[0]
        finally:
            h.remove()
        a = store["a"].detach().float()
        cam = F.relu((g.mean(dim=(2, 3, 4), keepdim=True) * a).sum(dim=1, keepdim=True))
        native = tuple(int(x) for x in cam.shape[2:])
        cam = F.interpolate(cam, size=v.shape[2:], mode="trilinear", align_corners=False)[:, 0]
        peak = cam.flatten(1).max(dim=1).values.clamp(min=1e-12)[:, None, None, None]
        cams.append((cam / peak).cpu().numpy())
    return np.concatenate(cams), native


@torch.no_grad()
def occlusion_map(model, vol, tab, sign: float, device, size: int = 8, stride: int = 4,
                  fill: float = 0.0) -> np.ndarray:
    """Hide a size^3 cube (fill 0 = -1000 HU, air) at every stride step; heat = how much the
    explained logit drops, averaged over the cubes covering each voxel; ReLU'd, scaled to [0, 1]."""
    n = vol.shape[-1]
    starts = list(range(0, n - size + 1, stride))
    pos = [(z, y, x) for z in starts for y in starts for x in starts]
    v1, t1 = vol[None].to(device), tab[None].to(device)
    base = float(model(v1, t1).float().squeeze())
    batch = v1.repeat(len(pos), 1, 1, 1, 1)
    for k, (z, y, x) in enumerate(pos):
        batch[k, 0, z:z + size, y:y + size, x:x + size] = fill
    out = []
    for i in range(0, len(pos), 128):
        out.append(model(batch[i:i + 128], t1.repeat(len(batch[i:i + 128]), 1)).float().squeeze(1).cpu())
    drop = sign * (base - torch.cat(out).numpy())
    heat, cnt = np.zeros((n, n, n)), np.zeros((n, n, n))
    for d, (z, y, x) in zip(drop, pos):
        heat[z:z + size, y:y + size, x:x + size] += d
        cnt[z:z + size, y:y + size, x:x + size] += 1
    heat = np.clip(heat / np.maximum(cnt, 1), 0, None)
    return (heat / heat.max()).astype(np.float32) if heat.max() > 0 else heat.astype(np.float32)


def auc(y, p) -> float:
    return float(roc_auc_score(y, p))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, required=True, help="one trained run, e.g. results/phase7_A/all/seed_42")
    p.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    p.add_argument("--no-occlusion", action="store_true", help="skip occlusion maps (the slow part on CPU)")
    p.add_argument("--repeats", type=int, default=10, help="shuffles per dependence / permutation test")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="DEBUG: first N test nodules only")
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    device = get_device(require_cuda=not args.allow_cpu)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out_dir = args.results_dir / "explain"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_cfg = json.loads((args.results_dir / "run_config.json").read_text())
    test_df = load_split("test", args.processed_dir)
    if args.limit:
        test_df = test_df.iloc[:args.limit].reset_index(drop=True)
    y = test_df["label"].to_numpy(int)
    diam = test_df["diameter_mm"].to_numpy(float)  # radiologist diameter: used ONLY to draw the bull's-eye
    prior = center_prior()

    rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"results_dir": str(args.results_dir), "n_test": int(len(y)),
                               "feature_set": run_cfg.get("feature_set", "all"), "models": {}}
    dep_rows, perm_rows, case_maps = [], [], {}
    for key in args.models:
        model, cols, scaler, ckpt = load_model(key, args.results_dir, device)
        name = MODELS[key][3]
        ds = NoduleDataset(test_df, scaler, feature_cols=cols)
        vols, tabs = ds.volumes, ds.tabular
        logit = logits_of(model, vols, tabs, device)
        prob = 1 / (1 + np.exp(-logit))
        thr = float(run_cfg["thresholds"][MODELS[key][2]])
        pred = (prob >= thr).astype(int)
        signs = np.where(pred == 1, 1.0, -1.0)  # explain the class the model predicted
        log.info("[%s] %d test nodules, AUC %.3f, threshold %.3f", name, len(y), auc(y, prob) if len(set(y)) > 1 else float("nan"), thr)

        maps: Dict[str, np.ndarray] = {}
        natives = {}
        for layer in LAYERS:
            maps[f"gradcam_{layer}"], natives[layer] = gradcam_batch(model, vols, tabs, signs, layer, device)
        rnd = random_copy(model, len(cols), args.seed + 1000, device)
        maps["random_weights_layer3"], _ = gradcam_batch(rnd, vols, tabs, signs, "layer3", device)
        if not args.no_occlusion:
            maps["occlusion_8mm"] = np.stack([occlusion_map(model, vols[i], tabs[i], float(signs[i]), device)
                                              for i in range(len(y))])
        maps["centre_prior"] = np.repeat(prior[None], len(y), axis=0)

        for method, m in maps.items():
            for i in range(len(y)):
                rows.append({"model": name, "method": method, "nodule_id": test_df["nodule_id"][i], "label": int(y[i]),
                             "prob": float(prob[i]), "pred": int(pred[i]), "diameter_mm": float(diam[i]),
                             **localisation(m[i], diam[i])})
        rho = [spearman_maps(maps["gradcam_layer3"][i], maps["random_weights_layer3"][i]) for i in range(len(y))]
        summary["models"][name] = {"threshold": thr, "native_shapes": {k: list(v) for k, v in natives.items()},
                                   "sanity_spearman_trained_vs_random_layer3_mean": float(np.nanmean(rho))}
        case_maps[name] = {k: v for k, v in maps.items() if k != "random_weights_layer3"}
        case_maps[name]["_prob"], case_maps[name]["_pred"] = prob, pred

        # ---- D: dependence on picture vs sheet (shuffle between nodules) -------------
        base_auc = auc(y, prob) if len(set(y)) > 1 else float("nan")
        for what in (["image", "table"] if key == "hybrid" else ["image"]):
            aucs = []
            for _ in range(args.repeats):
                perm = rng.permutation(len(y))
                v2, t2 = (vols[perm], tabs) if what == "image" else (vols, tabs[perm])
                aucs.append(auc(y, logits_of(model, v2, t2, device)))
            dep_rows.append({"model": name, "shuffled": what, "auc_original": base_auc,
                             "auc_shuffled_mean": float(np.mean(aucs)), "auc_shuffled_sd": float(np.std(aucs, ddof=1)),
                             "auc_drop": base_auc - float(np.mean(aucs))})
        if key == "hybrid":
            for j, c in enumerate(cols):
                drops = []
                for _ in range(args.repeats):
                    t2 = tabs.clone()
                    t2[:, j] = t2[rng.permutation(len(y)), j]
                    drops.append(base_auc - auc(y, logits_of(model, vols, t2, device)))
                perm_rows.append({"model": name, "feature": c, "auc_drop_mean": float(np.mean(drops)),
                                  "auc_drop_sd": float(np.std(drops, ddof=1))})
        del rnd

    per = pd.DataFrame(rows)
    per.to_csv(out_dir / "per_nodule.csv", index=False)
    dep = pd.DataFrame(dep_rows)
    dep.to_csv(out_dir / "dependence.csv", index=False)
    perm = pd.DataFrame(perm_rows).sort_values("auc_drop_mean", ascending=False) if perm_rows else pd.DataFrame()
    perm.to_csv(out_dir / "permutation.csv", index=False)
    table = (per.groupby(["model", "method"])
               .agg(concentration_median=("concentration", "median"), concentration_mean=("concentration", "mean"),
                    hit_rate=("hit", "mean"), mass_in_mean=("mass_in", "mean"), com_dist_mm_mean=("com_dist_mm", "mean"),
                    all_zero=("all_zero", "sum"))
               .reset_index())
    for name, g in table.groupby("model"):
        summary["models"][name]["methods"] = g.drop(columns="model").set_index("method").round(4).to_dict("index")
    summary["dependence"] = dep.round(4).to_dict("records")
    summary["permutation_top5"] = perm.head(5).round(4).to_dict("records") if len(perm) else []
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    make_figures(out_dir, test_df, case_maps, per)

    pd.set_option("display.width", 160)
    print("\n" + "=" * 100 + f"\nEXPLANATIONS ON {len(y)} TEST NODULES  (concentration: 1 = flat map; centre_prior = trivial reference)\n" + "=" * 100)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nSanity check, rank correlation trained vs random-weights Grad-CAM (layer3): " +
          ", ".join(f"{k} {v['sanity_spearman_trained_vs_random_layer3_mean']:.2f}" for k, v in summary["models"].items()))
    print("\nDependence (shuffle between nodules):\n" + dep.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if len(perm):
        print("\nHybrid: top table features by permutation importance (AUC drop):\n" +
              perm.head(5).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaved to {out_dir}")
    return 0


def make_figures(out_dir: Path, test_df: pd.DataFrame, case_maps: Dict[str, Dict[str, Any]], per: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from models.xai_metrics import ball_mask

    # 1) score overview: concentration (log) and hit rate per method
    methods = [m for m in ["gradcam_layer4", "gradcam_layer3", "gradcam_layer2", "occlusion_8mm",
                           "random_weights_layer3", "centre_prior"] if m in set(per["method"])]
    models = list(dict.fromkeys(per["model"]))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    w = 0.8 / len(models)
    for k, mname in enumerate(models):
        g = per[per["model"] == mname]
        data = [g[g["method"] == m]["concentration"].dropna().clip(lower=1e-2).to_numpy() for m in methods]
        pos = np.arange(len(methods)) + (k - (len(models) - 1) / 2) * w
        bp = ax[0].boxplot(data, positions=pos, widths=w * 0.9, patch_artist=True, showfliers=False)
        for b in bp["boxes"]:
            b.set_facecolor(["#2E6BD6", "#EE8A2F"][k % 2]); b.set_alpha(.7)
        hits = [g[g["method"] == m]["hit"].mean() for m in methods]
        ax[1].bar(pos, hits, width=w * 0.9, color=["#2E6BD6", "#EE8A2F"][k % 2], alpha=.8, label=mname)
    ax[0].axhline(1.0, color="k", ls="--", lw=1)
    ax[0].text(len(methods) - .5, 1.05, "flat map = 1", ha="right", va="bottom", fontsize=9)
    ax[0].set_yscale("log"); ax[0].set_ylabel("concentration on the nodule (log)")
    ax[1].set_ylabel("pointing game: hottest voxel on the nodule"); ax[1].set_ylim(0, 1); ax[1].legend(frameon=False)
    for a in ax:
        a.set_xticks(range(len(methods))); a.set_xticklabels([m.replace("_", "\n") for m in methods], fontsize=8)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Explanations scored on all {per['nodule_id'].nunique()} test nodules", fontsize=12)
    fig.tight_layout(); fig.savefig(out_dir / "fig_explain_scores.png", dpi=150); plt.close(fig)

    # 2) four cases x methods, Hybrid if present (axial slice through the centre)
    name = "C: Hybrid" if "C: Hybrid" in case_maps else next(iter(case_maps))
    cm = case_maps[name]
    y = test_df["label"].to_numpy(int)
    prob, pred = cm["_prob"], cm["_pred"]
    pick = []
    for cond, key in (((y == 1) & (pred == 1), -prob), ((y == 0) & (pred == 0), prob),
                      ((y == 1) & (pred == 0), -prob), ((y == 0) & (pred == 1), -prob)):
        idx = np.where(cond)[0]
        if len(idx):
            pick.append(int(idx[np.argsort(key[idx])[0]]))
    titles = ["true positive", "true negative", "missed cancer", "false alarm"]
    show = [m for m in ["gradcam_layer4", "gradcam_layer3", "gradcam_layer2", "occlusion_8mm"] if m in cm]
    if not pick:
        return
    fig, ax = plt.subplots(len(pick), len(show) + 1, figsize=(2.6 * (len(show) + 1), 2.7 * len(pick)), squeeze=False)
    for r, i in enumerate(pick):
        ct = np.load(resolve_patch_path(test_df["patch_path"][i], PROJECT_ROOT))
        ring = ball_mask(test_df["diameter_mm"][i])[16]
        ax[r, 0].imshow(ct[16], cmap="gray", vmin=0, vmax=1)
        ax[r, 0].contour(ring, levels=[.5], colors="#2F9E6B", linewidths=1.2)
        ax[r, 0].set_ylabel(f"{titles[r]}\nP={prob[i]:.2f}, {test_df['diameter_mm'][i]:.0f} mm", fontsize=8)
        for c, m in enumerate(show, start=1):
            ax[r, c].imshow(ct[16], cmap="gray", vmin=0, vmax=1)
            ax[r, c].imshow(cm[m][i][16], cmap="jet", alpha=.45, vmin=0, vmax=1)
            ax[r, c].contour(ring, levels=[.5], colors="white", linewidths=1)
            sc = localisation(cm[m][i], test_df["diameter_mm"][i])
            ax[r, c].set_title(f"{m.replace('gradcam_', 'Grad-CAM ').replace('_8mm', ' 8 mm')}\nconc {sc['concentration']:.1f}", fontsize=8)
        for a in ax[r]:
            a.set_xticks([]); a.set_yticks([])
    ax[0, 0].set_title("CT (ring = nodule)", fontsize=8)
    fig.suptitle(f"{name}: where each method says the evidence is (axial slice through the centre)", fontsize=10)
    fig.tight_layout(); fig.savefig(out_dir / "fig_explain_cases.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
