#!/usr/bin/env python3
"""
pipelines/03_evaluate_xai.py -- 3D Grad-CAM explanations for the best Hybrid model.

Loads ``results/best_hybrid_net.pth``, scores the held-out test set, picks four cases and, for each,
renders the central axial slice (z = 16) as  [ original CT | 3D Grad-CAM | alpha overlay ]:

    Case 1  True Positive, highest-confidence malignant      (max P(malignant) among TPs)
    Case 2  True Negative, highest-confidence benign         (min P(malignant) among TNs)
    Case 3  Subtle malignant, correctly identified            (smallest diameter_mm among remaining TPs)
    Case 4  False positive (highest-P benign predicted malignant); if the test set has none,
            the most borderline nodule (P(malignant) closest to the decision threshold)

Grad-CAM (Selvaraju et al.) is computed on the output of the last residual block of ``layer4`` of
the 3D CNN branch: alpha_k = mean_{d,h,w} dY/dA_k ; CAM = ReLU(sum_k alpha_k A_k), trilinearly
upsampled to 32^3 and scaled to [0, 1]. Y is the logit of the model's *predicted* class (sign-flipped
for predicted-benign cases), so the map always shows evidence *for the prediction being explained*.

Phase 7 step E: the default is now OCCLUSION (``--target-layer occlusion``): hide an 8 mm cube at a
time and map how much the explained logit drops. Step R scored all methods on all 151 test nodules
(results/phase7_R/EXPLAIN_SUMMARY.md): layer4 Grad-CAM (4x2x2) hit the nodule in 15% of cases, like
an untrained network; occlusion in ~50%. Grad-CAM stays available (``--target-layer layer3`` etc.).
Each case in the audit also gets the ball-mask scores of models/xai_metrics.py.

Outputs: ``results/gradcam_hybrid_overlay.png`` and ``results/xai_cases_audit.json``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.gradcam import OCCLUSION, occlusion_map  # noqa: E402
from models.hybrid_net import HybridNoduleNet  # noqa: E402
from models.xai_metrics import localisation  # noqa: E402
from pipelines.common import (  # noqa: E402
    FEATURE_COLS, FEATURE_SETS, PROCESSED_DIR, RESULTS_DIR, NoduleDataset, TabularScaler, get_device, load_split,
    make_loader, predict_probs, youden_threshold,
)

log = logging.getLogger("day2.xai")
SLICE_IDX = 16


# --------------------------------------------------------------------------- #
# Grad-CAM
# --------------------------------------------------------------------------- #
class GradCAM3D:
    """Grad-CAM for 5-D activations ``(B, C, D, H, W)`` of a chosen layer.

    Usage::

        cam_engine = GradCAM3D(model, model.backbone.layer4[-1])
        cam, logit = cam_engine(volume, tabular, sign=+1.0)
        cam_engine.close()
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self._acts: Optional[torch.Tensor] = None
        self._handle = target_layer.register_forward_hook(self._save)

    def _save(self, _m: nn.Module, _inp: Any, out: torch.Tensor) -> None:
        self._acts = out

    def close(self) -> None:
        self._handle.remove()

    def __call__(self, volume: torch.Tensor, tabular: torch.Tensor,
                 sign: float = 1.0) -> Tuple[np.ndarray, float, Tuple[int, int, int]]:
        """Return ``(cam, logit, native_shape)``; ``cam`` is ``(D, H, W)`` float32 in [0, 1]."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logit = self.model(volume, tabular).squeeze()
            if self._acts is None:
                raise RuntimeError("target layer did not fire during forward")
            grads = torch.autograd.grad(sign * logit, self._acts)[0]
        acts = self._acts.detach()
        weights = grads.mean(dim=(2, 3, 4), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))  # (1,1,d,h,w)
        native = tuple(int(s) for s in cam.shape[2:])
        cam = F.interpolate(cam, size=volume.shape[2:], mode="trilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy().astype(np.float32)
        peak = cam.max()
        cam = cam / peak if peak > 1e-12 else np.zeros_like(cam)
        return cam, float(logit.detach()), native  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Case selection
# --------------------------------------------------------------------------- #
def select_cases(df: pd.DataFrame, thr: float) -> List[Dict[str, Any]]:
    """Pick up to four distinct test nodules. ``df`` needs: label, prob, diameter_mm.

    Returns dicts with ``case``, ``title``, ``rule``, ``row`` (positional index) -- or ``row=None``
    when a category is empty (recorded in the audit instead of failing).
    """
    pred = (df["prob"] >= thr).astype(int)
    tp = df[(df["label"] == 1) & (pred == 1)]
    tn = df[(df["label"] == 0) & (pred == 0)]
    used: List[int] = []

    def pick(pool: pd.DataFrame, key: pd.Series | None, ascending: bool) -> Optional[int]:
        pool = pool[~pool.index.isin(used)]
        if pool.empty:
            return None
        idx = int((key.loc[pool.index] if key is not None else pool["prob"]).sort_values(ascending=ascending).index[0])
        used.append(idx)
        return idx

    out = [
        {"case": 1, "title": "TP: high-confidence malignant",
         "rule": "true malignant, predicted malignant, highest P(malignant)",
         "row": pick(tp, df["prob"], ascending=False)},
        {"case": 2, "title": "TN: high-confidence benign",
         "rule": "true benign, predicted benign, lowest P(malignant)",
         "row": pick(tn, df["prob"], ascending=True)},
        {"case": 3, "title": "Subtle malignant (correct)",
         "rule": "true malignant, predicted malignant, smallest diameter_mm (excluding case 1)",
         "row": pick(tp, df["diameter_mm"], ascending=True)},
        {"case": 4, "title": "False positive / borderline", "rule": "", "row": None},
    ]
    fp = df[(df["label"] == 0) & (pred == 1)]
    if not fp.empty:
        out[3]["title"] = "False positive"
        out[3]["rule"] = "true benign, predicted malignant, highest P(malignant)"
        out[3]["row"] = pick(fp, df["prob"], ascending=False)
    else:  # no FP on the test set: fall back to the most borderline nodule
        out[3]["title"] = "Borderline (no false positives)"
        out[3]["rule"] = "no false positives exist; P(malignant) closest to threshold (excluding cases 1-3)"
        out[3]["row"] = pick(df, (df["prob"] - thr).abs(), ascending=True)
    for c in out:
        if c["row"] is None:
            log.warning("Case %d (%s): no eligible nodule in the test set.", c["case"], c["title"])
    return out


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def render_figure(records: List[Dict[str, Any]], out_path: Path, alpha: float, dpi: int = 150,
                  method: str = "occlusion (8 mm cubes)") -> None:
    """3-column panel per case: CT slice | Grad-CAM | overlay."""
    n = len(records)
    fig, axes = plt.subplots(n, 3, figsize=(10.5, 3.4 * n), squeeze=False)
    for r, rec in enumerate(records):
        ct, cam = rec["_ct_slice"], rec["_cam_slice"]
        axes[r, 0].imshow(ct, cmap="gray", vmin=0, vmax=1)
        im = axes[r, 1].imshow(cam, cmap="jet", vmin=0, vmax=1)
        axes[r, 2].imshow(ct, cmap="gray", vmin=0, vmax=1)
        axes[r, 2].imshow(cam, cmap="jet", vmin=0, vmax=1, alpha=alpha)
        fig.colorbar(im, ax=axes[r, 1], fraction=0.046, pad=0.02)
        if not cam.any():  # ReLU(CAM) == 0: no positive evidence for the explained class
            for c in (1, 2):
                axes[r, c].text(0.5, 0.5, "CAM = 0\n(no positive evidence)", ha="center", va="center",
                                transform=axes[r, c].transAxes, color="white", fontsize=9,
                                bbox=dict(facecolor="black", alpha=0.6, pad=3))
        verdict = "correct" if rec["correct"] else "WRONG"
        axes[r, 0].set_ylabel(
            f"Case {rec['case']}: {rec['title']}\ntrue={'malignant' if rec['true_label'] else 'benign'}  "
            f"P(mal)={rec['prob_malignant']:.2f}  ({verdict})", fontsize=8)
        for c in range(3):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        if r == 0:
            for c, t in enumerate(("Original CT slice (z=16)", method, "Overlay")):
                axes[r, c].set_title(t, fontsize=10)
    fig.suptitle(f"Hybrid Net - {method}, evidence for the predicted class, test set", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_threshold(results_dir: Path, y_val: np.ndarray, p_val: np.ndarray) -> Tuple[float, str]:
    """Hybrid threshold from ``run_config.json`` (written by 02_train.py), else recompute on val."""
    cfg = results_dir / "run_config.json"
    if cfg.exists():
        for k, v in json.loads(cfg.read_text()).get("thresholds", {}).items():
            if k.startswith("C:"):
                return float(v), "run_config.json (validation Youden's J)"
    log.warning("run_config.json has no hybrid threshold; recomputing Youden's J on validation.")
    return youden_threshold(y_val, p_val), "recomputed on validation (Youden's J)"


def cam_statistics(cam: np.ndarray) -> Dict[str, Any]:
    """Compact, serialisable summary of a 32^3 CAM (for the audit trail)."""
    if not cam.any():
        return {"cam_all_zero": True}
    peak = np.unravel_index(int(cam.argmax()), cam.shape)
    mass = float(cam.sum())
    return {
        "cam_all_zero": False,
        "peak_voxel_dhw": [int(i) for i in peak],
        "peak_distance_from_center_vox": float(np.linalg.norm(np.array(peak) - 15.5)),
        "mass_fraction_in_central_16cube": float(cam[8:24, 8:24, 8:24].sum() / mass),
        "mean_activation_on_slice_16": float(cam[SLICE_IDX].mean()),
        "fraction_voxels_above_0.5": float((cam > 0.5).mean()),
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--checkpoint", type=Path, default=None, help="default: <results-dir>/best_hybrid_net.pth")
    p.add_argument("--target-layer", choices=(OCCLUSION, "layer4", "layer3", "layer2"), default=OCCLUSION,
                   help="occlusion (default, best in Phase 7 step R) or a Grad-CAM stage")
    p.add_argument("--alpha", type=float, default=0.4, help="overlay opacity of the heatmap")
    p.add_argument("--limit", type=int, default=None, help="DEBUG: subsample splits")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    ckpt_path = args.checkpoint or args.results_dir / "best_hybrid_net.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found -- run pipelines/02_train.py first.")
    device = get_device()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cols = ckpt.get("feature_cols", FEATURE_COLS)  # Phase 7: 'all' (15) or 'image' (12)
    if cols not in FEATURE_SETS.values():
        raise ValueError(f"checkpoint feature_cols are not a known feature set: {cols}")
    scaler = TabularScaler.from_dict(ckpt["scaler"])
    model = HybridNoduleNet(n_tabular=len(cols))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    log.info("Loaded %s (epoch %s, val_auc %.4f)", ckpt_path.name, ckpt.get("epoch"), ckpt.get("val_auc", float("nan")))

    val_df = load_split("val", args.processed_dir, args.limit, args.seed)
    test_df = load_split("test", args.processed_dir, args.limit, args.seed)
    loader = lambda d: make_loader(NoduleDataset(d, scaler, feature_cols=cols), 64, shuffle=False)  # noqa: E731
    p_val = predict_probs(model, loader(val_df), device)
    thr, thr_src = load_threshold(args.results_dir, val_df["label"].to_numpy(int), p_val)

    test_ds = NoduleDataset(test_df, scaler, feature_cols=cols)
    p_test = predict_probs(model, make_loader(test_ds, 64, shuffle=False), device)
    scored = test_df.assign(prob=p_test)
    log.info("Test: n=%d, threshold=%.4f (%s)", len(scored), thr, thr_src)

    use_occ = args.target_layer == OCCLUSION
    engine = None if use_occ else GradCAM3D(model, getattr(model.backbone, args.target_layer)[-1])
    records: List[Dict[str, Any]] = []
    audit_cases: List[Dict[str, Any]] = []
    try:
        for sel in select_cases(scored, thr):
            if sel["row"] is None:
                audit_cases.append({"case": sel["case"], "title": sel["title"], "rule": sel["rule"],
                                    "available": False})
                continue
            i = sel["row"]
            row = scored.iloc[i]
            item = test_ds[i]
            vol = item["volume"].unsqueeze(0).to(device)
            tab = item["tabular"].unsqueeze(0).to(device)
            pred = int(row["prob"] >= thr)
            sign = 1.0 if pred == 1 else -1.0
            if use_occ:
                cam = occlusion_map(model, vol, tab, sign=sign)
                with torch.no_grad():
                    logit, native = float(model(vol, tab).squeeze()), (7, 7, 7)
            else:
                cam, logit, native = engine(vol, tab, sign=sign)
            with torch.no_grad():
                _, gate = model(vol, tab, return_gate=True)
            ct = vol[0, 0].cpu().numpy()
            rec = {
                "case": sel["case"], "title": sel["title"], "prob_malignant": float(row["prob"]),
                "true_label": int(row["label"]), "correct": bool(pred == int(row["label"])),
                "_ct_slice": ct[SLICE_IDX], "_cam_slice": cam[SLICE_IDX],
            }
            records.append(rec)
            audit_cases.append({
                "case": sel["case"], "title": sel["title"], "rule": sel["rule"], "available": True,
                "nodule_id": row["nodule_id"], "series_uid": row.get("series_uid"),
                "patch_path": row["patch_path"], "true_label": int(row["label"]),
                "malignancy_mean_score": float(row["malignancy_mean"]) if "malignancy_mean" in row else None,
                "diameter_mm": float(row["diameter_mm"]), "prob_malignant": float(row["prob"]),
                "logit": logit, "threshold": thr, "predicted_label": pred,
                "correct": bool(pred == int(row["label"])),
                "explained_class": "malignant" if pred == 1 else "benign",
                "visual_gate_mean": float(gate.mean()),
                "slice_index_d": SLICE_IDX, "cam_native_shape_dhw": list(native),
                **cam_statistics(cam),
                **{f"ball_{k}": v for k, v in localisation(cam, float(row["diameter_mm"])).items()},
            })
    finally:
        if engine is not None:
            engine.close()

    if not records:
        raise RuntimeError("No cases could be selected; nothing to plot.")
    fig_path = args.results_dir / "gradcam_hybrid_overlay.png"
    render_figure(records, fig_path, args.alpha,
                  method="occlusion (8 mm cubes)" if use_occ else f"3D Grad-CAM ({args.target_layer})")
    audit = {
        "checkpoint": str(ckpt_path), "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_val_auc": ckpt.get("val_auc"),
        "target_layer": "occlusion (8 mm cube, stride 4 mm)" if use_occ else f"backbone.{args.target_layer}[-1]",
        "threshold": thr, "threshold_source": thr_src, "n_test": int(len(scored)),
        "notes": [
            "CAMs explain the model's predicted class (logit sign-flipped for predicted-benign cases).",
            "Occlusion maps: 7x7x7 grid of 8 mm cubes; Grad-CAM layer4 is only 4x2x2 (Phase 7 step R: hit rate 15%).",
            "ball_* = scores against a ball with the nodule's diameter (models/xai_metrics.py); concentration 1 = flat map.",
            "visual_gate_mean ~1 means the fusion gate favours the CT embedding over the tabular one.",
            "Grad-CAM shows where the CNN branch's evidence lies; it does not explain the tabular branch.",
        ],
        "cases": audit_cases,
    }
    (args.results_dir / "xai_cases_audit.json").write_text(json.dumps(audit, indent=2))
    log.info("Saved %s and xai_cases_audit.json", fig_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
