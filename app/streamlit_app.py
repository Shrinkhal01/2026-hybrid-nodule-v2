"""
app/streamlit_app.py -- interactive demo for the Hybrid Explainable-AI lung-nodule malignancy framework.

Run (from the project root):
    streamlit run app/streamlit_app.py
On a remote GPU server:
    streamlit run app/streamlit_app.py --server.port 8501 --server.headless true
    # then, on your laptop:  ssh -L 8501:localhost:8501 <user>@<server>   ->  http://localhost:8501

Needs the Day 2 artefacts: results/{best_hybrid_net.pth, best_pure_resnet.pth, baseline_rf.joblib,
model_comparison_metrics.csv} and data/processed/{train,test}.csv + patches/. Override locations with the
environment variables NODULE_RESULTS_DIR / NODULE_DATA_DIR (the *processed* folder).

RESEARCH PROTOTYPE -- not a medical device; not validated for clinical use.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402
from scipy import stats as sps  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.baseline_rf import BaselineRF  # noqa: E402
from models.gradcam import cam_summary, explain  # noqa: E402
from models.hybrid_net import HybridNoduleNet, Pure3DResNet  # noqa: E402
from pipelines.common import FEATURE_COLS, TabularScaler, resolve_patch_path  # noqa: E402

RESULTS_DIR = Path(os.environ.get("NODULE_RESULTS_DIR", PROJECT_ROOT / "results"))
PROCESSED_DIR = Path(os.environ.get("NODULE_DATA_DIR", PROJECT_ROOT / "data" / "processed"))

MODEL_NAMES = {"rf": "A: Random Forest", "resnet": "B: 3D ResNet-18", "hybrid": "C: Hybrid Net"}
FEATURE_GROUPS = {
    "Intensity statistics (HU)": ["mean_hu", "std_hu", "min_hu", "max_hu", "median_hu", "p25_hu", "p75_hu",
                                  "iqr_hu", "skewness", "kurtosis"],
    "Tissue density (%)": ["solid_fraction_pct", "hyperdense_fraction_pct"],
    "Radiologist consensus": ["diameter_mm", "calcification", "texture"],
}
LAYER_CHOICES = {"layer4 (native 4x2x2)": "layer4", "layer3 (native 8x4x4)": "layer3"}
PLANES = {"Axial (D)": 0, "Coronal (H)": 1, "Sagittal (W)": 2}
DISCLAIMER = ("Research prototype built on the public LIDC-IDRI benchmark. It is **not a medical device** and "
              "must not be used for diagnosis or treatment decisions.")


# --------------------------------------------------------------------------- #
# Loading (cached)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading models...")
def load_bundle(results_dir: str) -> Dict[str, Any]:
    """Load every available model once. Missing artefacts are reported, not fatal."""
    rd = Path(results_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle: Dict[str, Any] = {"device": device, "missing": [], "thresholds": {}, "scaler": None}

    def load_net(name: str, ckpt_name: str, factory):
        path = rd / ckpt_name
        if not path.exists():
            bundle["missing"].append(ckpt_name)
            return None
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if ckpt.get("feature_cols") not in (None, FEATURE_COLS):
            raise ValueError(f"{ckpt_name}: feature_cols mismatch")
        net = factory()
        net.load_state_dict(ckpt["state_dict"])
        net.to(device).eval()
        if bundle["scaler"] is None and "scaler" in ckpt:
            bundle["scaler"] = TabularScaler.from_dict(ckpt["scaler"])
        bundle[f"{name}_meta"] = {"epoch": ckpt.get("epoch"), "val_auc": ckpt.get("val_auc")}
        return net

    bundle["hybrid"] = load_net("hybrid", "best_hybrid_net.pth", HybridNoduleNet)
    bundle["resnet"] = load_net("resnet", "best_pure_resnet.pth", Pure3DResNet)
    rf_path = rd / "baseline_rf.joblib"
    if rf_path.exists():
        bundle["rf"] = BaselineRF.load(rf_path)
    else:
        bundle["rf"] = None
        bundle["missing"].append("baseline_rf.joblib")

    table = rd / "model_comparison_metrics.csv"
    if table.exists():
        t = pd.read_csv(table)
        for _, r in t.iterrows():
            bundle["thresholds"][{"A": "rf", "B": "resnet", "C": "hybrid"}[r["model"][0]]] = float(r["threshold"])
    else:
        bundle["missing"].append("model_comparison_metrics.csv (thresholds default to 0.5)")
    for k in ("rf", "resnet", "hybrid"):
        bundle["thresholds"].setdefault(k, 0.5)
    return bundle


@st.cache_data(show_spinner=False)
def load_table(path: str) -> Optional[pd.DataFrame]:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_test_frame(results_dir: str, processed_dir: str) -> Optional[pd.DataFrame]:
    """test.csv joined with the stored test-set probabilities of the three models."""
    test = load_table(str(Path(processed_dir) / "test.csv"))
    if test is None:
        return None
    preds = load_table(str(Path(results_dir) / "test_predictions.csv"))
    if preds is not None:
        test = test.merge(preds.drop(columns=["label"]), on="nodule_id", how="left")
    return test


@st.cache_data(show_spinner=False)
def train_medians(processed_dir: str) -> Dict[str, float]:
    tr = load_table(str(Path(processed_dir) / "train.csv"))
    if tr is None:
        return {c: 0.0 for c in FEATURE_COLS}
    return {c: float(tr[c].median()) for c in FEATURE_COLS}


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def scaled_tabular(bundle: Dict[str, Any], feats: np.ndarray) -> torch.Tensor:
    if bundle["scaler"] is None:
        raise RuntimeError("no tabular scaler available (needs a deep-model checkpoint)")
    return torch.from_numpy(bundle["scaler"].transform(feats[None]))


@torch.no_grad()
def predict_all(bundle: Dict[str, Any], patch: np.ndarray, feats: np.ndarray) -> Dict[str, Optional[float]]:
    """P(malignant) from each available model for one nodule."""
    dev = bundle["device"]
    vol = torch.from_numpy(patch.astype(np.float32))[None, None].to(dev)
    out: Dict[str, Optional[float]] = {"rf": None, "resnet": None, "hybrid": None}
    if bundle["rf"] is not None:
        out["rf"] = float(bundle["rf"].predict_proba(feats[None])[0])
    if bundle["resnet"] is not None:
        out["resnet"] = float(torch.sigmoid(bundle["resnet"](vol)).item())
    if bundle["hybrid"] is not None:
        tab = scaled_tabular(bundle, feats).to(dev)
        logit, gate = bundle["hybrid"](vol, tab, return_gate=True)
        out["hybrid"] = float(torch.sigmoid(logit).item())
        out["_gate"] = float(gate.mean().item())  # type: ignore[assignment]
    return out


@st.cache_data(show_spinner="Computing Grad-CAM...")
def cached_cam(_bundle: Dict[str, Any], model_key: str, patch: np.ndarray, feats: np.ndarray, layer: str,
               target: str) -> Dict[str, Any]:
    """Grad-CAM for one nodule (cached on its pixels/features + settings)."""
    dev = _bundle["device"]
    vol = torch.from_numpy(patch.astype(np.float32))[None, None].to(dev)
    tab = scaled_tabular(_bundle, feats).to(dev) if model_key == "hybrid" else None
    e = explain(_bundle[model_key], vol, tab, layer, _bundle["thresholds"][model_key], target)
    e["summary"] = cam_summary(e["cam"])
    return e


def estimate_image_features(patch: np.ndarray) -> Dict[str, float]:
    """Approximate the 12 image-derived features from a windowed patch (HU ~= x*1400 - 1000).

    Windowing clipped HU to [-1000, 400], so min/max/percentiles can differ slightly from the values
    computed on the raw CT in training. Use for convenience only.
    """
    hu = (patch.astype(np.float64) * 1400.0 - 1000.0).ravel()
    p25, med, p75 = np.percentile(hu, [25, 50, 75])
    flat = hu.std() < 1e-9
    return {
        "mean_hu": hu.mean(), "std_hu": hu.std(), "min_hu": hu.min(), "max_hu": hu.max(), "median_hu": med,
        "p25_hu": p25, "p75_hu": p75, "iqr_hu": p75 - p25,
        "skewness": 0.0 if flat else float(sps.skew(hu)), "kurtosis": 0.0 if flat else float(sps.kurtosis(hu)),
        "solid_fraction_pct": 100.0 * float(np.mean(hu > -400)),
        "hyperdense_fraction_pct": 100.0 * float(np.mean(hu > 100)),
    }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def take_slice(vol: np.ndarray, plane: int, idx: int) -> np.ndarray:
    return (vol[idx], vol[:, idx, :], vol[:, :, idx])[plane]


def render_slices(patch: np.ndarray, cam: np.ndarray, plane: int, idx: int, alpha: float) -> plt.Figure:
    ct, cm = take_slice(patch, plane, idx), take_slice(cam, plane, idx)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
    axes[0].imshow(ct, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(cm, cmap="jet", vmin=0, vmax=1)
    axes[2].imshow(ct, cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(cm, cmap="jet", vmin=0, vmax=1, alpha=alpha)
    for ax, t in zip(axes, ("CT (lung window)", "3D Grad-CAM", "Overlay")):
        ax.set_title(t, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    if not cam.any():
        for ax in axes[1:]:
            ax.text(0.5, 0.5, "CAM = 0\n(no positive evidence)", ha="center", va="center", transform=ax.transAxes,
                    color="white", fontsize=9, bbox=dict(facecolor="black", alpha=0.6, pad=3))
    fig.tight_layout()
    return fig


def decision_text(p: Optional[float], thr: float) -> str:
    if p is None:
        return "n/a"
    return "malignant" if p >= thr else "benign"


def show_predictions(bundle: Dict[str, Any], probs: Dict[str, Optional[float]], truth: Optional[int]) -> None:
    cols = st.columns(4 if truth is not None else 3)
    for col, key in zip(cols, ("hybrid", "resnet", "rf")):
        p = probs.get(key)
        thr = bundle["thresholds"][key]
        with col:
            if p is None:
                st.metric(MODEL_NAMES[key], "unavailable")
            else:
                st.metric(MODEL_NAMES[key], f"{p:.1%}")
                st.caption(f"P(malignant) -> **{decision_text(p, thr)}** at threshold {thr:.2f}")
    if truth is not None:
        with cols[3]:
            st.metric("Ground truth", "malignant" if truth else "benign")
            st.caption("radiologist consensus")
    if probs.get("_gate") is not None:
        st.caption(f"Fusion gate (mean) = {probs['_gate']:.2f} -> "
                   f"{'CT embedding' if probs['_gate'] > 0.5 else 'tabular embedding'} weighted slightly more. "
                   "Gate ~0.5 means both branches contribute about equally.")


def show_explanation(bundle: Dict[str, Any], patch: np.ndarray, feats: np.ndarray, key_prefix: str) -> None:
    """Sidebar-driven Grad-CAM viewer for one nodule."""
    avail = [k for k in ("hybrid", "resnet") if bundle.get(k) is not None]
    if not avail:
        st.info("No deep-model checkpoint found, so Grad-CAM is unavailable.")
        return
    c1, c2, c3, c4 = st.columns([1.3, 1.3, 1.2, 1])
    model_key = c1.selectbox("Explain model", avail, format_func=lambda k: MODEL_NAMES[k], key=f"{key_prefix}_m")
    layer = LAYER_CHOICES[c2.selectbox("Target layer", list(LAYER_CHOICES), key=f"{key_prefix}_l")]
    target = c3.radio("Explain evidence for", ["predicted", "malignant"], horizontal=True, key=f"{key_prefix}_t",
                      format_func=lambda s: "model's prediction" if s == "predicted" else "malignant class")
    alpha = c4.slider("Overlay opacity", 0.1, 0.9, 0.4, 0.05, key=f"{key_prefix}_a")
    c5, c6 = st.columns([1, 3])
    plane_name = c5.radio("Plane", list(PLANES), key=f"{key_prefix}_p")
    idx = c6.slider("Slice index (0-31)", 0, 31, 16, key=f"{key_prefix}_i")

    e = cached_cam(bundle, model_key, patch, feats, layer, target)
    st.pyplot(render_slices(patch, e["cam"], PLANES[plane_name], idx, alpha), clear_figure=True)
    s = e["summary"]
    if s.get("cam_all_zero"):
        st.warning("CAM = 0: no positive evidence for the explained class at this layer.")
    else:
        st.caption(
            f"Explaining **{e['explained']}** evidence · native map {e['native_shape']} (trilinearly upsampled) · "
            f"peak voxel (D,H,W) = {tuple(s['peak_voxel_dhw'])} · CAM mass in central 16³ cube = "
            f"{s['mass_fraction_in_central_16cube']:.2f} (a uniform map scores 0.125).")
    st.info("Interpretation caution: in the project's audit, CAM mass in the central cube was close to uniform "
            "(0.14-0.17 at layer4), so these heatmaps are coarse and do not reliably localise the nodule. "
            "Treat them as qualitative; Grad-CAM also says nothing about the tabular branch.")


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def tab_explorer(bundle: Dict[str, Any]) -> None:
    test = load_test_frame(str(RESULTS_DIR), str(PROCESSED_DIR))
    if test is None:
        st.error(f"test.csv not found in {PROCESSED_DIR}. Run the Day 1 pipeline or set NODULE_DATA_DIR.")
        return
    df = test.copy()
    thr = bundle["thresholds"]["hybrid"]
    if "prob_hybrid" in df:
        pred = (df["prob_hybrid"] >= thr).astype(int)
        df["outcome"] = np.select(
            [(df.label == 1) & (pred == 1), (df.label == 0) & (pred == 0), (df.label == 0) & (pred == 1)],
            ["True positive", "True negative", "False positive"], default="False negative")
    else:
        df["outcome"] = "n/a"
    f1, f2 = st.columns([1, 3])
    outcome = f1.selectbox("Filter (hybrid model outcome)", ["All"] + sorted(df["outcome"].unique()))
    view = df if outcome == "All" else df[df["outcome"] == outcome]
    if view.empty:
        st.info("No nodules in this category.")
        return
    options = view.index.tolist()

    def label(i: int) -> str:
        r = view.loc[i]
        p = f"P_hyb={r['prob_hybrid']:.2f}" if "prob_hybrid" in r and pd.notna(r["prob_hybrid"]) else ""
        return f"#{i:03d} · true={'malignant' if r.label else 'benign'} · {p} · {r.outcome} · {r.nodule_id[-28:]}"

    i = f2.selectbox(f"Test nodule ({len(view)} shown)", options, format_func=label)
    row = df.loc[i]
    path = resolve_patch_path(row["patch_path"], PROJECT_ROOT)
    if not path.exists():
        st.error(f"Patch file not found: {path}")
        return
    patch = np.load(path).astype(np.float32)
    feats = row[FEATURE_COLS].to_numpy(dtype=float)
    probs = predict_all(bundle, patch, feats)
    show_predictions(bundle, probs, int(row["label"]))
    with st.expander("Nodule features (the 15 tabular inputs)"):
        st.dataframe(pd.DataFrame({"value": row[FEATURE_COLS].astype(float)}).round(3))
        st.caption(f"Mean radiologist malignancy score: {row.get('malignancy_mean', float('nan')):.2f}  ·  "
                   f"series: {str(row.get('series_uid', ''))[-24:]}")
    show_explanation(bundle, patch, feats, "ex")


def fill_features(patch: np.ndarray) -> None:
    for k, v in estimate_image_features(patch).items():
        st.session_state[f"feat_{k}"] = float(v)


def tab_custom(bundle: Dict[str, Any]) -> None:
    st.write("Upload a **32×32×32 float32 `.npy` patch**, windowed to [0, 1] (lung window −1000…400 HU, axes "
             "depth×height×width) exactly like the project's `data/processed/patches/*.npy`, and enter the 15 features.")
    up = st.file_uploader("Patch (.npy)", type=["npy"])
    med = train_medians(str(PROCESSED_DIR))
    for c in FEATURE_COLS:  # defaults = training medians
        st.session_state.setdefault(f"feat_{c}", med[c])

    patch: Optional[np.ndarray] = None
    if up is not None:
        try:
            arr = np.load(up, allow_pickle=False)
            if arr.shape != (32, 32, 32):
                raise ValueError(f"expected shape (32, 32, 32), got {arr.shape}")
            arr = arr.astype(np.float32)
            if not np.isfinite(arr).all():
                raise ValueError("patch contains NaN/inf")
            if arr.min() < 0 or arr.max() > 1:
                st.warning("Values outside [0, 1] were clipped.")
                arr = np.clip(arr, 0.0, 1.0)
            patch = arr
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            st.error(f"Could not read the patch: {exc}")
    if patch is not None:
        st.button("Estimate the 12 image-derived features from this patch (approximate)",
                  on_click=fill_features, args=(patch,))
    for group, cols in FEATURE_GROUPS.items():
        st.markdown(f"**{group}**")
        grid = st.columns(5)
        for j, c in enumerate(cols):
            grid[j % 5].number_input(c, key=f"feat_{c}", format="%.3f", step=0.1)
    if patch is None:
        st.info("Upload a patch to get predictions.")
        return
    feats = np.array([st.session_state[f"feat_{c}"] for c in FEATURE_COLS], dtype=float)
    show_predictions(bundle, predict_all(bundle, patch, feats), None)
    show_explanation(bundle, patch, feats, "cu")


def tab_performance() -> None:
    table = load_table(str(RESULTS_DIR / "model_comparison_metrics.csv"))
    if table is None:
        st.error("results/model_comparison_metrics.csv not found - run pipelines/02_train.py.")
        return
    show = table[["model", "roc_auc", "auc_ci95_low", "auc_ci95_high", "accuracy", "sensitivity", "specificity",
                  "precision", "f1", "false_negatives", "false_positives", "threshold"]]
    st.subheader("Held-out test set (n = %d)" % int(table["n_test"].iloc[0]))
    st.dataframe(show.round(3), hide_index=True)
    sig = load_table(str(RESULTS_DIR / "figures" / "significance_tests.csv"))
    if sig is not None:
        st.subheader("Paired significance tests")
        st.dataframe(sig.round(3), hide_index=True)
        st.caption("AUC differences use a paired bootstrap on the same 151 test nodules; McNemar compares per-nodule "
                   "correctness at each model's validation threshold. The hybrid's AUC gain is not statistically "
                   "distinguishable from zero.")
    figs = [("fig_roc_curves.png", "ROC curves"), ("fig_metrics_comparison.png", "Metrics at validation thresholds"),
            ("fig_confusion_matrices.png", "Confusion matrices"), ("fig_auc_differences.png", "Paired AUC differences"),
            ("fig_training_curves.png", "Training curves"), ("fig_rf_feature_importance.png", "Random Forest importance"),
            ("fig_probability_distributions.png", "Predicted probabilities"), ("fig_cam_localization.png", "CAM localisation"),
            ("fig_dataset_splits.png", "Dataset splits")]
    present = [(RESULTS_DIR / "figures" / f, c) for f, c in figs if (RESULTS_DIR / "figures" / f).exists()]
    if not present:
        st.info("No figures yet - run `python pipelines/04_report_figures.py`.")
    cols = st.columns(2)
    for n, (p, cap) in enumerate(present):
        cols[n % 2].image(str(p), caption=cap)


def tab_about(bundle: Dict[str, Any]) -> None:
    st.markdown("""
**Task.** Predict whether an annotated lung nodule (32×32×32 CT patch from the LIDC-IDRI 1,176-nodule benchmark) is
benign (mean radiologist malignancy < 3) or malignant (> 3). Indeterminate nodules (= 3.0) are excluded.

**Models**
- **A - Random Forest**: 150 balanced trees on the 15 tabular features.
- **B - 3D ResNet-18**: image only (1-channel stem, 512→128 embedding, 128→64→1 head).
- **C - Hybrid Net**: the same 3D ResNet-18 plus a tabular MLP, combined by a learned gate
  `z = g·z_visual + (1−g)·W·z_tabular`.

**Explainability.** 3D Grad-CAM on the last residual block of `layer4` (or `layer3`).

**Known limitations**
- Detection is out of scope: patches are centred on radiologist-annotated nodules.
- Three of the 15 features (diameter, calcification, texture) are radiologist ratings; the Random Forest relies on
  them heavily (diameter alone ≈ 47% of its importance).
- The hybrid's AUC advantage over the baselines is not statistically significant on 151 test nodules.
- Splits are by CT series (no PatientID in the metadata), so patient-level leakage cannot be fully excluded.
- Grad-CAM maps at 32³ input are coarse and did not reliably concentrate on the nodule.
- Single training run (one seed); no external validation.
""")
    meta = []
    for k in ("hybrid", "resnet"):
        m = bundle.get(f"{k}_meta")
        if m:
            meta.append(f"{MODEL_NAMES[k]}: best epoch {m['epoch']}, validation AUC {m['val_auc']:.3f}")
    if meta:
        st.caption(" · ".join(meta))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Hybrid XAI - Lung Nodule Malignancy", page_icon="🫁", layout="wide")
    st.title("Hybrid Explainable AI for Lung Nodule Malignancy Prediction")
    st.warning(DISCLAIMER)
    bundle = load_bundle(str(RESULTS_DIR))
    with st.sidebar:
        st.header("Environment")
        st.write(f"Device: `{bundle['device']}`")
        st.write(f"Results: `{RESULTS_DIR}`")
        for k in ("hybrid", "resnet", "rf"):
            st.write(("✅ " if bundle.get(k) is not None else "❌ ") + MODEL_NAMES[k])
        for m in bundle["missing"]:
            st.caption(f"missing: {m}")
    if bundle["hybrid"] is None and bundle["resnet"] is None and bundle["rf"] is None:
        st.error("No model artefacts found. Train first: `python pipelines/02_train.py`, or set NODULE_RESULTS_DIR.")
        st.stop()
    t1, t2, t3, t4 = st.tabs(["Test-set explorer", "Custom input", "Model performance", "About & limitations"])
    with t1:
        tab_explorer(bundle)
    with t2:
        tab_custom(bundle)
    with t3:
        tab_performance()
    with t4:
        tab_about(bundle)


main()
