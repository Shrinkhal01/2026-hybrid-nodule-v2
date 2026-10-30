"""
models/gradcam.py -- 3D Grad-CAM for the 3D ResNet branch (reusable by scripts and the web app).

Grad-CAM (Selvaraju et al., 2017) on the output of the last residual block of a chosen ResNet stage:

    alpha_k = mean_{d,h,w} dY/dA_k        CAM = ReLU( sum_k alpha_k * A_k )

then trilinear upsampling to the input size and scaling to [0, 1]. ``Y`` is the logit of the class being
explained (``sign=+1`` malignant evidence, ``sign=-1`` benign evidence).

Native CAM resolution for a 32^3 input: layer4 -> 4x2x2, layer3 -> 8x4x4 (very coarse).
Works for both ``HybridNoduleNet`` and ``Pure3DResNet`` (both expose ``.backbone``).

Phase 7 step R measured these maps on all 151 test nodules (results/phase7_R/EXPLAIN_SUMMARY.md):
layer4 scored like an untrained network (hit rate 15%), occlusion localised best (hit rate ~50%).
So :func:`occlusion_map` is the default explanation from Phase 7 step E on.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VALID_LAYERS = ("layer1", "layer2", "layer3", "layer4")
OCCLUSION = "occlusion"  # explain(..., layer="occlusion") uses occlusion_map() instead of Grad-CAM


def get_target_block(model: nn.Module, layer: str = "layer4") -> nn.Module:
    """Last residual block of ``model.backbone.<layer>``."""
    if layer not in VALID_LAYERS:
        raise ValueError(f"layer must be one of {VALID_LAYERS}, got {layer!r}")
    if not hasattr(model, "backbone"):
        raise AttributeError("model has no .backbone (expected HybridNoduleNet or Pure3DResNet)")
    return getattr(model.backbone, layer)[-1]


class GradCAM3D:
    """Forward-hook based Grad-CAM for ``(B, C, D, H, W)`` activations.

    Use as a context manager (or call :meth:`close`) so the hook is always removed::

        with GradCAM3D(model, get_target_block(model, "layer4")) as engine:
            cam, logit, native = engine(volume, tabular, sign=+1.0)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self._acts: Optional[torch.Tensor] = None
        self._handle = target_layer.register_forward_hook(self._save)

    def _save(self, _m: nn.Module, _inp: Any, out: torch.Tensor) -> None:
        self._acts = out

    def close(self) -> None:
        self._handle.remove()

    def __enter__(self) -> "GradCAM3D":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __call__(self, volume: torch.Tensor, tabular: Optional[torch.Tensor] = None,
                 sign: float = 1.0) -> Tuple[np.ndarray, float, Tuple[int, int, int]]:
        """Return ``(cam, logit, native_shape)``; ``cam`` is ``(D, H, W)`` float32 in [0, 1] (all zeros if
        the ReLU'd map vanishes, i.e. no positive evidence for the explained class)."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self._acts = None
        with torch.enable_grad():
            logit = (self.model(volume, tabular) if tabular is not None else self.model(volume)).squeeze()
            if self._acts is None:
                raise RuntimeError("target layer did not fire during the forward pass")
            grads = torch.autograd.grad(sign * logit, self._acts)[0]
        acts = self._acts.detach()
        weights = grads.mean(dim=(2, 3, 4), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        native = tuple(int(s) for s in cam.shape[2:])
        cam = F.interpolate(cam, size=volume.shape[2:], mode="trilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy().astype(np.float32)
        peak = float(cam.max())
        cam = cam / peak if peak > 1e-12 else np.zeros_like(cam)
        return cam, float(logit.detach()), native  # type: ignore[return-value]


@torch.no_grad()
def occlusion_map(model: nn.Module, volume: torch.Tensor, tabular: Optional[torch.Tensor], sign: float = 1.0,
                  size: int = 8, stride: int = 4, fill: float = 0.0, batch: int = 128) -> np.ndarray:
    """Occlusion sensitivity for ONE sample: hide a ``size``^3 cube (``fill`` 0 = -1000 HU, air) at every
    ``stride`` step and record how much the explained logit (``sign`` * logit) drops. Each voxel gets the
    mean drop of the cubes covering it; ReLU'd and scaled to [0, 1].

    ``volume`` is ``(1, D, H, W)`` or ``(1, 1, D, H, W)``; ``tabular`` is ``(F,)`` / ``(1, F)`` or None.
    For 32^3 with size 8 / stride 4: 7^3 = 343 forward passes.
    """
    model.eval()
    vol = volume if volume.ndim == 5 else volume[None]
    tab = None if tabular is None else (tabular if tabular.ndim == 2 else tabular[None])
    run = (lambda v, t: model(v, t)) if tab is not None else (lambda v, t: model(v))
    n = vol.shape[-1]
    starts = list(range(0, n - size + 1, stride))
    pos = [(z, y, x) for z in starts for y in starts for x in starts]
    base = float(run(vol, tab).float().squeeze())
    drops = []
    for i in range(0, len(pos), batch):
        chunk = pos[i:i + batch]
        v = vol.repeat(len(chunk), 1, 1, 1, 1)
        for k, (z, y, x) in enumerate(chunk):
            v[k, 0, z:z + size, y:y + size, x:x + size] = fill
        t = None if tab is None else tab.repeat(len(chunk), 1)
        drops.append(sign * (base - run(v, t).float().squeeze(1).cpu().numpy()))
    drop = np.concatenate(drops)
    heat, cnt = np.zeros((n, n, n)), np.zeros((n, n, n))
    for d, (z, y, x) in zip(drop, pos):
        heat[z:z + size, y:y + size, x:x + size] += d
        cnt[z:z + size, y:y + size, x:x + size] += 1
    heat = np.clip(heat / np.maximum(cnt, 1), 0, None)
    return (heat / heat.max()).astype(np.float32) if heat.max() > 0 else heat.astype(np.float32)


def explain(model: nn.Module, volume: torch.Tensor, tabular: Optional[torch.Tensor], layer: str,
            threshold: float, target: str = "predicted") -> Dict[str, Any]:
    """Convenience wrapper: Grad-CAM + prediction for one sample.

    ``layer``: ``"occlusion"`` (default choice since Phase 7) or a ResNet stage for Grad-CAM.
    ``target``: ``"predicted"`` explains the model's own decision at ``threshold``; ``"malignant"`` always
    explains evidence for the malignant class. Returns cam, logit, prob, predicted label, native shape.
    """
    if target not in ("predicted", "malignant"):
        raise ValueError("target must be 'predicted' or 'malignant'")
    with torch.no_grad():
        model.eval()
        logit0 = float((model(volume, tabular) if tabular is not None else model(volume)).squeeze())
    prob = float(1.0 / (1.0 + np.exp(-logit0)))
    pred = int(prob >= threshold)
    sign = 1.0 if (target == "malignant" or pred == 1) else -1.0
    if layer == OCCLUSION:
        cam, logit = occlusion_map(model, volume, tabular, sign=sign), logit0
        native = (7, 7, 7)  # 8 mm cubes every 4 mm on a 32 mm patch
    else:
        with GradCAM3D(model, get_target_block(model, layer)) as engine:
            cam, logit, native = engine(volume, tabular, sign=sign)
    return {"cam": cam, "logit": logit, "prob_malignant": prob, "predicted": pred,
            "native_shape": native, "explained": "malignant" if sign > 0 else "benign"}


def cam_summary(cam: np.ndarray) -> Dict[str, Any]:
    """Localisation diagnostics: peak voxel and share of CAM mass inside the central 16^3 cube
    (a uniform map would score 0.125)."""
    if not cam.any():
        return {"cam_all_zero": True}
    peak = np.unravel_index(int(cam.argmax()), cam.shape)
    return {
        "cam_all_zero": False,
        "peak_voxel_dhw": [int(i) for i in peak],
        "mass_fraction_in_central_16cube": float(cam[8:24, 8:24, 8:24].sum() / cam.sum()),
        "fraction_voxels_above_0.5": float((cam > 0.5).mean()),
    }
