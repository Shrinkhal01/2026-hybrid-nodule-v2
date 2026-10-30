"""Phase 7 step E: occlusion maps find known evidence; the device picker falls back cleanly.

    python -m pytest tests/test_occlusion_device.py -v      (CPU, ~5 s)
"""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipelines.common as common  # noqa: E402
from models.gradcam import OCCLUSION, explain, occlusion_map  # noqa: E402
from models.xai_metrics import localisation  # noqa: E402


class CentreSum(torch.nn.Module):
    """Toy 'model' whose logit is the brightness of the central 8 mm cube: the evidence IS the nodule."""

    def forward(self, vol, tab=None):
        return vol[:, :, 12:20, 12:20, 12:20].sum(dim=(1, 2, 3, 4))[:, None] - 100.0


def test_occlusion_finds_the_central_evidence():
    vol = torch.full((1, 1, 32, 32, 32), 0.5)
    cam = occlusion_map(CentreSum(), vol, None, sign=1.0)
    assert cam.shape == (32, 32, 32) and cam.max() == pytest.approx(1.0)
    score = localisation(cam, diameter_mm=8)
    assert score["hit"] == 1.0 and score["concentration"] > 3
    assert cam[0, 0, 0] < cam[16, 16, 16]          # a corner hides nothing the model uses


def test_occlusion_sign_flips_for_benign_explanations():
    """Explaining the OTHER class: hiding the evidence raises that class -> no positive heat."""
    vol = torch.full((1, 1, 32, 32, 32), 0.5)
    assert occlusion_map(CentreSum(), vol, None, sign=-1.0).max() == 0.0


def test_explain_accepts_occlusion():
    vol = torch.full((1, 1, 32, 32, 32), 0.5)
    e = explain(CentreSum(), vol, torch.zeros(1, 15), OCCLUSION, threshold=0.5, target="malignant")
    assert e["native_shape"] == (7, 7, 7) and e["explained"] == "malignant"


def test_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(common, "_mps_works", lambda: False)
    assert common.get_device(require_cuda=False).type == "cpu"
    with pytest.raises(RuntimeError, match="allow-cpu"):
        common.get_device(require_cuda=True)


def test_device_prefers_mps_when_it_works(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(common, "_mps_works", lambda: True)
    assert common.get_device(require_cuda=False).type == "mps"
