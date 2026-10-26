"""
models/hybrid_net.py -- Dual-branch hybrid CNN + radiomics network and its vision-only ablation.

Components
----------
* ``build_visual_backbone``  3D ResNet-18 (torchvision ``r3d_18``) adapted to 1-channel CT.
* ``Pure3DResNet``           image-only model (Model B in the ablation).
* ``HybridNoduleNet``        image + 15 tabular features with gated cross-modal fusion (Model C).

Both networks share the *same* visual encoder (512 -> 128 projection) and the *same*
classification head (128 -> 64 -> 1), so the only difference between B and C is the tabular
branch + gated fusion. Both output a raw logit of shape ``(B, 1)`` (apply sigmoid for P(malignant)).

Input conventions
-----------------
* ``volume``:  float32 ``(B, 1, 32, 32, 32)`` in [0, 1], axes ``(D, H, W)``.
* ``tabular``: float32 ``(B, 15)``, **standardised with train-set statistics** (the BatchNorm in the
  tabular MLP does not replace feature scaling for inputs spanning -1000..3000 HU).
"""
from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn
from torchvision.models.video import r3d_18

N_TABULAR_FEATURES = 15
VISUAL_DIM = 128
TABULAR_DIM = 64
BACKBONE_FEATURES = 512  # r3d_18 global-pooled feature size


def build_visual_backbone(pretrained: bool = False) -> nn.Module:
    """Return a 1-channel 3D ResNet-18 whose ``fc`` is replaced by ``Identity`` (-> 512-d features).

    ``stem[0]`` (the 3-channel Conv3d) is swapped for a 1-channel Conv3d with identical geometry.
    With ``pretrained=True`` the Kinetics-400 weights are loaded (requires a download) and the RGB
    stem kernels are averaged into the single input channel; default is random init.
    """
    if pretrained:
        from torchvision.models.video import R3D_18_Weights

        net = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    else:
        net = r3d_18(weights=None)

    old: nn.Conv3d = net.stem[0]
    new = nn.Conv3d(
        1, old.out_channels, kernel_size=old.kernel_size, stride=old.stride,
        padding=old.padding, bias=old.bias is not None,
    )
    if pretrained:
        with torch.no_grad():
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
    net.stem[0] = new
    net.fc = nn.Identity()
    return net


class _VisualEncoder(nn.Module):
    """3D ResNet-18 backbone + 512 -> 128 embedding projection."""

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        self.backbone = build_visual_backbone(pretrained)
        self.proj = nn.Linear(BACKBONE_FEATURES, VISUAL_DIM)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(volume))


def _classifier_head() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(VISUAL_DIM, 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, 1)
    )


def _check_volume(volume: torch.Tensor) -> None:
    if volume.ndim != 5 or volume.shape[1] != 1:
        raise ValueError(f"volume must be (B, 1, D, H, W); got {tuple(volume.shape)}")


class Pure3DResNet(nn.Module):
    """Vision-only ablation: 3D ResNet-18 -> 128-d embedding -> classification head -> logit."""

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        self.visual = _VisualEncoder(pretrained)
        self.head = _classifier_head()

    @property
    def backbone(self) -> nn.Module:
        """The ResNet trunk (``layer1..layer4``) -- Grad-CAM hooks attach here."""
        return self.visual.backbone

    def forward(self, volume: torch.Tensor, tabular: torch.Tensor | None = None) -> torch.Tensor:
        """``tabular`` is accepted and ignored so both models share one call signature."""
        _check_volume(volume)
        return self.head(self.visual(volume))


class GatedCrossModalFusion(nn.Module):
    """``gate = sigmoid(W [z_v ; z_t])``;  ``z = gate * z_v + (1 - gate) * W_t z_t``.

    The tabular embedding (64-d) is projected into the visual space (128-d); the gate (128-d, per
    channel) decides how much of each modality passes through.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(VISUAL_DIM + TABULAR_DIM, VISUAL_DIM)
        self.tab_proj = nn.Linear(TABULAR_DIM, VISUAL_DIM)

    def forward(self, z_v: torch.Tensor, z_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate = torch.sigmoid(self.gate(torch.cat([z_v, z_t], dim=1)))
        fused = gate * z_v + (1.0 - gate) * self.tab_proj(z_t)
        return fused, gate


class HybridNoduleNet(nn.Module):
    """Dual-branch network: 3D ResNet-18 (volume) + MLP (15 radiomic/metadata features).

    ``forward(volume, tabular)`` returns logits ``(B, 1)``; with ``return_gate=True`` it returns
    ``(logits, gate)`` where ``gate`` is ``(B, 128)`` (values near 1 = visual-dominated).
    """

    def __init__(self, n_tabular: int = N_TABULAR_FEATURES, pretrained: bool = False) -> None:
        super().__init__()
        self.visual = _VisualEncoder(pretrained)
        self.tabular = nn.Sequential(
            nn.Linear(n_tabular, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, TABULAR_DIM),
            nn.GELU(),
        )
        self.fusion = GatedCrossModalFusion()
        self.head = _classifier_head()
        self.n_tabular = n_tabular

    @property
    def backbone(self) -> nn.Module:
        """The ResNet trunk (``layer1..layer4``) -- Grad-CAM hooks attach here."""
        return self.visual.backbone

    def forward(
        self, volume: torch.Tensor, tabular: torch.Tensor, return_gate: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        _check_volume(volume)
        if tabular.ndim != 2 or tabular.shape[1] != self.n_tabular:
            raise ValueError(f"tabular must be (B, {self.n_tabular}); got {tuple(tabular.shape)}")
        z_v = self.visual(volume)
        z_t = self.tabular(tabular)
        fused, gate = self.fusion(z_v, z_t)
        logits = self.head(fused)
        return (logits, gate) if return_gate else logits


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
