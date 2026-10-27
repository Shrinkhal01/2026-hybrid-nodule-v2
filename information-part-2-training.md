# Dual-Branch Gated Hybrid Network with 3D Explainability (XAI) for Lung Nodule Malignancy Classification

An end-to-end multimodal deep learning framework benchmarking a **Classical Radiomics Baseline**, a **Standalone Volumetric 3D CNN**, and a novel **Dual-Branch Gated Cross-Modal Hybrid Network** with **3D Grad-CAM** on the LIDC-IDRI thoracic CT cohort.

---

## 1. System Architecture: Dual-Branch with Gated Attention Fusion

Single-modality approaches to pulmonary nodule characterization suffer from fundamental trade-offs:
* **Standalone 3D CNNs** are prone to overfitting on small CT cohorts and struggle to learn exact volumetric scaling rules without external guidance.
* **Classical ML on Radiomics** leverages expert measurements (diameter, calcification, density ratios) but is blind to complex 3D spatial textures, border spiculation, and infiltrative boundaries.

Our **Dual-Branch Hybrid Network (`HybridNoduleNet`)** synergistically fuses both modalities using a learned **Gated Attention Mechanism**:

```text
[3D CT Patch: 1×32×32×32]                [15 Tabular Radiomic Features]
           │                                            │
           ▼                                            ▼
 [Branch 1: 3D ResNet-18]                   [Branch 2: Tabular MLP]
 (Adapted 1-channel 3D stem)              (Standardized + BatchNorm + GELU)
           │                                            │
    z_visual (128-dim)                          z_tabular (64-dim)
           │                                            │
           └─────────────────────┬──────────────────────┘
                                 ▼
               [Gated Cross-Modal Fusion Head]
                 gate = σ(W_g [z_vis, z_tab] + b)
                 z_fused = gate ⊙ z_vis + (1 - gate) ⊙ Proj(z_tab)
                                 │
                                 ▼
                     [Classification MLP Head]
                         (Linear + Dropout)
                                 │
                                 ▼
                     [Binary Malignancy Logit]


