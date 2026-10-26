# Hybrid Explainable AI (XAI) for 3D Lung Nodule Malignancy Prediction

An end-to-end framework combining **Volumetric 3D Deep Learning** and **Classical Radiomics** with **3D Explainability (Grad-CAM)** on the thoracic CT LIDC-IDRI benchmark.

---

## 📅 Day 1 Milestone: Dataset Curation, Radiomics & Leakage-Free Splitting

Day 1 focused on building a clean, reproducible, and leakage-free data processing pipeline from the LIDC-IDRI 1,176 nodule benchmark.

### 1. Data Processing Highlights
* **Raw Ingestion:** Ingested 1,176 3D CT NIfTI sub-volumes (`128 × 128 × 64`) and consensus metadata.
* **Consensus Exclusion:** Filtered out 176 indeterminate nodules (mean radiologist score $\bar{M} = 3.0$), yielding **1,000 clinically unambiguous nodules** (554 Benign, 446 Malignant).
* **Volumetric Preprocessing:**
  * Reoriented to PyTorch depth-first format: `(Depth, Height, Width)`.
  * Extracted central `32 × 32 × 32` voxel Regions of Interest (ROI).
  * Applied lung windowing (`[-1000, 400] HU`) and min-max scaled to `[0.0, 1.0]`.
* **15 Tabular Radiomic Features:**
  * **10 Intensity Statistics:** Mean, Std, Min, Max, Median, 25th percentile, 75th percentile, IQR, Skewness, Kurtosis.
  * **2 Tissue Density Ratios:** Solid fraction ($> -400\text{ HU}$) and Hyperdense/calcification fraction ($> 100\text{ HU}$).
  * **3 Radiologist Consensus Metrics:** Mean diameter (mm), mean calcification score, mean texture score.

---

## 📊 Dataset Distribution & Leakage-Free Split

Splitting was performed using `GroupShuffleSplit` strictly grouped by `SeriesInstanceUID` across 546 unique CT series. Evaluated over 100 random seeds (seed 60 selected) to achieve class balance across all partitions while maintaining zero patient/scan leakage:

| Split | Nodules | Benign (0) | Malignant (1) | Malignant % | Scan Groups | Split Share |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 703 | 389 | 314 | **44.67%** | 382 | 70.3% |
| **Validation** | 146 | 82 | 64 | **43.84%** | 82 | 14.6% |
| **Test (Held-out)**| 151 | 83 | 68 | **45.03%** | 82 | 15.1% |
| **Total** | **1,000** | **554** | **446** | **44.60%** | **546** | **100.0%** |

$$\text{Group Disjointness Verified: } \text{Train} \cap \text{Val} \cap \text{Test} = \emptyset$$

---

## 📁 Repository Directory Structure

```text
2026-hybrid-nodule-v2/
├── data/
│   ├── raw/
│   │   ├── LIDC-IDRI/              # 1,176 raw NIfTI sub-volumes
│   │   └── LIDC-IDRI_1176.npy      # Raw radiologist annotations
│   └── processed/
│       ├── patches/                # 1,000 normalized 32×32×32 .npy arrays
│       ├── train.csv               # 703 training cases
│       ├── val.csv                 # 146 validation cases
│       ├── test.csv                # 151 benchmark test cases
│       ├── metadata.csv            # Unified audit trail
│       └── dataset_summary.json    # JSON split and feature audit
├── 01_data_pipeline.py             # Reproducible preprocessing script
├── requirements.txt                # Environment dependencies
└── README.md                       # Documentation