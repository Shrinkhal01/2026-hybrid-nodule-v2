# Phase 7 decisions log (Claude's working log; your own notes go in PHASE_7.md)

## Step S: resampling to isotropic voxels

Numbers come from `python scripts/phase7_resampling_stats.py` (all 1,000 nodules, metadata only),
`python -m pytest tests/test_resample.py` and a 26-nodule sample run of `01_data_pipeline.py`.

| Used | Why | Rejected alternative | Number that justified it |
|---|---|---|---|
| **Resample to 1.0 mm isotropic** (`--spacing-mm 1.0`) | same mm per voxel for every scanner → shape in the patch = shape in the body | keep native voxels (legacy); 0.7 mm (32 mm would need 46³ → model change) | 10 different slice gaps, **0.45–2.5 mm**; legacy depth coverage **14.4–80 mm**; 12.8 mm nodule at 2.5 mm: **5×16×17 → 12×11×11** voxels |
| **Crop 32 mm cube** (`--crop-mm 32`) | still 32³ voxels → `PATCH_SHAPE`, models, Grad-CAM, app unchanged; little padding | 24 mm (86 nodules cut); 40 mm (12 padded, up to 30%); 48 mm (165 padded) | 32 mm: **4 padded** (12.5% of voxels each), **28 cut** (diameter > 32 mm); legacy: **106 cut** |
| **Trilinear** (`order=1`) | smooth, values stay between neighbours, fast | nearest (`order=0`, staircase edges); cubic (`order=3`, **12× slower**: 49 vs 3.9 ms, one-off timing) | cubic's accuracy gain: NOT MEASURED |
| **One step: `scipy.ndimage.map_coordinates`** on the 32³ target grid | resamples and crops together → only 32,768 points computed | `scipy.ndimage.zoom` on the whole 64×128×128 volume, then crop | **3.2 ms** per nodule (~35 nodules/s incl. file I/O, one-off timing); full-volume zoom time: NOT MEASURED |
| **Centre = (shape−1)/2** = (31.5, 63.5, 63.5) | identical centre to the legacy crop [16:48, 48:80, 48:80] → a 1 mm source gives the legacy crop exactly | shape/2 = (32, 64, 64) | centre of mass of 6 isolated nodules ≈ (32.4, 64.7, 64.5) (one-off check, not in a script): both choices within ~1 voxel; `test_1mm_source_equals_legacy_crop` passes |
| **Pad −1000 HU** where the source ends | air value, same as legacy padding | edge-replicate (`mode="nearest"`) | only **4 / 1,000** patches (the 0.45 mm scans), 2 slices at each end |
| **No anti-aliasing filter** before in-plane downsampling (0.46–0.98 mm → 1.0 mm) | keeps the core function simple; factor is at most ~2.2× | Gaussian prefilter, sigma = (1/s − 1)/2 voxels | NOT MEASURED. To measure: resample a pure-noise volume with and without the prefilter and compare the std of the outputs |
| **`--keep-split metadata.csv`** | old and new models are tested on the **same 151 nodules** → a fair before/after | draw a new split | 26-nodule sample: split identical for 26/26 |
| **`--spacing-mm 0` = legacy mode** | reproduce the old patches for comparisons and break-it runs | delete the old path | legacy re-run vs original features: max difference **1.8e-15** (26 nodules) |

### Limits still open after step S
- Radiomics still use the **whole crop, no nodule mask** → now the same 32 mm cube for every nodule, so the dilution is at least **consistent** (fixed in step C/mask later).
- **28 nodules > 32 mm** are still cut at the edges.
- Upsampling 2.5 mm slices to 1 mm **cannot add detail** → thick-slice nodules stay blurrier in depth.
- Patches are **interpolated** → HU values near edges are averages of neighbours.

### Break-it exercises (write your prediction first, then run)
1. `python scripts/phase7_resampling_stats.py`, then imagine `--crop-mm 40`: how many patches get padded? (answer is in the table the script prints)
2. Set `INTERP_ORDER = 0` in `01_data_pipeline.py`, re-run on a few nodules: what does the coronal view of a 2.5 mm nodule look like?
3. `--spacing-mm 2 --crop-mm 32` → 16³ patches. What breaks first in `pipelines/02_train.py`, and why?

### Hand-written core: `resample_crop_iso(vol, spacing_dhw, spacing_mm=1.0, crop_mm=32.0, pad_value=-1000.0)`
Spec (write your own in `tests/my_resample.py`, without looking at the pipeline code):
- Input: `vol` (D,H,W) float HU array, `spacing_dhw` = mm per voxel per axis.
- N = crop_mm / spacing_mm output voxels per axis.
- Centre c = (shape − 1) / 2 per axis.
- Output voxel k sits at offset (k − (N−1)/2) × spacing_mm mm from the centre.
- Its source index = c + offset / spacing of that axis → trilinear interpolation there.
- Outside [0, shape−1] on any axis → `pad_value`.
- Return (crop float32 (N,N,N), pad_fraction = share of padded voxels).
- Tests: `python -m pytest tests/test_resample.py -v` runs every test against your version too.
- Status: NOT YET WRITTEN
