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

## Step P: patient IDs from pylidc + patient-level leakage check

| Used | Why | Rejected alternative | Number that justified it |
|---|---|---|---|
| **pylidc's bundled database** (`pylidc.sqlite`, table `scans`) for SeriesInstanceUID → PatientID | the LUNA22 metadata has **no PatientID**; pylidc maps every LIDC scan to its patient without downloading DICOMs → also meets the "use pylidc" ground rule | TCIA metadata CSV download; keep series-level only | 1,018 scans → 1,010 patients in the database; your 546 series → **545 patients**, **1** with 2 series, **0** patients across splits (one-off check with pylidc 0.2.3; the pipeline now logs this on every run) |
| **Read the SQLite file directly**, don't `import pylidc` | the database is all we need; importing pylidc also loads DICOM/plot code it doesn't need here | `pylidc.query(pl.Scan)` | NOT MEASURED (whether `import pylidc` works with your numpy/SQLAlchemy versions) |
| **Unknown series = error**, not a blank | a silent gap would make the patient check pass while skipping rows | fill missing with the series UID | `test_unknown_series_is_an_error_not_a_gap` |
| **Keep the existing split** (`--keep-split`) after adding patient IDs | 0 patients straddle, so the split is already patient-disjoint; re-drawing would change the test set for nothing | re-split by patient | 545 vs 546 groups: a new `GroupShuffleSplit` draw would give a different test set |
| **Verification fails** if a patient is in two splits or a row has no patient_id | turns "we checked once" into a check that runs every time | a log message only | `test_same_patient_in_train_and_test_fails`: two different scans of one patient in train + test → FAILED, although every series is in one split only |

### Break-it exercise
- In `tests/test_patient_ids.py`, change the second row of `test_clean_patient_split_passes` from `"train"` to `"test"`. Predict first: does it pass or fail, and which check message appears?

## Step A: ablation (no radiologist ratings) + several seeds

Numbers: Run 1 (`results/phase7_SP`), `scripts/aggregate_runs.py` reference rows on the real
test set, and a CPU end-to-end run of `scripts/train_on_gpu.sh` on 26 sample nodules (mechanics only).

| Used | Why | Rejected alternative | Number that justified it |
|---|---|---|---|
| **`--features image`** = the 12 CT-measured features; drops diameter, calcification, texture | the 3 ratings come from the **same radiologists who gave the label** → we must see what the model does without them | drop only diameter; add noise to ratings | Run 1: **diameter alone** test AUC **0.941**; logistic regression on the **3 ratings** **0.969** [0.941, 0.991] > Hybrid 0.912 |
| Same split, same patches, same code; only the tabular columns change | one change at a time → the difference is caused by the ratings | new split per feature set | Hybrid weights 33,259,777 → 33,259,585 (**−192 = 3 × 64**, the first table layer only) |
| **5 seeds** (42–46) per feature set | one seed can't separate a real change from luck | 1 seed (Run 1); 10+ seeds | Run 1 val AUC jumps **±0.02 between epochs**; Run 1 vs Part 2 Hybrid change −0.025 was **not significant** (p = 0.16); one training run = **39 s** |
| **Seed ensemble** (average probability over the 5 seeds) + paired bootstrap on it | uses all 5 runs on the same 151 nodules → tighter, fair comparison | compare only mean ± SD | NOT MEASURED until Run 2 |
| **ResNet as a control row** in the ablation table | it uses no tabular features → its all − image difference must be ~0; anything else = GPU run-to-run noise | no control | CPU end-to-end test: **+0.000** (identical); on the GPU: NOT MEASURED (cuDNN may not be bit-exact) |
| **Reference models** with no training randomness (diameter alone, logistic regression on 3 ratings, logistic regression on 12 image features) | a floor and a ceiling for the deep models | none | logistic regression on **12 image features**: **0.641** [0.550, 0.729] → the hand-made radiomics carry little linear signal |
| Resume: a finished `seed_*` folder is skipped | a crash or SSH drop doesn't cost the finished runs | start over | n/a |

### Limits still open after step A
- The **image** set still has whole-crop radiomics (no nodule mask) → weak features (0.641) may undersell "image-only".
- The app (`app/streamlit_app.py`) only loads **15-feature** checkpoints → point `NODULE_RESULTS_DIR` at an `all/seed_*` folder.
- Calibration (Hybrid overconfident, Brier 0.151 vs RF 0.096 in Run 1) is not addressed yet.

### Break-it exercise
- Run `python scripts/aggregate_runs.py results/phase7_A` after deleting one `image/seed_*` folder's `test_predictions.csv`. Predict first: does it crash, skip, or silently use 4 seeds? Which line in `load_run()` decides?
