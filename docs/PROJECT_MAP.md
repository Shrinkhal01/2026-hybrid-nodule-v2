# Project map: where am I?

> **Now:** Phase 7 (SPARE), steps **S, P, A, R, E done** -> re-score explanations (plateau fix), write `docs/PHASE_7.md`, then merge.
> **Next phase:** Phase 8 = better model (**BASIC**: Bar, Architecture, Stronger augmentation, Integration, Calibration).
> **Work on branch:** `phase-7-resampling` (not `main`!)
> Read this file first whenever you come back after a break.

---

## 1. The metro line (4 branches, 3 stations)

Picture a **metro line**: each station adds something, and the train never goes backwards.

```
main ─── part-1 ───────── part-2 ───────────── phase-7-resampling ──> (R, E next)
         26 Oct           27 Oct                29-30 Oct
         DATA             MODELS                FIX THE GAPS
         (Day 1)          (Day 2)               (S, P, A)
                          = main stops here
```

- **`main` = Part 2** -> GitHub's front page shows `main`, **because** it is the default branch,
  so Phase 7 is **invisible there** until it is merged.
- **"Part" = branch name; "Phase" = learning-session number.** Phases 1-6 left no notes in
  `docs/`; their code lives in `part-1` + `part-2`. Phase 7 is the first phase with notes.
- Acronym for the stations: **D-M-F** = **D**ata -> **M**odels -> **F**ixes.

| Station | Branch | What it added | Key files |
|---|---|---|---|
| **D** Data | `part-1` | 1,176 LIDC-IDRI nodules -> **1,000** (score 3.0 dropped) -> 32^3 patches, HU window, **15 features**, split 703 / 146 / 151 | `01_data_pipeline.py`, `data/processed/` |
| **M** Models | `part-2` (= `main`) | the **3 models**, training on the H200, Grad-CAM, figures, app | `models/`, `pipelines/02-04`, `app/` |
| **F** Fixes | `phase-7-resampling` | **S** 1 mm resampling, **P** patient IDs, **A** ablation x 5 seeds | `docs/PHASE_7_decisions.md`, `scripts/`, `tests/` |

---

## 2. The "mixture" you decided on = the Hybrid (model C)

Your image: **three students** taking the same exam.

| Student | Reads | Model |
|---|---|---|
| **A** Random Forest | only the **sheet of numbers** (15 features) | `models/baseline_rf.py` |
| **B** 3D ResNet-18 | only the **picture** (32 mm cube) | `models/hybrid_net.py: Pure3DResNet` |
| **C** Hybrid | **picture + sheet**, and a **mixing knob** (gate) decides how much to trust each | `models/hybrid_net.py: HybridNoduleNet` |

- The mixture = **CNN** (picture) **+ MLP** (sheet) **+ gate** (knob): `gate = sigmoid(W [z_picture ; z_sheet])`
  -> `mix = gate x picture + (1 - gate) x sheet`, **because** each nodule can lean on the source it needs.
- A and B are there as **baselines**, **because** a hybrid only means something if it beats both parts alone.

---

## 3. Scoreboard: how the numbers moved (test AUC, 151 nodules)

| Run | Data | RF (A) | ResNet (B) | Hybrid (C) |
|---|---|:-:|:-:|:-:|
| Part 2 | old patches, 1 seed | 0.918 | 0.920 | 0.937 |
| Phase 7 run 1 | 1 mm patches + patient split, 1 seed | 0.944 | 0.892 | 0.912 |
| Phase 7 run 2, **all 15** | 5 seeds, mean +- SD | **0.942** +- 0.004 | 0.894 +- 0.010 | 0.915 +- 0.009 |
| Phase 7 run 2, **image 12** (no radiologist) | 5 seeds | **0.666** +- 0.010 | 0.896 +- 0.005 | 0.904 +- 0.016 |

What it means:
- **RF 0.942 -> 0.666** without the 3 radiologist ratings -> the RF mostly **read the answer key**
  (the same radiologists gave the labels).
- **Hybrid 0.915 -> 0.904** (p = 0.28, within noise) -> the deep models **learn from the CT image**.
- **One seed is not enough:** the same Hybrid run 3 times scored 0.912 / 0.930 / 0.919.
- References with no training: diameter alone **0.941**; logistic regression on the 3 ratings **0.969**.
- Hypothesis, NOT MEASURED: ResNet fell from 0.920 to ~0.89 after resampling, **because** the old
  crop cut off big nodules (all malignant), so "patch full of tissue = cancer" was an easy
  shortcut; 1 mm / 32 mm crops cut far fewer nodules (106 -> 28).

---

## 4. What is left

**Phase 7 (SPARE)**
- [x] **S** Spacing: 1 mm isotropic, 32 mm cube
- [x] **P** Patient check: 545 patients, 0 shared between splits
- [x] **A** Ablation + 5 seeds
- [x] **R** Re-explain: all 151 test nodules scored -> layer4 Grad-CAM = untrained network (hit 15%),
  occlusion best (hit >= 52%, re-score pending after the plateau fix), Hybrid ignores its table
- [x] **E** Extras (MEDIC): Mac MPS fallback, occlusion as default explanation, DOI + honest README,
  imbalance note (calibration moved to Phase 8)
- [ ] `docs/PHASE_7.md`: your own notes, from memory (still empty)
- [ ] Hand-written core: `resample_crop_iso` (spec + tests in `PHASE_7_decisions.md`)

**Extras candidates (E)**
- Calibration: Hybrid over-confident (Brier 0.151 vs RF 0.096 in run 1)
- Nodule-mask features (radiomics still use the whole cube; image-only LR = 0.641)
- Image-measured diameter (rays from the centre; no radiologist input)
- Shuffled-label check (RF with random labels should score ~0.5 -> proves no leak)
- 28 nodules > 32 mm are still cut at the edge

**Bigger open question:** detection in full CT scans -> to decide with the professor.

---

## 5. Git: find your place in 3 commands

```bash
git branch -a                          # which branches exist (* = where you are)
git log --oneline --graph --all -15    # the metro line, newest first
git status                             # anything not committed yet?
```

- Keep working on `phase-7-resampling` until Phase 7 is done.
- At the end of Phase 7: open a **pull request** `phase-7-resampling -> main` on GitHub and merge it,
  **because** then `main` (the front page) shows the real project again.
- Next phase: `git checkout main && git pull && git checkout -b phase-8-<name>`.
