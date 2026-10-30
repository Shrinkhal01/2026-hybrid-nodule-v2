# phase7_A: seeds x feature sets (test n = 151)

## 1. Test AUC over seeds

| features | model | seeds | mean ± SD | min–max | ensemble AUC [95% CI] | mean FN |
|:--|:--|:-:|:-:|:-:|:-:|:-:|
| all | A: Random Forest | 5 | 0.942 ± 0.004 | 0.937–0.948 | 0.943 [0.900, 0.980] | 7.4 |
| all | B: 3D ResNet-18 | 5 | 0.894 ± 0.010 | 0.881–0.907 | 0.901 [0.846, 0.950] | 12.6 |
| all | C: Hybrid Net | 5 | 0.915 ± 0.009 | 0.903–0.925 | 0.924 [0.876, 0.964] | 9.8 |
| image | A: Random Forest | 5 | 0.666 ± 0.010 | 0.659–0.683 | 0.668 [0.573, 0.754] | 31.6 |
| image | B: 3D ResNet-18 | 5 | 0.896 ± 0.005 | 0.888–0.899 | 0.908 [0.854, 0.954] | 9.0 |
| image | C: Hybrid Net | 5 | 0.904 ± 0.016 | 0.891–0.930 | 0.917 [0.865, 0.960] | 9.8 |

## 2. Ablation: AUC(all 15) − AUC(image 12)

| model | per-seed mean ± SD | seeds where 'all' wins | ensemble diff [95% CI] | p |
|:--|:-:|:-:|:-:|:-:|
| A: Random Forest | +0.276 ± 0.012 | 5/5 | +0.276 [+0.197, +0.364] | 0.000 |
| C: Hybrid Net | +0.010 ± 0.015 | 4/5 | +0.008 [-0.006, +0.022] | 0.276 |
| B: 3D ResNet-18 (control: uses no tabular features) | -0.002 ± 0.009 | 2/5 | -0.007 [-0.021, +0.005] | 0.278 |

## 3. Reference models (no training randomness)

| reference | test AUC [95% CI] |
|:--|:-:|
| diameter alone (no fitting) | 0.941 [0.897, 0.974] |
| logistic regression, 3 radiologist ratings | 0.969 [0.941, 0.991] |
| logistic regression, 12 image features | 0.641 [0.550, 0.729] |

Labels are mean radiologist ratings (not biopsy). Data: LIDC-IDRI, TCIA DOI 10.7937/K9/TCIA.2015.LO9QL9SX.
