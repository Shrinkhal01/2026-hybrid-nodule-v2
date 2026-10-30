# phase7_R: explanations scored on the test nodules, mean over seeds

Concentration = share of the map on the nodule / share of the cube the nodule fills (1.00 = flat map). Hit = hottest voxel on the nodule (±2 mm). The **centre prior** is the trivial 'always look at the centre' map; it is the bar a real explanation should approach.

| features | model | method | seeds | concentration (median per seed) | hit rate | centre-of-mass distance | all-zero maps |
|:--|:--|:--|:-:|:-:|:-:|:-:|:-:|
| all | B: 3D ResNet | gradcam_layer4 | 5 | 1.25 ± 0.02 | 15% ± 0% | 3.3 mm | 1% |
| all | B: 3D ResNet | gradcam_layer3 | 5 | 1.82 ± 0.46 | 42% ± 13% | 5.8 mm | 6% |
| all | B: 3D ResNet | gradcam_layer2 | 5 | 1.12 ± 0.57 | 27% ± 11% | 5.4 mm | 5% |
| all | B: 3D ResNet | occlusion_8mm | 5 | 2.55 ± 1.15 | 52% ± 4% | 2.6 mm | 0% |
| all | B: 3D ResNet | random_weights_layer3 | 5 | 0.97 ± 0.01 | 11% ± 1% | 6.2 mm | 0% |
| all | B: 3D ResNet | centre_prior | 5 | 8.40 ± 0.00 | 100% ± 0% | 0.0 mm | 0% |
| all | C: Hybrid | gradcam_layer4 | 5 | 1.23 ± 0.03 | 15% ± 0% | 3.0 mm | 1% |
| all | C: Hybrid | gradcam_layer3 | 5 | 1.83 ± 0.44 | 38% ± 6% | 5.7 mm | 4% |
| all | C: Hybrid | gradcam_layer2 | 5 | 1.07 ± 0.32 | 25% ± 9% | 5.1 mm | 0% |
| all | C: Hybrid | occlusion_8mm | 5 | 2.24 ± 1.13 | 52% ± 7% | 2.5 mm | 0% |
| all | C: Hybrid | random_weights_layer3 | 5 | 1.11 ± 0.07 | 27% ± 1% | 6.6 mm | 0% |
| all | C: Hybrid | centre_prior | 5 | 8.40 ± 0.00 | 100% ± 0% | 0.0 mm | 0% |
| image | B: 3D ResNet | gradcam_layer4 | 5 | 1.25 ± 0.03 | 15% ± 0% | 3.6 mm | 2% |
| image | B: 3D ResNet | gradcam_layer3 | 5 | 1.88 ± 0.55 | 40% ± 7% | 5.7 mm | 8% |
| image | B: 3D ResNet | gradcam_layer2 | 5 | 1.02 ± 0.38 | 29% ± 16% | 5.7 mm | 0% |
| image | B: 3D ResNet | occlusion_8mm | 5 | 1.84 ± 1.01 | 48% ± 6% | 2.6 mm | 0% |
| image | B: 3D ResNet | random_weights_layer3 | 5 | 0.97 ± 0.02 | 11% ± 1% | 6.4 mm | 0% |
| image | B: 3D ResNet | centre_prior | 5 | 8.40 ± 0.00 | 100% ± 0% | 0.0 mm | 0% |
| image | C: Hybrid | gradcam_layer4 | 5 | 1.28 ± 0.02 | 15% ± 0% | 2.8 mm | 1% |
| image | C: Hybrid | gradcam_layer3 | 5 | 1.77 ± 0.67 | 44% ± 5% | 5.3 mm | 11% |
| image | C: Hybrid | gradcam_layer2 | 5 | 0.93 ± 0.28 | 25% ± 9% | 5.5 mm | 1% |
| image | C: Hybrid | occlusion_8mm | 5 | 1.83 ± 0.91 | 50% ± 5% | 2.6 mm | 0% |
| image | C: Hybrid | random_weights_layer3 | 5 | 1.08 ± 0.00 | 18% ± 1% | 7.3 mm | 0% |
| image | C: Hybrid | centre_prior | 5 | 8.40 ± 0.00 | 100% ± 0% | 0.0 mm | 0% |

## Sanity check (Grad-CAM layer3, trained vs random weights)

Rank correlation near 1 = the map hardly depends on what the network learned (bad); near 0 = it does.

| features | model | correlation, mean ± SD over seeds |
|:--|:--|:-:|
| all | B: 3D ResNet | 0.13 ± 0.04 |
| all | C: Hybrid | 0.04 ± 0.09 |
| image | B: 3D ResNet | 0.16 ± 0.09 |
| image | C: Hybrid | -0.02 ± 0.09 |

## Which input does the model rely on? (shuffle it between nodules)

| features | model | shuffled | AUC original | AUC shuffled | AUC drop, mean ± SD over seeds |
|:--|:--|:--|:-:|:-:|:-:|
| all | B: 3D ResNet | image | 0.90 | 0.49 | +0.403 ± 0.013 |
| all | C: Hybrid | image | 0.91 | 0.59 | +0.319 ± 0.040 |
| all | C: Hybrid | table | 0.91 | 0.88 | +0.034 ± 0.034 |
| image | B: 3D ResNet | image | 0.90 | 0.50 | +0.398 ± 0.008 |
| image | C: Hybrid | image | 0.90 | 0.51 | +0.386 ± 0.015 |
| image | C: Hybrid | table | 0.90 | 0.90 | -0.001 ± 0.001 |

## Hybrid: table features by permutation importance (top 5 per feature set)

| features | feature | AUC drop when shuffled, mean ± SD over seeds |
|:--|:--|:-:|
| all | calcification | +0.014 ± 0.010 |
| all | diameter_mm | +0.010 ± 0.007 |
| all | texture | +0.005 ± 0.004 |
| all | max_hu | +0.001 ± 0.001 |
| all | min_hu | +0.001 ± 0.001 |
| image | min_hu | +0.000 ± 0.000 |
| image | hyperdense_fraction_pct | +0.000 ± 0.000 |
| image | max_hu | +0.000 ± 0.000 |
| image | median_hu | +0.000 ± 0.000 |
| image | iqr_hu | +0.000 ± 0.000 |

The ball uses the radiologists' mean diameter only to draw the target; no model saw it for this test.
Data: LIDC-IDRI, TCIA DOI 10.7937/K9/TCIA.2015.LO9QL9SX.
