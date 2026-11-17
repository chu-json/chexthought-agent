# trail_anatomy results: maira

300 of 300 test images scored (every condition produced a report). Reference: CheXbert labels of each image's report. Prediction: CheXbert labels of the generated findings (present or uncertain = positive call). 13 findings pooled over images x findings; 95% paired bootstrap CIs.

## All scored images (300); uncertain reference labels count as positive

693 positive reference labels.

| Condition | fixed | broken | new false | cleared | Sensitivity | Δ sens vs no_hint | Specificity | Δ spec | Δ F1 |
|---|---|---|---|---|---|---|---|---|---|
| **no_hint** | | | | | 0.582 [0.545, 0.617] | | 0.894 [0.879, 0.908] | | |
| neutral_text | 55 | 139 | 105 | 234 | 0.460 | -0.121 [-0.156, -0.088] | 0.934 | +0.040 [+0.028, +0.052] | -0.040 [-0.071, -0.009] |
| focus_text_own | 39 | 163 | 87 | 268 | 0.403 | -0.179 [-0.214, -0.144] | 0.950 | +0.056 [+0.044, +0.069] | -0.068 [-0.100, -0.031] |
| focus_text_other | 44 | 166 | 83 | 274 | 0.405 | -0.176 [-0.212, -0.139] | 0.953 | +0.060 [+0.046, +0.072] | -0.061 [-0.096, -0.026] |
| focus_text_random | 42 | 173 | 78 | 276 | 0.392 | -0.189 [-0.226, -0.148] | 0.955 | +0.062 [+0.049, +0.075] | -0.070 [-0.106, -0.031] |
| report_regions | 47 | 162 | 85 | 276 | 0.416 | -0.166 [-0.204, -0.127] | 0.953 | +0.060 [+0.047, +0.073] | -0.052 [-0.089, -0.014] |

## All scored images (300); uncertain reference labels left out

555 positive reference labels.

| Condition | fixed | broken | new false | cleared | Sensitivity | Δ sens vs no_hint | Specificity | Δ spec | Δ F1 |
|---|---|---|---|---|---|---|---|---|---|
| **no_hint** | | | | | 0.632 [0.592, 0.671] | | 0.894 [0.879, 0.908] | | |
| neutral_text | 41 | 106 | 105 | 234 | 0.515 | -0.117 [-0.157, -0.075] | 0.934 | +0.040 [+0.028, +0.052] | -0.020 [-0.054, +0.015] |
| focus_text_own | 33 | 124 | 87 | 268 | 0.468 | -0.164 [-0.200, -0.124] | 0.950 | +0.056 [+0.044, +0.069] | -0.030 [-0.062, +0.006] |
| focus_text_other | 39 | 127 | 83 | 274 | 0.474 | -0.159 [-0.199, -0.117] | 0.953 | +0.060 [+0.046, +0.072] | -0.020 [-0.056, +0.014] |
| focus_text_random | 36 | 134 | 78 | 276 | 0.456 | -0.177 [-0.217, -0.134] | 0.955 | +0.062 [+0.049, +0.075] | -0.031 [-0.070, +0.006] |
| report_regions | 38 | 117 | 85 | 276 | 0.490 | -0.142 [-0.185, -0.103] | 0.953 | +0.060 [+0.047, +0.073] | -0.006 [-0.044, +0.030] |

## Images whose own path has a focus region (147); uncertain reference labels count as positive

406 positive reference labels.

| Condition | fixed | broken | new false | cleared | Sensitivity | Δ sens vs no_hint | Specificity | Δ spec | Δ F1 |
|---|---|---|---|---|---|---|---|---|---|
| **no_hint** | | | | | 0.584 [0.539, 0.627] | | 0.885 [0.865, 0.904] | | |
| neutral_text | 33 | 81 | 56 | 113 | 0.466 | -0.118 [-0.163, -0.074] | 0.923 | +0.038 [+0.020, +0.055] | -0.049 [-0.088, -0.007] |
| focus_text_own | 22 | 93 | 58 | 133 | 0.409 | -0.175 [-0.222, -0.133] | 0.935 | +0.050 [+0.031, +0.069] | -0.085 [-0.129, -0.042] |
| focus_text_other | 27 | 97 | 48 | 132 | 0.411 | -0.172 [-0.215, -0.128] | 0.941 | +0.056 [+0.038, +0.074] | -0.076 [-0.117, -0.036] |
| focus_text_random | 25 | 103 | 49 | 141 | 0.392 | -0.192 [-0.241, -0.143] | 0.946 | +0.061 [+0.042, +0.080] | -0.089 [-0.134, -0.042] |
| report_regions | 29 | 96 | 45 | 138 | 0.419 | -0.165 [-0.218, -0.112] | 0.947 | +0.062 [+0.043, +0.080] | -0.063 [-0.111, -0.006] |

## Images whose own path has a focus region (147); uncertain reference labels left out

322 positive reference labels.

| Condition | fixed | broken | new false | cleared | Sensitivity | Δ sens vs no_hint | Specificity | Δ spec | Δ F1 |
|---|---|---|---|---|---|---|---|---|---|
| **no_hint** | | | | | 0.630 [0.580, 0.681] | | 0.885 [0.865, 0.904] | | |
| neutral_text | 24 | 64 | 56 | 113 | 0.506 | -0.124 [-0.174, -0.072] | 0.923 | +0.038 [+0.020, +0.055] | -0.039 [-0.081, +0.006] |
| focus_text_own | 20 | 69 | 58 | 133 | 0.478 | -0.152 [-0.207, -0.104] | 0.935 | +0.050 [+0.031, +0.069] | -0.045 [-0.094, +0.003] |
| focus_text_other | 24 | 73 | 48 | 132 | 0.478 | -0.152 [-0.203, -0.104] | 0.941 | +0.056 [+0.038, +0.074] | -0.037 [-0.079, +0.008] |
| focus_text_random | 23 | 79 | 49 | 141 | 0.457 | -0.174 [-0.231, -0.118] | 0.946 | +0.061 [+0.042, +0.080] | -0.047 [-0.094, +0.001] |
| report_regions | 22 | 68 | 45 | 138 | 0.488 | -0.143 [-0.198, -0.089] | 0.947 | +0.062 [+0.043, +0.080] | -0.020 [-0.068, +0.030] |

## Sensitivity per finding (uncertain = positive)

| Finding | positives | no_hint | neutral_text | focus_text_own | focus_text_other | focus_text_random | report_regions |
|---|---|---|---|---|---|---|---|
| Enlarged Cardiomediastinum | 28 | 0.18 | 0.07 | 0.00 | 0.00 | 0.00 | 0.11 |
| Cardiomegaly | 48 | 0.71 | 0.88 | 0.69 | 0.62 | 0.67 | 0.71 |
| Lung Opacity | 112 | 0.52 | 0.31 | 0.28 | 0.28 | 0.24 | 0.33 |
| Lung Lesion | 8 | 0.12 | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 |
| Edema | 83 | 0.48 | 0.27 | 0.17 | 0.17 | 0.19 | 0.22 |
| Consolidation | 39 | 0.08 | 0.10 | 0.05 | 0.10 | 0.08 | 0.00 |
| Pneumonia | 27 | 0.56 | 0.07 | 0.00 | 0.00 | 0.00 | 0.00 |
| Atelectasis | 71 | 0.58 | 0.34 | 0.18 | 0.24 | 0.20 | 0.21 |
| Pneumothorax | 22 | 0.27 | 0.23 | 0.36 | 0.45 | 0.32 | 0.27 |
| Pleural Effusion | 96 | 0.84 | 0.65 | 0.59 | 0.55 | 0.55 | 0.59 |
| Pleural Other | 9 | 0.11 | 0.11 | 0.11 | 0.11 | 0.00 | 0.00 |
| Fracture | 9 | 0.00 | 0.00 | 0.11 | 0.00 | 0.22 | 0.22 |
| Support Devices | 141 | 0.84 | 0.85 | 0.84 | 0.86 | 0.84 | 0.82 |

## Does the report name the hinted regions?

Share of images with a non-empty hint whose generated report names at least one hinted region, vs the no_hint report naming the same regions.

| Condition | images | names a hinted region | no_hint names them |
|---|---|---|---|
| focus_text_own | 147 | 0.61 | 0.20 |
| focus_text_other | 147 | 0.65 | 0.16 |
| focus_text_random | 147 | 0.70 | 0.09 |
| report_regions | 227 | 0.85 | 0.73 |

Truncated outputs (hit max tokens) per condition: {'no_hint': 0, 'neutral_text': 0, 'focus_text_own': 1, 'focus_text_other': 0, 'focus_text_random': 1, 'report_regions': 0}
