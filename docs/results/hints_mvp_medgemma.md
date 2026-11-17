# Results: hints_mvp_medgemma

82 of 100 images scored (37 with a finding, 45 normal); an image is dropped from every condition if any condition's output failed.
Reader: google/medgemma-1.5-4b-it via transformers (thinking None, effort None). Labels: the report, labeled by claude-sonnet-5-5.
Scored over 13 findings (No Finding left out), pooled over images x findings. 95% CIs: paired bootstrap over images, 1000 resamples, seed 0.

## Changes vs no_hint (uncertain counts as positive)

no_hint: 123 positive labels, of which it found 93 and missed 30; 943 negative labels, of which it called 105 (false findings). no_hint_rerun is an identical second request: its changes are chance.

| Condition | fixed (missed→found) | broken (found→missed) | new false findings | false findings cleared | recovery rate | break rate |
|---|---|---|---|---|---|---|
| no_hint_rerun | 8 | 12 | 43 | 44 | 0.267 [0.107, 0.396] | 0.129 [0.060, 0.205] |
| own_trail | 5 | 14 | 44 | 35 | 0.167 [0.038, 0.321] | 0.151 [0.075, 0.235] |
| other_trail | 6 | 13 | 46 | 34 | 0.200 [0.050, 0.391] | 0.140 [0.061, 0.229] |
| random_trail | 7 | 13 | 47 | 40 | 0.233 [0.071, 0.381] | 0.140 [0.060, 0.219] |

**Metrics (uncertain counts as positive)**

| Condition | Sensitivity | Specificity | F1 | Δ sensitivity vs no_hint | Δ specificity | Δ F1 |
|---|---|---|---|---|---|---|
| no_hint | 0.756 [0.674, 0.837] | 0.889 [0.858, 0.916] | 0.579 [0.500, 0.648] |  |  |  |
| no_hint_rerun | 0.724 [0.633, 0.804] | 0.890 [0.861, 0.916] | 0.563 [0.492, 0.623] | -0.033 [-0.111, +0.050] | +0.001 [-0.027, +0.032] | -0.016 [-0.073, +0.045] |
| own_trail | 0.683 [0.579, 0.783] | 0.879 [0.846, 0.910] | 0.523 [0.443, 0.587] | -0.073 [-0.148, -0.007] | -0.010 [-0.037, +0.018] | -0.056 [-0.108, -0.003] |
| other_trail | 0.699 [0.600, 0.794] | 0.876 [0.842, 0.907] | 0.528 [0.451, 0.597] | -0.057 [-0.128, +0.012] | -0.013 [-0.038, +0.011] | -0.052 [-0.102, +0.002] |
| random_trail | 0.707 [0.619, 0.802] | 0.881 [0.851, 0.908] | 0.540 [0.465, 0.601] | -0.049 [-0.129, +0.034] | -0.007 [-0.033, +0.017] | -0.039 [-0.093, +0.016] |
| leak_check | 0.154 [0.066, 0.255] | 0.928 [0.880, 0.969] | 0.181 [0.092, 0.258] |  |  |  |

leak_check (own trail, no image): sensitivity 0.154 [0.066, 0.255], specificity 0.928 [0.880, 0.969]. Youden's J within findings (sensitivity + specificity − 1 per finding, averaged; 0 = the coordinates alone say nothing about which images have a finding): +0.015 [-0.039, +0.083].

## Changes vs no_hint (uncertain labels left out)

no_hint: 104 positive labels, of which it found 82 and missed 22; 943 negative labels, of which it called 105 (false findings). no_hint_rerun is an identical second request: its changes are chance.

| Condition | fixed (missed→found) | broken (found→missed) | new false findings | false findings cleared | recovery rate | break rate |
|---|---|---|---|---|---|---|
| no_hint_rerun | 6 | 7 | 43 | 44 | 0.273 [0.095, 0.440] | 0.085 [0.029, 0.141] |
| own_trail | 3 | 10 | 44 | 35 | 0.136 [0.000, 0.269] | 0.122 [0.056, 0.194] |
| other_trail | 5 | 10 | 46 | 34 | 0.227 [0.040, 0.429] | 0.122 [0.056, 0.198] |
| random_trail | 5 | 6 | 47 | 40 | 0.227 [0.062, 0.417] | 0.073 [0.022, 0.127] |

**Metrics (uncertain labels left out)**

| Condition | Sensitivity | Specificity | F1 | Δ sensitivity vs no_hint | Δ specificity | Δ F1 |
|---|---|---|---|---|---|---|
| no_hint | 0.788 [0.714, 0.869] | 0.889 [0.858, 0.916] | 0.564 [0.486, 0.635] |  |  |  |
| no_hint_rerun | 0.779 [0.691, 0.862] | 0.890 [0.861, 0.916] | 0.561 [0.488, 0.623] | -0.010 [-0.075, +0.055] | +0.001 [-0.027, +0.032] | -0.003 [-0.061, +0.058] |
| own_trail | 0.721 [0.631, 0.817] | 0.879 [0.846, 0.910] | 0.512 [0.438, 0.577] | -0.067 [-0.128, -0.009] | -0.010 [-0.037, +0.018] | -0.052 [-0.106, +0.002] |
| other_trail | 0.740 [0.647, 0.832] | 0.876 [0.842, 0.907] | 0.517 [0.434, 0.590] | -0.048 [-0.106, +0.010] | -0.013 [-0.038, +0.011] | -0.047 [-0.095, +0.007] |
| random_trail | 0.779 [0.678, 0.874] | 0.881 [0.851, 0.908] | 0.545 [0.468, 0.608] | -0.010 [-0.063, +0.052] | -0.007 [-0.033, +0.017] | -0.018 [-0.068, +0.030] |
| leak_check | 0.183 [0.080, 0.294] | 0.928 [0.880, 0.969] | 0.199 [0.100, 0.282] |  |  |  |

leak_check (own trail, no image): sensitivity 0.183 [0.080, 0.294], specificity 0.928 [0.880, 0.969]. Youden's J within findings (sensitivity + specificity − 1 per finding, averaged; 0 = the coordinates alone say nothing about which images have a finding): +0.068 [-0.026, +0.186].

## Per finding (uncertain counts as positive)

Sensitivity / specificity per condition. Findings with few positives are noise.

| Finding | positives | no_hint | no_hint_rerun | own_trail | other_trail | random_trail | leak_check |
|---|---|---|---|---|---|---|---|
| Enlarged Cardiomediastinum | 1 | 1.00 / 0.95 | 0.00 / 0.89 | 0.00 / 0.85 | 1.00 / 0.90 | 1.00 / 0.91 | 0.00 / 0.89 |
| Cardiomegaly | 9 | 1.00 / 0.81 | 1.00 / 0.82 | 1.00 / 0.82 | 1.00 / 0.84 | 1.00 / 0.77 | 0.11 / 0.92 |
| Lung Opacity | 21 | 0.86 / 0.74 | 0.95 / 0.75 | 0.86 / 0.75 | 0.90 / 0.72 | 0.90 / 0.75 | 0.14 / 0.87 |
| Lung Lesion | 4 | 0.50 / 1.00 | 0.25 / 0.99 | 0.25 / 1.00 | 0.25 / 0.97 | 0.50 / 0.99 | 0.00 / 0.91 |
| Edema | 18 | 0.56 / 0.89 | 0.50 / 0.86 | 0.44 / 0.95 | 0.39 / 0.91 | 0.44 / 0.92 | 0.22 / 0.97 |
| Consolidation | 8 | 0.50 / 0.86 | 0.38 / 0.93 | 0.62 / 0.82 | 0.75 / 0.86 | 0.38 / 0.85 | 0.12 / 0.93 |
| Pneumonia | 2 | 1.00 / 0.88 | 0.50 / 0.91 | 0.50 / 0.78 | 0.50 / 0.78 | 0.50 / 0.82 | 0.00 / 0.93 |
| Atelectasis | 9 | 0.78 / 0.75 | 0.78 / 0.71 | 0.56 / 0.77 | 0.67 / 0.77 | 0.89 / 0.77 | 0.22 / 0.95 |
| Pneumothorax | 2 | 0.50 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 0.95 |
| Pleural Effusion | 15 | 0.67 / 0.97 | 0.67 / 0.93 | 0.53 / 0.97 | 0.47 / 0.96 | 0.47 / 0.96 | 0.07 / 0.96 |
| Pleural Other | 2 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 0.99 | 0.00 / 1.00 | 0.00 / 0.95 |
| Fracture | 4 | 0.25 / 0.99 | 0.25 / 1.00 | 0.25 / 1.00 | 0.25 / 0.99 | 0.25 / 1.00 | 0.00 / 0.95 |
| Support Devices | 28 | 1.00 / 0.57 | 1.00 / 0.65 | 1.00 / 0.61 | 1.00 / 0.59 | 1.00 / 0.59 | 0.25 / 0.89 |

## Reliability

| Condition | tasks | first-try parse failures | final failures | refusals | API errors | cost (all requests) |
|---|---|---|---|---|---|---|
| no_hint | 100 | 20 | 3 | 0 | 0 | $0.0 |
| no_hint_rerun | 100 | 19 | 3 | 0 | 0 | $0.0 |
| own_trail | 100 | 11 | 1 | 0 | 0 | $0.0 |
| other_trail | 100 | 15 | 3 | 0 | 0 | $0.0 |
| random_trail | 100 | 12 | 3 | 0 | 0 | $0.0 |
| leak_check | 100 | 34 | 11 | 0 | 0 | $0.0 |

Cost of all requests in this run: $0.0; everything hints_mvp has spent so far: $0.0.

## Observations

- **No trail helps MedGemma.** Every trail condition lowered sensitivity a little: own_trail −0.073 [−0.148, −0.007], other_trail −0.057, random_trail −0.049 (uncertain counted as positive). The noise floor (no_hint_rerun) is −0.033 [−0.111, +0.050], and own, other and random trails overlap. So whatever effect there is comes from adding trail text, not from where the radiologist looked. own_trail is the only CI that excludes 0, and only barely, among 4 conditions × 2 label rules, so it is weak evidence.
- **Unlike Sonnet, the trail doesn't cut false findings.** New false findings were 44–47 with a trail vs 43 for the rerun, and false findings cleared were 34–40 vs 44 for the rerun; Δ specificity −0.007 to −0.013. Sonnet's drop in false findings on normal images did not carry over.
- **The noise floor is large.** An identical rerun (sampling at temperature 1.0) fixed 8 misses, broke 12 true calls, added 43 false findings and cleared 44. Most changes in any condition are of this size, so per-finding differences mean little.
- **Baseline:** no_hint sensitivity 0.756 [0.674, 0.837] and specificity 0.889, vs Sonnet's 0.828 sensitivity. These are not the same images: 82 complete-case images here vs Sonnet's scored set. MedGemma calls Support Devices on most images (specificity 0.57).
- **Without the image, the coordinates say nothing:** leak_check within-finding J +0.015 [−0.039, +0.083] (Sonnet: +0.047 [+0.013, +0.093]). leak_check also has the most failures (11 of 100), because MedGemma walks through the coordinates one by one and runs out of tokens.
- **Run notes:** MedGemma 1.5 thinks before answering (`<unused94>thought … <unused95>`). The parser now reads only the text after the thinking, and max_tokens was raised from 2000 to 4000, because thinking plus the report cut off 2 of 10 smoke-test answers. 24 of 600 reports failed after one retry, spread over 18 images, so 82 images are scored. The run took 14.2 h on the RTX 3090.
