# trail_anatomy: standardized search paths and anatomy hints

**Question.** hints_mvp gave the reader the radiologist's raw search path (every click, as text) and it didn't help
either Sonnet 5.5 or MedGemma 1.5. Raw paths mix very different styles: some radiologists trace whole structures,
some click a detailed search path, some click sparsely. Does a **standardized** hint help a reader find
findings? That means one comparable across styles, built from where the radiologist spent more attention than
radiologists usually do, and named by anatomy. And does it depend on the hint coming from this image?

Code: [src/trail_anatomy/](../src/trail_anatomy/). Segmentation: CXAS
([ChestXRayAnatomySegmentation](https://github.com/ConstantinSeibold/ChestXRayAnatomySegmentation), 159 classes).

## What the first 100 paths look like (hints_mvp sample, one path per image)

These 100 images were already used for hints_mvp, and are now used to design the heuristics, so they are the
**dev set**. Results are reported on new images only (see Sample).

- **Styles vary a lot.** Points per path: 5 to 334 (median 54.5; 10th–90th percentile 30–99). Median step between
  consecutive clicks: 0.04 to 0.35 of the image width. Short steps mean tracing; long steps mean sparse jumps.
  More points go with shorter steps (r = −0.45).
- **Almost every path visits almost everything.** The median path touches 11 of the 15 anatomical regions, and
  41% touch 12 or more. The heart is touched by 100% of paths; trachea, hila, mediastinum and the upper and mid
  lungs by 90–97%. So "the regions that were clicked" carries almost no information, and a hint has to be about
  *relative* attention.
- **Normal and abnormal images get the same kind of path**: the same median points (54.5), regions touched (11)
  and step size.
- **Relative attention carries some signal, tangled with devices.** Exploratory, 100 images. A region's click
  share relative to the average path, AUC for positive vs negative images: Cardiomegaly / heart 0.70 (p 0.03);
  Support Devices / trachea + mediastinum + outside anatomy 0.68 (p 0.003). Some go the other way: Lung
  Opacity / mid and lower lungs 0.23 and Edema / hila and mid lungs 0.32. A likely reason is that radiologists
  follow lines and tubes, which takes clicks away from the lungs on abnormal (often device-laden) films.

## Anatomy partition

CXAS masks overlap (the lungs are projected through the heart), so every pixel gets the first region in this
order whose CXAS classes cover it: trachea, hila (CXAS pulmonary artery, the closest it has), heart,
mediastinum, right/left upper lung (apical + upper zone), mid lung, lower lung, right/left hemidiaphragm,
clavicles, spine, abdomen, other. Sides are the patient's. CXAS stretches images to 512 x 512 without cropping,
so a normalized click maps straight onto the masks. Checked on overlays: clicks land on the trachea, the
costophrenic angles and the cardiac borders, and the patient's right lung is on the image's left in all 100 images.

CXAS has no costophrenic-angle or hilum class. The lower lung zones plus the hemidiaphragms stand in for the
costophrenic angles, and the pulmonary arteries for the hila.

## Standardizing the path: focus regions

For path *i* with *c_ir* clicks in region *r* (*n_i* in total):
1. **Baseline:** *b_r* = the average click share of region *r* over all other paths (leave-one-out). This is how
   radiologists usually spread their clicks.
2. **Shrunk share:** *q_ir* = (*c_ir* + α·*b_r*) / (*n_i* + α). A sparse path (few clicks) is pulled toward the
   baseline, so 2 clicks out of 5 don't become a strong focus.
3. **Enrichment:** *e_ir* = *q_ir* / *b_r*.
4. **Focus regions:** up to K regions with *e_ir* ≥ τ and *c_ir* ≥ 3, highest enrichment first; "other" is
   never one. A path that spreads evenly over everything has no focus region; that is the answer to "clicks
   in every anatomy".

On the dev set, α = 10, τ = 1.5, K = 3 gives 0 / 1 / 2 / 3 focus regions for 6 / 36 / 30 / 28 paths. Most often
these are the left lower lung, left mid lung, hila, spine, right lower lung and mediastinum. α, τ and K are
tuned on the dev set only. Two things are compared there before the final run:
- a segmentation-free alternative: density hotspots, i.e. the path's kernel density minus the average path's,
  top peaks named by the region under them;
- a device-aware variant: drop clicks along a line or tube before counting. This needs device masks, which
  CXAS doesn't have, so it may not be possible.

The style is logged for every path (points, path length, median step, grid coverage, regions touched), so
results can be broken down by style.

## Predictors

| Model | Why | Hint channel | Fits on the 3090? | Status |
|---|---|---|---|---|
| **CheXOne** (StanfordAIMI, Qwen2.5-VL-3B, CC-BY-NC) | Instruction-following CXR model with reasoning mode; on ReXrank (CheXOne-R1: RRG 1/RadCliQ 1.056 on ReXGradient, #2 on ReXVQA) | free text in the prompt, a second image | yes, 7.6 GB weights; vLLM batched 1.8 s/request | **primary**; pilot done |
| **MedGemma 1.5 4B** | Same pipeline as hints_mvp, so a direct before/after; on ReXrank (1/RadCliQ 0.921) | free text, a second image | yes (~9 GB) | rerun via vLLM, since one at a time took 14 h for 600 requests |
| **MAIRA-2** (Microsoft, 7B, gated, research licence) | #3 on ReXGradient, #5 on CheXpert Plus | only fixed fields: indication / technique / comparison. Its card says other prompt changes degrade it. A focus hint can go in the indication ("Radiologist attention: right lower lung") | ~15 GB in bf16, needs transformers 4.48–4.51 (own env) | optional, phase 2 |
| Deepwise-RG | #1 on ReXrank | — | — | no public weights or API, so it can't be run |
| CARE-X, UniRG-CXR (Microsoft Research) | #2, #3 by 1/RadCliQ | — | — | no public weights |

**Output and scoring.** CheXOne and MAIRA-2 write findings reports, not structured labels. In the pilot,
CheXOne's reports were clean in 4 of 4 cases, while our 14-finding JSON failed to parse in 4 of 4. So every
predictor writes a findings report, and **CheXbert** labels both the predicted reports and the reference
reports. CheXbert is the standard CheXpert report labeler, behind ReXrank's SembScore and CheXbert F1. It runs
locally via `f1chexbert` with weights from StanfordAIMI/RRG_scorers, takes about 0.01 s per report and costs
nothing. The prediction and the reference then go through the same labeler.
- **Checked against Sonnet** on the 100 hints_mvp reports (uncertain counted as positive): 96.5% of the 1,300
  labels agree. Per-finding kappa is 0.71–1.00 for 11 of 13 findings and 0.66 for Pneumonia. The exception is
  Enlarged Cardiomediastinum: CheXbert has 12 positives vs Sonnet's 1, a known CheXbert over-call on phrases
  like "cardiomediastinal silhouette is normal". Report it, but don't lean on it.
- **The official CheXbert label file is not usable.** CheXpert Plus ships one (`chexpert_labels`,
  report_fixed.json), but its labels contradict the report text. For example, a report reading "lungs are
  clear without focal consolidation, effusion or pneumothorax" is labeled Edema and Pleural Effusion. It looks
  misaligned with its image paths, so we run CheXbert ourselves.

Scoring reuses hints_mvp: 13 findings, sensitivity / specificity / F1, changes vs no_hint, paired bootstrap
CIs, uncertain counted as positive and separately left out.

## Conditions (one read each, per predictor)

| # | Condition | The reader gets |
|---|---|---|
| 1 | no_hint | the image |
| 2 | no_hint_rerun | the same request again, sampled independently (noise floor) |
| 3 | coords_own | the raw path as coordinates (the hints_mvp sentence) |
| 4 | focus_text_own | "A radiologist reading this X-ray spent more attention than usual on: …" (focus regions of this image's path; "… spread their attention evenly" if none) |
| 5 | focus_text_other | the same sentence with another image's focus regions |
| 6 | focus_text_random | random regions, as many as the own path's, drawn with each region's dev-set focus frequency, so naming common regions earns nothing |
| 7 | focus_mask_own | the image plus a second copy with the focus regions outlined |
| 8 | focus_mask_other | the same with another image's focus regions |
| 9 | report_regions | regions where the report puts its findings, mapped onto the same 15 regions by Sonnet: a **ceiling** for anatomy hints |
| 10 | focus_text_own_assist | condition 4 with the previous researcher's framing ("use these … to assist you in the findings") instead of the neutral wording |

Main comparisons:
- **4 vs 1, against the 2 vs 1 noise floor:** do focus regions help at all?
- **4 vs 5 and 6:** is it this image's attention?
- **4 vs 3:** does standardizing beat raw coordinates?
- **7 vs 4:** does a visual hint beat a text hint?
- **9:** can the reader use an anatomy hint at all? If the ceiling doesn't help, no path-derived hint can.
- **10 vs 4:** does the framing matter?

## Sample

- **Test set:** 300 new frontal images (200 with a finding, 100 normal), drawn with hints_mvp's rules (one per
  patient, one path each, snapped paths skipped) and excluding the 100 dev images. More positives than in
  hints_mvp, because the question is mostly about sensitivity. hints_mvp's 82 scored images gave sensitivity
  CIs about ±0.08.
- **Every image has 2+ paths.** The full CSV (CheXthought v1.0 from Redivis, 103,592 rows) has 93,157 frontal
  paths over 45,243 frontal images and 16,836 patients. 42,789 images have exactly 2 paths, 2,368 have 3 and a
  few have more. So the check of whether two radiologists' focus regions for the same image agree more than
  for different images can run on thousands of images. It is free and runs before any predictor.
- **Each path also has the radiologist's written reasoning** (`chain_of_thought_reasoning`, median 281 words).
  It names the regions they looked at and why, so it can validate the click-based focus regions (do the
  regions the clicks emphasize match the regions the text discusses?). It could later be a hint of its own.

## Compute and cost (RTX 3090, 24 GB)

| Step | Estimate |
|---|---|
| CXAS segmentation | 0.26 s per image: 300 images in under 2 min |
| CheXOne, 10 conditions x 300 images | 3,000 requests at ~1.8 s batched: **~1.5–2 h** |
| MedGemma 1.5 via vLLM | likely 2–4 h (one at a time with transformers: ~70 h, not an option) |
| MAIRA-2 (optional), fewer conditions | several hours with transformers |
| CheXbert labeling of all reports | local, ~0.01 s per report, free |
| Claude API | not needed. The only optional use is the report_regions ceiling (condition 9): extracting where each report places its findings, about $1 for 300 reports. A local rule-based alternative is possible. |

vLLM pre-allocates GPU memory, so a predictor run needs the GPU to itself (set to 60% = ~14.7 GB in the pilot).
The CXAS and CheXOne environments are separate from the reader environment: `.venv-cxas` (cxas pins numpy 1.26)
and `.venv-chexone` (vLLM 0.10.1.1, transformers 4.56.2, on a uv-managed Python, because Triton needs the
Python headers the system Python lacks).

## Order of work

1. Full CheXthought CSV (done: `~/data/chexthought/chexthoughtdataset.csv`, md5 checked against Redivis).
   Label reports with CheXbert, build the test sample and download its images from Redivis.
2. Path analysis on dev + test: style breakdown, agreement between radiologists, and focus regions vs labels on
   the dev set only. If focus regions agree no better than chance between radiologists, stop and rethink the
   heuristic before spending GPU time.
3. CheXOne pilot on 10 dev images, all conditions: prompt wording, sampling settings, report labeling.
4. Full CheXOne run on the test set, then MedGemma; MAIRA-2 if wanted.
5. Score and write up.
