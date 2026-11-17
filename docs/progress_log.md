# Progress log

One running log of what was done, what was found and what's next. Newest first. Aggregate numbers only: no
reports, labels or image ids (research use agreement). Machine: the Linux RTX 3090 PC (24 GB).

---

## 2026-11-17: CheXOne scored; MedGemma and MAIRA-2 running

**Done**
- **CheXOne full run finished** (it had completed on 2026-11-12 in 19.5 min; nothing ran in between because the session
  had stopped). 3,300 of 3,300 requests; 300 of 300 images scored; 15 outputs hit the token limit (0–6 per condition).
- **Result: no hint changes CheXOne at all.** Baseline sensitivity 0.605 [0.568, 0.643], specificity 0.943. Every
  condition's Δ sensitivity is within ±0.006 of no_hint, the same size as an identical rerun (−0.006
  [−0.017, +0.004]). Even the ceiling (report_regions, regions taken from the real report) gives +0.001
  [−0.011, +0.013]. The outlined-image hints (focus_mask_own / _other) cost a little specificity (−0.005
  [−0.009, −0.001] and −0.004 [−0.008, −0.001]) and help nothing.
- **CheXOne does read the hints, a little.** With a focus hint, its report names a hinted region on 31% (own) /
  29% (other) / 22% (random) of images vs 20% / 14% / 1% without the hint. It mentions the region, but its
  findings don't change.
- **Same on the 147 images whose own path has a focus region** (where the hints actually differ): all Δs within
  ±0.012, focus_mask_own specificity −0.011 [−0.018, −0.003].
- Full tables: [results/trail_anatomy_chexone.md](results/trail_anatomy_chexone.md). hints_mvp MedGemma tables copied to
  [results/hints_mvp_medgemma.md](results/hints_mvp_medgemma.md).
- MedGemma via vLLM tested on 2 images (works, ~4 s per request batched; no thinking block in this mode).
- Added `AGENTS.md` / `CLAUDE.md`: every agent must push milestones to `main`.
- **Blocker: this PC can't push to GitHub** (no credentials: no gh, no SSH key, no token). Commits are on local
  `main` and `trail-anatomy` until the owner sets up push access.

**Running** (`outputs/trail_anatomy/run_rest.sh`): MedGemma 300 × 11 conditions, then scoring, then MAIRA-2
300 × 6 conditions, then scoring.

---

## 2026-11-12 (evening): test sample, path analysis, predictor runs started

**Done**
- **Test sample:** 300 new images (200 with a finding, 100 normal), drawn with the hints_mvp rules, dev patients
  excluded. CheXbert labeled 1,977 reports in candidate order, locally. 233 of the 300 images also have another
  radiologist's usable path. All images downloaded from Redivis and segmented (67 s).
- **Path analysis** (`src/trail_anatomy/analyze_paths.py`), 723 paths: 185 from dev images, 538 from test.
  - Focus rule tuned on dev by agreement between radiologists, using no labels: α = 20, τ = 2.0, up to 3
    regions, ≥ 3 clicks. About half of all paths have no focus region (test: 273 of 538). Sparse paths almost
    never do (80% none); tracing paths most often do.
  - **Radiologists agree beyond chance.** Two radiologists' focus regions on the same image overlap more than
    on different images: Jaccard 0.361 vs 0.268, 243 test pairs, permutation p < 0.005 (dev: 0.335 vs 0.261,
    p = 0.015). So focus regions carry modest image-specific information.
  - **Written reasoning doesn't single out focus regions.** Radiologists name 67% of their focus regions and
    68% of the other regions they clicked: they describe the whole chest either way.
  - **Focus regions vs findings (test, own paths).** A focus region lies in the relevant area more often on
    positive images for Pleural Effusion (0.43 vs 0.23), Pneumothorax (0.14 vs 0.01) and Enlarged
    Cardiomediastinum (0.14 vs 0.05). It is flat for Lung Opacity, Atelectasis, Consolidation, Edema and
    Cardiomegaly.
  - This image's path has any focus region on 54.5% of finding images vs 38% of normal ones.
- **Hints built** for dev and test. The report_regions ceiling names ≥ 1 region on 92% of finding images and
  44% of normal ones (neutral mentions such as "trachea midline").
- **CheXOne pilot**, 10 dev images × 11 conditions: 110 requests in 76 s. 2 answers got stuck repeating
  themselves; fixed by applying CheXOne's own repetition penalty of 1.05, which vLLM ignores when sampling
  settings are passed explicitly. The region outlines were checked visually.
- **CheXOne full run started:** 300 × 11 = 3,300 requests, temperature 0.7.
- MAIRA-2 weights downloaded (the token has access); its env is `.venv-maira` (transformers 4.51.3).

---

## 2026-11-12: trail_anatomy setup, data, labeler

**Done**
- Branch `trail-anatomy`. Plan: [docs/trail_anatomy_spec.md](trail_anatomy_spec.md).
- **CheXthought CSV** downloaded from Redivis (`aimi.chexthought:ds7j:v1_0`, a single file) to
  `~/data/chexthought/chexthoughtdataset.csv`. It is 508,908,453 bytes, the same as the Mac copy, and its md5
  matches Redivis. It was never on this PC before: only `hints_mvp_bundle.tgz` (100 images + sample.json) had
  been copied over.
- **The CSV:** 103,592 paths and 50,312 images. 93,157 of the paths are frontal, over 45,243 frontal images and
  16,836 patients. Every frontal image has at least 2 paths (2: 42,789 images; 3: 2,368; more: 86). Columns:
  `image_id`, `spatial_annotations` (x, y, click order), `report` (the same for every path of an image) and
  `chain_of_thought_reasoning` (the radiologist's written reasoning, median 281 words).
- **CXAS** anatomy segmentation in its own env, `.venv-cxas`, because cxas pins numpy 1.26. It needed gdown < 6.
  All 100 dev images segmented in 26 s, using 0.9 GB of GPU memory. Overlays confirm that clicks line up with
  the anatomy and that the patient's right is on the image's left.
- **Click analysis on the 100 dev paths:** 5–334 points (median 54.5). The median path touches 11 of 15
  anatomical regions, and the heart is touched by 100% of paths. Normal and abnormal images get the same kind
  of path. Relative attention: AUC 0.70 for Cardiomegaly / heart and 0.68 for Support Devices / trachea +
  mediastinum. It is reversed for Lung Opacity (0.23) and Edema (0.32), probably because clicks that follow
  devices take clicks away from the lungs.
- **Focus-region rule** prototyped (shrunk click share relative to the average path; α 10, τ 1.5, up to 3
  regions). Paths with 0 / 1 / 2 / 3 focus regions: 6 / 36 / 30 / 28.
- **CheXOne** (StanfordAIMI, Qwen2.5-VL-3B) in its own env, `.venv-chexone` (vLLM 0.10.1.1, transformers 4.56.2).
  It runs on a uv-managed Python, because Triton needs the Python headers the system Python lacks. Pilot on 4
  images: the findings reports were clean in 4 of 4, our JSON format failed in 4 of 4. 1.8 s per request with
  vLLM batching.
- **Labeler:** CheXbert, local and free, replaces Sonnet. It agrees with the Sonnet labels on 96.5% of the
  1,300 labels for the 100 reports. Kappa is 0.71–1.00 for 11 of 13 findings and 0.66 for Pneumonia;
  Enlarged Cardiomediastinum is over-called (12 vs 1). The official CheXpert Plus CheXbert file
  (report_fixed.json) contradicts the report text and looks misaligned, so it is not used.

**Next**
- Build the 300-image test sample from the full CSV (excluding the 100 dev images), label it with CheXbert and
  download its images from Redivis.
- Check whether the focus regions of two radiologists reading the same image agree (thousands of images).
- CheXOne pilot with all 10 conditions on 10 dev images.

---

## 2026-11-05: hints_mvp with MedGemma 1.5 4B (local GPU)

- The full run took 14.2 h (one request at a time with transformers, ~85 s per request). 600 requests, 82 of
  100 images scored.
- No search path helped. Δ sensitivity vs no_hint: own −0.073 [−0.148, −0.007], other −0.057, random −0.049,
  and an identical rerun −0.033 [−0.111, +0.050]. Unlike Sonnet, trails didn't reduce false findings.
  leak_check J +0.015 [−0.039, +0.083].
- Fixes: the parser now skips MedGemma's `<unused94>thought … <unused95>` block, and max_tokens went from 2000
  to 4000.
- Results: `outputs/hints_mvp/medgemma/medgemma_results.md` (local, gitignored).
