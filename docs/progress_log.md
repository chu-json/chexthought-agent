# Progress log

One running log of what was done, what was found and what's next. Newest first. Aggregate numbers only: no
reports, labels or image ids (research use agreement). Machine: the Linux RTX 3090 PC (24 GB).

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
