# cxr-second-look

Does a "helper" model that flags likely-missed regions help a "reader" model catch more findings on chest X-rays?
Both are vision-language models called through APIs; nothing is fine-tuned.

**Design (a second look):**
1. **Pass 1:** the reader reads the image alone and fills in a structured report.
2. **Helper:** sees the image and the pass-1 report, and proposes up to 3 boxes where the reader likely missed something.
3. **Pass 2:** the reader sees the clean image, its own pass-1 JSON, and (depending on condition) the image with boxes drawn, and revises.

Every pass-2 condition revises the same pass-1 read, so all comparisons are paired per image.

| Condition | What the reader gets in pass 2 |
|---|---|
| A. `alone` | nothing: the pass-1 report is the answer |
| B. `look_again` | clean image + its report, no boxes |
| C. `random_boxes` | 3 random boxes, sizes drawn from ground-truth box sizes |
| D. `checklist_boxes` | 3 of 7 standard review areas (apices, hila, retrocardiac, costophrenic angles) |
| E. `helper_boxes` | up to 3 helper boxes, padded to 3 with random boxes |
| F. `oracle_boxes` | boxes around ground-truth masks of present findings, padded to 3 (the ceiling) |

Rules that keep results meaningful:
- Every box condition shows exactly 3 boxes, drawn identically (red, same thickness, numbered 1–3, no text labels).
- Box order is shuffled with a fixed seed before numbering, so box 1 is not always the "real" one.
- Pass-2 wording is identical for C–F; B uses the same prompt without the box sentences ([src/prompts.py](src/prompts.py)).
- The helper's `suspected_finding` and `reason` are logged only; the reader never sees them.

## Data

CheXlocalize (unofficial Kaggle re-upload), unzipped **outside the repo** to `~/data/chexlocalize/`.
The images and labels are under a research use agreement: never commit them, and never send them anywhere except the model APIs.
`.gitignore` excludes `data/`, `outputs/`, images, CSVs and `.env`.

`python -m src.verify_data` checks counts, resolution, ID matching and mask sizes, and saves 4 overlays. Findings:
- Counts match (test 668 images / 500 patients, val 234 / 200); images are full resolution (median 2828×2320).
- The segmentation JSONs list only frontal images with ≥1 positive finding. Missing images are treated as having no masks.
- One test image has positive labels but no masks, so it is excluded from the test sample (`exclude_ids` in
  [configs/v0.yaml](configs/v0.yaml)).
- Two positive labels (one val, one test) have empty masks: labels are kept for scoring; that finding gets no
  oracle box and is left out of miss coverage.
- The JSONs call Lung Opacity "Airspace Opacity"; mapped in [src/data.py](src/data.py).

Rules: val for the dry run and any prompt adjustment, test only for the final v0 run. Frontal images only.
The v0 sample is 100 frontal test images (seed 0), stratified: up to 10 positives per finding, rarest first, then random fill.

## Setup and running

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
echo 'ANTHROPIC_API_KEY=...' > .env           # gitignored

.venv/bin/python -m pytest                     # unit tests + an end-to-end mock run
.venv/bin/python -m src.run   --config configs/mock_val.yaml     # 10 val images, mock backend, free
.venv/bin/python -m src.score --config configs/mock_val.yaml
.venv/bin/python -m src.viz   --config configs/mock_val.yaml     # sample pass-2 images with GT outlines
.venv/bin/python -m src.run   --config configs/dryrun_val.yaml   # 5 val images, Claude
.venv/bin/python -m src.run   --config configs/v0.yaml           # 100 test images, Claude
```

Each run writes `predictions.jsonl` (everything per image), `calls.jsonl` (every request with tokens, cost and
settings) and `summary.json` to its `output_dir`; `src.score` adds `results.md` and `predictions.csv`.
Every response is cached under `outputs/cache/`, keyed by (model, role, condition, image, prompt hash), where the
prompt hash covers the prompt text, image pixels and model settings. Reruns are free and interrupted runs resume.
A run stops once new spend reaches `max_usd`.

Layout: `configs/` (one YAML per run), `src/` (`data`, `boxes`, `prompts`, `backends`, `run`, `score`, `viz`,
`verify_data`), `tests/`, `outputs/` (gitignored).

## Choices worth knowing

- **Models:** v0 uses `claude-sonnet-5-5` as reader and helper. Backends: `anthropic` (used), `openai` (implemented,
  unused), `openai_compatible` (for Qwen3-VL / MedGemma on vLLM; untested stub), `mock` (deterministic, for tests;
  sometimes returns broken JSON or a refusal on purpose).
- **Temperature can't be fixed on Claude:** current Claude models reject `temperature`/`top_p`/`top_k`, so the
  config sets `temperature: null` (API default) and logs it. Thinking (`adaptive` or `between_tools` = none) and
  `effort` are fixed in the config and logged with every call.
- **Helper coordinates:** Claude is asked for pixel coordinates on the image it sees (its native format), and
  `boxes.to_norm` converts them to normalized [0,1] (x1, y1, x2, y2). All other box math is normalized.
  Formats `norm`, `norm1000` (Qwen3-VL) and `yxyx1000` (Gemma/MedGemma) are also supported and tested.
- **Images** are resized so the long edge is 1024 px (`image_long_edge`) and sent as PNG.
- **Invalid JSON** is retried once with a "return only valid JSON" nudge. Refusals are not retried and no fallback
  model is used, since a fallback would change which model answered.
- **Oracle boxes** are boxes around each mask's largest connected component. If more than 3 findings are present,
  findings the reader missed in pass 1 come first. A box with IoU > 0.5 to one already chosen (e.g. Cardiomegaly vs
  Enlarged Cardiomediastinum) is skipped.
- **Checklist boxes are crude:** 7 hand-placed normalized regions for a typical frontal film
  ([src/boxes.py](src/boxes.py)). They don't adapt to patient position or body habitus, and the apex boxes sit a
  little high.
- **Random boxes** draw (width, height) from all ground-truth boxes of the run's split and place them uniformly.
- **Scoring** is complete-case: an image is scored only if every condition produced a valid report, so all
  conditions are compared on the same images. CIs come from a paired bootstrap over images (1000 resamples, seed 0).
- **Box-quality caveat:** many masks are large (Lung Opacity, Edema, Enlarged Cardiomediastinum), so random boxes
  often "hit" (≥20% of the box inside any mask) and "cover" misses (any overlap). Compare C–F against C, not against 0.

## Ideas for later

- Stricter miss coverage (e.g. ≥20% of the box inside *that* finding's mask, or box-mask IoU).
- Prompt caching for the shared image prefix, and the Batch API (50% cheaper) for large runs.
- Structured outputs (JSON schema) instead of prompt-only JSON, once other backends are in play.
- Reader/helper pairs across models (Qwen3-VL, MedGemma via vLLM; GPT via `openai`).
- More than one sample per condition to separate revision effects from sampling noise.
- Better checklist regions (e.g. from a lung segmentation model), and a better random-box baseline (boxes inside
  the lung fields).
- The other radiologists' outlines (`hb_segmentations_test.json`) as a human-level box baseline.
