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

**v1:** neutral wording, a blind read (A), a repeat blind read (B), and a fresh read with only the
helper's regions drawn (E); xhigh thinking. Full spec: [docs/v1_spec.md](docs/v1_spec.md). Set by
`prompts: v1`, `pass2_mode: fresh` and `pad_boxes: false` in [configs/v1.yaml](configs/v1.yaml).
Result on 100 val images: E lowered specificity by 0.154 [0.116, 0.193] vs B with no gain in sensitivity, because
the helper boxed 60 of the 62 images where A had missed nothing and the reader treated boxes as findings.

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

Each run writes `predictions.jsonl` (everything per image), `calls.jsonl` (every request with tokens, cost,
settings and the path to its record), `summary.json` and `spend.json` to its `output_dir`; `src.score` adds
`results.md` and `predictions.csv`.

Every request is one JSON record under `outputs/cache/<model>/<role>/<condition>/<image>__<prompt hash>.json`,
keyed by (model, role, condition, image, prompt hash), where the hash covers prompt text, image pixels and settings.
Reruns are free and interrupted runs resume. Each record holds:
- `request`: the system prompt and every content part as sent. Text is stored verbatim. Images are stored as
  image id, variant (`clean` or `boxed`), the drawn boxes with their sources (e.g. `oracle:Edema`; only the numbered
  outlines are drawn), size and pixel hash.
- `raw`: the full API response (every content block, including thinking blocks, plus usage with `thinking_tokens`).
- `thinking`: the readable thinking text; `text`: the answer; `parsed` / `parse_error`: the validated output.
- `settings`, tokens, timing and stop reason.

`max_usd` caps the run's total spend across all passes and restarts (tracked in `spend.json`).

**Long runs (overnight, flaky wifi):**

```bash
nohup caffeinate -is .venv/bin/python -m src.run --config configs/v0.yaml --until-done >> outputs/v0_run.log 2>&1 &
tail -f outputs/v0_run.log      # progress every 25 requests; safe to close the terminal
```

- `caffeinate -is` keeps the Mac awake while the run is alive (`-s` only applies on AC power). Closing a MacBook's lid
  still sleeps it unless an external display is attached.
- Lost wifi or a sleeping laptop: the affected requests fail and are not cached. `--until-done` reruns them after
  1, 2, 4, ... up to 15 minutes, until none are left. Request errors (4xx) stop the run instead of looping.
- Laptop shut down or process killed: run the same command again. Finished requests come from the cache (each is
  written atomically), so nothing is paid for twice.

Layout: `configs/` (one YAML per run), `src/` (`data`, `boxes`, `prompts`, `backends`, `run`, `score`, `viz`,
`verify_data`), `tests/`, `outputs/` (gitignored).

## Choices worth knowing

- **Models:** v0 uses `claude-sonnet-5-5` as reader and helper. Backends: `anthropic` (used), `openai` (implemented,
  unused), `openai_compatible` (for Qwen3-VL / MedGemma on vLLM; untested stub), `mock` (deterministic, for tests;
  sometimes returns broken JSON or a refusal on purpose).
- **Temperature can't be fixed on Claude:** current Claude models reject `temperature`/`top_p`/`top_k`, so the
  config sets `temperature: null` (API default) and logs it. Thinking (`adaptive` or `between_tools` = none) and
  `effort` are fixed in the config and logged with every call.
- **Thinking traces are summaries:** Claude never returns its full chain of thought. With
  `thinking_display: summarized` each thinking block holds a readable summary (thinking tokens are billed in full).
  On this task Sonnet 5.5 chose not to think at `low`, `medium` or `high` effort (0 thinking tokens over 39 val
  calls); at `xhigh` it always thinks (200–350 thinking tokens per pass-1 call).
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
  often "hit" (≥20% of the box inside any mask) and "cover" misses (any overlap). Miss coverage is therefore also
  reported strictly (≥20% of a box inside *that* finding's mask). Compare C–F against C, not against 0.

## hints_mvp: a radiologist's click trail as a hint

A separate experiment in [src/hints_mvp/](src/hints_mvp/). Does giving the reader the radiologist's click trail
for an image (CheXthought `spatial_annotations`, normalized x, y in click order) help it find more of the image's
findings, and does that depend on the trail coming from this image? One read per condition, no revision:
`no_hint`, `no_hint_rerun` (an identical second request: the noise floor), `own_trail`, `other_trail` (another
sampled image's trail), `random_trail` (as many uniform random points) and `leak_check` (own trail, no image).
The trail goes in as text, with the same sentence for every trail condition ([src/hints_mvp/prompts.py](src/hints_mvp/prompts.py)).

- **Data:** the CheXthought CSV (`~/Downloads/chexthoughtdataset.csv`, Redivis `aimi.CheXthought`) and the sampled
  CheXpert Plus PNGs, downloaded one by one from Redivis `aimi.chexpert_plus:5yyj:v1_0.png_train:s6cj` to
  `~/data/chexpert_plus/train/...` (`fetch_images.py`; needs `REDIVIS_API_TOKEN` in `.env`).
- **Sample:** frontal images in a fixed random order (seed 0), one per patient, one trace each; traces with ≥80% of
  their x and y values within 0.001 of a 0.01 grid are skipped as snapped (11% of frontal traces). Reports are labeled
  in that order until there are 50 normal images (all 13 findings absent; about 5% of reports), then the first 50
  images with a finding are taken. 1,100 reports were labeled.
- **Labels:** each report is labeled by `claude-sonnet-5-5` as present / uncertain / absent for 13 CheXpert findings;
  No Finding follows CheXpert's rule. `outputs/hints_mvp/spot_check.md` has 20 for checking by hand.
- **Scoring:** the 13 findings, pooled over images x findings, against the report labels, with uncertain counted as
  positive and, separately, left out. Changes vs `no_hint` (fixed, broken, new false findings, cleared, recovery and
  break rates) and sensitivity, specificity and F1, with paired bootstrap CIs. `leak_check` also gets Youden's J
  within findings, so knowing which findings are common earns nothing.

```bash
.venv/bin/python -m src.hints_mvp.sample --config configs/hints_mvp.yaml        # labels and sample (cached)
.venv/bin/python -m src.hints_mvp.fetch_images --config configs/hints_mvp.yaml --table "aimi.chexpert_plus:5yyj:v1_0.png_train:s6cj"
.venv/bin/python -m src.hints_mvp.run   --config configs/hints_mvp.yaml --show-prompts   # prompt_examples.md, free
.venv/bin/python -m src.hints_mvp.run   --config configs/hints_mvp.yaml --until-done
.venv/bin/python -m src.hints_mvp.score --config configs/hints_mvp.yaml        # sonnet_results.md
```

`configs/hints_mvp_mock.yaml` runs the same pipeline for free. MedGemma (`google/medgemma-1.5-4b-it`) runs through
the `transformers` backend on a CUDA GPU (bf16, sampling at Gemma 3's recommended settings, so the rerun is a real
rerun) with `configs/hints_mvp_medgemma.yaml`; setup on a GPU machine is in
[docs/hints_mvp_medgemma_handoff.md](docs/hints_mvp_medgemma_handoff.md). One cap (`max_usd`, via `spend_file`)
covers labels and every Claude call.

**Result with Sonnet 5.5** (`outputs/hints_mvp/sonnet_results.md`, $11.45 in total): no trail raised sensitivity
(own_trail −0.012 [−0.058, +0.026]), and own, other and random trails gave the same results. Any trail cut false
findings on the 50 normal images from 88 to 67–68 (specificity +0.020 to +0.027), whatever its source, so that comes
from adding the trail text, not from where the radiologist looked. Without the image, the coordinates carry a little
information (within-finding J +0.047 [+0.013, +0.093], mostly Lung Opacity and Pleural Effusion).

## Ideas for later

From v1 (`outputs/v1/results.md`):
- A helper that abstains more: the v1 helper marked regions on 98 of 100 images. Options are asking for a
  confidence per region and drawing only confident ones, or allowing at most one region.
- Tell the reader how often the helper's regions turn out to contain a finding, so it can weigh them.
- Check why E dropped diffuse findings (Enlarged Cardiomediastinum 10, Edema 5 true calls lost) in the stored thinking.

From v0 (`outputs/v0/results.md`):
- A helper that flags likely *over-calls* (the reader had 180 false positives vs 40 misses on 100 images).
- An `xhigh`-effort reader as its own condition, to get reasoning traces (medium used no thinking at all).
- Say in the prompt whether external ECG leads count as Support Devices (most Support Devices false positives cite them).
- Repeat samples per condition to measure how much answers change between identical requests.

Earlier ideas:
- Box-mask IoU as a further box-quality measure.
- Prompt caching for the shared image prefix, and the Batch API (50% cheaper) for large runs.
- Structured outputs (JSON schema) instead of prompt-only JSON, once other backends are in play.
- Reader/helper pairs across models (Qwen3-VL, MedGemma via vLLM; GPT via `openai`).
- More than one sample per condition to separate revision effects from sampling noise.
- Better checklist regions (e.g. from a lung segmentation model), and a better random-box baseline (boxes inside
  the lung fields).
- The other radiologists' outlines (`hb_segmentations_test.json`) as a human-level box baseline.
