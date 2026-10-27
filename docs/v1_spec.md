# v1 spec: blind read, then the helper's regions

**Question.** If a reader first reads a chest X-ray blind, does seeing the regions another model marked as likely
missed (locations only, nothing else) change how well it reads the X-ray?

Config: [configs/v1.yaml](../configs/v1.yaml) (100 images) and [configs/v1_check.yaml](../configs/v1_check.yaml)
(5 images, to look at real outputs first).

## What changed from v0

| | v0 | v1 |
|---|---|---|
| Pass-2 wording | "A colleague flagged these regions for a second look. They may or may not contain abnormalities. Review your report and revise it if needed." | Describes the images and asks for the report. No colleague, no flagging, no "missed", no "revise". |
| Reader sees its earlier report in pass 2 | yes (revises it) | **no**: a fresh read (v0 showed it anchors to its first report) |
| Helper output | boxes + suspected finding + reason (finding and reason logged only) | **regions only**. Its thinking summary is logged and never shown to the reader. |
| Boxes drawn | always 3 (padded with random boxes) | **only the helper's real boxes** (0–3), no padding |
| Effort | medium (no thinking happened) | **xhigh** for reader and helper, the only level where Sonnet 5.5 thinks on this task |
| Conditions | A–F (6) | **A, B, E** (3); oracle F optional |

## Per-image flow

1. **A. Blind read.** The reader gets the clean image only.
2. **Helper.** Gets the clean image and A's report. Returns 0–3 regions where that report most likely missed a
   visible finding.
3. **B. Repeat blind read.** Exactly the same request as A, sampled again (its own API call and cache entry).
   This measures how much answers change with no hint at all.
4. **E. Read with the helper's regions.** The reader gets the clean image plus the same image with the helper's
   regions drawn. It does not see its earlier report. If the helper marked no regions, E gets the clean image
   alone, the same as a blind read.

| Condition | Reader gets | Calls |
|---|---|---|
| A. alone | clean image | 100 |
| B. repeat | clean image (same request as A) | 100 |
| E. helper_boxes | clean image + image with the helper's regions | 100 |
| Helper | clean image + A's report | 100 |
| F. oracle_boxes (optional, off) | clean image + image with boxes around the ground-truth masks | 100 |

**Main comparison: E vs B.** Both are second reads made independently of A and differ only by the hints.
B vs A gives the noise floor (how much a plain re-read changes answers).

## Exact prompts

Reader system prompt (A, B and E; same as v0):
```
You are an expert radiologist reading frontal chest X-rays.

For each of these 10 findings, decide whether it is present on the image: Atelectasis, Cardiomegaly, Consolidation,
Edema, Enlarged Cardiomediastinum, Lung Lesion, Lung Opacity, Pleural Effusion, Pneumothorax, Support Devices.

Respond with only a JSON object, no other text, in exactly this format:
{"findings": {"<finding>": {"present": "yes" or "no", "probability": <integer 0-100, your probability that the
finding is present>, "location": "<short description of where, or empty string>"}, ...}, "reasoning": "<short
summary of your reasoning>"}

"findings" must have exactly one entry for each of the 10 findings, using the exact names above.
```

A and B (one image: clean):
```
Read this chest X-ray and fill in the report.
```

E (two images: clean, then boxed):
```
The first image is a chest X-ray. The second image is the same X-ray with numbered boxes drawn on it by another model.

Read this chest X-ray and fill in the report.
```

Helper system prompt:
```
You are an expert radiologist reviewing chest X-ray reads.
```

Helper (one image: clean):
```
Another reader looked at this frontal chest X-ray and wrote this report:
<A's report as JSON>

Mark up to 3 regions of the image where this report most likely missed one of these findings: Atelectasis,
Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, Lung Lesion, Lung Opacity, Pleural Effusion,
Pneumothorax, Support Devices. A missed finding is one that is visible on the image but marked "no" in the report.
Mark fewer regions, or none, if you think nothing was missed. Give only the regions, with no labels or explanations.

Give each box in pixel coordinates of this image, which is <W> pixels wide and <H> pixels tall, with (0, 0) at
the top-left corner.

Respond with only a JSON object, no other text, in exactly this format:
{"boxes": [[x1, y1, x2, y2], ...]}
```

Changed after the 5-image check: the helper was first told to mark "a finding that is visible on the image", and
its thinking went to findings outside the 10 (fractures, subcutaneous emphysema, tube position), with boxes on
the shoulders. It now names the 10 scored findings. This is the helper's instruction only; the reader never sees it.

Invalid JSON is retried once with: `Return only valid JSON in exactly the format specified, with no other text.`
A test checks that the reader prompts contain none of: colleague, flag, second look, missed, revise, abnormal, review.

## Boxes

- The helper's pixel boxes (on the 1024-px image it saw) are converted to normalized [0,1] (x1, y1, x2, y2) in
  `boxes.to_norm`; all box math is normalized.
- Drawn identically: red outline, the same thickness, numbered 1..n, no labels. The order is shuffled with a fixed
  seed, so the numbers carry no ranking.
- No padding: the reader sees exactly the regions the helper marked.
- If the helper call fails (invalid JSON twice, or a refusal), that image gets no E read and is dropped from every
  condition when scoring, so the comparisons stay paired.

## Models and settings

| | Reader | Helper |
|---|---|---|
| Model | claude-sonnet-5-5 | claude-sonnet-5-5 |
| Thinking | adaptive, xhigh effort, summary returned | same |
| Temperature | API default (Claude rejects sampling parameters) | same |
| max_tokens | 16,000 | 16,000 |

Images are resized so the long edge is 1024 px and sent as PNG.

## Data

Val split: 100 of the 202 frontal val images (seed 0), stratified (up to 10 positives per finding, rarest first,
then random fill). 16 images are all-negative. Positives in the sample:

| Atelectasis | Cardiomegaly | Consolidation | Edema | Enl. Cardiomediastinum | Lung Lesion | Lung Opacity | Pleural Effusion | Pneumothorax | Support Devices |
|---|---|---|---|---|---|---|---|---|---|
| 37 | 32 | 12 | 15 | 53 | **1** | 58 | 32 | **7** | 47 |

Why val: the project rule is val for prompt changes and test only for final runs. Trade-off: val has only
1 Lung Lesion and 7 Pneumothorax (the v0 test sample has 9 and 9). To run on the v0 test images instead, set
`split: test` and `exclude_ids: [patient65086_study1_view1_frontal]`.

## Scoring (`python -m src.score --config configs/v1.yaml`)

Same as v0:
- Per finding and macro-averaged: sensitivity, specificity, F1 (from yes/no) and AUROC (from probability).
- Paired metric changes from A and from B.
- Answer changes (fixed, added, net, lost, cleared) vs A and vs B.
- Helper box quality: hit rate, and coverage of A's misses (any overlap and strict).
- 95% CIs from a paired bootstrap over images (1000 resamples, seed 0).

Output: `outputs/v1/results.md` and `outputs/v1/predictions.csv`.

## What is stored

Every request is one JSON record under `outputs/cache/claude-sonnet-5-5/<role>/<condition>/`, holding:
- the request: system prompt; text verbatim; each image's id, variant (clean or boxed), drawn boxes and pixel hash;
- the full raw response, including thinking blocks and the `thinking_tokens` count;
- the thinking summary text, the answer text, and the parsed output or parse error.

`outputs/v1/calls.jsonl` lists every call with the path to its record.

## Caching, resume and budget

- Finished requests are cached; failed ones are not. Rerunning the same command resumes, and finished requests
  are never paid for twice.
- `--until-done` reruns failed requests after network loss or sleep (waits 1, 2, 4 … up to 15 minutes) and stops
  on request errors (4xx).
- `max_usd: 15` covers the run's total spend across all passes and restarts (`outputs/v1/spend.json`).

## Cost and time (estimate)

Input is about 1.4k tokens per one-image read, 2.0k for the helper and 2.5k for E. Output at xhigh is about
0.8–1.1k tokens per read (measured on 2 probe calls) and less for the helper.

| Item | Estimate |
|---|---|
| Per image (4 calls) | ~$0.055 |
| v1, 100 images (400 calls) | **~$5.50** (range $4–8); cap $15 |
| v1_check, 5 images | ~$0.30 (its responses are reused by v1 for any image both share) |
| Optional oracle F | +~$1.70 |
| Optional Opus 5.5 helper | +~$1.20 |
| Wall time | ~15–20 minutes |

## Limits

- One sample per condition. E vs B compares two independent reads, so small differences can be noise; the B vs A
  change shows how big that noise is.
- 100 images; rare findings (Lung Lesion, Pneumothorax) have very few positives on val.
- The reader is told only that "another model" drew the boxes. It may still read them as a suggestion, which is
  the point of the hint, but the wording no longer says anything was missed or should change.
