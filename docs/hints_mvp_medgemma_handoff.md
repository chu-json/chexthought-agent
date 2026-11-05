# Handoff: run hints_mvp with MedGemma on the local GPU

For a Claude Code session on the PC with the RTX 3090. Your job: set up this repo, run the hints_mvp experiment
with MedGemma (`google/medgemma-1.5-4b-it`) as the reader on the local GPU, score it, and tell the user the results.
The code is done and was tested on a Mac (CPU, with a tiny random Gemma 3 model), but it has never run with
MedGemma itself or on a GPU. Expect to fix environment problems, not to change the experiment.

## What the experiment is

Read the hints_mvp section of [README.md](../README.md) first. In short: 100 frontal chest X-rays (50 whose report
has a finding, 50 normal), 6 conditions, one read each: `no_hint`, `no_hint_rerun`, `own_trail`, `other_trail`,
`random_trail` and `leak_check`. A radiologist's click trail goes in as text. The question is whether the trail
helps the reader find more findings, and whether it matters that the trail comes from this image.

The same experiment already ran with Claude Sonnet 5.5 on the Mac. With Sonnet, no trail raised sensitivity
(own_trail −0.012 [−0.058, +0.026]), and own, other and random trails behaved the same. Any trail cut false findings
on normal images (88 → 67–68). MedGemma is the second reader. A weaker reader should miss more, so it has more
room for a trail to help.

Code: [src/hints_mvp/](../src/hints_mvp/) (`run.py`, `score.py`, `prompts.py`, `data.py`); the backend is
`TransformersBackend` in [src/backends.py](../src/backends.py); config [configs/hints_mvp_medgemma.yaml](../configs/hints_mvp_medgemma.yaml).

## Rules

- **Data use agreement:** the images and `sample.json` (reports and labels) are CheXpert Plus data under a research
  use agreement. Never commit them, never upload them anywhere, and never send them to any API. MedGemma runs
  locally, so nothing leaves the machine. `outputs/`, `data/`, images and CSVs are gitignored; keep it that way.
- **Don't change the experiment:** that means the prompts, the sample, the seeds and the sampling settings
  (temperature 1.0, top_p 0.95, top_k 64, do_sample true). Changing any of them breaks comparability with the
  Sonnet run. If you think one has to change, stop and ask the user.
- **Don't regenerate the sample:** `python -m src.hints_mvp.sample` calls the Claude API and would rebuild it.
  The sample comes from the Mac (below).

## 1. Data (not in git)

The user copies `hints_mvp_bundle.tgz` from the Mac. It holds `outputs/hints_mvp/sample.json` and the 100 images
under `chexpert_plus/train/patientX/studyN/viewK_frontal.png`. From the repo root:

```bash
tar xzf <path to>/hints_mvp_bundle.tgz outputs/hints_mvp/sample.json
mkdir -p ~/data && tar xzf <path to>/hints_mvp_bundle.tgz -C ~/data chexpert_plus
```

`tar` also exists in Windows 10+ PowerShell. The config reads images from `~/data/chexpert_plus`; Python expands
`~` on Windows too, to `C:\Users\<name>\data\chexpert_plus`. Check this before going on: `sample.json` has 100 entries,
and all 100 paths in it (`train/...` under `~/data/chexpert_plus`) exist.

## 2. Environment

First find out the OS: Linux, WSL2 or native Windows.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt torch "transformers>=5" accelerate
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.venv/bin/python -m pytest -q          # 64 tests, no GPU needed; all should pass
```

- **Native Windows:** use `.venv\Scripts\python`. PyPI's Windows `torch` is CPU-only, so install the CUDA build
  with the command from https://pytorch.org/get-started/locally/ (pip, CUDA 12.x). `torch.cuda.is_available()`
  must print True before going on.
- **Package versions:** the Mac test used torch 2.15 and transformers 5.22. If transformers can't load MedGemma,
  upgrade it.
- **MedGemma is gated:** the user must accept its terms at https://huggingface.co/google/medgemma-1.5-4b-it with
  their Hugging Face account, then log in with `.venv/bin/hf auth login` (a read token). You can't do either for them.

## 3. Smoke test on 5 images

```bash
.venv/bin/python -m src.hints_mvp.run --config configs/hints_mvp_medgemma.yaml --limit 5
```

The first run downloads about 8 GB of weights; the model needs about 9 GB of GPU memory. The summary has one line per
condition. Check the following:
- **Parse failures:** `final_failures` should be 0 or close to it. The parser tolerates code fences, text around the
  JSON, trailing commas, case and spacing in finding names, `"Edema": "yes"` shorthand, and probabilities as
  `"70%"`; a bad answer gets one retry with a "return only valid JSON" nudge.
- **The raw answers:** each request is one JSON record under
  `outputs/cache/google/medgemma-1.5-4b-it/reader/<condition>/<image>__<hash>.json`, with fields `text`, `parsed`,
  `parse_error` and `stop_reason`. Read a few. If `stop_reason` is often `max_tokens` (2000 new tokens) or the text
  is not a report, show the user an example before changing anything.
- **Sampling:** `no_hint` and `no_hint_rerun` should differ on at least some answers. If they never do, sampling isn't on.
- **Time:** the summary prints the seconds taken. Divide by 30 requests (more if there were retries) and multiply by 600 to estimate the full run.

## 4. Full run and scoring

```bash
mkdir -p outputs/hints_mvp/medgemma
nohup .venv/bin/python -m src.hints_mvp.run --config configs/hints_mvp_medgemma.yaml --until-done \
  > outputs/hints_mvp/medgemma/run.log 2>&1 &
tail -f outputs/hints_mvp/medgemma/run.log
.venv/bin/python -m src.hints_mvp.score --config configs/hints_mvp_medgemma.yaml
```

- **Native Windows:** `nohup` doesn't exist; run it in a terminal that stays open, or with `Start-Process`.
- **Resuming:** every finished request is cached, so an interrupted run resumes for free; just run the same command again.
- **`--until-done`:** reruns only network errors, so a CUDA error stops the run. Fix the cause and run it again.
- **Output:** `score` writes `outputs/hints_mvp/medgemma/medgemma_results.md`.

## 5. Report back

1. **Observations:** write 3–5 plain ones into `outputs/hints_mvp/medgemma/observations.md`, like those at the
   end of the Sonnet results (the user has those on the Mac), and rescore so they're appended. Compare against
   `no_hint_rerun` before calling anything an effect. Also compare MedGemma's `no_hint` sensitivity with Sonnet's 0.828.
2. **For the user:** give them the main table: fixed, broken, new false findings and cleared per condition; recovery
   and break rates; Δ sensitivity and Δ specificity with CIs; and leak_check. Add the failure counts, the run time and
   any problems you hit.
3. **Copy back:** remind the user to copy `outputs/hints_mvp/medgemma/` back to the Mac. Don't commit it.
