"""Run MAIRA-2 (microsoft/maira-2) with hints in its indication field; same cache format as run_vllm.

Usage (in .venv-maira): python -m src.trail_anatomy.run_maira [--limit N]
MAIRA-2 only takes fixed inputs (frontal, lateral, prior, indication, technique, comparison); its card warns
that other prompt changes degrade it. So the only hint channel is the indication, the field a clinician uses
to direct a reader's attention. Frontal image only; no lateral, prior, technique or comparison.
Decoding is greedy (as on the model card), so an identical rerun would be identical: the noise floor is
neutral_text, an uninformative indication. Conditions: no_hint (no indication), neutral_text,
focus_text_own / _other / _random, report_regions (the ceiling). Non-grounded findings, max_new_tokens 300.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from ..hints_mvp import data
from . import focus as F
from .run_vllm import MODELS, NEUTRAL

CONDITIONS = ["no_hint", "neutral_text", "focus_text_own", "focus_text_other", "focus_text_random", "report_regions"]


def indication(cond, h):
    if cond == "no_hint":
        return None
    if cond == "neutral_text":
        return NEUTRAL
    if cond == "report_regions":
        return F.focus_text(h["report_regions"])
    return F.focus_text(h[cond.replace("_text", "")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="outputs/trail_anatomy/test/sample.json")
    ap.add_argument("--conditions", nargs="*", default=CONDITIONS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cache", default="outputs/trail_anatomy/cache")
    a = ap.parse_args()
    m = MODELS["maira"]
    settings = {k: m[k] for k in ("hf", "max_tokens", "temperature")}
    sample = json.load(open(a.sample))[: a.limit]
    hints = json.load(open(Path(a.sample).with_name("hints.json")))
    from transformers import AutoModelForCausalLM, AutoProcessor
    model = AutoModelForCausalLM.from_pretrained(m["hf"], trust_remote_code=True, torch_dtype=torch.bfloat16).eval().to("cuda")
    proc = AutoProcessor.from_pretrained(m["hf"], trust_remote_code=True)
    t0, n = time.time(), 0
    for s in sample:
        image = data.load_image("~/data/chexpert_plus", s["image_id"], 1024)
        for c in a.conditions:
            ind = indication(c, hints[s["image_id"]])
            key = hashlib.sha256(json.dumps([ind, hashlib.sha256(image.tobytes()).hexdigest(), settings]).encode()).hexdigest()[:16]
            path = Path(a.cache) / "maira" / c / f"{s['image_id']}__{key}.json"
            if path.exists():
                continue
            inp = proc.format_and_preprocess_reporting_input(
                current_frontal=image, current_lateral=None, prior_frontal=None, indication=ind, technique=None,
                comparison=None, prior_report=None, return_tensors="pt", get_grounding=False).to("cuda")
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=m["max_tokens"], use_cache=True, do_sample=False)
            gen = out[0][inp["input_ids"].shape[-1]:]
            text = proc.decode(gen, skip_special_tokens=True).lstrip()
            report = proc.convert_output_to_plaintext_or_grounded_sequence(text)
            rec = {"model": "maira", "condition": c, "image_id": s["image_id"], "settings": settings, "prompt": ind,
                   "images": [{"variant": "clean"}], "text": text,
                   "finish_reason": "length" if len(gen) >= m["max_tokens"] else "stop",
                   "output_tokens": int(len(gen)), "report": report if isinstance(report, str) else str(report)}
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            json.dump(rec, open(tmp, "w"))
            tmp.rename(path)
            n += 1
            if n % 50 == 0:
                print(f"  [{time.strftime('%H:%M:%S')}] {n} requests, {time.time() - t0:.0f}s", flush=True)
    print(f"done: {n} new requests in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
