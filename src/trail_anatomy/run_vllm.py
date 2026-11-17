"""Run a local predictor (CheXOne or MedGemma, via vLLM) on every image and condition; cache every request.

Usage (in .venv-chexone): python -m src.trail_anatomy.run_vllm --model chexone [--sample ...] [--limit N]
Every request is one JSON record under outputs/trail_anatomy/cache/<model>/<condition>/<image>__<hash>.json,
keyed by a hash of the prompt text, image pixels and sampling settings, so reruns are free and an interrupted
run resumes. Each record holds the prompt text, the image variants, the raw output and the extracted report.
All conditions write a findings report; scoring labels it with CheXbert (score.py).

Conditions (the text before the instruction differs; images are the clean image unless noted):
  no_hint, no_hint_rerun (same request, independent sample: the noise floor), neutral_text (an uninformative
  sentence: the effect of any added text), coords_own (the raw path, hints_mvp wording),
  focus_text_own / _other / _random, focus_text_own_assist (the previous researcher's "use ... to assist you"),
  focus_mask_own / _other (a second image with the focus regions outlined), report_regions (the ceiling).
A focus hint with no regions says the radiologist spread their attention evenly (one image only).
"""
import argparse
import base64
import hashlib
import io
import json
import re
import time
from pathlib import Path

from ..hints_mvp import data
from ..hints_mvp.prompts import trail_text
from . import anatomy as A
from . import focus as F

CONDITIONS = ["no_hint", "no_hint_rerun", "neutral_text", "coords_own", "focus_text_own", "focus_text_other",
              "focus_text_random", "focus_text_own_assist", "focus_mask_own", "focus_mask_other", "report_regions"]
NEUTRAL = "This chest X-ray was acquired as part of routine clinical care."
# Replication of CheXthought §4.3.2 on its base model (Qwen3-VL-8B-Thinking): the trajectory as structured text with
# a Broad / Central / Narrow strategy label (strategy.py), from this image (own), another image (other) or uniform
# random points (random, the paper's control). Run with --conditions no_hint no_hint_rerun strategy_own ...
QWEN3_CONDITIONS = ["no_hint", "no_hint_rerun", "strategy_own", "strategy_other", "strategy_random", "focus_text_own"]
_STRATEGY = None


def strategy_model():
    global _STRATEGY
    if _STRATEGY is None:
        _STRATEGY = json.load(open("outputs/trail_anatomy/strategy.json"))
    return _STRATEGY

MODELS = {
    "chexone": {"hf": "StanfordAIMI/CheXOne", "max_tokens": 3000, "temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.05,
                "instruction": "Write the findings section for the CXR. Please reason step by step, and put your final answer within \\boxed{}.",
                "llm": {"max_model_len": 8192, "limit_mm_per_prompt": {"image": 2},
                        "mm_processor_kwargs": {"min_pixels": 256 * 28 * 28, "max_pixels": 512 * 512}}},
    "medgemma": {"hf": "google/medgemma-1.5-4b-it", "max_tokens": 4000, "temperature": 1.0, "top_p": 0.95, "top_k": 64,
                 "instruction": "Write the findings section of the radiology report for this chest X-ray.",
                 "llm": {"max_model_len": 8192, "limit_mm_per_prompt": {"image": 2}}},
    # CheXthought's base model. FP8 (Qwen's own checkpoint) so the KV cache fits long thinking on a 24 GB card;
    # sampling as on the model card; the paper's reasoning prompt (its §4.2.1).
    "qwen3": {"hf": "Qwen/Qwen3-VL-8B-Thinking-FP8", "max_tokens": 12000, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
              "instruction": "Interpret this chest X-ray and provide step-by-step diagnostic reasoning followed by your final findings.",
              "llm": {"max_model_len": 14336, "limit_mm_per_prompt": {"image": 1},
                      "mm_processor_kwargs": {"min_pixels": 256 * 28 * 28, "max_pixels": 640 * 640}}},
    # Run by run_maira.py (transformers, its own env), not vLLM; listed here so score.py finds its settings.
    "maira": {"hf": "microsoft/maira-2", "max_tokens": 300, "temperature": 0.0},
}
VLLM_MODELS = ["chexone", "medgemma", "qwen3"]


def data_url(im):
    b = io.BytesIO()
    im.save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def digest(im):
    return hashlib.sha256(im.tobytes()).hexdigest()[:16]


def request(cond, s, h, image, model):
    """(text, images, image_meta) for one image and condition."""
    m = MODELS[model]
    hint, images, meta = None, [image], [{"variant": "clean"}]
    if cond in ("no_hint", "no_hint_rerun"):
        hint = None
    elif cond == "neutral_text":
        hint = NEUTRAL
    elif cond == "coords_own":
        hint = trail_text(h["coords_own"])
    elif cond in ("focus_text_own", "focus_text_other", "focus_text_random"):
        hint = F.focus_text(h[cond.replace("_text", "")])
    elif cond == "focus_text_own_assist":
        hint = F.focus_text(h["focus_own"], assist=True)
    elif cond == "report_regions":
        hint = F.focus_text(h["report_regions"])
    elif cond in ("strategy_own", "strategy_other", "strategy_random"):
        from .strategy import label, trajectory_text
        pts = s["trails"][cond.split("_")[1]]["points"]
        hint = trajectory_text(pts, label(pts, strategy_model()))
    elif cond in ("focus_mask_own", "focus_mask_other"):
        regions = h[cond.replace("_mask", "")]
        if not regions:
            hint = F.EVEN_SENTENCE
        else:
            lab = A.load_partition("outputs/trail_anatomy/masks", s["image_id"])
            images = [image, F.outline_image(image, lab, regions)]
            meta = [{"variant": "clean"}, {"variant": "outlined", "regions": regions}]
            hint = F.MASK_INTRO
    text = (hint + "\n\n" if hint else "") + m["instruction"]
    return text, images, meta


def extract_report(text, model):
    """The final report: CheXOne's last \\boxed{...}; MedGemma's text after its thinking block."""
    if model == "medgemma":
        return text.rsplit("<unused95>", 1)[-1].strip() if "<unused94>" in text else text.strip()
    if model == "qwen3":  # the answer after the thinking block; from a "final findings" heading on, if there is one
        if "</think>" not in text:  # the template opens <think> in the prompt: no end tag = cut off while thinking
            return ""
        ans = text.rsplit("</think>", 1)[-1]
        hits = list(re.finditer(r"final (?:findings|impression|diagnos[ie]s|answer)", ans, re.I))
        return (ans[hits[-1].end():] if hits else ans).strip(" :*#\n")
    k = text.rfind("\\boxed{")
    if k < 0:
        return text.split("Final Answer", 1)[-1].strip(" :*\n")
    depth, j = 0, k + len("\\boxed{") - 1
    for j in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            break
    return text[k + len("\\boxed{"):j].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=VLLM_MODELS, required=True)
    ap.add_argument("--sample", default="outputs/trail_anatomy/test/sample.json")
    ap.add_argument("--hints", default=None, help="default: hints.json next to the sample")
    ap.add_argument("--conditions", nargs="*", default=CONDITIONS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--cache", default="outputs/trail_anatomy/cache")
    a = ap.parse_args()
    m = MODELS[a.model]
    sample = json.load(open(a.sample))[: a.limit]
    hints = json.load(open(a.hints or Path(a.sample).with_name("hints.json")))
    settings = {k: m[k] for k in ("hf", "max_tokens", "temperature", "top_p", "top_k", "repetition_penalty") if k in m}
    jobs = []
    for s in sample:
        image = data.load_image("~/data/chexpert_plus", s["image_id"], 1024).convert("RGB")
        for c in a.conditions:
            text, images, meta = request(c, s, hints[s["image_id"]], image, a.model)
            key = hashlib.sha256(json.dumps([c if c == "no_hint_rerun" else "", text, [digest(i) for i in images],
                                             settings]).encode()).hexdigest()[:16]
            path = Path(a.cache) / a.model / c / f"{s['image_id']}__{key}.json"
            if not path.exists():
                jobs.append((path, s["image_id"], c, text, images, meta, int(key[:8], 16)))
    print(f"{a.model}: {len(sample)} images x {len(a.conditions)} conditions, {len(jobs)} requests to run", flush=True)
    if not jobs:
        return
    from vllm import LLM, SamplingParams
    llm = LLM(m["hf"], dtype="bfloat16", gpu_memory_utilization=a.gpu_mem, seed=0, **m["llm"])
    t0 = time.time()
    for k in range(0, len(jobs), a.chunk):
        chunk = jobs[k:k + a.chunk]
        convs = [[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url(i)}} for i in images]
                   + [{"type": "text", "text": text}]}] for _, _, _, text, images, _, _ in chunk]
        params = [SamplingParams(max_tokens=m["max_tokens"], temperature=m["temperature"], top_p=m["top_p"],
                                 top_k=m.get("top_k", -1), repetition_penalty=m.get("repetition_penalty", 1.0), seed=seed) for *_, seed in chunk]
        outs = llm.chat(convs, params, use_tqdm=False)
        for (path, iid, c, text, images, meta, seed), o in zip(chunk, outs):
            r = o.outputs[0]
            rec = {"model": a.model, "condition": c, "image_id": iid, "settings": settings, "seed": seed,
                   "prompt": text, "images": meta, "text": r.text, "finish_reason": r.finish_reason,
                   "output_tokens": len(r.token_ids), "report": extract_report(r.text, a.model)}
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            json.dump(rec, open(tmp, "w"))
            tmp.rename(path)
        done = min(k + a.chunk, len(jobs))
        print(f"  [{time.strftime('%H:%M:%S')}] {done}/{len(jobs)} requests, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
