"""Score a trail_anatomy run: CheXbert-label every generated report and compare conditions.

Usage (in .venv-chexone): python -m src.trail_anatomy.score --model chexone
Reference labels: CheXbert on the image's report (sample.json). Predictions: CheXbert on the generated findings,
present or uncertain counted as a positive call. Scored over the 13 findings, pooled over images x findings,
on the images where every condition produced a report. Two reference rules, as hints_mvp: uncertain counts as
positive (headline) and uncertain left out. Changes vs no_hint for every other condition; no_hint_rerun is the
noise floor and neutral_text the effect of any added text. 95% CIs: paired bootstrap over images (1000, seed 0).
Also scored on the subset whose own path has a focus region (where the focus hints actually differ), and a
manipulation check: how often the report names a hinted region, vs the same regions in no_hint.
Writes outputs/trail_anatomy/results_<model>.{md,json} (aggregate numbers only).
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

from ..hints_mvp.data import FINDINGS
from ..hints_mvp.score import changes, metrics
from . import regions_text as RT
from .run_vllm import CONDITIONS, MODELS, QWEN3_CONDITIONS

OUT = Path("outputs/trail_anatomy")
REF = "no_hint"
HINT_OF = {"focus_text_own": "focus_own", "focus_text_other": "focus_other", "focus_text_random": "focus_random",
           "focus_text_own_assist": "focus_own", "focus_mask_own": "focus_own", "focus_mask_other": "focus_other",
           "report_regions": "report_regions"}


def load_records(model, cache, conditions):
    """{(image_id, condition): record} for records made with the model's current settings."""
    want = {k: MODELS[model][k] for k in (("hf", "max_tokens", "temperature") if model == "maira" else ("hf", "max_tokens", "temperature", "top_p", "top_k", "repetition_penalty"))
            if k in MODELS[model]}
    recs = {}
    for f in glob.glob(f"{cache}/{model}/*/*.json"):
        r = json.load(open(f))
        if r["settings"] == want and r["condition"] in conditions:
            recs[(r["image_id"], r["condition"])] = r | {"_path": f}
    return recs


def label_all(recs, model):
    """CheXbert labels for every report, cached in OUT/labels_<model>.json by record path."""
    path = OUT / f"labels_{model}.json"
    cache = json.load(open(path)) if path.exists() else {}
    todo = [r for r in recs.values() if r["_path"] not in cache]
    if todo:
        from .chexbert import Labeler
        lab = Labeler()
        for r in todo:
            cache[r["_path"]] = lab(r["report"])
        json.dump(cache, open(path, "w"))
    return {k: cache[r["_path"]] for k, r in recs.items()}


def stats(y, preds, conds):
    out = {}
    for c in conds:
        for k, v in metrics(y, preds[c]).items():
            out[(c, k)] = v
        if c != REF:
            for k, v in changes(y, preds[REF], preds[c]).items():
                out[(c, k)] = v
            for k in ("sensitivity", "specificity", "f1"):
                out[(c, "d_" + k)] = out[(c, k)] - out[(REF, k)]
    return out


def boot(y, preds, conds, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    S = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        S.append(stats(y[idx], {c: p[idx] for c, p in preds.items()}, conds))
    return {k: tuple(np.nanpercentile([s[k] for s in S], [2.5, 97.5])) for k in S[0]}


def score_set(sample, labels, conds, rule):
    value = {"present": 1.0, "absent": 0.0, "uncertain": 1.0 if rule == "positive" else np.nan}
    y = np.array([[value[s["labels"][f]] for f in FINDINGS] for s in sample])
    preds = {c: np.array([[labels[(s["image_id"], c)][f] != "absent" for f in FINDINGS] for s in sample]) for c in conds}
    return y, preds, stats(y, preds, conds), boot(y, preds, conds)


def manipulation(sample, recs, hints, conds):
    """Share of images (with a non-empty hint) whose report names >=1 hinted region, vs no_hint naming them."""
    out = {}
    for c in conds:
        if c not in HINT_OF:
            continue
        rows = [(s["image_id"], set(hints[s["image_id"]][HINT_OF[c]])) for s in sample if hints[s["image_id"]][HINT_OF[c]]]
        if not rows:
            continue
        hit = np.mean([bool(RT.mentioned(recs[(i, c)]["report"]) & h) for i, h in rows])
        base = np.mean([bool(RT.mentioned(recs[(i, REF)]["report"]) & h) for i, h in rows])
        out[c] = {"images": len(rows), "names_hinted_region": float(hit), "no_hint_names_same": float(base)}
    return out


def f(v, ci=None, signed=False):
    s = (f"{v:+.3f}" if signed else f"{v:.3f}") if v == v else "–"
    return s + (f" [{ci[0]:+.3f}, {ci[1]:+.3f}]" if signed else f" [{ci[0]:.3f}, {ci[1]:.3f}]") if ci else s


def table(res, conds):
    pt, ci = res["point"], res["ci"]
    lines = ["| Condition | fixed | broken | new false | cleared | Sensitivity | Δ sens vs no_hint | Specificity | Δ spec | Δ F1 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for c in conds:
        if c == REF:
            lines.append(f"| **{c}** | | | | | {f(pt[(c, 'sensitivity')], ci[(c, 'sensitivity')])} | | "
                         f"{f(pt[(c, 'specificity')], ci[(c, 'specificity')])} | | |")
            continue
        lines.append(f"| {c} | {pt[(c, 'fixed')]} | {pt[(c, 'broken')]} | {pt[(c, 'new_false')]} | {pt[(c, 'cleared')]} | "
                     f"{f(pt[(c, 'sensitivity')])} | {f(pt[(c, 'd_sensitivity')], ci[(c, 'd_sensitivity')], True)} | "
                     f"{f(pt[(c, 'specificity')])} | {f(pt[(c, 'd_specificity')], ci[(c, 'd_specificity')], True)} | "
                     f"{f(pt[(c, 'd_f1')], ci[(c, 'd_f1')], True)} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sample", default="outputs/trail_anatomy/test/sample.json")
    ap.add_argument("--cache", default="outputs/trail_anatomy/cache")
    ap.add_argument("--conditions", nargs="*", default=CONDITIONS)
    a = ap.parse_args()
    sample = json.load(open(a.sample))
    hints = json.load(open(Path(a.sample).with_name("hints.json")))
    recs = load_records(a.model, a.cache, a.conditions)
    conds = [c for c in a.conditions if any(k[1] == c for k in recs)]
    complete = [s for s in sample if all((s["image_id"], c) in recs and recs[(s["image_id"], c)]["report"] for c in conds)]
    labels = label_all(recs, a.model)
    trunc = {c: sum(recs[k]["finish_reason"] == "length" for k in recs if k[1] == c) for c in conds}
    out = {"model": a.model, "images": len(sample), "scored": len(complete), "conditions": conds, "truncated": trunc}
    md = [f"# trail_anatomy results: {a.model}", "",
          f"{len(complete)} of {len(sample)} test images scored (every condition produced a report). Reference: CheXbert "
          "labels of each image's report. Prediction: CheXbert labels of the generated findings (present or uncertain = "
          "positive call). 13 findings pooled over images x findings; 95% paired bootstrap CIs.", ""]
    subsets = {"all": complete, "own_focus": [s for s in complete if hints[s["image_id"]]["focus_own"]]}
    for name, sub in subsets.items():
        for rule in ("positive", "excluded"):
            y, preds, pt, ci = score_set(sub, labels, conds, rule)
            out[f"{name}_{rule}"] = {"n": len(sub), "positives": int(np.nansum(y)), "point": {f"{k[0]}|{k[1]}": v for k, v in pt.items()},
                                     "ci": {f"{k[0]}|{k[1]}": v for k, v in ci.items()}}
            title = {"all": f"All scored images ({len(sub)})", "own_focus": f"Images whose own path has a focus region ({len(sub)})"}[name]
            md += [f"## {title}; uncertain reference labels {'count as positive' if rule == 'positive' else 'left out'}", "",
                   f"{int(np.nansum(y))} positive reference labels.", "", table({"point": pt, "ci": ci}, conds), ""]
    if "own_focus" in subsets:
        y, preds, *_ = score_set(complete, labels, conds, "positive")
        per = {c: {fd: metrics(y[:, j], preds[c][:, j])["sensitivity"] for j, fd in enumerate(FINDINGS)} for c in conds}
        out["per_finding_sensitivity"] = per
        md += ["## Sensitivity per finding (uncertain = positive)", "",
               "| Finding | positives | " + " | ".join(conds) + " |", "|---|---|" + "---|" * len(conds)]
        for j, fd in enumerate(FINDINGS):
            md.append(f"| {fd} | {int(np.nansum(y[:, j]))} | " + " | ".join(f"{per[c][fd]:.2f}" for c in conds) + " |")
        md.append("")
    man = manipulation(complete, recs, hints, conds)
    out["manipulation"] = man
    md += ["## Does the report name the hinted regions?", "",
           "Share of images with a non-empty hint whose generated report names at least one hinted region, vs the no_hint "
           "report naming the same regions.", "", "| Condition | images | names a hinted region | no_hint names them |", "|---|---|---|---|"]
    md += [f"| {c} | {v['images']} | {v['names_hinted_region']:.2f} | {v['no_hint_names_same']:.2f} |" for c, v in man.items()]
    md += ["", f"Truncated outputs (hit max tokens) per condition: {trunc}", ""]
    (OUT / f"results_{a.model}.md").write_text("\n".join(md))
    json.dump(out, open(OUT / f"results_{a.model}.json", "w"), indent=1, default=float)
    print("\n".join(md))


if __name__ == "__main__":
    main()
