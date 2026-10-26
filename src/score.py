"""Score a run: metrics per condition, changes vs A and B, box quality, paired bootstrap CIs.

Usage: python -m src.score --config configs/v0.yaml
Writes results.md and predictions.csv into the run's output_dir.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from . import boxes as bx
from . import data
from .run import BOX_CONDITIONS, CONDITIONS, load_config

LABEL = {
    "alone": "A. alone", "look_again": "B. look_again", "repeat": "B. repeat", "random_boxes": "C. random_boxes",
    "checklist_boxes": "D. checklist_boxes", "helper_boxes": "E. helper_boxes", "oracle_boxes": "F. oracle_boxes",
}
METRICS = ["sensitivity", "specificity", "f1", "auroc"]
REFS = ("alone", "look_again", "repeat")  # conditions that answer changes are counted against
HIT_FRACTION = 0.2  # a box "hits" if at least 20% of its area is inside a ground-truth mask


def report_for(r, cond):
    """The report for a condition, or None if that output failed."""
    if cond == "alone":
        return (r.get("pass1") or {}).get("report")
    return (r["pass2"].get(cond) or {}).get("report")


# --- Metrics on arrays of shape (images, findings). ---

def _ratio(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b > 0, a / np.maximum(b, 1), np.nan)


def auroc(y, s):
    """AUROC per column (Mann-Whitney U with average ranks for ties); nan if a column has one class."""
    out = np.full(y.shape[1], np.nan)
    for j in range(y.shape[1]):
        pos, n_pos = y[:, j] == 1, int(y[:, j].sum())
        n_neg = len(pos) - n_pos
        if n_pos and n_neg:
            ranks = rankdata(s[:, j])
            out[j] = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return out


def finding_metrics(y, p, s):
    """Per-finding sensitivity, specificity, F1 (from yes/no) and AUROC (from probability)."""
    y, p = y.astype(bool), p.astype(bool)
    tp, fp = (y & p).sum(0), (~y & p).sum(0)
    fn, tn = (y & ~p).sum(0), (~y & ~p).sum(0)
    return {
        "sensitivity": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "auroc": auroc(y.astype(int), s),
    }


def macro(m):
    return {k: float(np.nanmean(v)) if not np.all(np.isnan(v)) else np.nan for k, v in m.items()}


def changes(y, p_ref, p_new):
    """Answer changes from a reference condition, counted over images x findings."""
    y, a, b = y.astype(bool), p_ref.astype(bool), p_new.astype(bool)
    fixed = int((~a & b & y).sum())  # FN -> TP
    added = int((~a & b & ~y).sum())  # TN -> FP
    return {"fixed": fixed, "added": added, "net": fixed - added,
            "lost": int((a & ~b & y).sum()),  # TP -> FN
            "cleared": int((a & ~b & ~y).sum())}  # FP -> TN


def box_quality(r, cond, masks, missed):
    """Per-image counts for hit rate and miss coverage."""
    union = None
    for m in masks.values():
        union = m if union is None else union | m
    out = {"hits": 0, "boxes": 0, "real_hits": 0, "real_boxes": 0, "covered": 0, "covered_strict": 0, "misses": 0}
    for b in r["boxes"][cond]:
        hit = union is not None and bx.fraction_inside(b["box"], union) >= HIT_FRACTION
        out["hits"] += hit
        out["boxes"] += 1
        if b["source"] != "pad":
            out["real_hits"] += hit
            out["real_boxes"] += 1
    for f in missed:
        if f in masks:  # skip misses whose mask is empty
            out["misses"] += 1
            out["covered"] += any(bx.overlaps(b["box"], masks[f]) for b in r["boxes"][cond])
            out["covered_strict"] += any(bx.fraction_inside(b["box"], masks[f]) >= HIT_FRACTION
                                         for b in r["boxes"][cond])
    return out


def statistics(y, preds, probs, boxq, conds):
    """Every number that gets a bootstrap CI, computed on one (re)sample."""
    out = {}
    for c in conds:
        for k, v in macro(finding_metrics(y, preds[c], probs[c])).items():
            out[(c, k)] = v
    for c in conds:  # paired metric changes from each reference condition
        for ref in REFS:
            if ref in conds and c != ref and c != "alone":
                for k in METRICS:
                    out[(c, f"d_{k}_vs_{ref}")] = out[(c, k)] - out[(ref, k)]
    for c in conds:
        for ref in REFS:
            if ref in conds and c != ref:
                out[(c, f"net_vs_{ref}")] = changes(y, preds[ref], preds[c])["net"]
        if c in boxq:
            q = boxq[c]
            out[(c, "hit_rate")] = q["hits"].sum() / max(q["boxes"].sum(), 1)
            out[(c, "real_hit_rate")] = q["real_hits"].sum() / q["real_boxes"].sum() if q["real_boxes"].sum() else np.nan
            for k in ("covered", "covered_strict"):
                name = k.replace("covered", "miss_coverage")
                out[(c, name)] = q[k].sum() / q["misses"].sum() if q["misses"].sum() else np.nan
    return out


def bootstrap(y, preds, probs, boxq, conds, n_boot, seed):
    """Paired bootstrap over images: the same resampled images are used for every condition."""
    rng = np.random.default_rng(seed)
    n = len(y)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        samples.append(statistics(
            y[idx], {c: v[idx] for c, v in preds.items()}, {c: v[idx] for c, v in probs.items()},
            {c: {k: v[idx] for k, v in q.items()} for c, q in boxq.items()}, conds,
        ))
    return {k: tuple(np.nanpercentile([s[k] for s in samples], [2.5, 97.5])) for k in samples[0]}


def score(cfg):
    out_dir = Path(cfg["output_dir"])
    results = [json.loads(line) for line in open(out_dir / "predictions.jsonl")]
    conds = [c for c in CONDITIONS if c in cfg["conditions"]]
    complete = [r for r in results if all(report_for(r, c) for c in conds)]
    F = data.FINDINGS

    y = np.array([[r["labels"][f] for f in F] for r in complete])
    preds = {c: np.array([[report_for(r, c)["findings"][f]["present"] == "yes" for f in F] for r in complete])
             for c in conds}
    probs = {c: np.array([[report_for(r, c)["findings"][f]["probability"] for f in F] for r in complete], float)
             for c in conds}

    # Box quality needs the ground-truth masks of the scored images.
    rles = data.load_masks(cfg["data_root"], cfg["split"])
    boxq = {c: {} for c in conds if c in BOX_CONDITIONS}
    per_image = {c: [] for c in boxq}
    for r in complete:
        masks = {f: m for f, rle in rles.get(r["image_id"], {}).items() if (m := data.decode(rle)).any()}
        missed = [f for f in F if r["labels"][f] and r["pass1"]["report"]["findings"][f]["present"] == "no"]
        for c in boxq:
            per_image[c].append(box_quality(r, c, masks, missed))
    for c in boxq:
        boxq[c] = {k: np.array([q[k] for q in per_image[c]]) for k in per_image[c][0]} if complete else {}

    point = statistics(y, preds, probs, boxq, conds)
    boot = cfg.get("bootstrap", {"n": 1000, "seed": 0})
    ci = bootstrap(y, preds, probs, boxq, conds, boot["n"], boot["seed"])
    per_finding = {c: finding_metrics(y, preds[c], probs[c]) for c in conds}
    change_tables = {ref: {c: changes(y, preds[ref], preds[c]) for c in conds if c != ref}
                     for ref in REFS if ref in conds}

    summary = json.load(open(out_dir / "summary.json"))
    md = write_markdown(cfg, summary, results, complete, y, conds, point, ci, per_finding, change_tables, boxq, boot)
    (out_dir / "results.md").write_text(md)
    write_csv(results, conds, out_dir / "predictions.csv", {r["image_id"] for r in complete})
    print(md)


def fmt_ci(v, ci, digits=3):
    if v is None or np.isnan(v):
        return "n/a"
    return f"{v:.{digits}f} [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def write_markdown(cfg, summary, results, complete, y, conds, point, ci, per_finding, change_tables, boxq, boot):
    rd, hp = summary["reader"], summary["helper"]
    lines = [
        f"# Results: {cfg['name']}", "",
        f"{len(complete)} of {len(results)} {cfg['split']} images scored (images where any condition's output "
        f"failed are dropped from every condition, to keep comparisons paired).",
        f"Reader: {rd['model']} (thinking {rd['thinking']}, effort {rd['effort']}, temperature: API default). "
        f"Helper: {hp['model']} (thinking {hp['thinking']}, effort {hp['effort']}).",
        f"Prompts: {summary.get('prompts', 'v0')}; pass 2: {summary.get('pass2_mode', 'revise')} "
        f"({'reader sees its earlier report' if summary.get('pass2_mode', 'revise') == 'revise' else 'fresh read without its earlier report'}); "
        f"boxes {'padded to 3 with random boxes' if summary.get('pad_boxes', True) else 'not padded (real boxes only)'}.",
        f"95% CIs: paired bootstrap over images, {boot['n']} resamples, seed {boot['seed']}.", "",
        "## Macro-averaged metrics (10 findings)", "",
        "| Condition | Sensitivity | Specificity | F1 | AUROC |", "|---|---|---|---|---|",
    ]
    for c in conds:
        lines.append(f"| {LABEL[c]} | " + " | ".join(fmt_ci(point[(c, k)], ci[(c, k)]) for k in METRICS) + " |")

    for ref in REFS:
        rows = [c for c in conds if (c, f"d_auroc_vs_{ref}") in point]
        if not rows:
            continue
        lines += ["", f"**Change from {LABEL[ref].split('.')[0]} ({LABEL[ref]}), paired 95% CI**", "",
                  "| Condition | Sensitivity | Specificity | F1 | AUROC |", "|---|---|---|---|---|"]
        for c in rows:
            lines.append(f"| {LABEL[c]} | " + " | ".join(
                fmt_ci(point[(c, f'd_{k}_vs_{ref}')], ci[(c, f'd_{k}_vs_{ref}')]) for k in METRICS) + " |")

    lines += ["", "## Answer changes vs A and vs B", "",
              "Counted over images x findings. fixed = FN→TP, added = TN→FP (new false alarm), "
              "net = fixed − added; lost = TP→FN, cleared = FP→TN.", ""]
    for ref, table in change_tables.items():
        lines += [f"**vs {LABEL[ref]}**", "", "| Condition | fixed | added | net [95% CI] | lost | cleared |",
                  "|---|---|---|---|---|---|"]
        for c, ch in table.items():
            if ref != "alone" and c == "alone":
                continue
            net_ci = ci[(c, f"net_vs_{ref}")]
            lines.append(f"| {LABEL[c]} | {ch['fixed']} | {ch['added']} | {ch['net']} [{net_ci[0]:.0f}, {net_ci[1]:.0f}]"
                         f" | {ch['lost']} | {ch['cleared']} |")
        lines.append("")

    if boxq:
        lines += ["## Box quality", "",
                  f"Hit rate: share of boxes with at least {HIT_FRACTION:.0%} of their area inside any ground-truth mask. "
                  "Miss coverage: share of the reader's pass-1 misses where a box overlaps that finding's mask "
                  f"(any overlap); strict: a box has at least {HIT_FRACTION:.0%} of its area inside that finding's mask.",
                  "",
                  "| Condition | Hit rate | Hit rate, non-padding boxes (n) | Miss coverage | Miss coverage, strict "
                  "| Pass-1 misses |",
                  "|---|---|---|---|---|---|"]
        for c in boxq:
            q = boxq[c]
            lines.append(f"| {LABEL[c]} | {fmt_ci(point[(c, 'hit_rate')], ci[(c, 'hit_rate')])} | "
                         f"{fmt_ci(point[(c, 'real_hit_rate')], ci[(c, 'real_hit_rate')])} ({int(q['real_boxes'].sum())}) | "
                         f"{fmt_ci(point[(c, 'miss_coverage')], ci[(c, 'miss_coverage')])} | "
                         f"{fmt_ci(point[(c, 'miss_coverage_strict')], ci[(c, 'miss_coverage_strict')])} | "
                         f"{int(q['misses'].sum())} |")
        lines.append("")

    lines += ["## Per-finding metrics", ""]
    n_pos = y.sum(0)
    for k in METRICS:
        lines += [f"**{k}**", "", "| Finding | positives | " + " | ".join(LABEL[c][:2].strip(". ") for c in conds) + " |",
                  "|---|---|" + "---|" * len(conds)]
        for j, f in enumerate(data.FINDINGS):
            vals = " | ".join("n/a" if np.isnan(per_finding[c][k][j]) else f"{per_finding[c][k][j]:.2f}" for c in conds)
            lines.append(f"| {f} | {int(n_pos[j])} | {vals} |")
        lines.append("")

    rel = summary["reliability"]
    lines += ["## Reliability", "",
              f"- Reader: {rel['reader']}", f"- Helper: {rel['helper']}",
              f"- Helper real boxes per image (before padding): {summary['helper_real_boxes']}",
              f"- Tokens: {summary['tokens']}; cost of all requests: ${summary['cost_usd_all_requests']}", ""]
    # Hand-written observations live in observations.md so rescoring doesn't erase them.
    notes = Path(cfg["output_dir"]) / "observations.md"
    if notes.exists():
        lines += [notes.read_text()]
    return "\n".join(lines)


def write_csv(results, conds, path, complete_ids):
    rows = []
    for r in results:
        for c in conds:
            rep = report_for(r, c)
            for f in data.FINDINGS:
                v = rep["findings"][f] if rep else {}
                rows.append({"image_id": r["image_id"], "condition": c, "finding": f, "label": r["labels"][f],
                             "present": v.get("present"), "probability": v.get("probability"),
                             "location": v.get("location"), "scored": r["image_id"] in complete_ids})
    pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    score(load_config(ap.parse_args().config))
