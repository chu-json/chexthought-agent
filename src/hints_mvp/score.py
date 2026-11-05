"""Score a hints_mvp run against the report labels.

Usage: python -m src.hints_mvp.score --config configs/hints_mvp.yaml
Scored over the 13 findings, pooled over images x findings. No Finding is the absence of findings, so it is
not counted. Two label rules: uncertain counts as positive (headline), and uncertain cells left out.
For each condition vs no_hint: fixed (missed -> found), broken (found -> missed), new false findings,
false findings cleared, recovery rate (fixed / no_hint misses), break rate (broken / no_hint found), and the
change in sensitivity, specificity and F1. no_hint_rerun is always shown beside the others: its changes are
chance. 95% CIs come from a paired bootstrap over images. leak_check gets sensitivity, specificity and
Youden's J (sensitivity + specificity - 1) computed within each finding and averaged over findings with both
positive and negative labels: 0 means the coordinates alone say nothing about which images have a finding.
A pooled J would not be 0 without image-specific information, since saying yes only to common findings raises it.
Writes results_file (default results.md) into output_dir; observations.md there is appended if present.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .config import CONDITIONS, load_config
from .data import FINDINGS

REF = "no_hint"
CHANGE_ROWS = ["no_hint_rerun", "own_trail", "other_trail", "random_trail"]
METRICS = ["sensitivity", "specificity", "f1"]
CHANGES = ["fixed", "broken", "new_false", "cleared", "recovery", "break_rate"]
RULES = {"positive": "uncertain counts as positive", "excluded": "uncertain labels left out"}


def label_array(results, rule):
    """Images x 13 findings: 1 positive, 0 negative, nan left out."""
    value = {"present": 1.0, "absent": 0.0, "uncertain": 1.0 if rule == "positive" else np.nan}
    return np.array([[value[r["labels"][f]] for f in FINDINGS] for r in results])


def pred_array(results, cond):
    return np.array([[r["reports"][cond]["report"]["findings"][f]["present"] == "yes" for f in FINDINGS]
                     for r in results])


def _ratio(a, b):
    return a / b if b else np.nan


def metrics(y, p):
    pos, neg = y == 1, y == 0  # nan is neither
    tp, fn = (pos & p).sum(), (pos & ~p).sum()
    fp, tn = (neg & p).sum(), (neg & ~p).sum()
    return {"sensitivity": _ratio(tp, tp + fn), "specificity": _ratio(tn, tn + fp),
            "f1": _ratio(2 * tp, 2 * tp + fp + fn)}


def changes(y, base, new):
    pos, neg = y == 1, y == 0
    fixed, broken = (pos & ~base & new).sum(), (pos & base & ~new).sum()
    return {"fixed": int(fixed), "broken": int(broken),
            "new_false": int((neg & ~base & new).sum()), "cleared": int((neg & base & ~new).sum()),
            "recovery": _ratio(fixed, (pos & ~base).sum()), "break_rate": _ratio(broken, (pos & base).sum())}


def statistics(y, preds):
    """Every number that gets a bootstrap CI, on one (re)sample of images."""
    out = {}
    for c, p in preds.items():
        for k, v in metrics(y, p).items():
            out[(c, k)] = v
    for c in CHANGE_ROWS:
        if c in preds:
            for k, v in changes(y, preds[REF], preds[c]).items():
                out[(c, k)] = v
            for k in METRICS:
                out[(c, f"d_{k}")] = out[(c, k)] - out[(REF, k)]
    if "leak_check" in preds:  # within each finding, so knowing which findings are common earns nothing
        per_finding = [metrics(y[:, j], preds["leak_check"][:, j]) for j in range(y.shape[1])]
        js = [m["sensitivity"] + m["specificity"] - 1 for m in per_finding]
        out[("leak_check", "youden")] = np.nanmean(js) if not np.all(np.isnan(js)) else np.nan
    return out


def bootstrap(y, preds, n_boot, seed):
    """Paired bootstrap over images: every condition is scored on the same resampled images."""
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        samples.append(statistics(y[idx], {c: p[idx] for c, p in preds.items()}))
    return {k: tuple(np.nanpercentile([s[k] for s in samples], [2.5, 97.5])) for k in samples[0]}


def score(cfg):
    out_dir = Path(cfg["output_dir"])
    results = [json.loads(line) for line in open(out_dir / "predictions.jsonl")]
    conds = [c for c in CONDITIONS if c in cfg["conditions"]]
    complete = [r for r in results if all((r["reports"].get(c) or {}).get("report") for c in conds)]
    preds = {c: pred_array(complete, c) for c in conds}
    boot = cfg.get("bootstrap", {"n": 1000, "seed": 0})
    scored = {}
    for rule in RULES:
        y = label_array(complete, rule)
        scored[rule] = {"y": y, "point": statistics(y, preds), "ci": bootstrap(y, preds, boot["n"], boot["seed"])}
    summary = json.load(open(out_dir / "summary.json"))
    md = write_markdown(cfg, summary, results, complete, conds, preds, scored, boot)
    (out_dir / cfg.get("results_file", "results.md")).write_text(md)
    print(md)


def fmt(v, ci=None, digits=3, signed=False):
    if v is None or np.isnan(v):
        return "n/a"
    f = f"{{:{'+' if signed else ''}.{digits}f}}"
    return f.format(v) + (f" [{f.format(ci[0])}, {f.format(ci[1])}]" if ci else "")


def write_markdown(cfg, summary, results, complete, conds, preds, scored, boot):
    rd = summary["reader"]
    groups = {g: sum(r["group"] == g for r in complete) for g in ("finding", "normal")}
    lines = [
        f"# Results: {cfg['name']}", "",
        f"{len(complete)} of {len(results)} images scored ({groups['finding']} with a finding, {groups['normal']} "
        "normal); an image is dropped from every condition if any condition's output failed.",
        f"Reader: {rd['model']} via {rd['backend']} (thinking {rd.get('thinking')}, effort {rd.get('effort')}). "
        f"Labels: the report, labeled by {cfg['labeler']['model']}.",
        f"Scored over 13 findings (No Finding left out), pooled over images x findings. 95% CIs: paired bootstrap "
        f"over images, {boot['n']} resamples, seed {boot['seed']}.", "",
    ]
    for rule, desc in RULES.items():
        y, point, ci = scored[rule]["y"], scored[rule]["point"], scored[rule]["ci"]
        base = preds[REF]
        pos, neg = y == 1, y == 0
        lines += [
            f"## Changes vs no_hint ({desc})", "",
            f"no_hint: {int(pos.sum())} positive labels, of which it found {int((pos & base).sum())} and missed "
            f"{int((pos & ~base).sum())}; {int(neg.sum())} negative labels, of which it called "
            f"{int((neg & base).sum())} (false findings). no_hint_rerun is an identical second request: its "
            "changes are chance.", "",
            "| Condition | fixed (missed→found) | broken (found→missed) | new false findings | false findings cleared "
            "| recovery rate | break rate |", "|---|---|---|---|---|---|---|",
        ]
        for c in CHANGE_ROWS:
            if c in conds:
                lines.append(f"| {c} | {point[(c, 'fixed')]} | {point[(c, 'broken')]} | {point[(c, 'new_false')]} | "
                             f"{point[(c, 'cleared')]} | {fmt(point[(c, 'recovery')], ci[(c, 'recovery')])} | "
                             f"{fmt(point[(c, 'break_rate')], ci[(c, 'break_rate')])} |")
        lines += ["", f"**Metrics ({desc})**", "",
                  "| Condition | Sensitivity | Specificity | F1 | Δ sensitivity vs no_hint | Δ specificity | Δ F1 |",
                  "|---|---|---|---|---|---|---|"]
        for c in conds:
            cells = [fmt(point[(c, k)], ci[(c, k)]) for k in METRICS]
            deltas = [fmt(point[(c, f"d_{k}")], ci[(c, f"d_{k}")], signed=True) if (c, f"d_{k}") in point else ""
                      for k in METRICS]
            lines.append(f"| {c} | " + " | ".join(cells + deltas) + " |")
        if "leak_check" in conds:
            lines += ["", f"leak_check (own trail, no image): sensitivity "
                      f"{fmt(point[('leak_check', 'sensitivity')], ci[('leak_check', 'sensitivity')])}, specificity "
                      f"{fmt(point[('leak_check', 'specificity')], ci[('leak_check', 'specificity')])}. Youden's J "
                      "within findings (sensitivity + specificity − 1 per finding, averaged; 0 = the coordinates "
                      "alone say nothing about which images have a finding): "
                      f"{fmt(point[('leak_check', 'youden')], ci[('leak_check', 'youden')], signed=True)}."]
        lines.append("")

    y = scored["positive"]["y"]
    lines += ["## Per finding (uncertain counts as positive)", "",
              "Sensitivity / specificity per condition. Findings with few positives are noise.", "",
              "| Finding | positives | " + " | ".join(conds) + " |", "|---|---|" + "---|" * len(conds)]
    for j, f in enumerate(FINDINGS):
        cells = []
        for c in conds:
            m = metrics(y[:, j:j + 1], preds[c][:, j:j + 1])
            cells.append(f"{fmt(m['sensitivity'], digits=2)} / {fmt(m['specificity'], digits=2)}")
        lines.append(f"| {f} | {int((y[:, j] == 1).sum())} | " + " | ".join(cells) + " |")

    lines += ["", "## Reliability", "", "| Condition | tasks | first-try parse failures | final failures | refusals "
              "| API errors | cost (all requests) |", "|---|---|---|---|---|---|---|"]
    for c in conds:
        r, p = summary["reliability"][c], summary["per_condition"][c]
        lines.append(f"| {c} | {r['tasks']} | {r['first_try_parse_failures']} | {r['final_failures']} | "
                     f"{r['refusals']} | {r['api_errors']} | ${p['cost_usd']} |")
    lines += ["", f"Cost of all requests in this run: ${summary['cost_usd_all_requests']}; everything hints_mvp has "
              f"spent so far: ${summary['spend_usd_total']}.", ""]
    notes = Path(cfg["output_dir"]) / "observations.md"  # hand-written, so rescoring doesn't erase them
    if notes.exists():
        lines += [notes.read_text()]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    score(load_config(ap.parse_args().config))
