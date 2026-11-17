"""CheXthought Table 4 replication: results stratified by how the no_hint read did on each image.

Usage (in .venv-chexone, after score.py has labeled the reports):
    python -m src.trail_anatomy.score_strata --model qwen3
Strata (from the no_hint read; uncertain reference labels count as positive):
  pathology, 0 correct   image with a finding, no_hint caught none of its findings
  normal, hallucinated   normal image, no_hint called at least one finding
  pathology, >=1 correct image with a finding, no_hint caught at least one
  normal, correct        normal image, no_hint called nothing
Per condition: recovery (the first two strata: share now catching >= 1 finding / now calling nothing), change in correct
findings per image and change in false (hallucinated) findings per image. no_hint_rerun is the control the paper lacks:
the same request again, so its "recovery" is what chance alone gives (regression to the mean). 95% CIs: bootstrap over
images, 2000 resamples; "vs rerun" is a paired bootstrap of the difference.
Writes outputs/trail_anatomy/strata_<model>.{md,json}.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ..hints_mvp.data import FINDINGS
from .score import load_records, OUT

STRATA = ["pathology, 0 correct", "normal, hallucinated", "pathology, >=1 correct", "normal, correct"]


def sets(sample, labels, cond):
    ref = [{f for f in FINDINGS if s["labels"][f] != "absent"} for s in sample]
    pred = [{f for f in FINDINGS if labels[(s["image_id"], cond)][f] != "absent"} for s in sample]
    return ref, pred


def stratum(s, ref, pred):
    if s["group"] == "finding":
        return STRATA[0] if not (pred & ref) else STRATA[2]
    return STRATA[1] if pred else STRATA[3]


def per_image(ref, pred):
    return np.array([len(p & r) for r, p in zip(ref, pred)]), np.array([len(p - r) for r, p in zip(ref, pred)])


def boot_ci(f, n, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = [f(rng.integers(0, n, n)) for _ in range(B)]
    return [float(x) for x in np.nanpercentile(v, [2.5, 97.5])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3")
    ap.add_argument("--sample", default="outputs/trail_anatomy/test/sample.json")
    ap.add_argument("--conditions", nargs="*", default=None)
    a = ap.parse_args()
    from .run_vllm import QWEN3_CONDITIONS
    conds = a.conditions or QWEN3_CONDITIONS
    sample = json.load(open(a.sample))
    recs = load_records(a.model, "outputs/trail_anatomy/cache", conds)
    lab_cache = json.load(open(OUT / f"labels_{a.model}.json"))
    labels = {k: lab_cache[r["_path"]] for k, r in recs.items() if r["_path"] in lab_cache}
    conds = [c for c in conds if any(k[1] == c for k in labels)]
    sample = [s for s in sample if all((s["image_id"], c) in labels for c in conds)]
    ref, base = sets(sample, labels, "no_hint")
    st = np.array([stratum(s, r, p) for s, r, p in zip(sample, ref, base)])
    corr0, hall0 = per_image(ref, base)
    out = {"model": a.model, "images": len(sample), "strata": {k: int((st == k).sum()) for k in STRATA}, "rows": {}}
    rerun = None
    lines = [f"# CheXthought Table 4 replication: {a.model}", "",
             f"{len(sample)} images. Strata from the no_hint read: " + ", ".join(f"{k} {v}" for k, v in out["strata"].items()) + ".",
             "Recovery = share of the stratum now right (pathology: catches >= 1 finding; normal: calls nothing). "
             "ΔCorr / ΔHall = change in correct / false findings per image vs no_hint. no_hint_rerun = the same request "
             "again: its recovery is chance. 95% bootstrap CIs.", ""]
    for k in STRATA:
        idx = np.where(st == k)[0]
        if not len(idx):
            continue
        lines += [f"## {k} ({len(idx)} images)", "", "| Condition | Recovery | vs rerun | ΔCorr | ΔHall |", "|---|---|---|---|---|"]
        for c in conds[1:]:
            _, pred = sets(sample, labels, c)
            corr, hall = per_image(ref, pred)
            if k == STRATA[0]:
                ok = (corr > 0).astype(float)
            elif k == STRATA[1]:
                ok = np.array([len(p) == 0 for p in pred], dtype=float)
            else:
                ok = None
            dc, dh = corr - corr0, hall - hall0
            row = {"dcorr": float(dc[idx].mean()), "dcorr_ci": boot_ci(lambda j: dc[idx][j].mean(), len(idx)),
                   "dhall": float(dh[idx].mean()), "dhall_ci": boot_ci(lambda j: dh[idx][j].mean(), len(idx))}
            rec_s = vs_s = "–"
            if ok is not None:
                row["recovery"] = float(ok[idx].mean())
                row["recovery_ci"] = boot_ci(lambda j: ok[idx][j].mean(), len(idx))
                rec_s = f"{row['recovery']:.0%} [{row['recovery_ci'][0]:.0%}, {row['recovery_ci'][1]:.0%}]"
                if c == "no_hint_rerun":
                    rerun = ok
                elif rerun is not None:
                    d = ok - rerun
                    row["vs_rerun"] = float(d[idx].mean())
                    row["vs_rerun_ci"] = boot_ci(lambda j: d[idx][j].mean(), len(idx))
                    vs_s = f"{row['vs_rerun']:+.0%} [{row['vs_rerun_ci'][0]:+.0%}, {row['vs_rerun_ci'][1]:+.0%}]"
            out["rows"][f"{k}|{c}"] = row
            lines.append(f"| {c} | {rec_s} | {vs_s} | {row['dcorr']:+.2f} [{row['dcorr_ci'][0]:+.2f}, {row['dcorr_ci'][1]:+.2f}] | "
                         f"{row['dhall']:+.2f} [{row['dhall_ci'][0]:+.2f}, {row['dhall_ci'][1]:+.2f}] |")
        lines.append("")
    Path(OUT / f"strata_{a.model}.md").write_text("\n".join(lines))
    json.dump(out, open(OUT / f"strata_{a.model}.json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
