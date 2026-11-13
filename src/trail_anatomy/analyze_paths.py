"""Path analysis: styles, agreement between radiologists, focus vs written reasoning, focus vs labels.

Usage (main env): python -m src.trail_anatomy.analyze_paths
Pool: every usable path of the dev images (hints_mvp sample) and the test images. The baseline (how
radiologists usually spread their clicks) is the average over the whole pool; it uses no labels.
Tuning: alpha and tau are chosen on the dev images only, by agreement between two radiologists' focus regions
on the same image beyond chance (no labels used), then fixed and everything is reported on the test images.
Writes outputs/trail_anatomy/path_analysis.{json,md} (aggregate numbers only) and focus_params.json.
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from ..hints_mvp import data
from . import anatomy as A
from . import focus as F
from . import regions_text as RT

OUT = Path("outputs/trail_anatomy")
MASKS = OUT / "masks"
rng0 = np.random.default_rng(0)


def load_pool(csv, dev_sample, test_sample):
    traces = data.load_traces(csv)
    import pandas as pd
    cot = pd.read_csv(Path(csv).expanduser(), usecols=["chain_of_thought_reasoning"])["chain_of_thought_reasoning"]
    usable = traces[traces["points"].map(lambda p: len(p) > 0 and not data.is_snapped(p))]
    pool = []
    for split, path in (("dev", dev_sample), ("test", test_sample)):
        for s in json.load(open(path)):
            lab = A.load_partition(MASKS, s["image_id"])
            own = s["trace_id"]
            rows = usable[usable["image_id"] == s["image_id"]]
            for t, pts in zip(rows["trace_id"], rows["points"]):
                d = A.describe(pts, lab)
                pool.append({"split": split, "image_id": s["image_id"], "trace_id": int(t), "own": int(t) == own,
                             "counts": F.counts(pts, lab), "labels": s["labels"], "reasoning": cot.at[t],
                             **{k: v for k, v in d.items() if k != "counts"}})
    return pool


def style(p):
    if p["n_points"] <= 25:
        return "sparse"
    if p["median_step"] <= 0.07 and p["n_points"] >= 40:
        return "tracing"
    return "search"


def jaccard(a, b):
    a, b = set(a), set(b)
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def agreement(paths, b, params, n_perm=200):
    """Mean Jaccard of focus sets between two paths of the same image, vs paths of different images."""
    by_img = {}
    for p in paths:
        by_img.setdefault(p["image_id"], []).append(F.focus_regions(p["counts"], b, **params))
    pairs = [(x, y) for fs in by_img.values() if len(fs) > 1 for x, y in combinations(fs, 2)]
    if not pairs:
        return None
    same = np.mean([jaccard(x, y) for x, y in pairs])
    firsts, seconds = [x for x, _ in pairs], [y for _, y in pairs]
    null = []
    for _ in range(n_perm):
        perm = rng0.permutation(len(seconds))
        null.append(np.mean([jaccard(firsts[k], seconds[j]) for k, j in enumerate(perm) if j != k]))
    null = np.array(null)
    return {"pairs": len(pairs), "same_image": float(same), "different_images": float(null.mean()),
            "excess": float(same - null.mean()), "p": float((null >= same).mean())}


def reasoning_check(paths, b, params):
    """How often a region is named in the radiologist's own reasoning, focus regions vs other touched regions."""
    hit_f = n_f = hit_o = n_o = 0
    for p in paths:
        named = RT.mentioned(p["reasoning"])
        fr = set(F.focus_regions(p["counts"], b, **params))
        touched = {A.NAMES[r] for r in np.nonzero(p["counts"])[0] if r != A.OTHER}
        for r in touched:
            if r in fr:
                n_f += 1; hit_f += r in named
            else:
                n_o += 1; hit_o += r in named
    return {"focus_named": hit_f / max(n_f, 1), "n_focus": n_f, "other_touched_named": hit_o / max(n_o, 1), "n_other": n_o}


RELEVANT = {"Pleural Effusion": ["right lower lung", "left lower lung", "right hemidiaphragm", "left hemidiaphragm"],
            "Cardiomegaly": ["heart"], "Enlarged Cardiomediastinum": ["mediastinum", "heart"],
            "Edema": ["hila", "right mid lung", "left mid lung"],
            "Lung Opacity": ["right lower lung", "left lower lung", "right mid lung", "left mid lung"],
            "Atelectasis": ["right lower lung", "left lower lung"], "Consolidation": ["right lower lung", "left lower lung"],
            "Pneumothorax": ["right upper lung", "left upper lung"], "Support Devices": ["trachea", "mediastinum"]}


def label_check(paths, b, params):
    """Per finding: share of positive vs negative images whose own path has a focus region in the relevant regions."""
    out = {}
    own = [p for p in paths if p["own"]]
    for f, regs in RELEVANT.items():
        y = np.array([p["labels"][f] != "absent" for p in own])
        hit = np.array([bool(set(F.focus_regions(p["counts"], b, **params)) & set(regs)) for p in own])
        if y.sum() >= 5:
            out[f] = {"positives": int(y.sum()), "focus_rate_pos": float(hit[y].mean()), "focus_rate_neg": float(hit[~y].mean())}
    return out


def main():
    pool = load_pool("~/data/chexthought/chexthoughtdataset.csv", "outputs/hints_mvp/sample.json",
                     "outputs/trail_anatomy/test/sample.json")
    b = F.baseline([p["counts"] for p in pool])
    dev = [p for p in pool if p["split"] == "dev"]
    test = [p for p in pool if p["split"] == "test"]
    grid = [{"alpha": a, "tau": t, "k": 3, "min_count": 3} for a in (5.0, 10.0, 20.0) for t in (1.25, 1.5, 2.0)]
    tuning = [{**g, **agreement(dev, b, g, n_perm=50)} for g in grid]
    best = max(tuning, key=lambda r: r["excess"])
    params = {k: best[k] for k in ("alpha", "tau", "k", "min_count")}
    json.dump({"params": params, "baseline": dict(zip(A.NAMES, map(float, b)))}, open(OUT / "focus_params.json", "w"), indent=2)

    res = {"pool_paths": len(pool), "dev_paths": len(dev), "test_paths": len(test), "params": params, "tuning_dev": tuning,
           "baseline_share": dict(zip(A.NAMES, map(float, b)))}
    for name, ps in (("dev", dev), ("test", test)):
        styles = {}
        for p in ps:
            styles.setdefault(style(p), []).append(len(F.focus_regions(p["counts"], b, **params)))
        res[name] = {
            "agreement": agreement(ps, b, params),
            "reasoning": reasoning_check(ps, b, params),
            "styles": {s: {"paths": len(v), "mean_focus_regions": float(np.mean(v)), "no_focus": float(np.mean(np.array(v) == 0))}
                       for s, v in styles.items()},
            "n_focus": {int(k): int(v) for k, v in zip(*np.unique([len(F.focus_regions(p["counts"], b, **params)) for p in ps], return_counts=True))},
            "regions_touched_median": float(np.median([p["n_regions"] for p in ps])),
            "points_median": float(np.median([p["n_points"] for p in ps])),
            "labels": label_check(ps, b, params),
        }
        fr = {}
        for p in ps:
            if p["own"]:
                for r in F.focus_regions(p["counts"], b, **params):
                    fr[r] = fr.get(r, 0) + 1
        res[name]["focus_frequency_own"] = fr
    json.dump(res, open(OUT / "path_analysis.json", "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("pool_paths", "params")}, indent=1))
    for name in ("dev", "test"):
        r = res[name]
        print(f"\n[{name}] agreement {r['agreement']}\n  reasoning {r['reasoning']}\n  styles {r['styles']}\n  n_focus {r['n_focus']}")
        for f, v in r["labels"].items():
            print(f"  {f:27s} pos {v['positives']:3d}  focus-in-relevant-region: pos {v['focus_rate_pos']:.2f} vs neg {v['focus_rate_neg']:.2f}")


if __name__ == "__main__":
    main()
