"""Build every hint for every image of a sample: focus regions (own / other / random), report regions, coords.

Usage (main env): python -m src.trail_anatomy.hints --sample outputs/trail_anatomy/test/sample.json
Uses the focus parameters and baseline fixed by analyze_paths (focus_params.json).
- focus_own: focus regions of this image's own path.
- focus_other: focus regions of another sampled image's own path (the sample's "other" derangement).
- focus_random: as many regions as focus_own, drawn with each region's frequency as a focus region over the
  sample's own paths, so naming commonly emphasized regions earns nothing.
- report_regions: regions named in the report's positive (non-negated) findings/impression sentences, capped at 3
  in a fixed order (the ceiling: "attention" exactly where the report places its findings).
Writes hints.json next to the sample.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import anatomy as A
from . import focus as F
from . import regions_text as RT

ORDER = [n for n in A.NAMES if n != "other"]


def build(sample_path, params_path="outputs/trail_anatomy/focus_params.json"):
    sample = json.load(open(sample_path))
    fp = json.load(open(params_path))
    b = np.array([fp["baseline"][n] for n in A.NAMES])
    p = fp["params"]
    own = {}
    for s in sample:
        lab = A.load_partition(A_MASKS, s["image_id"])
        own[s["image_id"]] = F.focus_regions(F.counts(s["trails"]["own"]["points"], lab), b, **p)
    freq = {}
    for regs in own.values():
        for r in regs:
            freq[r] = freq.get(r, 0) + 1
    out = {}
    for s in sample:
        i = s["image_id"]
        rng = np.random.default_rng(int.from_bytes(i.encode(), "little") % 2**32)  # fixed per image
        rep = RT.mentioned(RT.report_findings_text(s.get("report", "")), positive_only=True)
        out[i] = {
            "focus_own": own[i],
            "focus_other": own[s["trails"]["other"]["from_image"]],
            "focus_random": F.random_regions(len(own[i]), freq, rng) if own[i] else [],
            "report_regions": [r for r in ORDER if r in rep][:3],
            "coords_own": s["trails"]["own"]["points"],
        }
    path = Path(sample_path).with_name("hints.json")
    json.dump(out, open(path, "w"))
    return out, path


A_MASKS = "outputs/trail_anatomy/masks"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="outputs/trail_anatomy/test/sample.json")
    a = ap.parse_args()
    out, path = build(a.sample)
    s = {x["image_id"]: x for x in json.load(open(a.sample))}
    for k in ("focus_own", "focus_other", "focus_random", "report_regions"):
        n = [len(v[k]) for v in out.values()]
        by_group = {g: float(np.mean([len(v[k]) > 0 for i, v in out.items() if s[i]["group"] == g])) for g in ("finding", "normal")}
        print(f"{k:15s} regions per image {dict(zip(*np.unique(n, return_counts=True)))}  share with any: {by_group}")
    print("wrote", path)
