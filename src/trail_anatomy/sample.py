"""Draw the trail_anatomy test sample: new images, labeled by CheXbert, with every radiologist's path.

Usage (in .venv-chexone): python -m src.trail_anatomy.sample [--n-finding 200 --n-normal 100]
Same rules as hints_mvp (data.candidate_order with seed 0: frontal, one image per patient, one usable path,
snapped paths skipped), minus the patients in the hints_mvp sample (the dev set). Reports are labeled with
CheXbert in candidate order until both groups are full:
- finding: at least one of the 13 findings present; normal: all 13 absent (no devices either);
- images whose only findings are uncertain are in neither group.
Each entry keeps the chosen ("own") path and every other usable path of the same image (other radiologists),
an "other" path (another sampled image's own path, a single-cycle derangement) and a "random" path (as many
uniform points). Writes output_dir/sample.json and image_paths.txt. Stays local (research use agreement).
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from ..hints_mvp import data
from ..hints_mvp.sample import group
from .chexbert import Labeler

SEED = 0


def build(csv, dev_sample, out_dir, n_finding, n_normal):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traces = data.load_traces(csv)
    cot = pd.read_csv(Path(csv).expanduser(), usecols=["chain_of_thought_reasoning"])["chain_of_thought_reasoning"]
    dev_patients = {s["image_id"].split("_")[0] for s in json.load(open(dev_sample))}
    order = [(i, t) for i, t in data.candidate_order(traces, SEED) if i.split("_")[0] not in dev_patients]
    usable = traces[traces["points"].map(lambda p: len(p) > 0 and not data.is_snapped(p))]
    by_image = usable.groupby("image_id")["trace_id"].apply(list).to_dict()

    labeler = Labeler()
    chosen = {"finding": [], "normal": []}
    want = {"finding": n_finding, "normal": n_normal}
    n_labeled = 0
    for i, t in order:
        if all(len(chosen[g]) >= n for g, n in want.items()):
            break
        labels = labeler(traces.at[t, "report"])
        n_labeled += 1
        g = group(labels)
        if g and len(chosen[g]) < want[g]:
            others = [{"trace_id": int(u), "points": traces.at[u, "points"]} for u in by_image[i] if u != t]
            chosen[g].append({
                "image_id": i, "group": g, "path": data.rel_path(i), "trace_id": int(t), "labels": labels,
                "report": traces.at[t, "report"], "reasoning": cot.at[t],
                "trails": {"own": {"trace_id": int(t), "points": traces.at[t, "points"]}},
                "other_radiologists": others,
            })
    sample = chosen["finding"] + chosen["normal"]
    ids = [s["image_id"] for s in sample]
    other = data.derangement(ids, SEED)
    by_id = {s["image_id"]: s for s in sample}
    for s in sample:
        o = by_id[other[s["image_id"]]]
        s["trails"]["other"] = {"from_image": o["image_id"], "points": o["trails"]["own"]["points"]}
        s["trails"]["random"] = {"points": data.random_trail(len(s["trails"]["own"]["points"]), SEED, s["image_id"])}
    json.dump(sample, open(out_dir / "sample.json", "w"))
    (out_dir / "image_paths.txt").write_text("\n".join(s["path"] for s in sample) + "\n")
    summary = {"labeled": n_labeled, "groups": dict(Counter(s["group"] for s in sample)),
               "other_radiologists_per_image": dict(Counter(len(s["other_radiologists"]) for s in sample)),
               "positives": {f: sum(s["labels"][f] != "absent" for s in sample) for f in data.FINDINGS}}
    json.dump(summary, open(out_dir / "sample_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="~/data/chexthought/chexthoughtdataset.csv")
    ap.add_argument("--dev-sample", default="outputs/hints_mvp/sample.json")
    ap.add_argument("--out", default="outputs/trail_anatomy/test")
    ap.add_argument("--n-finding", type=int, default=200)
    ap.add_argument("--n-normal", type=int, default=100)
    a = ap.parse_args()
    build(a.csv, a.dev_sample, a.out, a.n_finding, a.n_normal)
