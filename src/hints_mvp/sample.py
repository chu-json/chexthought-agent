"""Step 1: label reports with an LLM and draw the sample.

Usage: python -m src.hints_mvp.sample --config configs/hints_mvp.yaml
Candidates are frontal images in a fixed random order, one per patient, each with one usable trace
(data.candidate_order). Reports are labeled in that order, label_batch at a time, and each image goes in a group:
- finding: at least one of the 13 findings is present in the report.
- normal: all 13 are absent (no devices either). Reports whose only findings are uncertain are in neither group.
Labeling stops once there are n_normal normal images (about 4% of reports are normal), or after max_labeled
reports. The sample is the first n_normal normal images (fewer if labeling stopped first) and the first images
with a finding up to n_images, both in candidate order, so each group is a random sample of its kind.
Writes sample.json, image_paths.txt, labels.jsonl, spot_check.md and sample_summary.json to output_dir.
Labels are cached like every other request, so a rerun is free.
"""
import argparse
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

from ..run import BudgetExceeded, Caller
from . import data, prompts
from .config import load_config


def group(labels):
    if any(labels[f] == "present" for f in data.FINDINGS):
        return "finding"
    if all(labels[f] == "absent" for f in data.FINDINGS):
        return "normal"
    return None


def label_reports(cfg, caller, traces, candidates):
    """Label each candidate's report. Returns one row per candidate, in candidate order."""
    def label(candidate):
        i, t = candidate
        return caller.ask("labeler", "report_labels", i, prompts.LABELER_SYSTEM,
                          [prompts.labeler_text(traces.at[t, "report"])], prompts.parse_labels, [])

    with ThreadPoolExecutor(cfg["concurrency"]) as pool:
        outs = list(pool.map(label, candidates))
    errors = [info["error"] for _, info in outs if info["error"]]
    if errors:  # a missing label would change which images are chosen; rerun instead (cached)
        raise RuntimeError(f"{len(errors)} label requests failed, e.g. {errors[0]}. Rerun to retry.")
    return [{"image_id": i, "trace_id": t, "group": group(out["labels"]) if out else None,  # refused or
             "labels": out and out["labels"], "evidence": out and out["evidence"],  # unparseable: no group
             "refusal": info["refusal"], "parse_errors": info["parse_errors"]}
            for (i, t), (out, info) in zip(candidates, outs)]


def label_until_full(cfg, caller, traces, order):
    """Label reports in candidate order, a batch at a time, until both groups can be filled or max_labeled."""
    labeled, limit = [], min(cfg["max_labeled"], len(order))
    want = {"normal": cfg["n_normal"], "finding": cfg["n_images"] - cfg["n_normal"]}
    while len(labeled) < limit:
        labeled += label_reports(cfg, caller, traces, order[len(labeled):min(len(labeled) + cfg["label_batch"], limit)])
        have = Counter(r["group"] for r in labeled)
        print(f"  labeled {len(labeled)}: {have['finding']} finding, {have['normal']} normal, "
              f"hints_mvp spend ${caller.ledger['usd']:.2f}", flush=True)
        if all(have[g] >= n for g, n in want.items()):
            break
    return labeled


def choose(labeled, n_images, n_normal):
    """The first n_normal normal images, then images with a finding up to n_images, in candidate order."""
    normal = [r for r in labeled if r["group"] == "normal"][:n_normal]
    finding = [r for r in labeled if r["group"] == "finding"][:n_images - len(normal)]
    return finding + normal


def build(cfg):
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]
    traces = data.load_traces(cfg["csv"])
    n_snapped = int(traces["points"].map(data.is_snapped).sum())
    order = data.candidate_order(traces, seed)
    print(f"{len(traces)} frontal traces ({n_snapped} snapped, skipped); "
          f"{len(order)} candidate images, one per patient")

    caller = Caller(cfg, roles=("labeler",))
    try:
        labeled = label_until_full(cfg, caller, traces, order)
    except BudgetExceeded as e:
        print(f"STOPPED: budget cap reached ({e}). Labels so far are cached; raise max_usd and rerun.")
        return None
    with open(out_dir / "labels.jsonl", "w") as f:
        for row in labeled:
            f.write(json.dumps(row) + "\n")
    rows = choose(labeled, cfg["n_images"], cfg["n_normal"])
    n_normal = sum(r["group"] == "normal" for r in rows)
    if len(rows) < cfg["n_images"] or n_normal < cfg["n_normal"]:
        print(f"WARNING: {len(rows)} images ({n_normal} normal) after {len(labeled)} labeled reports")

    rows = sorted(rows, key=lambda r: r["image_id"])
    ids = [r["image_id"] for r in rows]
    trace_of = {r["image_id"]: r["trace_id"] for r in rows}
    other = data.derangement(ids, seed)
    sample = []
    for r in rows:
        i, t = r["image_id"], r["trace_id"]
        own = traces.at[t, "points"]
        sample.append({
            "image_id": i, "group": r["group"], "path": data.rel_path(i), "trace_id": t,
            "labels": r["labels"], "evidence": r["evidence"], "report": traces.at[t, "report"],
            "trails": {
                "own": {"trace_id": t, "points": own},
                "other": {"from_image": other[i], "trace_id": trace_of[other[i]],
                          "points": traces.at[trace_of[other[i]], "points"]},
                "random": {"points": data.random_trail(len(own), seed, i)},
            },
        })
    json.dump(sample, open(out_dir / "sample.json", "w"), indent=1)
    (out_dir / "image_paths.txt").write_text("\n".join(s["path"] for s in sample) + "\n")
    write_spot_check(sample, seed, out_dir / "spot_check.md")

    labeler_cost = sum(e["cost_usd"] for e in caller.log)
    groups = Counter(r["group"] for r in labeled)
    summary = {
        "frontal_traces": len(traces), "snapped_traces": n_snapped, "candidates": len(order),
        "reports_labeled": len(labeled), "groups_among_labeled": {str(k): v for k, v in groups.items()},
        "refusals": sum(r["refusal"] for r in labeled),
        "unparseable": sum(r["labels"] is None and not r["refusal"] for r in labeled),
        "sample": {g: sum(s["group"] == g for s in sample) for g in ("finding", "normal")},
        "positives_in_sample": {f: {v: sum(s["labels"][f] == v for s in sample) for v in ("present", "uncertain")}
                                for f in data.CHEXPERT},
        "trail_points": sorted(len(s["trails"]["own"]["points"]) for s in sample),
        "labeler_cost_usd_all_requests": round(labeler_cost, 4),
        "spend_usd_total": round(caller.ledger["usd"], 4),
    }
    json.dump(summary, open(out_dir / "sample_summary.json", "w"), indent=2)
    print_summary(summary)
    return sample


def write_spot_check(sample, seed, path, n=20):
    """n images drawn at random: up to half of them normal, the rest with a finding."""
    rng = random.Random(f"{seed}:spot_check")
    members = {g: [s for s in sample if s["group"] == g] for g in ("finding", "normal")}
    take = {"normal": min(n // 2, len(members["normal"]))}
    take["finding"] = min(n - take["normal"], len(members["finding"]))
    lines = ["# Report labels: spot check", "",
             f"{take['finding']} images with a finding and {take['normal']} normal images, drawn at random from the "
             "sample. Labels other than absent are listed with the labeler's evidence quote; every unlisted finding "
             "is absent.", ""]
    for g in ("finding", "normal"):
        for s in rng.sample(members[g], take[g]):
            lines += [f"## {s['image_id']} ({g})", "", "```", s["report"].strip(), "```", ""]
            listed = [(f, v) for f, v in s["labels"].items() if v != "absent"]
            lines += [f"- **{f}**: {v} — \"{s['evidence'].get(f, '')}\"" for f, v in listed] or ["- (all absent)"]
            lines.append("")
    path.write_text("\n".join(lines))


def print_summary(s):
    print(f"\nLabeled {s['reports_labeled']} reports: {s['groups_among_labeled']} "
          f"(refusals {s['refusals']}, unparseable {s['unparseable']})")
    print(f"Sample: {s['sample']}. Positives in the sample (present / uncertain):")
    for f, c in s["positives_in_sample"].items():
        print(f"  {f:28s} {c['present']:3d} / {c['uncertain']}")
    pts = s["trail_points"]
    print(f"Own-trail points per image: min {pts[0]}, median {pts[len(pts) // 2]}, max {pts[-1]}")
    print(f"Labeler cost ${s['labeler_cost_usd_all_requests']} (cached or not); hints_mvp spend so far "
          f"${s['spend_usd_total']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    build(load_config(ap.parse_args().config))
