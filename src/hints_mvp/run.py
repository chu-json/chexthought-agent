"""Run every hints_mvp condition on every sampled image: one read per condition, no revision.

Usage: python -m src.hints_mvp.run --config configs/hints_mvp.yaml [--until-done] [--show-prompts] [--limit N]
Conditions (all with the same system prompt; the text after the image differs):
- no_hint, no_hint_rerun: the image alone. The rerun is an identical, independent request (the noise floor).
- own_trail: this image's click trail as text. other_trail: another sampled image's trail (data.derangement).
- random_trail: as many uniform random points as the own trail.
- leak_check: the own trail with no image.
Responses are cached per condition, so the rerun is a separate request and a restart resumes for free.
--show-prompts writes prompt_examples.md (one image, every condition, exactly as sent) without calling the model.
"""
import argparse
import json
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

from ..run import BudgetExceeded, Caller
from . import data, prompts
from .config import load_config

TRAIL_OF = {"own_trail": "own", "other_trail": "other", "random_trail": "random", "leak_check": "own"}


def request_for(s, cond, image):
    """(parts, image_meta) sent for one image and condition."""
    points = s["trails"][TRAIL_OF[cond]]["points"] if cond in TRAIL_OF else None
    if cond == "leak_check":
        return [prompts.reader_text(points, with_image=False)], []
    return [image, prompts.reader_text(points)], [{"image_id": s["image_id"], "variant": "clean"}]


def load_sample(cfg):
    sample = json.load(open(cfg.get("sample_file") or Path(cfg["output_dir"]) / "sample.json"))
    if cfg.get("limit"):  # a fixed random subset, e.g. for the dry run
        sample = sorted(random.Random(f"{cfg['seed']}:limit").sample(sample, cfg["limit"]), key=lambda s: s["image_id"])
    return sample


def run(cfg):
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = load_sample(cfg)
    by_id = {s["image_id"]: s for s in sample}
    images = {i: data.load_image(cfg["image_root"], i, cfg["image_long_edge"]) for i in by_id}
    conds = cfg["conditions"]
    print(f"{len(sample)} images ({Counter(s['group'] for s in sample)}), {len(conds)} conditions")
    caller = Caller(cfg, roles=("reader",))
    results = {i: {"image_id": i, "group": s["group"], "labels": s["labels"], "other_trail_from":
                   s["trails"]["other"]["from_image"], "reports": {}} for i, s in by_id.items()}

    def read(job):
        i, cond = job
        parts, meta = request_for(by_id[i], cond, images[i])
        report, info = caller.ask("reader", cond, i, prompts.READER_SYSTEM, parts, prompts.parse_reader, meta)
        results[i]["reports"][cond] = {"report": report, **info}

    t0 = time.time()
    stopped_by_budget = False
    try:
        with ThreadPoolExecutor(cfg["concurrency"]) as pool:
            list(pool.map(read, [(i, c) for i in by_id for c in conds]))
    except BudgetExceeded as e:
        stopped_by_budget = True
        print(f"STOPPED: budget cap reached ({e}). Cached responses are kept; raise max_usd and rerun to resume.")
    finally:
        with open(out_dir / "predictions.jsonl", "w") as f:
            for r in results.values():
                f.write(json.dumps(r) + "\n")
        with open(out_dir / "calls.jsonl", "w") as f:
            for entry in caller.log:
                f.write(json.dumps(entry) + "\n")
        summary = summarize(cfg, results, caller, time.time() - t0) | {"stopped_by_budget": stopped_by_budget}
        json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
        print_summary(summary)
    return results, summary


def summarize(cfg, results, caller, seconds):
    def rel(infos):
        return {
            "tasks": len(infos),
            "first_try_parse_failures": sum(bool(t["parse_errors"]) for t in infos),
            "final_failures": sum(t["report"] is None for t in infos),
            "refusals": sum(t["refusal"] for t in infos),
            "api_errors": sum(t["error"] is not None for t in infos),
        }

    infos = {c: [r["reports"][c] for r in results.values() if c in r["reports"]] for c in cfg["conditions"]}
    every = [t for ts in infos.values() for t in ts]
    log = caller.log
    per_cond = {c: {"requests": sum(e["condition"] == c for e in log),
                    "input_tokens": sum(e["input_tokens"] for e in log if e["condition"] == c),
                    "output_tokens": sum(e["output_tokens"] for e in log if e["condition"] == c),
                    "cost_usd": round(sum(e["cost_usd"] for e in log if e["condition"] == c), 4)}
                for c in cfg["conditions"]}
    return {
        "name": cfg["name"], "n_images": len(results), "reader": caller.backends["reader"].settings(),
        "max_usd": cfg["max_usd"],
        "reliability": {c: rel(ts) for c, ts in infos.items()},
        "retryable_errors": sum(t["error"] is not None and t["retryable"] for t in every),
        "fatal_errors": sum(t["error"] is not None and not t["retryable"] for t in every),
        "parse_errors_sample": [e for t in every for e in t["parse_errors"]][:10],
        "api_errors_sample": [t["error"] for t in every if t["error"]][:5],
        "per_condition": per_cond,
        "requests": len(log), "requests_new": sum(not e["cached"] for e in log),
        "cost_usd_all_requests": round(sum(e["cost_usd"] for e in log), 4),
        "cost_usd_new_this_run": round(caller.new_cost, 4),
        "spend_usd_total": round(caller.ledger["usd"], 4),  # everything hints_mvp has spent (spend_file)
        "seconds": round(seconds, 1),
    }


def print_summary(s):
    print(f"\n{s['name']}: {s['n_images']} images, {s['requests']} requests ({s['requests_new']} new), {s['seconds']}s")
    for c, r in s["reliability"].items():
        p = s["per_condition"][c]
        print(f"  {c:14s} {r}  tokens in/out {p['input_tokens']}/{p['output_tokens']}  ${p['cost_usd']}")
    if s["parse_errors_sample"]:
        print(f"  parse errors, e.g. {s['parse_errors_sample'][:3]}")
    print(f"  cost of all requests ${s['cost_usd_all_requests']}; new this run ${s['cost_usd_new_this_run']}; "
          f"hints_mvp total spend ${s['spend_usd_total']} (cap ${s['max_usd']})")


def run_until_done(cfg, wait=60, max_wait=900):
    """Rerun until no request is left failing on a network or server error. Each pass reuses the cache."""
    while True:
        results, summary = run(cfg)
        if summary["stopped_by_budget"]:
            return results, summary
        if summary["fatal_errors"]:
            print(f"STOPPED: {summary['fatal_errors']} requests failed with non-retryable errors, e.g. "
                  f"{summary['api_errors_sample'][:2]}. Fix the cause and rerun.")
            return results, summary
        if not summary["retryable_errors"]:
            print("DONE: every request has a response.")
            return results, summary
        print(f"[{time.strftime('%H:%M:%S')}] {summary['retryable_errors']} requests failed on network/server "
              f"errors; retrying in {wait}s", flush=True)
        time.sleep(wait)
        wait = min(wait * 2, max_wait)


def show_prompts(cfg):
    """One image (the one whose trail length is closest to the median), every condition, exactly as sent."""
    sample = load_sample(cfg)
    n = sorted(len(s["trails"]["own"]["points"]) for s in sample)[len(sample) // 2]
    s = min(sample, key=lambda s: (abs(len(s["trails"]["own"]["points"]) - n), s["image_id"]))
    image = data.load_image(cfg["image_root"], s["image_id"], cfg["image_long_edge"])
    lines = ["# hints_mvp prompt examples", "", f"Image {s['image_id']} ({s['group']}); its own trail has "
             f"{len(s['trails']['own']['points'])} points; other_trail comes from {s['trails']['other']['from_image']} "
             f"({len(s['trails']['other']['points'])} points).", "", "## System prompt (every condition)", "",
             "```", prompts.READER_SYSTEM, "```", ""]
    for cond in cfg["conditions"]:
        parts, _ = request_for(s, cond, image)
        lines += [f"## {cond}", ""]
        for p in parts:
            lines += ["```", p if isinstance(p, str) else f"[image: {s['image_id']}, {p.width}x{p.height} grayscale PNG]",
                      "```"]
        lines.append("")
    path = Path(cfg["output_dir"]) / "prompt_examples.md"
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--until-done", action="store_true", help="rerun until no request fails on network/server errors")
    ap.add_argument("--show-prompts", action="store_true", help="write prompt_examples.md; no model calls")
    ap.add_argument("--limit", type=int, help="run on a fixed random subset of this many images")
    args = ap.parse_args()
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    cfg = load_config(args.config)
    if args.limit:
        cfg["limit"] = args.limit
    if args.show_prompts:
        show_prompts(cfg)
    else:
        run_until_done(cfg) if args.until_done else run(cfg)
