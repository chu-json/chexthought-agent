"""Run the second-look experiment: pass 1, helper, then pass 2 for every condition.

Usage: python -m src.run --config configs/mock_val.yaml
Reruns are free: every response is cached on disk, so an interrupted run resumes where it stopped.
"""
import argparse
import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from dotenv import load_dotenv

from . import boxes as bx
from . import data, prompts
from .backends import make_backend, request_hash

BOX_CONDITIONS = ["random_boxes", "checklist_boxes", "helper_boxes", "oracle_boxes"]
PASS2_CONDITIONS = ["look_again"] + BOX_CONDITIONS
CONDITIONS = ["alone"] + PASS2_CONDITIONS  # alone = the pass-1 report, no revision


class BudgetExceeded(RuntimeError):
    pass


class RateLimiter:
    """Spaces request starts at least 60/rpm seconds apart, across threads."""

    def __init__(self, rpm):
        self.interval = 60.0 / rpm
        self.next_time = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_time)
            self.next_time = start + self.interval
        time.sleep(start - now)


class Caller:
    """Model calls for one run, with an on-disk cache, rate limit, budget cap and token/cost log."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.backends = {role: make_backend(cfg[role]) for role in ("reader", "helper")}
        self.cache_dir = Path(cfg["cache_dir"])
        self.limiter = RateLimiter(cfg["requests_per_minute"])
        self.lock = threading.Lock()
        self.log = []  # one entry per request, cached or not
        self.new_cost = 0.0  # USD for uncached requests in this invocation

    def cost(self, rec):
        price = self.cfg["prices_per_mtok"][rec["settings"]["model"]]
        return (rec["input_tokens"] * price["input"] + rec["output_tokens"] * price["output"]) / 1e6

    def request(self, role, condition, image_id, system, parts):
        backend = self.backends[role]
        settings = backend.settings()
        key = request_hash(system, parts, settings)
        path = self.cache_dir / settings["model"] / role / condition / f"{image_id}__{key[:16]}.json"
        cached = path.exists()
        if cached:
            rec = json.load(open(path))
        else:
            with self.lock:
                if self.new_cost >= self.cfg["max_usd"]:
                    raise BudgetExceeded(f"spent ${self.new_cost:.2f}, max_usd is {self.cfg['max_usd']}")
            self.limiter.wait()
            t0 = time.time()
            r = backend.call(system, parts, role)
            rec = {
                "role": role, "condition": condition, "image_id": image_id, "prompt_hash": key,
                "settings": settings, "text": r.text, "stop_reason": r.stop_reason,
                "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                "seconds": round(time.time() - t0, 2), "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "raw": r.raw,
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            json.dump(rec, open(tmp, "w"))
            tmp.rename(path)
        cost = self.cost(rec)
        entry = {k: rec[k] for k in ("role", "condition", "image_id", "prompt_hash", "settings", "stop_reason",
                                     "input_tokens", "output_tokens")}
        with self.lock:
            if not cached:
                self.new_cost += cost
            self.log.append(entry | {"cached": cached, "cost_usd": round(cost, 6)})
        return rec

    def ask(self, role, condition, image_id, system, parts, parse):
        """Request and validate. On invalid output, retry once with a 'return only valid JSON' nudge.

        Returns (parsed or None, info). Refusals and API errors are not retried here
        (the SDK already retries rate limits and server errors).
        """
        info = {"attempts": 0, "parse_errors": [], "refusal": False, "error": None}
        for attempt in range(2):
            p = parts if attempt == 0 else parts[:-1] + [parts[-1] + prompts.RETRY_NUDGE]
            info["attempts"] += 1
            try:
                rec = self.request(role, condition, image_id, system, p)
            except BudgetExceeded:
                raise
            except Exception as e:
                info["error"] = f"{type(e).__name__}: {e}"
                return None, info
            if rec["stop_reason"] == "refusal":
                info["refusal"] = True
                return None, info
            try:
                return parse(rec["text"]), info
            except prompts.ParseError as e:
                note = " (stopped at max_tokens)" if rec["stop_reason"] in ("max_tokens", "length") else ""
                info["parse_errors"].append(str(e) + note)
        return None, info


def run(cfg):
    root, split, seed = cfg["data_root"], cfg["split"], cfg["box_seed"]
    conditions = cfg["conditions"]
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = data.load_labels(root, split)
    ids = data.sample_images(labels, cfg["n_images"], cfg["sample_seed"], cfg["min_positives_per_finding"],
                             cfg.get("exclude_ids", []))
    print(f"{len(ids)} {split} images. Positives per finding:")
    for f in data.FINDINGS:
        print(f"  {f:28s} {sum(labels[i][f] for i in ids)}")

    gt = data.gt_boxes(root, split, cfg["cache_dir"])
    sizes = [(b[2] - b[0], b[3] - b[1]) for by_finding in gt.values() for b in by_finding.values()]
    images = {i: data.load_image(root, split, i, cfg["image_long_edge"]) for i in ids}
    fmt = cfg["helper"].get("box_format", "norm")
    caller = Caller(cfg)
    results = {i: {"image_id": i, "split": split, "labels": labels[i], "boxes": {}, "pass2": {}} for i in ids}

    def pass1(i):
        report, info = caller.ask("reader", "pass1", i, prompts.READER_SYSTEM, [images[i], prompts.pass1_text()],
                                  prompts.parse_reader)
        results[i]["pass1"] = {"report": report, **info}

    def helper(i):
        im = images[i]
        text = prompts.helper_text(results[i]["pass1"]["report"], fmt, im.width, im.height)
        out, info = caller.ask("helper", "helper", i, prompts.HELPER_SYSTEM, [im, text],
                               lambda t: prompts.parse_helper(t, fmt, im.width, im.height))
        results[i]["helper"] = {"boxes": out, "n_real": len(out or []), **info}

    def make_box_sets(i):
        r = results[i]
        present = [f for f in data.FINDINGS if r["labels"][f]]
        missed = {f for f in present if r["pass1"]["report"]["findings"][f]["present"] == "no"}
        makers = {
            "random_boxes": lambda rng: bx.random_boxes(rng, sizes),
            "checklist_boxes": lambda rng: bx.checklist_boxes(rng),
            "helper_boxes": lambda rng: bx.helper_boxes(r.get("helper", {}).get("boxes"), rng, sizes),
            "oracle_boxes": lambda rng: bx.oracle_boxes(gt.get(i, {}), present, missed, rng, sizes),
        }
        for c in BOX_CONDITIONS:
            if c in conditions:
                r["boxes"][c] = makers[c](bx.rng_for(seed, i, c))
                assert len(r["boxes"][c]) == bx.N_BOXES

    def pass2(job):
        i, cond = job
        r = results[i]
        report = r["pass1"]["report"]
        if cond == "look_again":
            parts = [images[i], prompts.pass2_text(report, with_boxes=False)]
        else:
            boxed = bx.draw_boxes(images[i], [b["box"] for b in r["boxes"][cond]])
            parts = [images[i], boxed, prompts.pass2_text(report, with_boxes=True)]
        revised, info = caller.ask("reader", cond, i, prompts.READER_SYSTEM, parts, prompts.parse_reader)
        r["pass2"][cond] = {"report": revised, **info}

    t0 = time.time()
    try:
        with ThreadPoolExecutor(cfg["concurrency"]) as pool:
            list(pool.map(pass1, ids))
            ok = [i for i in ids if results[i]["pass1"]["report"]]
            if "helper_boxes" in conditions:
                list(pool.map(helper, ok))
            for i in ok:
                make_box_sets(i)
            list(pool.map(pass2, [(i, c) for i in ok for c in conditions if c in PASS2_CONDITIONS]))
    except BudgetExceeded as e:
        print(f"STOPPED: budget cap reached ({e}). Cached responses are kept; raise max_usd and rerun to resume.")
    finally:
        with open(out_dir / "predictions.jsonl", "w") as f:
            for i in ids:
                f.write(json.dumps(results[i]) + "\n")
        with open(out_dir / "calls.jsonl", "w") as f:
            for entry in caller.log:
                f.write(json.dumps(entry) + "\n")
        summary = summarize(cfg, ids, results, caller, time.time() - t0)
        json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
        print_summary(summary)
    return results


def summarize(cfg, ids, results, caller, seconds):
    tasks = []  # (role, condition, info) per validated output we asked for
    for i in ids:
        r = results[i]
        if "pass1" in r:
            tasks.append(("reader", "pass1", r["pass1"]))
        if "helper" in r:
            tasks.append(("helper", "helper", r["helper"]))
        tasks += [("reader", c, v) for c, v in r["pass2"].items()]

    def rates(ts):
        n = len(ts)
        failed = sum(t.get("report", t.get("boxes")) is None for t in ts)  # reader: report, helper: boxes
        return {
            "tasks": n,
            "first_try_parse_failures": sum(bool(t["parse_errors"]) for t in ts),
            "final_parse_failures": sum(len(t["parse_errors"]) == 2 for t in ts),
            "refusals": sum(t["refusal"] for t in ts),
            "api_errors": sum(t["error"] is not None for t in ts),
            "final_failure_rate": round(failed / n, 4) if n else None,
        }

    log = caller.log
    return {
        "name": cfg["name"], "split": cfg["split"], "n_images": len(ids),
        "reader": caller.backends["reader"].settings(), "helper": caller.backends["helper"].settings(),
        "helper_box_format": cfg["helper"].get("box_format", "norm"),
        "reliability": {
            "reader": rates([t for role, _, t in tasks if role == "reader"]),
            "helper": rates([t for role, _, t in tasks if role == "helper"]),
            "parse_errors_sample": [e for _, _, t in tasks for e in t["parse_errors"]][:10],
            "api_errors_sample": [t["error"] for _, _, t in tasks if t["error"]][:5],
        },
        "helper_real_boxes": dict(Counter(r["helper"]["n_real"] for r in results.values() if "helper" in r)),
        "requests": len(log), "requests_new": sum(not e["cached"] for e in log),
        "tokens": {
            "input": sum(e["input_tokens"] for e in log), "output": sum(e["output_tokens"] for e in log),
            "input_new": sum(e["input_tokens"] for e in log if not e["cached"]),
            "output_new": sum(e["output_tokens"] for e in log if not e["cached"]),
        },
        "cost_usd_all_requests": round(sum(e["cost_usd"] for e in log), 4),
        "cost_usd_new_this_run": round(caller.new_cost, 4),
        "seconds": round(seconds, 1),
    }


def print_summary(s):
    print(f"\n{s['name']}: {s['n_images']} images, {s['requests']} requests ({s['requests_new']} new), "
          f"{s['seconds']}s")
    for role in ("reader", "helper"):
        print(f"  {role}: {s['reliability'][role]}")
    print(f"  helper real boxes: {s['helper_real_boxes']}")
    print(f"  tokens: {s['tokens']}")
    print(f"  cost: ${s['cost_usd_all_requests']} for all requests, ${s['cost_usd_new_this_run']} new this run")


def load_config(path):
    cfg = yaml.safe_load(open(path))
    unknown = set(cfg["conditions"]) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"unknown conditions {unknown}")
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    run(load_config(args.config))
