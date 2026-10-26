"""Run the second-look experiment: pass 1, helper, then pass 2 for every condition.

Usage: python -m src.run --config configs/mock_val.yaml [--until-done]
Reruns are free: every response is cached on disk, so an interrupted run resumes where it stopped.
--until-done keeps rerunning (waiting longer each time) until no request is left failing on a
network or server error, which is how a run survives lost wifi or a sleeping laptop.
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
from .backends import image_digest, make_backend, request_hash

BOX_CONDITIONS = ["random_boxes", "checklist_boxes", "helper_boxes", "oracle_boxes"]
# look_again: pass 2 without boxes, revising the earlier report; repeat: a second blind read (fresh mode).
PASS2_CONDITIONS = ["look_again", "repeat"] + BOX_CONDITIONS
CONDITIONS = ["alone"] + PASS2_CONDITIONS  # alone = the pass-1 report, no revision


class BudgetExceeded(RuntimeError):
    pass


def is_retryable(e):
    """Network problems, timeouts, rate limits and server errors are worth retrying later; request errors are not."""
    status = getattr(e, "status_code", None)
    if status is not None:
        return status in (408, 409, 429) or status >= 500
    name = type(e).__name__
    return isinstance(e, (ConnectionError, TimeoutError)) or "Connection" in name or "Timeout" in name


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


def write_json(path, obj):
    """Write atomically, so a killed process never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    json.dump(obj, open(tmp, "w"))
    tmp.rename(path)


def describe_request(system, parts, image_meta):
    """What was sent, as JSON: text verbatim; each image by id, variant (clean/boxed), drawn boxes and pixel hash."""
    meta = iter(image_meta)
    content = [
        {"type": "text", "text": p} if isinstance(p, str)
        else {"type": "image", **next(meta), "size": list(p.size), "mode": p.mode, "pixel_sha256": image_digest(p)}
        for p in parts
    ]
    return {"system": system, "content": content}


class Caller:
    """Model calls for one run, with an on-disk cache, rate limit, budget cap and token/cost log.

    Each request is one JSON file in the cache holding the request (prompt text, image ids), the full raw
    response (including thinking blocks), the extracted thinking text and the parsed output.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.backends = {role: make_backend(cfg[role]) for role in ("reader", "helper")}
        self.cache_dir = Path(cfg["cache_dir"])
        self.limiter = RateLimiter(cfg["requests_per_minute"])
        self.lock = threading.Lock()
        self.log = []  # one entry per request, cached or not
        self.new_cost = 0.0  # USD for uncached requests in this invocation
        # Total new spend for this run across passes and restarts; max_usd applies to this.
        self.ledger_path = Path(cfg["output_dir"]) / "spend.json"
        self.ledger = json.load(open(self.ledger_path)) if self.ledger_path.exists() else {"usd": 0.0, "requests": 0}

    def cost(self, rec):
        price = self.cfg["prices_per_mtok"][rec["settings"]["model"]]
        return (rec["input_tokens"] * price["input"] + rec["output_tokens"] * price["output"]) / 1e6

    def request(self, role, condition, image_id, system, parts, image_meta):
        """Returns (record, path). Cached records missing a request description get one added."""
        backend = self.backends[role]
        settings = backend.settings()
        key = request_hash(system, parts, settings)
        path = self.cache_dir / settings["model"] / role / condition / f"{image_id}__{key[:16]}.json"
        cached = path.exists()
        if cached:
            rec = json.load(open(path))
            if "request" not in rec:
                rec["request"] = describe_request(system, parts, image_meta)
                write_json(path, rec)
        else:
            with self.lock:
                if self.ledger["usd"] >= self.cfg["max_usd"]:
                    raise BudgetExceeded(f"run has spent ${self.ledger['usd']:.2f}, max_usd is {self.cfg['max_usd']}")
            self.limiter.wait()
            t0 = time.time()
            r = backend.call(system, parts, role)
            rec = {
                "role": role, "condition": condition, "image_id": image_id, "prompt_hash": key,
                "settings": settings, "request": describe_request(system, parts, image_meta),
                "text": r.text, "thinking": r.thinking, "stop_reason": r.stop_reason,
                "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                "seconds": round(time.time() - t0, 2), "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "raw": r.raw,
            }
            write_json(path, rec)
        cost = self.cost(rec)
        entry = {k: rec[k] for k in ("role", "condition", "image_id", "prompt_hash", "settings", "stop_reason",
                                     "input_tokens", "output_tokens")}
        with self.lock:
            if not cached:
                self.new_cost += cost
                self.ledger = {"usd": self.ledger["usd"] + cost, "requests": self.ledger["requests"] + 1}
                write_json(self.ledger_path, self.ledger)
            self.log.append(entry | {"cached": cached, "cost_usd": round(cost, 6), "record": str(path)})
            n_new = sum(not e["cached"] for e in self.log)
            if not cached and n_new % 25 == 0:
                print(f"  [{time.strftime('%H:%M:%S')}] {n_new} new requests this pass, run total "
                      f"${self.ledger['usd']:.2f}", flush=True)
        return rec, path

    def ask(self, role, condition, image_id, system, parts, parse, image_meta):
        """Request and validate. On invalid output, retry once with a 'return only valid JSON' nudge.

        The parsed output (or parse error) is saved into each request's record. Returns (parsed or None,
        info). Refusals and API errors are not retried here: the SDK already retries briefly, failed
        requests are not cached, and --until-done reruns them later.
        """
        info = {"attempts": 0, "parse_errors": [], "refusal": False, "error": None, "retryable": False}
        for attempt in range(2):
            p = parts if attempt == 0 else parts[:-1] + [parts[-1] + prompts.RETRY_NUDGE]
            info["attempts"] += 1
            try:
                rec, path = self.request(role, condition, image_id, system, p, image_meta)
            except BudgetExceeded:
                raise
            except Exception as e:
                info["error"] = f"{type(e).__name__}: {e}"
                info["retryable"] = is_retryable(e)
                return None, info
            parsed, error = None, None
            if rec["stop_reason"] == "refusal":
                info["refusal"] = True
            else:
                try:
                    parsed = parse(rec["text"])
                except prompts.ParseError as e:
                    note = " (stopped at max_tokens)" if rec["stop_reason"] in ("max_tokens", "length") else ""
                    error = str(e) + note
                    info["parse_errors"].append(error)
            if "parsed" not in rec or rec.get("parse_error") != error:
                write_json(path, rec | {"parsed": parsed, "parse_error": error})
            if parsed is not None or info["refusal"]:
                return parsed, info
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
    # v0: flagged wording, padded boxes, reader revises its report. v1: neutral wording, real boxes only,
    # and in "fresh" mode the reader reads again without seeing its earlier report.
    style, mode, pad = cfg.get("prompts", "v0"), cfg.get("pass2_mode", "revise"), cfg.get("pad_boxes", True)
    caller = Caller(cfg)
    results = {i: {"image_id": i, "split": split, "labels": labels[i], "boxes": {}, "pass2": {}} for i in ids}

    def clean(i):
        return {"image_id": i, "variant": "clean"}

    def pass1(i):
        report, info = caller.ask("reader", "pass1", i, prompts.READER_SYSTEM, [images[i], prompts.pass1_text()],
                                  prompts.parse_reader, [clean(i)])
        results[i]["pass1"] = {"report": report, **info}

    def helper(i):
        im = images[i]
        text = prompts.helper_text(results[i]["pass1"]["report"], fmt, im.width, im.height, style)
        parse = prompts.parse_helper if style == "v0" else prompts.parse_helper_boxes
        out, info = caller.ask("helper", "helper", i, prompts.helper_system(style), [im, text],
                               lambda t: parse(t, fmt, im.width, im.height), [clean(i)])
        results[i]["helper"] = {"boxes": out, "n_real": len(out or []), **info}

    def make_box_sets(i):
        r = results[i]
        present = [f for f in data.FINDINGS if r["labels"][f]]
        missed = {f for f in present if r["pass1"]["report"]["findings"][f]["present"] == "no"}
        makers = {
            "random_boxes": lambda rng: bx.random_boxes(rng, sizes),
            "checklist_boxes": lambda rng: bx.checklist_boxes(rng),
            "helper_boxes": lambda rng: bx.helper_boxes(r.get("helper", {}).get("boxes"), rng, sizes, pad),
            "oracle_boxes": lambda rng: bx.oracle_boxes(gt.get(i, {}), present, missed, rng, sizes, pad),
        }
        for c in BOX_CONDITIONS:
            if c not in conditions:
                continue
            if c == "helper_boxes" and not pad and r.get("helper", {}).get("boxes") is None:
                continue  # the helper call failed: no helper read for this image (dropped when scoring)
            r["boxes"][c] = makers[c](bx.rng_for(seed, i, c))
            assert len(r["boxes"][c]) == bx.N_BOXES if pad else len(r["boxes"][c]) <= bx.N_BOXES

    def pass2(job):
        i, cond = job
        r = results[i]
        report = r["pass1"]["report"] if mode == "revise" else None
        if not r["boxes"].get(cond):  # look_again, repeat, or a v1 helper that marked no regions
            parts = [images[i], prompts.pass2_text(report, False, style)]
            meta = [clean(i)]
        else:
            boxed = bx.draw_boxes(images[i], [b["box"] for b in r["boxes"][cond]])
            parts = [images[i], boxed, prompts.pass2_text(report, True, style)]
            # Box sources (e.g. "oracle:Edema") are logged for analysis; only the numbered outlines are drawn.
            meta = [clean(i), {"image_id": i, "variant": "boxed", "drawn_boxes": r["boxes"][cond]}]
        revised, info = caller.ask("reader", cond, i, prompts.READER_SYSTEM, parts, prompts.parse_reader, meta)
        r["pass2"][cond] = {"report": revised, **info}

    t0 = time.time()
    stopped_by_budget = False
    try:
        with ThreadPoolExecutor(cfg["concurrency"]) as pool:
            list(pool.map(pass1, ids))
            ok = [i for i in ids if results[i]["pass1"]["report"]]
            print(f"  pass 1 done: {len(ok)}/{len(ids)} images ok", flush=True)
            if "helper_boxes" in conditions:
                list(pool.map(helper, ok))
                print("  helper done", flush=True)
            for i in ok:
                make_box_sets(i)
            list(pool.map(pass2, [(i, c) for i in ok for c in conditions
                                  if c in PASS2_CONDITIONS and (c not in BOX_CONDITIONS or c in results[i]["boxes"])]))
    except BudgetExceeded as e:
        stopped_by_budget = True
        print(f"STOPPED: budget cap reached ({e}). Cached responses are kept; raise max_usd and rerun to resume.")
    finally:
        with open(out_dir / "predictions.jsonl", "w") as f:
            for i in ids:
                f.write(json.dumps(results[i]) + "\n")
        with open(out_dir / "calls.jsonl", "w") as f:
            for entry in caller.log:
                f.write(json.dumps(entry) + "\n")
        summary = summarize(cfg, ids, results, caller, time.time() - t0) | {"stopped_by_budget": stopped_by_budget}
        json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
        print_summary(summary)
    return results, summary


def run_until_done(cfg, wait=60, max_wait=900):
    """Rerun until no request is left failing on a retryable error. Each pass reuses the cache."""
    while True:
        results, summary = run(cfg)
        rel = summary["reliability"]
        if summary["stopped_by_budget"]:
            return results, summary
        if rel["fatal_errors"]:
            print(f"STOPPED: {rel['fatal_errors']} requests failed with non-retryable errors, e.g. "
                  f"{rel['api_errors_sample'][:2]}. Fix the cause and rerun.")
            return results, summary
        if not rel["retryable_errors"]:
            print("DONE: every request has a response.")
            return results, summary
        print(f"[{time.strftime('%H:%M:%S')}] {rel['retryable_errors']} requests failed on network/server errors; "
              f"retrying in {wait}s", flush=True)
        time.sleep(wait)
        wait = min(wait * 2, max_wait)


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
        "name": cfg["name"], "split": cfg["split"], "n_images": len(ids), "max_usd": cfg["max_usd"],
        "reader": caller.backends["reader"].settings(), "helper": caller.backends["helper"].settings(),
        "helper_box_format": cfg["helper"].get("box_format", "norm"),
        "prompts": cfg.get("prompts", "v0"), "pass2_mode": cfg.get("pass2_mode", "revise"),
        "pad_boxes": cfg.get("pad_boxes", True),
        "reliability": {
            "reader": rates([t for role, _, t in tasks if role == "reader"]),
            "helper": rates([t for role, _, t in tasks if role == "helper"]),
            "retryable_errors": sum(t["error"] is not None and t["retryable"] for _, _, t in tasks),
            "fatal_errors": sum(t["error"] is not None and not t["retryable"] for _, _, t in tasks),
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
        "cost_usd_run_total": round(caller.ledger["usd"], 4),  # all new spend for this run, every pass
        "seconds": round(seconds, 1),
    }


def print_summary(s):
    print(f"\n{s['name']}: {s['n_images']} images, {s['requests']} requests ({s['requests_new']} new), "
          f"{s['seconds']}s")
    for role in ("reader", "helper"):
        print(f"  {role}: {s['reliability'][role]}")
    print(f"  helper real boxes: {s['helper_real_boxes']}")
    print(f"  tokens: {s['tokens']}")
    print(f"  cost: ${s['cost_usd_new_this_run']} new this pass; run total ${s['cost_usd_run_total']} "
          f"(cap ${s['max_usd']})")


def load_config(path):
    cfg = yaml.safe_load(open(path))
    unknown = set(cfg["conditions"]) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"unknown conditions {unknown}")
    style, mode = cfg.get("prompts", "v0"), cfg.get("pass2_mode", "revise")
    if style not in ("v0", "v1") or mode not in ("revise", "fresh"):
        raise ValueError("prompts must be v0 or v1; pass2_mode must be revise or fresh")
    if style == "v0" and mode == "fresh":
        raise ValueError("v0 prompts always revise the earlier report")
    if mode == "fresh" and "look_again" in cfg["conditions"]:
        raise ValueError("look_again revises the earlier report; in fresh mode use repeat")
    if mode == "revise" and "repeat" in cfg["conditions"]:
        raise ValueError("repeat is a fresh second read; use pass2_mode: fresh")
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--until-done", action="store_true", help="rerun until no request fails on network/server errors")
    args = ap.parse_args()
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    cfg = load_config(args.config)
    run_until_done(cfg) if args.until_done else run(cfg)
