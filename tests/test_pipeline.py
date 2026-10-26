"""End-to-end run with the mock backend on a few val images."""
import json
import shutil
from pathlib import Path

import pytest

from src import boxes as bx
from src import run, score

DATA = Path("~/data/chexlocalize").expanduser()
GT_CACHE = Path("outputs/cache/gt_boxes_val.json")


def mock_config(tmp_path, name="run"):
    cfg = run.load_config("configs/mock_val.yaml")
    cfg |= {"n_images": 4, "cache_dir": str(tmp_path / "cache"), "output_dir": str(tmp_path / name),
            "bootstrap": {"n": 20, "seed": 0}}
    if GT_CACHE.exists() and not (tmp_path / "cache" / GT_CACHE.name).exists():
        # decoding every val mask takes ~90 s; reuse the cached boxes
        (tmp_path / "cache").mkdir(parents=True, exist_ok=True)
        shutil.copy(GT_CACHE, tmp_path / "cache")
    return cfg


@pytest.mark.skipif(not DATA.exists(), reason="dataset not available")
def test_mock_pipeline_end_to_end(tmp_path):
    cfg = mock_config(tmp_path, "out")
    results, _ = run.run(cfg)
    assert len(results) == 4
    for r in results.values():
        if r["pass1"]["report"] is None:
            continue
        assert set(r["boxes"]) == set(run.BOX_CONDITIONS)
        assert all(len(b) == bx.N_BOXES for b in r["boxes"].values())
        assert set(r["pass2"]) == set(cfg["conditions"]) - {"alone"}
        if r["helper"]["boxes"] is not None:
            n_real = sum(b["source"] == "helper" for b in r["boxes"]["helper_boxes"])
            assert n_real == r["helper"]["n_real"]

    summary = json.load(open(tmp_path / "out" / "summary.json"))
    assert summary["requests_new"] > 0
    calls = [json.loads(line) for line in open(tmp_path / "out" / "calls.jsonl")]
    assert all(c["settings"]["model"] == "mock" and "temperature" in c["settings"] for c in calls)

    # Rerun: everything comes from the cache.
    run.run(cfg)
    assert json.load(open(tmp_path / "out" / "summary.json"))["requests_new"] == 0

    score.score(cfg)
    assert (tmp_path / "out" / "results.md").exists()
    assert (tmp_path / "out" / "predictions.csv").exists()


class FlakyConnection:
    """Wraps MockBackend.call: requests number `start` to `start + n - 1` fail as if wifi dropped."""

    def __init__(self, call, start, n):
        self.call, self.start, self.n, self.count = call, start, n, 0

    def __call__(self, backend, *args):
        self.count += 1
        if self.start <= self.count < self.start + self.n:
            raise ConnectionError("network is unreachable")
        return self.call(backend, *args)


@pytest.mark.skipif(not DATA.exists(), reason="dataset not available")
def test_until_done_recovers_from_lost_connection(tmp_path, monkeypatch):
    from src.backends import MockBackend
    reference, _ = run.run(mock_config(tmp_path / "a"))  # uninterrupted run, separate cache

    flaky = FlakyConnection(MockBackend.call, start=5, n=12)
    monkeypatch.setattr(MockBackend, "call", lambda self, *a: flaky(self, *a))
    results, summary = run.run_until_done(mock_config(tmp_path / "b"), wait=0)
    assert flaky.count > 12  # the outage happened and was retried
    assert summary["reliability"]["retryable_errors"] == 0
    for i in reference:
        assert results[i]["pass1"]["report"] == reference[i]["pass1"]["report"]
        assert {c: v["report"] for c, v in results[i]["pass2"].items()} == \
               {c: v["report"] for c, v in reference[i]["pass2"].items()}


@pytest.mark.skipif(not DATA.exists(), reason="dataset not available")
def test_until_done_stops_on_request_errors(tmp_path, monkeypatch):
    from src.backends import MockBackend

    class BadRequest(Exception):
        status_code = 400

    def fail(self, *a):
        raise BadRequest("invalid request")

    monkeypatch.setattr(MockBackend, "call", fail)
    _, summary = run.run_until_done(mock_config(tmp_path), wait=0)  # must return, not loop
    assert summary["reliability"]["fatal_errors"] > 0


def test_is_retryable():
    class Status(Exception):
        def __init__(self, code):
            self.status_code = code

    assert run.is_retryable(ConnectionError()) and run.is_retryable(TimeoutError())
    assert run.is_retryable(Status(429)) and run.is_retryable(Status(529)) and run.is_retryable(Status(500))
    assert not run.is_retryable(Status(400)) and not run.is_retryable(Status(401))
    assert not run.is_retryable(KeyError("bug"))


@pytest.mark.skipif(not DATA.exists(), reason="dataset not available")
def test_each_call_record_has_request_thinking_and_parsed_output(tmp_path):
    cfg = mock_config(tmp_path)
    results, _ = run.run(cfg)
    calls = [json.loads(line) for line in open(tmp_path / "run" / "calls.jsonl")]
    oracle = [c for c in calls if c["condition"] == "oracle_boxes" and c["stop_reason"] != "refusal"]
    rec = json.load(open(oracle[0]["record"]))
    images = [p for p in rec["request"]["content"] if p["type"] == "image"]
    assert [p["variant"] for p in images] == ["clean", "boxed"]
    assert all(p["image_id"] == rec["image_id"] for p in images)
    assert len(images[1]["drawn_boxes"]) == bx.N_BOXES
    assert rec["request"]["system"] and "A colleague flagged" in rec["request"]["content"][-1]["text"]
    assert rec["thinking"] == ["mock thinking summary"] and rec["raw"]
    assert "parsed" in rec and "parse_error" in rec
    if rec["parsed"] is not None:
        assert rec["parsed"] == results[rec["image_id"]]["pass2"]["oracle_boxes"]["report"]
    for c in calls:  # every call has its record on disk, with the parse outcome
        r = json.load(open(c["record"]))
        assert "request" in r and "parsed" in r


@pytest.mark.skipif(not DATA.exists(), reason="dataset not available")
def test_budget_cap_holds_across_passes(tmp_path):
    cfg = mock_config(tmp_path)
    cfg["prices_per_mtok"] = {"mock": {"input": 1e5, "output": 1e5}}  # every call costs > max_usd
    _, s1 = run.run(cfg)
    assert s1["stopped_by_budget"] and 0 < s1["requests_new"] <= cfg["concurrency"]
    _, s2 = run.run(cfg)  # a restart must not reset the spend
    assert s2["stopped_by_budget"] and s2["requests_new"] == 0
    assert s2["cost_usd_run_total"] == s1["cost_usd_run_total"]


def v1_config(tmp_path):
    cfg = mock_config(tmp_path, "v1")
    cfg |= {"prompts": "v1", "pass2_mode": "fresh", "pad_boxes": False, "n_images": 6,
            "conditions": ["alone", "repeat", "helper_boxes", "oracle_boxes"]}
    return cfg


@pytest.mark.skipif(not DATA.exists(), reason="dataset not available")
def test_v1_fresh_pipeline(tmp_path):
    cfg = v1_config(tmp_path)
    results, _ = run.run(cfg)
    calls = [json.loads(line) for line in open(tmp_path / "v1" / "calls.jsonl")]
    first = {}  # (image, condition) -> first request record
    for c in calls:
        first.setdefault((c["image_id"], c["condition"]), json.load(open(c["record"])))
    for i, r in results.items():
        if r["pass1"]["report"] is None:
            continue
        # A repeat read is exactly the pass-1 request, sampled again.
        assert first[(i, "repeat")]["prompt_hash"] == first[(i, "pass1")]["prompt_hash"]
        # The helper's regions only, unpadded; the reader never sees its earlier report.
        n_real = r["helper"]["n_real"]
        assert len(r["boxes"]["helper_boxes"]) == n_real
        assert all(b["source"] == "helper" for b in r["boxes"]["helper_boxes"])
        rec = first[(i, "helper_boxes")]
        images = [p for p in rec["request"]["content"] if p["type"] == "image"]
        assert len(images) == (2 if n_real else 1)
        text = rec["request"]["content"][-1]["text"]
        assert "earlier report" not in text and "suspected" not in text
    score.score(cfg)


@pytest.mark.skipif(not DATA.exists(), reason="dataset not available")
def test_v1_failed_helper_drops_the_image(tmp_path, monkeypatch):
    from src.backends import MockBackend, Response
    real_call = MockBackend.call

    def refuse_helper(self, system, parts, role):
        if role == "helper":
            return Response("", "refusal", 10, 0, {})
        return real_call(self, system, parts, role)

    monkeypatch.setattr(MockBackend, "call", refuse_helper)
    results, _ = run.run(v1_config(tmp_path))
    assert all("helper_boxes" not in r["pass2"] and "helper_boxes" not in r["boxes"] for r in results.values())


def test_config_rejects_incompatible_modes(tmp_path):
    import yaml
    base = yaml.safe_load(open("configs/v1.yaml"))
    for change in [{"conditions": ["alone", "look_again"]}, {"prompts": "v0"}, {"pass2_mode": "revise"}]:
        path = tmp_path / "c.yaml"
        yaml.safe_dump(base | change, open(path, "w"))
        with pytest.raises(ValueError):
            run.load_config(path)
