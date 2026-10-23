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
        assert set(r["pass2"]) == set(run.PASS2_CONDITIONS)
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
