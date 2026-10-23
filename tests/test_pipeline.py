"""End-to-end run with the mock backend on a few val images."""
import json
import shutil
from pathlib import Path

import pytest

from src import boxes as bx
from src import run, score

DATA = Path("~/data/chexlocalize").expanduser()
GT_CACHE = Path("outputs/cache/gt_boxes_val.json")


@pytest.mark.skipif(not DATA.exists(), reason="dataset not available")
def test_mock_pipeline_end_to_end(tmp_path, capsys):
    cfg = run.load_config("configs/mock_val.yaml")
    cfg |= {"n_images": 4, "cache_dir": str(tmp_path / "cache"), "output_dir": str(tmp_path / "out"),
            "bootstrap": {"n": 20, "seed": 0}}
    if GT_CACHE.exists():  # decoding every val mask takes ~90 s; reuse the cached boxes
        (tmp_path / "cache").mkdir()
        shutil.copy(GT_CACHE, tmp_path / "cache")

    results = run.run(cfg)
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
