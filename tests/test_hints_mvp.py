import json

import pandas as pd
import pytest
import yaml

from src.hints_mvp import data, prompts, sample
from src.hints_mvp.config import load_config
from src.prompts import ParseError


def test_snapped():
    assert data.is_snapped([(0.45, 0.14), (0.4601, 0.2), (0.3, 0.7)])
    assert not data.is_snapped([(0.4507, 0.1404), (0.4601, 0.1653), (0.4758, 0.1901)])
    # 3 of 4 values on the grid is 75%, under the 80% threshold
    assert not data.is_snapped([(0.45, 0.14), (0.30, 0.1653)])


def test_derangement_is_one_cycle():
    ids = [f"patient{k:05d}_study1_view1_frontal" for k in range(100)]
    d = data.derangement(ids, 0)
    assert sorted(d.values()) == sorted(ids)
    assert all(d[i] != i for i in ids)
    assert d == data.derangement(ids, 0)


def test_trail_text():
    t = prompts.trail_text([(0.4507, 0.1404), (0.46, 0.1653)])
    assert t == ("A radiologist's search path for this image (x, y as fractions of width and height, in viewing "
                 "order): (0.451, 0.140), (0.460, 0.165). It shows where they looked and may not indicate any finding.")
    assert prompts.reader_text(None) == prompts.READ_TEXT
    assert prompts.reader_text([(0.1, 0.2)], with_image=False).endswith(prompts.NO_IMAGE_TEXT)


def test_rel_path():
    assert data.rel_path("patient00002_study1_view1_frontal") == "train/patient00002/study1/view1_frontal.png"


def reader_json(**over):
    findings = {f: {"present": "no", "probability": 5, "location": ""} for f in data.CHEXPERT}
    findings.update(over)
    return {"findings": findings, "reasoning": "r"}


def test_parse_reader_tolerant():
    obj = reader_json(Edema={"present": "Yes", "probability": "80%"})
    obj["findings"]["pleural effusion"] = obj["findings"].pop("Pleural Effusion")
    obj["findings"]["Cardiomegaly"] = "yes"
    text = "```json\n" + json.dumps(obj)[:-1] + ",}\n```"  # code fence and a trailing comma
    out = prompts.parse_reader(text)["findings"]
    assert out["Edema"] == {"present": "yes", "probability": 80.0, "location": ""}
    assert out["Pleural Effusion"]["present"] == "no"
    assert out["Cardiomegaly"]["present"] == "yes" and out["Cardiomegaly"]["probability"] is None


def test_parse_reader_skips_thinking():
    """MedGemma 1.5 thinks first and often drafts JSON there; only the answer after <unused95> counts."""
    obj = reader_json(Edema={"present": "yes", "probability": 90, "location": "bilateral"})
    text = ('<unused94>thought\nDraft: {"findings": {"Edema": {"present": "no"}}} Check format. Yes.'
            "<unused95>```json\n" + json.dumps(obj) + "\n```")
    assert prompts.parse_reader(text)["findings"]["Edema"]["present"] == "yes"
    with pytest.raises(ParseError, match="thinking"):
        prompts.parse_reader('<unused94>thought\nStill thinking {"findings": ')


def test_parse_reader_rejects_missing_or_bad():
    obj = reader_json()
    del obj["findings"]["Fracture"]
    with pytest.raises(ParseError, match="Fracture"):
        prompts.parse_reader(json.dumps(obj))
    with pytest.raises(ParseError):
        prompts.parse_reader(json.dumps(reader_json(Edema={"present": "maybe"})))


def test_parse_labels_derives_no_finding():
    labels = {f: "absent" for f in data.FINDINGS}
    out = prompts.parse_labels(json.dumps({"labels": labels | {"Support Devices": "present"}, "evidence": {}}))
    assert out["labels"]["No Finding"] == "present"  # support devices alone keep No Finding
    out = prompts.parse_labels(json.dumps({"labels": labels | {"Edema": "Uncertain"}}))
    assert out["labels"]["No Finding"] == "absent" and out["labels"]["Edema"] == "uncertain"
    with pytest.raises(ParseError):
        prompts.parse_labels(json.dumps({"labels": labels | {"Edema": "likely"}}))


def test_sample_end_to_end_mock(tmp_path):
    rows = []
    for k in range(400):
        pts = [{"n": n, "x": round(0.1 + 0.0137 * n, 4), "y": round(0.2 + 0.0071 * n, 4)} for n in range(1, 30)]
        view = "view1_frontal" if k % 10 else "view2_lateral"
        rows.append({"image_id": f"patient{k // 2:05d}_study{k % 2 + 1}_{view}", "chain_of_thought_reasoning": "",
                     "spatial_annotations": json.dumps(pts), "report": f"report {k}"})
    snapped = [{"n": n, "x": 0.1 * n, "y": 0.05} for n in range(1, 6)]
    rows.append({"image_id": "patient99999_study1_view1_frontal", "chain_of_thought_reasoning": "",
                 "spatial_annotations": json.dumps(snapped), "report": "snapped"})
    pd.DataFrame(rows).to_csv(tmp_path / "traces.csv", index=False)

    cfg = yaml.safe_load(open("configs/hints_mvp_mock.yaml"))
    cfg.update(csv=str(tmp_path / "traces.csv"), n_images=15, n_normal=5, label_batch=20, max_labeled=200,
               cache_dir=str(tmp_path / "cache"), output_dir=str(tmp_path / "out"),
               spend_file=str(tmp_path / "out" / "spend.json"))
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    s = sample.build(load_config(path))
    assert len(s) == 15 and sum(x["group"] == "normal" for x in s) == 5
    assert len({x["image_id"].split("_")[0] for x in s}) == 15  # one image per patient
    assert all(x["image_id"].endswith("frontal") and x["report"] != "snapped" for x in s)
    for x in s:
        t = x["trails"]
        assert t["other"]["from_image"] != x["image_id"]
        assert len(t["random"]["points"]) == len(t["own"]["points"])
        assert (x["group"] == "normal") == all(x["labels"][f] == "absent" for f in data.FINDINGS)
    assert (tmp_path / "out" / "image_paths.txt").read_text().count("\n") == 15
    assert json.dumps(sample.build(load_config(path))) == json.dumps(s)  # rerun: same sample, from the cache


def test_requests_per_condition():
    import re

    from PIL import Image

    from src.hints_mvp import run
    from src.hints_mvp.config import CONDITIONS

    s = {"image_id": "patient00001_study1_view1_frontal",
         "trails": {"own": {"points": [(0.1, 0.2)]}, "other": {"points": [(0.3, 0.4), (0.5, 0.6)]},
                    "random": {"points": [(0.7, 0.8)]}}}
    im = Image.new("L", (10, 10))
    parts = {c: run.request_for(s, c, im)[0] for c in CONDITIONS}
    assert parts["no_hint"] == parts["no_hint_rerun"] == [im, prompts.READ_TEXT]
    assert parts["own_trail"] == [im, prompts.trail_text([(0.1, 0.2)]) + "\n\n" + prompts.READ_TEXT]
    assert "(0.300, 0.400), (0.500, 0.600)" in parts["other_trail"][1]
    assert parts["leak_check"] == [prompts.trail_text([(0.1, 0.2)]) + "\n\n" + prompts.NO_IMAGE_TEXT]
    # conditions 3-6 use the identical sentence; only the points differ
    sentences = {re.sub(r"\(\d\.\d{3}, \d\.\d{3}\)(, )?", "", p[-1].split("\n\n")[0])
                 for c, p in parts.items() if c in run.TRAIL_OF}
    assert len(sentences) == 1


def test_score_changes_and_uncertain_rule():
    import numpy as np

    from src.hints_mvp import score

    # one image x 6 findings: labels positive, positive, uncertain, negative, negative, negative
    labels = ["present", "present", "uncertain", "absent", "absent", "absent"]
    base = np.array([[False, True, False, False, True, True]])
    new = np.array([[True, False, True, True, False, True]])
    r = [{"labels": dict(zip(data.FINDINGS, labels + ["absent"] * 7))}]
    y = score.label_array(r, "positive")[:, :6]
    ch = score.changes(y, base, new)
    assert ch == {"fixed": 2, "broken": 1, "new_false": 1, "cleared": 1, "recovery": 1.0, "break_rate": 1.0}
    y_ex = score.label_array(r, "excluded")[:, :6]
    assert np.isnan(y_ex[0, 2])
    assert score.changes(y_ex, base, new)["fixed"] == 1  # the uncertain cell no longer counts
    m = score.metrics(y_ex, new)
    assert m["sensitivity"] == 0.5 and m["specificity"] == 1 / 3
