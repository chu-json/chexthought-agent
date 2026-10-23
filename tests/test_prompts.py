import json

import pytest

from src import prompts
from src.data import FINDINGS


def report(**overrides):
    findings = {f: {"present": "no", "probability": 10, "location": ""} for f in FINDINGS}
    findings.update(overrides)
    return {"findings": findings, "reasoning": "ok"}


def test_valid_report_parses():
    r = report(**{"Pleural Effusion": {"present": "yes", "probability": 85.5, "location": "left base"}})
    assert prompts.parse_reader(json.dumps(r)) == r


def test_code_fenced_report_parses():
    assert prompts.parse_reader("```json\n" + json.dumps(report()) + "\n```") == report()


@pytest.mark.parametrize("bad", [
    "no json here",
    "{not json}",
    json.dumps({"findings": {}, "reasoning": ""}),  # missing findings
    json.dumps(report() | {"reasoning": 3}),
    json.dumps(report(Edema={"present": "maybe", "probability": 10, "location": ""})),
    json.dumps(report(Edema={"present": "no", "probability": 150, "location": ""})),
    json.dumps(report(Edema={"present": "no", "probability": True, "location": ""})),
    json.dumps(report(Edema={"present": "no", "probability": "10", "location": ""})),
    json.dumps(report(Edema={"present": "no", "probability": 10})),  # no location
    json.dumps({"findings": report()["findings"] | {"Pneumonia": {}}, "reasoning": ""}),  # extra finding
])
def test_invalid_reports_raise(bad):
    with pytest.raises(prompts.ParseError):
        prompts.parse_reader(bad)


def test_helper_boxes_convert_to_normalized():
    text = json.dumps({"boxes": [{"box": [102.4, 168, 614.4, 756], "suspected_finding": "Lung Lesion", "reason": "r"}]})
    out = prompts.parse_helper(text, "pixel", 1024, 840)
    assert out[0]["box"] == pytest.approx((0.1, 0.2, 0.6, 0.9))
    assert out[0]["raw_box"] == [102.4, 168, 614.4, 756]
    assert prompts.parse_helper('{"boxes": []}', "pixel", 1024, 840) == []


@pytest.mark.parametrize("boxes", [
    [{"box": [0, 0, 10, 10], "suspected_finding": "other", "reason": ""}] * 4,  # too many
    [{"box": [0, 0, 10], "suspected_finding": "other", "reason": ""}],
    [{"box": [0, 0, 10, 10], "suspected_finding": "Pneumonia", "reason": ""}],  # not one of the 10
    [{"box": [2000, 2000, 3000, 3000], "suspected_finding": "other", "reason": ""}],  # outside image
    [{"box": [0, 0, 10, 10], "suspected_finding": "other"}],  # no reason
])
def test_invalid_helper_output_raises(boxes):
    with pytest.raises(prompts.ParseError):
        prompts.parse_helper(json.dumps({"boxes": boxes}), "pixel", 1024, 840)


SPEC_SENTENCES = ("A colleague flagged these regions for a second look. They may or may not contain abnormalities. "
                  "Review your report and revise it if needed.")


def test_pass2_wording():
    with_boxes = prompts.pass2_text(report(), with_boxes=True)
    without = prompts.pass2_text(report(), with_boxes=False)
    assert SPEC_SENTENCES in with_boxes
    assert "colleague" not in without and "Review your report and revise it if needed." in without
    assert json.dumps(report()) in with_boxes and json.dumps(report()) in without


def test_helper_prompt_states_image_size_for_pixel_boxes():
    text = prompts.helper_text(report(), "pixel", 1024, 840)
    assert "1024 pixels wide and 840 pixels tall" in text
