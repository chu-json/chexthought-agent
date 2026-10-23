"""Prompt text for the reader and helper, and strict validation of their JSON outputs."""
import json

from . import boxes as bx
from .data import FINDINGS

FINDING_LIST = ", ".join(FINDINGS)

READER_SYSTEM = f"""You are an expert radiologist reading frontal chest X-rays.

For each of these 10 findings, decide whether it is present on the image: {FINDING_LIST}.

Respond with only a JSON object, no other text, in exactly this format:
{{"findings": {{"<finding>": {{"present": "yes" or "no", "probability": <integer 0-100, your probability that the finding is present>, "location": "<short description of where, or empty string>"}}, ...}}, "reasoning": "<short summary of your reasoning>"}}

"findings" must have exactly one entry for each of the 10 findings, using the exact names above."""

PASS1_TEXT = "Read this chest X-ray and fill in the report."

# Pass-2 wording. The three sentences in BOX_SENTENCES + REVISE are identical for every box condition;
# look_again uses the same prompt without the box sentences.
BOX_SENTENCES = (
    "A colleague flagged these regions for a second look. They may or may not contain abnormalities. "
)
REVISE = "Review your report and revise it if needed."
RETURN_FULL = "Return the complete report in the same JSON format."

PASS2_TEXT_BOXES = (
    "The first image is the chest X-ray you read. The second image is the same X-ray with numbered boxes drawn on it.\n\n"
    "Your earlier report:\n{report}\n\n" + BOX_SENTENCES + REVISE + " " + RETURN_FULL
)
PASS2_TEXT_NO_BOXES = (
    "This is the chest X-ray you read.\n\n"
    "Your earlier report:\n{report}\n\n" + REVISE + " " + RETURN_FULL
)

RETRY_NUDGE = "\n\nReturn only valid JSON in exactly the format specified, with no other text."

HELPER_SYSTEM = (
    "You are an expert radiologist double-checking a colleague's chest X-ray report. "
    "Your job is to point out regions where the colleague most likely missed a finding."
)

HELPER_TEXT = """Your colleague read this frontal chest X-ray and wrote this report:
{report}

Propose up to 3 boxes around regions where the colleague most likely missed something: a finding that is visible on the image but marked "no" in the report. Return fewer boxes, or none, if you do not think anything was missed.

{coords}

Respond with only a JSON object, no other text, in exactly this format:
{{"boxes": [{{"box": {placeholder}, "suspected_finding": "<one of: {findings}, or other>", "reason": "<short reason>"}}]}}"""

COORDS = {
    "pixel": ("Give each box in pixel coordinates of this image, which is {width} pixels wide and {height} pixels "
              "tall, with (0, 0) at the top-left corner.", "[x1, y1, x2, y2]"),
    "norm": ("Give each box as fractions of the image width and height (each between 0 and 1), with (0, 0) at the "
             "top-left corner.", "[x1, y1, x2, y2]"),
    "norm1000": ("Give each box on a 0-1000 scale of the image width and height, with (0, 0) at the top-left "
                 "corner.", "[x1, y1, x2, y2]"),
    "yxyx1000": ("Give each box on a 0-1000 scale of the image height and width, with (0, 0) at the top-left "
                 "corner.", "[y1, x1, y2, x2]"),
}


def report_json(report):
    return json.dumps(report)


def pass1_text():
    return PASS1_TEXT


def pass2_text(report, with_boxes):
    template = PASS2_TEXT_BOXES if with_boxes else PASS2_TEXT_NO_BOXES
    return template.format(report=report_json(report))


def helper_text(report, fmt, width, height):
    coords, placeholder = COORDS[fmt]
    return HELPER_TEXT.format(
        report=report_json(report), coords=coords.format(width=width, height=height),
        placeholder=placeholder, findings=FINDING_LIST,
    )


class ParseError(ValueError):
    pass


def extract_json(text):
    """Parse the outermost JSON object in a response (tolerates code fences or text around it)."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ParseError("no JSON object in response")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise ParseError(f"invalid JSON: {e}") from None


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def parse_reader(text):
    """Validate a reader report. Returns {"findings": {finding: {present, probability, location}}, "reasoning"}."""
    obj = extract_json(text)
    findings = obj.get("findings") if isinstance(obj, dict) else None
    if not isinstance(findings, dict) or set(findings) != set(FINDINGS):
        got = sorted(findings) if isinstance(findings, dict) else findings
        raise ParseError(f"'findings' must have exactly the 10 findings; got {got}")
    out = {}
    for f in FINDINGS:
        v = findings[f]
        if not isinstance(v, dict):
            raise ParseError(f"{f}: entry must be an object")
        if v.get("present") not in ("yes", "no"):
            raise ParseError(f"{f}: 'present' must be 'yes' or 'no'")
        if not _is_number(v.get("probability")) or not 0 <= v["probability"] <= 100:
            raise ParseError(f"{f}: 'probability' must be a number 0-100")
        if not isinstance(v.get("location"), str):
            raise ParseError(f"{f}: 'location' must be a string")
        out[f] = {"present": v["present"], "probability": v["probability"], "location": v["location"]}
    if not isinstance(obj.get("reasoning"), str):
        raise ParseError("'reasoning' must be a string")
    return {"findings": out, "reasoning": obj["reasoning"]}


def parse_helper(text, fmt, width, height):
    """Validate helper output and convert its boxes to normalized coordinates.

    Returns [{"box", "raw_box", "suspected_finding", "reason"}], at most 3.
    """
    obj = extract_json(text)
    raw = obj.get("boxes") if isinstance(obj, dict) else None
    if not isinstance(raw, list) or len(raw) > bx.N_BOXES:
        raise ParseError(f"'boxes' must be a list of at most {bx.N_BOXES}")
    out = []
    for b in raw:
        if not isinstance(b, dict):
            raise ParseError("each box entry must be an object")
        box = b.get("box")
        if not (isinstance(box, list) and len(box) == 4 and all(_is_number(v) for v in box)):
            raise ParseError("'box' must be a list of 4 numbers")
        if b.get("suspected_finding") not in FINDINGS + ["other"]:
            raise ParseError(f"bad suspected_finding {b.get('suspected_finding')!r}")
        if not isinstance(b.get("reason"), str):
            raise ParseError("'reason' must be a string")
        norm = bx.to_norm(box, fmt, width, height)
        if bx.area(norm) == 0:
            raise ParseError(f"box {box} has no area inside the image")
        out.append({"box": norm, "raw_box": box, "suspected_finding": b["suspected_finding"], "reason": b["reason"]})
    return out
