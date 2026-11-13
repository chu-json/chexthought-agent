"""hints_mvp prompt text (report labeler and reader) and tolerant parsing of their JSON outputs."""
import json
import re

from ..prompts import ParseError
from .data import CHEXPERT, FINDINGS, PATHOLOGIES

# --- Report labeler: the reference labels, from the clinical report. ---

LABELER_SYSTEM = f"""You label chest X-ray radiology reports with CheXpert findings.

For each of these 13 findings, give the label the report supports: {", ".join(FINDINGS)}.
- "present": the report says the finding is there, including when it is described as stable, unchanged, improved or decreased but still there.
- "uncertain": the report is uncertain about it (for example "possible", "may represent", "cannot be excluded", "atelectasis versus pneumonia").
- "absent": the report says it is not there or has resolved, or does not mention it.

Lung Opacity covers any opacity, opacification, density, infiltrate or airspace disease in the lungs. Pleural Other covers pleural abnormalities other than effusion, such as thickening or plaques. Support Devices covers any line, tube, catheter, drain, pacemaker, wire, valve or other medical device. Fracture covers any fracture, including old or healed ones. The report may describe more than one view; label the whole report.

Respond with only a JSON object, no other text, in exactly this format:
{{"labels": {{"<finding>": "present" or "uncertain" or "absent", ...}}, "evidence": {{"<finding>": "<short quote from the report>", ...}}}}

"labels" must have exactly one entry for each of the 13 findings, using the exact names above. "evidence" has an entry only for findings labeled present or uncertain."""


def labeler_text(report):
    return f"Report:\n{report}"


# --- Reader: the model under test. ---

READER_SYSTEM = f"""You are an expert radiologist reading frontal chest X-rays.

For each of these 14 findings, decide whether it is present on the image: {", ".join(CHEXPERT)}.
"No Finding" means that none of the other findings, apart from Support Devices, is present.

Respond with only a JSON object, no other text, in exactly this format:
{{"findings": {{"<finding>": {{"present": "yes" or "no", "probability": <integer 0-100, your probability that the finding is present>, "location": "<short description of where, or empty string>"}}, ...}}, "reasoning": "<short summary of your reasoning>"}}

"findings" must have exactly one entry for each of the 14 findings, using the exact names above."""

READ_TEXT = "Read this chest X-ray and fill in the report."
# Identical for own_trail, other_trail, random_trail and leak_check; only the points differ.
TRAIL_TEXT = ("A radiologist's search path for this image (x, y as fractions of width and height, in viewing order): "
              "{points}. It shows where they looked and may not indicate any finding.")
NO_IMAGE_TEXT = "The image is not provided. Fill in the report as best you can from the search path alone."


def trail_text(points):
    return TRAIL_TEXT.format(points=", ".join(f"({x:.3f}, {y:.3f})" for x, y in points))


def reader_text(points=None, with_image=True):
    """The text sent after the image (or alone, for leak_check). points=None: no trail (no_hint)."""
    if points is None:
        return READ_TEXT
    return trail_text(points) + "\n\n" + (READ_TEXT if with_image else NO_IMAGE_TEXT)


# --- Tolerant parsing: small local models (MedGemma) often bend the format. ---

# MedGemma 1.5 may think first: "<unused94>thought ... <unused95>" and then the answer. The thinking often
# drafts JSON, so only the text after the end-of-thought marker is parsed.
THOUGHT_START, THOUGHT_END = "<unused94>", "<unused95>"


def strip_thought(text):
    if THOUGHT_END in text:
        return text.rsplit(THOUGHT_END, 1)[1]
    if THOUGHT_START in text:
        raise ParseError("response ended inside the thinking block")
    return text


def loads_tolerant(text):
    """The outermost JSON object in a response, tolerating a thinking block, code fences, text around it and
    trailing commas."""
    text = strip_thought(text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ParseError("no JSON object in response")
    s = text[start:end + 1]
    err = None
    for candidate in (s, re.sub(r",\s*([}\]])", r"\1", s)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            err = e
    raise ParseError(f"invalid JSON: {err}")


def _key(name):
    return re.sub(r"[^a-z]", "", str(name).lower())


def match_names(d, names):
    """Map a dict's keys onto the expected names, ignoring case, spaces and punctuation. Extra keys are ignored."""
    by_key = {_key(k): v for k, v in d.items()}
    missing = [n for n in names if _key(n) not in by_key]
    if missing:
        raise ParseError(f"missing findings: {missing}")
    return {n: by_key[_key(n)] for n in names}


YES, NO = {"yes", "true", "present", "y", "1"}, {"no", "false", "absent", "n", "0"}
LABEL_WORDS = {"present": "present", "positive": "present", "yes": "present",
               "uncertain": "uncertain", "possible": "uncertain", "equivocal": "uncertain",
               "absent": "absent", "negative": "absent", "no": "absent", "notmentioned": "absent"}


def _probability(v):
    """A number 0-100, or None if missing or unreadable (probability isn't used for the headline metrics)."""
    if isinstance(v, str):
        try:
            v = float(v.strip().rstrip("%"))
        except ValueError:
            return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 100:
        return None
    return v


def parse_reader(text):
    """Reader report -> {"findings": {finding: {present, probability, location}}, "reasoning"}, all 14 findings."""
    obj = loads_tolerant(text)
    findings = obj.get("findings") if isinstance(obj, dict) else None
    if not isinstance(findings, dict):
        raise ParseError("'findings' must be an object")
    out = {}
    for f, v in match_names(findings, CHEXPERT).items():
        if not isinstance(v, dict):  # tolerate "Edema": "yes"
            v = {"present": v}
        word = str(v.get("present", "")).strip().lower()
        if word not in YES | NO:
            raise ParseError(f"{f}: 'present' must be 'yes' or 'no', got {v.get('present')!r}")
        out[f] = {"present": "yes" if word in YES else "no", "probability": _probability(v.get("probability")),
                  "location": str(v.get("location") or "")}
    reasoning = obj.get("reasoning")
    return {"findings": out, "reasoning": reasoning if isinstance(reasoning, str) else ""}


def parse_labels(text):
    """Labeler output -> {"labels": {finding: present|uncertain|absent} for all 14, "evidence": {...}}.

    The labeler labels the 13 findings; No Finding follows CheXpert's rule from the pathologies.
    """
    obj = loads_tolerant(text)
    raw = obj.get("labels") if isinstance(obj, dict) else None
    if not isinstance(raw, dict):
        raise ParseError("'labels' must be an object")
    labels = {}
    for f, v in match_names(raw, FINDINGS).items():
        if _key(v) not in LABEL_WORDS:
            raise ParseError(f"{f}: label must be present, uncertain or absent, got {v!r}")
        labels[f] = LABEL_WORDS[_key(v)]
    normal = all(labels[f] == "absent" for f in PATHOLOGIES)
    evidence = obj.get("evidence") if isinstance(obj.get("evidence"), dict) else {}
    return {"labels": {"No Finding": "present" if normal else "absent", **labels},
            "evidence": {str(k): str(v) for k, v in evidence.items()}}
