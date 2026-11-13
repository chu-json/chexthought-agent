"""Regions named in text (a radiologist's reasoning, or a report), on the same 15-region partition.

A simple local keyword lexicon, no model. Used to
- check focus regions against the radiologist's own written reasoning (are emphasized regions discussed?), and
- build the report_regions ceiling hint: regions named in the report's positive (non-negated) sentences.
Negation is crude, sentence-level: a sentence with a negation cue is skipped.
"""
import re

R, L = "right", "left"
LOWER, MID, UPPER = "lower lung", "mid lung", "upper lung"

# (pattern, regions). Patterns are matched on lowercased text.
LEXICON = [
    (r"\bright (upper|apical|apex|apices)\b|\bright upper lobe\b|\brul\b|\bright lung apex\b", [f"{R} {UPPER}"]),
    (r"\bleft (upper|apical|apex|apices)\b|\bleft upper lobe\b|\blul\b|\bleft lung apex\b", [f"{L} {UPPER}"]),
    (r"\bright (mid|middle)\b|\bright middle lobe\b|\brml\b", [f"{R} {MID}"]),
    (r"\bleft (mid|middle)\b|\blingula", [f"{L} {MID}"]),
    (r"\bright (lower|base|basilar|basal|lung base)\b|\bright lower lobe\b|\brll\b|\bright costophrenic", [f"{R} {LOWER}"]),
    (r"\bleft (lower|base|basilar|basal|lung base)\b|\bleft lower lobe\b|\blll\b|\bleft costophrenic|\bretrocardiac", [f"{L} {LOWER}"]),
    (r"\bbibasilar\b|\bbibasal\b|\bbilateral (lower|basilar|basal|bases)\b|\b(both|lung) bases\b|\bbilateral pleural|\bbilateral effusions?\b",
     [f"{R} {LOWER}", f"{L} {LOWER}"]),
    (r"\bbilateral (upper|apical|apices)\b|\bbiapical\b|\b(both|lung) apices\b", [f"{R} {UPPER}", f"{L} {UPPER}"]),
    (r"\bright (hemi)?diaphragm", ["right hemidiaphragm"]),
    (r"\bleft (hemi)?diaphragm", ["left hemidiaphragm"]),
    (r"\bheart\b|\bcardiac\b|\bcardiomegaly\b|\bpericardi", ["heart"]),
    (r"\bmediastin|\baort(a|ic)\b|\bcardiomediastinal\b", ["mediastinum"]),
    (r"\bhil(um|a|ar)\b|\bperihilar\b", ["hila"]),
    (r"\btrache|\bendotracheal\b|\bett\b|\bcarina\b", ["trachea"]),
    (r"\bclavic", ["clavicles"]),
    (r"\bspine\b|\bvertebra|\bspinal\b", ["spine"]),
    (r"\babdom|\bstomach\b|\bbowel\b|\bsubdiaphragmatic\b|\bfree air\b", ["abdomen"]),
]
NEGATION = re.compile(r"\b(no|not|without|negative|clear|resolved|normal|unremarkable|absent|free of)\b")


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.;:\n])\s+|\n+", str(text)) if s.strip()]


def mentioned(text, positive_only=False):
    """Set of region names in the text. positive_only: skip sentences with a negation cue."""
    found = set()
    for s in sentences(text.lower() if isinstance(text, str) else ""):
        if positive_only and NEGATION.search(s):
            continue
        for pat, regs in LEXICON:
            if re.search(pat, s):
                found.update(regs)
    return found


def report_findings_text(report):
    """The impression and findings parts of a CheXpert Plus report (history and comparison dropped)."""
    t = str(report)
    m = re.search(r"(FINDINGS|IMPRESSION)\s*:", t, flags=re.I)
    t = t[m.start():] if m else t
    return re.split(r"(END OF IMPRESSION|SUMMARY\s*:|ACCESSION NUMBER)", t, flags=re.I)[0]
