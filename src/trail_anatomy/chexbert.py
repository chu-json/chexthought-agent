"""CheXbert report labels: present / uncertain / absent for the 13 CheXpert findings, run locally.

Runs in .venv-chexone (f1chexbert). Weights: hf download StanfordAIMI/RRG_scorers chexbert.pth --local-dir ~/.cache/chexbert
CheXbert's classes per finding are blank (not mentioned), positive, negative, uncertain. Blank and negative are
both "absent". No Finding follows CheXpert's rule from the 12 pathologies, as in hints_mvp.
The same labeler labels the reference reports and the predicted reports.
"""
from ..hints_mvp.data import FINDINGS, PATHOLOGIES

WORD = {1: "present", -1: "uncertain", 0: "absent", "": "absent"}


class Labeler:
    def __init__(self, device="cuda"):
        from f1chexbert import F1CheXbert
        self.model = F1CheXbert(device=device)
        self.names = self.model.target_names

    def __call__(self, report):
        if not report or not report.strip():
            return {"No Finding": "present", **{f: "absent" for f in FINDINGS}}
        out = dict(zip(self.names, self.model.get_label(report, mode="classification")))
        labels = {f: WORD[out[f]] for f in FINDINGS}
        normal = all(labels[f] == "absent" for f in PATHOLOGIES)
        return {"No Finding": "present" if normal else "absent", **labels}
