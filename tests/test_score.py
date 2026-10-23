import numpy as np
import pytest

from src import score

# Toy example, 4 images x 2 findings.
Y = np.array([[1, 0], [1, 1], [0, 0], [0, 1]])
P = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
S = np.array([[90, 20], [40, 80], [10, 20], [60, 80]], float)


def test_finding_metrics_known_answers():
    m = score.finding_metrics(Y, P, S)
    # Finding 1: TP=1 FN=1 TN=1 FP=1. Positives scored 90, 40; negatives 10, 60 -> 3 of 4 pairs ordered.
    assert m["sensitivity"][0] == 0.5 and m["specificity"][0] == 0.5 and m["f1"][0] == 0.5
    assert m["auroc"][0] == 0.75
    # Finding 2: perfect.
    assert m["sensitivity"][1] == 1 and m["specificity"][1] == 1 and m["f1"][1] == 1 and m["auroc"][1] == 1
    assert score.macro(m)["auroc"] == pytest.approx(0.875)


def test_auroc_ties_and_single_class():
    assert score.auroc(np.array([[1], [0], [1], [0]]), np.array([[50.0], [50], [50], [50]]))[0] == 0.5
    assert np.isnan(score.auroc(np.array([[1], [1]]), np.array([[1.0], [2]]))[0])


def test_changes_known_answers():
    ref = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    new = np.array([[1, 1], [0, 0], [0, 0], [1, 1]])
    y = np.array([[1, 0], [0, 1], [0, 1], [1, 1]])
    # (0,0) FN->TP fixed; (0,1) TN->FP added; (1,1) TP->FN lost; (2,0) FP->TN cleared.
    assert score.changes(y, ref, new) == {"fixed": 1, "added": 1, "net": 0, "lost": 1, "cleared": 1}


def test_box_quality():
    mask = np.zeros((100, 100), bool)
    mask[0:50, 0:50] = True
    r = {"boxes": {"oracle_boxes": [
        {"box": (0.0, 0.0, 0.5, 0.5), "source": "oracle:Edema"},  # fully inside: hit
        {"box": (0.4, 0.4, 0.9, 0.9), "source": "pad"},  # 4% inside: miss, but overlaps
        {"box": (0.6, 0.6, 0.9, 0.9), "source": "pad"},  # outside
    ]}}
    q = score.box_quality(r, "oracle_boxes", {"Edema": mask}, missed=["Edema", "Lung Lesion"])
    assert q == {"hits": 1, "boxes": 3, "real_hits": 1, "real_boxes": 1, "covered": 1, "covered_strict": 1,
                 "misses": 1}
    # Only the 4%-inside box: covered by any overlap, but not strictly.
    r["boxes"]["oracle_boxes"] = r["boxes"]["oracle_boxes"][1:]
    q = score.box_quality(r, "oracle_boxes", {"Edema": mask}, missed=["Edema"])
    assert q["covered"] == 1 and q["covered_strict"] == 0


def test_bootstrap_identical_conditions_have_zero_net_change():
    preds = {"alone": P, "look_again": P.copy()}
    probs = {"alone": S, "look_again": S.copy()}
    ci = score.bootstrap(Y, preds, probs, {}, ["alone", "look_again"], n_boot=50, seed=0)
    assert ci[("look_again", "net_vs_alone")] == (0, 0)
