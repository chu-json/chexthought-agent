import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src import boxes as bx

BOX = (0.1, 0.2, 0.6, 0.9)
SIZES = [(0.2, 0.3), (0.1, 0.1), (0.5, 0.4)]
DATA = Path("~/data/chexlocalize").expanduser()


@pytest.mark.parametrize("fmt", bx.FORMATS)
def test_round_trip(fmt):
    native = bx.from_norm(BOX, fmt, 1024, 840)
    assert bx.to_norm(native, fmt, 1024, 840) == pytest.approx(BOX)


def test_known_conversions():
    assert bx.to_norm([102.4, 168, 614.4, 756], "pixel", 1024, 840) == pytest.approx(BOX)
    assert bx.to_norm([100, 200, 600, 900], "norm1000") == pytest.approx(BOX)
    assert bx.to_norm([200, 100, 900, 600], "yxyx1000") == pytest.approx(BOX)


def test_corners_are_ordered_and_clipped():
    assert bx.to_norm([0.6, 0.9, 0.1, 0.2], "norm") == pytest.approx(BOX)
    assert bx.to_norm([-5, -5, 2000, 500], "pixel", 1000, 1000) == pytest.approx((0, 0, 1, 0.5))
    with pytest.raises(ValueError):
        bx.to_norm([0, 0, 1, 1], "xywh")


def test_drawn_boxes_land_where_expected():
    im = Image.new("L", (1000, 800), 0)
    box = (0.2, 0.25, 0.6, 0.75)  # pixels: x 200-600, y 200-600
    out = np.array(bx.draw_boxes(im, [box]))
    assert out.shape == (800, 1000, 3)
    red = (out[..., 0] == 255) & (out[..., 1] == 0) & (out[..., 2] == 0)
    assert red[400, 200] and red[400, 599]  # left and right edges, mid-height
    assert red[200, 400] and red[599, 400]  # top and bottom edges, mid-width
    assert not red[400, 400]  # interior untouched
    assert not red[400, 190] and not red[400, 610] and not red[610, 400]  # outside untouched
    assert red[198, 201]  # number tag sits just above the top-left corner
    assert not red[100, 900] and out[100, 900].tolist() == [0, 0, 0]  # rest of the image unchanged


def test_tag_goes_inside_box_at_top_edge():
    out = np.array(bx.draw_boxes(Image.new("L", (1000, 800), 0), [(0.2, 0.0, 0.3, 0.3)]))
    red = (out[..., 0] == 255) & (out[..., 1] == 0)
    assert red[4, 201] and red[20, 201]  # tag fills the box's top-left corner
    assert not red[260, 201]  # and nothing is drawn below the box


def test_boxes_drawn_identically():
    """Same pixels for a box whatever list it is in, apart from its number."""
    im = Image.new("L", (500, 500), 128)
    a = np.array(bx.draw_boxes(im, [(0.1, 0.5, 0.3, 0.7)]))
    b = np.array(bx.draw_boxes(im, [(0.6, 0.1, 0.9, 0.3), (0.1, 0.5, 0.3, 0.7)]))
    # Compare the second box's outline below its tag.
    assert (a[260:350, 40:160] == b[260:350, 40:160]).all()


def test_every_box_condition_has_three_boxes():
    gt = {"A": (0.1, 0.1, 0.3, 0.3), "B": (0.5, 0.5, 0.9, 0.9), "C": (0.1, 0.6, 0.3, 0.9),
          "D": (0.6, 0.1, 0.9, 0.3), "E": (0.4, 0.4, 0.5, 0.5)}
    sets = [bx.random_boxes(random.Random(0), SIZES), bx.checklist_boxes(random.Random(0)),
            bx.helper_boxes(None, random.Random(0), SIZES)]
    for n in range(4):
        sets.append(bx.helper_boxes([{"box": (0.1, 0.1, 0.3, 0.3)}] * n, random.Random(0), SIZES))
    for present in ([], ["A"], list(gt)):
        sets.append(bx.oracle_boxes(gt, present, set(), random.Random(0), SIZES))
    for s in sets:
        assert len(s) == bx.N_BOXES
        for b in s:
            x1, y1, x2, y2 = b["box"]
            assert 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1


def test_random_boxes_use_ground_truth_sizes():
    for b in bx.random_boxes(random.Random(1), SIZES):
        x1, y1, x2, y2 = b["box"]
        assert any((x2 - x1, y2 - y1) == pytest.approx(s) for s in SIZES)


def test_padding_position_is_shuffled():
    """A single real box must not always be box 1, or the numbering would reveal it."""
    positions = set()
    for seed in range(30):
        s = bx.helper_boxes([{"box": (0.1, 0.1, 0.3, 0.3)}], bx.rng_for(seed, "img", "helper_boxes"), SIZES)
        positions.add(next(k for k, b in enumerate(s) if b["source"] == "helper"))
    assert positions == {0, 1, 2}


def test_seeded_boxes_are_reproducible():
    a = bx.random_boxes(bx.rng_for(0, "img", "random_boxes"), SIZES)
    b = bx.random_boxes(bx.rng_for(0, "img", "random_boxes"), SIZES)
    assert a == b


def test_oracle_prefers_misses_and_skips_duplicates():
    gt = {"A": (0.1, 0.1, 0.3, 0.3), "A2": (0.1, 0.1, 0.3, 0.31), "B": (0.5, 0.5, 0.9, 0.9),
          "C": (0.1, 0.6, 0.3, 0.9), "D": (0.6, 0.1, 0.9, 0.3)}
    for seed in range(10):
        s = bx.oracle_boxes(gt, list(gt), {"D"}, random.Random(seed), SIZES)
        sources = [b["source"] for b in s]
        assert "oracle:D" in sources  # the pass-1 miss is always shown
        assert not ("oracle:A" in sources and "oracle:A2" in sources)  # near-duplicates skipped


def test_largest_component_box_and_overlap():
    mask = np.zeros((100, 200), bool)
    mask[10:50, 20:120] = True  # large component
    mask[80:90, 150:160] = True  # small component
    box = bx.largest_component_box(mask)
    assert box == pytest.approx((20 / 200, 10 / 100, 120 / 200, 50 / 100))
    assert bx.fraction_inside(box, mask) == pytest.approx(1.0)
    assert bx.overlaps(box, mask)
    assert not bx.overlaps((0.0, 0.6, 0.5, 1.0), mask)
    assert bx.largest_component_box(np.zeros((5, 5), bool)) is None


@pytest.mark.skipif(not DATA.exists(), reason="dataset not available")
def test_oracle_boxes_overlap_their_masks_on_real_data():
    from src import data
    masks = data.load_masks(DATA, "val")
    for iid in sorted(masks)[:5]:
        decoded = {f: m for f, rle in masks[iid].items() if (m := data.decode(rle)).any()}
        gt = {f: bx.largest_component_box(m) for f, m in decoded.items()}
        for b in bx.oracle_boxes(gt, list(gt), set(), random.Random(0), SIZES):
            if b["source"].startswith("oracle:"):
                assert bx.overlaps(b["box"], decoded[b["source"].split(":", 1)[1]])


def test_unpadded_boxes_keep_only_real_ones():
    real = [{"box": (0.1, 0.1, 0.3, 0.3)}, {"box": (0.5, 0.5, 0.7, 0.7)}]
    for n in range(3):
        s = bx.helper_boxes(real[:n], random.Random(0), SIZES, pad=False)
        assert len(s) == n and all(b["source"] == "helper" for b in s)
    gt = {"A": (0.1, 0.1, 0.3, 0.3)}
    assert [b["source"] for b in bx.oracle_boxes(gt, ["A"], set(), random.Random(0), SIZES, pad=False)] == ["oracle:A"]
