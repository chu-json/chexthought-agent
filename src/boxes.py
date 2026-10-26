"""Box math. Every box is normalized (x1, y1, x2, y2) in [0, 1], origin at the image's top-left.

Model-native formats are converted to and from normalized boxes only in `to_norm` / `from_norm`.
"""
import random

import numpy as np
from PIL import ImageDraw, ImageFont
from scipy import ndimage

N_BOXES = 3
BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)

# Native box formats: pixel = absolute pixels of the image the model saw; norm1000 = 0-1000 scale
# (e.g. Qwen3-VL); yxyx1000 = (y1, x1, y2, x2) on a 0-1000 scale (e.g. Gemma / MedGemma).
FORMATS = ("norm", "pixel", "norm1000", "yxyx1000")

# Standard review areas on a frontal film. The patient's right is on the image's left.
# Crude, hand-placed for a typical PA/AP chest X-ray.
CHECKLIST = {
    "right_apex": (0.15, 0.08, 0.45, 0.30),
    "left_apex": (0.55, 0.08, 0.85, 0.30),
    "right_hilum": (0.30, 0.32, 0.48, 0.52),
    "left_hilum": (0.52, 0.32, 0.70, 0.52),
    "retrocardiac": (0.50, 0.50, 0.72, 0.78),
    "right_costophrenic_angle": (0.08, 0.62, 0.35, 0.88),
    "left_costophrenic_angle": (0.65, 0.62, 0.92, 0.88),
}


def clean(box):
    """Order the corners and clip to [0, 1]."""
    x1, y1, x2, y2 = (min(max(float(v), 0.0), 1.0) for v in box)
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def to_norm(box, fmt, width=None, height=None):
    """Convert a box from a model's native format to a normalized (x1, y1, x2, y2) box."""
    a, b, c, d = (float(v) for v in box)
    if fmt == "norm":
        return clean((a, b, c, d))
    if fmt == "pixel":
        return clean((a / width, b / height, c / width, d / height))
    if fmt == "norm1000":
        return clean((a / 1000, b / 1000, c / 1000, d / 1000))
    if fmt == "yxyx1000":
        return clean((b / 1000, a / 1000, d / 1000, c / 1000))
    raise ValueError(f"unknown box format {fmt!r}")


def from_norm(box, fmt, width=None, height=None):
    """Inverse of `to_norm` (used by the mock backend and tests)."""
    x1, y1, x2, y2 = box
    if fmt == "norm":
        return [x1, y1, x2, y2]
    if fmt == "pixel":
        return [x1 * width, y1 * height, x2 * width, y2 * height]
    if fmt == "norm1000":
        return [x1 * 1000, y1 * 1000, x2 * 1000, y2 * 1000]
    if fmt == "yxyx1000":
        return [y1 * 1000, x1 * 1000, y2 * 1000, x2 * 1000]
    raise ValueError(f"unknown box format {fmt!r}")


def to_pixels(box, width, height):
    """Integer pixel corners (x1, y1, x2, y2), x2/y2 exclusive."""
    x1, y1, x2, y2 = box
    return (round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height))


def area(box):
    return max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)


def iou(a, b):
    inter = area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def largest_component_box(mask):
    """Normalized box around the largest connected component of a boolean mask, or None if empty."""
    if not mask.any():
        return None
    labeled, _ = ndimage.label(mask)
    biggest = np.bincount(labeled.ravel())[1:].argmax() + 1
    ys, xs = np.nonzero(labeled == biggest)
    h, w = mask.shape
    return (xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h)


def fraction_inside(box, mask):
    """Share of the box's area that lies inside the mask."""
    x1, y1, x2, y2 = to_pixels(box, mask.shape[1], mask.shape[0])
    n = (x2 - x1) * (y2 - y1)
    return float(mask[y1:y2, x1:x2].sum()) / n if n > 0 else 0.0


def overlaps(box, mask):
    """True if any mask pixel lies inside the box."""
    x1, y1, x2, y2 = to_pixels(box, mask.shape[1], mask.shape[0])
    return bool(mask[y1:y2, x1:x2].any())


# --- Box sets for the pass-2 conditions: lists of {"box", "source"}, exactly N_BOXES when padded. ---

def rng_for(seed, image_id, condition):
    """Deterministic RNG per (seed, image, condition); str seeds are hashed stably by `random`."""
    return random.Random(f"{seed}|{image_id}|{condition}")


def random_box(rng, sizes):
    """Random box whose (width, height) is drawn from `sizes` (ground-truth box sizes)."""
    w, h = rng.choice(sizes)
    x1, y1 = rng.uniform(0, 1 - w), rng.uniform(0, 1 - h)
    return (x1, y1, x1 + w, y1 + h)


def pad_and_shuffle(real, rng, sizes, pad=True):
    """Pad to N_BOXES with random boxes (if pad), then shuffle so the numbering doesn't reveal which are real.

    Without padding (v1) only the real boxes are shown; shuffling still keeps their order from carrying a ranking.
    """
    out = [dict(b) for b in real[:N_BOXES]]
    while pad and len(out) < N_BOXES:
        out.append({"box": random_box(rng, sizes), "source": "pad"})
    rng.shuffle(out)
    return out


def random_boxes(rng, sizes):
    return [{"box": random_box(rng, sizes), "source": "random"} for _ in range(N_BOXES)]


def checklist_boxes(rng):
    return [{"box": CHECKLIST[k], "source": f"checklist:{k}"} for k in rng.sample(sorted(CHECKLIST), N_BOXES)]


def helper_boxes(helper_out, rng, sizes, pad=True):
    real = [{"box": b["box"], "source": "helper"} for b in helper_out or []]
    return pad_and_shuffle(real, rng, sizes, pad)


def oracle_boxes(gt_for_image, present, missed, rng, sizes, pad=True):
    """Boxes around ground-truth masks of present findings.

    Findings the reader missed in pass 1 come first, ties broken at random; a box that nearly
    duplicates one already chosen (IoU > 0.5, e.g. Cardiomegaly vs Enlarged Cardiomediastinum) is skipped.
    """
    candidates = sorted(f for f in present if f in gt_for_image)
    rng.shuffle(candidates)
    candidates.sort(key=lambda f: f not in missed)
    chosen = []
    for f in candidates:
        box = tuple(gt_for_image[f])
        if all(iou(box, c["box"]) <= 0.5 for c in chosen):
            chosen.append({"box": box, "source": f"oracle:{f}"})
    return pad_and_shuffle(chosen, rng, sizes, pad)


def draw_boxes(image, boxes):
    """RGB copy of `image` with each box outlined and numbered 1..n (list order), all drawn identically."""
    im = image.convert("RGB")
    draw = ImageDraw.Draw(im)
    w, h = im.size
    lw = max(2, round(max(w, h) / 300))
    font = ImageFont.load_default(size=max(12, round(max(w, h) / 36)))
    for k, box in enumerate(boxes, 1):
        x1, y1, x2, y2 = to_pixels(box, w, h)
        draw.rectangle([x1, y1, x2 - 1, y2 - 1], outline=BOX_COLOR, width=lw)
        # Number tag just above the top-left corner, or inside that corner if there is no room above.
        left, top, right, bottom = draw.textbbox((0, 0), str(k), font=font)
        tag_w, tag_h = right - left + 2 * lw, bottom - top + 2 * lw
        ty = y1 - tag_h if y1 - tag_h >= 0 else y1
        draw.rectangle([x1, ty, x1 + tag_w, ty + tag_h], fill=BOX_COLOR)
        draw.text((x1 + lw - left, ty + lw - top), str(k), fill=TEXT_COLOR, font=font)
    return im
