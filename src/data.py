"""Load CheXlocalize frontal images, radiologist labels and ground-truth masks."""
import json
import random
from pathlib import Path

import pandas as pd
from PIL import Image
from pycocotools import mask as mask_utils

from .boxes import largest_component_box

FINDINGS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
    "Lung Lesion", "Lung Opacity", "Pleural Effusion", "Pneumothorax", "Support Devices",
]
# The segmentation JSONs call Lung Opacity "Airspace Opacity".
SEG_NAME = {f: f for f in FINDINGS} | {"Lung Opacity": "Airspace Opacity"}


def image_id(rel_path):
    """'test/patient64741/study1/view1_frontal.jpg' -> 'patient64741_study1_view1_frontal'.

    Uses the last three path parts because val_labels.csv paths start with 'CheXpert-v1.0/valid/'.
    """
    return "_".join(Path(rel_path).with_suffix("").parts[-3:])


def image_path(root, split, iid):
    patient, study, view = iid.split("_", 2)
    return Path(root).expanduser() / "CheXpert" / split / patient / study / f"{view}.jpg"


def load_labels(root, split):
    """Radiologist labels for frontal images: {image_id: {finding: 0 or 1}}."""
    df = pd.read_csv(Path(root).expanduser() / "CheXpert" / f"{split}_labels.csv")
    df["id"] = df["Path"].map(image_id)
    df = df[df["id"].str.endswith("frontal")].set_index("id")
    return df[FINDINGS].astype(int).to_dict("index")


def load_masks(root, split):
    """Ground-truth RLE masks for frontal images: {image_id: {finding: rle}}.

    The JSON only lists images with at least one positive finding; missing images have no masks.
    """
    path = Path(root).expanduser() / "CheXlocalize" / f"gt_segmentations_{split}.json"
    seg = json.load(open(path))
    return {
        iid: {f: v[SEG_NAME[f]] for f in FINDINGS if SEG_NAME[f] in v}
        for iid, v in seg.items() if iid.endswith("frontal")
    }


def decode(rle):
    return mask_utils.decode(rle).astype(bool)


def gt_boxes(root, split, cache_dir):
    """Normalized box around the largest component of every non-empty mask: {image_id: {finding: box}}.

    Decoding every mask is slow, so the result is cached as JSON.
    """
    path = Path(cache_dir) / f"gt_boxes_{split}.json"
    if path.exists():
        return json.load(open(path))
    out = {}
    for iid, by_finding in load_masks(root, split).items():
        out[iid] = {}
        for f, rle in by_finding.items():
            box = largest_component_box(decode(rle))
            if box is not None:
                out[iid][f] = box
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(path, "w"))
    return out


def sample_images(labels, n, seed, min_pos=10, exclude=()):
    """Stratified sample of image ids.

    Rarest finding first, take up to `quota` positives for each finding (fewer if not available),
    then fill the remaining slots at random. quota = min(min_pos, n // 10), at least 1, so small
    samples are not filled up by the rarest findings.
    """
    rng = random.Random(seed)
    pool = sorted(i for i in labels if i not in set(exclude))
    rng.shuffle(pool)
    quota = max(1, min(min_pos, n // len(FINDINGS)))
    chosen = []
    for f in sorted(FINDINGS, key=lambda f: sum(labels[i][f] for i in pool)):
        have = sum(labels[i][f] for i in chosen)
        for i in pool:
            if have >= quota or len(chosen) >= n:
                break
            if labels[i][f] and i not in chosen:
                chosen.append(i)
                have += 1
    for i in pool:
        if len(chosen) >= n:
            break
        if i not in chosen:
            chosen.append(i)
    return sorted(chosen)


def load_image(root, split, iid, long_edge=None):
    """Grayscale image, resized so its long edge is `long_edge` px (no resize if None)."""
    im = Image.open(image_path(root, split, iid)).convert("L")
    if long_edge:
        s = long_edge / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    return im
