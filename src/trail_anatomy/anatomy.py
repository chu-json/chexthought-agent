"""Map search-path points onto a fixed partition of the CXAS anatomy, and describe each path.

The partition: every pixel gets exactly one region, the first in REGIONS whose CXAS classes cover it. CXAS
masks overlap (the lungs are projected through the heart and mediastinum), so the order is a choice:
small, specific structures first, then the heart and mediastinum, then lung zones, then the rest.
So a click on the heart counts as heart even where the left lower lung lies behind it. Sides are the
patient's (CXAS "right lung" is the patient's right, on the left of a standard frontal image).
"""
from pathlib import Path

import numpy as np

SIZE = 512  # CXAS mask resolution; images are stretched to this, so normalized coords map directly

# (region, CXAS class ids). Order = priority.
REGIONS = [
    ("trachea", [154, 155]),
    ("hila", [131]),                       # pulmonary arteries: the closest CXAS has to the hila
    ("heart", [121]),
    ("mediastinum", [116, 115, 127, 130]),  # upper mediastinum, rest of cardiomediastinum, aorta
    ("right upper lung", [144, 141]),      # apical + upper zone
    ("right mid lung", [142]),
    ("right lower lung", [143]),
    ("left upper lung", [148, 145]),
    ("left mid lung", [146]),
    ("left lower lung", [147]),
    ("right hemidiaphragm", [107]),
    ("left hemidiaphragm", [106]),
    ("clavicles", [30]),
    ("spine", [0]),
    ("abdomen", [108, 109, 110, 111, 112, 113, 114]),
    ("other", []),                         # none of the above: soft tissue, outside the body, image edges
]
NAMES = [r for r, _ in REGIONS]
OTHER = NAMES.index("other")
LUNG = [i for i, n in enumerate(NAMES) if n.endswith("lung")]


def load_masks(path):
    packed = np.load(path)["packed"]
    return np.unpackbits(packed, axis=1, count=SIZE * SIZE).reshape(-1, SIZE, SIZE).astype(bool)


def partition(masks):
    """(512, 512) int array: the region index of every pixel."""
    lab = np.full((SIZE, SIZE), OTHER, dtype=np.int8)
    free = np.ones((SIZE, SIZE), dtype=bool)
    for k, (_, ids) in enumerate(REGIONS[:-1]):
        m = masks[ids].any(0) & free
        lab[m] = k
        free &= ~m
    return lab


def load_partition(mask_dir, image_id):
    return partition(load_masks(Path(mask_dir) / f"{image_id}.npz"))


def pixel(points):
    """Normalized (x, y) -> (row, col) indices into a 512 x 512 mask, clipped to the image."""
    p = np.clip(np.asarray(points, dtype=float), 0, 1 - 1e-9)
    return (p[:, 1] * SIZE).astype(int), (p[:, 0] * SIZE).astype(int)


def regions_of(points, lab):
    r, c = pixel(points)
    return lab[r, c]


def area_share(lab):
    """Fraction of the image in each region."""
    return np.bincount(lab.ravel(), minlength=len(NAMES)) / lab.size


def describe(points, lab):
    """Per-path numbers used to compare search styles. points: [(x, y), ...] in click order."""
    p = np.asarray(points, dtype=float)
    steps = np.linalg.norm(np.diff(p, axis=0), axis=1) if len(p) > 1 else np.zeros(0)
    reg = regions_of(p, lab)
    counts = np.bincount(reg, minlength=len(NAMES))
    cells = {(int(x * 10), int(y * 10)) for x, y in np.clip(p, 0, 0.999)}
    return {
        "n_points": len(p),
        "path_length": float(steps.sum()),
        "median_step": float(np.median(steps)) if len(steps) else 0.0,
        "grid_coverage": len(cells) / 100,  # share of a 10 x 10 grid with at least one point
        "n_regions": int((counts[:OTHER] > 0).sum()),  # anatomical regions with at least one point
        "share_other": float(counts[OTHER] / len(p)),
        "counts": {n: int(c) for n, c in zip(NAMES, counts)},
    }
