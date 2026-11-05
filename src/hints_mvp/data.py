"""CheXthought traces, click trails and CheXpert Plus images for hints_mvp."""
import json
import random
from pathlib import Path

import pandas as pd
from PIL import Image

# The 14 CheXpert observations, in CheXpert's order.
CHEXPERT = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
    "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other",
    "Fracture", "Support Devices",
]
FINDINGS = CHEXPERT[1:]  # the 13 findings that are labeled and scored; No Finding is derived from them
# CheXpert's rule: No Finding is present when no pathology (anything but Support Devices) is present or uncertain.
PATHOLOGIES = [f for f in FINDINGS if f != "Support Devices"]

# A trace is "snapped" when at least 80% of its x and y values lie within 0.001 of a multiple of 0.01.
# For uniform random values the share is 20%.
SNAP_GRID, SNAP_TOL, SNAP_SHARE = 0.01, 0.001, 0.8


def parse_points(s):
    """spatial_annotations JSON -> [(x, y), ...] in click order."""
    return [(p["x"], p["y"]) for p in sorted(json.loads(s), key=lambda p: p["n"])]


def is_snapped(points):
    values = [v for p in points for v in p]
    near = sum(abs(v - round(v / SNAP_GRID) * SNAP_GRID) <= SNAP_TOL + 1e-9 for v in values)
    return near >= SNAP_SHARE * len(values)


def load_traces(csv_path):
    """Frontal traces, one row per trace. trace_id is the row number in the CSV."""
    df = pd.read_csv(Path(csv_path).expanduser(), usecols=["image_id", "spatial_annotations", "report"])
    df["trace_id"] = df.index
    df = df[df["image_id"].str.endswith("_frontal")].copy()
    df["points"] = df["spatial_annotations"].map(parse_points)
    return df.drop(columns="spatial_annotations").set_index("trace_id", drop=False).rename_axis(None)


def candidate_order(traces, seed):
    """Frontal images in a fixed random order, one per patient, each with one usable trace.

    Snapped and empty traces are dropped. Each image's trace is drawn from its usable traces with an
    rng seeded by the image id, so it doesn't depend on the order. Returns [(image_id, trace_id)].
    """
    usable = traces[traces["points"].map(lambda p: len(p) > 0 and not is_snapped(p))]
    by_image = usable.groupby("image_id")["trace_id"].apply(lambda s: sorted(int(t) for t in s)).to_dict()
    ids = sorted(by_image)
    random.Random(seed).shuffle(ids)
    seen, out = set(), []
    for i in ids:
        patient = i.split("_")[0]
        if patient not in seen:
            seen.add(patient)
            out.append((i, random.Random(f"{seed}:{i}:trace").choice(by_image[i])))
    return out


def derangement(ids, seed):
    """{image_id: another image_id}, a single random cycle (Sattolo), so no image gets its own."""
    order = list(ids)
    rng = random.Random(f"{seed}:other_trail")
    for k in range(len(order) - 1, 0, -1):
        j = rng.randrange(k)
        order[k], order[j] = order[j], order[k]
    return {ids[k]: order[k] for k in range(len(ids))}


def random_trail(n, seed, image_id):
    rng = random.Random(f"{seed}:{image_id}:random_trail")
    return [(round(rng.random(), 4), round(rng.random(), 4)) for _ in range(n)]


def rel_path(image_id):
    """'patient00002_study1_view1_frontal' -> 'train/patient00002/study1/view1_frontal.png' (CheXpert Plus PNG)."""
    patient, study, view = image_id.split("_", 2)
    return f"train/{patient}/{study}/{view}.png"


def load_image(root, image_id, long_edge=None):
    """Grayscale image, resized so its long edge is `long_edge` px (no resize if None)."""
    im = Image.open(Path(root).expanduser() / rel_path(image_id)).convert("L")
    if long_edge:
        s = long_edge / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    return im
