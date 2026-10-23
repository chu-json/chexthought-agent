"""Step 1 data checks for CheXlocalize. Prints a report and saves 4 overlay images.

Usage: python -m src.verify_data [--root ~/data/chexlocalize] [--out outputs/verify]
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pycocotools import mask as mask_utils

FINDINGS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
    "Lung Lesion", "Lung Opacity", "Pleural Effusion", "Pneumothorax", "Support Devices",
]
# The segmentation JSONs call Lung Opacity "Airspace Opacity".
SEG_NAME = {f: f for f in FINDINGS} | {"Lung Opacity": "Airspace Opacity"}
EXPECTED = {"test": (668, 500), "val": (234, 200)}


def image_id(rel_path):
    """'test/patient64741/study1/view1_frontal.jpg' -> 'patient64741_study1_view1_frontal'."""
    parts = Path(rel_path).with_suffix("").parts
    return "_".join(parts[-3:])


def check_split(root, split):
    print(f"\n=== {split} ===")
    img_dir = root / "CheXpert" / split
    paths = sorted(img_dir.rglob("*.jpg"))
    frontal = [p for p in paths if "frontal" in p.name]
    patients = {p.parts[-3] for p in paths}
    exp_imgs, exp_pts = EXPECTED[split]
    print(f"images: {len(paths)} (expected {exp_imgs}), patients: {len(patients)} (expected {exp_pts}), "
          f"frontal: {len(frontal)}, lateral: {len(paths) - len(frontal)}")

    # Resolution and mode of frontal images (header read only).
    sizes, modes = {}, Counter()
    for p in frontal:
        with Image.open(p) as im:
            sizes[image_id(p)] = (im.height, im.width)
            modes[im.mode] += 1
    hs = np.array([s[0] for s in sizes.values()])
    ws = np.array([s[1] for s in sizes.values()])
    print(f"frontal height px: min {hs.min()}, median {int(np.median(hs))}, max {hs.max()}")
    print(f"frontal width  px: min {ws.min()}, median {int(np.median(ws))}, max {ws.max()}")
    print(f"image modes: {dict(modes)}; frontal images with long edge < 1500 px: "
          f"{int((np.maximum(hs, ws) < 1500).sum())}")

    # Labels CSV. Val paths look like 'CheXpert-v1.0/valid/patientX/...'; ids use the last 3 parts.
    labels = pd.read_csv(root / "CheXpert" / f"{split}_labels.csv")
    labels["id"] = labels["Path"].map(image_id)
    print(f"labels rows: {len(labels)}, duplicate ids: {labels['id'].duplicated().sum()}")
    missing_cols = [f for f in FINDINGS if f not in labels.columns]
    print(f"finding columns missing from CSV: {missing_cols or 'none'}")
    vals = sorted(pd.unique(labels[FINDINGS].values.ravel()).tolist())
    print(f"label values present: {vals}")

    # Segmentation JSON.
    seg = json.load(open(root / "CheXlocalize" / f"gt_segmentations_{split}.json"))
    seg_names = Counter(n for v in seg.values() for n in v)
    print(f"seg entries: {len(seg)}; finding names in seg JSON: {dict(seg_names)}")

    frontal_ids = set(sizes)
    label_ids = set(labels["id"])
    all_img_ids = {image_id(p) for p in paths}
    report = {
        "frontal images missing from seg JSON": sorted(frontal_ids - set(seg)),
        "seg JSON ids with no image": sorted(set(seg) - all_img_ids),
        "seg JSON ids that are not frontal": sorted(set(seg) - frontal_ids),
        "frontal images missing from labels CSV": sorted(frontal_ids - label_ids),
        "labels CSV ids with no image": sorted(label_ids - all_img_ids),
    }
    for k, v in report.items():
        print(f"{k}: {len(v)}" + (f" -> {v[:10]}{' ...' if len(v) > 10 else ''}" if v else ""))

    # Frontal images not in seg JSON: what are their labels?
    lab = labels.set_index("id")
    no_seg = [i for i in report["frontal images missing from seg JSON"] if i in lab.index]
    if no_seg:
        pos = lab.loc[no_seg, FINDINGS].sum(axis=1)
        print(f"  of those, positives per image (10 findings): {Counter(pos.astype(int).tolist())}")

    # Mask dimensions vs image dimensions, and mask non-empty vs label positive.
    dim_mismatch, agree = [], Counter()
    nonempty = {}
    for iid, by_finding in seg.items():
        if iid not in sizes:
            continue
        for f in FINDINGS:
            rle = by_finding.get(SEG_NAME[f])
            if rle is None:
                agree[(f, "no mask key")] += 1
                continue
            if tuple(rle["size"]) != sizes[iid]:
                dim_mismatch.append((iid, f, rle["size"], sizes[iid]))
            m = mask_utils.decode(rle)
            if m.shape != sizes[iid]:
                dim_mismatch.append((iid, f, m.shape, sizes[iid]))
            has = bool(m.any())
            nonempty[(iid, f)] = has
            label = lab.loc[iid, f] if iid in lab.index else np.nan
            agree[(f, f"label={label:.0f} mask={'yes' if has else 'no'}")] += 1
    print(f"mask/image dimension mismatches: {len(dim_mismatch)}"
          + (f" -> {dim_mismatch[:5]}" if dim_mismatch else ""))
    print("label vs mask agreement (images in seg JSON):")
    for f in FINDINGS:
        row = {k[1]: v for k, v in agree.items() if k[0] == f}
        print(f"  {f:28s} {row}")

    # Are Cardiomegaly and Enlarged Cardiomediastinum masks identical?
    same = sum(
        1 for iid, v in seg.items()
        if v.get("Cardiomegaly", {}).get("counts") == v.get("Enlarged Cardiomediastinum", {}).get("counts")
        and nonempty.get((iid, "Cardiomegaly"))
    )
    print(f"images where non-empty Cardiomegaly mask == Enlarged Cardiomediastinum mask: {same}")
    return seg, sizes, nonempty, labels


def save_overlays(root, split, seg, nonempty, ids, out_dir):
    colors = plt.cm.tab10.colors
    for iid in ids:
        patient, study, view = iid.split("_", 2)
        img = np.array(Image.open(root / "CheXpert" / split / patient / study / f"{view}.jpg").convert("L"))
        fig, ax = plt.subplots(figsize=(8, 8 * img.shape[0] / img.shape[1]))
        ax.imshow(img, cmap="gray")
        handles = []
        for c, f in zip(colors, FINDINGS):
            if not nonempty.get((iid, f)):
                continue
            m = mask_utils.decode(seg[iid][SEG_NAME[f]])
            ax.contour(m, levels=[0.5], colors=[c], linewidths=1.5)
            handles.append(plt.Line2D([], [], color=c, label=f))
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
        ax.set_title(f"{split}: {iid}  ({img.shape[1]}x{img.shape[0]})", fontsize=9)
        ax.axis("off")
        path = out_dir / f"overlay_{split}_{iid}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/data/chexlocalize")
    ap.add_argument("--out", default="outputs/verify")
    args = ap.parse_args()
    root = Path(args.root).expanduser()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(0)
    for split in ["val", "test"]:
        seg, sizes, nonempty, _ = check_split(root, split)
        # 2 overlays per split, from images with masks for at least 3 findings.
        rich = sorted(i for i in seg if i in sizes and sum(nonempty.get((i, f), False) for f in FINDINGS) >= 3)
        save_overlays(root, split, seg, nonempty, rng.sample(rich, 2), out_dir)


if __name__ == "__main__":
    main()
