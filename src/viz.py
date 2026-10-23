"""Save sample pass-2 images: the exact boxed image the reader saw, with ground-truth outlines overlaid.

Usage: python -m src.viz --config configs/mock_val.yaml --conditions random_boxes helper_boxes oracle_boxes --n 2
"""
import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from . import boxes as bx
from . import data
from .run import load_config
from .score import LABEL

# One color per finding; no red (the box color) and no gray (invisible on an X-ray).
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#ffd700", "#9467bd", "#8c564b", "#e377c2", "#7fff00", "#bcbd22",
          "#17becf"]


def box_note(k, b, r):
    """Caption for box k, for checking by eye; never shown to the reader."""
    note = f"{k}: {b['source']} {[round(v, 3) for v in b['box']]}"
    if b["source"] == "helper":
        match = next(h for h in r["helper"]["boxes"] if list(h["box"]) == list(b["box"]))
        note += f"  raw={match['raw_box']}  suspected={match['suspected_finding']}"
    return note


def save_sample(cfg, r, cond, rles, path):
    im = data.load_image(cfg["data_root"], cfg["split"], r["image_id"], cfg["image_long_edge"])
    boxed = bx.draw_boxes(im, [b["box"] for b in r["boxes"][cond]])
    fig, ax = plt.subplots(figsize=(9, 9 * im.height / im.width))
    ax.imshow(boxed)
    handles = []
    for color, f in zip(COLORS, data.FINDINGS):
        rle = rles.get(r["image_id"], {}).get(f)
        mask = data.decode(rle) if rle else None
        if mask is None or not mask.any():
            continue
        small = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize(im.size, Image.NEAREST)) > 0
        ax.contour(small, levels=[0.5], colors=[color], linewidths=1.2)
        handles.append(plt.Line2D([], [], color=color, label=f))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, title="ground truth")
    notes = "\n".join(box_note(k, b, r) for k, b in enumerate(r["boxes"][cond], 1))
    ax.set_title(f"{LABEL[cond]} | {r['image_id']}\n{notes}", fontsize=8, loc="left")
    ax.axis("off")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--conditions", nargs="+", default=["random_boxes", "helper_boxes", "oracle_boxes"])
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    results = [json.loads(line) for line in open(Path(cfg["output_dir"]) / "predictions.jsonl")]
    rles = data.load_masks(cfg["data_root"], cfg["split"])
    out = Path(cfg["output_dir"]) / "samples"
    out.mkdir(exist_ok=True)
    rng = random.Random(args.seed)
    for cond in args.conditions:
        # Images that reached pass 2 and have at least one outline to compare against.
        pool = [r for r in results if cond in r["boxes"] and r["image_id"] in rles]
        for r in rng.sample(pool, min(args.n, len(pool))):
            path = out / f"{cond}_{r['image_id']}.png"
            save_sample(cfg, r, cond, rles, path)
            print(f"saved {path}")


if __name__ == "__main__":
    main()
