"""Segment every sampled image into CXAS's 159 anatomy classes and store the masks.

Runs in its own environment (.venv-cxas: cxas pins numpy 1.26, which the reader environment can't take):
    CXAS_NO_WELCOME=1 .venv-cxas/bin/python -m src.trail_anatomy.segment --sample outputs/hints_mvp/sample.json

CXAS stretches each image to 512 x 512 (no crop or pad), so a click at normalized (x, y) falls on mask pixel
(int(y * 512), int(x * 512)) whatever the image's shape. Masks are saved bit-packed per image to
<out>/<image_id>.npz (key "packed", shape (159, 512 * 512 / 8)); unpack with anatomy.load_masks.
Images and masks stay local (research use agreement).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="outputs/hints_mvp/sample.json")
    ap.add_argument("--image-root", default="~/data/chexpert_plus")
    ap.add_argument("--out", default="outputs/trail_anatomy/masks")
    ap.add_argument("--gpus", default="0")
    args = ap.parse_args()

    from cxas import CXAS  # imported here so --help works without the cxas environment

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.image_root).expanduser()
    sample = json.load(open(args.sample))
    todo = [s for s in sample if not (out / f"{s['image_id']}.npz").exists()]
    print(f"{len(sample)} images, {len(todo)} to segment")
    model = CXAS(model_name="UNet_ResNet50_default", gpus=args.gpus)
    t0 = time.time()
    for k, s in enumerate(todo):
        pred = model.process_file(filename=str(root / s["path"]))["segmentation_preds"][0]
        masks = pred.cpu().numpy().astype(bool)
        assert masks.shape == (159, 512, 512), masks.shape
        tmp = out / f"{s['image_id']}.tmp.npz"
        np.savez_compressed(tmp, packed=np.packbits(masks.reshape(159, -1), axis=1))
        tmp.rename(out / f"{s['image_id']}.npz")
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(todo)} in {time.time() - t0:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
