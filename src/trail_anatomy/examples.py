"""Local-only example page with real images: raw dots, CXAS regions, focus outline, and model reports per condition.

Usage (main env): python -m src.trail_anatomy.examples [--n 6]
Writes outputs/trail_anatomy/examples/examples.html (gitignored). It contains real images, paths and reports
under the research use agreement: view it on this machine only; never commit, upload or publish it.
"""
import argparse
import base64
import glob
import html
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..hints_mvp import data
from ..hints_mvp.data import FINDINGS
from . import anatomy as A
from . import focus as F
from .analyze_paths import style

OUT = Path("outputs/trail_anatomy")
PALETTE = np.array([(232, 176, 74), (217, 138, 58), (217, 102, 122), (201, 162, 106), (90, 155, 224), (63, 127, 200),
                    (45, 98, 168), (120, 180, 240), (80, 140, 215), (55, 110, 185), (63, 174, 122), (90, 200, 150),
                    (185, 192, 201), (142, 151, 163), (154, 123, 92), (0, 0, 0)], dtype=np.uint8)
SHOW = [("no_hint", "No hint"), ("coords_own", "Raw dots"), ("focus_text_own", "Focus areas (this X-ray)"),
        ("focus_mask_own", "Outlined areas"), ("report_regions", "Cheat: report areas")]


def b64(im, size=420):
    im = im.copy()
    im.thumbnail((size, size))
    b = io.BytesIO()
    im.save(b, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def dots(im, points, color=(255, 211, 77)):
    im = im.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    w, h = im.size
    r = max(3, w // 160)
    xy = [(x * w, y * h) for x, y in points]
    if len(xy) > 1:
        d.line(xy, fill=(255, 211, 77), width=max(1, r // 3))
    for x, y in xy:
        d.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=(0, 0, 0))
    return im


def regions_overlay(im, lab):
    rgb = np.array(im.convert("RGB").resize((512, 512))).astype(float)
    col = PALETTE[lab].astype(float)
    m = (lab != A.OTHER)[..., None]
    out = np.where(m, rgb * 0.45 + col * 0.55, rgb)
    return Image.fromarray(out.astype(np.uint8)).resize(im.size)


def records(model):
    recs, labs = {}, {}
    lp = OUT / f"labels_{model}.json"
    labels = json.load(open(lp)) if lp.exists() else {}
    for f in glob.glob(f"{OUT}/cache/{model}/*/*.json"):
        r = json.load(open(f))
        recs[(r["image_id"], r["condition"])] = r
        labs[(r["image_id"], r["condition"])] = labels.get(f)
    return recs, labs


def calls(lab):
    return [f for f in FINDINGS if lab and lab[f] != "absent"] if lab else None


CONDS = [("no_hint", "No hint"), ("neutral_text", "Meaningless sentence"), ("coords_own", "Raw dots as numbers"),
         ("focus_text_own", "Focus areas, this X-ray"), ("focus_text_other", "Focus areas, another X-ray"),
         ("focus_text_random", "Random areas"), ("focus_text_own_assist", "Focus + 'assist you'"),
         ("focus_mask_own", "Outlined, this X-ray"), ("focus_mask_other", "Outlined, another X-ray"),
         ("report_regions", "Cheat: report areas")]


def hint_of(cond, h):
    from ..hints_mvp.prompts import trail_text
    from .run_vllm import NEUTRAL
    return {"no_hint": "", "neutral_text": NEUTRAL, "coords_own": trail_text(h["coords_own"]),
            "focus_text_own": F.focus_text(h["focus_own"]), "focus_text_other": F.focus_text(h["focus_other"]),
            "focus_text_random": F.focus_text(h["focus_random"]), "focus_text_own_assist": F.focus_text(h["focus_own"], assist=True),
            "focus_mask_own": F.MASK_INTRO if h["focus_own"] else F.EVEN_SENTENCE,
            "focus_mask_other": F.MASK_INTRO if h["focus_other"] else F.EVEN_SENTENCE,
            "report_regions": F.focus_text(h["report_regions"])}[cond]


def regions_of_cond(cond, h):
    return {"focus_text_own": h["focus_own"], "focus_text_own_assist": h["focus_own"], "focus_mask_own": h["focus_own"],
            "focus_text_other": h["focus_other"], "focus_mask_other": h["focus_other"],
            "focus_text_random": h["focus_random"], "report_regions": h["report_regions"]}.get(cond, [])


def build_example(s, h, b, params, models):
    i = s["image_id"]
    im = data.load_image("~/data/chexpert_plus", i, 1024).convert("RGB")
    lab = A.load_partition(OUT / "masks", i)
    pts = s["trails"]["own"]["points"]
    c = F.counts(pts, lab)
    e = F.enrichment(c.astype(float), b, params["alpha"])
    ref = [f for f in FINDINGS if s["labels"][f] != "absent"]
    imgs = {"plain": b64(im, 560), "dots": b64(dots(im, pts), 560), "seg": b64(dots(regions_overlay(im, lab), pts), 560)}
    for key in ("focus_own", "focus_other", "focus_random", "report_regions"):
        imgs[key] = b64(F.outline_image(im, lab, h[key]) if h[key] else im, 560)
    conds = []
    for cond, name in CONDS:
        reps = []
        for m, (recs, labs) in models.items():
            r = recs.get((i, cond))
            if r:
                got = calls(labs.get((i, cond)))
                reps.append({"model": m, "report": r["report"], "calls": got, "trunc": r.get("finish_reason") == "length"})
        conds.append({"id": cond, "name": name, "hint": hint_of(cond, h), "regions": regions_of_cond(cond, h), "reports": reps})
    return {"id": i, "group": s["group"], "style": style(A.describe(pts, lab)), "n": len(pts), "ref": ref,
            "report": s.get("report", ""), "imgs": imgs, "conds": conds,
            "table": [{"r": A.NAMES[r], "c": int(c[r]), "share": float(c[r] / max(c.sum(), 1)), "b": float(b[r]), "e": float(e[r])}
                      for r in np.argsort(-e) if c[r] > 0], "focus": h["focus_own"]}


TEMPLATE = Path(__file__).with_name("examples_template.html")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    sample = json.load(open(OUT / "test/sample.json"))
    hints = json.load(open(OUT / "test/hints.json"))
    fp = json.load(open(OUT / "focus_params.json"))
    b = np.array([fp["baseline"][n] for n in A.NAMES])
    models = {m: records(m) for m in ("chexone", "medgemma", "maira") if (OUT / "cache" / m).exists()}
    by = {}
    for s in sample:  # per style: images with a finding, with and without a focus region; plus a normal one
        lab = A.load_partition(OUT / "masks", s["image_id"])
        st = style(A.describe(s["trails"]["own"]["points"], lab))
        key = (st, s["group"], bool(hints[s["image_id"]]["focus_own"]))
        if len(by.setdefault(key, [])) < (2 if s["group"] == "finding" else 1):
            by[key].append(s)
    order = [k for st in ("search", "tracing", "sparse") for k in [(st, "finding", True), (st, "finding", False), (st, "normal", True), (st, "normal", False)]]
    picks = [s for k in order for s in by.get(k, [])][: a.n]
    ex = [build_example(s, hints[s["image_id"]], b, fp["params"], models) for s in picks]
    out = OUT / "examples/examples.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(TEMPLATE.read_text().replace("__DATA__", json.dumps(ex).replace("</", "<\\/")))
    print("wrote", out, len(ex), "examples", round(out.stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
