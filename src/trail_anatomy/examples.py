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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    sample = json.load(open(OUT / "test/sample.json"))
    hints = json.load(open(OUT / "test/hints.json"))
    fp = json.load(open(OUT / "focus_params.json"))
    b = np.array([fp["baseline"][n] for n in A.NAMES])
    models = {m: records(m) for m in ("chexone", "medgemma", "maira") if (OUT / "cache" / m).exists()}
    # One example per style, with and without a focus region, preferring images with a finding.
    by = {}
    for s in sample:
        lab = A.load_partition(OUT / "masks", s["image_id"])
        pts = s["trails"]["own"]["points"]
        st = style(A.describe(pts, lab))
        key = (st, bool(hints[s["image_id"]]["focus_own"]))
        if s["group"] == "finding" and len(by.setdefault(key, [])) < 2:
            by[key].append(s)
    picks = [s for k in sorted(by) for s in by[k]][: a.n]
    parts = []
    for s in picks:
        i = s["image_id"]
        im = data.load_image("~/data/chexpert_plus", i, 1024).convert("RGB")
        lab = A.load_partition(OUT / "masks", i)
        pts = s["trails"]["own"]["points"]
        foc = hints[i]["focus_own"]
        c = F.counts(pts, lab)
        e = F.enrichment(c.astype(float), b, fp["params"]["alpha"])
        st = style(A.describe(pts, lab))
        rows = "".join(f"<tr{' class=f' if A.NAMES[r] in foc else ''}><td>{A.NAMES[r]}</td><td>{c[r]}</td>"
                       f"<td>{c[r] / max(c.sum(), 1):.0%}</td><td>{b[r]:.1%}</td><td>{e[r]:.2f}×</td></tr>"
                       for r in np.argsort(-e) if c[r] > 0)
        ref = [f for f in FINDINGS if s["labels"][f] != "absent"]
        reps = ""
        for m, (recs, labs) in models.items():
            for cond, name in SHOW:
                r = recs.get((i, cond))
                if not r:
                    continue
                got = calls(labs.get((i, cond)))
                tag = "" if got is None else " · calls: " + (", ".join(
                    f"<b class={'hit' if g in ref else 'miss'}>{g}</b>" for g in got) or "none")
                reps += f"<details><summary>{m} · {name}{tag}</summary><pre>{html.escape(r['report'])}</pre></details>"
        parts.append(f"""<section><h2>{i} · {st} path · {len(pts)} dots</h2>
<p><b>Real report's findings:</b> {', '.join(ref) or 'none'}<br><b>Focus areas:</b> {', '.join(foc) or 'none (spread evenly)'}
· <b>another X-ray's:</b> {', '.join(hints[i]['focus_other']) or 'none'} · <b>report areas:</b> {', '.join(hints[i]['report_regions']) or 'none'}</p>
<div class=row><figure><img src="{b64(dots(im, pts))}"><figcaption>Raw dots</figcaption></figure>
<figure><img src="{b64(regions_overlay(im, lab))}"><figcaption>CXAS regions (16 areas)</figcaption></figure>
<figure><img src="{b64(dots(F.outline_image(im, lab, foc) if foc else im, pts))}"><figcaption>Focus areas outlined</figcaption></figure></div>
<table><tr><th>Area</th><th>Dots</th><th>This path</th><th>Average</th><th>Adjusted vs average</th></tr>{rows}</table>
<h3>Model reports (green = real finding, red = false finding)</h3>{reps}
<details><summary>Real report</summary><pre>{html.escape(s.get('report', ''))}</pre></details></section>""")
    page = f"""<!doctype html><meta charset=utf-8><title>Real examples (local only)</title><style>
body{{font:15px/1.5 system-ui;max-width:1100px;margin:auto;padding:16px;background:#f6f7f9;color:#15191f}}
.warn{{background:#fde8e4;padding:10px 14px;border-radius:8px}} section{{background:#fff;border-radius:10px;padding:16px;margin:20px 0}}
.row{{display:flex;flex-wrap:wrap;gap:10px}} figure{{margin:0;flex:1;min-width:250px}} img{{width:100%;border-radius:6px}}
figcaption{{font-size:13px;color:#555}} table{{border-collapse:collapse;font-size:13px;margin:10px 0}} td,th{{padding:3px 8px;border-bottom:1px solid #ddd;text-align:left}}
tr.f td{{background:#fde8e4}} pre{{white-space:pre-wrap;background:#f3f4f6;padding:8px;border-radius:6px}} .hit{{color:#1e7a4a}} .miss{{color:#b23a2e}}
summary{{cursor:pointer;padding:3px 0}}</style>
<h1>Real examples (local only)</h1><p class=warn>Contains real CheXpert Plus images, CheXthought paths and reports under the
research use agreement. Keep on this machine; never commit, upload or publish.</p>
<p>Region colours: amber = trachea/hila/mediastinum, pink = heart, blues = lung thirds, green = diaphragm, grey = clavicles/spine, brown = abdomen.</p>
{''.join(parts)}"""
    out = OUT / "examples/examples.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(page)
    print("wrote", out, len(picks), "examples")


if __name__ == "__main__":
    main()
