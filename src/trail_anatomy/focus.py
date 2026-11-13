"""Standardized search paths: focus regions, and the hint text and outlines built from them.

Focus regions of a path (counts c_r over the 16 regions, n points in total), against a baseline b_r (the
average click share of region r over a pool of paths: how radiologists usually spread their clicks):
    q_r = (c_r + alpha * b_r) / (n + alpha)     shrunk share: a sparse path is pulled toward the baseline
    e_r = q_r / b_r                             enrichment: > 1 means more attention than usual
    focus = up to k regions with e_r >= tau and c_r >= min_count, highest e_r first ("other" never counts)
A path that spreads its clicks like everyone else has no focus region. alpha, tau, k, min_count are tuned on the
dev set only (see analyze_paths).
"""
import numpy as np

from . import anatomy as A

DEFAULT = {"alpha": 10.0, "tau": 1.5, "k": 3, "min_count": 3}


def counts(points, lab):
    return np.bincount(A.regions_of(points, lab), minlength=len(A.NAMES))


def baseline(count_rows):
    """Average click share per region over a pool of paths."""
    c = np.asarray(count_rows, dtype=float)
    return (c / c.sum(1, keepdims=True)).mean(0)


def enrichment(c, b, alpha):
    q = (c + alpha * b) / (c.sum() + alpha)
    return q / np.maximum(b, 1e-9)


def focus_regions(c, b, alpha=10.0, tau=1.5, k=3, min_count=3):
    e = enrichment(np.asarray(c, dtype=float), b, alpha)
    out = [r for r in np.argsort(-e) if r != A.OTHER and e[r] >= tau and c[r] >= min_count]
    return [A.NAMES[r] for r in out[:k]]


def random_regions(n, freq, rng):
    """n distinct regions drawn with probability proportional to how often each is a focus region (dev set)."""
    names = [r for r in A.NAMES if r != "other" and freq.get(r, 0) > 0]
    p = np.array([freq[r] for r in names], dtype=float)
    pick = rng.choice(len(names), size=min(n, len(names)), replace=False, p=p / p.sum())
    return [names[i] for i in pick]


def phrase(regions):
    """['right lower lung', 'heart'] -> 'the right lower lung and the heart'."""
    parts = [("the " + r) for r in regions]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


FOCUS_SENTENCE = "A radiologist who read this X-ray spent more attention than usual on {regions}."
EVEN_SENTENCE = "A radiologist who read this X-ray spread their attention evenly across the chest."
ASSIST_PREFIX = "Use the following to assist you in your findings. "
MASK_INTRO = ("The first image is a chest X-ray. The second image is the same X-ray with outlines around the regions "
              "where a radiologist who read it spent more attention than usual.")


def focus_text(regions, assist=False):
    s = FOCUS_SENTENCE.format(regions=phrase(regions)) if regions else EVEN_SENTENCE
    return (ASSIST_PREFIX + s) if assist else s


def outline_image(image, lab, regions, width=3):
    """RGB copy of `image` with each region's boundary drawn in red (numbered regions are not labeled)."""
    from PIL import Image
    from scipy import ndimage
    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]
    for name in regions:
        m = lab == A.NAMES.index(name)
        edge = m & ~ndimage.binary_erosion(m, iterations=max(1, width * 512 // max(h, w) or 1))
        big = np.array(Image.fromarray(edge.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 0
        big = ndimage.binary_dilation(big, iterations=width)
        rgb[big] = (255, 0, 0)
    return Image.fromarray(rgb)
