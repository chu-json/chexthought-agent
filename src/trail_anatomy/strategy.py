"""Search-strategy labels (Broad / Central / Narrow) and the structured-text trajectory hint, after CheXthought §2.4.7 / §4.3.2.

The paper clusters search trajectories with k-means into Broad (35.5%), Central (32.3%) and Narrow (32.2%) and gives
base models the trajectory "encoded as structured text", but publishes neither the features nor the wording. This is
our reconstruction: k-means (k=3, standardized) on three point-only features, fitted on the test sample's own paths, and
clusters named by their centroids: the most central one is Central, of the other two the wider one is Broad.
  coverage  share of an 8x8 grid's cells visited
  central   share of points in the paramediastinal band, |x - 0.5| < 0.15
  spread    mean distance of points from their centroid
Usage (main env): python -m src.trail_anatomy.strategy  -> outputs/trail_anatomy/strategy.json
"""
import json
from pathlib import Path

import numpy as np

OUT = Path("outputs/trail_anatomy")
DESCRIBE = {"Broad": "a broad sweep covering most of the chest",
            "Central": "a search concentrated on the central chest (mediastinum, hila and adjacent lung)",
            "Narrow": "a focused search concentrated on a few areas"}


def features(points):
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    cells = {(min(int(x * 8), 7), min(int(y * 8), 7)) for x, y in p}
    return np.array([len(cells) / 64, float(np.mean(np.abs(p[:, 0] - 0.5) < 0.15)),
                     float(np.mean(np.linalg.norm(p - p.mean(0), axis=1)))])


def kmeans(X, k=3, seed=0, iters=100):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(10):
        C = X[rng.choice(len(X), k, replace=False)]
        for _ in range(iters):
            lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
            C2 = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
            if np.allclose(C, C2):
                break
            C = C2
        sse = ((X - C[lab]) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, C, lab)
    return best[1], best[2]


def fit(sample):
    X = np.array([features(s["trails"]["own"]["points"]) for s in sample])
    mu, sd = X.mean(0), X.std(0)
    C, lab = kmeans((X - mu) / sd)
    Craw = C * sd + mu
    central = int(np.argmax(Craw[:, 1]))
    rest = [j for j in range(3) if j != central]
    broad = max(rest, key=lambda j: Craw[j, 0] + Craw[j, 2] * 2)
    narrow = [j for j in rest if j != broad][0]
    names = {central: "Central", broad: "Broad", narrow: "Narrow"}
    return {"mu": mu.tolist(), "sd": sd.tolist(), "centroids": C.tolist(), "names": [names[j] for j in range(3)],
            "centroids_raw": Craw.tolist(), "shares": {names[j]: float(np.mean(lab == j)) for j in range(3)}}


def label(points, model):
    z = (features(points) - np.array(model["mu"])) / np.array(model["sd"])
    return model["names"][int(np.argmin(((np.array(model["centroids"]) - z) ** 2).sum(1)))]


def trajectory_text(points, strategy):
    pts = ", ".join(f"({x:.2f}, {y:.2f})" for x, y in points)
    return ("Expert radiologist search trajectory for this image (structured):\n"
            f"- Search strategy: {strategy} ({DESCRIBE[strategy]})\n"
            f"- Number of fixation points: {len(points)}\n"
            f"- Fixation points in viewing order (x, y as fractions of image width and height): {pts}\n"
            "Use this expert search trajectory to guide your interpretation.")


if __name__ == "__main__":
    sample = json.load(open(OUT / "test/sample.json"))
    m = fit(sample)
    json.dump(m, open(OUT / "strategy.json", "w"), indent=1)
    print("cluster shares:", m["shares"])
    for n, c in zip(m["names"], m["centroids_raw"]):
        print(f"  {n:8s} coverage {c[0]:.2f}  central {c[1]:.2f}  spread {c[2]:.3f}")
