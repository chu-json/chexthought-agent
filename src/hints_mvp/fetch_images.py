"""Download only the sampled CheXpert Plus PNGs from Redivis.

Usage (REDIVIS_API_TOKEN is read from the environment or .env):
  python -m src.hints_mvp.fetch_images --config configs/hints_mvp.yaml --table <reference> --inspect
  python -m src.hints_mvp.fetch_images --config configs/hints_mvp.yaml --table <reference>
<reference> is the PNG_train table: "aimi.chexpert_plus:5yyj:v1_0.png_train:s6cj" (CheXpert Plus v1.0).
--inspect prints the table's variables and a few rows. Without it, a query keeps only the rows for the sampled
paths (output_dir/image_paths.txt), and each file is saved to
image_root/train/patientX/studyN/viewK_frontal.png. Files already there are skipped.
"""
import argparse
from pathlib import Path

from dotenv import load_dotenv

from .config import load_config


def inspect(table):
    print([v.name for v in table.list_variables()])
    print(table.to_pandas_dataframe(max_results=3).to_string())


def fetch(cfg, reference):
    import redivis

    root = Path(cfg["image_root"]).expanduser()
    paths = (Path(cfg["output_dir"]) / "image_paths.txt").read_text().split()
    todo = [p for p in paths if not (root / p).exists()]
    print(f"{len(paths)} sampled images, {len(paths) - len(todo)} already in {root}")
    if not todo:
        return
    # The table's file_name is "patientX/studyN/viewK_frontal.png", without the "train/" prefix.
    tails = {"/".join(Path(p).parts[-3:]): p for p in todo}
    names = ", ".join(f"'{t}'" for t in tails)
    q = redivis.query(f"SELECT * FROM `{reference}` WHERE file_name IN ({names})")
    found = q.to_directory().list(mode="files", recursive=True)
    by_tail = {}
    for f in found:
        tail = "/".join(f.path.parts[-3:])
        if tail in tails:
            by_tail.setdefault(tail, []).append(f)
    ambiguous = [t for t, fs in by_tail.items() if len(fs) > 1]
    if ambiguous:
        raise RuntimeError(f"{len(ambiguous)} paths match more than one file, e.g. "
                           f"{[str(f.path) for f in by_tail[ambiguous[0]]]}")
    for tail, (f,) in by_tail.items():
        target = root / tails[tail]
        target.parent.mkdir(parents=True, exist_ok=True)
        f.download(str(target))
    missing = sorted(set(tails) - set(by_tail))
    print(f"downloaded {len(by_tail)}; not found in the table: {len(missing)}{' e.g. ' + missing[0] if missing else ''}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--table", required=True, help="Redivis reference of the PNG_train table")
    ap.add_argument("--inspect", action="store_true", help="print the table's variables and a few rows")
    args = ap.parse_args()
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    if args.inspect:
        import redivis
        inspect(redivis.table(args.table))
    else:
        fetch(load_config(args.config), args.table)
