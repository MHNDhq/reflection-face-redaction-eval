"""Download the selected Open Images files from the public CVDF S3 mirror, resize to max side 1024."""
import io
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "images"
URL = "https://open-images-dataset.s3.amazonaws.com/{split}/{id}.jpg"


def fetch(row):
    dst = OUT / row.split / f"{row.ImageID}.jpg"
    if dst.exists():
        return "skip"
    for _ in range(3):
        try:
            r = requests.get(URL.format(split=row.split, id=row.ImageID), timeout=30)
            if r.status_code != 200:
                return f"http{r.status_code}"
            im = Image.open(io.BytesIO(r.content)).convert("RGB")
            im.thumbnail((1024, 1024))
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, quality=92)
            return "ok"
        except Exception as e:  # retry, then report
            err = repr(e)
    return "err " + err


def main():
    src = pd.read_csv(ROOT / "data" / "sources.csv")
    with ThreadPoolExecutor(24) as ex:
        res = list(ex.map(fetch, src.itertuples()))
    print(pd.Series([r.split()[0] for r in res]).value_counts().to_dict())
    bad = [(r, s) for r, s in zip(src.ImageID, res) if s not in ("ok", "skip")]
    print("failures", len(bad), bad[:10])


if __name__ == "__main__":
    sys.exit(main())
