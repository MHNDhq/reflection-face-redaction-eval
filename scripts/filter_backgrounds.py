"""Drop background photos where YuNet (>=0.5) or BlazeFace full-range (>=0.5) fires.

Backgrounds are labelled person-free in Open Images, but a stray unlabelled face would become an
unscored false alarm in every composite built on it. This filter runs on BACKGROUNDS ONLY; hard
negatives (toys, dolls, prints) are never filtered by a detector, since that would remove exactly
the cases the eval is meant to test. Writes data/bg_rejected.txt.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import detectors as D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
src = pd.read_csv(ROOT / "data" / "sources.csv")
bg = src[src.role == "background"]
y, b = D.YuNet(score=0.5), D.BlazeFace("full", score=0.5)
rej = []
for r in bg.itertuples():
    p = ROOT / "data" / "images" / r.split / f"{r.ImageID}.jpg"
    if not p.exists():
        continue
    a = np.asarray(Image.open(p).convert("RGB"))
    if y(a) or b(a):
        rej.append(r.ImageID)
(ROOT / "data" / "bg_rejected.txt").write_text("\n".join(rej) + "\n")
print(f"backgrounds {len(bg)}, rejected {len(rej)}")
