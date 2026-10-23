"""Side-by-side sample: SYNTHETIC eval frames before and after ft_B (cutoff from tune) + solid fill."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import redact as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ann = json.load(open(ROOT / "data" / "eval" / "annotations.json"))["images"]
dets = json.load(open(ROOT / "results" / "eval_ft_B.json"))
cut = json.load(open(ROOT / "results" / "metrics.json"))["detectors"]["ft_B"]["cutoff_from_tune"]
if cut is None:
    raise SystemExit("ft_B has no cutoff that meets the rule; no sample sheet")
pick = []
for fam in ["mirror", "appliance", "shower", "screen"]:
    pick += [r for r in ann if r["family"] == fam and r["label"] == "positive"][3:4]
rows = []
for r in pick:
    im = np.asarray(Image.open(ROOT / "data" / "eval" / "images" / r["file"]).convert("RGB"))
    boxes = [d[:4] for d in dets[r["file"]] if d[4] >= cut]
    out = R.redact(im, boxes)
    a, b = Image.fromarray(im).resize((480, 270)), Image.fromarray(out).resize((480, 270))
    tile = Image.new("RGB", (960, 270))
    tile.paste(a, (0, 0)); tile.paste(b, (480, 0))
    d = ImageDraw.Draw(tile)
    d.rectangle([0, 0, 300, 18], fill=(0, 0, 0)); d.text((6, 4), f"SYNTHETIC | {r['family']} | source", fill=(255, 255, 255))
    d.rectangle([480, 0, 840, 18], fill=(0, 0, 0)); d.text((486, 4), f"ft_B at {cut} + solid fill", fill=(255, 255, 255))
    rows.append(tile)
sheet = Image.new("RGB", (960, 270 * len(rows)))
for i, t in enumerate(rows):
    sheet.paste(t, (0, 270 * i))
sheet.save(ROOT / "results" / "sample_redactions.jpg", quality=88)
print("results/sample_redactions.jpg", len(rows))
