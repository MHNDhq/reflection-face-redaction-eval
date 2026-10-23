"""Contact sheet of eval composites with ground truth drawn (green = scoring face, yellow = ignore).

Usage: .venv/bin/python scripts/contact_sheet.py [set] [label] [n_per_family] [out]
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
set_name = sys.argv[1] if len(sys.argv) > 1 else "eval"
label = sys.argv[2] if len(sys.argv) > 2 else "positive"
k = int(sys.argv[3]) if len(sys.argv) > 3 else 3
out = sys.argv[4] if len(sys.argv) > 4 else f"results/contact_{set_name}_{label}.jpg"
ann = json.load(open(ROOT / "data" / set_name / "annotations.json"))["images"]
fams = sorted({r["family"] for r in ann})
tiles = []
for f in fams:
    rs = [r for r in ann if r["family"] == f and r["label"] == label][:k]
    for r in rs:
        im = Image.open(ROOT / "data" / set_name / "images" / r["file"]).convert("RGB")
        d = ImageDraw.Draw(im)
        for b in r["boxes"]:
            d.rectangle(b, outline=(0, 255, 0), width=3)
        for b in r["ignore"]:
            d.rectangle(b, outline=(255, 220, 0), width=2)
        d.rectangle([0, 0, 330, 34], fill=(0, 0, 0))
        d.text((8, 8), f"{f} | {r['source_category'] or r['source_role']} | SYNTHETIC", fill=(255, 255, 255))
        tiles.append(im.resize((480, 270)))
cols = k
rows = (len(tiles) + cols - 1) // cols
sheet = Image.new("RGB", (480 * cols, 270 * rows), (255, 255, 255))
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % cols) * 480, (i // cols) * 270))
sheet.save(ROOT / out, quality=88)
print(out, len(tiles))
