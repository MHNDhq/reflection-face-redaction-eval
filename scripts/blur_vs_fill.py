"""Does a redaction still leave a findable face? Light blur vs heavy blur vs pixelation vs solid fill.

For every scoring face in the 'plain' eval family that YuNet finds on the clean image (score >= 0.5,
IoU >= 0.5), apply each redaction to that face's box (25% margin) and run YuNet again. A face that is
still detected after redaction still carries face structure. Writes results/blur_vs_fill.json.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import detectors as D  # noqa: E402
import redact as R  # noqa: E402
from evaluate import iou  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def apply(img, box, mode):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = R.expand(box, 0.25, w, h)
    out = img.copy()
    roi = out[y0:y1, x0:x1]
    if mode == "blur_light":
        out[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (0, 0), 3)
    elif mode == "blur_heavy":
        out[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (0, 0), max(4, (x1 - x0) / 6))
    elif mode == "pixelate_8":
        small = cv2.resize(roi, (max(1, (x1 - x0) // 8), max(1, (y1 - y0) // 8)), interpolation=cv2.INTER_AREA)
        out[y0:y1, x0:x1] = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    elif mode == "fill":
        out = R.redact(img, [box])
    return out


def main():
    ann = json.load(open(ROOT / "data" / "eval" / "annotations.json"))["images"]
    det = D.YuNet(score=0.5)
    modes = ["blur_light", "blur_heavy", "pixelate_8", "fill"]
    still = {m: 0 for m in modes}
    n = 0
    for r in ann:
        if r["family"] != "plain" or not r["boxes"]:
            continue
        img = np.asarray(Image.open(ROOT / "data" / "eval" / "images" / r["file"]).convert("RGB"))
        found = det(img)
        for g in r["boxes"]:
            if not any(iou(d, g) >= 0.5 for d in found):
                continue
            n += 1
            for m in modes:
                red = apply(img, g, m)
                if any(iou(d, g) >= 0.3 for d in det(red)):
                    still[m] += 1
    res = {"faces_tested": n, "still_detected_after": still,
           "rate": {m: round(still[m] / n, 3) if n else None for m in modes},
           "note": ("SYNTHETIC eval set, plain family: the source photo resized and centre-cropped, mirrored half the time, no composite; faces kept where "
                    "YuNet at 0.5 finds them at IoU >= 0.5, re-detection after redaction counted at IoU >= 0.3")}
    json.dump(res, open(ROOT / "results" / "blur_vs_fill.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
