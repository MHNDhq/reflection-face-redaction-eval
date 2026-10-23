"""Score detectors on the SYNTHETIC eval (or tune) set.

Usage:
  .venv/bin/python scripts/evaluate.py --set eval --det yunet blazeface_short blazeface_full ckpt:models/ft_B.pt
Writes results/<set>_<det>.json (raw detections) and prints metrics at fixed cutoffs.

Matching: detections sorted by score; a detection is a TRUE POSITIVE if IoU >= 0.5 with an unmatched
scoring face; IGNORED if it overlaps an ignore region (IoU >= 0.5, or >= 50% of the detection lies inside
it) or a face already matched; otherwise a FALSE POSITIVE.
Frame-level: a positive frame is FULLY COVERED when every scoring face in it is a true positive; a
negative frame is a FALSE-ALARM FRAME when it holds any false positive.
"""
import argparse
import json
from pathlib import Path

import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def ioa(det, region):
    ix = max(0, min(det[2], region[2]) - max(det[0], region[0]))
    iy = max(0, min(det[3], region[3]) - max(det[1], region[1]))
    a = (det[2] - det[0]) * (det[3] - det[1])
    return ix * iy / a if a > 0 else 0.0


def match_image(dets, gts, ign, thr):
    """Return (tp, fp, n_gt, matched_flags) at score threshold thr."""
    dets = sorted([d for d in dets if d[4] >= thr], key=lambda d: -d[4])
    used = [False] * len(gts)
    tp = fp = 0
    for d in dets:
        best, bi = 0.0, -1
        for i, g in enumerate(gts):
            v = iou(d, g)
            if v > best:
                best, bi = v, i
        if best >= 0.5 and not used[bi]:
            used[bi] = True
            tp += 1
        elif best >= 0.5:
            continue  # duplicate on a matched face: ignored, not a false alarm
        elif any(iou(d, r) >= 0.5 or ioa(d, r) >= 0.5 for r in ign):
            continue
        else:
            fp += 1
    return tp, fp, len(gts), used


def coverage_image(dets, gts, ign, thr, margin=0.25, need=0.7):
    """Redaction view. A face is COVERED when >= need of its box lies inside the union of the boxes that
    would actually be filled (detections expanded by the 25% redaction margin). A detection is a
    FALSE ALARM when its center is outside every face and ignore box and its IoU with each is < 0.1."""
    import numpy as np
    from redact import expand
    W, H = 960, 540
    keep = [d for d in dets if d[4] >= thr]
    m = np.zeros((H, W), bool)
    for d in keep:
        x0, y0, x1, y1 = expand(d, margin, W, H)
        m[y0:y1, x0:x1] = True
    cov = []
    for g in gts:
        x0, y0, x1, y1 = int(g[0]), int(g[1]), int(np.ceil(g[2])), int(np.ceil(g[3]))
        a = max(1, (x1 - x0) * (y1 - y0))
        cov.append(m[y0:y1, x0:x1].sum() / a >= need)
    fa = 0
    for d in keep:
        cx, cy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
        if not any((r[0] <= cx <= r[2] and r[1] <= cy <= r[3]) or iou(d, r) >= 0.1 for r in gts + ign):
            fa += 1
    return cov, fa, len(keep)


def metrics(records, dets_by_file, thr, family=None):
    TP = FP = NG = 0
    pos_frames = covered = neg_frames = fa_frames = 0
    c_hit = c_fa = c_n = c_frames = c_neg_fa = 0
    for r in records:
        if family and r["family"] != family:
            continue
        tp, fp, ng, used = match_image(dets_by_file[r["file"]], r["boxes"], r["ignore"], thr)
        TP, FP, NG = TP + tp, FP + fp, NG + ng
        if ng:
            pos_frames += 1
            covered += int(all(used))
        if r["label"] == "negative":
            neg_frames += 1
            fa_frames += int(fp > 0)
        cov, fa, nk = coverage_image(dets_by_file[r["file"]], r["boxes"], r["ignore"], thr)
        c_hit += sum(cov)
        c_fa += fa
        c_n += nk
        c_frames += int(bool(cov) and all(cov))
        if r["label"] == "negative":
            c_neg_fa += int(fa > 0)
    red = {"redaction_recall": c_hit / NG if NG else None,
           "redaction_precision": (c_n - c_fa) / c_n if c_n else None,
           "redaction_frames_fully_covered": c_frames / pos_frames if pos_frames else None,
           "redaction_false_alarm_frame_rate": c_neg_fa / neg_frames if neg_frames else None}
    return {**red, "thr": thr, "recall": TP / NG if NG else None, "precision": TP / (TP + FP) if TP + FP else None,
            "tp": TP, "fp": FP, "faces": NG, "frames_fully_covered": covered / pos_frames if pos_frames else None,
            "pos_frames": pos_frames, "neg_frames": neg_frames,
            "false_alarm_frame_rate": fa_frames / neg_frames if neg_frames else None}


def make_detector(spec):
    import detectors as D
    if spec == "yunet":
        return D.YuNet(score=0.05)
    if spec == "yunet@0.9":
        return D.YuNet(score=0.9)
    if spec.startswith("blazeface_"):
        return D.BlazeFace(spec.split("_")[1], score=0.05)
    if spec.startswith("ckpt:"):
        return D.FRCNN(ROOT / spec[5:])
    raise SystemExit(f"unknown detector {spec}")


def run(spec, set_name):
    ann = json.load(open(ROOT / "data" / set_name / "annotations.json"))
    det = make_detector(spec)
    name = spec.replace("ckpt:models/", "").replace(".pt", "").replace("@", "_at_")
    out = {}
    recs = ann["images"]
    if hasattr(det, "batch"):
        for i in range(0, len(recs), 8):
            chunk = recs[i:i + 8]
            arrs = [np.asarray(Image.open(ROOT / "data" / set_name / "images" / r["file"]).convert("RGB")) for r in chunk]
            for r, d in zip(chunk, det.batch(arrs)):
                out[r["file"]] = [list(map(float, x)) for x in d]
    else:
        for r in recs:
            a = np.asarray(Image.open(ROOT / "data" / set_name / "images" / r["file"]).convert("RGB"))
            out[r["file"]] = [list(map(float, x)) for x in det(a)]
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(out, open(ROOT / "results" / f"{set_name}_{name}.json", "w"))
    return name, recs, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="eval")
    ap.add_argument("--det", nargs="+", required=True)
    ap.add_argument("--thr", nargs="+", type=float, default=[0.5, 0.7, 0.9])
    ap.add_argument("--rescore", action="store_true", help="reuse results/<set>_<name>.json instead of re-running")
    a = ap.parse_args()
    for spec in a.det:
        if a.rescore:
            name = spec.replace("ckpt:models/", "").replace(".pt", "").replace("@", "_at_")
            recs = json.load(open(ROOT / "data" / a.set / "annotations.json"))["images"]
            out = json.load(open(ROOT / "results" / f"{a.set}_{name}.json"))
        else:
            name, recs, out = run(spec, a.set)
        for t in a.thr:
            print(name, json.dumps(metrics(recs, out, t)))


if __name__ == "__main__":
    main()
