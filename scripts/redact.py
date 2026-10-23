"""Temporal padding, gap interpolation and irreversible redaction.

Redaction = SOLID FILL of each padded face box with one constant colour. The output pixels inside a
box are a constant, so they hold zero information about the face that was there: no deblurring,
super-resolution or recognition model can recover what is not in the file. A Gaussian blur or a
small-block mosaic keeps a function of the face pixels and is not used.

Pixels outside the union of boxes are copied, never re-written (numpy slice assignment only touches
the box slices). tests/test_redaction.py checks both properties byte for byte.
"""
import numpy as np

FILL = (128, 128, 128)


def expand(box, margin, w, h):
    x0, y0, x1, y1 = box[:4]
    mw, mh = (x1 - x0) * margin, (y1 - y0) * margin
    # Clamp all four edges into the frame. A box fully off-frame used to give a NEGATIVE x1/y1,
    # which numpy reads as "all but the last N" and filled a band across the frame (review 2026-10-23).
    cx = lambda v: max(0, min(w, v))
    cy = lambda v: max(0, min(h, v))
    ex0, ey0 = cx(int(np.floor(x0 - mw))), cy(int(np.floor(y0 - mh)))
    ex1, ey1 = cx(int(np.ceil(x1 + mw))), cy(int(np.ceil(y1 + mh)))
    return (ex0, ey0, max(ex0, ex1), max(ey0, ey1))


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def _covers(outer, inner, margin=0.25):
    """True when the filled area of `outer` (expanded by the redaction margin, as redact() does)
    includes the filled area of `inner`. Checked on the same integer rectangles the fill uses, with
    no frame clamp, so skipping `inner` never loses a pixel."""
    BIG = 1 << 30
    o, n = expand(outer, margin, BIG, BIG), expand(inner, margin, BIG, BIG)
    return o[0] <= n[0] and o[1] <= n[1] and o[2] >= n[2] and o[3] >= n[3]


def interpolate_gaps(dets, max_gap=30, min_iou=0.1):
    """PROTOTYPE, privacy over cost (a track linker replaces it). dets: list (per frame) of boxes.
    Each box at frame t is bridged to EVERY box at t+g
    (1<g<=max_gap) that overlaps it (IoU >= min_iou), with linearly interpolated boxes in between.
    An interpolated box is skipped on frame t+k only when an ORIGINAL detection there covers its
    whole filled area, so skipping never loses a pixel; a neighbouring face, a
    small false positive or a jittered box never cancels it (review 2026-10-23: touching the span,
    then IoU >= 0.5, then a half-pixel tolerance were each too weak).
    Boxes in between never cancel a bridge (review 2026-10-23: a jittered box at t+1, then t+2,
    used to leave mid-gap frames unredacted), and nothing is merged or dropped, so the redacted
    area can only grow. The cost is some over-redaction when a face moves non-linearly."""
    dets = [[tuple(float(v) for v in b[:4]) for b in fr] for fr in dets]   # lists, arrays, scored 5-tuples
    out = [list(d) for d in dets]
    n = len(dets)
    for t in range(n):
        for a in dets[t]:
            for g in range(2, min(max_gap, n - 1 - t) + 1):
                for b in dets[t + g]:
                    if _iou(a, b) < min_iou:
                        continue
                    for k in range(1, g):
                        f = k / g
                        box = tuple(a[i] + (b[i] - a[i]) * f for i in range(4))   # exact when a == b
                        if not any(_covers(c, box) for c in dets[t + k]):
                            out[t + k].append(box)
    # Exact copies from overlapping pairs add no area; drop them (order kept).
    return [list(dict.fromkeys(fr)) for fr in out]


def pad(dets, before=8, after=8):
    """Every box found at frame t is also applied to frames t-before .. t+after."""
    n = len(dets)
    out = [[] for _ in range(n)]
    for t, boxes in enumerate(dets):
        boxes = [tuple(float(v) for v in b[:4]) for b in boxes]
        for k in range(max(0, t - before), min(n, t + after + 1)):
            out[k].extend(boxes)
    return [list(dict.fromkeys(fr)) for fr in out]   # exact copies add no area


def redact(frame, boxes, margin=0.25, fill=FILL):
    """Return a copy of frame (H,W,3 uint8) with every expanded box solid-filled."""
    h, w = frame.shape[:2]
    out = frame.copy()
    for b in boxes:
        x0, y0, x1, y1 = expand(b, margin, w, h)
        out[y0:y1, x0:x1] = fill
    return out


def redaction_mask(shape, boxes, margin=0.25):
    h, w = shape[:2]
    m = np.zeros((h, w), dtype=bool)
    for b in boxes:
        x0, y0, x1, y1 = expand(b, margin, w, h)
        m[y0:y1, x0:x1] = True
    return m


def plan(dets, before=8, after=8, max_gap=30):
    """Detections per frame -> boxes to redact per frame (interpolate gaps, then pad)."""
    return pad(interpolate_gaps(dets, max_gap=max_gap), before, after)
