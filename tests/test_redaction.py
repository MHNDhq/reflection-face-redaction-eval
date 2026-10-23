"""Byte-level tests for scripts/redact.py. Run: .venv/bin/python tests/test_redaction.py"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import redact as R  # noqa: E402

rng = np.random.default_rng(0)
H, W = 540, 960
results = []


def check(name, cond):
    results.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name)


# 1. Pixels outside the boxes are byte-identical to the source; pixels inside are the fill colour.
frame = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
boxes = [(100.3, 50.7, 180.2, 150.9), (700, 300, 760, 380), (940, 520, 990, 560)]  # last one clipped at the edge
src_copy = frame.copy()  # snapshot BEFORE the first redact(), so in-place writes are caught
out = R.redact(frame, boxes)
mask = R.redaction_mask(frame.shape, boxes)
check("outside boxes byte-identical to the original source", np.array_equal(out[~mask], src_copy[~mask]))
check("inside boxes all fill colour", np.all(out[mask] == np.array(R.FILL, np.uint8)))
check("output is a separate buffer", not np.shares_memory(out, frame))
check("source frame bytes unchanged after redact", np.array_equal(frame, src_copy))
# Oracle independent of expand(): a box fully off-frame must change zero pixels.
off = R.redact(frame, [(-60, 10, -20, 50), (1000, 600, 1100, 700)])
check("box fully off-frame changes no pixels", np.array_equal(off, src_copy))
print(f"     changed pixels {int(mask.sum())} of {H * W} ({100 * mask.mean():.2f}%)")

# 2. Irreversibility: two different faces under the same boxes give byte-identical outputs in the boxes.
other = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
out2 = R.redact(other, boxes)
check("box content independent of input (zero information)", np.array_equal(out[mask], out2[mask]))

# 3. Temporal padding: a face found only at frame 100 stays redacted on frames 92..108.
dets = [[] for _ in range(200)]
dets[100] = [(10, 10, 50, 50)]
p = R.plan(dets, before=8, after=8)
check("frame 101..108 redacted after a hit at 100", all(p[k] for k in range(101, 109)))
check("frame 92..99 redacted before a hit at 100", all(p[k] for k in range(92, 100)))
check("frames 91 and 109 not redacted", not p[91] and not p[109])

# 4. Gap interpolation: hits at 100 and 125 (a 24-frame miss) are bridged.
dets = [[] for _ in range(200)]
dets[100] = [(10, 10, 50, 50)]
dets[125] = [(20, 12, 60, 52)]
p = R.plan(dets, before=8, after=8, max_gap=30)
check("frames 100..125 all redacted across a 24-frame miss", all(p[k] for k in range(100, 126)))

# 5. Lossless round trip: encode redacted frames with FFV1, decode, compare.
n = 12
src = [rng.integers(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(n)]
bx = [[(200 + 5 * i, 100, 280 + 5 * i, 200)] for i in range(n)]
red = [R.redact(f, b) for f, b in zip(src, bx)]
with tempfile.TemporaryDirectory() as td:
    raw = Path(td) / "in.rgb"
    raw.write_bytes(b"".join(f.tobytes() for f in red))
    mkv = Path(td) / "out.mkv"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                    "-r", "30", "-i", str(raw), "-c:v", "ffv1", "-pix_fmt", "rgb24", str(mkv)], check=True)
    dec = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(mkv), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         check=True, capture_output=True).stdout
dec = np.frombuffer(dec, np.uint8).reshape(n, H, W, 3)
check("FFV1 decode equals redacted frames byte for byte", all(np.array_equal(d, r) for d, r in zip(dec, red)))
ok = all(np.array_equal(d[~R.redaction_mask(d.shape, b)], s[~R.redaction_mask(d.shape, b)])
         for d, s, b in zip(dec, src, bx))
check("FFV1 decode: outside boxes equals SOURCE byte for byte", ok)

# Gap boundary: a 29-frame miss is bridged, a 30-frame miss is not (pad 8 covers only the edges).
def _filled(pl, f):
    return len(pl[f]) > 0
d29 = [[] for _ in range(200)]; d29[100] = [(10, 10, 50, 50)]; d29[130] = [(12, 10, 52, 50)]
p29 = R.plan(d29, 8, 8, 30)
check("29-frame miss bridged (frames 101..129 all filled)", all(_filled(p29, f) for f in range(101, 130)))
d30 = [[] for _ in range(200)]; d30[100] = [(10, 10, 50, 50)]; d30[131] = [(12, 10, 52, 50)]
p30 = R.plan(d30, 8, 8, 30)
check("30-frame miss NOT bridged (frames 109..122 unfilled)", not any(_filled(p30, f) for f in range(109, 123)))
# A jittered box at t+1 must not cancel the bridge (review 2026-10-23 repro).
dj = [[] for _ in range(200)]; dj[100] = [(10, 10, 50, 50)]; dj[101] = [(40, 10, 80, 50)]; dj[125] = [(0, 10, 40, 50)]
pj = R.plan(dj, 8, 8, 30)
check("jittered t+1 box does not cancel the bridge (frames 102..124 filled)", all(_filled(pj, f) for f in range(102, 125)))

# Jitter at EVERY offset between the two hits must not cancel the bridge.
for kk in range(1, 25):
    dk = [[] for _ in range(200)]; dk[100] = [(10, 10, 50, 50)]; dk[100 + kk] = [(40, 10, 80, 50)]; dk[125] = [(0, 10, 40, 50)]
    pk = R.plan(dk, 8, 8, 30)
    if not all(pk[f] for f in range(101, 125)):
        check(f"jitter at t+{kk} does not cancel the bridge", False)
        break
else:
    check("jitter at any offset t+1..t+24 does not cancel the bridge", True)
failed = [n for n, c in results if not c]
print(f"{len(results) - len(failed)}/{len(results)} passed")
sys.exit(1 if failed else 0)
