"""Measure detector throughput on this machine (Apple M5, 10-core GPU, 24 GB) and 4K HEVC decode speed.

Usage: .venv/bin/python scripts/throughput.py
Writes results/throughput.json. Frames are taken from the synthetic eval set and resized.
"""
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import detectors as D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def frames(size, n=64):
    files = sorted((ROOT / "data" / "eval" / "images").glob("*.jpg"))[:n]
    return [np.asarray(Image.open(f).convert("RGB").resize(size, Image.BILINEAR)) for f in files]


def bench(fn, arrs, batch=1, warm=8):
    for i in range(0, warm, batch):
        fn(arrs[i:i + batch]) if batch > 1 else fn(arrs[i])
    t = time.time()
    k = 0
    for i in range(0, len(arrs) - batch + 1, batch):
        fn(arrs[i:i + batch]) if batch > 1 else fn(arrs[i])
        k += batch
    return k / (time.time() - t)


def main():
    out = {"machine": platform.platform(), "cpu_threads_cv2": cv2.getNumThreads(), "runs": []}
    for size in [(960, 540), (1920, 1080)]:
        arrs = frames(size)
        y = D.YuNet(score=0.9)
        out["runs"].append({"det": "yunet_2023mar (CPU, OpenCV)", "size": size, "fps": round(bench(y, arrs), 2)})
        b = D.BlazeFace("full", score=0.5)
        out["runs"].append({"det": "blazeface_full (CPU, MediaPipe)", "size": size, "fps": round(bench(b, arrs), 2)})
        ck = ROOT / "models" / "ft_B.pt"
        if ck.exists():
            f = D.FRCNN(ck)
            out["runs"].append({"det": f"ft_B Faster R-CNN MobileNetV3 ({str(f.dev).upper()}, batch 8)", "size": size,
                                "fps": round(bench(f.batch, arrs, batch=8), 2)})
            print(out["runs"][-1], flush=True)
    # 4K HEVC decode speed: synthetic 10 s 3840x2160 30 fps HEVC clip, hardware decode via VideoToolbox.
    clip = ROOT / "results" / "synthetic_4k_hevc.mp4"
    if not clip.exists():
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=3840x2160:rate=30",
                        "-t", "10", "-c:v", "hevc_videotoolbox", "-b:v", "40M", "-tag:v", "hvc1", str(clip)], check=True)
    for hw in (["-hwaccel", "videotoolbox"], []):
        t = time.time()
        subprocess.run(["ffmpeg", "-loglevel", "error", *hw, "-i", str(clip), "-f", "null", "-"], check=True)
        out["runs"].append({"det": "ffmpeg HEVC 4K decode " + ("VideoToolbox" if hw else "software"),
                            "size": [3840, 2160], "fps": round(300 / (time.time() - t), 2)})
    json.dump(out, open(ROOT / "results" / "throughput.json", "w"), indent=1)
    for r in out["runs"]:
        print(r)


if __name__ == "__main__":
    main()
