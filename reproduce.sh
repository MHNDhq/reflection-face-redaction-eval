#!/usr/bin/env bash
# Rebuild every number in README.md from scratch. Needs Python 3.12, ffmpeg, and about 2 GB of disk.
# Training picks Apple MPS, then CUDA, then CPU. throughput.py uses hevc_videotoolbox, so that step needs macOS.
set -euo pipefail
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt; }
PY=.venv/bin/python
mkdir -p data/oi_meta models results
for u in v5/test-annotations-bbox.csv v5/validation-annotations-bbox.csv \
         2018_04/test/test-images-with-rotation.csv 2018_04/validation/validation-images-with-rotation.csv \
         v5/validation-annotations-human-imagelabels-boxable.csv v5/test-annotations-human-imagelabels-boxable.csv; do
  [ -f "data/oi_meta/$(basename $u)" ] || curl -sSfL -o "data/oi_meta/$(basename $u)" "https://storage.googleapis.com/openimages/$u"
done
[ -f models/face_detection_yunet_2023mar.onnx ] || curl -sSfL -o models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
for v in short full; do
  [ -f "models/blaze_face_${v}_range.tflite" ] || curl -sSfL -o "models/blaze_face_${v}_range.tflite" \
    "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_${v}_range/float16/latest/blaze_face_${v}_range.tflite"
done
$PY scripts/select_sources.py          # -> data/sources.csv (per-image licence + author), data/source_boxes.csv
$PY scripts/select_tune2.py 500 200    # + pool tune2 (validation)
$PY scripts/select_pool.py eval2 test 300 220 160         # round 3: fresh eval set
$PY scripts/select_pool.py tune3 validation 0 250 0       # round 3: tune negatives
$PY scripts/select_pool.py train_hn validation 0 400 0    # round 3: training hard negatives
$PY scripts/select_pool.py eval3 test 300 260 160         # round 4: fresh eval set
$PY scripts/download_sources.py        # -> data/images/{test,validation}/ (re-run if a download fails)
$PY scripts/filter_backgrounds.py      # -> data/bg_rejected.txt
$PY scripts/composite.py               # -> data/{eval,tune,train_comp,train_plain}/
$PY scripts/composite.py --tune2
for p in eval2 tune3 train_hn train_hn2 eval3; do $PY scripts/composite.py --pool $p; done
$PY scripts/train.py --stage A         # -> models/ft_A.pt
$PY scripts/train.py --stage B         # -> models/ft_B.pt
$PY scripts/train.py --stage control   # -> models/ft_control.pt
$PY scripts/train.py --stage B --steps 1500 --tag _long   # -> models/ft_B_long.pt
$PY scripts/train.py --stage HN --steps 1500              # -> models/ft_HN.pt
$PY scripts/train.py --stage HN2 --steps 2000             # -> models/ft_HN2.pt (the reported "after")
for s in tune tune2 tune3 eval eval2 eval3; do
  $PY scripts/evaluate.py --set $s --det yunet blazeface_short blazeface_full ckpt:models/ft_A.pt \
      ckpt:models/ft_control.pt ckpt:models/ft_B.pt ckpt:models/ft_B_long.pt ckpt:models/ft_HN.pt ckpt:models/ft_HN2.pt --thr 0.5
done
for s in eval eval2 eval3; do $PY scripts/report.py --eval $s; done   # -> results/metrics*.md, .json, pr_curve*.png
$PY scripts/blur_vs_fill.py            # -> results/blur_vs_fill.json
$PY scripts/sample_redactions.py       # -> results/sample_redactions.jpg
$PY scripts/contact_sheet.py eval positive 3 && $PY scripts/contact_sheet.py eval negative 3
$PY scripts/throughput.py              # -> results/throughput.json
$PY tests/test_redaction.py            # byte-level redaction tests
