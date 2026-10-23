#!/usr/bin/env bash
# Rebuild every number in README.md from scratch. Needs Python 3.12, ffmpeg, and about 2 GB of disk.
# Training uses Apple MPS; on another machine change device="mps" in scripts/train.py and scripts/detectors.py.
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
$PY scripts/download_sources.py        # -> data/images/{test,validation}/
$PY scripts/filter_backgrounds.py      # -> data/bg_rejected.txt
$PY scripts/composite.py               # -> data/{eval,tune,train_comp,train_plain}/
$PY scripts/train.py --stage A         # -> models/ft_A.pt
$PY scripts/train.py --stage B         # -> models/ft_B.pt
$PY scripts/train.py --stage control   # -> models/ft_control.pt
for s in eval tune; do
  $PY scripts/evaluate.py --set $s --det yunet blazeface_short blazeface_full \
      ckpt:models/ft_A.pt ckpt:models/ft_control.pt ckpt:models/ft_B.pt --thr 0.5
done
$PY scripts/report.py                  # -> results/metrics.md, metrics.json, pr_curve.png
$PY scripts/blur_vs_fill.py            # -> results/blur_vs_fill.json
$PY scripts/sample_redactions.py       # -> results/sample_redactions.jpg
$PY scripts/contact_sheet.py eval positive 3 && $PY scripts/contact_sheet.py eval negative 3
$PY scripts/throughput.py              # -> results/throughput.json
$PY tests/test_redaction.py            # byte-level redaction tests
