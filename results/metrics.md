# Results on the SYNTHETIC eval set (not the client's footage)

Eval: 860 composite images (test-split sources). Tune: 240 images (validation-split sources held out of training). Cutoff per detector = highest redaction recall with redaction precision >= 95% on TUNE, applied unchanged to EVAL.

| detector | cutoff | redaction recall | redaction precision | frames fully covered | false-alarm frames (negatives) | recall IoU0.5 | precision IoU0.5 |
|---|---|---|---|---|---|---|---|
| YuNet 2023mar (OpenCV) at 0.9 (client setting) | 0.9 | 30.7% | 99.5% | 26.7% | 0.0% | 29.5% | 98.3% |
| YuNet 2023mar (OpenCV) | 0.8 | 49.7% | 96.5% | 45.4% | 2.4% | 47.5% | 94.0% |
| BlazeFace short-range | 0.65 | 28.7% | 89.8% | 30.7% | 3.8% | 23.8% | 78.0% |
| BlazeFace full-range | 0.55 | 43.8% | 93.1% | 40.0% | 3.5% | 38.4% | 81.2% |
| Faster R-CNN, plain faces (before) | 0.9 | 51.2% | 88.0% | 49.6% | 8.8% | 47.5% | 82.7% |
| Faster R-CNN, control: +plain & negatives | 0.7 | 52.0% | 91.2% | 50.6% | 5.6% | 49.3% | 86.7% |
| Faster R-CNN, +reflection composites (after) | 0.65 | 65.8% | 90.2% | 62.3% | 9.4% | 63.1% | 86.4% |

## Comparison at equal precision (cutoff chosen on EVAL itself; for comparison only, not a deployable number)

| detector | cutoff | redaction recall | redaction precision |
|---|---|---|---|
| YuNet 2023mar (OpenCV) | 0.8 | 49.7% | 96.5% |
| BlazeFace short-range | 0.7 | 26.2% | 95.6% |
| BlazeFace full-range | 0.6 | 39.9% | 97.6% |
| Faster R-CNN, plain faces (before) | 0.99 | 36.6% | 97.9% |
| Faster R-CNN, control: +plain & negatives | 0.9 | 47.1% | 95.8% |
| Faster R-CNN, +reflection composites (after) | 0.9 | 55.2% | 96.9% |

## Redaction recall by family (shower = held out of training)

| detector | mirror | appliance | shower | photo | screen | plain |
|---|---|---|---|---|---|---|
| YuNet 2023mar (OpenCV) at 0.9 | 50.5% | 22.4% | 10.6% | 55.0% | 0.0% | 40.0% |
| YuNet 2023mar (OpenCV) at 0.8 | 75.2% | 37.4% | 31.7% | 74.0% | 7.4% | 64.2% |
| BlazeFace short-range at 0.65 | 40.4% | 23.4% | 23.1% | 26.0% | 2.5% | 52.6% |
| BlazeFace full-range at 0.55 | 59.6% | 40.2% | 38.5% | 46.0% | 9.9% | 62.1% |
| Faster R-CNN, plain faces (before) at 0.9 | 78.0% | 41.1% | 29.8% | 67.0% | 11.1% | 72.6% |
| Faster R-CNN, control: +plain & negatives at 0.7 | 82.6% | 38.3% | 29.8% | 69.0% | 4.9% | 78.9% |
| Faster R-CNN, +reflection composites (after) at 0.65 | 81.7% | 68.2% | 43.3% | 77.0% | 43.2% | 76.8% |

## False-alarm frames on hard negatives by family

| detector | mirror | appliance | shower | photo | screen | plain |
|---|---|---|---|---|---|---|
| YuNet 2023mar (OpenCV) at 0.8 | 0.0% | 3.6% | 5.3% | 1.7% | 0.0% | 3.6% |
| BlazeFace short-range at 0.65 | 3.6% | 5.4% | 5.3% | 3.4% | 0.0% | 5.4% |
| BlazeFace full-range at 0.55 | 3.6% | 0.0% | 7.0% | 5.2% | 0.0% | 5.4% |
| Faster R-CNN, plain faces (before) at 0.9 | 7.1% | 3.6% | 10.5% | 12.1% | 1.8% | 17.9% |
| Faster R-CNN, control: +plain & negatives at 0.7 | 3.6% | 3.6% | 3.5% | 8.6% | 0.0% | 14.3% |
| Faster R-CNN, +reflection composites (after) at 0.65 | 3.6% | 10.7% | 3.5% | 8.6% | 7.0% | 23.2% |
