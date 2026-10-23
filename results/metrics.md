# Results on the SYNTHETIC eval set (not the client's footage)

Eval: 860 composite images (test-split sources). Tune: 240 images (validation-split sources held out of training). Cutoff per detector = highest redaction recall on TUNE (1800 images: tune + tune2) among cutoffs whose tune precision has a cluster-bootstrap 5th percentile >= 95% (resampling source photos, 1,000 draws, one-sided 95%), applied unchanged to EVAL. Pre-registered before tune2 detections or a new eval read; see README for the two earlier rules.

| detector | cutoff | redaction recall | redaction precision | frames fully covered | false-alarm frames (negatives) | recall IoU0.5 | precision IoU0.5 |
|---|---|---|---|---|---|---|---|
| YuNet 2023mar (OpenCV) at 0.9 (client setting) | 0.9 | 30.7% | 99.5% | 26.7% | 0.0% | 29.5% | 98.3% |
| YuNet 2023mar (OpenCV) | 0.75 | 54.0% | 94.5% | 49.2% | 4.1% | 51.7% | 91.7% |
| BlazeFace short-range | 0.7 | 26.2% | 95.6% | 27.9% | 1.8% | 23.2% | 87.9% |
| BlazeFace full-range | 0.55 | 43.8% | 93.1% | 40.0% | 3.5% | 38.4% | 81.2% |
| Faster R-CNN, plain faces (before) | 0.9 | 51.2% | 88.0% | 49.6% | 8.8% | 47.5% | 82.7% |
| Faster R-CNN, control: +plain & negatives | 0.65 | 52.7% | 90.6% | 51.3% | 6.2% | 49.7% | 85.8% |
| Faster R-CNN, +reflection composites (500 steps) | 0.55 | 67.8% | 87.2% | 64.2% | 12.6% | 64.6% | 81.9% |
| Faster R-CNN, +reflection composites, 1,500 steps (after; chosen on tune) | 0.6 | 71.6% | 87.6% | 68.1% | 12.4% | 68.5% | 82.9% |

## Comparison at equal precision (cutoff chosen on EVAL itself; for comparison only, not a deployable number)

| detector | cutoff | redaction recall | redaction precision |
|---|---|---|---|
| YuNet 2023mar (OpenCV) | 0.8 | 49.7% | 96.5% |
| BlazeFace short-range | 0.7 | 26.2% | 95.6% |
| BlazeFace full-range | 0.6 | 39.9% | 97.6% |
| Faster R-CNN, plain faces (before) | 0.98 | 40.8% | 96.3% |
| Faster R-CNN, control: +plain & negatives | 0.9 | 47.1% | 95.8% |
| Faster R-CNN, +reflection composites (500 steps) | 0.9 | 55.2% | 96.9% |
| Faster R-CNN, +reflection composites, 1,500 steps (after; chosen on tune) | 0.92 | 59.7% | 95.2% |

## Redaction recall by family (shower = held out of training)

| detector | mirror | appliance | shower | photo | screen | plain |
|---|---|---|---|---|---|---|
| YuNet 2023mar (OpenCV) at 0.9 | 50.5% | 22.4% | 10.6% | 55.0% | 0.0% | 40.0% |
| YuNet 2023mar (OpenCV) at 0.75 | 77.1% | 42.1% | 34.6% | 78.0% | 9.9% | 74.7% |
| BlazeFace short-range at 0.7 | 34.9% | 19.6% | 22.1% | 23.0% | 1.2% | 52.6% |
| BlazeFace full-range at 0.55 | 59.6% | 40.2% | 38.5% | 46.0% | 9.9% | 62.1% |
| Faster R-CNN, plain faces (before) at 0.9 | 78.0% | 41.1% | 29.8% | 67.0% | 11.1% | 72.6% |
| Faster R-CNN, control: +plain & negatives at 0.65 | 82.6% | 39.3% | 29.8% | 70.0% | 7.4% | 78.9% |
| Faster R-CNN, +reflection composites (500 steps) at 0.55 | 81.7% | 69.2% | 44.2% | 78.0% | 48.1% | 82.1% |
| Faster R-CNN, +reflection composites, 1,500 steps (after; chosen on tune) at 0.6 | 86.2% | 72.0% | 49.0% | 80.0% | 55.6% | 84.2% |

## False-alarm frames on hard negatives by family

| detector | mirror | appliance | shower | photo | screen | plain |
|---|---|---|---|---|---|---|
| YuNet 2023mar (OpenCV) at 0.75 | 3.6% | 5.4% | 5.3% | 1.7% | 0.0% | 8.9% |
| BlazeFace short-range at 0.7 | 1.8% | 1.8% | 3.5% | 1.7% | 0.0% | 1.8% |
| BlazeFace full-range at 0.55 | 3.6% | 0.0% | 7.0% | 5.2% | 0.0% | 5.4% |
| Faster R-CNN, plain faces (before) at 0.9 | 7.1% | 3.6% | 10.5% | 12.1% | 1.8% | 17.9% |
| Faster R-CNN, control: +plain & negatives at 0.65 | 3.6% | 3.6% | 3.5% | 8.6% | 0.0% | 17.9% |
| Faster R-CNN, +reflection composites (500 steps) at 0.55 | 7.1% | 16.1% | 5.3% | 8.6% | 10.5% | 28.6% |
| Faster R-CNN, +reflection composites, 1,500 steps (after; chosen on tune) at 0.6 | 5.4% | 21.4% | 8.8% | 5.2% | 8.8% | 25.0% |
