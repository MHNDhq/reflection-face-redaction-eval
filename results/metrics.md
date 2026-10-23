# Results on the SYNTHETIC eval set (not the client's footage)

eval: 860 composite images (test-split sources). Tune: 240 images (validation-split sources held out of training). Cutoff per detector = highest redaction recall on TUNE (2550 images: tune + tune2) among cutoffs whose tune precision has a cluster-bootstrap 5th percentile >= 95% (resampling source photos, 1,000 draws, one-sided 95%), applied unchanged to EVAL. Pre-registered before tune2 detections or a new eval read; see README for the two earlier rules.

| detector | cutoff | redaction recall | redaction precision | frames fully covered | false-alarm frames (negatives) | recall IoU0.5 | precision IoU0.5 |
|---|---|---|---|---|---|---|---|
| YuNet 2023mar (OpenCV) at 0.9 (client setting) | 0.9 | 30.7% | 99.5% | 26.7% | 0.0% | 29.5% | 98.3% |
| YuNet 2023mar (OpenCV) | 0.8 | 49.7% | 96.5% | 45.4% | 2.4% | 47.5% | 94.0% |
| BlazeFace short-range | 0.7 | 26.2% | 95.6% | 27.9% | 1.8% | 23.2% | 87.9% |
| BlazeFace full-range | 0.6 | 39.9% | 97.6% | 35.8% | 0.6% | 36.2% | 88.5% |
| Faster R-CNN, plain faces (before) | 0.94 | 48.8% | 90.8% | 47.5% | 6.2% | 45.6% | 86.3% |
| Faster R-CNN, control: +plain & negatives | 0.75 | 51.0% | 91.5% | 50.1% | 5.0% | 48.3% | 87.3% |
| Faster R-CNN, +reflection composites (500 steps) | 0.65 | 65.8% | 90.2% | 62.3% | 9.4% | 63.1% | 86.4% |
| Faster R-CNN, +reflection composites, 1,500 steps | 0.75 | 68.1% | 91.1% | 64.4% | 8.2% | 65.6% | 87.9% |
| Faster R-CNN, + hard negatives x2, mirror-weighted, 2,000 steps | 0.6 | 70.5% | 89.6% | 66.0% | 9.1% | 68.0% | 85.8% |

## Comparison at equal precision (cutoff chosen on EVAL itself; for comparison only, not a deployable number)

| detector | cutoff | redaction recall | redaction precision |
|---|---|---|---|
| YuNet 2023mar (OpenCV) | 0.8 | 49.7% | 96.5% |
| BlazeFace short-range | 0.7 | 26.2% | 95.6% |
| BlazeFace full-range | 0.6 | 39.9% | 97.6% |
| Faster R-CNN, plain faces (before) | 0.98 | 40.8% | 96.3% |
| Faster R-CNN, control: +plain & negatives | 0.9 | 47.1% | 95.8% |
| Faster R-CNN, +reflection composites (500 steps) | 0.9 | 55.2% | 96.9% |
| Faster R-CNN, +reflection composites, 1,500 steps | 0.92 | 59.7% | 95.2% |
| Faster R-CNN, + hard negatives x2, mirror-weighted, 2,000 steps | 0.85 | 63.3% | 95.0% |

## Redaction recall by family (shower = held out of training)

| detector | mirror | appliance | shower | photo | screen | plain |
|---|---|---|---|---|---|---|
| YuNet 2023mar (OpenCV) at 0.9 | 50.5% | 22.4% | 10.6% | 55.0% | 0.0% | 40.0% |
| YuNet 2023mar (OpenCV) at 0.8 | 75.2% | 37.4% | 31.7% | 74.0% | 7.4% | 64.2% |
| BlazeFace short-range at 0.7 | 34.9% | 19.6% | 22.1% | 23.0% | 1.2% | 52.6% |
| BlazeFace full-range at 0.6 | 56.0% | 37.4% | 36.5% | 45.0% | 3.7% | 53.7% |
| Faster R-CNN, plain faces (before) at 0.94 | 74.3% | 40.2% | 26.9% | 64.0% | 11.1% | 69.5% |
| Faster R-CNN, control: +plain & negatives at 0.75 | 81.7% | 37.4% | 28.8% | 69.0% | 3.7% | 76.8% |
| Faster R-CNN, +reflection composites (500 steps) at 0.65 | 81.7% | 68.2% | 43.3% | 77.0% | 43.2% | 76.8% |
| Faster R-CNN, +reflection composites, 1,500 steps at 0.75 | 81.7% | 71.0% | 46.2% | 76.0% | 49.4% | 81.1% |
| Faster R-CNN, + hard negatives x2, mirror-weighted, 2,000 steps at 0.6 | 85.3% | 75.7% | 41.3% | 79.0% | 54.3% | 84.2% |

## False-alarm frames on hard negatives by family

| detector | mirror | appliance | shower | photo | screen | plain |
|---|---|---|---|---|---|---|
| YuNet 2023mar (OpenCV) at 0.8 | 0.0% | 3.6% | 5.3% | 1.7% | 0.0% | 3.6% |
| BlazeFace short-range at 0.7 | 1.8% | 1.8% | 3.5% | 1.7% | 0.0% | 1.8% |
| BlazeFace full-range at 0.6 | 0.0% | 0.0% | 1.8% | 1.7% | 0.0% | 0.0% |
| Faster R-CNN, plain faces (before) at 0.94 | 5.4% | 1.8% | 3.5% | 10.3% | 1.8% | 14.3% |
| Faster R-CNN, control: +plain & negatives at 0.75 | 3.6% | 3.6% | 1.8% | 6.9% | 0.0% | 14.3% |
| Faster R-CNN, +reflection composites (500 steps) at 0.65 | 3.6% | 10.7% | 3.5% | 8.6% | 7.0% | 23.2% |
| Faster R-CNN, +reflection composites, 1,500 steps at 0.75 | 3.6% | 16.1% | 8.8% | 1.7% | 3.5% | 16.1% |
| Faster R-CNN, + hard negatives x2, mirror-weighted, 2,000 steps at 0.6 | 3.6% | 19.6% | 5.3% | 1.7% | 5.3% | 19.6% |
