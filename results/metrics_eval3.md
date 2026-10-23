# Results on the SYNTHETIC eval3 set (not the client's footage)

eval3: 1120 composite images (test-split sources). Tune: 240 images (validation-split sources held out of training). Cutoff per detector = highest redaction recall on TUNE (2550 images: tune + tune2) among cutoffs whose tune precision has a cluster-bootstrap 5th percentile >= 95% (resampling source photos, 1,000 draws, one-sided 95%), applied unchanged to EVAL. Pre-registered before tune2 detections or a new eval read; see README for the two earlier rules.

| detector | cutoff | redaction recall | redaction precision | frames fully covered | false-alarm frames (negatives) | recall IoU0.5 | precision IoU0.5 |
|---|---|---|---|---|---|---|---|
| YuNet 2023mar (OpenCV) at 0.9 (client setting) | 0.9 | 26.1% | 99.0% | 22.5% | 0.4% | 25.3% | 98.0% |
| YuNet 2023mar (OpenCV) | 0.8 | 45.9% | 95.7% | 40.1% | 1.7% | 44.2% | 93.9% |
| BlazeFace short-range | 0.7 | 20.5% | 96.9% | 21.9% | 0.8% | 17.8% | 87.2% |
| BlazeFace full-range | 0.6 | 36.1% | 95.8% | 32.0% | 1.5% | 31.2% | 84.1% |
| Faster R-CNN, plain faces (before) | 0.94 | 43.8% | 96.5% | 41.1% | 1.5% | 40.6% | 91.4% |
| Faster R-CNN, control: +plain & negatives | 0.75 | 47.4% | 94.5% | 43.7% | 1.3% | 43.3% | 88.9% |
| Faster R-CNN, +reflection composites (500 steps) | 0.65 | 60.4% | 93.8% | 58.1% | 3.5% | 56.0% | 87.9% |
| Faster R-CNN, +reflection composites, 1,500 steps | 0.75 | 63.1% | 96.1% | 61.3% | 2.5% | 58.3% | 90.1% |
| Faster R-CNN, +reflection composites + hard negatives, 1,500 steps | 0.65 | 65.4% | 92.4% | 63.6% | 4.4% | 59.8% | 84.8% |
| Faster R-CNN, + hard negatives x2, mirror-weighted, 2,000 steps | 0.6 | 65.1% | 95.4% | 62.6% | 2.3% | 60.0% | 89.3% |

## Comparison at equal precision (cutoff chosen on EVAL itself; for comparison only, not a deployable number)

| detector | cutoff | redaction recall | redaction precision |
|---|---|---|---|
| YuNet 2023mar (OpenCV) | 0.8 | 45.9% | 95.7% |
| BlazeFace short-range | 0.7 | 20.5% | 96.9% |
| BlazeFace full-range | 0.6 | 36.1% | 95.8% |
| Faster R-CNN, plain faces (before) | 0.94 | 43.8% | 96.5% |
| Faster R-CNN, control: +plain & negatives | 0.8 | 45.8% | 95.6% |
| Faster R-CNN, +reflection composites (500 steps) | 0.75 | 57.1% | 95.7% |
| Faster R-CNN, +reflection composites, 1,500 steps | 0.7 | 64.2% | 95.5% |
| Faster R-CNN, +reflection composites + hard negatives, 1,500 steps | 0.8 | 61.4% | 96.2% |
| Faster R-CNN, + hard negatives x2, mirror-weighted, 2,000 steps | 0.6 | 65.1% | 95.4% |

## Redaction recall by family (shower = held out of training)

| detector | mirror | appliance | shower | photo | screen | plain |
|---|---|---|---|---|---|---|
| YuNet 2023mar (OpenCV) at 0.9 | 44.4% | 15.7% | 0.7% | 43.3% | 0.0% | 52.3% |
| YuNet 2023mar (OpenCV) at 0.8 | 65.4% | 28.1% | 11.1% | 75.8% | 19.5% | 76.2% |
| BlazeFace short-range at 0.7 | 27.1% | 17.4% | 15.6% | 16.7% | 0.0% | 44.6% |
| BlazeFace full-range at 0.6 | 54.9% | 30.6% | 17.8% | 36.7% | 13.0% | 62.3% |
| Faster R-CNN, plain faces (before) at 0.94 | 62.4% | 28.9% | 18.5% | 59.2% | 14.6% | 78.5% |
| Faster R-CNN, control: +plain & negatives at 0.75 | 69.9% | 29.8% | 18.5% | 68.3% | 13.8% | 83.1% |
| Faster R-CNN, +reflection composites (500 steps) at 0.65 | 73.7% | 53.7% | 25.9% | 74.2% | 50.4% | 85.4% |
| Faster R-CNN, +reflection composites, 1,500 steps at 0.75 | 72.9% | 63.6% | 31.1% | 70.8% | 56.9% | 84.6% |
| Faster R-CNN, +reflection composites + hard negatives, 1,500 steps at 0.65 | 75.9% | 64.5% | 26.7% | 78.3% | 61.8% | 86.9% |
| Faster R-CNN, + hard negatives x2, mirror-weighted, 2,000 steps at 0.6 | 75.9% | 65.3% | 26.7% | 76.7% | 60.2% | 87.7% |

## False-alarm frames on hard negatives by family

| detector | mirror | appliance | shower | photo | screen | plain |
|---|---|---|---|---|---|---|
| YuNet 2023mar (OpenCV) at 0.8 | 2.3% | 0.0% | 0.0% | 0.0% | 0.0% | 8.1% |
| BlazeFace short-range at 0.7 | 1.1% | 1.1% | 1.1% | 1.2% | 0.0% | 0.0% |
| BlazeFace full-range at 0.6 | 0.0% | 2.3% | 2.3% | 1.2% | 1.2% | 2.3% |
| Faster R-CNN, plain faces (before) at 0.94 | 1.1% | 0.0% | 2.3% | 2.3% | 0.0% | 3.5% |
| Faster R-CNN, control: +plain & negatives at 0.75 | 1.1% | 0.0% | 2.3% | 0.0% | 2.3% | 2.3% |
| Faster R-CNN, +reflection composites (500 steps) at 0.65 | 2.3% | 2.3% | 5.7% | 1.2% | 5.8% | 3.5% |
| Faster R-CNN, +reflection composites, 1,500 steps at 0.75 | 2.3% | 1.1% | 1.1% | 0.0% | 5.8% | 4.7% |
| Faster R-CNN, +reflection composites + hard negatives, 1,500 steps at 0.65 | 3.4% | 2.3% | 0.0% | 3.5% | 12.8% | 4.7% |
| Faster R-CNN, + hard negatives x2, mirror-weighted, 2,000 steps at 0.6 | 1.1% | 4.5% | 1.1% | 0.0% | 1.2% | 5.8% |
