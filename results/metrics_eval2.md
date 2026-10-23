# Results on the SYNTHETIC eval2 set (not the client's footage)

eval2: 1040 composite images (test-split sources). Tune: 240 images (validation-split sources held out of training). Cutoff per detector = highest redaction recall on TUNE (2550 images: tune + tune2) among cutoffs whose tune precision has a cluster-bootstrap 5th percentile >= 95% (resampling source photos, 1,000 draws, one-sided 95%), applied unchanged to EVAL. Pre-registered before tune2 detections or a new eval read; see README for the two earlier rules.

| detector | cutoff | redaction recall | redaction precision | frames fully covered | false-alarm frames (negatives) | recall IoU0.5 | precision IoU0.5 |
|---|---|---|---|---|---|---|---|
| YuNet 2023mar (OpenCV) at 0.9 (client setting) | 0.9 | 27.6% | 100.0% | 25.9% | 0.0% | 27.5% | 99.5% |
| YuNet 2023mar (OpenCV) | 0.8 | 50.0% | 98.3% | 47.3% | 0.9% | 48.2% | 95.4% |
| BlazeFace short-range | 0.7 | 25.6% | 95.7% | 29.0% | 0.9% | 22.7% | 87.1% |
| BlazeFace full-range | 0.6 | 38.9% | 97.9% | 38.3% | 0.9% | 34.9% | 86.6% |
| Faster R-CNN, plain faces (before) | 0.94 | 44.2% | 95.3% | 44.2% | 2.5% | 41.5% | 90.7% |
| Faster R-CNN, control: +plain & negatives | 0.75 | 46.1% | 96.1% | 45.9% | 1.1% | 43.0% | 90.5% |
| Faster R-CNN, +reflection composites (500 steps) | 0.65 | 61.7% | 93.9% | 61.3% | 4.3% | 57.9% | 88.8% |
| Faster R-CNN, +reflection composites, 1,500 steps | 0.75 | 64.6% | 95.0% | 64.4% | 3.9% | 60.2% | 89.4% |
| Faster R-CNN, +reflection composites + hard negatives, 1,500 steps | 0.65 | 68.0% | 93.7% | 66.9% | 5.0% | 63.6% | 87.5% |

## Comparison at equal precision (cutoff chosen on EVAL itself; for comparison only, not a deployable number)

| detector | cutoff | redaction recall | redaction precision |
|---|---|---|---|
| YuNet 2023mar (OpenCV) | 0.7 | 55.1% | 96.3% |
| BlazeFace short-range | 0.7 | 25.6% | 95.7% |
| BlazeFace full-range | 0.6 | 38.9% | 97.9% |
| Faster R-CNN, plain faces (before) | 0.94 | 44.2% | 95.3% |
| Faster R-CNN, control: +plain & negatives | 0.7 | 46.9% | 95.1% |
| Faster R-CNN, +reflection composites (500 steps) | 0.75 | 58.0% | 96.0% |
| Faster R-CNN, +reflection composites, 1,500 steps | 0.8 | 63.3% | 95.8% |
| Faster R-CNN, +reflection composites + hard negatives, 1,500 steps | 0.7 | 66.5% | 95.5% |

## Redaction recall by family (shower = held out of training)

| detector | mirror | appliance | shower | photo | screen | plain |
|---|---|---|---|---|---|---|
| YuNet 2023mar (OpenCV) at 0.9 | 48.1% | 18.4% | 6.1% | 47.1% | 4.5% | 50.5% |
| YuNet 2023mar (OpenCV) at 0.8 | 76.0% | 34.4% | 19.1% | 80.7% | 25.0% | 76.3% |
| BlazeFace short-range at 0.7 | 45.2% | 14.4% | 25.2% | 16.8% | 1.8% | 59.1% |
| BlazeFace full-range at 0.6 | 57.7% | 35.2% | 27.5% | 46.2% | 8.9% | 65.6% |
| Faster R-CNN, plain faces (before) at 0.94 | 73.1% | 30.4% | 17.6% | 58.0% | 14.3% | 86.0% |
| Faster R-CNN, control: +plain & negatives at 0.75 | 73.1% | 34.4% | 16.8% | 63.9% | 16.1% | 86.0% |
| Faster R-CNN, +reflection composites (500 steps) at 0.65 | 81.7% | 58.4% | 28.2% | 72.3% | 51.8% | 89.2% |
| Faster R-CNN, +reflection composites, 1,500 steps at 0.75 | 81.7% | 68.0% | 31.3% | 73.9% | 52.7% | 90.3% |
| Faster R-CNN, +reflection composites + hard negatives, 1,500 steps at 0.65 | 84.6% | 72.0% | 32.1% | 77.3% | 61.6% | 90.3% |

## False-alarm frames on hard negatives by family

| detector | mirror | appliance | shower | photo | screen | plain |
|---|---|---|---|---|---|---|
| YuNet 2023mar (OpenCV) at 0.8 | 1.4% | 0.0% | 0.0% | 1.4% | 0.0% | 2.8% |
| BlazeFace short-range at 0.7 | 1.4% | 2.7% | 1.4% | 0.0% | 0.0% | 0.0% |
| BlazeFace full-range at 0.6 | 1.4% | 1.4% | 1.4% | 1.4% | 0.0% | 0.0% |
| Faster R-CNN, plain faces (before) at 0.94 | 1.4% | 2.7% | 4.1% | 1.4% | 2.7% | 2.8% |
| Faster R-CNN, control: +plain & negatives at 0.75 | 1.4% | 0.0% | 0.0% | 2.7% | 1.4% | 1.4% |
| Faster R-CNN, +reflection composites (500 steps) at 0.65 | 9.6% | 8.1% | 0.0% | 4.1% | 1.4% | 2.8% |
| Faster R-CNN, +reflection composites, 1,500 steps at 0.75 | 8.2% | 4.1% | 2.7% | 4.1% | 1.4% | 2.8% |
| Faster R-CNN, +reflection composites + hard negatives, 1,500 steps at 0.65 | 13.7% | 5.4% | 4.1% | 2.7% | 1.4% | 2.8% |
