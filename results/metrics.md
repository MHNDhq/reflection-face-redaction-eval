# Results on the SYNTHETIC eval set (not the client's footage)

Eval: 860 composite images (test-split sources). Tune: 240 images (validation-split sources held out of training). Cutoff per detector = highest redaction recall on TUNE among cutoffs where the lower end of the two-sided 95% Wilson interval for tune precision is >= 95%, applied unchanged to EVAL. This rule and the finer cutoff grid were chosen AFTER eval had been read under the first rule (point precision >= 95% on tune), which over-fit the small tune split.

| detector | cutoff | redaction recall | redaction precision | frames fully covered | false-alarm frames (negatives) | recall IoU0.5 | precision IoU0.5 |
|---|---|---|---|---|---|---|---|
| YuNet 2023mar (OpenCV) | no cutoff met the rule | n/a | n/a | n/a | n/a | n/a | n/a |
| YuNet 2023mar (OpenCV) at 0.9 (client setting) | 0.9 | 30.7% | 99.5% | 26.7% | 0.0% | 29.5% | 98.3% |
| BlazeFace short-range | no cutoff met the rule | n/a | n/a | n/a | n/a | n/a | n/a |
| BlazeFace full-range | 0.65 | 37.1% | 99.6% | 33.5% | 0.3% | 33.9% | 91.0% |
| Faster R-CNN, plain faces (before) | 0.99 | 36.6% | 97.9% | 34.9% | 1.2% | 35.1% | 95.9% |
| Faster R-CNN, control: +plain & negatives | 0.99 | 32.4% | 99.5% | 30.2% | 0.0% | 31.2% | 98.4% |
| Faster R-CNN, +reflection composites (after) | 0.9 | 55.2% | 96.9% | 53.2% | 1.8% | 53.4% | 94.9% |

## Comparison at equal precision (cutoff chosen on EVAL itself; for comparison only, not a deployable number)

| detector | cutoff | redaction recall | redaction precision |
|---|---|---|---|
| YuNet 2023mar (OpenCV) | 0.8 | 49.7% | 96.5% |
| BlazeFace short-range | 0.7 | 26.2% | 95.6% |
| BlazeFace full-range | 0.6 | 39.9% | 97.6% |
| Faster R-CNN, plain faces (before) | 0.98 | 40.8% | 96.3% |
| Faster R-CNN, control: +plain & negatives | 0.9 | 47.1% | 95.8% |
| Faster R-CNN, +reflection composites (after) | 0.9 | 55.2% | 96.9% |

## Redaction recall by family (shower = held out of training)

| detector | mirror | appliance | shower | photo | screen | plain |
|---|---|---|---|---|---|---|
| YuNet 2023mar (OpenCV) at 0.9 | 50.5% | 22.4% | 10.6% | 55.0% | 0.0% | 40.0% |
| BlazeFace full-range at 0.65 | 52.3% | 34.6% | 35.6% | 42.0% | 3.7% | 47.4% |
| Faster R-CNN, plain faces (before) at 0.99 | 55.0% | 31.8% | 16.3% | 51.0% | 4.9% | 54.7% |
| Faster R-CNN, control: +plain & negatives at 0.99 | 49.5% | 28.0% | 10.6% | 46.0% | 1.2% | 53.7% |
| Faster R-CNN, +reflection composites (after) at 0.9 | 76.1% | 54.2% | 32.7% | 62.0% | 32.1% | 69.5% |

## False-alarm frames on hard negatives by family

| detector | mirror | appliance | shower | photo | screen | plain |
|---|---|---|---|---|---|---|
| BlazeFace full-range at 0.65 | 0.0% | 0.0% | 1.8% | 0.0% | 0.0% | 0.0% |
| Faster R-CNN, plain faces (before) at 0.99 | 0.0% | 1.8% | 1.8% | 1.7% | 0.0% | 1.8% |
| Faster R-CNN, control: +plain & negatives at 0.99 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Faster R-CNN, +reflection composites (after) at 0.9 | 0.0% | 1.8% | 1.8% | 1.7% | 0.0% | 5.4% |
