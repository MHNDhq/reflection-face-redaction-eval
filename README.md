# reflection-face-redaction-eval

Public repo: https://github.com/MHNDhq/reflection-face-redaction-eval

A small, checkable experiment for one failure class in first-person video redaction: faces seen in
mirrors, appliance doors, shower glass, photos and screens, next to face-like things that are not
faces (toys, dolls, sculptures, cartoon prints, packaging, screens without people).

> **SYNTHETIC DATA.** Every eval and training image here is a composite built from Open Images photos
> (CC BY 2.0). None of it is real head-mounted footage, and no number here is a claim about how any
> detector performs on real footage. Treat the numbers as a method demonstration, not a benchmark.

Built in October 2026 by Muhannad Ahmad for an application to a face-redaction trial. It's the first
face detector he has fine-tuned.

## Result (eval3: fresh, pre-registered, read once)

eval3: 1,120 synthetic composites from Open Images TEST-split photos never used before (300 face photos,
260 hard negatives incl. 48 sculptures, 160 new backgrounds; 46% no-face frames). Every cutoff was picked
on the tune sets (2,550 images) by the rule committed before this read: highest tune recall among cutoffs
whose tune precision has a cluster-bootstrap 5th percentile of at least 95%. The "after" model was also
chosen on tune. Confidence intervals: 95%, the wider of a bootstrap over source photos and one over
background photos (`scripts/intervals.py` -> `results/intervals.json`).

| detector | cutoff | redaction recall | redaction precision | false-alarm frames |
|---|---|---|---|---|
| YuNet 2023mar at 0.9 (the client's current setting) | 0.9 | 26.1% | 99.0% | 0.4% |
| YuNet 2023mar, cutoff from tune | 0.8 | 45.9% (41.1 to 50.5) | 95.7% (92.5 to 98.1) | 1.7% |
| Faster R-CNN ft_A, plain faces (before) | 0.94 | 43.8% (39.4 to 48.3) | 96.5% (94.3 to 98.3) | 1.5% |
| Faster R-CNN ft_control, + plain photos and hard negatives | 0.75 | 47.4% | 94.5% | 1.3% |
| Faster R-CNN ft_HN2, + reflection composites + hard negatives (after) | 0.6 | 65.1% (60.6 to 69.8) | 95.4% (93.5 to 97.1) | 2.3% |

Read honestly:
- On this clean read the fine-tuned model clears a 95% precision bar at its tune-picked cutoff, and covers
  about 19 points more faces than YuNet with its own tuned cutoff, at about the same precision. The
  precision interval (93.5 to 97.1%) includes values below 95%, so "at the bar", not "safely above it".
- Not robust yet: the SAME model and cutoff give 94.1% precision on eval2 and 89.6% on the original eval
  set, which has more verified hard negatives (82 sculptures, 80 toys). Those two sets were read before
  this model existed (eval three times, eval2 once), so they are reported, not used to choose anything.
- Shower glass, never in training: YuNet at 0.9 0.7%, ft_A 18.5%, ft_HN2 26.7%. Screens with people:
  YuNet at 0.9 0.0%, ft_HN2 60.2%.
- One training seed per stage; run-to-run noise is not measured.
- Tune composites reuse the training BACKGROUND photos (eval2 and eval3 do not), so tune precision runs
  optimistic and the cutoffs it picks lean low. eval3 itself has its own backgrounds, so its read is fair.
- The tune-side bootstrap resamples source photos only; backgrounds are shared across those clusters, so
  the rule's bound is somewhat too generous.
- No detector is near 99% recall.

`results/sample_redactions.jpg` shows four eval3 frames, picked by position (the fourth face frame of each
family), not by result, before and after ft_HN2 at 0.6 plus the fill: 2 of their 5 scoring faces are
covered (mirror 1 of 1, appliance 1 of 2, shower 0 of 1, screen 0 of 1). Full tables:
`results/metrics_eval3.md`, `results/metrics_eval2.md`, `results/metrics.md` (the original eval).

## How it got here (four rounds, every rule committed before the read it governs)

| round | eval set (reads) | rule | "after" model | recall | precision |
|---|---|---|---|---|---|
| 1 | eval (1st) | point precision >= 95% on a 240-image tune split | ft_B | 65.8% | 90.2% |
| 2 | eval (3rd) | cluster bootstrap on tune + tune2 (6.5% no-face, no sculptures) | ft_B_long | 71.6% | 87.6% |
| 3 | eval2 (1st) | same, tune + tune2 + tune3 (34% no-face) | ft_HN | 68.0% | 93.7% |
| 4 | eval3 (1st) | same | ft_HN2 | 65.1% | 95.4% |

A per-box Wilson-bound rule was also tried between rounds 1 and 2, chosen after the first eval read; it is
superseded and not counted as a round. What moved the result: a tune set whose no-face share matches the
eval sets (precision depends on that share), and hard-negative training aimed at mirror-backed false alarms.

## Pre-registration (2026-10-23T18:22Z, committed before the step it governs)

Written before any detector was run on tune2 and before eval was read again:
- Tune set: tune (240 images) + tune2 (1,560 composites from 520 new validation-split photos, none used
  in training, tune or eval; `scripts/select_tune2.py`, `scripts/composite.py --tune2`).
- Cutoff rule: highest redaction recall on tune + tune2 among grid cutoffs whose tune precision has a
  cluster-bootstrap 5th percentile of at least 95% (resampling source photos, 1,000 draws, seed 20261024).
- Grid: unchanged from the previous commit.
- Model: ft_B (500 steps) vs ft_B_long (1,500 steps, same data): the one with the higher tune recall at
  its own rule cutoff is reported as "after". Eval is read once, after that choice.

## Pre-registration, round 3 (2026-10-23T19:31Z, committed before the step it governs)

Round 2 read: ft_B_long, chosen on tune, gave 71.6% recall at 87.6% precision on eval, below 95%.
Cause found on tune and eval composition, not on detector output: tune + tune2 had 6.5% no-face frames
and no sculptures; eval has 40%. Round 3, written before any detector runs on the new sets:
- eval2: a FRESH test-split set (1,040 composites: 300 face photos, 220 negatives in 8 categories incl.
  sculptures, 160 new backgrounds; 42% no-face frames). Negatives need only "no person box or label",
  not the verified label (the test split ran out of those); an unlabeled face there counts as a false
  alarm, so this errs toward LOWER precision. The original eval set has now been read three times and is
  reported for continuity only.
- tune3: 250 new validation negatives (750 composites). Tune = tune + tune2 + tune3 = 2,550 images, 34%
  no-face frames.
- ft_B_hn: ft_A + plain + reflection composites + 179 new validation hard negatives (data/train_hn),
  1,500 steps, disjoint from tune3 and eval2.
- Rule and grid: unchanged (cluster-bootstrap 5th percentile of tune precision >= 95%).
- Model: ft_B_long vs ft_B_hn, the higher tune recall at its rule cutoff is "after". eval2 is read once.

## Pre-registration, round 4 (2026-10-23T20:03Z, committed before the step it governs)

Round 3 read (eval2, once): ft_HN, chosen on tune, 68.0% recall at 93.7% precision, 1.3 points under 95%;
the false alarms concentrate on mirror-backed no-face frames (13.7% of them). Round 4, written before
training ft_HN2 and before any detector runs on eval3:
- train_hn2: the 179 training-only validation negatives (data/train_hn) re-composited 6 times each with
  mostly mirror and appliance EFFECTS (1,074 images; the background photo is drawn at random, as everywhere). Validation negatives are otherwise used up.
- ft_HN2: ft_A + plain + reflection composites + train_hn + train_hn2, 2,000 steps.
- eval3: a FOURTH, fresh test-split set (1,120 composites: 300 face photos, 260 negatives incl. 48
  sculptures, 160 new backgrounds; 46% no-face frames). Same negative rule as eval2.
- Tune set, rule and grid: unchanged from round 3.
- Model: ft_HN vs ft_HN2, the higher tune recall at its rule cutoff is "after". eval3 is read once.
  eval and eval2 are reported for continuity; eval2 has been read once, eval three times.

### How the numbers are scored

- Redaction recall: a real face counts as covered when at least 70% of its box sits inside the area that
  would be filled (the detections, each grown by a 25% margin). This is the property redaction needs.
- Redaction precision: share of detections that touch a real face (center inside a face box, or IoU >= 0.1).
  The rest are false alarms.
- Frames fully covered: share of face frames where every scoring face is covered.
- False-alarm frames: share of hard-negative frames (no real face) with at least one false alarm.
- Strict recall and precision at IoU >= 0.5 are reported too, for comparison with detector papers.
- Faces narrower than 16 px or less than 60% visible are "ignore" regions: neither a hit nor a miss.
- Cutoffs: see the cutoff rule above (a confidence bound on tune precision, chosen after a first look at
  eval). YuNet is also reported at 0.9.

## The data

- Sources: Open Images V7. Eval images come only from the TEST split; training and tune images come only
  from the VALIDATION split, so no source photo is in both. Only rows whose licence column is CC BY 2.0
  are used. `data/sources.csv` lists every source image with its licence, author, author page and Flickr page.
- Faces: Open Images human-drawn "Human face" boxes; images with group or depiction face boxes are
  skipped, so every positive is a real person's face.
- Hard negatives: doll, teddy bear, toy, sculpture, snowman, screen, packaging and poster images with no
  person boxes. The original eval set also needs a human-verified "no face" label; eval2 and eval3 do not
  (the test split ran out of verified ones), so an unlabeled face there counts as a false alarm and
  lowers precision. Training and tune negatives only need no person boxes or labels.
- Pools added after round 1 (`scripts/select_tune2.py`, `scripts/select_pool.py`, `pool` column in
  `data/sources.csv`): tune2 and tune3 (validation, tune only), train_hn (validation, training only),
  eval2 and eval3 (test, with their own backgrounds). No photo is in two pools.
- Backgrounds: fridge, microwave, oven, mirror, shower, bath and sink photos with no person boxes. Any
  background where YuNet or BlazeFace fired at 0.5 was dropped (85 of 659 across all pools). Detectors never filtered the
  hard negatives, so the hard ones stayed in.
- Composite families (`scripts/composite.py`): mirror (flipped, dimmed, sheen, frame); appliance door
  (flipped, 25 to 60% opacity over a dark panel, tint, blur, optional microwave mesh); shower glass
  (fog, haze, droplets, ghost double image); photo (a print stuck on a surface); screen (bezel, blue
  cast, scanlines, glare); plain (the photo itself). Random occluders on 30% of composites.
- Per the client's definition, a real person in a photo or on a screen IS a face; a toy, doll,
  cartoon or artwork is NOT. Photo and screen composites of real faces are therefore positives, and of
  toys and prints are negatives.
- The shower-glass family is never used in training. It measures how far the fine-tune carries to a
  reflection type it did not see.

## The fine-tune

- Base: torchvision `fasterrcnn_mobilenet_v3_large_fpn`, COCO weights, head replaced for one class.
- Stage A (before): plain face photos only. A generic face detector.
- Stage B (after): A, then plain photos + hard negatives + reflection composites (no shower).
- Control: A, then plain photos + hard negatives, no composites, same steps and learning rate as B. The
  gap between B and the control is what the reflection composites added; the gap between A and the
  control is what the hard negatives and extra steps added.
- B_long: stage B for 1,500 steps. HN: B data + train_hn hard negatives, 1,500 steps. HN2 (the reported
  "after"): HN data + train_hn2 (the same negatives re-composited with mostly mirror and appliance effects),
  2,000 steps. Each was chosen against the previous one on tune only.
- Training logs with loss, steps and seconds: `models/ft_*_trainlog.json`.
- The fine-tuned weight files (`models/ft_*.pt`) are NOT in this repo; they are available on request. The reason is
  the licence: the base model's lineage (torchvision COCO weights on an ImageNet backbone) is unresolved, see
  `LICENCES.md`. The training logs are here, and `./reproduce.sh` retrains the weights.

## Redaction

`scripts/redact.py`, tested in `tests/test_redaction.py`.

- Solid fill, not blur. The pixels inside a filled box are one constant colour, so the output carries no
  information about the face: a test redacts two different random frames under the same boxes and checks
  the boxes come out byte-identical.
- Why not blur: `scripts/blur_vs_fill.py` re-runs YuNet on faces after each redaction. On 78 faces that YuNet found in clean "plain" eval frames (`.venv/bin/python scripts/blur_vs_fill.py`):
YuNet still found 75 after a light Gaussian blur (sigma 3), 39 after an 8-pixel mosaic, 2 after a heavy blur
(sigma = box width / 6), and 0 after the solid fill. A face a detector can still find is a face that is still there.
- Rest of the frame untouched: a test checks that every pixel outside the filled boxes is byte-identical
  to the source, then encodes the frames with FFV1 (lossless), decodes them, and checks again. This proves
  the redaction step changes only pixels inside the padded face boxes. It does not hold after a lossy
  re-encode (HEVC, H.264): that changes pixels everywhere, so the pipeline's final encode settings decide
  how close the output stays to the source.
- Time: every box found at frame t is also applied from t-8 to t+8, and a face found at t and again at
  t+g (g up to 30, so up to 29 missed frames, overlapping boxes) is bridged with interpolated boxes. Tested: a hit at frame 100
  keeps 92 to 108 filled; hits at 100 and 125 keep 100 to 125 filled; a 29-frame miss is bridged and a
  30-frame miss is not; a jittered box between the hits does not cancel the bridge (tested at every offset and for coverage).

## Throughput and cost

Measured on an Apple M5 (10-core GPU, 24 GB) with `.venv/bin/python scripts/throughput.py` (results/throughput.json):

| step | input | frames per second |
|---|---|---|
| ft_B Faster R-CNN, MPS GPU, batch 8 (same architecture as ft_HN2) | 960x540 | 28.2 |
| ft_B Faster R-CNN, MPS GPU, batch 8 | 1920x1080 (the model resizes to 960x540 inside) | 18.8 |
| YuNet, CPU (OpenCV) | 960x540 | 155.9 |
| YuNet, CPU (OpenCV) | 1920x1080 | 19.7 |
| BlazeFace full-range, CPU (MediaPipe, 128x128 inside) | 960x540 | 660.1 |
| 4K HEVC decode, VideoToolbox | 3840x2160 synthetic test pattern | 159.9 |
| 4K HEVC decode, software | 3840x2160 synthetic test pattern | 251.2 |

Cost per video-hour, as an ESTIMATE (confidence: low):
- One video-hour at 30 fps is 108,000 frames. ft_B at 28.2 fps needs about 1.06 hours of M5 GPU time per video-hour.
- Lambda's on-demand price for one NVIDIA A10 was $1.29 per GPU-hour (read from lambda.ai/pricing on 2026-10-23).
  If an A10 ran this model only as fast as the M5, the detector would cost about $1.37 per video-hour. That
  speed was NOT measured on an A10; a real benchmark on the target GPU replaces this line. At 1920x1080 input
  (18.8 fps, resized to 960x540 inside), the same estimate is about $2.06 per video-hour.
- Decode is not the bottleneck on this machine: 4K HEVC decoded at 160 to 251 fps on a test pattern. Real
  GoPro footage will decode slower than a test pattern; that was not measured.
- Running the detector on every second frame and relying on the +/-8 frame padding would halve detector
  cost; its effect on recall was not measured.

## Licences

See `LICENCES.md` for the audit with source URLs. Short version:
- YuNet: MIT model files, trained on WIDER FACE (CC BY-NC-ND). BlazeFace: Apache 2.0, consented in-house
  data. torchvision: BSD-3 code; COCO and ImageNet pretraining behind the weights. InsightFace pretrained
  models: non-commercial research only (rejected). Ultralytics: AGPL-3.0 or enterprise (rejected).
- Whether weights trained on non-commercial data inherit the restriction is an open question for counsel.
- This repo's code: MIT. Source images: CC BY 2.0, attribution in `data/sources.csv`. Boxes: CC BY 4.0 (Google).

## Limits, stated plainly

- Synthetic composites, not real GoPro frames: no wide-angle distortion, no motion blur, no 4K HEVC
  artefacts, no wearer with a head-mounted camera in the mirror.
- Open Images face boxes are drawn by people but can miss small faces; a detector that finds an unboxed
  face is scored as a false alarm.
- One training seed, one run per stage. Differences of a few points are within run-to-run noise that was not measured.
- The cutoff rule is a choice made after a first look at eval; the client's own cutoff would come from their
  answer key's hold-out, not from this set.
- "Mirror" backgrounds in Open Images include car mirrors; they are not all domestic.

## Known method limits (found in a code review on 2026-10-23, not yet fixed)

Every number above was measured under these rules. They are listed so you can weigh the numbers.

- Ground-truth boxes are clipped to the 960x540 canvas, not to the composite panel. In the eval set, 125 of 501 scoring boxes in panel families extend past the panel edge, 39 of them by more than 20% of their area. A detector that boxes only the visible face scores lower than it should.
- Faces 5% to 60% visible are "ignore" regions in scoring, but `train.py` trains ignore regions 8 px or wider as faces. That is label noise and may explain part of the fine-tune's higher false-alarm rate.
- Matching pairs each detection with its single best ground truth, not with the best unmatched one (the VOC/COCO rule). With two adjacent faces this can undercount recall.
- `interpolate_gaps` bridges a miss only when the face box after the gap still overlaps the box before it. A face that moves far during a long miss is not bridged; padding (8 frames each side) still applies.
- The `ignore` column that `select_sources.py` writes for very small faces is not read by `composite.py`.
- `download_sources.py` does not retry a throttled download, so one image can go missing and shift the training draws.

Fixed in the same review: a box between two hits no longer cancels the gap bridge (a jittered box at t+1, then at t+2, used to leave mid-gap frames unredacted). The bridge is now a PROTOTYPE that favours privacy over cost: each hit bridges to every overlapping hit within 30 frames, and an interpolated box is skipped only when an original detection on that frame covers its whole filled area, so skipping never loses a pixel. Exact copies are removed. Boxes may be lists, arrays or scored 5-tuples; only the first four values are used. The no-loss rule assumes the default 0.25 fill margin. Tests check face-pixel coverage: jitter at every offset and a 3-hit track with no padding, a second face beside the gap with padding; curved motion between hits is not yet tested. Cost, measured (max boxes on one frame after padding): a static face 1 when seen on every frame or on every 3rd; a face seen on every frame with 2 px detector jitter about 7,000 (6,977 on 600 frames); on a sine path 6,146. `interpolate_gaps` on 10 static faces over 600 frames takes 37.9 s in pure Python. That is too slow for production: a track linker that bridges only real gaps is the trial-grade replacement. The per-video-hour estimate in the throughput section counts the detector only. `plan()` is a tested library function; no script in this repo runs it over a video yet. `expand()` now clamps all four box edges into the frame (a box fully off-frame used to fill a band of the frame; no stored detection was off-frame, so no metric changed), and training and inference pick MPS, then CUDA, then CPU.

Logged from review round 8 (not fixed): the 0.25 margin in the no-loss check is fixed while `redact()` takes a margin argument (pass the default); `plan()` applies no score threshold, so filter stored detections by score first, as `evaluate.py` does; malformed or NaN boxes fail late, not at input; the cost figures have no script that reproduces them.

## Reproduce

`./reproduce.sh` rebuilds everything from public downloads. Single commands are listed inside it.

Each step in `reproduce.sh` was run on its own while this repo was built; the script as a whole was
syntax-checked but has not been run end to end. One validation-split image failed to download
(`data/download_missing.csv`). A rerun that fetches it shifts the random draws for the training composites,
so retrained numbers will land close to, not byte-equal with, the ones above. The eval set (test split) had
no missing images and rebuilds exactly from its seed.
