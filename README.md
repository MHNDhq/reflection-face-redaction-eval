# reflection-face-redaction-eval

Intended public repo name: `reflection-face-redaction-eval` (not yet published).

A small, checkable experiment for one failure class in first-person video redaction: faces seen in
mirrors, appliance doors, shower glass, photos and screens, next to face-like things that are not
faces (toys, dolls, sculptures, cartoon prints, packaging, screens without people).

> **SYNTHETIC DATA.** Every eval and training image here is a composite built from Open Images photos
> (CC BY 2.0). None of it is real head-mounted footage, and no number here is a claim about how any
> detector performs on real footage. Treat the numbers as a method demonstration, not a benchmark.

Built in October 2026 by Muhannad Ahmad for an application to a face-redaction trial. It's the first
face detector he has fine-tuned.

## What was measured

Eval set: 860 synthetic images (596 scoring faces in 427 face frames, 340 hard-negative frames), all
from Open Images TEST-split photos. Command: `.venv/bin/python scripts/report.py` (after `scripts/evaluate.py`).

| detector | cutoff | redaction recall | redaction precision | false-alarm frames |
|---|---|---|---|---|
| YuNet 2023mar at 0.9 (the client's current setting) | 0.9 | 30.7% | 99.5% | 0.0% |
| YuNet 2023mar, cutoff from tune split | 0.8 | 49.7% | 96.5% | 2.4% |
| BlazeFace full-range, cutoff from tune split | 0.55 | 43.8% | 93.1% | 3.5% |
| Faster R-CNN ft_A, plain faces (before) | 0.9 | 51.2% | 88.0% | 8.8% |
| Faster R-CNN ft_control, + plain photos and hard negatives | 0.7 | 52.0% | 91.2% | 5.6% |
| Faster R-CNN ft_B, + reflection composites (after) | 0.65 | 65.8% | 90.2% | 9.4% |

What moved, read honestly:
- Before to after (ft_A to ft_B, each at its tune-split cutoff): redaction recall 51.2% to 65.8%, precision 88.0% to 90.2%.
- The control run (same steps, no composites) reached 52.0% at 91.2%, so most of the recall gain came from
  the reflection composites, not from the extra steps or the hard negatives.
- At an equal precision floor (cutoff chosen on the eval set itself, for comparison only): ft_B 55.2% recall
  at 96.9% precision; YuNet 49.7% at 96.5%; ft_control 47.1% at 95.8%; ft_A 36.6% at 97.9%.
- Shower glass, never in training: YuNet at 0.9 10.6%, ft_A 29.8%, ft_control 29.8%, ft_B 43.3%.
- Screens with people: YuNet at 0.9 0.0%, ft_B 43.2%.
- No detector here meets a 95% precision bar on eval at its tune-split cutoff, and none is near 99% recall.
  The tune split comes from the same photo pool as training, so it flattered the cutoff: the three fine-tuned
  models held at least 95% precision on tune and came in at 88.0% to 91.2% on eval. ft_B also raises false alarms on plain hard-negative photos (23.2% of those frames,
  against 14.3% for the control).


Full tables: `results/metrics.md`. Raw detections: `results/eval_*.json`. Curve: `results/pr_curve.png`.

`results/sample_redactions.jpg` shows four eval frames before and after ft_B plus the fill. They were picked
by position (the fourth face frame of each family), not by result. All four still show at least one missed
face (5 of their 6 scoring faces are missed). Overall recall is 65.8%; this sample is worse than average. `results/contact_eval_positive.jpg` and
`results/contact_eval_negative.jpg` show the ground truth (green = scoring face, yellow = ignore).

### How the numbers are scored

- Redaction recall: a real face counts as covered when at least 70% of its box sits inside the area that
  would be filled (the detections, each grown by a 25% margin). This is the property redaction needs.
- Redaction precision: share of detections that touch a real face (center inside a face box, or IoU >= 0.1).
  The rest are false alarms.
- Frames fully covered: share of face frames where every scoring face is covered.
- False-alarm frames: share of hard-negative frames (no real face) with at least one false alarm.
- Strict recall and precision at IoU >= 0.5 are reported too, for comparison with detector papers.
- Faces narrower than 16 px or less than 60% visible are "ignore" regions: neither a hit nor a miss.
- Cutoffs: each detector's cutoff is the highest redaction recall with redaction precision >= 95% on the
  TUNE split, then applied unchanged to EVAL. YuNet is also reported at 0.9.

## The data

- Sources: Open Images V7. Eval images come only from the TEST split; training and tune images come only
  from the VALIDATION split, so no source photo is in both. Only rows whose licence column is CC BY 2.0
  are used. `data/sources.csv` lists every source image with its licence, author, author page and Flickr page.
- Faces: Open Images human-drawn "Human face" boxes; images with group or depiction face boxes are
  skipped, so every positive is a real person's face.
- Hard negatives (eval): doll, teddy bear, toy, sculpture, snowman, screen, packaging and poster images
  with no person boxes and a human-verified "no face" label. Training negatives only need no person
  boxes or labels.
- Backgrounds: fridge, microwave, oven, mirror, shower, bath and sink photos with no person boxes. Any
  background where YuNet or BlazeFace fired at 0.5 was dropped (45 of 339). Detectors never filtered the
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
  t+g (g up to 30 frames, overlapping boxes) is bridged with interpolated boxes. Tested: a hit at frame 100
  keeps 92 to 108 filled; hits at 100 and 125 keep 100 to 125 filled.

## Throughput and cost

Measured on an Apple M5 (10-core GPU, 24 GB) with `.venv/bin/python scripts/throughput.py` (results/throughput.json):

| step | input | frames per second |
|---|---|---|
| ft_B Faster R-CNN, MPS GPU, batch 8 | 960x540 | 28.2 |
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
  speed was NOT measured on an A10; a real benchmark on the target GPU replaces this line.
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
- The cutoff rule (95% precision on tune) is a choice; the client's own cutoff would come from their
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

Fixed in the same review: `expand()` now clamps all four box edges into the frame (a box fully off-frame used to fill a band of the frame; no stored detection was off-frame, so no metric changed), and training and inference pick MPS, then CUDA, then CPU.

## Reproduce

`./reproduce.sh` rebuilds everything from public downloads. Single commands are listed inside it.

Each step in `reproduce.sh` was run on its own while this repo was built; the script as a whole was
syntax-checked but has not been run end to end. One validation-split image failed to download
(`data/download_missing.csv`). A rerun that fetches it shifts the random draws for the training composites,
so retrained numbers will land close to, not byte-equal with, the ones above. The eval set (test split) had
no missing images and rebuilds exactly from its seed.
