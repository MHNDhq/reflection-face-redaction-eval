# Model card: ft_HN2 (reflection- and hard-negative-tuned face detector), with ft_A, ft_control, ft_B, ft_B_long, ft_HN

The weight files are not published by default (see README). This card describes the models `./reproduce.sh` trains.

SYNTHETIC-DATA EXPERIMENT. Not validated on real head-mounted footage. Not for production use as is.

## Model

- Architecture: torchvision `fasterrcnn_mobilenet_v3_large_fpn`, one class ("face"), input resized so the
  short side is 540 px and the long side at most 960 px.
- Initialised from torchvision COCO weights (`FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1`).
- ft_A: fine-tuned on plain Open Images face photos. ft_B: ft_A fine-tuned on reflection composites, plain
  photos and hard negatives. ft_control: ft_A fine-tuned on plain photos and hard negatives only.
- ft_B_long: ft_B's data, 1,500 steps. ft_HN: + train_hn hard negatives. ft_HN2 (reported "after"): + train_hn2
  (the same negatives re-composited with mostly mirror and appliance effects), 2,000 steps.
- Training settings and logs: `models/ft_*_trainlog.json` (SGD, momentum 0.9, weight decay 1e-4, batch 8,
  cosine schedule, horizontal flip and brightness jitter). One seed per stage.
- Hardware: Apple M5 (10-core GPU, 24 GB), PyTorch MPS backend.

## Intended use

- A demonstration of a method: build a reflection eval set, measure a baseline, fine-tune, re-measure,
  choose a cutoff on a held-out split. The method transfers to real footage with a real answer key; these
  weights are not expected to.

## Out of scope

- Identity recognition, tracking of named people, or any surveillance use.
- Any claim about real-world recall without evaluation on the target footage.

## Data

- Open Images V7 photos listed as CC BY 2.0, with Google's CC BY 4.0 face boxes. Training uses the
  VALIDATION split only; evaluation uses the TEST split only. Attribution list: `data/sources.csv`.
- The shower-glass composite family is held out of training.

## Results

Headline (README > Result): on eval3, a fresh pre-registered set read once, ft_HN2 at its tune-picked cutoff 0.6
covers 65.1% of faces at 95.4% precision (bootstrap range 93.5 to 97.1%). The same cutoff gives 94.1% on
eval2 (whose false alarms shaped train_hn2, so optimistic) and 89.6% on the original eval set. Tables:
`results/metrics_eval3.md`, `results/metrics_eval2.md`, `results/metrics.md`. Synthetic composites only.

## Licence of the weights

- The fine-tuning code in this repo is MIT; the fine-tuning data is CC BY 2.0 / CC BY 4.0.
- The starting weights were trained on COCO (images under the Flickr Terms of Use) on an ImageNet backbone.
  Whether those dataset terms carry into derived weights is an open question; see `LICENCES.md`.
