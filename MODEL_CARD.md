# Model card: ft_B (reflection-tuned face detector), with ft_A and ft_control

The weight files are not published by default (see README). This card describes the models `./reproduce.sh` trains.

SYNTHETIC-DATA EXPERIMENT. Not validated on real head-mounted footage. Not for production use as is.

## Model

- Architecture: torchvision `fasterrcnn_mobilenet_v3_large_fpn`, one class ("face"), input resized so the
  short side is 540 px and the long side at most 960 px.
- Initialised from torchvision COCO weights (`FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1`).
- ft_A: fine-tuned on plain Open Images face photos. ft_B: ft_A fine-tuned on reflection composites, plain
  photos and hard negatives. ft_control: ft_A fine-tuned on plain photos and hard negatives only.
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

See `README.md` and `results/metrics.md`. Numbers are on synthetic composites.

## Licence of the weights

- The fine-tuning code in this repo is MIT; the fine-tuning data is CC BY 2.0 / CC BY 4.0.
- The starting weights were trained on COCO (images under the Flickr Terms of Use) on an ImageNet backbone.
  Whether those dataset terms carry into derived weights is an open question; see `LICENCES.md`.
