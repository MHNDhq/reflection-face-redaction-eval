"""Fine-tune torchvision Faster R-CNN (MobileNetV3-Large FPN, COCO weights) as a 1-class face detector.

Stages (all data from the Open Images VALIDATION split; eval uses the TEST split):
  A        COCO weights -> plain face photos only                       ("before": generic face detector)
  B        A -> reflection composites (mirror/appliance/photo/screen) + plain photos + hard negatives
  control  A -> plain photos + hard negatives, no composites, same steps as B (isolates the composite effect)
The shower-glass family is never in training.

Usage: .venv/bin/python scripts/train.py --stage A|B|control [--steps N]
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import detectors as D

ROOT = Path(__file__).resolve().parents[1]


def load_set(name, positives_only=False):
    ann = json.load(open(ROOT / "data" / name / "annotations.json"))
    items = []
    for r in ann["images"]:
        if positives_only and r["source_role"] != "face":
            continue
        boxes = r["boxes"] + [b for b in r["ignore"] if b[2] - b[0] >= 8]  # partial/small real faces still train as faces
        items.append((ROOT / "data" / name / "images" / r["file"], boxes))
    return items


def to_target(boxes, flip, w, dev):
    b = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    if flip and len(b):
        b = torch.stack([w - b[:, 2], b[:, 1], w - b[:, 0], b[:, 3]], 1)
    return {"boxes": b.to(dev), "labels": torch.ones(len(b), dtype=torch.int64, device=dev)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["A", "B", "control", "HN"])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--tag", default="", help="suffix for the saved weights and log, e.g. _long")
    a = ap.parse_args()
    random.seed(0)
    torch.manual_seed(0)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    if a.stage == "A":
        items = load_set("train_plain", positives_only=True)
        model = D.build_frcnn(pretrained=True)
        steps, lr = a.steps or 900, 0.01
    else:
        model = D.build_frcnn(pretrained=False)
        model.load_state_dict(torch.load(ROOT / "models" / "ft_A.pt", map_location="cpu", weights_only=True))
        items = load_set("train_plain")
        if a.stage in ("B", "HN"):
            items += load_set("train_comp")
        if a.stage == "HN":  # round 3: + hard negatives from a disjoint validation pool (data/train_hn)
            items += load_set("train_hn")
        steps, lr = a.steps or 500, 0.005
    model.to(dev).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 50) * 0.5 * (1 + np.cos(np.pi * s / steps)))
    t0 = time.time()
    log = []
    for step in range(steps):
        batch = random.sample(items, a.bs)
        imgs, tgts = [], []
        for path, boxes in batch:
            arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255
            flip = random.random() < 0.5
            if flip:
                arr = arr[:, ::-1]
            arr = np.clip(arr * random.uniform(0.75, 1.25), 0, 1)
            imgs.append(torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).to(dev))
            tgts.append(to_target(boxes, flip, arr.shape[1], dev))
        losses = model(imgs, tgts)
        loss = sum(losses.values())
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 50 == 0 or step == steps - 1:
            rec = {"step": step, "loss": round(float(loss), 4), "lr": round(sched.get_last_lr()[0], 5),
                   "elapsed_s": round(time.time() - t0, 1)}
            log.append(rec)
            print(json.dumps(rec), flush=True)
    torch.save(model.state_dict(), ROOT / "models" / f"ft_{a.stage}{a.tag}.pt")
    json.dump({"stage": a.stage, "steps": steps, "batch": a.bs, "lr": lr, "n_items": len(items),
               "train_seconds": round(time.time() - t0, 1), "log": log},
              open(ROOT / "models" / f"ft_{a.stage}{a.tag}_trainlog.json", "w"), indent=1)


if __name__ == "__main__":
    main()
