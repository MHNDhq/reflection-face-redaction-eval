"""Synthetic reflection / domestic-surface composites with exact box bookkeeping.

SYNTHETIC: every image this module writes is a composite of Open Images (CC BY 2.0) photos.
None of it is real head-mounted footage.

Families
  mirror     flipped, near-full opacity, dimmed, specular sheen, dark frame
  appliance  flipped, low opacity over a dark door panel, tint, blur, streaks, optional mesh (microwave)
  shower     flipped, fogged (heavy blur + haze), droplets, ghost double image   <- HELD OUT of training
  photo      unflipped print stuck on a surface (white border), full opacity, small
  screen     unflipped, inside a black bezel, blue cast, scanlines, glare
  plain      the source image resized to the canvas (no composite)

Ground truth: real-person face boxes are mapped through the same resize / flip / offset as the
pixels. A face becomes an IGNORE region (neither hit nor miss) when its visible area after
clipping and occlusion is < 60% or its width on the 960x540 canvas is < 16 px.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

W, H = 960, 540
MIN_W = 16
FAMILIES_ALL = ["mirror", "appliance", "shower", "photo", "screen", "plain"]
FAMILIES_TRAIN = ["mirror", "appliance", "photo", "screen", "plain"]  # shower held out
ROOT = Path(__file__).resolve().parents[1]


def cover(im, w, h):
    """Resize+center-crop to exactly w x h. Returns image and (scale, dx, dy) with x' = x*s - dx."""
    s = max(w / im.width, h / im.height)
    nw, nh = max(w, round(im.width * s)), max(h, round(im.height * s))
    im2 = im.resize((nw, nh), Image.BILINEAR)
    dx, dy = (nw - w) // 2, (nh - h) // 2
    return im2.crop((dx, dy, dx + w, dy + h)), (s, dx, dy)


def map_boxes(boxes_rel, src_w, src_h, s, dx, dy, pw, ph, flip, ox, oy):
    out = []
    for x0, x1, y0, y1 in boxes_rel:
        a, b = x0 * src_w * s - dx, x1 * src_w * s - dx
        c, d = y0 * src_h * s - dy, y1 * src_h * s - dy
        if flip:
            a, b = pw - b, pw - a
        out.append([a + ox, c + oy, b + ox, d + oy])
    return out


def finalize(boxes, vis_mask):
    """Split mapped boxes into scoring boxes and ignore boxes using the visibility mask."""
    keep, ign = [], []
    for x0, y0, x1, y1 in boxes:
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if area <= 0 or cx1 <= cx0 or cy1 <= cy0:
            continue
        m = vis_mask[int(cy0):int(np.ceil(cy1)), int(cx0):int(np.ceil(cx1))]
        vis = float(m.sum()) / area
        b = [round(cx0, 1), round(cy0, 1), round(cx1, 1), round(cy1, 1)]
        if vis < 0.6 or (cx1 - cx0) < MIN_W:
            if vis > 0.05:
                ign.append(b)
        else:
            keep.append(b)
    return keep, ign


def specular(arr, rng, strength):
    h, w = arr.shape[:2]
    ang = rng.uniform(0, np.pi)
    yy, xx = np.mgrid[0:h, 0:w]
    t = (np.cos(ang) * xx / w + np.sin(ang) * yy / h)
    band = np.exp(-((t - rng.uniform(0.2, 0.8)) ** 2) / (2 * rng.uniform(0.01, 0.05)))
    return np.clip(arr + strength * 255 * band[..., None], 0, 255)


def make(rng, family, src, src_boxes, bg):
    """src: PIL RGB; src_boxes: list of (xmin,xmax,ymin,ymax) relative; bg: PIL RGB."""
    canvas, _ = cover(bg, W, H)
    canvas = np.asarray(canvas, dtype=np.float32)
    vis = np.zeros((H, W), dtype=np.float32)
    meta = {"family": family}
    if family == "plain":
        im, (s, dx, dy) = cover(src, W, H)
        flip = bool(rng.random() < 0.5)
        if flip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        boxes = map_boxes(src_boxes, src.width, src.height, s, dx, dy, W, H, flip, 0, 0)
        vis[:] = 1
        out = np.asarray(im, dtype=np.float32)
        keep, ign = finalize(boxes, vis)
        return out, keep, ign, meta

    if family in ("photo", "screen"):
        pw = int(W * rng.uniform(0.2, 0.42))
    else:
        pw = int(W * rng.uniform(0.38, 0.75))
    aspect = rng.uniform(0.6, 1.4) if family != "screen" else rng.uniform(0.55, 0.75)
    ph = min(int(pw * aspect), H - 10)
    ox, oy = int(rng.integers(0, W - pw)), int(rng.integers(0, H - ph))
    flip = family in ("mirror", "appliance", "shower")
    im, (s, dx, dy) = cover(src, pw, ph)
    if flip:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    boxes = map_boxes(src_boxes, src.width, src.height, s, dx, dy, pw, ph, flip, ox, oy)
    p = np.asarray(im, dtype=np.float32)
    region = canvas[oy:oy + ph, ox:ox + pw].copy()

    if family == "mirror":
        gain = rng.uniform(0.55, 1.0)
        p = p * gain
        p = np.asarray(Image.fromarray(p.astype(np.uint8)).filter(ImageFilter.GaussianBlur(rng.uniform(0, 1.0))), np.float32)
        p = specular(p, rng, rng.uniform(0.0, 0.35))
        alpha = rng.uniform(0.8, 0.97)
        mix = alpha * p + (1 - alpha) * region
        meta.update(gain=round(gain, 2), alpha=round(alpha, 2))
    elif family == "appliance":
        door = np.full_like(region, rng.uniform(15, 70)) * rng.uniform(0.8, 1.2, size=3)[None, None, :]
        tint = rng.uniform(0.6, 1.1, size=3)
        pb = np.asarray(Image.fromarray(p.astype(np.uint8)).filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 2.2))), np.float32)
        alpha = rng.uniform(0.25, 0.6)
        mix = alpha * pb * tint + (1 - alpha) * door
        mix = specular(mix, rng, rng.uniform(0.05, 0.4))
        if rng.random() < 0.4:  # microwave mesh
            step = int(rng.integers(4, 8))
            mix[::step, :, :] *= 0.55
            mix[:, ::step, :] *= 0.55
        meta.update(alpha=round(alpha, 2))
    elif family == "shower":
        pb = np.asarray(Image.fromarray(p.astype(np.uint8)).filter(ImageFilter.GaussianBlur(rng.uniform(2.0, 4.5))), np.float32)
        shift = int(rng.integers(5, 16))
        ghost = np.roll(pb, shift, axis=1)
        alpha = rng.uniform(0.35, 0.65)
        haze = rng.uniform(150, 220)
        mix = alpha * (0.75 * pb + 0.25 * ghost) + (1 - alpha) * (0.5 * region + 0.5 * haze)
        dimg = Image.fromarray(np.clip(mix, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(dimg)
        for _ in range(int(rng.integers(40, 160))):
            x, y, r = rng.integers(0, pw), rng.integers(0, ph), rng.integers(1, 5)
            d.ellipse([x - r, y - r, x + r, y + r], fill=(235, 240, 245))
        mix = np.asarray(dimg, np.float32)
        meta.update(alpha=round(alpha, 2), ghost_px=shift)
    elif family == "photo":
        mix = p
    else:  # screen
        mix = p * np.array([0.85, 0.95, 1.1]) * rng.uniform(0.7, 1.0)
        mix[::3, :, :] *= 0.8
        mix = specular(mix, rng, rng.uniform(0.0, 0.3))

    canvas[oy:oy + ph, ox:ox + pw] = np.clip(mix, 0, 255)
    vis[oy:oy + ph, ox:ox + pw] = 1
    if family == "photo":
        b = int(rng.integers(4, 12))
        canvas[max(0, oy - b):oy, max(0, ox - b):min(W, ox + pw + b)] = 245
        canvas[oy + ph:min(H, oy + ph + b), max(0, ox - b):min(W, ox + pw + b)] = 245
        canvas[max(0, oy - b):min(H, oy + ph + b), max(0, ox - b):ox] = 245
        canvas[max(0, oy - b):min(H, oy + ph + b), ox + pw:min(W, ox + pw + b)] = 245
    if family in ("screen", "mirror"):
        b = int(rng.integers(5, 14))
        col = 10 if family == "screen" else rng.uniform(20, 90)
        canvas[max(0, oy - b):oy, max(0, ox - b):min(W, ox + pw + b)] = col
        canvas[oy + ph:min(H, oy + ph + b), max(0, ox - b):min(W, ox + pw + b)] = col
        canvas[max(0, oy - b):min(H, oy + ph + b), max(0, ox - b):ox] = col
        canvas[max(0, oy - b):min(H, oy + ph + b), ox + pw:min(W, ox + pw + b)] = col
    if rng.random() < 0.3:  # partial occluder (a towel, a door edge, a hand)
        cw, chh = int(pw * rng.uniform(0.15, 0.4)), int(ph * rng.uniform(0.2, 0.6))
        cx, cy = ox + int(rng.integers(0, max(1, pw - cw))), oy + int(rng.integers(0, max(1, ph - chh)))
        canvas[cy:cy + chh, cx:cx + cw] = rng.uniform(30, 230, size=3)
        vis[cy:cy + chh, cx:cx + cw] = 0
        meta["occluder"] = [cx, cy, cx + cw, cy + chh]
    keep, ign = finalize(boxes, vis)
    meta["panel"] = [ox, oy, ox + pw, oy + ph]
    return canvas, keep, ign, meta


def build(split, out_dir, families, per_source, seed, subset=None, jpeg_q=85):
    """Generate composites for one split. subset: optional set of ImageIDs to use as content sources."""
    rng = np.random.default_rng(seed)
    src = pd.read_csv(ROOT / "data" / "sources.csv")
    src = src[src.split == split]
    fb = pd.read_csv(ROOT / "data" / "source_boxes.csv")
    img_dir = ROOT / "data" / "images" / split
    have = {p.stem for p in img_dir.glob("*.jpg")}
    src = src[src.ImageID.isin(have)]
    rejected = set((ROOT / "data" / "bg_rejected.txt").read_text().split())
    bgs = [i for i in src[src.role == "background"].ImageID if i not in rejected]
    content = src[src.role.isin(["face", "negative"])]
    if subset is not None:
        content = content[content.ImageID.isin(subset)]
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    recs = []
    k = 0
    for row in content.itertuples():
        im = Image.open(img_dir / f"{row.ImageID}.jpg").convert("RGB")
        boxes = []
        if row.role == "face":
            f = fb[fb.ImageID == row.ImageID]
            boxes = f[["XMin", "XMax", "YMin", "YMax"]].values.tolist()
        for j in range(per_source):
            fam = families[(k + j) % len(families)]
            bg_id = bgs[int(rng.integers(0, len(bgs)))]
            bg = Image.open(img_dir / f"{bg_id}.jpg").convert("RGB")
            arr, keep, ign, meta = make(rng, fam, im, boxes, bg)
            name = f"{split}_{row.ImageID}_{j}_{fam}.jpg"
            Image.fromarray(arr.astype(np.uint8)).save(out_dir / "images" / name, quality=jpeg_q)
            recs.append({"file": name, "family": fam, "label": "positive" if row.role == "face" and keep else
                         ("positive_ignore_only" if row.role == "face" else "negative"),
                         "source_role": row.role, "source_category": row.category if isinstance(row.category, str) else "",
                         "content_id": row.ImageID, "background_id": bg_id if fam != "plain" else "",
                         "boxes": keep, "ignore": ign, "meta": meta, "synthetic": True})
        k += 1
    with open(out_dir / "annotations.json", "w") as fh:
        json.dump({"note": "SYNTHETIC composites of Open Images CC BY 2.0 photos; not real head-mounted footage.",
                   "split": split, "families": families, "seed": seed, "images": recs}, fh, indent=1)
    return recs


def main():
    src = pd.read_csv(ROOT / "data" / "sources.csv")
    val = src[(src.split == "validation") & src.role.isin(["face", "negative"])].ImageID.tolist()
    rng = np.random.default_rng(7)
    rng.shuffle(val)
    tune = set(val[: len(val) // 10])
    train = set(val[len(val) // 10:])
    r = build("test", ROOT / "data" / "eval", FAMILIES_ALL, 2, seed=101)
    print("eval", len(r))
    r = build("validation", ROOT / "data" / "tune", FAMILIES_TRAIN, 2, seed=202, subset=tune)
    print("tune", len(r))
    r = build("validation", ROOT / "data" / "train_comp", FAMILIES_TRAIN, 3, seed=303, subset=train)
    print("train composites", len(r))
    r = build("validation", ROOT / "data" / "train_plain", ["plain"], 1, seed=404, subset=train)
    print("train plain", len(r))


if __name__ == "__main__":
    sys.exit(main())
