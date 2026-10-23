"""Select source images from Open Images V7 (test split -> eval, validation split -> train).

Only images whose metadata License is CC BY 2.0 and Rotation == 0 are used.
Writes data/sources.csv with per-image provenance (ImageID, split, role, licence,
author, Flickr landing URL) and data/source_boxes.csv with face boxes.

Roles:
  face        real-person face boxes (IsDepiction=0, IsGroupOf=0); tiny faces become 'ignore'
  negative    toy / doll / teddy / sculpture / screen / packaging / poster with NO face or person boxes
              and a human-verified 'Human face = absent' image label
  background  fridge / microwave / oven / mirror / shower / bath / sink with no face or person boxes
              and a human-verified 'Human face = absent' image label
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "oi_meta"
SEED = 20261023
CCBY = "https://creativecommons.org/licenses/by/2.0/"

FACE = "/m/0dzct"
PEOPLE = {"/m/01g317", "/m/04yx4", "/m/03bt1vf", "/m/01bl7v", "/m/05r655", "/m/04hgtk",
          "/m/02p0tk3", "/m/014sv8", "/m/0k0pj", "/m/0283dt1", FACE}
NEG = {"/m/0167gd": "doll", "/m/0kmg4": "teddy_bear", "/m/0138tl": "toy", "/m/06msq": "sculpture",
       "/m/0152hh": "snowman", "/m/07c52": "screen", "/m/02522": "screen", "/m/01c648": "screen",
       "/m/050k8": "screen", "/m/0bh9flk": "screen", "/m/025dyy": "packaging", "/m/02jnhm": "packaging",
       "/m/04dr76w": "packaging", "/m/01n5jq": "poster", "/m/01knjb": "poster"}
BG = {"/m/040b_t": "fridge", "/m/0fx9l": "microwave", "/m/029bxz": "oven", "/m/054_l": "mirror",
      "/m/02f9f_": "shower", "/m/03dnzn": "bathtub", "/m/0130jx": "sink", "/m/04y4h8h": "bathroom_cabinet",
      "/m/0h99cwc": "kitchen_appliance"}

QUOTA = {  # split -> role -> n
    "test": {"face": 260, "negative": 170, "background": 160},
    "validation": {"face": 900, "negative": 300, "background": 260},
}
MIN_FACE_REL = 0.07  # face width >= 7% of image width to count as a scoring face; smaller -> ignore


def load(split):
    pre = "test" if split == "test" else "validation"
    boxes = pd.read_csv(META / f"{pre}-annotations-bbox.csv")
    meta = pd.read_csv(META / f"{pre}-images-with-rotation.csv",
                       usecols=["ImageID", "OriginalLandingURL", "License", "Author", "AuthorProfileURL", "Rotation"])
    labels = pd.read_csv(META / f"{pre}-annotations-human-imagelabels-boxable.csv")
    return boxes, meta, labels


def pick(split):
    boxes, meta, labels = load(split)
    meta = meta[(meta.License == CCBY) & (meta.Rotation.fillna(0) == 0)]
    ok_ids = set(meta.ImageID)
    boxes = boxes[boxes.ImageID.isin(ok_ids)]
    face_absent = set(labels[(labels.LabelName == FACE) & (labels.Confidence == 0)].ImageID)
    people_pos = set(labels[labels.LabelName.isin(PEOPLE) & (labels.Confidence == 1)].ImageID)
    people_imgs = set(boxes[boxes.LabelName.isin(PEOPLE)].ImageID) | people_pos

    faces = boxes[boxes.LabelName == FACE].copy()
    faces["w"] = faces.XMax - faces.XMin
    g = faces.groupby("ImageID")
    bad = set(faces[(faces.IsDepiction == 1) | (faces.IsGroupOf == 1)].ImageID)
    has_big = set(faces[faces.w >= MIN_FACE_REL].ImageID)
    nfaces = g.size()
    face_ids = sorted((has_big - bad) & set(nfaces[nfaces <= 6].index))

    rows = []
    rng = pd.Series(face_ids).sample(frac=1.0, random_state=SEED).tolist()[: QUOTA[split]["face"]]
    for i in rng:
        rows.append((i, split, "face", ""))

    def by_class(table, quota, role, strict):
        cand = boxes[boxes.LabelName.isin(table)]
        cand = cand[~cand.ImageID.isin(people_imgs)]
        if strict:  # human-verified "Human face: absent" label required
            cand = cand[cand.ImageID.isin(face_absent)]
        cand = cand.drop_duplicates("ImageID")
        cand = cand.sample(frac=1.0, random_state=SEED)
        cand["cat"] = cand.LabelName.map(table)
        # round-robin across categories so one class cannot dominate
        per = {c: list(d.ImageID) for c, d in cand.groupby("cat")}
        out, k = [], 0
        while len(out) < quota and any(per.values()):
            for c in sorted(per):
                if per[c] and len(out) < quota:
                    out.append((per[c].pop(0), c))
            k += 1
        return [(i, split, role, c) for i, c in out]

    # eval negatives need the verified-absent label; training pools only need no person boxes/labels
    rows += by_class(NEG, QUOTA[split]["negative"], "negative", strict=(split == "test"))
    rows += by_class(BG, QUOTA[split]["background"], "background", strict=False)
    src = pd.DataFrame(rows, columns=["ImageID", "split", "role", "category"])
    src = src.drop_duplicates("ImageID")
    src = src.merge(meta, on="ImageID", how="left")
    fb = faces[faces.ImageID.isin(set(src[src.role == "face"].ImageID))][
        ["ImageID", "XMin", "XMax", "YMin", "YMax", "IsOccluded", "IsTruncated", "w"]].copy()
    fb["ignore"] = (fb.w < MIN_FACE_REL).astype(int)
    return src, fb


def main():
    srcs, fbs = [], []
    for split in ("test", "validation"):
        s, f = pick(split)
        srcs.append(s)
        fbs.append(f)
        print(split, s.groupby(["role"]).size().to_dict(), "face boxes", len(f), "ignore", int(f.ignore.sum()))
    src = pd.concat(srcs)
    assert not set(src[src.split == "test"].ImageID) & set(src[src.split == "validation"].ImageID)
    src.to_csv(ROOT / "data" / "sources.csv", index=False)
    pd.concat(fbs).drop(columns=["w"]).to_csv(ROOT / "data" / "source_boxes.csv", index=False)
    print(src.groupby(["split", "role", "category"]).size().to_string())


if __name__ == "__main__":
    sys.exit(main())
