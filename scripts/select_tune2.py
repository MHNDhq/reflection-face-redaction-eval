"""Select a SECOND, larger tune pool from the Open Images VALIDATION split, disjoint from every image in
data/sources.csv (so no training, tune or eval photo is reused as content). Added 2026-10-23 because the
240-image tune split was too small for a stable cutoff (one tune box moved the chosen cutoff).

Same filters as select_sources.py (CC BY 2.0, Rotation 0, real faces, no depictions or groups, at most 6
faces, at least one face >= 7% of the image width; negatives with the human-verified 'Human face = absent'
label, as for eval). Backgrounds are NOT selected here: tune2 reuses the validation background pool, as the
first tune split does.

Appends rows with pool='tune2' to data/sources.csv and data/source_boxes.csv (provenance stays in one file);
composite.py keeps the original train/tune split by ignoring pool='tune2' rows there.

Usage: .venv/bin/python scripts/select_tune2.py [n_faces] [n_negatives]
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_sources as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED2 = 20261024


def main():
    n_face = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    n_neg = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    src_path, fb_path = ROOT / "data" / "sources.csv", ROOT / "data" / "source_boxes.csv"
    src0, fb0 = pd.read_csv(src_path), pd.read_csv(fb_path)
    if "pool" in src0.columns and (src0["pool"] == "tune2").any():
        raise SystemExit("tune2 rows already present in data/sources.csv; refusing to append twice")
    used = set(src0.ImageID)

    boxes, meta, labels = S.load("validation")
    meta = meta[(meta.License == S.CCBY) & (meta.Rotation.fillna(0) == 0)]
    boxes = boxes[boxes.ImageID.isin(set(meta.ImageID))]
    face_absent = set(labels[(labels.LabelName == S.FACE) & (labels.Confidence == 0)].ImageID)
    people_pos = set(labels[labels.LabelName.isin(S.PEOPLE) & (labels.Confidence == 1)].ImageID)
    people_imgs = set(boxes[boxes.LabelName.isin(S.PEOPLE)].ImageID) | people_pos

    faces = boxes[boxes.LabelName == S.FACE].copy()
    faces["w"] = faces.XMax - faces.XMin
    bad = set(faces[(faces.IsDepiction == 1) | (faces.IsGroupOf == 1)].ImageID)
    has_big = set(faces[faces.w >= S.MIN_FACE_REL].ImageID)
    nf = faces.groupby("ImageID").size()
    face_ids = sorted(((has_big - bad) & set(nf[nf <= 6].index)) - used)
    face_pick = pd.Series(face_ids).sample(frac=1.0, random_state=SEED2).tolist()[:n_face]

    cand = boxes[boxes.LabelName.isin(S.NEG)]
    cand = cand[~cand.ImageID.isin(people_imgs) & cand.ImageID.isin(face_absent) & ~cand.ImageID.isin(used)]
    cand = cand.drop_duplicates("ImageID").sample(frac=1.0, random_state=SEED2)
    cand["cat"] = cand.LabelName.map(S.NEG)
    per = {c: list(d.ImageID) for c, d in cand.groupby("cat")}
    neg = []
    while len(neg) < n_neg and any(per.values()):
        for c in sorted(per):
            if per[c] and len(neg) < n_neg:
                neg.append((per[c].pop(0), c))

    rows = [(i, "validation", "face", "") for i in face_pick] + [(i, "validation", "negative", c) for i, c in neg]
    add = pd.DataFrame(rows, columns=["ImageID", "split", "role", "category"]).merge(meta, on="ImageID", how="left")
    add["pool"] = "tune2"
    assert not set(add.ImageID) & used
    fb = faces[faces.ImageID.isin(set(face_pick))][
        ["ImageID", "XMin", "XMax", "YMin", "YMax", "IsOccluded", "IsTruncated", "w"]].copy()
    fb["ignore"] = (fb.w < S.MIN_FACE_REL).astype(int)
    fb = fb.drop(columns=["w"])

    if "pool" not in src0.columns:
        src0["pool"] = ""
    pd.concat([src0, add], ignore_index=True).to_csv(src_path, index=False)
    pd.concat([fb0, fb], ignore_index=True).to_csv(fb_path, index=False)
    if len(neg) < n_neg or len(face_pick) < n_face:
        print(f"SHORTFALL: asked {n_face} faces / {n_neg} negatives, got {len(face_pick)} / {len(neg)} (candidate pool ran out)")
    print(f"tune2: {len(face_pick)} face images ({len(fb)} face boxes), {len(neg)} negatives; "
          f"{len(face_ids)} eligible face images were unused")


if __name__ == "__main__":
    sys.exit(main())
