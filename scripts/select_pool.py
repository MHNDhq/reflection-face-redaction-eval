"""Select an extra source pool from Open Images, disjoint from every image already in data/sources.csv.
Added 2026-10-24 for the third, pre-registered round (README > Pre-registration, round 3).

  .venv/bin/python scripts/select_pool.py POOL SPLIT N_FACE N_NEG N_BG [--strict]

POOL   name written to the `pool` column (eval2, tune3, train_hn)
SPLIT  test | validation
--strict  negatives need the human-verified 'Human face = absent' label (as for eval); without it they
          only need no person boxes or positive person labels (as for the training pools).

Same filters as select_sources.py. Appends to data/sources.csv and data/source_boxes.csv; refuses a pool
name that already exists. composite.py --pool POOL builds it.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_sources as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    pool, split = args[0], args[1]
    n_face, n_neg, n_bg = (int(x) for x in args[2:5])
    strict = "--strict" in sys.argv
    seed = {"eval2": 31, "tune3": 32, "train_hn": 33, "eval3": 35}.get(pool, 34)
    src_path, fb_path = ROOT / "data" / "sources.csv", ROOT / "data" / "source_boxes.csv"
    src0, fb0 = pd.read_csv(src_path), pd.read_csv(fb_path)
    if "pool" in src0.columns and (src0["pool"] == pool).any():
        raise SystemExit(f"pool {pool} already present; refusing to append twice")
    used = set(src0.ImageID)

    boxes, meta, labels = S.load(split)
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
    face_pick = pd.Series(face_ids, dtype=object).sample(frac=1.0, random_state=seed).tolist()[:n_face]

    def by_class(table, quota, need_absent):
        cand = boxes[boxes.LabelName.isin(table)]
        cand = cand[~cand.ImageID.isin(people_imgs) & ~cand.ImageID.isin(used) & ~cand.ImageID.isin(set(face_pick))]
        if need_absent:
            cand = cand[cand.ImageID.isin(face_absent)]
        cand = cand.drop_duplicates("ImageID").sample(frac=1.0, random_state=seed)
        cand["cat"] = cand.LabelName.map(table)
        per = {c: list(d.ImageID) for c, d in cand.groupby("cat")}
        out = []
        while len(out) < quota and any(per.values()):
            for c in sorted(per):
                if per[c] and len(out) < quota:
                    out.append((per[c].pop(0), c))
        return out

    neg = by_class(S.NEG, n_neg, strict)
    bg = by_class(S.BG, n_bg, False)
    rows = ([(i, split, "face", "") for i in face_pick] + [(i, split, "negative", c) for i, c in neg]
            + [(i, split, "background", c) for i, c in bg])
    add = pd.DataFrame(rows, columns=["ImageID", "split", "role", "category"]).drop_duplicates("ImageID")
    add = add.merge(meta, on="ImageID", how="left")
    add["pool"] = pool
    assert not set(add.ImageID) & used
    fb = faces[faces.ImageID.isin(set(face_pick))][
        ["ImageID", "XMin", "XMax", "YMin", "YMax", "IsOccluded", "IsTruncated", "w"]].copy()
    fb["ignore"] = (fb.w < S.MIN_FACE_REL).astype(int)
    fb = fb.drop(columns=["w"])
    if "pool" not in src0.columns:
        src0["pool"] = ""
    pd.concat([src0, add], ignore_index=True).to_csv(src_path, index=False)
    pd.concat([fb0, fb], ignore_index=True).to_csv(fb_path, index=False)
    cats = add[add.role == "negative"].category.value_counts().to_dict()
    if len(neg) < n_neg or len(face_pick) < n_face:
        print(f"SHORTFALL: asked {n_face} faces / {n_neg} negatives, got {len(face_pick)} / {len(neg)} (candidate pool ran out)")
    print(f"{pool}/{split}: {len(face_pick)} faces ({len(fb)} boxes), {len(neg)} negatives {cats}, {len(bg)} backgrounds")


if __name__ == "__main__":
    sys.exit(main())
