"""Build results/metrics.json, results/metrics.md and results/pr_curve.png from saved detections.

Cutoffs are chosen on the TUNE set (validation-split sources, never trained on) as the highest
redaction recall among cutoffs where the lower end of the two-sided 95% confidence interval (Wilson) for tune precision is at least 95%,
then applied unchanged to the EVAL set. A detector with no such cutoff gets no deployable row.
YuNet is also reported at 0.9, the client's current setting.

Usage: .venv/bin/python scripts/report.py
"""
import json

import numpy as np
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DETS = [("yunet", "YuNet 2023mar (OpenCV)"), ("blazeface_short", "BlazeFace short-range"),
        ("blazeface_full", "BlazeFace full-range"), ("ft_A", "Faster R-CNN, plain faces (before)"),
        ("ft_control", "Faster R-CNN, control: +plain & negatives"),
        ("ft_B", "Faster R-CNN, +reflection composites (500 steps)"),
        ("ft_B_long", "Faster R-CNN, +reflection composites, 1,500 steps"),
        ("ft_HN", "Faster R-CNN, +reflection composites + hard negatives, 1,500 steps")]
GRID = sorted(set([round(0.05 * i, 2) for i in range(1, 20)] + [0.92, 0.94, 0.96, 0.97, 0.98, 0.99]))
FAMS = ["mirror", "appliance", "shower", "photo", "screen", "plain"]
KEYS = ["redaction_recall", "redaction_precision", "redaction_frames_fully_covered",
        "redaction_false_alarm_frame_rate", "recall", "precision"]


def load(set_name, name):
    p = ROOT / "results" / f"{set_name}_{name}.json"
    return json.load(open(p)) if p.exists() else None


BOOT_N, BOOT_SEED, BOOT_Q = 1000, 20261024, 0.05
# --eval eval2 reports on the fresh round-3 set (results/metrics_eval2.*); default is the original eval set
EVAL = sys.argv[sys.argv.index("--eval") + 1] if "--eval" in sys.argv else "eval"
SUFFIX = "" if EVAL == "eval" else f"_{EVAL}"


def _per_cluster(recs, dets, thr):
    """content_id -> [kept boxes, false alarms, faces, faces covered] at cutoff thr."""
    from evaluate import coverage_image
    out = {}
    for r in recs:
        cov, fa, nk = coverage_image(dets[r["file"]], r["boxes"], r["ignore"], thr)
        c = out.setdefault(r.get("content_id", r["file"]), [0, 0, 0, 0])
        c[0] += nk; c[1] += fa; c[2] += len(cov); c[3] += sum(cov)
    return np.array(list(out.values()), dtype=float)


def boot_precision_q(arr, n=BOOT_N, q=BOOT_Q, seed=BOOT_SEED):
    """One-sided lower bound: the q-quantile of tune precision over a bootstrap that resamples SOURCE
    PHOTOS (content_id), so the repeated composites of one photo count once (review 2026-10-23: boxes
    cluster by photo and background, so a per-box interval was too generous)."""
    if not len(arr) or arr[:, 0].sum() == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n, len(arr)))
    kept, fa = arr[idx, 0].sum(1), arr[idx, 1].sum(1)
    prec = np.where(kept > 0, (kept - fa) / np.maximum(kept, 1), 1.0)
    return float(np.quantile(prec, q))


def pick_cutoff(recs, dets):
    """PRE-REGISTERED 2026-10-23, committed before any detector was run on tune2 and before eval was
    re-read: the cutoff with the highest redaction recall on TUNE (tune + tune2) among cutoffs whose
    tune precision has a cluster-bootstrap 5th percentile >= 0.95 (one-sided 95%). History: rule 1
    (point precision on the 240-image tune split) over-fit it; rule 2 (per-box Wilson bound) was chosen
    after a first look at eval and treated clustered boxes as independent. Returns None when no cutoff
    qualifies, and the report then shows no deployable row."""
    best = None
    for t in GRID:
        arr = _per_cluster(recs, dets, t)
        if not len(arr) or arr[:, 0].sum() == 0:
            continue
        if boot_precision_q(arr) >= 0.95:
            rec = arr[:, 3].sum() / max(1, arr[:, 2].sum())
            if best is None or rec > best[1]:
                best = (t, rec)
    return best[0] if best else None


def main():
    ev = json.load(open(ROOT / "data" / EVAL / "annotations.json"))["images"]
    tu = json.load(open(ROOT / "data" / "tune" / "annotations.json"))["images"]
    t2p = ROOT / "data" / "tune2" / "annotations.json"
    tu2 = json.load(open(t2p))["images"] if t2p.exists() else []
    t3p = ROOT / "data" / "tune3" / "annotations.json"
    tu2 += json.load(open(t3p))["images"] if t3p.exists() else []   # round 3: matched no-face share
    out = {"note": "SYNTHETIC eval set (Open Images CC BY 2.0 composites); not the client's footage.",
           "eval_images": len(ev), "tune_images": len(tu), "detectors": {}}
    plt.figure(figsize=(7, 5))
    for name, label in DETS:
        de, dt, dt2 = load(EVAL, name), load("tune", name), load("tune2", name)
        dt3 = load("tune3", name) if t3p.exists() else {}
        if tu2 and (dt2 is None or dt3 is None):
            print("missing tune2/tune3 detections for", name)
            continue
        if dt is not None and dt2:
            dt = {**dt, **dt2, **(dt3 or {})}
        if de is None or dt is None:
            print("missing", name)
            continue
        cut = pick_cutoff(tu + tu2, dt)
        row = {"label": label, "cutoff_from_tune": cut,
               "overall": metrics(ev, de, cut) if cut is not None else None,
               "by_family": {f: metrics(ev, de, cut, family=f) for f in FAMS} if cut is not None else None}
        if name == "yunet":
            row["at_0.9"] = {"overall": metrics(ev, de, 0.9),
                             "by_family": {f: metrics(ev, de, 0.9, family=f) for f in FAMS}}
        curve = [metrics(ev, de, t) for t in GRID]
        ok = [(c["redaction_recall"], t, c) for t, c in zip(GRID, curve)
              if c["redaction_precision"] is not None and c["redaction_precision"] >= 0.95]
        if ok:
            r_, t_, c_ = max(ok, key=lambda x: x[0])
            row["eval_oracle_at_95_precision"] = {"thr": t_, "redaction_recall": r_,
                                                  "redaction_precision": c_["redaction_precision"],
                                                  "note": "cutoff chosen ON EVAL; comparison only, not deployable"}
        row["curve"] = [{"thr": t, "redaction_recall": c["redaction_recall"],
                         "redaction_precision": c["redaction_precision"]} for t, c in zip(GRID, curve)]
        out["detectors"][name] = row
        xs = [c["redaction_recall"] for c in curve]
        ys = [c["redaction_precision"] if c["redaction_precision"] is not None else 1.0 for c in curve]
        plt.plot(xs, ys, marker=".", label=label)
    plt.axhline(0.95, color="grey", lw=0.8, ls="--")
    plt.axvline(0.99, color="grey", lw=0.8, ls=":")
    plt.xlabel("redaction recall (faces >=70% covered)")
    plt.ylabel("redaction precision")
    plt.title("SYNTHETIC reflection eval set (not real footage)", fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.legend(fontsize=7, loc="lower left")
    plt.tight_layout()
    plt.savefig(ROOT / "results" / f"pr_curve{SUFFIX}.png", dpi=130)
    json.dump(out, open(ROOT / "results" / f"metrics{SUFFIX}.json", "w"), indent=1)

    def pct(v):
        return "n/a" if v is None else f"{100 * v:.1f}%"

    lines = [f"# Results on the SYNTHETIC {EVAL} set (not the client's footage)", "",
             f"{EVAL}: {len(ev)} composite images (test-split sources). Tune: {len(tu)} images (validation-split sources held out of training). "
             f"Cutoff per detector = highest redaction recall on TUNE ({len(tu) + len(tu2)} images: tune + tune2) among cutoffs whose tune "
             "precision has a cluster-bootstrap 5th percentile >= 95% (resampling source photos, 1,000 draws, one-sided 95%), "
             "applied unchanged to EVAL. Pre-registered before tune2 detections or a new eval read; see README for the two earlier rules.", "",
             "| detector | cutoff | redaction recall | redaction precision | frames fully covered | false-alarm frames (negatives) | recall IoU0.5 | precision IoU0.5 |",
             "|---|---|---|---|---|---|---|---|"]
    for name, row in out["detectors"].items():
        rows = [(row["label"], row["cutoff_from_tune"], row["overall"])] if row["overall"] else []
        if "at_0.9" in row:
            rows.insert(0, (row["label"] + " at 0.9 (client setting)", 0.9, row["at_0.9"]["overall"]))
        if not row["overall"]:
            lines.append(f"| {row['label']} | no cutoff met the rule | n/a | n/a | n/a | n/a | n/a | n/a |")
        for lab, c, m in rows:
            lines.append(f"| {lab} | {c} | {pct(m['redaction_recall'])} | {pct(m['redaction_precision'])} | "
                         f"{pct(m['redaction_frames_fully_covered'])} | {pct(m['redaction_false_alarm_frame_rate'])} | "
                         f"{pct(m['recall'])} | {pct(m['precision'])} |")
    lines += ["", "## Comparison at equal precision (cutoff chosen on EVAL itself; for comparison only, not a deployable number)", "",
              "| detector | cutoff | redaction recall | redaction precision |", "|---|---|---|---|"]
    for name, row in out["detectors"].items():
        o = row.get("eval_oracle_at_95_precision")
        lines.append(f"| {row['label']} | {o['thr'] if o else 'none'} | {pct(o['redaction_recall']) if o else 'never reaches 95%'} | {pct(o['redaction_precision']) if o else ''} |")
    lines += ["", "## Redaction recall by family (shower = held out of training)", "",
              "| detector | " + " | ".join(FAMS) + " |", "|---" * (len(FAMS) + 1) + "|"]
    for name, row in out["detectors"].items():
        if "at_0.9" in row:
            lines.append(f"| {row['label']} at 0.9 | " + " | ".join(pct(row['at_0.9']['by_family'][f]['redaction_recall']) for f in FAMS) + " |")
        if row["by_family"]:
            lines.append(f"| {row['label']} at {row['cutoff_from_tune']} | " + " | ".join(pct(row['by_family'][f]['redaction_recall']) for f in FAMS) + " |")
    lines += ["", "## False-alarm frames on hard negatives by family", "",
              "| detector | " + " | ".join(FAMS) + " |", "|---" * (len(FAMS) + 1) + "|"]
    for name, row in out["detectors"].items():
        if row["by_family"]:
            lines.append(f"| {row['label']} at {row['cutoff_from_tune']} | " + " | ".join(pct(row['by_family'][f]['redaction_false_alarm_frame_rate']) for f in FAMS) + " |")
    (ROOT / "results" / f"metrics{SUFFIX}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
