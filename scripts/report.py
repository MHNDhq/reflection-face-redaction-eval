"""Build results/metrics.json, results/metrics.md and results/pr_curve.png from saved detections.

Cutoffs are chosen on the TUNE set (validation-split sources, never trained on) as the highest
redaction recall among cutoffs where the lower end of the two-sided 95% confidence interval (Wilson) for tune precision is at least 95%,
then applied unchanged to the EVAL set. A detector with no such cutoff gets no deployable row.
YuNet is also reported at 0.9, the client's current setting.

Usage: .venv/bin/python scripts/report.py
"""
import json
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
        ("ft_B", "Faster R-CNN, +reflection composites (after)")]
GRID = sorted(set([round(0.05 * i, 2) for i in range(1, 20)] + [0.92, 0.94, 0.96, 0.97, 0.98, 0.99]))
FAMS = ["mirror", "appliance", "shower", "photo", "screen", "plain"]
KEYS = ["redaction_recall", "redaction_precision", "redaction_frames_fully_covered",
        "redaction_false_alarm_frame_rate", "recall", "precision"]


def load(set_name, name):
    p = ROOT / "results" / f"{set_name}_{name}.json"
    return json.load(open(p)) if p.exists() else None


def _wilson_lb(p, n, z=1.96):
    """Lower end of the 95% Wilson interval for a proportion p measured on n boxes."""
    if not n:
        return 0.0
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - s) / d


def pick_cutoff(recs, dets):
    """Highest redaction recall on TUNE among cutoffs where the lower end of the two-sided 95% Wilson
    interval for tune precision is >= 0.95 (z = 1.96, so 97.5% one-sided). The first rule (point precision >= 95% on the 240-image
    tune split) over-fit that small split: 95-96% on tune became 88-91% on eval. Returns None when no
    cutoff clears the bound, so the report never shows a fallback as a rule choice."""
    best = None
    for t in GRID:
        m = metrics(recs, dets, t)
        p, n = m["redaction_precision"], m.get("redaction_boxes")
        if p is not None and _wilson_lb(p, n) >= 0.95:
            if best is None or m["redaction_recall"] > best[1]:
                best = (t, m["redaction_recall"])
    return best[0] if best else None


def main():
    ev = json.load(open(ROOT / "data" / "eval" / "annotations.json"))["images"]
    tu = json.load(open(ROOT / "data" / "tune" / "annotations.json"))["images"]
    out = {"note": "SYNTHETIC eval set (Open Images CC BY 2.0 composites); not the client's footage.",
           "eval_images": len(ev), "tune_images": len(tu), "detectors": {}}
    plt.figure(figsize=(7, 5))
    for name, label in DETS:
        de, dt = load("eval", name), load("tune", name)
        if de is None or dt is None:
            print("missing", name)
            continue
        cut = pick_cutoff(tu, dt)
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
    plt.savefig(ROOT / "results" / "pr_curve.png", dpi=130)
    json.dump(out, open(ROOT / "results" / "metrics.json", "w"), indent=1)

    def pct(v):
        return "n/a" if v is None else f"{100 * v:.1f}%"

    lines = ["# Results on the SYNTHETIC eval set (not the client's footage)", "",
             f"Eval: {len(ev)} composite images (test-split sources). Tune: {len(tu)} images (validation-split sources held out of training). "
             "Cutoff per detector = highest redaction recall on TUNE among cutoffs where the lower end of the two-sided 95% Wilson "
             "interval for tune precision is >= 95%, applied unchanged to EVAL. This rule and the finer cutoff grid were chosen "
             "AFTER eval had been read under the first rule (point precision >= 95% on tune), which over-fit the small tune split.", "",
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
    (ROOT / "results" / "metrics.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
