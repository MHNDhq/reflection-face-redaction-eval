"""Rebuild the README's headline figures that report.py does not print: the tune recall behind each
pre-registered model choice, and the eval3 95% intervals (bootstrap over source photos and, separately,
over background photos; the README quotes their union: the lower of the two lower bounds and the higher of the two upper bounds).

Usage: .venv/bin/python scripts/intervals.py        -> results/intervals.json (+ printed)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report as RP  # noqa: E402
import evaluate as E  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TUNES = ["tune", "tune2", "tune3"]
CHOICES = [("round 2", ["ft_B", "ft_B_long"], ["tune", "tune2"]), ("round 3", ["ft_B_long", "ft_HN"], TUNES),
           ("round 4", ["ft_HN", "ft_HN2"], TUNES)]
EVAL3 = [("ft_HN2", "after"), ("ft_A", "before"), ("yunet", "YuNet")]


def recs(sets):
    return sum((json.load(open(ROOT / "data" / s / "annotations.json"))["images"] for s in sets), [])


def boot(cl, n=2000, seed=1):
    a = np.array(list(cl.values()), float)
    idx = np.random.default_rng(seed).integers(0, len(a), size=(n, len(a)))
    k, fa, f, c = (a[idx, j].sum(1) for j in range(4))
    p, r = (k - fa) / np.maximum(k, 1), c / np.maximum(f, 1)
    return [float(np.quantile(p, .025)), float(np.quantile(p, .975))], [float(np.quantile(r, .025)), float(np.quantile(r, .975))]


def main():
    out = {"model_choice": {}, "eval3": {}}
    for label, models, sets in CHOICES:
        tu = recs(sets)
        row = {}
        for m in models:
            dt = {}
            for s in sets:
                dt.update(RP.load(s, m))
            cut = RP.pick_cutoff(tu, dt)
            a = RP._per_cluster(tu, dt, cut) if cut is not None else None
            row[m] = {"cutoff": cut, "tune_recall": None if a is None else float(a[:, 3].sum() / a[:, 2].sum())}
        out["model_choice"][label] = row
    ev = recs(["eval3"])
    tu = recs(TUNES)
    for m, lab in EVAL3:
        dt = {}
        for s in TUNES:
            dt.update(RP.load(s, m))
        cut = RP.pick_cutoff(tu, dt)
        dets = RP.load("eval3", m)
        by = {"photo": {}, "background": {}}
        for r in ev:
            cov, fa, nk = E.coverage_image(dets[r["file"]], r["boxes"], r["ignore"], cut)
            for key, cl in (("photo", r["content_id"]), ("background", r.get("background_id") or r["file"])):
                c = by[key].setdefault(cl, [0, 0, 0, 0])
                c[0] += nk; c[1] += fa; c[2] += len(cov); c[3] += sum(cov)
        res = {"cutoff": cut}
        for key, cl in by.items():
            p, rr = boot(cl)
            res[key] = {"precision_ci": p, "recall_ci": rr, "clusters": len(cl)}
        res["quoted_precision_ci"] = [min(res["photo"]["precision_ci"][0], res["background"]["precision_ci"][0]),
                                      max(res["photo"]["precision_ci"][1], res["background"]["precision_ci"][1])]
        res["quoted_recall_ci"] = [min(res["photo"]["recall_ci"][0], res["background"]["recall_ci"][0]),
                                   max(res["photo"]["recall_ci"][1], res["background"]["recall_ci"][1])]
        out["eval3"][m] = res
    json.dump(out, open(ROOT / "results" / "intervals.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.exit(main())
