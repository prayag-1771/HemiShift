"""S3 confound check: do non-motor channels decode imagery vs rest? (dev, exploratory)

The sweep's occipital control decoded Task B at ~79 % (k = 5). It used the 59-channel common
average, which subtracts the mean of all channels from every channel, so a widespread motor
ERD reaches the occipital channels through the reference. Here each channel group is also
re-referenced to its own average, which removes that shared term exactly.
  motor14    the template (F/FC/C/CP/P rows), the decoders' channels
  occipital  PO7 PO5 PO3 POz PO4 PO6 PO8 O1 Oz O2 (visual cortex: the screen, attention)
  frontal    Fpz Fp1 Fp2 AF3 AF4 AF7 AF8 (eye movements and blinks; also 1-8 Hz)
  temporal   FT7 FT8 T7 T8 TP7 TP8 (jaw and neck muscle)
Second part: where the sweep's best variant (pooled_fb4: theta, mu, low and high beta) gets
its gain. Theta is where eye activity lives, so fb4 is also run without theta, theta alone,
and on the strict motor strip (FC/C/CP rows only, away from the eyes and visual cortex).
own and pooled TS + LR at k = 5 and 10 (protocol P3), donors the other dev patients.
Results: results/summary/s3_confound_check_dev.csv.

Usage: .venv/Scripts/python -I scripts/s3_confound_check.py
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from hemishift.data.channels import EXTENDED_TEMPLATE
from hemishift.paths import RESULTS_DIR


def _script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = _script("s3_sweep")
GROUPS = {
    "motor14": EXTENDED_TEMPLATE,
    "occipital": sweep.OCCIPITAL,
    "frontal": ("Fpz", "Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8"),
    "temporal": ("FT7", "FT8", "T7", "T8", "TP7", "TP8"),
}
STRICT_MOTOR = ("FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CP4")
FB4 = {"theta": (4.0, 8.0), "mu": (8.0, 13.0), "b1": (13.0, 20.0), "b2": (20.0, 30.0)}
FB_VARIANTS = {"fb4": ("theta", "mu", "b1", "b2"), "fb4-theta": ("mu", "b1", "b2"),
               "theta": ("theta",)}
OUT = RESULTS_DIR / "summary" / "s3_confound_check_dev.csv"


def main():
    X59 = sweep.load("dev", "pre", sweep.ALL59, sweep.WINDOWS["std"])
    subjects = list(X59)
    y = {s: X59[s][1] for s in subjects}
    variants = {}  # name -> list of per-band covariance dicts (tangent vectors concatenated)
    for group, channels in GROUPS.items():
        for reference in ("local", "car59"):
            bands = [("830", (8.0, 30.0))] + ([("1-8", (1.0, 8.0))] if group == "frontal" else [])
            for band, hz in bands:
                variants[f"{group}_{reference}_{band}"] = [{
                    s: sweep.covariances(sweep.subset(X59[s][0], channels, reference == "local"), hz)
                    for s in subjects}]
    for group, channels in (("motor14", EXTENDED_TEMPLATE), ("strict8", STRICT_MOTOR)):
        per_band = {b: {s: sweep.covariances(sweep.subset(X59[s][0], channels), hz) for s in subjects}
                    for b, hz in FB4.items()}
        for name, bands in FB_VARIANTS.items():
            variants[f"{group}_local_{name}"] = [per_band[b] for b in bands]
        if group == "strict8":
            variants["strict8_local_830"] = [{s: sweep.covariances(sweep.subset(X59[s][0], channels),
                                                                   (8.0, 30.0)) for s in subjects}]
    rows = []
    for name, Cs in variants.items():
        TF = {s: np.hstack([sweep.tangent(C[s], np.arange(len(C[s]))) for C in Cs]) for s in subjects}
        for s in subjects:
            for k in sweep.KS:
                cal, test = np.arange(2 * k), np.arange(2 * k, len(y[s]))
                T = np.hstack([sweep.tangent(C[s], cal) for C in Cs])
                own = sweep.lr().fit(T[cal], y[s][cal]).predict_proba(T[test])[:, 1]
                others = [o for o in subjects if o != s]
                T_d = np.vstack([TF[o] for o in others])
                y_d = np.concatenate([y[o] for o in others])
                pooled = sweep.lr().fit(np.vstack([T[cal], T_d]), np.concatenate([y[s][cal], y_d]),
                                        sample_weight=sweep.donor_weights(len(cal), len(y_d), 1.0)
                                        ).predict_proba(T[test])[:, 1]
                rows.append({"variant": name, "subject": s, "k": k,
                             "own": 100 * np.mean((own > 0.5) == y[s][test]),
                             "pooled": 100 * np.mean((pooled > 0.5) == y[s][test])})
        t = pd.DataFrame(rows)
        t = t[t["variant"] == name]
        print(f"{name:22s} " + " | ".join(
            f"k={k}: own {t[t.k == k]['own'].mean():5.1f} pooled {t[t.k == k]['pooled'].mean():5.1f}"
            for k in sweep.KS), flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"results in {OUT}")


if __name__ == "__main__":
    main()
