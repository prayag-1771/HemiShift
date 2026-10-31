"""Sensitivity of the confirmed S3 locked-test result to ICA cleaning (declared in
ACHIEVEMENTS.md before running, 2026-10-31). Runs once; not a new hypothesis.

The confirmed comparison (own vs pooled TS + LR, 14 channels, 8-30 Hz, protocol P3) is
repeated on the authors' ICA-cleaned release of the 8 locked-test patients, with the 16 dev
patients' cleaned recordings as donors. The band-bank pooled decoder (mu, low beta, high beta;
S4's H-R2 decoder) is reported alongside. Results: results/summary/s3_locked_sensitivity.csv.

Usage: .venv/Scripts/python -I scripts/s3_locked_sensitivity.py
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from hemishift.data.channels import EXTENDED_TEMPLATE
from hemishift.data.loaders import load_wang2026sf_epochs
from hemishift.paths import CONFIG_DIR, RESULTS_DIR


def _script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = _script("s3_sweep")
check = _script("s3_processed_check")
BANDS = {"830": ((8.0, 30.0),), "bank": ((8.0, 13.0), (13.0, 20.0), (20.0, 30.0))}
OUT = RESULTS_DIR / "summary" / "s3_locked_sensitivity.csv"


def load_test_cleaned():
    split = json.loads((CONFIG_DIR / "splits" / "wang2026sf_locked_test.json").read_text())
    print(f"locked test patients: {split['test']} (split hash {split['hash']})")
    patients = {}
    for subject in split["test"]:
        e = load_wang2026sf_epochs([subject], template=check.COMMON42, source="processed")
        mi = np.flatnonzero(e.labels == "affected_mi")
        paired = mi[(mi > 0) & (e.labels[mi - 1] == "rest")]
        order = np.ravel(np.column_stack([paired - 1, paired]))
        patients[subject] = (e.X[order].astype(np.float32),
                             (e.labels[order] == "affected_mi").astype(int))
        print(f"  cleaned sub {subject}: {order.size} epochs", flush=True)
    return patients


def main():
    donors = check.load("processed")
    test = load_test_cleaned()
    covs = {name: {s: [sweep.covariances(check.subset(X, EXTENDED_TEMPLATE), b) for b in bands]
                   for s, (X, _) in {**donors, **test}.items()} for name, bands in BANDS.items()}
    rows = []
    for s, (_, y) in test.items():
        for k in sweep.KS:
            cal, tst = np.arange(2 * k), np.arange(2 * k, len(y))
            row = {"subject": s, "k": k, "n_test": len(tst)}
            for name in BANDS:
                T = np.hstack([sweep.tangent(c, cal) for c in covs[name][s]])
                donor_T = np.vstack([np.hstack([sweep.tangent(c, np.arange(len(c)))
                                                for c in covs[name][o]]) for o in donors])
                donor_y = np.concatenate([donors[o][1] for o in donors])
                own = sweep.lr().fit(T[cal], y[cal]).predict_proba(T[tst])[:, 1]
                pooled = sweep.lr().fit(
                    np.vstack([T[cal], donor_T]), np.concatenate([y[cal], donor_y]),
                    sample_weight=sweep.donor_weights(len(cal), len(donor_y), 1.0)
                ).predict_proba(T[tst])[:, 1]
                row[f"own_{name}"] = 100 * np.mean((own > 0.5) == y[tst])
                row[f"pooled_{name}"] = 100 * np.mean((pooled > 0.5) == y[tst])
            rows.append(row)
            print(f"sub {s:2d} k={k:2d}: " + " ".join(f"{n} {v:.1f}" for n, v in row.items()
                                                      if n.startswith(("own", "pooled"))))
    table = pd.DataFrame(rows)
    table.to_csv(OUT, index=False)
    for k in sweep.KS:
        t = table[table["k"] == k]
        for a, b in (("pooled_830", "own_830"), ("pooled_bank", "own_830"),
                     ("pooled_bank", "pooled_830")):
            diff = t[a] - t[b]
            nonzero = diff[diff != 0]
            p = stats.wilcoxon(nonzero, alternative="greater").pvalue if len(nonzero) else 1.0
            print(f"k = {k}: {a} {t[a].mean():.1f} % vs {b} {t[b].mean():.1f} % | mean diff "
                  f"{diff.mean():+.1f}, helped {(diff > 0).sum()}, harmed {(diff < 0).sum()}, "
                  f"one-sided Wilcoxon p = {p:.4f}")
    print(f"results in {OUT}")


if __name__ == "__main__":
    main()
