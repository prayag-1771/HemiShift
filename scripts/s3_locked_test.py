"""S3 locked-test confirmation of patient pooling (pre-registered in ACHIEVEMENTS.md, 2026-10-24).

Runs once, on the 8 S3 locked-test patients. For each, the first k (rest, MI) pairs calibrate
and the rest test. own TS + LR (calibration only) vs pooled TS + LR with the 16 dev patients
as donors, using step 3's code unchanged (s3_augment.pooled_proba: each domain re-centred on
its own mean, donors together weigh as much as the calibration).
Primary: pooled > own at k = 5, one-sided Wilcoxon. Secondary: k = 10.

Usage: .venv/Scripts/python -I scripts/s3_locked_test.py
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from hemishift.eval.stats import chance_threshold
from hemishift.paths import CONFIG_DIR, RESULTS_DIR


def _script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s3 = _script("s3_transfer")
augment = _script("s3_augment")
OUT = RESULTS_DIR / "summary" / "s3_locked_test.csv"


def main():
    split = json.loads((CONFIG_DIR / "splits" / "wang2026sf_locked_test.json").read_text())
    print(f"locked test patients: {split['test']} (split hash {split['hash']})")
    donors = s3.load_patients("dev")
    C_donors = {s: s3.premise.covariances(X) for s, (X, _) in donors.items()}
    donor_C = np.concatenate([augment.recentred_by_domain(C, np.zeros(len(C))) for C in C_donors.values()])
    donor_y = np.concatenate([y for _, y in donors.values()])
    patients = s3.load_patients("test")

    rows = []
    for subject, (X, y) in patients.items():
        C = s3.premise.covariances(X)
        for k in s3.KS:
            cal, test = s3.calibration_split(len(y), k)
            own = s3.own_proba(X, y, cal, test)
            pooled = augment.pooled_proba(C[cal], y[cal], C[test], donor_C, donor_y)
            rows.append({"subject": subject, "k": k,
                         "own": 100 * np.mean((own > 0.5).astype(int) == y[test]),
                         "pooled": 100 * np.mean((pooled > 0.5).astype(int) == y[test])})
            print(f"sub {subject:2d} k={k:2d}: own {rows[-1]['own']:.1f} | pooled {rows[-1]['pooled']:.1f}")
    table = pd.DataFrame(rows)
    table.to_csv(OUT, index=False)

    for k in s3.KS:
        t = table[table["k"] == k]
        diff = t["pooled"] - t["own"]
        nonzero = diff[diff != 0]
        p = stats.wilcoxon(nonzero, alternative="greater").pvalue if len(nonzero) else float("nan")
        threshold = 100 * chance_threshold(80 - 2 * k)
        label = "PRIMARY" if k == 5 else "secondary"
        print(f"\nk = {k} ({label}): own {t['own'].mean():.1f} % | pooled {t['pooled'].mean():.1f} % | "
              f"mean diff {diff.mean():+.1f} pts, helped {(diff > 0).sum()}, harmed {(diff < 0).sum()}, "
              f"one-sided Wilcoxon p = {p:.4f} | at or above chance ({threshold:.0f} %): "
              f"own {(t['own'] >= threshold).sum()}, pooled {(t['pooled'] >= threshold).sum()}")
    print(f"results in {OUT}")


if __name__ == "__main__":
    main()
