"""S4 (Liu2025) replication of patient pooling (pre-registered in ACHIEVEMENTS.md, 2026-10-31).

Runs once, on all 27 Liu2025 patients' pre-treatment sessions (gait MI vs idle). For each
patient the first k trials per class in recording order calibrate and the rest test.
  own     s3_transfer.own_proba (calibration only)
  pooled  s3_augment.pooled_proba with the other 26 patients as donors, each re-centred on
          its own mean (the S3-confirmed code, unchanged)
H-R1 (primary): pooled > own at k = 5, one-sided Wilcoxon. Secondary: k = 10.

Usage: .venv/Scripts/python -I scripts/s4_replication.py
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from hemishift.data.channels import EXTENDED_TEMPLATE
from hemishift.data.loaders import load_moabb_epochs, moabb_cohort
from hemishift.data.splits import calibration_split
from hemishift.eval.stats import chance_threshold, paired_comparison
from hemishift.paths import HARMONISED_DIR, RESULTS_DIR


def _script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s3 = _script("s3_transfer")
augment = _script("s3_augment")
KS = (5, 10)
MIN_TRIALS = 20
OUT = RESULTS_DIR / "summary" / "s4_replication.csv"


def load_patients():
    """Pre-session epochs per patient in recording order: X (n, 14, 448), y (MI 1, idle 0)."""
    cache = HARMONISED_DIR / "s4_liu2025_pre_t14.npz"
    if cache.exists():
        data = np.load(cache)
        return {int(key[1:]): (data[key], data["y" + key[1:]])
                for key in data.files if key.startswith("X")}
    dataset, sides, group = moabb_cohort("liu2025")
    patients = {}
    for subject in dataset.subject_list:
        try:
            e = load_moabb_epochs(dataset, sides, "liu2025", group, events=("gait_imagery", "rest"),
                                  subjects=[subject], template=EXTENDED_TEMPLATE)
        except Exception as exc:  # pre-registered: unreadable sessions are excluded
            print(f"sub {subject}: excluded, cannot be read ({type(exc).__name__}: {exc})")
            continue
        patients[subject] = (e.X, (e.labels == "gait_imagery").astype(int))
        print(f"sub {subject} ({sides[subject]}): {len(e.labels)} trials, "
              f"{patients[subject][1].sum()} MI", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **{f"X{s}": X for s, (X, _) in patients.items()},
             **{f"y{s}": y for s, (_, y) in patients.items()})
    return patients


def main():
    patients = load_patients()
    for s in list(patients):
        counts = np.bincount(patients[s][1], minlength=2)
        if counts.min() < MIN_TRIALS:
            print(f"sub {s}: excluded, {counts.tolist()} trials per class (pre-registered minimum "
                  f"{MIN_TRIALS})")
            del patients[s]
    print(f"{len(patients)} patients analysed")
    C = {s: s3.premise.covariances(X) for s, (X, _) in patients.items()}
    centred = {s: augment.recentred_by_domain(C[s], np.zeros(len(C[s]))) for s in patients}

    rows = []
    for s, (X, y) in patients.items():
        others = [o for o in patients if o != s]
        donor_C = np.concatenate([centred[o] for o in others])
        donor_y = np.concatenate([patients[o][1] for o in others])
        for k in KS:
            cal, test = calibration_split(y, k, mode="first")
            own = s3.own_proba(X, y, cal, test)
            pooled = augment.pooled_proba(C[s][cal], y[cal], C[s][test], donor_C, donor_y)
            rows.append({"subject": s, "k": k, "n_test": len(test),
                         "own": 100 * np.mean((own > 0.5).astype(int) == y[test]),
                         "pooled": 100 * np.mean((pooled > 0.5).astype(int) == y[test])})
            print(f"sub {s:2d} k={k:2d}: own {rows[-1]['own']:.1f} | pooled {rows[-1]['pooled']:.1f}",
                  flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT, index=False)

    for k in KS:
        t = table[table["k"] == k]
        diff = t["pooled"] - t["own"]
        nonzero = diff[diff != 0]
        p = stats.wilcoxon(nonzero, alternative="greater").pvalue if len(nonzero) else float("nan")
        boot = np.random.default_rng(0).choice(diff.to_numpy(), size=(10_000, len(diff))).mean(axis=1)
        low, high = np.percentile(boot, [2.5, 97.5])
        above = {m: int(sum(acc >= 100 * chance_threshold(n)
                            for acc, n in zip(t[m], t["n_test"]))) for m in ("own", "pooled")}
        label = "PRIMARY (H-R1)" if k == 5 else "secondary"
        print(f"\nk = {k} ({label}): own {t['own'].mean():.1f} % | pooled {t['pooled'].mean():.1f} % | "
              f"mean diff {diff.mean():+.1f} pts (95 % CI {low:+.1f} to {high:+.1f}), median "
              f"{100 * paired_comparison(t['pooled'] / 100, t['own'] / 100)['median_diff']:+.1f}, "
              f"helped {(diff > 0).sum()}, harmed {(diff < 0).sum()}, one-sided Wilcoxon p = {p:.4f} | "
              f"at or above chance: own {above['own']}, pooled {above['pooled']} of {len(t)}")
    print(f"results in {OUT}")


if __name__ == "__main__":
    main()
