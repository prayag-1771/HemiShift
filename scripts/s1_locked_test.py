"""S1 locked-test evaluation of the frozen ens16 pipeline (pre-registered in ACHIEVEMENTS.md).

Runs once, on the 15 locked-test patients only. The pipeline code is imported unchanged from
s1_classical.py (step 3e): the 16 window/band TS + LR candidates and their mean probability,
against the base decoder (MI 0.5-4 s, 8-30 Hz). Blocked 4-fold CV per patient, first 39 trials.

Usage: .venv/Scripts/python -I scripts/s1_locked_test.py
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from hemishift.data.liu2024_raw import EXCLUDED, load_raw
from hemishift.data.splits import blocked_folds
from hemishift.eval.stats import chance_threshold, paired_comparison
from hemishift.paths import CONFIG_DIR, RESULTS_DIR

_spec = importlib.util.spec_from_file_location("s1_classical",
                                               Path(__file__).with_name("s1_classical.py"))
classical = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(classical)

OUT = RESULTS_DIR / "summary" / "s1_locked_test.csv"


def load(subject):
    """The 12 (window, band) covariance sets, cut to the trials all windows share."""
    raw = load_raw(subject)
    epochs, labels = {}, None
    for name, window in classical.WINDOWS.items():
        X, y, _ = classical.prepare_epochs(raw, classical.Preprocessing(
            template=classical.EXTENDED_TEMPLATE, window=window))
        epochs[name] = X
        labels = y if name == "MI 0.5-4" else labels
    n = min(len(X) for X in epochs.values())
    data = {(w, b): classical.covariances(epochs[w][:n], band)
            for w in classical.WINDOWS for b, band in classical.BANDS.items()}
    return data, (np.asarray(labels[:n]) == "right_hand").astype(int)


def main():
    split = json.loads((CONFIG_DIR / "splits" / "liu2024_locked_test.json").read_text())
    print(f"locked test patients: {split['test']} (split hash {split['hash']})")
    rows = []
    for subject in split["test"]:
        if subject in EXCLUDED:
            print(f"sub {subject}: excluded by the label-free quality rule")
            continue
        data, y = load(subject)
        hits = {"base": 0, "ens16": 0}
        for train, test in blocked_folds(len(y), 4):
            probs = [classical.candidate_proba(data, y, c, train, test) for c in classical.CANDIDATES]
            base = classical.candidate_proba(data, y, ("MI 0.5-4", ("8-30",)), train, test)
            hits["base"] += int(((base > 0.5).astype(int) == y[test]).sum())
            hits["ens16"] += int(((np.mean(probs, axis=0) > 0.5).astype(int) == y[test]).sum())
        rows.append({"subject": subject, "n_trials": len(y),
                     **{k: 100 * v / len(y) for k, v in hits.items()}})
        print(f"sub {subject:2d}: base {rows[-1]['base']:.1f} | ens16 {rows[-1]['ens16']:.1f}")
    table = pd.DataFrame(rows)
    table.to_csv(OUT, index=False)

    diff = table["ens16"] - table["base"]
    one_sided = stats.wilcoxon(diff[diff != 0], alternative="greater").pvalue
    r = paired_comparison(table["ens16"] / 100, table["base"] / 100)
    threshold = 100 * chance_threshold(39)
    print(f"\n{len(table)} locked-test patients, blocked 4-fold CV (chance threshold {threshold:.0f} %)")
    for name in ("base", "ens16"):
        print(f"{name:6s} mean {table[name].mean():5.1f} % | at or above chance {(table[name] >= threshold).sum()}")
    print(f"ens16 - base: mean {diff.mean():+.1f} pts, median {100 * r['median_diff']:+.1f} "
          f"(95 % CI {100 * r['ci95'][0]:+.1f} to {100 * r['ci95'][1]:+.1f}), helped {(diff > 0).sum()}, "
          f"harmed {(diff < 0).sum()}, one-sided Wilcoxon p = {one_sided:.4f} (pre-registered), "
          f"two-sided p = {r['p']:.4f}")
    print(f"results in {OUT}")


if __name__ == "__main__":
    main()
