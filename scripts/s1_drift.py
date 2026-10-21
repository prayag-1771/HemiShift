"""S1 step 3b: does correcting within-session drift help? (development patients only)

Two patients score significantly below chance, which suggests the signal drifts during the
session and flips the learned rule. Blocked 4-fold CV (10 consecutive trials per fold, 40
scored), own decoder (TS + LR, 8-30 Hz, 14 channels), MI 0.5-4 s and rebound 4-6 s:
  global   every trial re-centred by the training folds' Riemannian mean (as before)
  block    every block of 10 trials re-centred by its own Riemannian mean. This is label-free;
           each block holds 5 left + 5 right trials, so no class information leaks. Online it
           corresponds to adaptive re-centring on a running mean of recent trials.

--causal runs the realistic version instead: calibrate on the first 10 + 10 trials, then
decode the last 20 in order, re-centring each trial on the 10 trials just before it.

Usage: .venv/Scripts/python -I scripts/s1_drift.py [--causal]
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hemishift.data.channels import EXTENDED_TEMPLATE
from hemishift.data.liu2024_raw import EXCLUDED, load_raw
from hemishift.data.splits import blocked_folds, calibration_split
from hemishift.eval.stats import chance_threshold, paired_comparison
from hemishift.paths import CONFIG_DIR, RESULTS_DIR
from hemishift.preprocess.pipeline import Preprocessing, prepare_epochs

_spec = importlib.util.spec_from_file_location("premise_check",
                                               Path(__file__).with_name("premise_check.py"))
premise = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(premise)

WINDOWS = {"MI 0.5-4 s": (0.5, 4.0), "rebound 4-6 s": (4.0, 6.0)}


def accuracy(C, y, mode, n_folds=4):
    folds = list(blocked_folds(len(y), n_folds))
    if mode == "block":  # each block centred on itself, unlabeled
        centred = np.empty_like(C)
        for _, block in folds:
            centred[block] = premise.recenter(C[block], np.arange(block.size))[: block.size]
    correct = 0
    for train, test in folds:
        aligned = premise.recenter(C, train) if mode == "global" else centred
        model = premise.model().fit(aligned[train], y[train])
        correct += int((model.predict(aligned[test]) == y[test]).sum())
    return 100 * correct / len(y)


def local_recenter(C, t, memory):
    """Trial t re-centred on the mean of the `memory` trials just before it (past only; the
    first trials of a session use the first `memory` trials, all of them calibration data)."""
    from pyriemann.utils.base import invsqrtm
    from pyriemann.utils.mean import mean_riemann

    start = max(0, t - memory)
    window = np.arange(start, t) if t - start >= memory else np.arange(memory)
    W = invsqrtm(mean_riemann(C[window]))
    return W @ C[t] @ W


def causal_accuracy(C, y, k=10, memory=10, adaptive=True, consistent=False):
    """Realistic session: calibrate on the first k trials per class, then decode the rest in
    order. adaptive=True re-centres each new trial on the Riemannian mean of the `memory`
    trials just before it (past only, unlabelled), which is what a real-time BCI can do.
    adaptive=False keeps the calibration mean (static). consistent=True normalises the
    calibration trials the same local way, so training and test match."""
    from pyriemann.utils.base import invsqrtm
    from pyriemann.utils.mean import mean_riemann

    calib, test = calibration_split(y, k, mode="first")
    if consistent:
        train = np.stack([local_recenter(C, t, memory) for t in calib])
    else:
        train = premise.recenter(C, calib)[calib]
    model = premise.model().fit(train, y[calib])
    static = invsqrtm(mean_riemann(C[calib]))
    correct = 0
    for t in test:
        if adaptive:
            trial = local_recenter(C, t, memory)
        else:
            trial = static @ C[t] @ static
        correct += int(model.predict(trial[None])[0] == y[t])
    return 100 * correct / len(test)


def main_causal():
    """Causal check of the block result: first 10 + 10 calibrate, last 20 decoded in order."""
    split = json.loads((CONFIG_DIR / "splits" / "liu2024_locked_test.json").read_text())
    rows = []
    for subject in split["dev"]:
        if subject in EXCLUDED:
            continue
        raw = load_raw(subject)
        row = {"subject": subject}
        for name, window in WINDOWS.items():
            X, y, _ = prepare_epochs(raw, Preprocessing(template=EXTENDED_TEMPLATE, window=window))
            C = premise.covariances(X)
            row[f"{name}, static"] = causal_accuracy(C, y, adaptive=False)
            row[f"{name}, causal adaptive"] = causal_accuracy(C, y, adaptive=True)
            row[f"{name}, causal consistent"] = causal_accuracy(C, y, adaptive=True, consistent=True)
        rows.append(row)
        print(f"sub {subject:2d}: " + " | ".join(f"{k} {v:.0f}" for k, v in row.items() if k != "subject"))
    table = pd.DataFrame(rows)
    out = RESULTS_DIR / "summary" / "s1_drift_causal_dev.csv"
    table.to_csv(out, index=False)
    threshold = 100 * chance_threshold(20)
    print(f"\n{len(table)} dev patients, first 20 trials calibrate, last 20 decoded in order "
          f"(per-patient chance threshold {threshold:.0f} %; results in {out})")
    for name in WINDOWS:
        s = table[f"{name}, static"]
        for variant in ("causal adaptive", "causal consistent"):
            a = table[f"{name}, {variant}"]
            r = paired_comparison(a / 100, s / 100)
            print(f"{name:14s} static {s.mean():.1f} % -> {variant} {a.mean():.1f} % | median "
                  f"{100 * r['median_diff']:+.1f} pts, p = {r['p']:.4f}, helped {(a > s).sum()}, "
                  f"harmed {(a < s).sum()}")


def main():
    split = json.loads((CONFIG_DIR / "splits" / "liu2024_locked_test.json").read_text())
    rows = []
    for subject in split["dev"]:
        if subject in EXCLUDED:
            continue
        raw = load_raw(subject)
        row = {"subject": subject}
        for name, window in WINDOWS.items():
            X, y, _ = prepare_epochs(raw, Preprocessing(template=EXTENDED_TEMPLATE, window=window))
            C = premise.covariances(X)
            for mode in ("global", "block"):
                row[f"{name}, {mode}"] = accuracy(C, y, mode)
        rows.append(row)
        print(f"sub {subject:2d}: " + " | ".join(f"{k} {v:.0f}" for k, v in row.items() if k != "subject"))
    table = pd.DataFrame(rows)
    out = RESULTS_DIR / "summary" / "s1_drift_dev.csv"
    table.to_csv(out, index=False)

    threshold = 100 * chance_threshold(40)
    print(f"\n{len(table)} dev patients (results in {out})")
    for name in WINDOWS:
        g, b = table[f"{name}, global"], table[f"{name}, block"]
        r = paired_comparison(b / 100, g / 100)
        print(f"{name:14s} global {g.mean():.1f} % ({(g >= threshold).sum()} above chance, "
              f"{(g <= 30).sum()} below) -> block {b.mean():.1f} % ({(b >= threshold).sum()} above, "
              f"{(b <= 30).sum()} below) | median {100 * r['median_diff']:+.1f} pts, p = {r['p']:.4f}, "
              f"helped {(b > g).sum()}, harmed {(b < g).sum()}")


if __name__ == "__main__":
    import sys

    main_causal() if "--causal" in sys.argv else main()
