"""S1 deep-dive step 3a: is there left/right information outside the classic place?

Development patients only. Each patient's own decoder (tangent space + LR on 8-30 Hz OAS
covariances, extended 14-channel template, CAR over the picked channels) is scored with
blocked 4-fold CV: 10 consecutive trials per fold, all 40 scored, references from the training
folds only. Variants:
  MI 0.5-4 s                  the classic window (reference)
  rebound 4-6 s               post-imagery beta rebound, during the break
  MI + rebound 0.5-6 s
  healthy side, MI            contralesional hemisphere + midline only (Mansour et al. 2022)
  healthy side, MI + rebound
Exploratory; paired Wilcoxon against the reference, Holm-corrected.

Usage: .venv/Scripts/python -I scripts/s1_information.py
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hemishift.data.channels import EXTENDED_TEMPLATE, hemisphere
from hemishift.data.liu2024_raw import EXCLUDED, load_raw, participants
from hemishift.data.splits import blocked_folds
from hemishift.eval.stats import chance_threshold, holm, paired_comparison
from hemishift.paths import CONFIG_DIR, RESULTS_DIR
from hemishift.preprocess.pipeline import Preprocessing, prepare_epochs

_spec = importlib.util.spec_from_file_location("premise_check",
                                               Path(__file__).with_name("premise_check.py"))
premise = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(premise)

WINDOWS = {"MI 0.5-4 s": (0.5, 4.0), "rebound 4-6 s": (4.0, 6.0), "MI + rebound 0.5-6 s": (0.5, 6.0)}


def blocked_accuracy(X, y, n_folds=4):
    C = premise.covariances(X)
    correct = 0
    for train, test in blocked_folds(len(y), n_folds):
        reference = premise.recenter(C, train)
        model = premise.model().fit(reference[train], y[train])
        correct += int((model.predict(reference[test]) == y[test]).sum())
    return 100 * correct / len(y)


def healthy_side_channels(paralysis_side):
    """Template indices of the healthy (contralesional) hemisphere plus the midline.

    The lesion is opposite the paralysed side, so the healthy hemisphere is on the same side as
    the paralysis.
    """
    healthy = "left" if paralysis_side == "left" else "right"
    return [i for i, ch in enumerate(EXTENDED_TEMPLATE) if hemisphere(ch) in (healthy, "midline")]


def main():
    split = json.loads((CONFIG_DIR / "splits" / "liu2024_locked_test.json").read_text())
    clinical = participants()
    rows = []
    for subject in split["dev"]:
        if subject in EXCLUDED:
            continue
        raw = load_raw(subject)
        keep = healthy_side_channels(clinical.loc[subject, "ParalysisSide"])
        row = {"subject": subject}
        for name, window in WINDOWS.items():
            X, y, _ = prepare_epochs(raw, Preprocessing(template=EXTENDED_TEMPLATE, window=window))
            row[name] = blocked_accuracy(X, y)
            if name != "rebound 4-6 s":
                row[f"healthy side, {name.split(' 0')[0]}"] = blocked_accuracy(X[:, keep], y)
        rows.append(row)
        print(f"sub {subject:2d}: " + " | ".join(f"{k} {v:.0f}" for k, v in row.items() if k != "subject"))
    table = pd.DataFrame(rows)
    out = RESULTS_DIR / "summary" / "s1_information_dev.csv"
    table.to_csv(out, index=False)

    variants = [c for c in table.columns if c != "subject"]
    base = variants[0]
    threshold = 100 * chance_threshold(40)
    print(f"\n{len(table)} dev patients, blocked 4-fold CV over 40 trials "
          f"(per-patient chance threshold {threshold:.0f} %)")
    print(f"{base:28s} mean {table[base].mean():5.1f} % | above chance {(table[base] >= threshold).sum()}")
    tests = {v: paired_comparison(table[v] / 100, table[base] / 100) for v in variants[1:]}
    adjusted = dict(zip(tests, holm([r["p"] for r in tests.values()])))
    for v, r in tests.items():
        print(f"{v:28s} mean {table[v].mean():5.1f} % | above chance {(table[v] >= threshold).sum()} | "
              f"vs MI median {100 * r['median_diff']:+.1f} pts, p = {r['p']:.3f} (Holm {adjusted[v]:.3f})")
    print(f"results in {out}")


if __name__ == "__main__":
    main()
