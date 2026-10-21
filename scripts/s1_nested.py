"""S1 step 3d: per-patient tuning done honestly (nested selection), development patients only.

Each patient picks a time window and frequency band from a menu, but only from its own
training trials (inner blocked CV), and the pick is scored on held-out trials it never saw.
Outer: blocked 4-fold CV (10 consecutive trials per fold, all 40 scored). Inner: the 3
training blocks, each held out once. Menu = 4 windows x (3 single bands + 1 filter bank of
all three) = 16 candidates; model = tangent space + LR, references from training trials only.
Reported per patient:
  fixed    MI 0.5-4 s, 8-30 Hz (the default used so far)
  nested   the honest per-patient choice
  oracle   the best candidate in hindsight (optimistic; shows how much "best of" is luck)

Usage: .venv/Scripts/python -I scripts/s1_nested.py
"""

import json
from itertools import product

import numpy as np
import pandas as pd
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.mean import mean_riemann
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression

from hemishift.data.channels import EXTENDED_TEMPLATE
from hemishift.data.harmonise import TARGET_SFREQ
from hemishift.data.liu2024_raw import EXCLUDED, load_raw
from hemishift.data.splits import blocked_folds
from hemishift.eval.stats import chance_threshold, paired_comparison
from hemishift.paths import CONFIG_DIR, RESULTS_DIR
from hemishift.preprocess.pipeline import Preprocessing, prepare_epochs

WINDOWS = {"MI 0.5-4": (0.5, 4.0), "MI 0.5-2.5": (0.5, 2.5), "MI 2-4": (2.0, 4.0),
           "rebound 4-6": (4.0, 6.0)}
BANDS = {"mu 8-13": (8.0, 13.0), "beta 13-30": (13.0, 30.0), "8-30": (8.0, 30.0)}
CANDIDATES = [(w, (b,)) for w, b in product(WINDOWS, BANDS)] + \
             [(w, tuple(BANDS)) for w in WINDOWS]  # filter bank: all three bands together
DEFAULT = ("MI 0.5-4", ("8-30",))


def covariance_sets(raw):
    """Covariances for every (window, band), all 40 trials; and the labels."""
    sets, labels = {}, None
    for w_name, window in WINDOWS.items():
        X, labels, _ = prepare_epochs(raw, Preprocessing(template=EXTENDED_TEMPLATE, window=window))
        for b_name, band in BANDS.items():
            sos = butter(4, band, btype="bandpass", fs=TARGET_SFREQ, output="sos")
            sets[(w_name, b_name)] = Covariances("oas").transform(sosfiltfilt(sos, X, axis=-1))
    return sets, labels


def features(sets, candidate, train, rows):
    """Tangent vectors (one block per band) for `rows`, with references from `train`."""
    window, bands = candidate
    blocks = []
    for band in bands:
        C = sets[(window, band)]
        W = invsqrtm(mean_riemann(C[train]))
        aligned = W @ C @ W
        space = TangentSpace(metric="riemann").fit(aligned[train])
        blocks.append(space.transform(aligned[rows]))
    return np.hstack(blocks)


def fit_predict(sets, y, candidate, train, test):
    model = LogisticRegression(C=1.0, max_iter=1000).fit(features(sets, candidate, train, train), y[train])
    return model.predict(features(sets, candidate, train, test))


def evaluate(sets, y):
    folds = list(blocked_folds(len(y), 4))
    hits = {"fixed": 0, "nested": 0}
    per_candidate = {c: 0 for c in CANDIDATES}
    chosen = []
    for train, test in folds:
        inner_blocks = np.array_split(train, 3)  # the 3 training blocks, in time order
        inner = {}
        for candidate in CANDIDATES:
            correct = 0
            for i, held in enumerate(inner_blocks):
                fit_on = np.concatenate([b for j, b in enumerate(inner_blocks) if j != i])
                correct += int((fit_predict(sets, y, candidate, fit_on, held) == y[held]).sum())
            inner[candidate] = correct
        best = max(CANDIDATES, key=lambda c: (inner[c], c == DEFAULT))  # ties keep the default
        chosen.append(best)
        hits["nested"] += int((fit_predict(sets, y, best, train, test) == y[test]).sum())
        for candidate in CANDIDATES:
            per_candidate[candidate] += int((fit_predict(sets, y, candidate, train, test) == y[test]).sum())
    hits["fixed"] = per_candidate[DEFAULT]
    result = {k: 100 * v / len(y) for k, v in hits.items()}
    result["oracle"] = 100 * max(per_candidate.values()) / len(y)
    result["choices"] = "; ".join(f"{w} {'+'.join(b)}" for w, b in chosen)
    return result


def main():
    split = json.loads((CONFIG_DIR / "splits" / "liu2024_locked_test.json").read_text())
    rows = []
    for subject in split["dev"]:
        if subject in EXCLUDED:
            continue
        sets, labels = covariance_sets(load_raw(subject))
        row = {"subject": subject, **evaluate(sets, np.asarray(labels))}
        rows.append(row)
        print(f"sub {subject:2d}: fixed {row['fixed']:.0f} | nested {row['nested']:.0f} | "
              f"oracle {row['oracle']:.0f} | picks: {row['choices']}")
    table = pd.DataFrame(rows)
    out = RESULTS_DIR / "summary" / "s1_nested_dev.csv"
    table.to_csv(out, index=False)
    threshold = 100 * chance_threshold(40)
    print(f"\n{len(table)} dev patients, blocked 4-fold outer CV (chance threshold {threshold:.0f} %)")
    for name in ("fixed", "nested", "oracle"):
        print(f"{name:7s} mean {table[name].mean():5.1f} % | above chance {(table[name] >= threshold).sum()}")
    r = paired_comparison(table["nested"] / 100, table["fixed"] / 100)
    print(f"nested vs fixed: median {100 * r['median_diff']:+.1f} pts, p = {r['p']:.3f}, helped "
          f"{(table['nested'] > table['fixed']).sum()}, harmed {(table['nested'] < table['fixed']).sum()}")
    print(f"results in {out}")


if __name__ == "__main__":
    main()
