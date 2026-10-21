"""S1 step 3f: slow movement-related potentials as a second information source (dev patients).

Every decoder so far used rhythm power (ERD, 8-30 Hz). Imagined movement also produces slow
cortical potentials (the movement-related cortical potential, MRCP, with its lateralised part,
the LRP). The EDF keeps power below 1 Hz, so these are recoverable here.
Features: 0.1-3 Hz, MI window 0.5-4 s, the 8 motor-strip channels (FC3 FCz FC4 C3 Cz C4 CP3 CP4,
common average over those 8), mean amplitude in seven 0.5 s bins -> shrinkage LDA.
Slow EEG is where eye movements live, and EOG is class-dependent in 8 dev patients
(ACHIEVEMENTS.md, data audit), so two controls run alongside:
  eog-only   the same features from the two EOG channels alone (how much eyes alone decode)
  +eogreg    slow features after regressing EOG out of the EEG (label-free, whole recording)
Variants: base (ERD TS + LR, as in step 3a), slow, slow+eogreg, eog-only, and the hybrids
(mean of the ERD and slow probabilities). Blocked 4-fold CV, all 40 trials.

Usage: .venv/Scripts/python -I scripts/s1_slow.py
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from hemishift.data.channels import EXTENDED_TEMPLATE
from hemishift.data.harmonise import TARGET_SFREQ
from hemishift.data.liu2024_raw import EXCLUDED, load_raw
from hemishift.data.splits import blocked_folds
from hemishift.eval.stats import chance_threshold, holm, paired_comparison
from hemishift.paths import CONFIG_DIR, RESULTS_DIR
from hemishift.preprocess.pipeline import Preprocessing, prepare_epochs

_spec = importlib.util.spec_from_file_location("s1_classical",
                                               Path(__file__).with_name("s1_classical.py"))
classical = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(classical)

MOTOR8 = classical.MOTOR8
SLOW_HZ = (0.1, 3.0)
BIN_S = 0.5
OUT = RESULTS_DIR / "summary" / "s1_slow_dev.csv"


def binned(X):
    """Mean amplitude per channel in consecutive BIN_S bins: (n, channels * bins)."""
    width = int(BIN_S * TARGET_SFREQ)
    n_bins = X.shape[-1] // width
    return X[..., :n_bins * width].reshape(*X.shape[:2], n_bins, width).mean(-1).reshape(len(X), -1)


def load(subject):
    raw = load_raw(subject)
    erd, labels, _ = prepare_epochs(raw, Preprocessing(template=EXTENDED_TEMPLATE))
    slow = prepare_epochs(raw, Preprocessing(template=MOTOR8, band=SLOW_HZ))[0]
    slow_reg = prepare_epochs(raw, Preprocessing(template=MOTOR8, band=SLOW_HZ,
                                                 eog_regression=True))[0]
    eog = prepare_epochs(raw, Preprocessing(template=("HEOL", "VEOR"), band=SLOW_HZ,
                                            reference="all"))[0]
    y = (np.asarray(labels) == "right_hand").astype(int)
    return {"erd": classical.covariances(erd), "slow": binned(slow), "slow+eogreg": binned(slow_reg),
            "eog-only": binned(eog)}, y


def lda_proba(F, y, train, test):
    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(F[train], y[train])
    return model.predict_proba(F[test])[:, 1]


def variants(data, y, train, test):
    out = {"base": classical.ts_proba(data["erd"], y, train, test, classical.lr())}
    for name in ("slow", "slow+eogreg", "eog-only"):
        out[name] = lda_proba(data[name], y, train, test)
    out["hybrid"] = (out["base"] + out["slow"]) / 2
    out["hybrid+eogreg"] = (out["base"] + out["slow+eogreg"]) / 2
    return out


def main():
    split = json.loads((CONFIG_DIR / "splits" / "liu2024_locked_test.json").read_text())
    rows = []
    for subject in split["dev"]:
        if subject in EXCLUDED:
            continue
        data, y = load(subject)
        hits = {}
        for train, test in blocked_folds(len(y), 4):
            for name, p in variants(data, y, train, test).items():
                hits[name] = hits.get(name, 0) + int(((p > 0.5).astype(int) == y[test]).sum())
        rows.append({"subject": subject, **{k: 100 * v / len(y) for k, v in hits.items()}})
        print(f"sub {subject:2d}: " + " ".join(f"{k} {v:.0f}" for k, v in rows[-1].items()
                                                if k != "subject"))
    table = pd.DataFrame(rows)
    table.to_csv(OUT, index=False)

    threshold = 100 * chance_threshold(40)
    names = [c for c in table.columns if c not in ("subject", "base")]
    results = {name: paired_comparison(table[name] / 100, table["base"] / 100) for name in names}
    adjusted = dict(zip(names, holm([results[n]["p"] for n in names])))
    print(f"\n{len(table)} dev patients, blocked 4-fold CV; base mean {table['base'].mean():.1f} %, "
          f"{(table['base'] >= threshold).sum()} at or above {threshold:.0f} %")
    for name in names:
        r = results[name]
        print(f"{name:14s} mean {table[name].mean():5.1f} % | >= chance {(table[name] >= threshold).sum():2d}"
              f" | vs base median {100 * r['median_diff']:+.1f}, p = {r['p']:.3f} (Holm {adjusted[name]:.3f})"
              f", helped {(table[name] > table['base']).sum()}, harmed {(table[name] < table['base']).sum()}")
    print(f"results in {OUT}")


if __name__ == "__main__":
    main()
