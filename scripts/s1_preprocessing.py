"""S1 deep-dive step 2: which preprocessing helps a patient's own decoder? (dev patients only)

Each variant changes one thing at a time relative to the one before it. All are scored with
the same R0 (tangent space + LR on 8-30 Hz OAS covariances, first 10 + 10 trials, tested on
the remaining 20 in recording order), and compared with the baseline by paired Wilcoxon
across development patients. Excluded recordings (liu2024_raw.EXCLUDED) are skipped.
Exploratory: this is where choices are made, so the locked test set stays untouched.

Usage: .venv/Scripts/python -I scripts/s1_preprocessing.py
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hemishift.data.channels import EXTENDED_TEMPLATE
from hemishift.data.liu2024_raw import EXCLUDED, load_raw
from hemishift.data.splits import calibration_split
from hemishift.eval.stats import holm, paired_comparison
from hemishift.paths import CONFIG_DIR, RESULTS_DIR
from hemishift.preprocess.pipeline import Preprocessing, prepare_epochs

_spec = importlib.util.spec_from_file_location("premise_check",
                                               Path(__file__).with_name("premise_check.py"))
premise = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(premise)

ALL_29 = ("Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FCz", "FC3", "FC4", "FT7", "FT8", "Cz",
          "C3", "C4", "T7", "T8", "CP3", "CP4", "TP7", "TP8", "Pz", "P3", "P4", "P7", "P8",
          "Oz", "O1", "O2")
EXT = EXTENDED_TEMPLATE
VARIANTS = {
    "baseline (core 9, CAR over them)": Preprocessing(),
    "extended 14": Preprocessing(template=EXT),
    "extended 14, CAR over all 29": Preprocessing(template=EXT, reference="all"),
    "+ interpolate faulty channels": Preprocessing(template=EXT, reference="all", interpolate=True),
    "+ EOG regression": Preprocessing(template=EXT, reference="all", interpolate=True,
                                      eog_regression=True),
    "Laplacian instead of CAR": Preprocessing(template=EXT, reference="laplacian",
                                              interpolate=True, eog_regression=True),
    "all 29 channels (confound risk)": Preprocessing(template=ALL_29, reference="all",
                                                     interpolate=True, eog_regression=True),
}


def own_accuracy(X, y, k=10):
    calib, test = calibration_split(y, k, mode="first")
    reference = premise.recenter(premise.covariances(X), calib)
    model = premise.model().fit(reference[calib], y[calib])
    return 100 * np.mean(model.predict(reference[test]) == y[test])


def main():
    split = json.loads((CONFIG_DIR / "splits" / "liu2024_locked_test.json").read_text())
    rows = []
    for subject in split["dev"]:
        if subject in EXCLUDED:
            continue
        raw = load_raw(subject)
        row = {"subject": subject}
        for name, config in VARIANTS.items():
            X, y, info = prepare_epochs(raw, config)
            row[name] = own_accuracy(X, y)
        rows.append(row)
        print(f"sub {subject:2d}: " + " | ".join(f"{row[n]:.0f}" for n in VARIANTS))
    table = pd.DataFrame(rows)
    out = RESULTS_DIR / "summary" / "s1_preprocessing_dev.csv"
    table.to_csv(out, index=False)

    base = list(VARIANTS)[0]
    tests = {name: paired_comparison(table[name] / 100, table[base] / 100) for name in list(VARIANTS)[1:]}
    adjusted = dict(zip(tests, holm([r["p"] for r in tests.values()])))
    print(f"\n{len(table)} dev patients | {base}: mean {table[base].mean():.1f} %")
    for name, r in tests.items():
        print(f"{name:34s} mean {table[name].mean():5.1f} % | vs baseline median {100 * r['median_diff']:+.1f} "
              f"pts, p = {r['p']:.3f} (Holm {adjusted[name]:.3f}) | helped {(table[name] > table[base]).sum()}, "
              f"harmed {(table[name] < table[base]).sum()}")
    print(f"results in {out}")


if __name__ == "__main__":
    main()
