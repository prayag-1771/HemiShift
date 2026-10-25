"""S1 step 5: does patient pooling, confirmed on S3 Task B, also help Task A? (dev patients)

S3's locked test confirmed that pooling other patients' data into a re-centred Riemannian
decoder beats the patient's own decoder at k = 5 (ACHIEVEMENTS.md, 2026-10-24). Here the same
code (s3_augment.pooled_proba: each domain re-centred on its own mean, tangent space at the
identity, donors together weigh as much as the calibration) runs on S1 Task A (affected vs
unaffected hand, lesion normalised so labels mean the same in every patient).
Protocol P3: the first k trials per class calibrate (alternating trials, so the first 2k
trials), the remaining trials test; k = 5, 10. Donors = the other 33 dev patients (all trials).
Dev only: the S1 locked test has been used.

Usage: .venv/Scripts/python -I scripts/s1_pooling.py
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hemishift.data.channels import EXTENDED_TEMPLATE
from hemishift.data.harmonise import normalise_lesion_side
from hemishift.data.liu2024_raw import EXCLUDED, load_raw, participants
from hemishift.eval.stats import paired_comparison
from hemishift.paths import CONFIG_DIR, RESULTS_DIR
from hemishift.preprocess.pipeline import Preprocessing, prepare_epochs


def _script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


augment = _script("s3_augment")
premise = augment.s3.premise
KS = (5, 10)
OUT = RESULTS_DIR / "summary" / "s1_pooling_dev.csv"


def load_patients():
    split = json.loads((CONFIG_DIR / "splits" / "liu2024_locked_test.json").read_text())
    clinical = participants()
    patients = {}
    for subject in split["dev"]:
        if subject in EXCLUDED:
            continue
        X, labels, _ = prepare_epochs(load_raw(subject), Preprocessing(template=EXTENDED_TEMPLATE))
        side = "L" if clinical.loc[subject, "ParalysisSide"] == "left" else "R"
        X, labels = normalise_lesion_side(X, labels, EXTENDED_TEMPLATE, side)
        patients[subject] = (premise.covariances(X), (np.asarray(labels) == "right_hand").astype(int))
    return patients


def main():
    patients = load_patients()
    recentred = {s: augment.recentred_by_domain(C, np.zeros(len(C))) for s, (C, _) in patients.items()}
    rows = []
    for subject, (C, y) in patients.items():
        others = [s for s in patients if s != subject]
        donor_C = np.concatenate([recentred[s] for s in others])
        donor_y = np.concatenate([patients[s][1] for s in others])
        for k in KS:
            cal, test = np.arange(2 * k), np.arange(2 * k, len(y))
            A = premise.recenter(C, cal)
            own = premise.model().fit(A[cal], y[cal]).predict_proba(A[test])[:, 1]
            pooled = augment.pooled_proba(C[cal], y[cal], C[test], donor_C, donor_y)
            rows.append({"subject": subject, "k": k,
                         "own": 100 * np.mean((own > 0.5).astype(int) == y[test]),
                         "pooled": 100 * np.mean((pooled > 0.5).astype(int) == y[test])})
        print(f"sub {subject:2d}: " + " | ".join(f"k={r['k']} own {r['own']:.0f} pooled {r['pooled']:.0f}"
                                                for r in rows[-len(KS):]))
    table = pd.DataFrame(rows)
    table.to_csv(OUT, index=False)
    for k in KS:
        t = table[table["k"] == k]
        r = paired_comparison(t["pooled"] / 100, t["own"] / 100)
        print(f"\nk = {k} per class ({len(t)} dev patients, {40 - 2 * k} test trials): own {t['own'].mean():.1f} % "
              f"| pooled {t['pooled'].mean():.1f} % | median {100 * r['median_diff']:+.1f}, p = {r['p']:.3f}, "
              f"helped {(t['pooled'] > t['own']).sum()}, harmed {(t['pooled'] < t['own']).sum()}")
    print(f"results in {OUT}")


if __name__ == "__main__":
    main()
