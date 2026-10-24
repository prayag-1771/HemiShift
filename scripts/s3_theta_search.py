"""S3 step 4: LesionSim rung 2, learned operator settings (Task B, k = 5, dev patients).

Step 3 showed that real other-patient donors lift the own Riemannian decoder (84.0 % vs 81.7 %
at k = 5) while unmodified healthy donors do not (78.9 %). Hand-set LesionSim closed little of
that gap. Rung 2 (PLAN.md §4.0) searches the operator settings instead:
  alpha  ipsilesional mu ERD kept (O1): 1.0, 0.6, 0.2
  beta   share moved to the contralesional side (O2): 0.0, 0.4, 0.8
  spread trial-to-trial ERD variability, mu and beta (O4b): 0.0, 0.5, 1.0
27 settings, the same for every donor (population mode); (1, 0, 0) is the unmodified donors.
Each setting transforms the healthy donors' MI epochs, and the step 3 pooled decoder is
scored on every dev patient. The setting is chosen honestly: 4 folds over patients, choose on
12, score the other 4. Reported:
  identity      unmodified healthy donors
  oracle        best setting in hindsight (optimistic)
  nested        the honest choice
  gap closed    (nested - identity) / (patients - identity), with "+patients" from step 3
Results: results/summary/s3_theta_search_dev.csv (patients x settings).

Usage: .venv/Scripts/python -I scripts/s3_theta_search.py
"""

import importlib.util
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from hemishift.augment.lesionsim import BETA_HZ, MU_HZ, attenuate_erd, shift_to_contralesional, vary_erd
from hemishift.data.harmonise import TARGET_SFREQ, common_average
from hemishift.eval.stats import paired_comparison
from hemishift.paths import RESULTS_DIR


def _script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s3 = _script("s3_transfer")
augment = _script("s3_augment")

K = 5
SETTINGS = list(product((1.0, 0.6, 0.2), (0.0, 0.4, 0.8), (0.0, 0.5, 1.0)))
IDENTITY = (1.0, 0.0, 0.0)
OUT = RESULTS_DIR / "summary" / "s3_theta_search_dev.csv"


def transformed_mi(X, y, subjects, mirrored, rest_amp, alpha, beta, spread, seed=0):
    """Donor MI epochs under one operator setting, in the order of y == 1."""
    rng = np.random.default_rng(seed)
    mi = np.flatnonzero(y == 1)
    out = X[mi].astype(float)
    if (alpha, beta, spread) == IDENTITY:
        return out.astype(np.float32)
    for s in np.unique(subjects):
        for flipped in (False, True):
            rows = np.flatnonzero((subjects[mi] == s) & (mirrored[mi] == flipped))
            if rows.size == 0:
                continue
            mu, bt = (rest_amp[s][band][s3.MIRROR] if flipped else rest_amp[s][band]
                      for band in (MU_HZ, BETA_HZ))
            x = out[rows]
            if alpha != 1.0:
                x = attenuate_erd(x, TARGET_SFREQ, s3.CHANNELS, mu, alpha=alpha)
            if beta:
                x = shift_to_contralesional(x, TARGET_SFREQ, s3.CHANNELS, mu, beta=beta)
            if spread:
                x = vary_erd(x, TARGET_SFREQ, s3.CHANNELS, mu, spread=spread, rng=rng, band=MU_HZ)
                x = vary_erd(x, TARGET_SFREQ, s3.CHANNELS, bt, spread=spread, rng=rng, band=BETA_HZ)
            out[rows] = common_average(x)
    return out.astype(np.float32)


def main():
    patients = s3.load_patients()
    X_h, y_h, subjects_h, mirrored, rest_amp = s3.load_healthy()
    C_h = s3.premise.covariances(X_h)
    C_patients = {s: s3.premise.covariances(X) for s, (X, _) in patients.items()}
    subjects = list(patients)

    scores = {}  # setting -> per-patient accuracy
    for setting in SETTINGS:
        C = C_h.copy()
        C[y_h == 1] = s3.premise.covariances(transformed_mi(X_h, y_h, subjects_h, mirrored, rest_amp, *setting))
        donors = augment.recentred_by_domain(C, subjects_h)
        accuracy = []
        for s in subjects:
            X, y = patients[s]
            cal, test = s3.calibration_split(len(y), K)
            Cp = C_patients[s]
            p = augment.pooled_proba(Cp[cal], y[cal], Cp[test], donors, y_h)
            accuracy.append(100 * np.mean((p > 0.5).astype(int) == y[test]))
        scores[setting] = np.array(accuracy)
        print(f"alpha {setting[0]:.1f} beta {setting[1]:.1f} spread {setting[2]:.1f}: "
              f"mean {scores[setting].mean():.1f} %")
        pd.DataFrame({f"{a}/{b}/{c}": v for (a, b, c), v in scores.items()}, index=subjects) \
            .rename_axis("subject").to_csv(OUT)

    # Honest choice: 4 folds over patients; choose on 12, score the other 4. Ties keep identity.
    nested = np.empty(len(subjects))
    picks = []
    for held in np.array_split(np.arange(len(subjects)), 4):
        rest = np.setdiff1d(np.arange(len(subjects)), held)
        best = max(SETTINGS, key=lambda st: (scores[st][rest].mean(), st == IDENTITY))
        picks.append(best)
        nested[held] = scores[best][held]
    identity = scores[IDENTITY]
    oracle_setting = max(SETTINGS, key=lambda st: scores[st].mean())
    step3 = pd.read_csv(RESULTS_DIR / "summary" / "s3_augment_dev.csv")
    step3 = step3[step3["k"] == K].set_index("subject").loc[subjects]
    own, real = step3["own"].to_numpy(), step3["+patients"].to_numpy()

    print(f"\nk = {K}, {len(subjects)} dev patients")
    print(f"  own (calibration only)       {own.mean():5.1f} %")
    print(f"  + healthy donors (identity)  {identity.mean():5.1f} %")
    print(f"  + LesionSim, oracle setting  {scores[oracle_setting].mean():5.1f} %  {oracle_setting}")
    print(f"  + LesionSim, nested choice   {nested.mean():5.1f} %  picks {picks}")
    print(f"  + real other patients        {real.mean():5.1f} %  (step 3)")
    r = paired_comparison(nested / 100, identity / 100)
    print(f"  nested vs identity: median {100 * r['median_diff']:+.1f}, p = {r['p']:.3f}, "
          f"helped {(nested > identity).sum()}, harmed {(nested < identity).sum()}")
    gap = real.mean() - identity.mean()
    print(f"  gap closed: {(nested.mean() - identity.mean()) / gap:.0%} of the {gap:.1f}-point "
          f"healthy-to-patient donor gap (oracle {(scores[oracle_setting].mean() - identity.mean()) / gap:.0%})")
    print(f"results in {OUT}")


if __name__ == "__main__":
    main()
