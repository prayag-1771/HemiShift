"""External test of the combination decoders at a second hospital: S1 (Beijing), pre-registered
in ACHIEVEMENTS.md (2026-11-01). Runs once.

S1 differs from the cohorts the decoders were developed on in hospital, amplifier (a portable
29-channel system), stage (acute) and task (affected- vs unaffected-hand imagery instead of
imagery vs rest), and it has one session per patient. The combination decoders were never run
on S1. The S1 patients were used before for other decoders (development, and two pre-registered
locked-test analyses), so all 49 usable patients are tested together here, like S4.
Frozen decoders (hemishift.models.combo, 14-channel template; S1 lacks the 28-channel strip):
  own                the patient's own decoder, 8-30 Hz
  pooled_bank        the other 48 patients + the mu / low-beta / high-beta bank
  own_sup            co-adaptive recalibration of the own decoder (the known technique)
  own_sup_bank, pooled_sup_bank, prior_sup_bank
  pooled_unsup_bank  pooled bank with online re-centring, no labels after calibration
  bma                online model averaging over pooled_bank, own_sup_bank, pooled_sup_bank and
                     prior_sup_bank (the 14-channel members of the S3 system)
S1's trials alternate between the hands, so every online decoder and the averaging weights are
updated only between (left, right) pairs (block = 2; the audit of 2026-11-01).
Protocol P3: the first k trials per class (the first 2k trials) calibrate, k = 5 primary, 10
descriptive. Hypotheses, one-sided Wilcoxon over patients, fixed sequence at alpha = 0.05:
  H-E1  bma > own
  H-E2  bma > own_sup
  H-E3  pooled_unsup_bank > own
Results: results/summary/s1_combo_external.csv.

Usage: .venv/Scripts/python -I scripts/s1_combo_external.py [--dry-run]
  --dry-run runs 4 patients with shuffled labels (checks the code path, shows nothing real).
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from hemishift.data.channels import EXTENDED_TEMPLATE
from hemishift.data.harmonise import normalise_lesion_side
from hemishift.data.liu2024_raw import EXCLUDED, load_raw, participants
from hemishift.data.splits import calibration_split
from hemishift.eval.stats import clean_differences, signed_rank_p
from hemishift.models.combo import (bma, centred, online, own_exact, population_prior, static,
                                    tangent_all)
from hemishift.paths import CONFIG_DIR, HARMONISED_DIR, RESULTS_DIR
from hemishift.preprocess.pipeline import Preprocessing, prepare_epochs

KS = (5, 10)
BLOCK = 2
ALPHA = 0.05
BANDS = {"830": (8.0, 30.0), "mu": (8.0, 13.0), "b1": (13.0, 20.0), "b2": (20.0, 30.0)}
BANK = ("mu", "b1", "b2")
EXPERTS = ("pooled_bank", "own_sup_bank", "pooled_sup_bank", "prior_sup_bank")
HYPOTHESES = (("H-E1", "bma", "own"), ("H-E2", "bma", "own_sup"), ("H-E3", "pooled_unsup_bank", "own"))
CACHE = HARMONISED_DIR / "s1_combo_t14.npz"
OUT = RESULTS_DIR / "summary" / "s1_combo_external.csv"


def covariances(X, band):
    from pyriemann.estimation import Covariances
    from scipy.signal import butter, sosfiltfilt

    from hemishift.data.harmonise import TARGET_SFREQ
    sos = butter(4, band, btype="bandpass", fs=TARGET_SFREQ, output="sos")
    return Covariances(estimator="oas").transform(sosfiltfilt(sos, X, axis=-1))


def load():
    """Covariances per view and labels (affected hand 1) of all usable S1 patients, cached."""
    if not CACHE.exists():
        split = json.loads((CONFIG_DIR / "splits" / "liu2024_locked_test.json").read_text())
        clinical = participants()
        arrays = {}
        for subject in sorted(split["dev"] + split["test"]):
            if subject in EXCLUDED:
                continue
            X, labels, _ = prepare_epochs(load_raw(subject), Preprocessing(template=EXTENDED_TEMPLATE))
            side = "L" if clinical.loc[subject, "ParalysisSide"] == "left" else "R"
            X, labels = normalise_lesion_side(X, labels, EXTENDED_TEMPLATE, side)
            arrays[f"y|{subject}"] = (np.asarray(labels) == "right_hand").astype(int)
            for name, band in BANDS.items():
                arrays[f"{name}|{subject}"] = covariances(np.asarray(X, dtype=float), band)
            print(f"  S1 sub {subject}: {len(labels)} trials", flush=True)
        np.savez(CACHE, **arrays)
    data = np.load(CACHE)
    cov, labels = {}, {}
    for key in data.files:
        view, s = key.split("|")
        if view == "y":
            labels[int(s)] = data[key]
        else:
            cov.setdefault(int(s), {})[view] = data[key]
    return cov, labels


def decode(s, cov, y, others, others_C, prior):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    rows = []
    for k in KS:
        cal, test = calibration_split(y, k, mode="first")
        yt = y[test]
        p = {
            "own": own_exact(cov["830"], y, cal, test),
            "pooled_bank": static(cov, y, cal, test, BANK, [others["bank"]]),
            "own_sup": online(cov, y, cal, test, ("830",), block=BLOCK),
            "own_sup_bank": online(cov, y, cal, test, BANK, block=BLOCK),
            "pooled_sup_bank": online(cov, y, cal, test, BANK, groups=[others["bank"]], block=BLOCK),
            "prior_sup_bank": online(cov, y, cal, test, BANK, prior=prior, block=BLOCK),
            "pooled_unsup_bank": online(cov, y, cal, test, BANK, labels="cal", groups=[others["bank"]],
                                        block=BLOCK),
        }
        p["bma"] = bma([p[e] for e in EXPERTS], yt, BLOCK)
        rows.append({"subject": s, "k": k, "n_test": len(test),
                     **{n: 100 * np.mean((q > 0.5).astype(int) == yt) for n, q in p.items()}})
    return rows


def run(cov, labels, jobs, subjects=None):
    pool = sorted(labels)
    TF = {s: {v: tangent_all(cov[s][v]) for v in ("830",) + BANK} for s in pool}
    W = {}
    for s in pool:
        model = LogisticRegression(C=1.0, max_iter=1000).fit(np.hstack([TF[s][v] for v in BANK]), labels[s])
        W[s] = np.append(model.coef_[0], model.intercept_[0])

    def tasks():
        for s in subjects or pool:
            rest = [o for o in pool if o != s]
            others = {"bank": (np.vstack([np.hstack([TF[o][v] for v in BANK]) for o in rest]),
                               np.concatenate([labels[o] for o in rest]))}
            others_C = (np.concatenate([centred(cov[o]["830"]) for o in rest]),
                        np.concatenate([labels[o] for o in rest]))
            prior = population_prior([W[o] for o in rest])
            yield delayed(decode)(s, cov[s], labels[s], others, others_C, prior)

    os.environ["HEMISHIFT_NO_TORCH"] = "1"
    rows = []
    for r in Parallel(n_jobs=jobs, return_as="generator")(tasks()):
        rows += r
        print(f"  sub {r[0]['subject']:2d} done", flush=True)
    return pd.DataFrame(rows).sort_values(["subject", "k"]).reset_index(drop=True)


def report(table):
    print("\n=== pre-registered hypotheses (k = 5; one-sided Wilcoxon; fixed sequence) ===")
    t = table[table.k == 5].set_index("subject")
    going = True
    for name, decoder, baseline in HYPOTHESES:
        diff = clean_differences(t[decoder] - t[baseline])
        p = signed_rank_p(diff, "greater")
        verdict = ("CONFIRMED" if p < ALPHA else "not confirmed") if going else "not tested; p is descriptive"
        going = going and p < ALPHA
        print(f"  {name}: {decoder} {t[decoder].mean():.1f} % vs {baseline} {t[baseline].mean():.1f} % "
              f"(mean diff {diff.mean():+.1f}, helped {(diff > 0).sum()}, harmed {(diff < 0).sum()}, "
              f"n {len(t)}); p = {p:.4f} -> {verdict}")
    for k in KS:
        t = table[table.k == k]
        print(f"descriptive, k = {k}: " + ", ".join(f"{n} {t[n].mean():.1f}" for n in t.columns
                                                    if n not in ("subject", "k", "n_test")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    cov, labels = load()
    print(f"{len(labels)} S1 patients", flush=True)
    if args.dry_run:
        rng = np.random.default_rng(1)
        shuffled = {s: rng.permutation(y) for s, y in labels.items()}
        table = run(cov, shuffled, args.jobs, subjects=sorted(labels)[:4])
        print(f"dry run on shuffled labels: {len(table)} rows, columns {list(table.columns)}")
        return
    table = run(cov, labels, args.jobs)
    table.to_csv(OUT, index=False)
    report(table)


if __name__ == "__main__":
    main()
