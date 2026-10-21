"""S3 step 1: few-trial calibration with transfer on Task B (affected-hand MI vs rest), dev.

Protocol P3 (chronological calibration): each development patient's first k (rest, MI) pairs
calibrate, and the remaining pairs of the pre-training phase (40 pairs in all) are the test
set. k = 5 and 10. Decoders:
  own         TS + LR on the 2k calibration epochs
  scratch     EEGNet trained on the calibration epochs only
  zeroshot    healthy-pretrained EEGNet, no patient training (alignment from calibration)
  healthy     healthy-pretrained EEGNet, fine-tuned on the calibration epochs
  +lesionsim  pretrained on healthy + a LesionSim copy of the healthy MI epochs
  +generic    pretrained on healthy + a generic-augmentation copy (the control)
  +patients   pretrained on healthy + the other dev patients (4 patient groups)
Healthy source: PhysioNet's 109 subjects, 14 channels. MI = right-hand trials plus mirrored
left-hand trials (both then show the ERD over the left, "lesioned", hemisphere); rest = rest
epochs, a random half of them mirrored too, so mirroring says nothing about the label.
Alignment: per healthy subject from its original epochs; per patient from its calibration
epochs. Results: results/summary/s3_transfer_dev.csv.

Usage: .venv/Scripts/python -I scripts/s3_transfer.py
"""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from hemishift.augment.lesionsim import (
    BETA_HZ,
    MU_HZ,
    attenuate_erd,
    band_split,
    shift_to_contralesional,
    vary_erd,
)
from hemishift.data.channels import EXTENDED_TEMPLATE, mirror
from hemishift.data.harmonise import TARGET_SFREQ, common_average
from hemishift.data.loaders import load_moabb_epochs, load_wang2026sf_epochs, moabb_cohort
from hemishift.eval.stats import chance_threshold, holm, paired_comparison
from hemishift.models.deep import euclidean_alignment, make_eegnet, predict_proba, train
from hemishift.paths import CONFIG_DIR, RESULTS_DIR


def _script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


premise = _script("premise_check")
s1 = _script("s1_lesionsim")

CHANNELS = EXTENDED_TEMPLATE
MIRROR = [CHANNELS.index(mirror(ch)) for ch in CHANNELS]
CROP = 256
KS = (5, 10)
OUT = RESULTS_DIR / "summary" / "s3_transfer_dev.csv"


# ---------- data ----------

def load_healthy(seed=0):
    """Healthy epochs (MI = 1, rest = 0), subject ids, mirrored flags, rest amplitudes."""
    dataset, side, group = moabb_cohort("physionet")
    e = load_moabb_epochs(dataset, side, "physionet", group, template=CHANNELS,
                          events=("left_hand", "right_hand", "rest"))
    rng = np.random.default_rng(seed)
    subjects = e.meta["patient"].to_numpy()
    is_rest = e.labels == "rest"
    mirrored = (e.labels == "left_hand") | (is_rest & (rng.random(len(e.labels)) < 0.5))
    X = e.X.copy()
    X[mirrored] = X[mirrored][:, MIRROR]
    rest_amp = {}
    for s in np.unique(subjects):
        rest = e.X[(subjects == s) & is_rest]
        rest_amp[s] = {band: band_split(rest, TARGET_SFREQ, band)[1].mean(axis=(0, 2))
                       for band in (MU_HZ, BETA_HZ)}
    return X, (~is_rest).astype(int), subjects, mirrored, rest_amp


def lesionsim_copy(X, y, subjects, mirrored, rest_amp, seed):
    """MI epochs made stroke-like with all three operators (as in s1_lesionsim); rest unchanged.
    Mirrored epochs use the mirrored rest amplitudes."""
    rng = np.random.default_rng(seed)
    out = X.astype(float)
    for s in np.unique(subjects):
        for flipped in (False, True):
            rows = np.flatnonzero((subjects == s) & (y == 1) & (mirrored == flipped))
            if rows.size == 0:
                continue
            mu, beta = (rest_amp[s][band][MIRROR] if flipped else rest_amp[s][band]
                        for band in (MU_HZ, BETA_HZ))
            x = attenuate_erd(out[rows], TARGET_SFREQ, CHANNELS, mu,
                              alpha=rng.uniform(0.6, 1.0, rows.size))
            x = shift_to_contralesional(x, TARGET_SFREQ, CHANNELS, mu, beta=rng.uniform(0.2, 0.8))
            x = vary_erd(x, TARGET_SFREQ, CHANNELS, mu, spread=0.5, rng=rng, band=MU_HZ)
            x = vary_erd(x, TARGET_SFREQ, CHANNELS, beta, spread=0.5, rng=rng, band=BETA_HZ)
            out[rows] = common_average(x)
    return out.astype(np.float32)


def load_patients():
    """Dev patients' (rest, MI) pairs in recording order: X (80, 14, 448), y (rest 0, MI 1)."""
    split = json.loads((CONFIG_DIR / "splits" / "wang2026sf_locked_test.json").read_text())
    patients = {}
    for subject in split["dev"]:
        e = load_wang2026sf_epochs([subject], phases=("pre",), template=CHANNELS)
        mi = np.flatnonzero(e.labels == "affected_mi")
        paired = mi[(mi > 0) & (e.labels[mi - 1] == "rest")]
        order = np.ravel(np.column_stack([paired - 1, paired]))  # rest, MI, rest, MI, ...
        patients[subject] = (e.X[order].astype(np.float32),
                             (e.labels[order] == "affected_mi").astype(int))
    return patients


# ---------- decoders ----------

def calibration_split(n, k):
    """First k pairs calibrate, the rest test."""
    return np.arange(2 * k), np.arange(2 * k, n)


def own_proba(X, y, cal, test):
    A = premise.recenter(premise.covariances(X), cal)
    return premise.model().fit(A[cal], y[cal]).predict_proba(A[test])[:, 1]


def deep_proba(model, X, y, cal, test, epochs=40, lr=5e-4):
    Xa = euclidean_alignment(X, cal)
    if epochs:
        model = train(copy.deepcopy(model), Xa[cal], y[cal], crop=CROP, epochs=epochs, lr=lr,
                      batch=16)
    return predict_proba(model, Xa[test], CROP)[:, 1]


def scratch_proba(X, y, cal, test):
    Xa = euclidean_alignment(X, cal)
    model = train(make_eegnet(len(CHANNELS), CROP), Xa[cal], y[cal], crop=CROP, epochs=100,
                  lr=1e-3, batch=16)
    return predict_proba(model, Xa[test], CROP)[:, 1]


# ---------- bookkeeping ----------

class Table:
    def __init__(self, patients):
        self.patients = patients
        self.rows = {(s, k): {"subject": s, "k": k} for s in patients for k in KS}

    def record(self, name, decode, subjects=None):
        for s in subjects if subjects is not None else self.patients:
            X, y = self.patients[s]
            for k in KS:
                cal, test = calibration_split(len(y), k)
                p = decode(X, y, cal, test)
                self.rows[(s, k)][name] = 100 * np.mean((p > 0.5) == y[test])
                self.rows[(s, k)][f"{name}_auc"] = roc_auc_score(y[test], p)

    def report(self):
        table = pd.DataFrame(self.rows.values())
        table.to_csv(OUT, index=False)
        methods = [c for c in table.columns if c not in ("subject", "k", "own")
                   and not c.endswith("_auc")]
        for k in KS:
            t = table[table["k"] == k]
            n_test = 80 - 2 * k
            threshold = 100 * chance_threshold(n_test)
            print(f"\nk = {k} per class ({len(t)} dev patients, {n_test} test epochs, chance "
                  f"threshold {threshold:.0f} %) | own mean {t['own'].mean():.1f} %, "
                  f"AUC {t['own_auc'].mean():.3f}")
            results = {m: paired_comparison(t[m] / 100, t["own"] / 100) for m in methods}
            adjusted = dict(zip(methods, holm([results[m]["p"] for m in methods])))
            for m in methods:
                r = results[m]
                print(f"  {m:11s} mean {t[m].mean():5.1f} % AUC {t[m + '_auc'].mean():.3f} | vs own "
                      f"median {100 * r['median_diff']:+.1f}, p = {r['p']:.3f} (Holm {adjusted[m]:.3f}), "
                      f"helped {(t[m] > t['own']).sum()}, harmed (>= 2 pts) {(t[m] <= t['own'] - 2).sum()}")


def main():
    patients = load_patients()
    table = Table(patients)
    table.record("own", own_proba)
    table.record("scratch", scratch_proba)
    print(f"loaded {len(patients)} dev patients")
    table.report()

    X_h, y_h, subjects_h, mirrored, rest_amp = load_healthy()
    print(f"healthy: {len(y_h)} epochs ({y_h.sum()} MI) from {len(np.unique(subjects_h))} subjects")
    copies = {
        "healthy": [],
        "+lesionsim": [lesionsim_copy(X_h, y_h, subjects_h, mirrored, rest_amp, seed=4)],
        "+generic": [s1.generic_copy(X_h, seed=5)],
    }
    for name, extra in copies.items():
        parts = s1.align_with_original([X_h] + extra, subjects_h)
        print(f"[{name}] aligned scale (99.99th pct of |x|) per part: "
              + ", ".join(f"{np.percentile(np.abs(p), 99.99):.1f}" for p in parts))
        model = s1.pretrain(np.concatenate(parts), np.concatenate([y_h] * len(parts)))
        if name == "healthy":
            table.record("zeroshot", lambda X, y, cal, test: deep_proba(model, X, y, cal, test, 0))
        table.record(name, lambda X, y, cal, test: deep_proba(model, X, y, cal, test))
        table.report()

    # Real stroke data: healthy + the other dev patients, by 4 patient groups.
    X_h_aligned = s1.align_with_original([X_h], subjects_h)[0]
    for group in np.array_split(np.array(list(patients)), 4):
        others = [s for s in patients if s not in set(group)]
        X_p = np.concatenate([euclidean_alignment(patients[s][0], np.arange(len(patients[s][1])))
                              for s in others])
        y_p = np.concatenate([patients[s][1] for s in others])
        model = s1.pretrain(np.concatenate([X_h_aligned, X_p]), np.concatenate([y_h, y_p]))
        table.record("+patients", lambda X, y, cal, test: deep_proba(model, X, y, cal, test),
                     subjects=list(group))
    table.report()
    print(f"results in {OUT}")


if __name__ == "__main__":
    main()
