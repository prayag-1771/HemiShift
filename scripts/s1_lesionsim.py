"""S1 step 4: does LesionSim pretraining shrink the negative transfer? (dev patients)

EEGNet is pretrained on different data, then fine-tuned per development patient with blocked
4-fold CV (as in s1_deep.py), and every variant is compared with the patient's own TS + LR
decoder:
  healthy            PhysioNet healthy MI only (the negative-transfer baseline)
  +variability       + a LesionSim copy: per-trial ERD gain spread in mu and beta (O4b)
  +shift             + a LesionSim copy: relative ERD shifted to the healthy hemisphere (O2)
  +weaker            + a LesionSim copy: per-trial ERD attenuation on the lesioned side (O1)
  +lesionsim         + a LesionSim copy with all three
  +generic           + a generic-augmentation copy (per-channel gain and noise), the control
  +patients          healthy + the other dev patients (4 patient-group folds), real stroke data
Operator strengths follow the gap diagnosis (ACHIEVEMENTS.md): trial-to-trial variability up
about 20 %, laterality shifted part of the way to the healthy side, weak attenuation. They are
drawn per subject or per trial (domain randomisation). Each LesionSim copy uses each healthy
subject's own rest amplitude (PhysioNet rest epochs) as its baseline.
Results are written per variant as they finish, to results/summary/s1_lesionsim_dev.csv.

Usage: .venv/Scripts/python -I scripts/s1_lesionsim.py
"""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hemishift.augment.lesionsim import (
    BETA_HZ,
    MU_HZ,
    attenuate_erd,
    band_split,
    shift_to_contralesional,
    vary_erd,
)
from hemishift.data.channels import EXTENDED_TEMPLATE
from hemishift.data.harmonise import TARGET_SFREQ, common_average, normalise_lesion_side
from hemishift.data.liu2024_raw import EXCLUDED, load_raw, participants
from hemishift.data.loaders import load_moabb_epochs, moabb_cohort
from hemishift.data.splits import blocked_folds
from hemishift.eval.stats import paired_comparison
from hemishift.models.deep import (
    euclidean_alignment,
    inverse_sqrt,
    make_eegnet,
    predict_proba,
    train,
)
from hemishift.paths import CONFIG_DIR, RESULTS_DIR
from hemishift.preprocess.pipeline import Preprocessing, prepare_epochs

_spec = importlib.util.spec_from_file_location("premise_check",
                                               Path(__file__).with_name("premise_check.py"))
premise = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(premise)

CROP = 256
CLASS_ID = {"left_hand": 0, "right_hand": 1}
CHANNELS = EXTENDED_TEMPLATE
OUT = RESULTS_DIR / "summary" / "s1_lesionsim_dev.csv"


# ---------- healthy data and LesionSim copies ----------

def load_healthy():
    """Healthy MI trials (µV), labels, subject ids, and per-subject rest amplitudes."""
    dataset, side, group = moabb_cohort("physionet")
    e = load_moabb_epochs(dataset, side, "physionet", group, template=CHANNELS,
                          events=("left_hand", "right_hand", "rest"))
    subjects = e.meta["patient"].to_numpy()
    is_mi = e.labels != "rest"
    rest_amp = {}
    for s in np.unique(subjects):
        rest = e.X[(subjects == s) & ~is_mi]
        rest_amp[s] = {band: band_split(rest, TARGET_SFREQ, band)[1].mean(axis=(0, 2))
                       for band in (MU_HZ, BETA_HZ)}
    return e.X[is_mi], e.labels[is_mi], subjects[is_mi], rest_amp


def lesionsim_copy(X, subjects, rest_amp, ops, seed):
    """A stroke-like copy of the healthy trials, transformed subject by subject."""
    rng = np.random.default_rng(seed)
    out = np.empty_like(X)
    for s in np.unique(subjects):
        rows = np.flatnonzero(subjects == s)
        x = X[rows].astype(float)
        mu, beta = rest_amp[s][MU_HZ], rest_amp[s][BETA_HZ]
        if "weaker" in ops:
            x = attenuate_erd(x, TARGET_SFREQ, CHANNELS, mu, alpha=rng.uniform(0.6, 1.0, len(rows)))
        if "shift" in ops:
            x = shift_to_contralesional(x, TARGET_SFREQ, CHANNELS, mu, beta=rng.uniform(0.2, 0.8))
        if "variability" in ops:
            x = vary_erd(x, TARGET_SFREQ, CHANNELS, mu, spread=0.5, rng=rng, band=MU_HZ)
            x = vary_erd(x, TARGET_SFREQ, CHANNELS, beta, spread=0.5, rng=rng, band=BETA_HZ)
        out[rows] = common_average(x)  # the edits act per channel; re-reference like real data
    return out.astype(np.float32)


def generic_copy(X, seed):
    """The control: random per-channel gain (log-sd 0.3) plus white noise at 30 % of each
    channel's standard deviation, with no physiology in it."""
    rng = np.random.default_rng(seed)
    gain = np.exp(rng.normal(0.0, 0.3, size=X.shape[:2]))[..., None]
    noise = rng.normal(size=X.shape) * 0.3 * X.std(axis=-1, keepdims=True)
    return common_average(X * gain + noise).astype(np.float32)


def align_with_original(X_parts, subjects):
    """Euclidean-align every part with the same per-subject reference (the original trials),
    so augmented copies stay in their subject's frame."""
    original = X_parts[0]
    out = [np.empty_like(part) for part in X_parts]
    for s in np.unique(subjects):
        rows = np.flatnonzero(subjects == s)
        x = original[rows].astype(float)
        W = inverse_sqrt(np.einsum("nct,ndt->ncd", x, x).mean(axis=0) / x.shape[-1])
        for part, aligned in zip(X_parts, out):
            aligned[rows] = np.einsum("cd,ndt->nct", W, part[rows])
    return out


def pretrain(X, y, seed=0):
    return train(make_eegnet(len(CHANNELS), CROP, seed=seed), X, y, crop=CROP, epochs=60,
                 lr=1e-3, batch=64, seed=seed)


# ---------- patients ----------

def load_patients():
    split = json.loads((CONFIG_DIR / "splits" / "liu2024_locked_test.json").read_text())
    clinical = participants()
    patients = {}
    for subject in split["dev"]:
        if subject in EXCLUDED:
            continue
        X, labels, _ = prepare_epochs(load_raw(subject), Preprocessing(template=CHANNELS))
        side = "L" if clinical.loc[subject, "ParalysisSide"] == "left" else "R"
        X, labels = normalise_lesion_side(X, labels, CHANNELS, side)
        patients[subject] = (X.astype(np.float32), np.array([CLASS_ID[label] for label in labels]))
    return patients


def finetune_accuracy(pretrained, X, y):
    correct = 0
    for fold, (tr, te) in enumerate(blocked_folds(len(y), 4)):
        Xa = euclidean_alignment(X, tr)
        tuned = train(copy.deepcopy(pretrained), Xa[tr], y[tr], crop=CROP, epochs=40, lr=5e-4,
                      batch=16, seed=fold)
        correct += int((predict_proba(tuned, Xa[te], CROP).argmax(1) == y[te]).sum())
    return 100 * correct / len(y)


def own_accuracy(X, y):
    C = premise.covariances(X)
    correct = 0
    for tr, te in blocked_folds(len(y), 4):
        reference = premise.recenter(C, tr)
        model = premise.model().fit(reference[tr], y[tr])
        correct += int((model.predict(reference[te]) == y[te]).sum())
    return 100 * correct / len(y)


def report(table):
    table.to_csv(OUT, index=False)
    print(f"\n{len(table)} dev patients | own TS+LR mean {table['own'].mean():.1f} %")
    for variant in [c for c in table.columns if c not in ("subject", "own")]:
        r = paired_comparison(table[variant] / 100, table["own"] / 100)
        harmed = (table[variant] < table["own"]).sum()
        line = (f"{variant:12s} mean {table[variant].mean():5.1f} % | vs own median "
                f"{100 * r['median_diff']:+.1f}, p = {r['p']:.3f}, helped "
                f"{(table[variant] > table['own']).sum()}, harmed {harmed}")
        if variant != "healthy" and "healthy" in table:
            h = paired_comparison(table[variant] / 100, table["healthy"] / 100)
            line += f" | vs healthy-pretrained median {100 * h['median_diff']:+.1f}, p = {h['p']:.3f}"
        print(line)


def main():
    patients = load_patients()
    table = pd.DataFrame({"subject": list(patients)})
    table["own"] = [own_accuracy(*patients[s]) for s in patients]
    print(f"loaded {len(patients)} dev patients; own TS+LR mean {table['own'].mean():.1f} %")

    X_h, labels_h, subjects_h, rest_amp = load_healthy()
    y_h = np.array([CLASS_ID[label] for label in labels_h])
    print(f"healthy: {len(y_h)} MI trials from {len(np.unique(subjects_h))} subjects")

    copies = {
        "healthy": [],
        "+variability": [lesionsim_copy(X_h, subjects_h, rest_amp, {"variability"}, seed=1)],
        "+shift": [lesionsim_copy(X_h, subjects_h, rest_amp, {"shift"}, seed=2)],
        "+weaker": [lesionsim_copy(X_h, subjects_h, rest_amp, {"weaker"}, seed=3)],
        "+lesionsim": [lesionsim_copy(X_h, subjects_h, rest_amp,
                                      {"variability", "shift", "weaker"}, seed=4)],
        "+generic": [generic_copy(X_h, seed=5)],
    }
    for name, extra in copies.items():
        parts = align_with_original([X_h] + extra, subjects_h)
        print(f"[{name}] aligned scale (99.99th pct of |x|) per part: "
              + ", ".join(f"{np.percentile(np.abs(p), 99.99):.1f}" for p in parts))
        model = pretrain(np.concatenate(parts), np.concatenate([y_h] * len(parts)))
        table[name] = [finetune_accuracy(model, *patients[s]) for s in patients]
        print(f"[{name}] pretrained on {sum(len(p) for p in parts)} trials; "
              f"fine-tuned mean {table[name].mean():.1f} %")
        report(table)

    # Real stroke data: healthy + the other dev patients, by 4 patient groups.
    X_h_aligned = align_with_original([X_h], subjects_h)[0]
    groups = np.array_split(np.array(list(patients)), 4)
    scores = {}
    for group in groups:
        others = [s for s in patients if s not in set(group)]
        X_p = np.concatenate([euclidean_alignment(patients[s][0], np.arange(40)) for s in others])
        y_p = np.concatenate([patients[s][1] for s in others])
        model = pretrain(np.concatenate([X_h_aligned, X_p]), np.concatenate([y_h, y_p]))
        for s in group:
            scores[s] = finetune_accuracy(model, *patients[s])
    table["+patients"] = [scores[s] for s in patients]
    print(f"[+patients] fine-tuned mean {table['+patients'].mean():.1f} %")
    report(table)


if __name__ == "__main__":
    main()
