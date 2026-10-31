"""S3 sweep: the untried variants of the pooled decoder on Tianjin Task B (dev only, exploratory).

The pooled decoder (re-centred Riemannian TS + LR with other patients as donors) is the
confirmed result (S3 locked test, 2026-10-24). It was only ever run in one setting: 14
channels, 8-30 Hz, the 0.5-4.0 s window, LR(C = 1), all donors weighing as much as the
calibration. This sweep runs every variant not yet tried on S3, on the 16 dev patients only.
The locked test is spent, so a gain found here needs fresh data to be confirmed.

  structure  channels (14, 28 central-strip, all 59) x window (0.5-4.0 s, or 0.5-5.5 s of the
             6 s imagery and of the 8 s rest) x band (8-30 Hz, or a mu + beta + 8-30 Hz bank
             with tangent vectors concatenated): all 12 combinations
  bands      mu, beta, 4-30 Hz, and a 4-band bank (theta, mu, low beta, high beta)
  ensemble   mean probability over 3 bands x 3 windows (S1's ens16 idea, inside the imagery)
  weighting  donors weigh 0.5-8 times the calibration; LR regularisation C = 0.1 or 10
  donors     only the 8 nearest donors (Riemannian distance between mean covariances), or the
             8 whose own decoder best fits the patient's calibration epochs
  alignment  each domain stretched to unit dispersion (Riemannian Procrustes, second step)
  classifier linear SVM on the pooled tangent vectors
  mixup      own decoder plus tangent-space midpoints of same-class calibration epochs
  no labels  donors only; the patient is re-centred on its first 2k epochs, labels unused
  online     causal adaptation during the session: supervised (the cue gives each trial's
             label once it is decoded) or unsupervised (running re-centring only)
  controls   occipital channels only, own and pooled (a visual confound would show there)

Protocol P3 as before: the first k (rest, MI) pairs calibrate (k = 5 primary, 10 secondary),
the remaining pairs test. Donors are the other dev patients, each re-centred on its own mean.
Every variant is compared with the pooled decoder (two-sided Wilcoxon, Holm over the
variants). "Pick the best variant" is estimated without hindsight by leave-one-patient-out
selection: each patient is scored with the variant that did best on the other patients.
Results: results/summary/s3_sweep_<phase>_dev.csv.

Usage: .venv/Scripts/python -I scripts/s3_sweep.py [--phase pre|post|follow]
"""

import argparse
import importlib.util
import json
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from pyriemann.estimation import Covariances
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import tangent_space
from scipy.signal import butter, sosfiltfilt
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from hemishift.data.channels import EXTENDED_TEMPLATE
from hemishift.data.harmonise import TARGET_SFREQ, common_average
from hemishift.data.loaders import load_wang2026sf_epochs
from hemishift.eval.stats import holm, paired_comparison
from hemishift.paths import CONFIG_DIR, HARMONISED_DIR, RAW_DIR, RESULTS_DIR


def _script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s3 = _script("s3_transfer")
augment = _script("s3_augment")

ALL59 = ("Fpz", "Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8", "Fz", "F1", "F2", "F3", "F4", "F5",
         "F6", "F7", "F8", "FCz", "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FT7", "FT8", "Cz",
         "C1", "C2", "C3", "C4", "C5", "C6", "T7", "T8", "CP1", "CP2", "CP3", "CP4", "CP5",
         "CP6", "TP7", "TP8", "Pz", "P3", "P4", "P5", "P6", "P7", "P8", "POz", "PO3", "PO4",
         "PO5", "PO6", "PO7", "PO8", "Oz", "O1", "O2")
CENTRAL28 = ("F3", "F1", "Fz", "F2", "F4", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
             "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CP2", "CP4", "CP6",
             "P3", "Pz", "P4")
OCCIPITAL = ("PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "O1", "Oz", "O2")
WINDOWS = {"std": (0.5, 4.0), "long": (0.5, 5.5)}
BANDS = {"830": (8.0, 30.0), "mu": (8.0, 13.0), "beta": (13.0, 30.0), "theta": (4.0, 8.0),
         "b1": (13.0, 20.0), "b2": (20.0, 30.0), "wide": (4.0, 30.0)}
CROPS = {"w1": (0.0, 2.0), "w2": (1.5, 3.5)}  # 0.5-2.5 s and 2.0-4.0 s after onset
KS = s3.KS
BASE = "14_std_830"  # the confirmed decoder's view
CONTROLS = ("occ_own", "occ_pooled")
ONLINE = ("own_sup", "pooled_sup", "pooled_unsup", "donor_unsup")


# ---------- data ----------

def load(group, phase, template, window):
    """(rest, MI) pairs in recording order per patient, cached under data/harmonised."""
    split = json.loads((CONFIG_DIR / "splits" / "wang2026sf_locked_test.json").read_text())
    tag = {EXTENDED_TEMPLATE: "t14", ALL59: "t59"}[template]
    path = HARMONISED_DIR / f"s3sweep_{group}_{phase}_{tag}_{window[0]}-{window[1]}.npz"
    if path.exists():
        data = np.load(path)
        return {s: (data[f"X{s}"], data[f"y{s}"]) for s in split[group] if f"X{s}" in data.files}
    patients = {}
    for subject in split[group]:
        if not (RAW_DIR / "wang2026sf" / "sourcedata" / f"sub-{subject:02d}" / phase).is_dir():
            continue
        e = load_wang2026sf_epochs([subject], phases=(phase,), window=window, template=template)
        mi = np.flatnonzero(e.labels == "affected_mi")
        paired = mi[(mi > 0) & (e.labels[mi - 1] == "rest")]
        order = np.ravel(np.column_stack([paired - 1, paired]))  # rest, MI, rest, MI, ...
        patients[subject] = (e.X[order].astype(np.float32),
                             (e.labels[order] == "affected_mi").astype(int))
        print(f"  loaded {phase} sub {subject} {tag} {window}: {order.size} epochs", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{f"X{s}": X for s, (X, _) in patients.items()},
             **{f"y{s}": y for s, (_, y) in patients.items()})
    return patients


def subset(X59, channels, rereference=True):
    """Channels picked from the 59-channel epochs, re-referenced to their own average (which
    equals loading them with that template) unless `rereference` is False."""
    X = X59[:, [ALL59.index(ch) for ch in channels]]
    return common_average(X).astype(np.float32) if rereference else X


def covariances(X, band, crop=None):
    """OAS covariances after a 4th-order zero-phase band-pass; `crop` in s from epoch start.
    With the 8-30 Hz band and no crop this is premise_check.covariances."""
    sos = butter(4, band, btype="bandpass", fs=TARGET_SFREQ, output="sos")
    Xf = sosfiltfilt(sos, X, axis=-1)
    if crop is not None:
        Xf = Xf[..., int(round(crop[0] * TARGET_SFREQ)):int(round(crop[1] * TARGET_SFREQ))]
    return Covariances(estimator="oas").transform(Xf)


def tangent(C, reference):
    """Tangent vectors at the identity after whitening by the Riemannian mean of C[reference]."""
    W = invsqrtm(mean_riemann(C[reference]))
    return tangent_space(W @ C @ W, np.eye(C.shape[-1]), metric="riemann")


class Data:
    """Covariances per patient and view, donor tangent vectors (each patient re-centred on all
    its epochs) and target tangent vectors per k (re-centred on the first 2k epochs)."""

    def __init__(self, X14, X59, X59L):
        self.subjects = list(X14)
        self.X14 = {s: X for s, (X, _) in X14.items()}
        self.y = {s: y for s, (_, y) in X14.items()}
        self.C = {}
        for s in self.subjects:
            assert np.array_equal(self.y[s], X59[s][1]) and np.array_equal(self.y[s], X59L[s][1])
            x59, x59L = X59[s][0], X59L[s][0]
            arrays = {("14", "std"): self.X14[s], ("28", "std"): subset(x59, CENTRAL28),
                      ("59", "std"): x59, ("14", "long"): subset(x59L, EXTENDED_TEMPLATE),
                      ("28", "long"): subset(x59L, CENTRAL28), ("59", "long"): x59L,
                      ("occ", "std"): subset(x59, OCCIPITAL, rereference=False)}
            views = {}
            for (ch, win), X in arrays.items():
                bands = ("830",) if ch == "occ" else ("830", "mu", "beta")
                if (ch, win) == ("14", "std"):
                    bands = ("830", "mu", "beta", "theta", "b1", "b2", "wide")
                    for crop in CROPS:
                        for band in ("830", "mu", "beta"):
                            views[f"14_{crop}_{band}"] = covariances(X, BANDS[band], CROPS[crop])
                for band in bands:
                    views[f"{ch}_{win}_{band}"] = covariances(X, BANDS[band])
            self.C[s] = views
        self.TF = {s: {v: tangent(C, np.arange(len(C))) for v, C in self.C[s].items()}
                   for s in self.subjects}
        self.TK = {s: {k: {v: tangent(C, np.arange(2 * k)) for v, C in self.C[s].items()}
                       for k in KS} for s in self.subjects}
        self.mean = {s: mean_riemann(self.C[s][BASE]) for s in self.subjects}
        self.single = {s: lr().fit(self.TF[s][BASE], self.y[s]) for s in self.subjects}

    def others(self, s):
        return [o for o in self.subjects if o != s]


# ---------- decoders ----------

def lr(C=1.0):
    return LogisticRegression(C=C, max_iter=1000)


def donor_weights(n_target, n_donor, lam):
    """Target epochs weigh 1; all donors together weigh `lam` times the target."""
    return np.concatenate([np.ones(n_target), np.full(n_donor, lam * n_target / n_donor)])


def features(T, views, reference=None, stretch=False):
    """Tangent vectors of several views side by side; `stretch` scales each view to unit
    dispersion, measured on the `reference` rows (all rows if None)."""
    blocks = []
    for v in views:
        t = T[v]
        if stretch:
            ref = t if reference is None else t[reference]
            t = t / np.sqrt(np.mean(np.sum(ref ** 2, axis=1)))
        blocks.append(t)
    return np.hstack(blocks)


def split(d, s, k):
    return s3.calibration_split(len(d.y[s]), k)


def own(d, s, k, views=(BASE,), mix=False):
    y = d.y[s]
    cal, test = split(d, s, k)
    T = features(d.TK[s][k], views)
    X, Y, w = T[cal], y[cal], np.ones(len(cal))
    if mix:  # tangent-space midpoints of every same-class pair, together weighing as the real
        syn = [((T[cal][i] + T[cal][j]) / 2, c) for c in (0, 1)
               for i, j in combinations(np.flatnonzero(y[cal] == c), 2)]
        X = np.vstack([X, [t for t, _ in syn]])
        Y = np.concatenate([Y, [c for _, c in syn]])
        w = np.concatenate([w, np.full(len(syn), len(cal) / len(syn))])
    return lr().fit(X, Y, sample_weight=w).predict_proba(T[test])[:, 1]


def pooled(d, s, k, views=(BASE,), lam=1.0, C=1.0, donors=None, stretch=False, svm=False):
    y = d.y[s]
    cal, test = split(d, s, k)
    T = features(d.TK[s][k], views, cal, stretch)
    donors = d.others(s) if donors is None else donors
    T_d = np.vstack([features(d.TF[o], views, None, stretch) for o in donors])
    y_d = np.concatenate([d.y[o] for o in donors])
    X, Y = np.vstack([T[cal], T_d]), np.concatenate([y[cal], y_d])
    w = donor_weights(len(cal), len(y_d), lam)
    if svm:
        score = LinearSVC(C=C, max_iter=20000).fit(X, Y, sample_weight=w).decision_function(T[test])
        return 1 / (1 + np.exp(-score))
    return lr(C).fit(X, Y, sample_weight=w).predict_proba(T[test])[:, 1]


def donor_only(d, s, k, views=(BASE,)):
    _, test = split(d, s, k)
    T_d = np.vstack([features(d.TF[o], views) for o in d.others(s)])
    y_d = np.concatenate([d.y[o] for o in d.others(s)])
    return lr().fit(T_d, y_d).predict_proba(features(d.TK[s][k], views)[test])[:, 1]


def nearest(d, s, k, m=8):
    reference = mean_riemann(d.C[s][BASE][:2 * k])
    return sorted(d.others(s), key=lambda o: distance_riemann(reference, d.mean[o]))[:m]


def best_fitting(d, s, k, m=8):
    """Donors whose single-donor decoder gives the patient's calibration labels the highest
    mean log-likelihood."""
    cal = np.arange(2 * k)
    T, y = d.TK[s][k][BASE][cal], d.y[s][cal]

    def loglik(o):
        p = d.single[o].predict_proba(T)[np.arange(len(y)), y]
        return np.mean(np.log(np.clip(p, 1e-12, None)))

    return sorted(d.others(s), key=loglik, reverse=True)[:m]


def online(d, s, k, mode, view=BASE):
    """Causal decoding pair by pair after calibration; the decoder is refitted before each pair.
      own_sup / pooled_sup  every earlier epoch is labelled (by its cue) and re-centres
      pooled_unsup          labels from calibration only; every earlier epoch re-centres
      donor_unsup           no labels at all; every earlier epoch re-centres"""
    y, C = d.y[s], d.C[s][view]
    first = 2 * k
    identity = np.eye(C.shape[-1])
    T_d = np.vstack([d.TF[o][view] for o in d.others(s)])
    y_d = np.concatenate([d.y[o] for o in d.others(s)])
    fixed = lr().fit(T_d, y_d) if mode == "donor_unsup" else None
    p = np.empty(len(y) - first)
    for start in range(first, len(y), 2):
        pair = np.arange(start, min(start + 2, len(y)))
        seen = np.arange(start)
        W = invsqrtm(mean_riemann(C[seen]))

        def tv(rows):
            return tangent_space(W @ C[rows] @ W, identity, metric="riemann")

        if mode == "own_sup":
            model = lr().fit(tv(seen), y[seen])
        elif mode in ("pooled_sup", "pooled_unsup"):
            labelled = seen if mode == "pooled_sup" else np.arange(first)
            model = lr().fit(np.vstack([tv(labelled), T_d]), np.concatenate([y[labelled], y_d]),
                             sample_weight=donor_weights(len(labelled), len(y_d), 1.0))
        else:
            model = fixed
        p[pair - first] = model.predict_proba(tv(pair))[:, 1]
    return p


def structure_views(ch, win, bank):
    return (f"{ch}_{win}_830",) if bank == "830" else tuple(f"{ch}_{win}_{b}" for b in ("mu", "beta", "830"))


ENSEMBLE = tuple(f"14_{w}_{b}" for w in ("std", "w1", "w2") for b in ("830", "mu", "beta"))


def candidates(d):
    """name -> function(s, k) returning P(MI) for the test epochs."""
    def exact_pooled(s, k):
        cal, test = split(d, s, k)
        C = d.C[s][BASE]
        others = d.others(s)
        donor_C = np.concatenate([augment.recentred_by_domain(d.C[o][BASE], np.zeros(len(d.y[o])))
                                  for o in others])
        donor_y = np.concatenate([d.y[o] for o in others])
        return augment.pooled_proba(C[cal], d.y[s][cal], C[test], donor_C, donor_y)

    c = {
        "own": lambda s, k: s3.own_proba(d.X14[s], d.y[s], *split(d, s, k)),
        "pooled": exact_pooled,
    }
    for ch in ("14", "28", "59"):
        for win in ("std", "long"):
            for bank in ("830", "fb3"):
                name = f"pooled_{ch}ch_{win}_{bank}"
                c[name] = (lambda views: lambda s, k: pooled(d, s, k, views))(structure_views(ch, win, bank))
    c.update({
        "own_long": lambda s, k: own(d, s, k, ("14_long_830",)),
        "own_28ch": lambda s, k: own(d, s, k, ("28_std_830",)),
        "own_mix": lambda s, k: own(d, s, k, mix=True),
        "pooled_mu": lambda s, k: pooled(d, s, k, ("14_std_mu",)),
        "pooled_beta": lambda s, k: pooled(d, s, k, ("14_std_beta",)),
        "pooled_wide": lambda s, k: pooled(d, s, k, ("14_std_wide",)),
        "pooled_fb4": lambda s, k: pooled(d, s, k, tuple(f"14_std_{b}" for b in ("theta", "mu", "b1", "b2"))),
        "pooled_ens9": lambda s, k: np.mean([pooled(d, s, k, (v,)) for v in ENSEMBLE], axis=0),
        "pooled_near8": lambda s, k: pooled(d, s, k, donors=nearest(d, s, k)),
        "pooled_fit8": lambda s, k: pooled(d, s, k, donors=best_fitting(d, s, k)),
        "pooled_stretch": lambda s, k: pooled(d, s, k, stretch=True),
        "pooled_svm": lambda s, k: pooled(d, s, k, svm=True),
        "donor_only": lambda s, k: donor_only(d, s, k),
        "occ_own": lambda s, k: own(d, s, k, ("occ_std_830",)),
        "occ_pooled": lambda s, k: pooled(d, s, k, ("occ_std_830",)),
    })
    for lam in (0.5, 2, 4, 8):
        c[f"pooled_lam{lam:g}"] = (lambda lam: lambda s, k: pooled(d, s, k, lam=lam))(lam)
    for C in (0.1, 10):
        c[f"pooled_C{C:g}"] = (lambda C: lambda s, k: pooled(d, s, k, C=C))(C)
    for mode in ONLINE:
        c[mode] = (lambda mode: lambda s, k: online(d, s, k, mode))(mode)
    return c


# ---------- report ----------

def compare(table, names, reference="pooled"):
    for k in KS:
        t = table[table["k"] == k]
        print(f"\nk = {k} ({len(t)} patients): own {t['own'].mean():.1f} %, pooled "
              f"{t[reference].mean():.1f} %  |  variant: mean, mean diff and median diff vs "
              f"pooled, p (Holm), helped/harmed")
        rows = []
        for n in names:
            r = paired_comparison(t[n] / 100, t[reference] / 100)
            rows.append((n, t[n].mean(), t[n].mean() - t[reference].mean(), 100 * r["median_diff"],
                         r["p"], int((t[n] > t[reference]).sum()), int((t[n] < t[reference]).sum())))
        adjusted = holm([r[4] for r in rows])
        for r, a in sorted(zip(rows, adjusted), key=lambda z: -z[0][1]):
            print(f"  {r[0]:24s} {r[1]:5.1f} %  {r[2]:+5.1f} / {r[3]:+5.1f}  p {r[4]:.3f} "
                  f"({a:.3f})  {r[5]:2d}/{r[6]:2d}")


def nested(table, names, k):
    """Each patient scored with the variant that has the best mean on the other patients."""
    t = table[table["k"] == k].reset_index(drop=True)
    picks, scores = [], []
    for i in range(len(t)):
        rest = t.drop(index=i)
        best = max(names, key=lambda n: rest[n].mean())
        picks.append(best)
        scores.append(t.loc[i, best])
    return np.array(scores), picks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="pre", choices=("pre", "post", "follow"))
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    out = RESULTS_DIR / "summary" / f"s3_sweep_{args.phase}_dev.csv"

    started = time.time()
    X14 = load("dev", args.phase, EXTENDED_TEMPLATE, WINDOWS["std"])
    X59 = load("dev", args.phase, ALL59, WINDOWS["std"])
    X59L = load("dev", args.phase, ALL59, WINDOWS["long"])
    if args.phase == "pre":
        reference = s3.load_patients("dev")
        assert all(np.array_equal(reference[s][0], X14[s][0]) for s in reference), \
            "the cached 14-channel data must equal s3_transfer.load_patients"
    d = Data(X14, X59, X59L)
    print(f"{len(d.subjects)} dev patients ({args.phase}), {len(d.C[d.subjects[0]])} views; "
          f"data ready in {time.time() - started:.0f} s", flush=True)

    decoders = candidates(d)
    rows = []
    for s in d.subjects:
        for k in KS:
            _, test = split(d, s, k)
            row = {"subject": s, "k": k}
            for name, decode in decoders.items():
                row[name] = 100 * np.mean((decode(s, k) > 0.5).astype(int) == d.y[s][test])
            rows.append(row)
        pd.DataFrame(rows).to_csv(out, index=False)
        t5 = rows[-2]
        print(f"sub {s:2d} ({time.time() - started:.0f} s) k=5: own {t5['own']:.1f} pooled "
              f"{t5['pooled']:.1f} best {max((n for n in t5 if n not in ('subject', 'k')), key=t5.get)} "
              f"{max(v for n, v in t5.items() if n not in ('subject', 'k')):.1f}", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(out, index=False)
    same = (table["pooled"] == table["pooled_14ch_std_830"]).mean()
    print(f"\nsanity: generic pooled equals the confirmed pooled_proba in {100 * same:.0f} % of rows")

    fixed = [n for n in decoders if n not in ("pooled", "pooled_14ch_std_830") + CONTROLS + ONLINE]
    print("\n=== fixed calibration (protocol P3) ===")
    compare(table, fixed)
    print("\n=== online adaptation over the session (same test epochs) ===")
    compare(table, list(ONLINE))
    print("\n=== controls ===")
    compare(table, list(CONTROLS))
    print("\n=== honest 'pick the best variant' (leave-one-patient-out selection) ===")
    for k in KS:
        t = table[table["k"] == k]
        scores, picks = nested(table, ["pooled"] + fixed, k)
        r = paired_comparison(scores / 100, t["pooled"].to_numpy() / 100)
        print(f"k = {k}: selected {scores.mean():.1f} % vs pooled {t['pooled'].mean():.1f} % "
              f"(median {100 * r['median_diff']:+.1f}, p {r['p']:.3f}); picks "
              f"{pd.Series(picks).value_counts().to_dict()}")
    print(f"results in {out} ({time.time() - started:.0f} s)")


if __name__ == "__main__":
    main()
