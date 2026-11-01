"""Combination study (exploratory; S3 dev and S4), declared in ACHIEVEMENTS.md (2026-11-01).

Each confirmed gain is a few points: pooling other patients (S3 locked test, S4), the
mu / low-beta / high-beta band bank and the patient's own earlier sessions (S4). This study
asks whether they add up, and tries new ways of combining them. S4's confirmatory tests are
spent, so S4 is development data here, like the 16 S3 dev patients; the locked tests are not
touched. Every number is exploratory.

All decoders are re-centred Riemannian tangent space + LR (C = 1), 14-channel template unless
named "28" (the 28-channel central strip). "bank" = 8-13, 13-20 and 20-30 Hz, tangent vectors
side by side; otherwise 8-30 Hz.
  static, fixed after calibration
    own, pooled          the confirmed baselines (s3_transfer.own_proba, s3_augment.pooled_proba)
    own_bank             the bank on the calibration only
    pooled_bank          + the other patients' same session (S4's H-R2 decoder)
    pooled_bank28        the same on 28 channels
    pooled_bank_crops    four 2-s sub-windows per epoch as samples; test probabilities averaged
                         over the sub-windows
    prior_bank           new: population-prior LR. Instead of pooling the other patients'
                         trials, the weights get a Gaussian prior whose mean and per-weight
                         variance are those of the other patients' own decoders (empirical Bayes)
    later sessions only
    prev_bank            + the patient's own earlier sessions (H-R3's decoder, with the bank)
    prevoth_bank         + earlier sessions and the other patients, each group weighing as much
                         as the calibration
    prevoth_bank28       the same on 28 channels
    prevzero_bank        no labels: the earlier sessions' decoder, the patient re-centred on the
                         calibration epochs (H-R4's decoder, with the bank)
  online: each test trial is decoded by a decoder refitted just before it, with the cue labels
  of the calibration and every earlier test trial, re-centred on all of them
    own_sup, pooled_sup  S4's descriptive co-adaptive decoders (8-30 Hz)
    own_sup_bank, pooled_sup_bank, pooled_sup_bank28
    prior_sup_bank       new: the population prior keeps its weight while the patient's own
                         labelled trials accumulate
    later sessions only: prevoth_sup_bank, prevoth_sup_bank28
  online, unsupervised: labels from the calibration only (or none), re-centred on every epoch
  seen so far
    pooled_unsup_bank; later sessions only: prevzero_unsup_bank (no labels at all)
  model averaging
    bma                  new: Bayesian model averaging over the online decoders and the best
                         static ones; each decoder's weight is its likelihood on the test trials
                         decoded so far (each cue label arrives after the decision)
Donor groups weigh as much as the patient's labelled trials. Online, they keep doing so as the
labelled trials accumulate (as in S4's co-adaptive decoders).

Protocol P3: the first k pairs (S3) or the first k trials per class (S4) calibrate, k = 5 and
10; every decoder is scored on the same test trials. Later sessions: patients with every
earlier session (as in H-R3). Results: results/summary/combo_<cohort>.csv; per-trial
probabilities in results/combo/<cohort>_probs.npz.

Usage: .venv/Scripts/python -I scripts/combo_explore.py [--cohort s3|s4|both] [--jobs 8]
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # one BLAS thread per worker; the workers run in parallel

import argparse
import importlib.util
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from hemishift.data.channels import EXTENDED_TEMPLATE
from hemishift.data.harmonise import common_average
from hemishift.data.loaders import load_wang2026sf_epochs
from hemishift.data.splits import calibration_split
from hemishift.eval.stats import clean_differences, holm, signed_rank_p
from hemishift.models.combo import (bma, centred, crop_features, crops_static, online,
                                    online_many, own_exact, pooled_exact, population_prior,
                                    static, tangent_all)
from hemishift.paths import CONFIG_DIR, HARMONISED_DIR, RAW_DIR, RESULTS_DIR


def _script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SESSIONS = ("pre", "post", "follow")
LATER = {"post": ("pre",), "follow": ("pre", "post")}
KS = (5, 10)
BANDS = {"830": (8.0, 30.0), "mu": (8.0, 13.0), "b1": (13.0, 20.0), "b2": (20.0, 30.0)}
BANK = ("mu", "b1", "b2")
BANK28 = tuple(f"28_{b}" for b in BANK)
CROPS = ((0.0, 2.0), (0.5, 2.5), (1.0, 3.0), (1.5, 3.5))  # s from epoch start (0.5 s after the cue)
STATIC = ("own", "pooled", "own_bank", "pooled_bank", "pooled_bank28", "pooled_bank_crops",
          "prior_bank")
STATIC_LATER = ("prev_bank", "prevoth_bank", "prevoth_bank28", "prevzero_bank")
ONLINE = ("own_sup", "pooled_sup", "own_sup_bank", "pooled_sup_bank", "pooled_sup_bank28",
          "prior_sup_bank")
ONLINE_LATER = ("prevoth_sup_bank", "prevoth_sup_bank28")
UNSUP, UNSUP_LATER = ("pooled_unsup_bank",), ("prevzero_unsup_bank",)
R2 = ("pooled_unsup_bank28",)  # round 2 (declared 2026-11-01, after round 1 on S3)
R2_LATER = ("prevoth_unsup_bank", "prevoth_unsup_bank28", "prevothzero_bank28",
            "prevothzero_unsup_bank28")
EARLY = tuple(f"{b}@0" for b in BANK)  # 0.5-2.5 s after the cue
LATE = tuple(f"{b}@3" for b in BANK)  # 2.0-4.0 s after the cue
R3 = ("pooled_sup_bank_fix", "pooled_sup_bank_hl20", "pooled_sup_bank_all", "pooled_sup_early",
      "pooled_sup_late", "bma3")  # round 3 (exploratory, dev only)
R3_LATER = ("prevoth_sup_bank28_fix", "prevoth_sup_bank28_hl20")
EXPERTS = ("pooled_bank", "pooled_bank28", "own_sup_bank", "pooled_sup_bank", "pooled_sup_bank28",
           "prior_sup_bank")
EXPERTS_LATER = ("prevoth_bank", "prevoth_bank28", "prevoth_sup_bank", "prevoth_sup_bank28")
FLOOR = 1e-4  # prior variance floor


# ---------- data (main process) ----------

def load_clean(phase, group="dev"):
    """S3 (rest, MI) pairs of the split's `group` from the authors' ICA-cleaned release, on the
    42 channels common with the raw recording (s3_processed_check), cached under
    data/harmonised."""
    check = _script("s3_processed_check")
    path = HARMONISED_DIR / f"s3proc_{group}_{phase}_processed_t42.npz"
    split = json.loads((CONFIG_DIR / "splits" / "wang2026sf_locked_test.json").read_text())
    if not path.exists():
        patients = {}
        for subject in split[group]:
            if not (RAW_DIR / "wang2026sf" / "processeddata" / f"sub-{subject:02d}" / phase).is_dir():
                continue
            e = load_wang2026sf_epochs([subject], phases=(phase,), template=check.COMMON42,
                                       source="processed")
            mi = np.flatnonzero(e.labels == "affected_mi")
            paired = mi[(mi > 0) & (e.labels[mi - 1] == "rest")]
            order = np.ravel(np.column_stack([paired - 1, paired]))
            patients[subject] = (e.X[order].astype(np.float32),
                                 (e.labels[order] == "affected_mi").astype(int))
            print(f"  cleaned {phase} sub {subject}: {order.size} epochs", flush=True)
        np.savez(path, **{f"X{s}": X for s, (X, _) in patients.items()},
                 **{f"y{s}": y for s, (_, y) in patients.items()})
    data = np.load(path)
    return {s: (data[f"X{s}"], data[f"y{s}"]) for s in split[group] if f"X{s}" in data.files}


def load(cohort, phase, channels):
    """Epochs in recording order and labels (MI 1) per patient; 14 or 28 channels, each group
    re-referenced to its own average."""
    sweep, s4 = _script("s3_sweep"), _script("s4_replication")
    if cohort in ("s3clean", "s3cleanlocked"):
        common = _script("s3_processed_check").COMMON42
        template = EXTENDED_TEMPLATE if channels == 14 else sweep.CENTRAL28
        group = "dev" if cohort == "s3clean" else "test"
        return {s: (common_average(X[:, [common.index(ch) for ch in template]]).astype(np.float32), y)
                for s, (X, y) in load_clean(phase, group).items()}
    if cohort in ("s3", "s3locked"):
        group = "dev" if cohort == "s3" else "test"
        if channels == 14:
            return sweep.load(group, phase, EXTENDED_TEMPLATE, sweep.WINDOWS["std"])
        data = sweep.load(group, phase, sweep.ALL59, sweep.WINDOWS["std"])
    else:
        if channels == 14:
            return s4.load_patients(phase)
        data = s4.load_patients(phase, sweep.ALL59)
    return {s: (sweep.subset(X, sweep.CENTRAL28), y) for s, (X, y) in data.items()}


def prepare(cohort):
    """Covariances per session, patient and view (cached under data/harmonised), and labels."""
    cov, labels = {}, {}
    for phase in SESSIONS:
        path = HARMONISED_DIR / f"combo_{cohort}_{phase}.npz"
        if not path.exists():  # the loaders are imported only here: a cached run needs no torch
            covariances = _script("s3_sweep").covariances
            X14, X28 = load(cohort, phase, 14), load(cohort, phase, 28)
            assert set(X14) == set(X28)
            arrays = {}
            for s, (X, y) in X14.items():
                assert np.array_equal(y, X28[s][1])
                arrays[f"y|{s}"] = y
                for b, band in BANDS.items():
                    arrays[f"{b}|{s}"] = covariances(X, band)
                for b in BANK:
                    arrays[f"28_{b}|{s}"] = covariances(X28[s][0], BANDS[b])
                    for i, crop in enumerate(CROPS):
                        arrays[f"{b}@{i}|{s}"] = covariances(X, BANDS[b], crop)
            np.savez(path, **arrays)
            print(f"  {cohort} {phase}: covariances of {len(X14)} patients cached", flush=True)
        data = np.load(path)
        cov[phase], labels[phase] = {}, {}
        for key in data.files:
            view, s = key.split("|")
            if view == "y":
                labels[phase][int(s)] = data[key]
            else:
                cov[phase].setdefault(int(s), {})[view] = data[key]
    return cov, labels


def crops_of(cov):
    """Sub-window covariances per bank band, (trials, sub-windows, c, c)."""
    return [np.stack([cov[f"{b}@{i}"] for i in range(len(CROPS))], axis=1) for b in BANK]


def domain_features(cov):
    """Per patient: tangent vectors of every view, re-centred on all its epochs, and the
    sub-window tangent vectors, each band re-centred on the mean over all its sub-windows."""
    TF = {v: tangent_all(cov[v]) for v in ("830",) + BANK + BANK28 + EARLY + LATE}
    TF["crops"] = crop_features(crops_of(cov))
    return TF


def donor_group(TF, labels, subjects, views):
    """Donor tangent vectors (views side by side) and labels, stacked over patients."""
    T = np.vstack([np.hstack([TF[s][v] for v in views]) for s in subjects])
    return T, np.concatenate([labels[s] for s in subjects])


def priors(TF, labels, pool, targets, views):
    """prior[s]: the population prior from the own decoders (each fitted on all its epochs) of
    the pool patients other than s."""
    W = {}
    for s in pool:
        model = LogisticRegression(C=1.0, max_iter=1000).fit(
            np.hstack([TF[s][v] for v in views]), labels[s])
        W[s] = np.append(model.coef_[0], model.intercept_[0])
    return {s: population_prior([W[o] for o in pool if o != s], FLOOR) for s in targets}


# ---------- one patient-session (runs in the workers) ----------

def online_options(others, prior, previous, n_cal):
    """Every online decoder of rounds 1 and 3: name -> (views, options of combo.online)."""
    options = {
        "own_sup": (("830",), {}),
        "pooled_sup": (("830",), {"groups": [others["830"]]}),
        "own_sup_bank": (BANK, {}),
        "pooled_sup_bank": (BANK, {"groups": [others["bank"]]}),
        "pooled_sup_bank28": (BANK28, {"groups": [others["bank28"]]}),
        "prior_sup_bank": (BANK, {"prior": prior}),
        "pooled_unsup_bank": (BANK, {"labels": "cal", "groups": [others["bank"]]}),
        "pooled_sup_bank_fix": (BANK, {"groups": [others["bank"]], "donor_total": n_cal}),
        "pooled_sup_bank_hl20": (BANK, {"groups": [others["bank"]], "half_life": 20}),
        "pooled_sup_bank_all": (BANK, {"groups": [others["bank_all"]]}),
        "pooled_sup_early": (EARLY, {"groups": [others["early"]]}),
        "pooled_sup_late": (LATE, {"groups": [others["late"]]}),
    }
    if previous is not None:
        both28 = [previous["bank28"], others["bank28"]]
        options.update({
            "prevoth_sup_bank": (BANK, {"groups": [previous["bank"], others["bank"]]}),
            "prevoth_sup_bank28": (BANK28, {"groups": both28}),
            "prevzero_unsup_bank": (BANK, {"zero": previous["bank"]}),
            "prevoth_sup_bank28_fix": (BANK28, {"groups": both28, "donor_total": n_cal}),
            "prevoth_sup_bank28_hl20": (BANK28, {"groups": both28, "half_life": 20}),
        })
    return options


def online_decoders(names, cov, y, cal, test, others, prior, previous, block):
    """The named online decoders. Those on the same views share one re-centring pass
    (combo.online_many: the same numbers as one combo.online call each, computed faster)."""
    options = online_options(others, prior, previous, len(cal))
    by_views = {}
    for name in names:
        views, o = options[name]
        by_views.setdefault(views, {})[name] = o
    p = {}
    for views, specs in by_views.items():
        p.update(online_many(cov, y, cal, test, views, specs, block))
    return p


def decoders(cov, y, cal, test, others, others_C, crops, prior, previous, block, more=()):
    """P(MI) of the test trials from round 1's decoders and their bma, then the online decoders
    named in `more`."""
    later = previous is not None
    p = {
        "own": own_exact(cov["830"], y, cal, test),
        "pooled": pooled_exact(cov["830"], y, cal, test, *others_C),
        "own_bank": static(cov, y, cal, test, BANK),
        "pooled_bank": static(cov, y, cal, test, BANK, [others["bank"]]),
        "pooled_bank28": static(cov, y, cal, test, BANK28, [others["bank28"]]),
        "pooled_bank_crops": crops_static(crops_of(cov), y, cal, test, crops),
        "prior_bank": static(cov, y, cal, test, BANK, prior=prior),
    }
    if later:
        p.update({
            "prev_bank": static(cov, y, cal, test, BANK, [previous["bank"]]),
            "prevoth_bank": static(cov, y, cal, test, BANK, [previous["bank"], others["bank"]]),
            "prevoth_bank28": static(cov, y, cal, test, BANK28, [previous["bank28"], others["bank28"]]),
            "prevzero_bank": static(cov, y, cal, test, BANK, zero=previous["bank"]),
        })
    first = ONLINE + UNSUP + (ONLINE_LATER + UNSUP_LATER if later else ())
    p.update(online_decoders(first + tuple(more), cov, y, cal, test, others, prior, previous, block))
    order = STATIC + ONLINE + UNSUP + (STATIC_LATER + ONLINE_LATER + UNSUP_LATER if later else ())
    out = {n: p[n] for n in order}
    out["bma"] = bma([p[e] for e in EXPERTS + (EXPERTS_LATER if later else ())], y[test], block)
    out.update({n: p[n] for n in more})
    return out


def record(rows, probs, phase, s, k, p, yt):
    rows.append({"session": phase, "subject": s, "k": k, "n_test": len(yt),
                 **{n: 100 * np.mean((q > 0.5).astype(int) == yt) for n, q in p.items()}})
    probs.update({f"{phase}|{s}|{k}|{n}": q.astype(np.float32) for n, q in p.items()})
    probs[f"{phase}|{s}|{k}|y"] = yt


def run_patient(phase, s, cov, y, splits, others, others_C, crops, prior, previous, block=1):
    """All decoders for one patient-session and every k. others: view set -> donor group of the
    other patients in this session; previous: view set -> the patient's earlier sessions."""
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    rows, probs = [], {}
    for k, (cal, test) in splits.items():
        p = decoders(cov, y, cal, test, others, others_C, crops, prior, previous, block)
        record(rows, probs, phase, s, k, p, y[test])
    return rows, probs


def run_patient_r2(phase, s, cov, y, splits, others, others_C, crops, prior, previous, block=1):
    """Round 2: the unsupervised (re-centring only) versions of round 1's best stacks.
    prevothzero_* use no patient labels at all: earlier sessions + other patients."""
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    rows, probs = [], {}
    for k, (cal, test) in splits.items():
        yt = y[test]
        p = {"pooled_unsup_bank28": online(cov, y, cal, test, BANK28, labels="cal",
                                           groups=[others["bank28"]], block=block)}
        if previous is not None:
            both, both28 = [previous["bank"], others["bank"]], [previous["bank28"], others["bank28"]]
            p.update({
                "prevoth_unsup_bank": online(cov, y, cal, test, BANK, labels="cal", groups=both, block=block),
                "prevoth_unsup_bank28": online(cov, y, cal, test, BANK28, labels="cal", groups=both28, block=block),
                "prevothzero_bank28": static(cov, y, cal, test, BANK28, zero=both28),
                "prevothzero_unsup_bank28": online(cov, y, cal, test, BANK28, zero=both28, block=block),
            })
        rows.append({"session": phase, "subject": s, "k": k,
                     **{n: 100 * np.mean((q > 0.5).astype(int) == yt) for n, q in p.items()}})
        probs.update({f"{phase}|{s}|{k}|{n}": q.astype(np.float32) for n, q in p.items()})
    return rows, probs


def run_patient_r3(phase, s, cov, y, splits, others, others_C, crops, prior, previous, block=1,
                   saved=None):
    """Round 3 (exploratory): round 1's decoders recomputed in the same run, then
      pooled_sup_bank_fix   donors keep the calibration's total weight as labelled trials accrue
      pooled_sup_bank_hl20  forgetting: labelled trials weigh 0.5 ** (age / 20 trials)
      pooled_sup_bank_all   donors from the other patients' every session, not only this one
      pooled_sup_early/late co-adaptive pooled banks on 0.5-2.5 s and 2.0-4.0 s after the cue
      prevoth_sup_bank28_fix / _hl20   the same two changes to the returning-patient stack
      bma3                  model averaging over round 1's members and every decoder above
    The new online decoders share round 1's re-centring passes (`decoders`).
    saved: k -> (probabilities, accuracies, test labels) of round 1 from a committed run with the
    same settings (`run(reuse=...)`); round 1 is then taken from it, not recomputed. bma3 always
    averages round 1's members as saved (float32), as declared."""
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    later = previous is not None
    more = R3[:-1] + (R3_LATER if later else ())
    experts = EXPERTS + (EXPERTS_LATER if later else ())
    rows, probs = [], {}
    for k, (cal, test) in splits.items():
        yt = y[test]
        if saved is None:
            p = decoders(cov, y, cal, test, others, others_C, crops, prior, previous, block, more)
        else:
            first, exact, y_saved = saved[k]
            assert np.array_equal(y_saved, yt), "the committed run had other test trials"
            p = {**first, **online_decoders(more, cov, y, cal, test, others, prior, previous, block)}
        p["bma3"] = bma([p[e].astype(np.float32) for e in experts] + [p[n] for n in more], yt, block)
        record(rows, probs, phase, s, k, p, yt)
        if saved is not None:
            rows[-1].update(exact)  # round 1's accuracies exactly as committed
    return rows, probs


# ---------- driver ----------

def split(cohort, y, k, chrono=False):
    """Calibration and test trials. S3: the first k (rest, MI) pairs. S4: the first k trials per
    class (P3); with `chrono`, every trial up to the one that completes k per class calibrates,
    so that no test trial precedes a calibration trial (an audit of P3)."""
    if cohort.startswith("s3"):
        return np.arange(2 * k), np.arange(2 * k, len(y))  # first k (rest, MI) pairs
    if chrono:
        counts = np.stack([np.cumsum(y == 0), np.cumsum(y == 1)])
        end = int(np.argmax(counts.min(axis=0) >= k)) + 1
        return np.arange(end), np.arange(end, len(y))
    return calibration_split(y, k, mode="first")


def run(cohort, jobs, worker, targets=None, block=1, permute=None, chrono=False, checkpoint=None,
        reuse=False):
    """Decode every patient of `cohort`, the other patients being the donors. With `targets`,
    decode only those: a second cohort (e.g. "s3locked", with all of `cohort` as the donor
    pool), or a list of `cohort`'s own patients (held out of the pool; a dry run).
    Audits: `block` (updates between blocks of trials only, see combo.online); `permute`, a
    seed: every patient's labels are shuffled within each session, donors included, so every
    decoder must fall to chance unless information leaks; `chrono` (see `split`).
    checkpoint: a directory; each patient-session is saved there as soon as it is done, and a
    rerun skips the saved ones (a crash loses only the patient-sessions in progress).
    The longest patient-sessions are dispatched first, so none is left running alone at the
    end; results are collected in the order they finish.
    reuse (round 3 only): take round 1 from the committed run of `cohort` (combo_<cohort>.csv
    and its per-trial probabilities), which used the same settings, instead of recomputing it."""
    started = time.time()
    assert not reuse or (worker is run_patient_r3 and targets is None and block == 1
                         and permute is None and not chrono)
    if reuse:
        saved_probs = np.load(RESULTS_DIR / "combo" / f"{cohort}_probs.npz")
        saved_table = pd.read_csv(RESULTS_DIR / "summary" / f"combo_{cohort}.csv").set_index(
            ["session", "subject", "k"])
        saved_keys = set(saved_probs.files)
        first_round = [n for n in saved_table.columns if n != "n_test"]

        def saved(phase, s):
            out = {}
            for k in KS:
                names = [n for n in first_round if f"{phase}|{s}|{k}|{n}" in saved_keys]
                row = saved_table.loc[(phase, s, k)]
                out[k] = ({n: saved_probs[f"{phase}|{s}|{k}|{n}"] for n in names},
                          {n: float(row[n]) for n in names}, saved_probs[f"{phase}|{s}|{k}|y"])
            return out
    done = {}
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        checkpoint.mkdir(parents=True, exist_ok=True)
        for path in checkpoint.glob("*.pkl"):
            done[path.stem] = pickle.loads(path.read_bytes())
        print(f"checkpoint {checkpoint}: {len(done)} patient-sessions already done", flush=True)
    cov, labels = prepare(cohort)
    if permute is not None:
        rng = np.random.default_rng(permute)
        labels = {ph: {s: rng.permutation(y) for s, y in sorted(data.items())} for ph, data in labels.items()}
    pool = {ph: sorted(labels[ph]) for ph in SESSIONS}
    decode = pool
    if isinstance(targets, str):
        extra_cov, extra_labels = prepare(targets)
        assert not any(set(extra_labels[ph]) & set(labels[ph]) for ph in SESSIONS)
        assert permute is None
        for ph in SESSIONS:
            cov[ph].update(extra_cov[ph])
            labels[ph].update(extra_labels[ph])
        decode = {ph: sorted(extra_labels[ph]) for ph in SESSIONS}
    elif targets is not None:
        decode = {ph: [s for s in pool[ph] if s in targets] for ph in SESSIONS}
        pool = {ph: [s for s in pool[ph] if s not in targets] for ph in SESSIONS}
    print(f"{cohort}{'' if targets is None else f' -> {targets}'}: data ready "
          f"({time.time() - started:.0f} s); patients decoded per session "
          f"{ {ph: len(decode[ph]) for ph in SESSIONS} }", flush=True)
    TF = {ph: {s: domain_features(c) for s, c in cov[ph].items()} for ph in SESSIONS}
    print(f"donor features ready ({time.time() - started:.0f} s)", flush=True)

    def tasks():  # lazily, so only the dispatched patients' donor stacks are in memory
        for phase in SESSIONS[::-1]:  # later sessions first: they have the most decoders
            todo = [s for s in decode[phase] if f"{phase}_{s}" not in done]
            if not todo:
                continue
            prior = priors(TF[phase], labels[phase], pool[phase], todo, BANK)
            for s in sorted(todo, key=lambda s: -len(labels[phase][s])):  # most trials first
                previous = None
                if phase in LATER:
                    if any(s not in labels[p] for p in LATER[phase]):
                        continue
                    previous = {
                        name: tuple(np.concatenate(parts) for parts in zip(*[
                            donor_group(TF[p], labels[p], [s], views) for p in LATER[phase]]))
                        for name, views in (("bank", BANK), ("bank28", BANK28))}
                rest = [o for o in pool[phase] if o != s]
                others = {name: donor_group(TF[phase], labels[phase], rest, views)
                          for name, views in (("830", ("830",)), ("bank", BANK), ("bank28", BANK28),
                                              ("early", EARLY), ("late", LATE))}
                every = [donor_group(TF[ph], labels[ph], [o for o in pool[ph] if o != s], BANK)
                         for ph in SESSIONS if [o for o in pool[ph] if o != s]]
                others["bank_all"] = (np.vstack([T for T, _ in every]), np.concatenate([y_ for _, y_ in every]))
                others_C = (np.concatenate([centred(cov[phase][o]["830"]) for o in rest]),
                            np.concatenate([labels[phase][o] for o in rest]))
                crops = (np.concatenate([TF[phase][o]["crops"] for o in rest]), others["bank"][1])
                y = labels[phase][s]
                assert s not in rest and all(o in pool[phase] for o in rest)
                extra = {"saved": saved(phase, s)} if reuse else {}
                yield delayed(worker)(phase, s, cov[phase][s], y,
                                      {k: split(cohort, y, k, chrono) for k in KS}, others, others_C,
                                      crops, prior[s], previous, block, **extra)

    os.environ["HEMISHIFT_NO_TORCH"] = "1"  # workers spawned from here skip torch (hemishift.models.combo)
    rows, probs = [], {}
    for r, p in done.values():
        rows += r
        probs.update(p)
    # pre_dispatch = n_jobs: one queued task per worker keeps fewer donor stacks in memory
    for r, p in Parallel(n_jobs=jobs, return_as="generator_unordered", pre_dispatch="n_jobs")(tasks()):
        rows += r
        probs.update(p)
        first = r[0]
        if checkpoint is not None:
            path = checkpoint / f"{first['session']}_{first['subject']}.pkl"
            path.with_suffix(".tmp").write_bytes(pickle.dumps((r, p)))
            os.replace(path.with_suffix(".tmp"), path)
        print(f"  {first['session']:6s} sub {first['subject']:2d} k=5: " + " ".join(
            f"{n} {v:.1f}" for n, v in first.items() if n not in ("session", "subject", "k", "n_test"))
            + f" ({time.time() - started:.0f} s)", flush=True)
    table = pd.DataFrame(rows).sort_values(["session", "subject", "k"]).reset_index(drop=True)
    return table, probs


# ---------- report ----------

FAMILIES = {  # what the decoder needs during the session
    "no labels after calibration": STATIC + STATIC_LATER + UNSUP + UNSUP_LATER + R2 + R2_LATER,
    "cue labels during the session": ONLINE + ONLINE_LATER + ("bma",),
}


def candidates(t):
    """Every decoder with a result for every patient of this setting."""
    return [n for members in FAMILIES.values() for n in members
            if n != "own" and n in t and t[n].notna().all()]


def per_setting(table, setting, k):
    """Per-patient accuracies: the pre session, or each patient's mean over post and follow-up."""
    t = table[table["k"] == k]
    if setting == "pre":
        return t[t["session"] == "pre"].set_index("subject")
    later = t[t["session"].isin(LATER)]
    return later.groupby("subject").mean(numeric_only=True)


def summary(t, names, reference):
    rows = []
    for n in names:
        diff = clean_differences(t[n] - t["own"])
        rows.append({"decoder": n, "mean": t[n].mean(), "vs_own": diff.mean(),
                     "vs_ref": (t[n] - t[reference]).mean(), "helped": int((diff > 0).sum()),
                     "harmed": int((diff < 0).sum()), "p": signed_rank_p(diff)})
    out = pd.DataFrame(rows)
    out["holm"] = holm(out["p"].tolist())
    return out.sort_values("mean", ascending=False)


def lopo(t, names):
    """Each patient scored with the decoder that has the best mean on the other patients."""
    scores = []
    for s in t.index:
        rest = t.drop(index=s)
        scores.append(t.loc[s, max(names, key=lambda n: rest[n].mean())])
    return np.array(scores)


def report(tables):
    best = {}
    for cohort, table in tables.items():
        for setting in ("pre", "later"):
            reference = "pooled_bank" if setting == "pre" else "prev_bank"
            for k in KS:
                t = per_setting(table, setting, k)
                names = candidates(t)
                print(f"\n=== {cohort.upper()} {setting}, k = {k}: {len(t)} patients, own "
                      f"{t['own'].mean():.1f} %, reference {reference} {t[reference].mean():.1f} % ===")
                s = summary(t, names, reference)
                for _, r in s.iterrows():
                    print(f"  {r['decoder']:20s} {r['mean']:5.1f} %  vs own {r['vs_own']:+5.1f} "
                          f"({r['helped']:2d}/{r['harmed']:2d}, p {r['p']:.4f}, Holm {r['holm']:.4f})  "
                          f"vs {reference} {r['vs_ref']:+5.1f}")
                for family, members in FAMILIES.items():
                    members = [n for n in members if n in names]
                    scores = lopo(t, members)
                    diff = clean_differences(scores - t["own"].to_numpy())
                    print(f"  honest pick ({family}; leave-one-patient-out): {scores.mean():.1f} %, "
                          f"vs own {diff.mean():+.1f} ({(diff > 0).sum()}/{(diff < 0).sum()}, "
                          f"p {signed_rank_p(diff):.4f})")
                    best[(cohort, setting, k, family)] = max(members, key=lambda n: t[n].mean())
    if len(tables) == 2:
        print("\n=== across cohorts: the best decoder of one cohort, scored on the other ===")
        for source, target in (("s3", "s4"), ("s4", "s3")):
            for setting in ("pre", "later"):
                for k in KS:
                    t = per_setting(tables[target], setting, k)
                    for family in FAMILIES:
                        pick = best[(source, setting, k, family)]
                        diff = clean_differences(t[pick] - t["own"])
                        print(f"  {family}, {setting}, k={k}: best on {source.upper()} = {pick}; on "
                              f"{target.upper()} {t[pick].mean():.1f} % vs own {t['own'].mean():.1f} "
                              f"({diff.mean():+.1f}, {(diff > 0).sum()}/{(diff < 0).sum()}, "
                              f"p {signed_rank_p(diff):.4f})")


def check(cohort, table):
    """The baselines must reproduce the logged decoders."""
    logs = RESULTS_DIR / "summary"
    if cohort == "s4":
        logged = pd.read_csv(logs / "s4_replication.csv")
        pairs = [("S4 pre", logged[logged["analysis"] == "pooling"],
                  [("own", "own"), ("pooled", "pooled"), ("pooled_bank", "pooled_fb"),
                   ("own_sup", "own_coadaptive"), ("pooled_sup", "pooled_coadaptive")]),
                 ("S4 later", logged[logged["analysis"] == "cross_session"], [("own", "own")])]
    else:
        sweeps = pd.concat([pd.read_csv(logs / f"s3_sweep_{ph}_dev.csv").assign(session=ph)
                            for ph in ("pre", "post")])
        cross = pd.read_csv(logs / "s3_cross_session_dev.csv").rename(columns={"target": "session"})
        pairs = [("S3 sweep", sweeps, [("own", "own"), ("pooled", "pooled")]),
                 ("S3 cross-session", cross, [("own", "own")])]
    mine = table.set_index(["session", "subject", "k"])
    for name, ref, columns in pairs:
        ref = ref.set_index(["session", "subject", "k"])
        for a, b in columns:
            same = np.isclose(mine.loc[ref.index, a], ref[b]).mean()
            print(f"check {name}: {a} equals the logged {b} in {100 * same:.0f} % of rows")


def report_r3(tables):
    """Round 3: each new decoder against the own decoder and against round 1's bma, recomputed in
    the same run; selection without hindsight within the cue-label family."""
    family = list(FAMILIES["cue labels during the session"]) + list(R3) + list(R3_LATER)
    best = {}
    for cohort, table in tables.items():
        for setting in ("pre", "later"):
            for k in KS:
                t = per_setting(table, setting, k)
                names = [n for n in family if n in t and t[n].notna().all()]
                print(f"\n=== round 3, {cohort.upper()} {setting}, k = {k}: own {t['own'].mean():.1f} %, "
                      f"bma {t['bma'].mean():.1f} % ===")
                for n in sorted(names, key=lambda n: -t[n].mean())[:8]:
                    d_own, d_bma = clean_differences(t[n] - t["own"]), clean_differences(t[n] - t["bma"])
                    print(f"  {n:24s} {t[n].mean():5.1f} %  vs own {d_own.mean():+5.1f}  vs bma {d_bma.mean():+5.1f} "
                          f"({(d_bma > 0).sum()}/{(d_bma < 0).sum()}, p {signed_rank_p(d_bma):.3f})")
                scores = lopo(t, names)
                print(f"  honest pick (leave-one-patient-out): {scores.mean():.1f} % "
                      f"(bma {t['bma'].mean():.1f} %)")
                best[(cohort, setting, k)] = max(names, key=lambda n: t[n].mean())
    if len(tables) == 2:
        print("\n=== round 3 across cohorts ===")
        for source, target in (("s3", "s4"), ("s4", "s3")):
            for setting in ("pre", "later"):
                t = per_setting(tables[target], setting, 5)
                pick = best[(source, setting, 5)]
                print(f"  {setting:5s}: best on {source.upper()} = {pick}; on {target.upper()} {t[pick].mean():.1f} % "
                      f"vs bma {t['bma'].mean():.1f} % ({clean_differences(t[pick] - t['bma']).mean():+.1f})")


def load_tables(cohort):
    """Round 1, with round 2's decoders joined in once they exist."""
    table = pd.read_csv(RESULTS_DIR / "summary" / f"combo_{cohort}.csv")
    second = RESULTS_DIR / "summary" / f"combo_{cohort}_r2.csv"
    if second.exists():
        table = table.merge(pd.read_csv(second), on=["session", "subject", "k"], how="left",
                            validate="one_to_one")
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="both", choices=("s3", "s4", "s3clean", "both"))
    parser.add_argument("--round", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--stats-only", action="store_true",
                        help="recompute the report from the saved per-patient results")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    cohorts = ("s3", "s4") if args.cohort == "both" else (args.cohort,)
    workers = {1: run_patient, 2: run_patient_r2, 3: run_patient_r3}
    suffix = {1: "", 2: "_r2", 3: "_r3"}[args.round]
    if args.round == 3:  # its own table: round 1 recomputed with pair-wise updates where trials alternate
        tables = {}
        for cohort in cohorts:
            out = RESULTS_DIR / "summary" / f"combo_{cohort}_r3.csv"
            if not args.stats_only:
                block = 2 if cohort.startswith("s3") else 1
                # S4's committed round 1 used these settings (per-trial updates); S3's did not
                table, _ = run(cohort, args.jobs, run_patient_r3, block=block, reuse=block == 1,
                               checkpoint=RESULTS_DIR / "combo" / "checkpoints" / f"{cohort}_r3_block{block}")
                table.to_csv(out, index=False)
            tables[cohort] = pd.read_csv(out)
        report_r3(tables)
        return
    if not args.stats_only:
        for cohort in cohorts:
            table, probs = run(cohort, args.jobs, workers[args.round])
            table.to_csv(RESULTS_DIR / "summary" / f"combo_{cohort}{suffix}.csv", index=False)
            (RESULTS_DIR / "combo").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(RESULTS_DIR / "combo" / f"{cohort}_probs{suffix}.npz", **probs)
            if args.round == 1 and cohort != "s3clean":
                check(cohort, table)
    report({cohort: load_tables(cohort) for cohort in cohorts})


if __name__ == "__main__":
    main()
