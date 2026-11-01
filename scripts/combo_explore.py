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
                                    own_exact, pooled_exact, population_prior, static,
                                    tangent_all)
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
EXPERTS = ("pooled_bank", "pooled_bank28", "own_sup_bank", "pooled_sup_bank", "pooled_sup_bank28",
           "prior_sup_bank")
EXPERTS_LATER = ("prevoth_bank", "prevoth_bank28", "prevoth_sup_bank", "prevoth_sup_bank28")
FLOOR = 1e-4  # prior variance floor


# ---------- data (main process) ----------

def load_clean(phase):
    """S3 dev (rest, MI) pairs from the authors' ICA-cleaned release on the 42 channels common
    with the raw recording (s3_processed_check), cached under data/harmonised."""
    check = _script("s3_processed_check")
    path = HARMONISED_DIR / f"s3proc_dev_{phase}_processed_t42.npz"
    split = json.loads((CONFIG_DIR / "splits" / "wang2026sf_locked_test.json").read_text())
    if not path.exists():
        patients = {}
        for subject in split["dev"]:
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
    return {s: (data[f"X{s}"], data[f"y{s}"]) for s in split["dev"] if f"X{s}" in data.files}


def load(cohort, phase, channels):
    """Epochs in recording order and labels (MI 1) per patient; 14 or 28 channels, each group
    re-referenced to its own average."""
    sweep, s4 = _script("s3_sweep"), _script("s4_replication")
    if cohort == "s3clean":
        common = _script("s3_processed_check").COMMON42
        template = EXTENDED_TEMPLATE if channels == 14 else sweep.CENTRAL28
        return {s: (common_average(X[:, [common.index(ch) for ch in template]]).astype(np.float32), y)
                for s, (X, y) in load_clean(phase).items()}
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
    covariances = _script("s3_sweep").covariances
    cov, labels = {}, {}
    for phase in SESSIONS:
        path = HARMONISED_DIR / f"combo_{cohort}_{phase}.npz"
        if not path.exists():
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
    TF = {v: tangent_all(cov[v]) for v in ("830",) + BANK + BANK28}
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

def run_patient(phase, s, cov, y, splits, others, others_C, crops, prior, previous):
    """All decoders for one patient-session and every k. others: view set -> donor group of the
    other patients in this session; previous: view set -> the patient's earlier sessions."""
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    rows, probs = [], {}
    for k, (cal, test) in splits.items():
        yt = y[test]
        p = {
            "own": own_exact(cov["830"], y, cal, test),
            "pooled": pooled_exact(cov["830"], y, cal, test, *others_C),
            "own_bank": static(cov, y, cal, test, BANK),
            "pooled_bank": static(cov, y, cal, test, BANK, [others["bank"]]),
            "pooled_bank28": static(cov, y, cal, test, BANK28, [others["bank28"]]),
            "pooled_bank_crops": crops_static(crops_of(cov), y, cal, test, crops),
            "prior_bank": static(cov, y, cal, test, BANK, prior=prior),
            "own_sup": online(cov, y, cal, test, ("830",)),
            "pooled_sup": online(cov, y, cal, test, ("830",), groups=[others["830"]]),
            "own_sup_bank": online(cov, y, cal, test, BANK),
            "pooled_sup_bank": online(cov, y, cal, test, BANK, groups=[others["bank"]]),
            "pooled_sup_bank28": online(cov, y, cal, test, BANK28, groups=[others["bank28"]]),
            "prior_sup_bank": online(cov, y, cal, test, BANK, prior=prior),
            "pooled_unsup_bank": online(cov, y, cal, test, BANK, labels="cal", groups=[others["bank"]]),
        }
        experts = list(EXPERTS)
        if previous is not None:
            p.update({
                "prev_bank": static(cov, y, cal, test, BANK, [previous["bank"]]),
                "prevoth_bank": static(cov, y, cal, test, BANK, [previous["bank"], others["bank"]]),
                "prevoth_bank28": static(cov, y, cal, test, BANK28,
                                         [previous["bank28"], others["bank28"]]),
                "prevzero_bank": static(cov, y, cal, test, BANK, zero=previous["bank"]),
                "prevoth_sup_bank": online(cov, y, cal, test, BANK,
                                           groups=[previous["bank"], others["bank"]]),
                "prevoth_sup_bank28": online(cov, y, cal, test, BANK28,
                                             groups=[previous["bank28"], others["bank28"]]),
                "prevzero_unsup_bank": online(cov, y, cal, test, BANK, zero=previous["bank"]),
            })
            experts += EXPERTS_LATER
        p["bma"] = bma([p[e] for e in experts], yt)
        rows.append({"session": phase, "subject": s, "k": k, "n_test": len(test),
                     **{n: 100 * np.mean((q > 0.5).astype(int) == yt) for n, q in p.items()}})
        probs.update({f"{phase}|{s}|{k}|{n}": q.astype(np.float32) for n, q in p.items()})
        probs[f"{phase}|{s}|{k}|y"] = yt
    return rows, probs


def run_patient_r2(phase, s, cov, y, splits, others, others_C, crops, prior, previous):
    """Round 2: the unsupervised (re-centring only) versions of round 1's best stacks.
    prevothzero_* use no patient labels at all: earlier sessions + other patients."""
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    rows, probs = [], {}
    for k, (cal, test) in splits.items():
        yt = y[test]
        p = {"pooled_unsup_bank28": online(cov, y, cal, test, BANK28, labels="cal",
                                           groups=[others["bank28"]])}
        if previous is not None:
            both, both28 = [previous["bank"], others["bank"]], [previous["bank28"], others["bank28"]]
            p.update({
                "prevoth_unsup_bank": online(cov, y, cal, test, BANK, labels="cal", groups=both),
                "prevoth_unsup_bank28": online(cov, y, cal, test, BANK28, labels="cal", groups=both28),
                "prevothzero_bank28": static(cov, y, cal, test, BANK28, zero=both28),
                "prevothzero_unsup_bank28": online(cov, y, cal, test, BANK28, zero=both28),
            })
        rows.append({"session": phase, "subject": s, "k": k,
                     **{n: 100 * np.mean((q > 0.5).astype(int) == yt) for n, q in p.items()}})
        probs.update({f"{phase}|{s}|{k}|{n}": q.astype(np.float32) for n, q in p.items()})
    return rows, probs


# ---------- driver ----------

def split(cohort, y, k):
    if cohort.startswith("s3"):
        return np.arange(2 * k), np.arange(2 * k, len(y))  # first k (rest, MI) pairs
    return calibration_split(y, k, mode="first")


def run(cohort, jobs, worker, targets=None):
    """Decode every patient of `cohort`, the other patients being the donors. With `targets`,
    decode only those: a second cohort (e.g. "s3locked", with all of `cohort` as the donor
    pool), or a list of `cohort`'s own patients (held out of the pool; a dry run)."""
    started = time.time()
    cov, labels = prepare(cohort)
    pool = {ph: sorted(labels[ph]) for ph in SESSIONS}
    decode = pool
    if isinstance(targets, str):
        extra_cov, extra_labels = prepare(targets)
        assert not any(set(extra_labels[ph]) & set(labels[ph]) for ph in SESSIONS)
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
        for phase in SESSIONS:
            prior = priors(TF[phase], labels[phase], pool[phase], decode[phase], BANK)
            for s in decode[phase]:
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
                          for name, views in (("830", ("830",)), ("bank", BANK), ("bank28", BANK28))}
                others_C = (np.concatenate([centred(cov[phase][o]["830"]) for o in rest]),
                            np.concatenate([labels[phase][o] for o in rest]))
                crops = (np.concatenate([TF[phase][o]["crops"] for o in rest]), others["bank"][1])
                y = labels[phase][s]
                yield delayed(worker)(phase, s, cov[phase][s], y,
                                      {k: split(cohort, y, k) for k in KS}, others, others_C,
                                      crops, prior[s], previous)

    rows, probs = [], {}
    for r, p in Parallel(n_jobs=jobs, return_as="generator")(tasks()):
        rows += r
        probs.update(p)
        first = r[0]
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
    parser.add_argument("--round", type=int, default=1, choices=(1, 2))
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--stats-only", action="store_true",
                        help="recompute the report from the saved per-patient results")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    cohorts = ("s3", "s4") if args.cohort == "both" else (args.cohort,)
    if not args.stats_only:
        suffix = "" if args.round == 1 else "_r2"
        for cohort in cohorts:
            table, probs = run(cohort, args.jobs, run_patient if args.round == 1 else run_patient_r2)
            table.to_csv(RESULTS_DIR / "summary" / f"combo_{cohort}{suffix}.csv", index=False)
            (RESULTS_DIR / "combo").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(RESULTS_DIR / "combo" / f"{cohort}_probs{suffix}.npz", **probs)
            if args.round == 1 and cohort != "s3clean":
                check(cohort, table)
    report({cohort: load_tables(cohort) for cohort in cohorts})


if __name__ == "__main__":
    main()
