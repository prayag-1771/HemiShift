"""S4 (Liu2025) confirmations, pre-registered in ACHIEVEMENTS.md (2026-10-31). Runs once.

27 stroke patients, gait MI vs idle, three sessions each (pre, post, follow-up). In every
session the first k trials per class in recording order calibrate and the rest test.
Decoders use the S3 code unchanged (14 channels, re-centred tangent space + LR):
  own            s3_transfer.own_proba (calibration only, 8-30 Hz)
  pooled         s3_augment.pooled_proba, the other patients' pre sessions as donors
  pooled_fb      pooled with a mu / low-beta / high-beta bank (8-13, 13-20, 20-30 Hz)
  previous       the patient's own earlier sessions as the donors (s3_cross_session)
  previous_zero  the earlier sessions' decoder, no labels: the later session is re-centred
                 on its calibration epochs, whose labels are not used
H-R1 (primary): pooled > own, pre session, k = 5.
H-R2: pooled_fb > pooled, pre session, k = 5.
H-R3: previous > own in the later sessions, k = 5 (each patient's mean over post and follow-up).
H-R4: previous_zero >= own - 2 points in the later sessions, k = 5 (non-inferiority).
All one-sided Wilcoxon signed-rank over patients. Fixed sequence: H-R1 at alpha = 0.05; only
if it is confirmed, H-R2 to H-R4 with Holm at alpha = 0.05.
Descriptive only: k = 10, previous + other patients, a no-label decoder from the other
patients (re-centred on the calibration epochs, labels unused), and co-adaptive own and
pooled decoders (refitted before every test trial on all earlier labelled trials).

Usage: .venv/Scripts/python -I scripts/s4_replication.py
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import tangent_space
from scipy import stats

from hemishift.data.channels import EXTENDED_TEMPLATE
from hemishift.data.loaders import load_moabb_epochs, moabb_cohort
from hemishift.data.splits import calibration_split
from hemishift.eval.stats import chance_threshold, holm
from hemishift.paths import HARMONISED_DIR, RESULTS_DIR


def _script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s3 = _script("s3_transfer")
augment = _script("s3_augment")
cross = _script("s3_cross_session")
sweep = _script("s3_sweep")
KS = (5, 10)
MIN_TRIALS = 20
MARGIN = 2.0  # points; the pre-registered negative-transfer margin
ALPHA = 0.05
FB = ((8.0, 13.0), (13.0, 20.0), (20.0, 30.0))
SESSIONS = ("pre", "post", "follow")
LATER = (("post", ("pre",)), ("follow", ("pre", "post")))
OUT = RESULTS_DIR / "summary" / "s4_replication.csv"


def load_patients(session="pre"):
    """One session's epochs per patient in recording order: X (n, 14, 448), y (MI 1, idle 0).
    Patients excluded by the pre-registered rules are left out."""
    import moabb.datasets as moabb_datasets

    cache = HARMONISED_DIR / f"s4_liu2025_{session}_t14.npz"
    if cache.exists():
        data = np.load(cache)
        patients = {int(key[1:]): (data[key], data["y" + key[1:]])
                    for key in data.files if key.startswith("X")}
    else:
        _, sides, group = moabb_cohort("liu2025")
        dataset = moabb_datasets.Liu2025(sessions=[session])
        patients = {}
        for subject in dataset.subject_list:
            try:
                e = load_moabb_epochs(dataset, sides, "liu2025", group,
                                      events=("gait_imagery", "rest"), subjects=[subject],
                                      template=EXTENDED_TEMPLATE)
            except Exception as exc:  # pre-registered: unreadable sessions are excluded
                print(f"{session} sub {subject}: excluded, cannot be read "
                      f"({type(exc).__name__}: {exc})")
                continue
            patients[subject] = (e.X, (e.labels == "gait_imagery").astype(int))
            print(f"{session} sub {subject} ({sides[subject]}): {len(e.labels)} trials, "
                  f"{patients[subject][1].sum()} MI", flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **{f"X{s}": X for s, (X, _) in patients.items()},
                 **{f"y{s}": y for s, (_, y) in patients.items()})
    for s in list(patients):
        counts = np.bincount(patients[s][1], minlength=2)
        if counts.min() < MIN_TRIALS:
            print(f"{session} sub {s}: excluded, {counts.tolist()} trials per class "
                  f"(pre-registered minimum {MIN_TRIALS})")
            del patients[s]
    return patients


def accuracy(p, y):
    return 100 * np.mean((p > 0.5).astype(int) == y)


def fb_pooled_proba(Cb, y, cal, test, donor_T, donor_y):
    """Pooled TS + LR on the band bank: each band re-centred and mapped at the identity, the
    bands' tangent vectors side by side (s3_sweep's pooled decoder)."""
    T = np.hstack([sweep.tangent(C, cal) for C in Cb])
    model = sweep.lr().fit(np.vstack([T[cal], donor_T]), np.concatenate([y[cal], donor_y]),
                           sample_weight=sweep.donor_weights(len(cal), len(donor_y), 1.0))
    return model.predict_proba(T[test])[:, 1]


def coadaptive_proba(C, y, cal, test, donor_T=None, donor_y=None):
    """Each test trial is decoded by a decoder refitted on the calibration and all earlier
    test trials, with their cue labels (descriptive)."""
    identity = np.eye(C.shape[-1])
    p = np.empty(len(test))
    for i, trial in enumerate(test):
        labelled = np.concatenate([cal, test[:i]])
        W = invsqrtm(mean_riemann(C[labelled]))
        T = tangent_space(W @ C[labelled] @ W, identity, metric="riemann")
        if donor_T is None:
            model = sweep.lr().fit(T, y[labelled])
        else:
            model = sweep.lr().fit(np.vstack([T, donor_T]), np.concatenate([y[labelled], donor_y]),
                                   sample_weight=sweep.donor_weights(len(labelled), len(donor_y), 1.0))
        p[i] = model.predict_proba(tangent_space(W @ C[[trial]] @ W, identity, metric="riemann"))[0, 1]
    return p


def pooling_rows(patients, C):
    """H-R1, H-R2 and the co-adaptive decoders in the pre session; the other patients donate."""
    centred = {s: augment.recentred_by_domain(C[s], np.zeros(len(C[s]))) for s in patients}
    Cb = {s: [sweep.covariances(X, band) for band in FB] for s, (X, _) in patients.items()}
    Tb = {s: np.hstack([sweep.tangent(c, np.arange(len(c))) for c in Cb[s]]) for s in patients}
    T830 = {s: sweep.tangent(C[s], np.arange(len(C[s]))) for s in patients}
    rows = []
    for s, (X, y) in patients.items():
        others = [o for o in patients if o != s]
        donor_C = np.concatenate([centred[o] for o in others])
        donor_y = np.concatenate([patients[o][1] for o in others])
        for k in KS:
            cal, test = calibration_split(y, k, mode="first")
            rows.append({
                "analysis": "pooling", "session": "pre", "subject": s, "k": k, "n_test": len(test),
                "own": accuracy(s3.own_proba(X, y, cal, test), y[test]),
                "pooled": accuracy(augment.pooled_proba(C[s][cal], y[cal], C[s][test], donor_C,
                                                        donor_y), y[test]),
                "pooled_fb": accuracy(fb_pooled_proba(Cb[s], y, cal, test,
                                                      np.vstack([Tb[o] for o in others]), donor_y),
                                      y[test]),
                "donor_only": accuracy(cross.donor_only(C[s][cal], C[s][test], donor_C, donor_y),
                                       y[test]),
                "own_coadaptive": accuracy(coadaptive_proba(C[s], y, cal, test), y[test]),
                "pooled_coadaptive": accuracy(coadaptive_proba(
                    C[s], y, cal, test, np.vstack([T830[o] for o in others]), donor_y), y[test]),
            })
        first = rows[-len(KS)]
        print(f"pre sub {s:2d} k={first['k']}: " + " ".join(
            f"{n} {first[n]:.1f}" for n in ("own", "pooled", "pooled_fb", "own_coadaptive",
                                              "pooled_coadaptive")), flush=True)
    return rows


def cross_session_rows(sessions, C):
    """H-R3 and H-R4: later sessions with the patient's own earlier sessions as donors."""
    centred = {ph: {s: augment.recentred_by_domain(c, np.zeros(len(c))) for s, c in data.items()}
               for ph, data in C.items()}
    rows = []
    for target, previous in LATER:
        for s, (X, y) in sessions[target].items():
            if any(s not in sessions[p] for p in previous):
                continue
            prev = (np.concatenate([centred[p][s] for p in previous]),
                    np.concatenate([sessions[p][s][1] for p in previous]))
            others = [o for o in sessions[target] if o != s]
            oth = (np.concatenate([centred[target][o] for o in others]),
                   np.concatenate([sessions[target][o][1] for o in others]))
            Cs = C[target][s]
            for k in KS:
                cal, test = calibration_split(y, k, mode="first")
                rows.append({
                    "analysis": "cross_session", "session": target, "subject": s, "k": k,
                    "n_test": len(test),
                    "own": accuracy(s3.own_proba(X, y, cal, test), y[test]),
                    "previous": accuracy(cross.pooled_groups(Cs[cal], y[cal], Cs[test], [prev]),
                                         y[test]),
                    "previous_zero": accuracy(cross.donor_only(Cs[cal], Cs[test], *prev), y[test]),
                    "previous+others": accuracy(cross.pooled_groups(Cs[cal], y[cal], Cs[test],
                                                                    [prev, oth]), y[test]),
                })
            first = rows[-len(KS)]
            print(f"{target} sub {s:2d} k={first['k']}: " + " ".join(
                f"{n} {first[n]:.1f}" for n in ("own", "previous", "previous_zero",
                                                  "previous+others")), flush=True)
    return rows


def one_sided(diff):
    """One-sided Wilcoxon signed-rank p for median(diff) > 0 (zeros dropped)."""
    diff = np.asarray(diff, dtype=float)
    nonzero = diff[diff != 0]
    return stats.wilcoxon(nonzero, alternative="greater").pvalue if nonzero.size else 1.0


def describe(label, a, b, t):
    diff = (t[a] - t[b]).to_numpy()
    boot = np.random.default_rng(0).choice(diff, size=(10_000, len(diff))).mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    above = {m: int(sum(acc >= 100 * chance_threshold(n) for acc, n in zip(t[m], t["n_test"])))
             for m in (a, b)}
    print(f"  {label}: {a} {t[a].mean():.1f} % vs {b} {t[b].mean():.1f} % | mean diff "
          f"{diff.mean():+.1f} (95 % CI {low:+.1f} to {high:+.1f}), median {np.median(diff):+.1f}, "
          f"helped {(diff > 0).sum()}, harmed {(diff < 0).sum()} | above chance: "
          f"{a} {above[a]}, {b} {above[b]} of {len(t)}")


def main():
    sessions = {ph: load_patients(ph) for ph in SESSIONS}
    print({ph: len(p) for ph, p in sessions.items()}, "patients per session")
    C = {ph: {s: s3.premise.covariances(X) for s, (X, _) in data.items()}
         for ph, data in sessions.items()}
    table = pd.DataFrame(pooling_rows(sessions["pre"], C["pre"]) + cross_session_rows(sessions, C))
    table.to_csv(OUT, index=False)

    pool = table[table["analysis"] == "pooling"]
    later = table[table["analysis"] == "cross_session"]
    means = ["own", "previous", "previous_zero", "previous+others"]
    per_patient = later.groupby(["k", "subject"])[means].mean()
    per_patient["n_test"] = later.groupby(["k", "subject"])["n_test"].min()
    p5, l5 = pool[pool["k"] == 5], per_patient.loc[5]

    print("\n=== pre-registered hypotheses (one-sided Wilcoxon; fixed sequence, then Holm) ===")
    p1 = one_sided(p5["pooled"] - p5["own"])
    print(f"  H-R1 pooled > own (pre, k = 5): p = {p1:.4f} -> "
          f"{'CONFIRMED' if p1 < ALPHA else 'not confirmed'}")
    family = {
        "H-R2 pooled_fb > pooled (pre, k = 5)": one_sided(p5["pooled_fb"] - p5["pooled"]),
        "H-R3 previous > own (later sessions, k = 5)": one_sided(l5["previous"] - l5["own"]),
        f"H-R4 previous_zero >= own - {MARGIN:g} (later sessions, k = 5)":
            one_sided(l5["previous_zero"] - l5["own"] + MARGIN),
    }
    adjusted = holm(list(family.values()))
    for (name, p), a in zip(family.items(), adjusted):
        verdict = ("CONFIRMED" if a < ALPHA else "not confirmed") if p1 < ALPHA else \
            "not tested (H-R1 not confirmed); p is descriptive"
        print(f"  {name}: p = {p:.4f}, Holm {a:.4f} -> {verdict}")

    print("\n=== details ===")
    for k in KS:
        print(f"k = {k}")
        t = pool[pool["k"] == k]
        describe("pre", "pooled", "own", t)
        describe("pre", "pooled_fb", "pooled", t)
        describe("pre (descriptive): no labels, other patients only", "donor_only", "own", t)
        describe("pre (descriptive)", "own_coadaptive", "own", t)
        describe("pre (descriptive)", "pooled_coadaptive", "pooled", t)
        t = per_patient.loc[k]
        for m in ("previous", "previous_zero", "previous+others"):
            describe("later sessions, patient means", m, "own", t)
        for target, _ in LATER:
            describe(f"{target} only", "previous", "own",
                     later[(later["k"] == k) & (later["session"] == target)])
    print(f"results in {OUT}")


if __name__ == "__main__":
    main()
