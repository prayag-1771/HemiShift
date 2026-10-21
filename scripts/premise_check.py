"""Premise check: does pooling other patients beat a patient's own few trials?

--cohort liu2024 (S1, Task A, the default) or wang2026sf (S3, Task B: affected-hand MI vs rest).

Everything in HemiShift assumes that more (aligned) data from other people helps a new
patient. Before building on that, test it on the hardest cohort with a fast Riemannian model:

  R0 own       : tangent-space + logistic regression on the patient's first k trials per class
  R1 pooled    : the same model trained on all other S1 patients plus those k trials
  R1 balanced  : pooled, but the target's k trials weigh as much as all other patients together
  R1 rotate    : re-centre + Riemannian rotation of every source patient onto the target's
                 class means (pyriemann TLCenter + TLRotate), then balanced TS + LR
  R1 mdwm      : minimum distance to a weighted mean of source and target class means
                 (MDWM, domain trade-off 0.5)

Each patient's covariances are re-centred to the identity (Riemannian Procrustes /
"Riemannian alignment"). The reference for the target patient comes from its calibration
trials only, and for training patients from their own trials (PLAN.md §7, rule 3). Both
models are scored on the target's remaining trials. Exploratory; not pre-registered.

Usage: .venv/Scripts/python -I scripts/premise_check.py [--k 10]
"""

import argparse

import numpy as np
import pandas as pd
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.mean import mean_riemann
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from hemishift.data.harmonise import TARGET_SFREQ
from hemishift.data.loaders import load_moabb_epochs, moabb_cohort
from hemishift.data.splits import calibration_split
from hemishift.eval.stats import paired_comparison
from hemishift.paths import RESULTS_DIR


VARIANTS = ("own", "pooled", "balanced", "mdwm")  # "rotate" is opt-in: --rotate


def covariances(X):
    sos = butter(4, (8.0, 30.0), btype="bandpass", fs=TARGET_SFREQ, output="sos")
    return Covariances(estimator="oas").transform(sosfiltfilt(sos, X, axis=-1))


def recenter(C, reference_trials):
    """Whiten every covariance by the Riemannian mean of `reference_trials`."""
    W = invsqrtm(mean_riemann(C[reference_trials]))
    return W @ C @ W


def model():
    return make_pipeline(TangentSpace(metric="riemann"), LogisticRegression(C=1.0, max_iter=1000))


def transfer_predict(kind, C_src, y_src, d_src, C_cal, y_cal, C_test, target):
    """Fit a pyriemann transfer pipeline on sources + target calibration; predict target test.

    Centering statistics for the target come from its calibration trials only, the only
    target trials passed to fit().
    """
    from pyriemann.transfer import MDWM, TLCenter, TLClassifier, TLRotate, encode_domains

    X = np.concatenate([C_src, C_cal])
    y = np.concatenate([y_src, y_cal])
    domains = np.concatenate([d_src, np.full(len(y_cal), target)])
    X, y_encoded = encode_domains(X, y, domains)
    if kind == "rotate":
        weight = len(y_cal) / len(y_src)  # all sources together weigh as much as the target
        weights = {d: (1.0 if d == target else weight) for d in np.unique(domains)}
        steps = [TLCenter(target_domain=target), TLRotate(target_domain=target),
                 TLClassifier(target_domain=target, estimator=model(), domain_weight=weights)]
    elif kind == "mdwm":
        steps = [TLCenter(target_domain=target), MDWM(domain_tradeoff=0.5, target_domain=target)]
    else:
        raise ValueError(kind)
    return make_pipeline(*steps).fit(X, y_encoded).predict(C_test)


def load_task(cohort):
    """(X, y, patients) in recording order within each patient.

    liu2024    Task A: affected vs unaffected hand (left/right, lesion normalised)
    wang2026sf Task B: affected-hand MI vs the rest epoch right before it, pre-training phase
    """
    if cohort == "liu2024":
        dataset, affected_side, group = moabb_cohort("liu2024")
        epochs = load_moabb_epochs(dataset, affected_side, cohort="liu2024", group=group)
        return epochs.X, epochs.labels, epochs.meta["patient"].to_numpy()
    if cohort == "wang2026sf":
        from hemishift.data.loaders import load_wang2026sf_epochs, wang2026sf_affected_side

        X_all, y_all, p_all = [], [], []
        for subject in sorted(wang2026sf_affected_side()):
            e = load_wang2026sf_epochs([subject], phases=("pre",))
            mi = np.flatnonzero(e.labels == "affected_mi")
            paired = mi[(mi > 0) & (e.labels[mi - 1] == "rest")]
            order = np.ravel(np.column_stack([paired - 1, paired]))  # rest, MI, rest, MI, ...
            X_all.append(e.X[order])
            y_all.append(np.where(e.labels[order] == "rest", "rest", "mi"))
            p_all.append(np.full(order.size, f"wang2026sf:{subject}"))
            print(f"loaded wang2026sf:{subject} ({order.size} epochs)")
        return np.concatenate(X_all), np.concatenate(y_all), np.concatenate(p_all)
    raise ValueError(cohort)


def main():
    parser = argparse.ArgumentParser(description="Pooled vs own-trials premise check.")
    parser.add_argument("--cohort", default="liu2024", choices=["liu2024", "wang2026sf"])
    parser.add_argument("--k", type=int, default=10, help="calibration trials per class")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N patients")
    parser.add_argument("--rotate", action="store_true",
                        help="also run TLRotate (~13 s per source patient per target: hours for S1)")
    args = parser.parse_args()
    global VARIANTS
    if args.rotate:
        VARIANTS = VARIANTS + ("rotate",)

    X, y_all, patients = load_task(args.cohort)
    C_all = covariances(X)

    # Training patients are aligned with all of their own trials (labels unused).
    aligned = np.empty_like(C_all)
    for p in np.unique(patients):
        rows = np.flatnonzero(patients == p)
        aligned[rows] = recenter(C_all[rows], np.arange(rows.size))

    rows_out = []
    for target in pd.unique(patients)[: args.limit]:
        rows = np.flatnonzero(patients == target)
        calib, test = calibration_split(y_all[rows], args.k, mode="first")
        C_target = recenter(C_all[rows], calib)  # reference from calibration trials only
        y_target = y_all[rows]
        own = model().fit(C_target[calib], y_target[calib])
        others = patients != target
        X_pool = np.concatenate([aligned[others], C_target[calib]])
        y_pool = np.concatenate([y_all[others], y_target[calib]])
        pooled = model().fit(X_pool, y_pool)
        balance = np.concatenate([np.full(others.sum(), calib.size / others.sum()),
                                  np.ones(calib.size)])
        balanced = model().fit(X_pool, y_pool, logisticregression__sample_weight=balance)

        def accuracy(predicted):
            return 100 * np.mean(predicted == y_target[test])

        row = {"patient": target, "n_test": test.size,
               "own": accuracy(own.predict(C_target[test])),
               "pooled": accuracy(pooled.predict(C_target[test])),
               "balanced": accuracy(balanced.predict(C_target[test]))}
        for kind in [v for v in ("mdwm", "rotate") if v in VARIANTS]:  # align raw covariances
            row[kind] = accuracy(transfer_predict(kind, C_all[others], y_all[others],
                                                  patients[others], C_all[rows][calib],
                                                  y_target[calib], C_all[rows][test], target))
        rows_out.append(row)
        print(target + ": " + "  ".join(f"{k} {row[k]:.0f}" for k in VARIANTS))

    table = pd.DataFrame(rows_out)
    suffix = f"_first{args.limit}" if args.limit else ""  # test runs never overwrite full ones
    out = RESULTS_DIR / "summary" / f"premise_check_{args.cohort}_k{args.k}{suffix}.csv"
    table.to_csv(out, index=False)
    print(f"\nown (R0) mean {table['own'].mean():.1f} %   (results in {out})")
    for variant in VARIANTS[1:]:
        r = paired_comparison(table[variant] / 100, table["own"] / 100)
        print(f"{variant:9s} mean {table[variant].mean():5.1f} % | vs own: median "
              f"{100 * r['median_diff']:+.1f} pts (95 % CI {100 * r['ci95'][0]:+.1f}.."
              f"{100 * r['ci95'][1]:+.1f}), p = {r['p']:.4f} | helped "
              f"{(table[variant] > table['own']).sum()}, harmed {(table[variant] < table['own']).sum()}")


if __name__ == "__main__":
    main()
