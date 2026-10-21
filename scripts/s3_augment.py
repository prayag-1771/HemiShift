"""S3 step 3: LesionSim as extra training data for the own Riemannian decoder (Task B, dev).

The deep transfer route (step 1) did not respond to LesionSim, and it trails the own TS + LR
decoder. Here LesionSim feeds the strong decoder directly, in PLAN.md §4.0's patient-matched
mode. Protocol P3: the first k (rest, MI) pairs calibrate (k = 5, 10), the rest test.
Pooled TS + LR, each domain re-centred on its own Riemannian mean (the patient on its
calibration epochs). All donor epochs together weigh as much as the calibration epochs.
  own        calibration only
  +healthy   + the 109 healthy donors as recorded
  +lesionsim + donors with population-mode LesionSim (as in step 1: attenuation U(0.6, 1),
             shift U(0.2, 0.8), ERD variability 0.5)
  +matched   + donors transformed with THIS patient's operator settings, estimated from the
             calibration epochs (lesion on the left after harmonisation):
               alpha = patient / healthy relative mu ERD on FC3 C3 CP3 (clipped to [0, 1])
               beta  = extra contralesional ERD (FC4 C4 CP4) beyond healthy, as a share of
                       the healthy ipsilateral ERD (clipped to [0, 1])
  +patients  + the other dev patients' epochs (real stroke data; the reference)
Results: results/summary/s3_augment_dev.csv.

Usage: .venv/Scripts/python -I scripts/s3_augment.py
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import tangent_space
from sklearn.linear_model import LogisticRegression

from hemishift.augment.lesionsim import MU_HZ, attenuate_erd, band_split, shift_to_contralesional
from hemishift.data.harmonise import TARGET_SFREQ, common_average
from hemishift.eval.stats import holm, paired_comparison
from hemishift.paths import RESULTS_DIR

_spec = importlib.util.spec_from_file_location("s3_transfer", Path(__file__).with_name("s3_transfer.py"))
s3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s3)

CHANNELS = s3.CHANNELS
IPSI = [CHANNELS.index(ch) for ch in ("FC3", "C3", "CP3")]
CONTRA = [CHANNELS.index(ch) for ch in ("FC4", "C4", "CP4")]
OUT = RESULTS_DIR / "summary" / "s3_augment_dev.csv"


# ---------- Riemannian pooling ----------

def recentred_by_domain(C, domains):
    """Each domain's covariances whitened by that domain's own Riemannian mean."""
    out = np.empty_like(C)
    for d in np.unique(domains):
        rows = domains == d
        W = invsqrtm(mean_riemann(C[rows]))
        out[rows] = W @ C[rows] @ W
    return out


def pooled_proba(C_cal, y_cal, C_test, donor_C, donor_y):
    """TS + LR on the calibration epochs plus re-centred donor epochs.

    Every domain sits at the identity after re-centring, so the tangent space is taken there.
    All donor epochs together weigh as much as the calibration epochs.
    """
    W = invsqrtm(mean_riemann(C_cal))
    identity = np.eye(C_cal.shape[-1])
    X = tangent_space(np.concatenate([W @ C_cal @ W, donor_C]), identity, metric="riemann")
    y = np.concatenate([y_cal, donor_y])
    w = np.concatenate([np.ones(len(y_cal)), np.full(len(donor_y), len(y_cal) / len(donor_y))])
    model = LogisticRegression(C=1.0, max_iter=1000).fit(X, y, sample_weight=w)
    return model.predict_proba(tangent_space(W @ C_test @ W, identity, metric="riemann"))[:, 1]


# ---------- patient-matched operator settings ----------

def relative_erd(X, y, rest_amplitude=None):
    """Relative mu ERD per channel, (MI - rest) / rest, from mean Hilbert envelopes."""
    amplitude = band_split(X, TARGET_SFREQ, MU_HZ)[1].mean(axis=-1)  # (n, channels)
    rest = amplitude[y == 0].mean(axis=0) if rest_amplitude is None else rest_amplitude
    return (amplitude[y == 1].mean(axis=0) - rest) / rest


def matched_settings(X_cal, y_cal, healthy_erd):
    erd = relative_erd(X_cal, y_cal)
    h_ipsi, h_contra = healthy_erd[IPSI].mean(), healthy_erd[CONTRA].mean()
    alpha = np.clip(erd[IPSI].mean() / h_ipsi, 0.0, 1.0)
    beta = np.clip((erd[CONTRA].mean() - h_contra) / h_ipsi, 0.0, 1.0)
    return float(alpha), float(beta)


def matched_mi(X, y, subjects, mirrored, rest_amp, alpha, beta):
    """Donor MI epochs with this patient's attenuation and shift, in the order of y == 1."""
    mi = np.flatnonzero(y == 1)
    out = X[mi].astype(float)
    for s in np.unique(subjects):
        for flipped in (False, True):
            rows = np.flatnonzero((subjects[mi] == s) & (mirrored[mi] == flipped))
            if rows.size == 0:
                continue
            mu = rest_amp[s][MU_HZ][s3.MIRROR] if flipped else rest_amp[s][MU_HZ]
            x = attenuate_erd(out[rows], TARGET_SFREQ, CHANNELS, mu, alpha=alpha)
            x = shift_to_contralesional(x, TARGET_SFREQ, CHANNELS, mu, beta=beta)
            out[rows] = common_average(x)
    return out.astype(np.float32)


def main():
    patients = s3.load_patients()
    X_h, y_h, subjects_h, mirrored, rest_amp = s3.load_healthy()
    print(f"{len(patients)} dev patients; {len(y_h)} healthy donor epochs")
    C_h = s3.premise.covariances(X_h)
    C_ls = s3.premise.covariances(s3.lesionsim_copy(X_h, y_h, subjects_h, mirrored, rest_amp, seed=4))
    donors_h, donors_ls = recentred_by_domain(C_h, subjects_h), recentred_by_domain(C_ls, subjects_h)
    healthy_erd = np.mean([relative_erd(X_h[subjects_h == s], y_h[subjects_h == s])
                           for s in np.unique(subjects_h)], axis=0)
    print("healthy relative mu ERD, ipsilateral (FC3 C3 CP3) "
          f"{healthy_erd[IPSI].mean():+.3f}, contralateral {healthy_erd[CONTRA].mean():+.3f}")
    C_patients = {s: s3.premise.covariances(X) for s, (X, _) in patients.items()}

    rows = []
    for subject, (X, y) in patients.items():
        C = C_patients[subject]
        others = [s for s in patients if s != subject]
        donors_p = np.concatenate([recentred_by_domain(C_patients[s], np.zeros(len(C_patients[s])))
                                   for s in others])
        y_p = np.concatenate([patients[s][1] for s in others])
        for k in s3.KS:
            cal, test = s3.calibration_split(len(y), k)
            alpha, beta = matched_settings(X[cal], y[cal], healthy_erd)
            C_matched = C_h.copy()
            C_matched[y_h == 1] = s3.premise.covariances(
                matched_mi(X_h, y_h, subjects_h, mirrored, rest_amp, alpha, beta))
            probs = {
                "own": s3.own_proba(X, y, cal, test),
                "+healthy": pooled_proba(C[cal], y[cal], C[test], donors_h, y_h),
                "+lesionsim": pooled_proba(C[cal], y[cal], C[test], donors_ls, y_h),
                "+matched": pooled_proba(C[cal], y[cal], C[test],
                                         recentred_by_domain(C_matched, subjects_h), y_h),
                "+patients": pooled_proba(C[cal], y[cal], C[test], donors_p, y_p),
            }
            row = {"subject": subject, "k": k, "alpha": alpha, "beta": beta,
                   **{n: 100 * np.mean((p > 0.5).astype(int) == y[test]) for n, p in probs.items()}}
            rows.append(row)
            print(f"sub {subject:2d} k={k:2d} (alpha {alpha:.2f}, beta {beta:.2f}): "
                  + " ".join(f"{n} {row[n]:.0f}" for n in probs))
        pd.DataFrame(rows).to_csv(OUT, index=False)

    table = pd.DataFrame(rows)
    names = ["+healthy", "+lesionsim", "+matched", "+patients"]
    for k in s3.KS:
        t = table[table["k"] == k]
        print(f"\nk = {k} per class ({len(t)} dev patients): own mean {t['own'].mean():.1f} %")
        results = {n: paired_comparison(t[n] / 100, t["own"] / 100) for n in names}
        adjusted = dict(zip(names, holm([results[n]["p"] for n in names])))
        for n in names:
            r = results[n]
            line = (f"  {n:11s} mean {t[n].mean():5.1f} % | vs own median {100 * r['median_diff']:+.1f}, "
                    f"p = {r['p']:.3f} (Holm {adjusted[n]:.3f}), helped {(t[n] > t['own']).sum()}, "
                    f"harmed (>= 2 pts) {(t[n] <= t['own'] - 2).sum()}")
            if n in ("+lesionsim", "+matched"):
                h = paired_comparison(t[n] / 100, t["+healthy"] / 100)
                line += f" | vs +healthy median {100 * h['median_diff']:+.1f}, p = {h['p']:.3f}"
            print(line)
    print(f"results in {OUT}")


if __name__ == "__main__":
    main()
