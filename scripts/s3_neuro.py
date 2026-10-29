"""S3 neuroscience analysis (PLAN.md §9): descriptive, with a fixed decoder and no tuning.

All 24 Tianjin patients (Task B, affected-hand MI vs rest, lesion normalised to the left).
Nothing here changes a method, so the already-completed locked-test evaluation is unaffected.
1. ERD maps, stroke vs healthy: relative mu (8-13 Hz) and beta (13-30 Hz) ERD per channel,
   (MI - rest) / rest from Hilbert envelopes over 0.5-4 s. Healthy: PhysioNet's 109 subjects,
   right-hand MI plus mirrored left-hand MI vs rest. Per channel: Mann-Whitney, Holm.
   Also the trial-to-trial spread of each patient's ERD at C3 and C4.
2. Why real stroke donors help and healthy donors do not: each patient's MI-vs-rest direction
   in the re-centred tangent space (class-mean difference), its cosine with (a) the pooled
   healthy direction and (b) the pooled direction of the other patients (leave-one-out), and
   its split-half reliability (odd vs even pairs) as the ceiling.
3. Longitudinal: own TS + LR accuracy (blocked 4-fold CV over the 40 pairs) and C3/C4 mu ERD
   at pre, post and follow-up (19 patients have follow-up).
Results: results/summary/s3_neuro_{erd,alignment,longitudinal}.csv.

Usage: .venv/Scripts/python -I scripts/s3_neuro.py
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from pyriemann.utils.tangentspace import tangent_space
from scipy import stats

from hemishift.augment.lesionsim import BETA_HZ, MU_HZ, band_split
from hemishift.data.channels import EXTENDED_TEMPLATE
from hemishift.data.harmonise import TARGET_SFREQ
from hemishift.data.loaders import load_wang2026sf_epochs, wang2026sf_affected_side
from hemishift.data.splits import blocked_folds
from hemishift.eval.stats import holm
from hemishift.models.pooling import recentre_by_domain
from hemishift.paths import RESULTS_DIR


def _script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s3 = _script("s3_transfer")
premise = s3.premise
CHANNELS = EXTENDED_TEMPLATE
C3, C4 = CHANNELS.index("C3"), CHANNELS.index("C4")
SUMMARY = RESULTS_DIR / "summary"


def pairs(subject, phase):
    """(X, y) of the (rest, MI) pairs of one phase, recording order; None if absent."""
    try:
        e = load_wang2026sf_epochs([subject], phases=(phase,), template=CHANNELS)
    except (FileNotFoundError, ValueError):
        return None
    mi = np.flatnonzero(e.labels == "affected_mi")
    paired = mi[(mi > 0) & (e.labels[mi - 1] == "rest")]
    if paired.size < 8:
        return None
    order = np.ravel(np.column_stack([paired - 1, paired]))
    return e.X[order], (e.labels[order] == "affected_mi").astype(int)


def relative_erd(X, y, band):
    """Per channel (MI - rest) / rest of the mean envelope; and per MI trial."""
    amplitude = band_split(X, TARGET_SFREQ, band)[1].mean(axis=-1)  # (n, channels)
    rest = amplitude[y == 0].mean(axis=0)
    per_trial = (amplitude[y == 1] - rest) / rest
    return per_trial.mean(axis=0), per_trial


def own_accuracy(X, y):
    C = premise.covariances(X)
    correct = 0
    for train, test in blocked_folds(len(y), 4):
        A = premise.recenter(C, train)
        correct += int((premise.model().fit(A[train], y[train]).predict(A[test]) == y[test]).sum())
    return 100 * correct / len(y)


def direction(T, y):
    d = T[y == 1].mean(axis=0) - T[y == 0].mean(axis=0)
    return d / np.linalg.norm(d)


def tangent(C, domains):
    return tangent_space(recentre_by_domain(C, domains), np.eye(C.shape[-1]), metric="riemann")


def main():
    subjects = sorted(wang2026sf_affected_side())
    data = {s: pairs(s, "pre") for s in subjects}

    # 1. ERD maps
    X_h, y_h, subjects_h, _, _ = s3.load_healthy()
    rows = []
    for s in np.unique(subjects_h):
        rows_s = subjects_h == s
        for band_name, band in (("mu", MU_HZ), ("beta", BETA_HZ)):
            erd, _ = relative_erd(X_h[rows_s], y_h[rows_s], band)
            rows.append({"group": "healthy", "subject": s, "band": band_name, **dict(zip(CHANNELS, erd))})
    spread = []
    for s, (X, y) in data.items():
        for band_name, band in (("mu", MU_HZ), ("beta", BETA_HZ)):
            erd, per_trial = relative_erd(X, y, band)
            rows.append({"group": "stroke", "subject": s, "band": band_name, **dict(zip(CHANNELS, erd))})
            spread.append({"subject": s, "band": band_name, "sd_C3": per_trial[:, C3].std(),
                           "sd_C4": per_trial[:, C4].std()})
    erd = pd.DataFrame(rows)
    erd.to_csv(SUMMARY / "s3_neuro_erd.csv", index=False)
    print("1. Relative ERD, stroke (24, lesion left) vs healthy (109, 'lesion' = left hemisphere)")
    for band_name in ("mu", "beta"):
        h = erd[(erd.group == "healthy") & (erd.band == band_name)]
        p = erd[(erd.group == "stroke") & (erd.band == band_name)]
        pvals = [stats.mannwhitneyu(h[ch], p[ch]).pvalue for ch in CHANNELS]
        adjusted = holm(pvals)
        print(f"  {band_name}: " + " | ".join(
            f"{ch} {100 * h[ch].mean():+.0f}/{100 * p[ch].mean():+.0f}{'*' if q < 0.05 else ''}"
            for ch, q in zip(CHANNELS, adjusted)))
        li_h = (h["C4"] - h["C3"]) / (h["C4"].abs() + h["C3"].abs())
        li_p = (p["C4"] - p["C3"]) / (p["C4"].abs() + p["C3"].abs())
        print(f"     laterality (C4 - C3) / (|C4| + |C3|): healthy {li_h.median():+.2f}, stroke {li_p.median():+.2f} "
              f"(Mann-Whitney p {stats.mannwhitneyu(li_h, li_p).pvalue:.3f}; negative = stronger ERD at C3)")
    spread = pd.DataFrame(spread)
    for band_name in ("mu", "beta"):
        t = spread[spread.band == band_name]
        print(f"     {band_name} trial-to-trial SD of stroke ERD: C3 {t.sd_C3.median():.2f}, C4 {t.sd_C4.median():.2f}")

    # 2. Direction alignment in the re-centred tangent space
    C_h = premise.covariances(X_h)
    T_h = tangent(C_h, subjects_h)
    w_h = direction(T_h, y_h)
    T_p = {s: tangent(premise.covariances(X), np.zeros(len(y))) for s, (X, y) in data.items()}
    rows = []
    for s, (X, y) in data.items():
        w_s = direction(T_p[s], y)
        others = [o for o in data if o != s]
        w_others = direction(np.concatenate([T_p[o] for o in others]),
                             np.concatenate([data[o][1] for o in others]))
        n_pairs = len(y) // 2
        odd = np.ravel([[2 * i, 2 * i + 1] for i in range(0, n_pairs, 2)])
        even = np.ravel([[2 * i, 2 * i + 1] for i in range(1, n_pairs, 2)])
        rows.append({"subject": s, "cos_healthy": float(w_s @ w_h), "cos_patients": float(w_s @ w_others),
                     "split_half": float(direction(T_p[s][odd], y[odd]) @ direction(T_p[s][even], y[even]))})
    align = pd.DataFrame(rows)
    align.to_csv(SUMMARY / "s3_neuro_alignment.csv", index=False)
    w = stats.wilcoxon(align.cos_patients, align.cos_healthy)
    print("\n2. Cosine between each patient's MI-vs-rest direction and the pooled direction of")
    print(f"   healthy donors {align.cos_healthy.median():.2f} | other patients {align.cos_patients.median():.2f} "
          f"(Wilcoxon p {w.pvalue:.4f}; higher for patients in {(align.cos_patients > align.cos_healthy).sum()}/{len(align)}) "
          f"| split-half within patient {align.split_half.median():.2f}")

    # 3. Longitudinal
    rows = []
    for s in subjects:
        for phase in ("pre", "post", "follow"):
            d = data[s] if phase == "pre" else pairs(s, phase)
            if d is None:
                continue
            X, y = d
            mu, _ = relative_erd(X, y, MU_HZ)
            rows.append({"subject": s, "phase": phase, "accuracy": own_accuracy(X, y),
                         "erd_C3": mu[C3], "erd_C4": mu[C4], "n_pairs": len(y) // 2})
    longitudinal = pd.DataFrame(rows)
    longitudinal.to_csv(SUMMARY / "s3_neuro_longitudinal.csv", index=False)
    wide = longitudinal.pivot(index="subject", columns="phase", values="accuracy")
    print("\n3. Own TS + LR accuracy (blocked 4-fold CV) by phase: " + ", ".join(
        f"{p} {wide[p].mean():.1f} % (n {wide[p].notna().sum()})" for p in ("pre", "post", "follow")))
    for a, b in (("pre", "post"), ("pre", "follow"), ("post", "follow")):
        both = wide[[a, b]].dropna()
        r = stats.wilcoxon(both[b], both[a])
        print(f"   {b} - {a}: mean {(both[b] - both[a]).mean():+.1f} points, Wilcoxon p {r.pvalue:.3f} (n {len(both)})")
    for column in ("erd_C3", "erd_C4"):
        wide_erd = longitudinal.pivot(index="subject", columns="phase", values=column)
        both = wide_erd[["pre", "post"]].dropna()
        print(f"   mu ERD {column[-2:]}: pre {100 * both.pre.mean():+.1f} %, post {100 * both.post.mean():+.1f} % "
              f"(Wilcoxon p {stats.wilcoxon(both.post, both.pre).pvalue:.3f})")


if __name__ == "__main__":
    main()
