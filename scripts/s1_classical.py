"""S1 step 3e: classical decoder upgrades, one at a time (dev patients).

Same evaluation as steps 3a-3d: blocked 4-fold CV per development patient, with every
feature, reference and filter fitted on training trials only. Every recording's last trial
is dropped for all variants, because the rebound window (used by the ensembles) runs past the
end of the file. The baseline is therefore the step 3d "fixed" decoder.
Variants, each compared with the baseline (Holm-corrected across variants):
  base      TS + LR (C = 1), 14 channels, MI 0.5-4 s, 8-30 Hz, OAS, training-mean re-centring
  C=0.1     stronger L2 penalty
  lda       TS + shrinkage LDA
  mdm       minimum distance to the class means (no tangent space)
  fgmdm     MDM after geodesic filtering
  csp       CSP (3 pairs) + LDA on log-variance, the classic recipe
  motor8    the 8 motor-strip channels only (FC3 FCz FC4 C3 Cz C4 CP3 CP4)
  full29    all 29 scalp channels
  crops2s   five 2 s crops per trial for training; test probabilities averaged over crops
  potato    training trials far from the training mean (robust z > 2.5) dropped
  ens12     probabilities averaged over the 12 MI-window candidates of step 3d
  ens16     the same over all 16 candidates (adds the rebound window)

Usage: .venv/Scripts/python -I scripts/s1_classical.py
"""

import json
from itertools import product

import numpy as np
import pandas as pd
from pyriemann.classification import MDM, FgMDM
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.mean import mean_riemann
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from hemishift.data.channels import EXTENDED_TEMPLATE
from hemishift.data.harmonise import TARGET_SFREQ
from hemishift.data.liu2024_raw import EXCLUDED, load_raw
from hemishift.data.splits import blocked_folds
from hemishift.eval.stats import chance_threshold, holm, paired_comparison
from hemishift.paths import CONFIG_DIR, RESULTS_DIR
from hemishift.preprocess.pipeline import Preprocessing, prepare_epochs

WINDOWS = {"MI 0.5-4": (0.5, 4.0), "MI 0.5-2.5": (0.5, 2.5), "MI 2-4": (2.0, 4.0),
           "rebound 4-6": (4.0, 6.0)}
BANDS = {"mu": (8.0, 13.0), "beta": (13.0, 30.0), "8-30": (8.0, 30.0)}
CANDIDATES = [(w, (b,)) for w, b in product(WINDOWS, BANDS)] + [(w, tuple(BANDS)) for w in WINDOWS]
MOTOR8 = ("FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CP4")
FULL29 = ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FT7", "FC3", "FCz", "FC4", "FT8", "T7", "C3",
          "Cz", "C4", "T8", "TP7", "CP3", "CP4", "TP8", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2")
CROP = 256  # 2 s at 128 Hz
OUT = RESULTS_DIR / "summary" / "s1_classical_dev.csv"


def covariances(X, band=BANDS["8-30"]):
    sos = butter(4, band, btype="bandpass", fs=TARGET_SFREQ, output="sos")
    return Covariances("oas").transform(sosfiltfilt(sos, X, axis=-1))


def recentre(C, reference):
    W = invsqrtm(mean_riemann(C[reference]))
    return W @ C @ W


def load(subject):
    """Covariances for every variant, cut to the trials all windows share; and labels."""
    raw = load_raw(subject)
    epochs, labels = {}, None
    for name, window in WINDOWS.items():
        X, y, _ = prepare_epochs(raw, Preprocessing(template=EXTENDED_TEMPLATE, window=window))
        epochs[name] = X
        labels = y if name == "MI 0.5-4" else labels
    n = min(len(X) for X in epochs.values())  # the rebound window loses the last trial
    data = {(w, b): covariances(epochs[w][:n], band) for w in WINDOWS for b, band in BANDS.items()}
    data["motor8"] = covariances(prepare_epochs(raw, Preprocessing(template=MOTOR8))[0][:n])
    data["full29"] = covariances(prepare_epochs(raw, Preprocessing(template=FULL29))[0][:n])
    sos = butter(4, BANDS["8-30"], btype="bandpass", fs=TARGET_SFREQ, output="sos")
    filtered = sosfiltfilt(sos, epochs["MI 0.5-4"][:n], axis=-1)
    starts = np.linspace(0, filtered.shape[-1] - CROP, 5).astype(int)
    crops = np.stack([filtered[..., s:s + CROP] for s in starts], axis=1)  # (n, 5, ch, CROP)
    data["crops"] = Covariances("oas").transform(crops.reshape(-1, *crops.shape[2:])) \
        .reshape(n, len(starts), crops.shape[2], crops.shape[2])
    y = (np.asarray(labels[:n]) == "right_hand").astype(int)
    return data, y


def ts_proba(C, y, train, test, classifier):
    A = recentre(C, train)
    space = TangentSpace(metric="riemann").fit(A[train])
    classifier.fit(space.transform(A[train]), y[train])
    return classifier.predict_proba(space.transform(A[test]))[:, 1]


def lr():
    return LogisticRegression(C=1.0, max_iter=1000)


def candidate_proba(data, y, candidate, train, test):
    """Step 3d candidate: one tangent block per band, joined, then LR."""
    window, bands = candidate
    blocks_train, blocks_test = [], []
    for band in bands:
        A = recentre(data[(window, band)], train)
        space = TangentSpace(metric="riemann").fit(A[train])
        blocks_train.append(space.transform(A[train]))
        blocks_test.append(space.transform(A[test]))
    model = lr().fit(np.hstack(blocks_train), y[train])
    return model.predict_proba(np.hstack(blocks_test))[:, 1]


def crops_proba(data, y, train, test):
    C = data["crops"]
    n_crops = C.shape[1]
    flat = C.reshape(-1, *C.shape[2:])
    rows = lambda trials: (trials[:, None] * n_crops + np.arange(n_crops)).ravel()  # noqa: E731
    A = recentre(flat, rows(train))
    space = TangentSpace(metric="riemann").fit(A[rows(train)])
    model = lr().fit(space.transform(A[rows(train)]), np.repeat(y[train], n_crops))
    return model.predict_proba(space.transform(A[rows(test)]))[:, 1].reshape(-1, n_crops).mean(1)


def potato_proba(data, y, train, test, z_max=2.5):
    C = data[("MI 0.5-4", "8-30")]
    A = recentre(C, train)
    log_eigs = np.log(np.linalg.eigvalsh(A[train]))
    distance = np.sqrt((log_eigs ** 2).sum(axis=1))  # Riemannian distance to the training mean
    mad = 1.4826 * np.median(np.abs(distance - np.median(distance)))
    keep = train[(distance - np.median(distance)) / mad <= z_max]
    return ts_proba(C, y, keep, test, lr())


def variants(data, y, train, test):
    """P(right hand) on the test trials, per variant."""
    base = data[("MI 0.5-4", "8-30")]
    out = {
        "base": ts_proba(base, y, train, test, lr()),
        "C=0.1": ts_proba(base, y, train, test, LogisticRegression(C=0.1, max_iter=1000)),
        "lda": ts_proba(base, y, train, test,
                        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        "mdm": MDM(metric="riemann").fit(recentre(base, train)[train], y[train])
        .predict_proba(recentre(base, train)[test])[:, 1],
        "fgmdm": FgMDM(metric="riemann").fit(recentre(base, train)[train], y[train])
        .predict_proba(recentre(base, train)[test])[:, 1],
        "motor8": ts_proba(data["motor8"], y, train, test, lr()),
        "full29": ts_proba(data["full29"], y, train, test, lr()),
        "crops2s": crops_proba(data, y, train, test),
        "potato": potato_proba(data, y, train, test),
    }
    csp = CSP(nfilter=6, log=True).fit(base[train], y[train])
    out["csp"] = LinearDiscriminantAnalysis().fit(csp.transform(base[train]), y[train]) \
        .predict_proba(csp.transform(base[test]))[:, 1]
    probs = {c: candidate_proba(data, y, c, train, test) for c in CANDIDATES}
    out["ens12"] = np.mean([p for c, p in probs.items() if not c[0].startswith("rebound")], axis=0)
    out["ens16"] = np.mean(list(probs.values()), axis=0)
    return out


def main():
    split = json.loads((CONFIG_DIR / "splits" / "liu2024_locked_test.json").read_text())
    rows = []
    for subject in split["dev"]:
        if subject in EXCLUDED:
            continue
        data, y = load(subject)
        hits = {}
        for train, test in blocked_folds(len(y), 4):
            for name, p in variants(data, y, train, test).items():
                hits[name] = hits.get(name, 0) + int(((p > 0.5).astype(int) == y[test]).sum())
        rows.append({"subject": subject, "n_trials": len(y),
                     **{k: 100 * v / len(y) for k, v in hits.items()}})
        print(f"sub {subject:2d}: " + " ".join(f"{k} {v:.0f}" for k, v in rows[-1].items()
                                                if k not in ("subject", "n_trials")))
    table = pd.DataFrame(rows)
    table.to_csv(OUT, index=False)

    threshold = 100 * chance_threshold(40)
    names = [c for c in table.columns if c not in ("subject", "n_trials", "base")]
    results = {name: paired_comparison(table[name] / 100, table["base"] / 100) for name in names}
    adjusted = dict(zip(names, holm([results[n]["p"] for n in names])))
    print(f"\n{len(table)} dev patients, blocked 4-fold CV; base mean {table['base'].mean():.1f} %, "
          f"{(table['base'] >= threshold).sum()} at or above {threshold:.0f} %")
    for name in sorted(names, key=lambda n: -table[n].mean()):
        r = results[name]
        print(f"{name:8s} mean {table[name].mean():5.1f} % | >= chance {(table[name] >= threshold).sum():2d}"
              f" | vs base median {100 * r['median_diff']:+.1f}, p = {r['p']:.3f} (Holm {adjusted[name]:.3f})"
              f", helped {(table[name] > table['base']).sum()}, harmed {(table[name] < table['base']).sum()}")
    print(f"results in {OUT}")


if __name__ == "__main__":
    main()
