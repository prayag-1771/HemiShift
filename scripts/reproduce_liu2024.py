"""Gate G1: reproduce Liu et al. 2024's within-patient baselines on S1.

Paper (Sci Data 11:131): CSP+LDA 55.57 %, FBCSP+SVM 57.57 %. Gate G1 passes when the
reproduction lands within ±3 points of the paper for both pipelines (PLAN.md §11).

--mode paper (default) mirrors the authors' MATLAB code (figshare code.zip: CSP_LDA.m,
FBCSP_SVM.m, CSP.m, CSPFilter.m, FBCSP.m, BandpassFilter.m, NotchFilter.m) line by line:
  - epoch MI onset -1.6 .. +4.0 s at 500 Hz, all 29 EEG channels, microvolts
  - causal 50 Hz notch (iirnotch, Q = 6), then a causal 2nd-order Butterworth band-pass
    4-30 Hz, then keep 0-4 s
  - CSP on trace-normalised trial covariances, 2 filter pairs; training features log(var),
    test features log(1 + var) (CSPFilter.m does this); LDA
  - FBCSP: causal 4th-order IIR band-pass per band 10-13, 13-16, 16-19, 19-22, 22-25 Hz,
    2 pairs each, all 20 log-variance features, linear SVM (fitcecoc default)
  - 10 random 24/16 splits per patient, not stratified
--mode moabb is the first attempt: MOABB filtering, MNE CSP, stratified CV.

Usage: .venv/Scripts/python -I scripts/reproduce_liu2024.py [--mode paper|moabb] [--subjects N]
"""

import argparse

import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, lfilter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
from sklearn.svm import SVC

from hemishift.data.loaders import configure_moabb
from hemishift.eval.stats import chance_threshold
from hemishift.paths import RESULTS_DIR

PAPER = {"CSP+LDA": 55.57, "FBCSP+SVM": 57.57}
FS = 500
FB_BANDS = [(10, 13), (13, 16), (16, 19), (19, 22), (22, 25)]


# ----- the authors' pipeline, ported from their MATLAB code -----

def paper_epochs(raw):
    """(40 trials, 29 channels, 2000 samples) and labels 1/2, as CSP_LDA.m builds them."""
    events = mne.find_events(raw, shortest_event=0, verbose=False)
    events = events[np.isin(events[:, 2], [1, 2])]
    data = raw.get_data(picks="eeg") * 1e6  # volts -> microvolts, like their .mat files
    b_notch, a_notch = iirnotch(50 / (FS / 2), (50 / (FS / 2)) / 6)
    b_band, a_band = butter(2, [4 / (FS / 2), 30 / (FS / 2)], btype="band")
    trials = []
    for onset in events[:, 0]:
        segment = data[:, onset - 800: onset + 2000]  # MI onset -1.6 .. +4.0 s
        segment = lfilter(b_band, a_band, lfilter(b_notch, a_notch, segment, axis=1), axis=1)
        trials.append(segment[:, 800:])  # keep MI onset .. +4.0 s
    return np.stack(trials), events[:, 2]


def csp_filters(X, y, pairs=2):
    """CSP.m: trace-normalised covariances, whitening, 2 * pairs filters (rows)."""
    covs = np.einsum("nct,ndt->ncd", X, X)
    covs /= np.trace(covs, axis1=1, axis2=2)[:, None, None]
    classes = np.unique(y)
    c1, c2 = (covs[y == c].mean(0) for c in classes)
    values, vectors = np.linalg.eigh(c1 + c2)
    order = np.argsort(values)[::-1]
    whitening = np.diag(1 / np.sqrt(values[order])) @ vectors[:, order].T
    values1, vectors1 = np.linalg.eigh(whitening @ c1 @ whitening.T)
    rotation = vectors1[:, np.argsort(values1)[::-1]]
    filters = rotation.T @ whitening
    return filters[np.r_[0:pairs, len(filters) - pairs:len(filters)]]


def log_var(filters, X, plus_one=False):
    variance = np.var(np.einsum("fc,nct->nft", filters, X), axis=2, ddof=1)
    return np.log(1 + variance) if plus_one else np.log(variance)


def paper_csp_lda(X_train, y_train, X_test):
    filters = csp_filters(X_train, y_train)
    model = LinearDiscriminantAnalysis().fit(log_var(filters, X_train), y_train)
    return model.predict(log_var(filters, X_test, plus_one=True))  # CSPFilter.m: log(1 + var)


def paper_fbcsp_svm(X_train, y_train, X_test):
    train_features, test_features = [], []
    for low, high in FB_BANDS:
        b, a = butter(2, [low / (FS / 2), high / (FS / 2)], btype="band")  # 4th-order bandpass
        band_train, band_test = lfilter(b, a, X_train, axis=2), lfilter(b, a, X_test, axis=2)
        filters = csp_filters(band_train, y_train)
        train_features.append(log_var(filters, band_train))
        test_features.append(log_var(filters, band_test))
    model = SVC(kernel="linear", C=1.0).fit(np.hstack(train_features), y_train)
    return model.predict(np.hstack(test_features))


def run_paper(dataset, subjects, seed=0):
    rows = []
    splitter = ShuffleSplit(n_splits=10, train_size=24, test_size=16, random_state=seed)
    for subject in subjects:
        raw = dataset.get_data(subjects=[subject])[subject]["0"]["0"]
        X, y = paper_epochs(raw)
        for name, pipeline in (("CSP+LDA", paper_csp_lda), ("FBCSP+SVM", paper_fbcsp_svm)):
            scores = [np.mean(pipeline(X[tr], y[tr], X[te]) == y[te]) for tr, te in splitter.split(X)]
            rows.append({"mode": "paper", "pipeline": name, "protocol": "10x random 24/16",
                         "subject": subject, "n_trials": len(y), "accuracy": 100 * np.mean(scores)})
        print(f"subject {subject}: CSP+LDA {rows[-2]['accuracy']:.1f}  FBCSP+SVM {rows[-1]['accuracy']:.1f}")
    return rows


# ----- first attempt: MOABB preprocessing with MNE/sklearn defaults -----

def run_moabb(dataset, subjects, seed=0):
    from mne.decoding import CSP
    from moabb.paradigms import FilterBankMotorImagery, MotorImagery
    from moabb.pipelines.utils import FilterBank
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline

    events = ["left_hand", "right_hand"]
    setups = {
        "CSP+LDA": (MotorImagery(n_classes=2, events=events, fmin=8, fmax=30, tmin=0, tmax=4,
                                 resample=128),
                    lambda: make_pipeline(CSP(n_components=6, log=True), LinearDiscriminantAnalysis())),
        "FBCSP+SVM": (FilterBankMotorImagery(n_classes=2, events=events, tmin=0, tmax=4, resample=128,
                                             filters=[[f, f + 4] for f in range(4, 40, 4)]),
                      lambda: make_pipeline(FilterBank(make_pipeline(CSP(n_components=4, log=True))),
                                            SelectKBest(mutual_info_classif, k=8), SVC(kernel="linear"))),
    }
    protocols = {"10-fold": StratifiedKFold(10, shuffle=True, random_state=seed),
                 "10x 60/40": StratifiedShuffleSplit(10, test_size=0.4, random_state=seed)}
    rows = []
    for name, (paradigm, make) in setups.items():
        X, y, meta = paradigm.get_data(dataset, subjects=subjects)
        for subject in subjects:
            mask = meta["subject"].to_numpy() == subject
            for protocol, cv in protocols.items():
                scores = cross_val_score(make(), X[mask], y[mask], cv=cv)
                rows.append({"mode": "moabb", "pipeline": name, "protocol": protocol,
                             "subject": subject, "n_trials": int(mask.sum()),
                             "accuracy": 100 * scores.mean()})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Reproduce Liu2024 within-patient baselines.")
    parser.add_argument("--mode", choices=["paper", "moabb"], default="paper")
    parser.add_argument("--subjects", type=int, default=None, help="first N subjects only")
    args = parser.parse_args()

    configure_moabb()
    from moabb.datasets import Liu2024

    dataset = Liu2024()
    subjects = dataset.subject_list[: args.subjects] if args.subjects else dataset.subject_list
    rows = (run_paper if args.mode == "paper" else run_moabb)(dataset, subjects)

    results = pd.DataFrame(rows)
    out = RESULTS_DIR / "summary" / f"reproduce_liu2024_{args.mode}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)

    threshold = 100 * chance_threshold(16)  # each split tests 16 trials
    summary = results.groupby(["pipeline", "protocol"])["accuracy"].agg(["mean", "std", "count"])
    summary["paper"] = [PAPER[p] for p, _ in summary.index]
    summary["diff"] = summary["mean"] - summary["paper"]
    print(summary.round(2).to_string())
    print(f"\nchance threshold for one 16-trial test split: {threshold:.1f} %")
    passed = all((summary.loc[name, "diff"].abs() <= 3).any() for name in PAPER)
    print("Gate G1:", "PASS" if passed else "FAIL", f"(results in {out})")


if __name__ == "__main__":
    main()
