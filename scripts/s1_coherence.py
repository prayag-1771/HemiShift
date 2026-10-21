"""S1 step 3j: phase coupling between channels (lagged coherence) as another source (dev).

Covariance-based decoders see power and zero-lag coupling. Lagged coherence (pyriemann
Coherences, coh="lagged") keeps only the phase-lagged part of the coupling, so volume
conduction and the reference contribute little. Stroke changes interhemispheric coupling, so
it may carry class information the covariances miss.
Features: lagged coherence over MI 0.5-4 s (1 s windows, 75 % overlap), averaged within mu
(8-13 Hz) and beta (13-30 Hz), upper triangle (91 pairs per band) -> shrinkage LDA.
  coh-mu, coh-beta, coh-both   the three feature sets
  ens16+coh                    mean of the ens16 and coh-both probabilities (complementarity)
Blocked 4-fold CV, first 39 trials, vs base and ens16 (step 3e).

Usage: .venv/Scripts/python -I scripts/s1_coherence.py
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyriemann.estimation import Coherences
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hemishift.data.harmonise import TARGET_SFREQ
from hemishift.data.liu2024_raw import EXCLUDED, load_raw
from hemishift.data.splits import blocked_folds
from hemishift.eval.stats import chance_threshold, holm, paired_comparison
from hemishift.paths import CONFIG_DIR, RESULTS_DIR


def _script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


classical = _script("s1_classical")
OUT = RESULTS_DIR / "summary" / "s1_coherence_dev.csv"


def coherence_features(X):
    estimator = Coherences(window=128, overlap=0.75, fmin=8.0, fmax=30.0, fs=TARGET_SFREQ, coh="lagged")
    coh = estimator.transform(X)  # (n, ch, ch, freqs)
    freqs = estimator.freqs_  # 1 Hz bins, 8-30 Hz
    upper = np.triu_indices(X.shape[1], k=1)
    out = {}
    for name, (lo, hi) in {"mu": (8.0, 13.0), "beta": (13.0, 30.0)}.items():
        band = coh[..., (freqs >= lo) & (freqs <= hi)].mean(axis=-1)
        out[name] = band[:, upper[0], upper[1]]
    out["both"] = np.hstack([out["mu"], out["beta"]])
    return out


def load(subject):
    raw = load_raw(subject)
    epochs, labels = {}, None
    for name, window in classical.WINDOWS.items():
        X, y, _ = classical.prepare_epochs(raw, classical.Preprocessing(
            template=classical.EXTENDED_TEMPLATE, window=window))
        epochs[name] = X
        labels = y if name == "MI 0.5-4" else labels
    n = min(len(X) for X in epochs.values())
    data = {(w, b): classical.covariances(epochs[w][:n], band)
            for w in classical.WINDOWS for b, band in classical.BANDS.items()}
    return data, coherence_features(epochs["MI 0.5-4"][:n]), (np.asarray(labels[:n]) == "right_hand").astype(int)


def lda_proba(F, y, train, test):
    model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    return model.fit(F[train], y[train]).predict_proba(F[test])[:, 1]


def main():
    split = json.loads((CONFIG_DIR / "splits" / "liu2024_locked_test.json").read_text())
    rows = []
    for subject in split["dev"]:
        if subject in EXCLUDED:
            continue
        data, coh, y = load(subject)
        hits = {}
        for train, test in blocked_folds(len(y), 4):
            p = {"base": classical.candidate_proba(data, y, ("MI 0.5-4", ("8-30",)), train, test),
                 "ens16": np.mean([classical.candidate_proba(data, y, c, train, test)
                                   for c in classical.CANDIDATES], axis=0)}
            for band in ("mu", "beta", "both"):
                p[f"coh-{band}"] = lda_proba(coh[band], y, train, test)
            p["ens16+coh"] = (p["ens16"] + p["coh-both"]) / 2
            for name, prob in p.items():
                hits[name] = hits.get(name, 0) + int(((prob > 0.5).astype(int) == y[test]).sum())
        rows.append({"subject": subject, **{k: 100 * v / len(y) for k, v in hits.items()}})
        print(f"sub {subject:2d}: " + " ".join(f"{k} {v:.0f}" for k, v in rows[-1].items() if k != "subject"))
    table = pd.DataFrame(rows)
    table.to_csv(OUT, index=False)
    threshold = 100 * chance_threshold(39)
    names = ["coh-mu", "coh-beta", "coh-both", "ens16", "ens16+coh"]
    results = {n: paired_comparison(table[n] / 100, table["base"] / 100) for n in names}
    adjusted = dict(zip(names, holm([results[n]["p"] for n in names])))
    print(f"\n{len(table)} dev patients; base mean {table['base'].mean():.1f} %")
    for n in names:
        r = results[n]
        print(f"{n:9s} mean {table[n].mean():5.1f} % | >= chance {(table[n] >= threshold).sum():2d} | vs base "
              f"median {100 * r['median_diff']:+.1f}, p = {r['p']:.3f} (Holm {adjusted[n]:.3f}), "
              f"helped {(table[n] > table['base']).sum()}, harmed {(table[n] < table['base']).sum()}")
    r = paired_comparison(table["ens16+coh"] / 100, table["ens16"] / 100)
    print(f"ens16+coh vs ens16: median {100 * r['median_diff']:+.1f}, p = {r['p']:.3f}")
    print(f"results in {OUT}")


if __name__ == "__main__":
    main()
