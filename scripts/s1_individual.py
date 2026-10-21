"""S1 step 3g: individual frequency band and one long window (dev patients).

Two label-free, per-patient refinements of the step 3e decoders:
  iband     mu band centred on the patient's own rhythm peak (peak +- 2 Hz; the peak is found
            above the 1/f background over the whole recording at C3/C4/Pz, as in the audit;
            8-13 Hz when there is no peak)
  long      one 0.5-6 s window (MI + rebound) instead of separate windows, 8-30 Hz
  ens16-i   the step 3e ensemble with the individual band in place of 8-13 Hz
Compared with base and ens16 (step 3e). Blocked 4-fold CV, first 39 trials. Dev only.

Usage: .venv/Scripts/python -I scripts/s1_individual.py
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hemishift.data.liu2024_raw import EXCLUDED, load_raw
from hemishift.data.splits import blocked_folds
from hemishift.eval.stats import chance_threshold, holm, paired_comparison
from hemishift.paths import CONFIG_DIR, RESULTS_DIR
from hemishift.preprocess.quality import aperiodic_peak

_spec = importlib.util.spec_from_file_location("s1_classical",
                                               Path(__file__).with_name("s1_classical.py"))
classical = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(classical)

OUT = RESULTS_DIR / "summary" / "s1_individual_dev.csv"


def rhythm_peak(raw):
    """Peak above 1/f at C3/C4/Pz over the whole recording (label-free), or NaN."""
    raw = raw.copy().load_data(verbose="error").filter(1.0, 40.0, verbose="error")
    psd = raw.compute_psd(fmin=1.0, fmax=40.0, n_fft=int(4 * raw.info["sfreq"]),
                          picks=["C3", "C4", "Pz"], verbose="error")
    return aperiodic_peak(psd.freqs, psd.get_data().mean(axis=0))


def load(subject):
    raw = load_raw(subject)
    peak = rhythm_peak(raw)
    iband = (peak - 2.0, peak + 2.0) if np.isfinite(peak) else classical.BANDS["mu"]
    epochs, labels = {}, None
    windows = dict(classical.WINDOWS, **{"long 0.5-6": (0.5, 6.0)})
    for name, window in windows.items():
        X, y, _ = classical.prepare_epochs(raw, classical.Preprocessing(
            template=classical.EXTENDED_TEMPLATE, window=window))
        epochs[name] = X
        labels = y if name == "MI 0.5-4" else labels
    n = min(len(X) for X in epochs.values())
    data = {(w, b): classical.covariances(epochs[w][:n], band)
            for w in classical.WINDOWS for b, band in classical.BANDS.items()}
    for w in classical.WINDOWS:
        data[(w, "imu")] = classical.covariances(epochs[w][:n], iband)
    data[("long 0.5-6", "8-30")] = classical.covariances(epochs["long 0.5-6"][:n])
    return data, (np.asarray(labels[:n]) == "right_hand").astype(int), peak


def variants(data, y, train, test):
    p = lambda c: classical.candidate_proba(data, y, c, train, test)  # noqa: E731
    individual = [(w, tuple("imu" if b == "mu" else b for b in bands))
                  for w, bands in classical.CANDIDATES]
    return {
        "base": p(("MI 0.5-4", ("8-30",))),
        "iband": p(("MI 0.5-4", ("imu",))),
        "long": p(("long 0.5-6", ("8-30",))),
        "ens16": np.mean([p(c) for c in classical.CANDIDATES], axis=0),
        "ens16-i": np.mean([p(c) for c in individual], axis=0),
    }


def main():
    split = json.loads((CONFIG_DIR / "splits" / "liu2024_locked_test.json").read_text())
    rows = []
    for subject in split["dev"]:
        if subject in EXCLUDED:
            continue
        data, y, peak = load(subject)
        hits = {}
        for train, test in blocked_folds(len(y), 4):
            for name, prob in variants(data, y, train, test).items():
                hits[name] = hits.get(name, 0) + int(((prob > 0.5).astype(int) == y[test]).sum())
        rows.append({"subject": subject, "peak_hz": peak, **{k: 100 * v / len(y) for k, v in hits.items()}})
        print(f"sub {subject:2d} (peak {peak:4.1f} Hz): " + " ".join(
            f"{k} {v:.0f}" for k, v in rows[-1].items() if k not in ("subject", "peak_hz")))
    table = pd.DataFrame(rows)
    table.to_csv(OUT, index=False)
    threshold = 100 * chance_threshold(39)
    print(f"\n{len(table)} dev patients; rhythm peak found in {table['peak_hz'].notna().sum()}")
    names = ["iband", "long", "ens16", "ens16-i"]
    results = {n: paired_comparison(table[n] / 100, table["base"] / 100) for n in names}
    adjusted = dict(zip(names, holm([results[n]["p"] for n in names])))
    print(f"base     mean {table['base'].mean():5.1f} % | >= chance {(table['base'] >= threshold).sum()}")
    for n in names:
        r = results[n]
        print(f"{n:8s} mean {table[n].mean():5.1f} % | >= chance {(table[n] >= threshold).sum():2d} | vs base "
              f"median {100 * r['median_diff']:+.1f}, p = {r['p']:.3f} (Holm {adjusted[n]:.3f}), "
              f"helped {(table[n] > table['base']).sum()}, harmed {(table[n] < table['base']).sum()}")
    r = paired_comparison(table["ens16-i"] / 100, table["ens16"] / 100)
    print(f"ens16-i vs ens16: median {100 * r['median_diff']:+.1f}, p = {r['p']:.3f}")
    print(f"results in {OUT}")


if __name__ == "__main__":
    main()
