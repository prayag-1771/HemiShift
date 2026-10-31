"""S3 artifact check on the authors' cleaned release: does pooling survive ICA cleaning? (dev)

The confound check found that frontal 1-8 Hz activity (eye movements, blinks) separates rest
from imagery in S3 at ~77 %. The raw EOG channels cannot be regressed out (in most runs they
are flat or far outside the physiological range), so this check uses the dataset authors'
own preprocessed release instead: 43 channels, average reference, ICA, artifact segments cut.
The same decoders run on both versions, each with its own trials:
  motor14 8-30 Hz   the confirmed own and pooled decoders
  motor14 bank      S4's H-R2 decoder (mu, low beta, high beta) and fb4 (with theta), pooled
  frontal 1-8 Hz    Fpz Fp1 Fp2 AF3 AF4, the eye control
  posterior 8-30 Hz PO5 PO3 POz PO4 PO6, the visual-cortex control (no O row in the release)
Channels common to both versions (42, CPz left out: it is the raw recording's reference);
each group is re-referenced to its own average. Protocol P3, k = 5 and 10; donors are the
other dev patients of the same version. Results: results/summary/s3_processed_check_dev.csv.

Usage: .venv/Scripts/python -I scripts/s3_processed_check.py
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hemishift.data.channels import EXTENDED_TEMPLATE
from hemishift.data.loaders import load_wang2026sf_epochs
from hemishift.eval.stats import paired_comparison
from hemishift.paths import CONFIG_DIR, HARMONISED_DIR, RESULTS_DIR


def _script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = _script("s3_sweep")
COMMON42 = ("Fpz", "Fp1", "Fp2", "AF3", "AF4", "Fz", "F1", "F2", "F3", "F4", "F5", "F6", "FCz",
            "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "Cz", "C1", "C2", "C3", "C4", "C5", "C6",
            "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "Pz", "P3", "P4", "P5", "P6", "POz", "PO3",
            "PO4", "PO5", "PO6")
MU, B1, B2 = (8.0, 13.0), (13.0, 20.0), (20.0, 30.0)
VARIANTS = {  # name: (channels, bands)
    "motor14_830": (EXTENDED_TEMPLATE, ((8.0, 30.0),)),
    "motor14_bank": (EXTENDED_TEMPLATE, (MU, B1, B2)),
    "motor14_fb4": (EXTENDED_TEMPLATE, ((4.0, 8.0), MU, B1, B2)),
    "frontal_1-8": (("Fpz", "Fp1", "Fp2", "AF3", "AF4"), ((1.0, 8.0),)),
    "posterior_830": (("PO5", "PO3", "POz", "PO4", "PO6"), ((8.0, 30.0),)),
}
OUT = RESULTS_DIR / "summary" / "s3_processed_check_dev.csv"


def load(source):
    """Dev patients' pre-session (rest, MI) pairs on the 42 common channels (cached)."""
    path = HARMONISED_DIR / f"s3proc_dev_pre_{source}_t42.npz"
    split = json.loads((CONFIG_DIR / "splits" / "wang2026sf_locked_test.json").read_text())
    if path.exists():
        data = np.load(path)
        return {s: (data[f"X{s}"], data[f"y{s}"]) for s in split["dev"]}
    patients = {}
    for subject in split["dev"]:
        e = load_wang2026sf_epochs([subject], template=COMMON42, source=source)
        mi = np.flatnonzero(e.labels == "affected_mi")
        paired = mi[(mi > 0) & (e.labels[mi - 1] == "rest")]
        order = np.ravel(np.column_stack([paired - 1, paired]))
        patients[subject] = (e.X[order].astype(np.float32),
                             (e.labels[order] == "affected_mi").astype(int))
        print(f"  {source} sub {subject}: {order.size} epochs", flush=True)
    np.savez(path, **{f"X{s}": X for s, (X, _) in patients.items()},
             **{f"y{s}": y for s, (_, y) in patients.items()})
    return patients


def subset(X, channels):
    return sweep.common_average(X[:, [COMMON42.index(ch) for ch in channels]])


def evaluate(data, source):
    rows = []
    for name, (channels, bands) in VARIANTS.items():
        C = {s: [sweep.covariances(subset(X, channels), band) for band in bands]
             for s, (X, _) in data.items()}
        TF = {s: np.hstack([sweep.tangent(c, np.arange(len(c))) for c in C[s]]) for s in data}
        for s, (_, y) in data.items():
            others = [o for o in data if o != s]
            donor_T = np.vstack([TF[o] for o in others])
            donor_y = np.concatenate([data[o][1] for o in others])
            for k in sweep.KS:
                cal, test = np.arange(2 * k), np.arange(2 * k, len(y))
                T = np.hstack([sweep.tangent(c, cal) for c in C[s]])
                own = sweep.lr().fit(T[cal], y[cal]).predict_proba(T[test])[:, 1]
                pooled = sweep.lr().fit(
                    np.vstack([T[cal], donor_T]), np.concatenate([y[cal], donor_y]),
                    sample_weight=sweep.donor_weights(len(cal), len(donor_y), 1.0)
                ).predict_proba(T[test])[:, 1]
                rows.append({"source": source, "variant": name, "subject": s, "k": k,
                             "n_test": len(test),
                             "own": 100 * np.mean((own > 0.5) == y[test]),
                             "pooled": 100 * np.mean((pooled > 0.5) == y[test])})
    return rows


def main():
    data = {source: load(source) for source in ("raw", "processed")}
    print("epochs per patient, raw vs processed:",
          {s: (len(data["raw"][s][1]), len(data["processed"][s][1])) for s in data["raw"]})
    table = pd.DataFrame(evaluate(data["raw"], "raw") + evaluate(data["processed"], "processed"))
    table.to_csv(OUT, index=False)
    for k in sweep.KS:
        print(f"\nk = {k} (16 dev patients): own / pooled, raw -> authors' cleaned release")
        base = {src: table[(table["source"] == src) & (table["variant"] == "motor14_830")
                           & (table["k"] == k)].set_index("subject") for src in ("raw", "processed")}
        for name in VARIANTS:
            t = {src: table[(table["source"] == src) & (table["variant"] == name)
                            & (table["k"] == k)].set_index("subject") for src in ("raw", "processed")}
            p = t["processed"]
            gain = paired_comparison(p["pooled"] / 100, p["own"] / 100)
            line = (f"  {name:14s} {t['raw']['own'].mean():5.1f} / {t['raw']['pooled'].mean():5.1f} -> "
                    f"{p['own'].mean():5.1f} / {p['pooled'].mean():5.1f}  (cleaned: pooled - own "
                    f"median {100 * gain['median_diff']:+.1f}, p {gain['p']:.3f}, helped "
                    f"{(p['pooled'] > p['own']).sum()}, harmed {(p['pooled'] < p['own']).sum()})")
            if name in ("motor14_bank", "motor14_fb4"):
                vs = paired_comparison(p["pooled"] / 100, base["processed"]["pooled"] / 100)
                line += (f" | vs cleaned pooled 8-30: mean "
                         f"{(p['pooled'] - base['processed']['pooled']).mean():+.1f}, p {vs['p']:.3f}")
            print(line)
    print(f"results in {OUT}")


if __name__ == "__main__":
    main()
