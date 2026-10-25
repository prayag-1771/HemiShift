"""S3 step 5: EEG foundation-model features (CBraMod, frozen) on Task B, dev patients.

Completes the C4 arm on the second cohort (S1: step 3h, no gain). Protocol P3 as in S3 steps
1-4: the first k (rest, MI) pairs calibrate, the rest test; k = 5, 10. CBraMod embeddings of
each epoch (0.5-3.5 s, i.e. three 1 s patches at 200 Hz, units of 100 µV) feed a shrinkage-LDA
classifier trained on the calibration epochs only.
  fm-mean   embeddings averaged over channels and patches (200 features)
  fm-chan   averaged over patches (14 x 200 features)
Compared with own TS + LR.

Usage: .venv/Scripts/python -I scripts/s3_foundation.py
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly

from hemishift.eval.stats import holm, paired_comparison
from hemishift.paths import RESULTS_DIR, device


def _script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s3 = _script("s3_transfer")
fm = _script("s1_foundation")
PATCHES = 3
OUT = RESULTS_DIR / "summary" / "s3_foundation_dev.csv"


@torch.no_grad()
def embed(encoder, X):
    """(n, channels, 200) CBraMod embeddings of 128 Hz µV epochs, first PATCHES seconds."""
    x200 = resample_poly(X, 25, 16, axis=-1)[..., :200 * PATCHES] / 100.0
    out = []
    for start in range(0, len(x200), 16):
        batch = torch.as_tensor(x200[start:start + 16], dtype=torch.float32, device=device())
        out.append(encoder(batch).mean(dim=2).cpu().numpy())
    return np.concatenate(out)


def main():
    encoder = fm.load_encoder()
    patients = s3.load_patients()
    rows = []
    for subject, (X, y) in patients.items():
        E = embed(encoder, X)
        features = {"fm-mean": E.mean(axis=1), "fm-chan": E.reshape(len(E), -1)}
        for k in s3.KS:
            cal, test = s3.calibration_split(len(y), k)
            row = {"subject": subject, "k": k,
                   "own": 100 * np.mean((s3.own_proba(X, y, cal, test) > 0.5) == y[test])}
            for name, F in features.items():
                p = fm.lda().fit(F[cal], y[cal]).predict_proba(F[test])[:, 1]
                row[name] = 100 * np.mean((p > 0.5) == y[test])
            rows.append(row)
        print(f"sub {subject:2d}: " + " | ".join(
            f"k={r['k']} own {r['own']:.0f} fm-mean {r['fm-mean']:.0f} fm-chan {r['fm-chan']:.0f}"
            for r in rows[-len(s3.KS):]))
    table = pd.DataFrame(rows)
    table.to_csv(OUT, index=False)
    names = ["fm-mean", "fm-chan"]
    for k in s3.KS:
        t = table[table["k"] == k]
        results = {n: paired_comparison(t[n] / 100, t["own"] / 100) for n in names}
        adjusted = dict(zip(names, holm([results[n]["p"] for n in names])))
        print(f"\nk = {k} ({len(t)} dev patients): own {t['own'].mean():.1f} %")
        for n in names:
            r = results[n]
            print(f"  {n:8s} mean {t[n].mean():5.1f} % | vs own median {100 * r['median_diff']:+.1f}, "
                  f"p = {r['p']:.3f} (Holm {adjusted[n]:.3f}), helped {(t[n] > t['own']).sum()}, "
                  f"harmed {(t[n] < t['own']).sum()}")
    print(f"results in {OUT}")


if __name__ == "__main__":
    main()
