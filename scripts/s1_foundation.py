"""S1 step 3h: EEG foundation-model features (C4 arm), linear probe, dev patients.

CBraMod (Wang et al., ICLR 2025), pretrained on the large TUEG clinical corpus, loaded through
braindecode ("braindecode/cbramod-pretrained"). The encoder is frozen. Its patch embeddings
feed a shrinkage-LDA classifier trained per patient: with 30 training trials, fine-tuning
4.9 M parameters would only overfit.
Input as CBraMod expects it: 200 Hz, units of 100 µV, 1 s patches. The window is MI 0.5-4.5 s,
4 patches; the 40 trials all fit inside the recording.
  fm-mean   embeddings averaged over channels and patches (200 features)
  fm-chan   averaged over patches only (14 channels x 200 = 2800 features)
  fm+erd    mean of the fm-chan and base probabilities
Blocked 4-fold CV, compared with the base TS + LR decoder (MI 0.5-4 s, 8-30 Hz).

Usage: .venv/Scripts/python -I scripts/s1_foundation.py
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hemishift.data.channels import EXTENDED_TEMPLATE
from hemishift.data.liu2024_raw import EXCLUDED, load_raw
from hemishift.data.splits import blocked_folds
from hemishift.eval.stats import chance_threshold, holm, paired_comparison
from hemishift.paths import CONFIG_DIR, RESULTS_DIR, device
from hemishift.preprocess.pipeline import Preprocessing, prepare_epochs

_spec = importlib.util.spec_from_file_location("s1_classical",
                                               Path(__file__).with_name("s1_classical.py"))
classical = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(classical)

FM_WINDOW = (0.5, 4.5)
OUT = RESULTS_DIR / "summary" / "s1_foundation_dev.csv"


def load_encoder():
    from braindecode.models import CBraMod

    encoder = CBraMod.from_pretrained("braindecode/cbramod-pretrained", return_encoder_output=True)
    return encoder.to(device()).eval()


@torch.no_grad()
def embed(encoder, X):
    """(n, channels, 200) patch-averaged CBraMod embeddings of 128 Hz µV epochs."""
    x200 = resample_poly(X, 25, 16, axis=-1) / 100.0  # 128 -> 200 Hz; µV -> 100 µV units
    out = []
    for start in range(0, len(x200), 16):
        batch = torch.as_tensor(x200[start:start + 16], dtype=torch.float32, device=device())
        out.append(encoder(batch).mean(dim=2).cpu().numpy())  # mean over patches
    return np.concatenate(out)


def lda():
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))


def main():
    encoder = load_encoder()
    split = json.loads((CONFIG_DIR / "splits" / "liu2024_locked_test.json").read_text())
    rows = []
    for subject in split["dev"]:
        if subject in EXCLUDED:
            continue
        raw = load_raw(subject)
        X_mi, labels, _ = prepare_epochs(raw, Preprocessing(template=EXTENDED_TEMPLATE))
        X_fm, _, _ = prepare_epochs(raw, Preprocessing(template=EXTENDED_TEMPLATE, window=FM_WINDOW))
        y = (np.asarray(labels) == "right_hand").astype(int)
        C = classical.covariances(X_mi)
        E = embed(encoder, X_fm)
        features = {"fm-mean": E.mean(axis=1), "fm-chan": E.reshape(len(E), -1)}
        hits = {}
        for train, test in blocked_folds(len(y), 4):
            p = {"base": classical.ts_proba(C, y, train, test, classical.lr())}
            for name, F in features.items():
                p[name] = lda().fit(F[train], y[train]).predict_proba(F[test])[:, 1]
            p["fm+erd"] = (p["fm-chan"] + p["base"]) / 2
            for name, prob in p.items():
                hits[name] = hits.get(name, 0) + int(((prob > 0.5).astype(int) == y[test]).sum())
        rows.append({"subject": subject, **{k: 100 * v / len(y) for k, v in hits.items()}})
        print(f"sub {subject:2d}: " + " ".join(f"{k} {v:.0f}" for k, v in rows[-1].items() if k != "subject"))
    table = pd.DataFrame(rows)
    table.to_csv(OUT, index=False)
    threshold = 100 * chance_threshold(40)
    names = ["fm-mean", "fm-chan", "fm+erd"]
    results = {n: paired_comparison(table[n] / 100, table["base"] / 100) for n in names}
    adjusted = dict(zip(names, holm([results[n]["p"] for n in names])))
    print(f"\n{len(table)} dev patients; base mean {table['base'].mean():.1f} %, "
          f">= chance {(table['base'] >= threshold).sum()}")
    for n in names:
        r = results[n]
        print(f"{n:8s} mean {table[n].mean():5.1f} % | >= chance {(table[n] >= threshold).sum():2d} | vs base "
              f"median {100 * r['median_diff']:+.1f}, p = {r['p']:.3f} (Holm {adjusted[n]:.3f}), "
              f"helped {(table[n] > table['base']).sum()}, harmed {(table[n] < table['base']).sum()}")
    print(f"results in {OUT}")


if __name__ == "__main__":
    main()
