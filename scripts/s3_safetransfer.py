"""S3 step 2: the SafeTransfer gate (C3, H2) on Task B with 5 or 10 calibration pairs, dev.

Candidates (PLAN.md §8): own (R0, TS + LR), transfer (R3: healthy + LesionSim pretrained
EEGNet, fine-tuned on the calibration epochs) and their ensemble (mean probability).
Per patient, each candidate is scored by leave-one-pair-out CV on the calibration pairs only.
`hemishift.eval.safetransfer.choose` then picks the best candidate (ties favour own), or
abstains when even the best does not beat chance (exact binomial, Bonferroni over the three).
An abstaining patient falls back to own for the accuracy figures, and is flagged.
Reported on the held-out pairs:
  - mean accuracy of always-own, always-transfer, always-ensemble and the gate
  - negative-transfer rate (>= 2 points below own) of always-transfer vs the gate (H2)
  - abstention rate, and the correlation between calibration-CV and held-out accuracy
Results: results/summary/s3_safetransfer_dev.csv.

Usage: .venv/Scripts/python -I scripts/s3_safetransfer.py
"""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from hemishift.eval.safetransfer import choose
from hemishift.eval.stats import paired_comparison
from hemishift.models.deep import euclidean_alignment, predict_proba, train
from hemishift.paths import RESULTS_DIR

_spec = importlib.util.spec_from_file_location("s3_transfer", Path(__file__).with_name("s3_transfer.py"))
s3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s3)

OUT = RESULTS_DIR / "summary" / "s3_safetransfer_dev.csv"
MODEL_PATH = RESULTS_DIR / "models" / "s3_lesionsim_eegnet.pt"


def pretrained_r3():
    """The healthy + LesionSim EEGNet of s3_transfer.py, cached on disk."""
    import torch

    from hemishift.models.deep import make_eegnet

    model = make_eegnet(len(s3.CHANNELS), s3.CROP)
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        return model
    X_h, y_h, subjects_h, mirrored, rest_amp = s3.load_healthy()
    copy_ = s3.lesionsim_copy(X_h, y_h, subjects_h, mirrored, rest_amp, seed=4)
    parts = s3.s1.align_with_original([X_h, copy_], subjects_h)
    model = s3.s1.pretrain(np.concatenate(parts), np.concatenate([y_h, y_h]))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    return model


def transfer_proba(model, X, y, fit_on, predict_on):
    """Fine-tune on `fit_on` (alignment from `fit_on` too) and predict `predict_on`."""
    Xa = euclidean_alignment(X, fit_on)
    tuned = train(copy.deepcopy(model), Xa[fit_on], y[fit_on], crop=s3.CROP, epochs=40, lr=5e-4,
                  batch=16)
    return predict_proba(tuned, Xa[predict_on], s3.CROP)[:, 1]


def candidates(model, X, y, fit_on, predict_on):
    own = s3.own_proba(X, y, fit_on, predict_on)
    transfer = transfer_proba(model, X, y, fit_on, predict_on)
    return {"own": own, "transfer": transfer, "ensemble": (own + transfer) / 2}


def calibration_scores(model, X, y, cal):
    """Leave-one-pair-out correct counts per candidate, on the calibration pairs only."""
    correct = {"own": 0, "transfer": 0, "ensemble": 0}
    for pair in range(len(cal) // 2):
        held = cal[2 * pair:2 * pair + 2]
        fit_on = np.setdiff1d(cal, held)
        for name, p in candidates(model, X, y, fit_on, held).items():
            correct[name] += int(((p > 0.5).astype(int) == y[held]).sum())
    return correct


def main():
    patients = s3.load_patients()
    model = pretrained_r3()
    rows = []
    for subject, (X, y) in patients.items():
        for k in s3.KS:
            cal, test = s3.calibration_split(len(y), k)
            correct = calibration_scores(model, X, y, cal)
            choice, cal_accuracy = choose(correct, len(cal))
            final = candidates(model, X, y, cal, test)
            accuracy = {name: 100 * np.mean((p > 0.5).astype(int) == y[test]) for name, p in final.items()}
            rows.append({"subject": subject, "k": k, **accuracy,
                         **{f"cal_{name}": 100 * c / len(cal) for name, c in correct.items()},
                         "choice": choice or "abstain",
                         "gate": accuracy[choice or "own"]})
            print(f"sub {subject:2d} k={k:2d}: own {accuracy['own']:.0f} transfer {accuracy['transfer']:.0f} "
                  f"ensemble {accuracy['ensemble']:.0f} | calibration {correct} -> {choice or 'ABSTAIN'} "
                  f"({accuracy[choice or 'own']:.0f})")
    table = pd.DataFrame(rows)
    table.to_csv(OUT, index=False)

    for k in s3.KS:
        t = table[table["k"] == k]
        print(f"\nk = {k} per class ({len(t)} dev patients, {80 - 2 * k} test epochs)")
        for name in ("own", "transfer", "ensemble", "gate"):
            line = f"  {name:9s} mean {t[name].mean():5.1f} %"
            if name != "own":
                r = paired_comparison(t[name] / 100, t["own"] / 100)
                line += (f" | vs own median {100 * r['median_diff']:+.1f}, p = {r['p']:.3f}, "
                         f"harmed (>= 2 pts) {(t[name] <= t['own'] - 2).sum()}")
            print(line)
        abstain = t["choice"] == "abstain"
        print(f"  choices: {t['choice'].value_counts().to_dict()}; abstained {abstain.sum()}/{len(t)}")
        if (~abstain).any():
            print(f"  selective accuracy (gate, non-abstained): {t.loc[~abstain, 'gate'].mean():.1f} %")
        cal_best = t[["cal_own", "cal_transfer", "cal_ensemble"]].max(axis=1)
        rho = stats.spearmanr(cal_best, t["gate"])
        print(f"  calibration-CV vs held-out accuracy: Spearman rho {rho.statistic:.2f} (p {rho.pvalue:.3f})")
    print(f"results in {OUT}")


if __name__ == "__main__":
    main()
