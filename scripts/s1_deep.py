"""S1 step 3c: does deep learning with healthy pretraining help left-vs-right? (dev patients)

1. Pretrain EEGNet on PhysioNet's 109 healthy subjects (left/right MI, 14 motor channels,
   Euclidean alignment per subject, random 2 s crops). 11 subjects are held out to report
   cross-subject accuracy as a sanity check.
2. For each development patient (lesion normalised to the left), blocked 4-fold CV
   (10 consecutive trials per fold, all 40 scored). Alignment uses the training folds only.
     tslr      own tangent-space + LR decoder (the reference, as in s1_drift.py)
     scratch   EEGNet trained on the patient's 30 training trials only
     zeroshot  the healthy-pretrained EEGNet, no patient training
     finetune  the healthy-pretrained EEGNet fine-tuned on the 30 training trials
Exploratory; paired Wilcoxon against tslr.

Usage: .venv/Scripts/python -I scripts/s1_deep.py
"""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hemishift.data.channels import EXTENDED_TEMPLATE
from hemishift.data.harmonise import normalise_lesion_side
from hemishift.data.liu2024_raw import EXCLUDED, load_raw, participants
from hemishift.data.loaders import load_moabb_epochs, moabb_cohort
from hemishift.data.splits import blocked_folds
from hemishift.eval.stats import chance_threshold, paired_comparison
from hemishift.models.deep import euclidean_alignment, make_eegnet, predict_proba, train
from hemishift.paths import CONFIG_DIR, RESULTS_DIR
from hemishift.preprocess.pipeline import Preprocessing, prepare_epochs

_spec = importlib.util.spec_from_file_location("premise_check",
                                               Path(__file__).with_name("premise_check.py"))
premise = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(premise)

CROP = 256  # 2 s at 128 Hz
CLASS_ID = {"left_hand": 0, "right_hand": 1}  # right_hand = affected hand after normalisation


def aligned_by_subject(X, subjects):
    out = np.empty_like(X)
    for s in np.unique(subjects):
        rows = np.flatnonzero(subjects == s)
        out[rows] = euclidean_alignment(X[rows], np.arange(rows.size))
    return out


def pretrain():
    dataset, side, group = moabb_cohort("physionet")
    healthy = load_moabb_epochs(dataset, side, "physionet", group, template=EXTENDED_TEMPLATE)
    subjects = healthy.meta["patient"].to_numpy()
    X = aligned_by_subject(healthy.X, subjects)
    y = np.array([CLASS_ID[label] for label in healthy.labels])
    held_out = np.isin(subjects, np.unique(subjects)[::10])  # every 10th subject
    model = train(make_eegnet(len(EXTENDED_TEMPLATE), CROP), X[~held_out], y[~held_out],
                  crop=CROP, epochs=60, lr=1e-3, batch=64)
    accuracy = np.mean(predict_proba(model, X[held_out], CROP).argmax(1) == y[held_out])
    print(f"pretrained on {(~held_out).sum()} healthy trials; held-out healthy subjects: "
          f"{100 * accuracy:.1f} % ({held_out.sum()} trials)")
    return model


def score_patient(X, y, pretrained):
    C = premise.covariances(X)
    correct = {"tslr": 0, "scratch": 0, "zeroshot": 0, "finetune": 0}
    for fold, (tr, te) in enumerate(blocked_folds(len(y), 4)):
        reference = premise.recenter(C, tr)
        tslr = premise.model().fit(reference[tr], y[tr])
        correct["tslr"] += int((tslr.predict(reference[te]) == y[te]).sum())

        Xa = euclidean_alignment(X, tr)
        correct["zeroshot"] += int((predict_proba(pretrained, Xa[te], CROP).argmax(1) == y[te]).sum())
        tuned = train(copy.deepcopy(pretrained), Xa[tr], y[tr], crop=CROP, epochs=40, lr=5e-4,
                      batch=16, seed=fold)
        correct["finetune"] += int((predict_proba(tuned, Xa[te], CROP).argmax(1) == y[te]).sum())
        scratch = train(make_eegnet(X.shape[1], CROP, seed=fold), Xa[tr], y[tr], crop=CROP,
                        epochs=100, lr=1e-3, batch=16, seed=fold)
        correct["scratch"] += int((predict_proba(scratch, Xa[te], CROP).argmax(1) == y[te]).sum())
    return {k: 100 * v / len(y) for k, v in correct.items()}


def main():
    pretrained = pretrain()
    split = json.loads((CONFIG_DIR / "splits" / "liu2024_locked_test.json").read_text())
    clinical = participants()
    rows = []
    for subject in split["dev"]:
        if subject in EXCLUDED:
            continue
        X, labels, _ = prepare_epochs(load_raw(subject), Preprocessing(template=EXTENDED_TEMPLATE))
        side = "L" if clinical.loc[subject, "ParalysisSide"] == "left" else "R"
        X, labels = normalise_lesion_side(X, labels, EXTENDED_TEMPLATE, side)
        y = np.array([CLASS_ID[label] for label in labels])
        row = {"subject": subject, **score_patient(X.astype(np.float32), y, pretrained)}
        rows.append(row)
        print(f"sub {subject:2d}: " + " | ".join(f"{k} {v:.0f}" for k, v in row.items() if k != "subject"))
    table = pd.DataFrame(rows)
    out = RESULTS_DIR / "summary" / "s1_deep_dev.csv"
    table.to_csv(out, index=False)

    threshold = 100 * chance_threshold(40)
    print(f"\n{len(table)} dev patients, blocked 4-fold CV (chance threshold {threshold:.0f} %)")
    print(f"tslr      mean {table['tslr'].mean():5.1f} % | above chance {(table['tslr'] >= threshold).sum()}")
    for method in ("scratch", "zeroshot", "finetune"):
        r = paired_comparison(table[method] / 100, table["tslr"] / 100)
        print(f"{method:9s} mean {table[method].mean():5.1f} % | above chance {(table[method] >= threshold).sum()} "
              f"| vs tslr median {100 * r['median_diff']:+.1f} pts, p = {r['p']:.3f}, "
              f"helped {(table[method] > table['tslr']).sum()}, harmed {(table[method] < table['tslr']).sum()}")
    print(f"results in {out}")


if __name__ == "__main__":
    main()
