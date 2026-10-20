"""How decodable is each cohort for a patient's own decoder? (exploratory, R0 only)

Per patient, tangent-space + logistic regression trained on the first k trials per class (in
recording order) and tested on the rest, the same R0 as premise_check.py:
  Task A  affected-hand vs unaffected-hand MI (left vs right, lesion normalised to the left)
  Task B  affected-hand MI vs rest, with rest cut from a quiet window before each trial's
          cue (REST_WINDOWS), so every rest epoch is matched in time to a real trial
Reports mean accuracy and the share of patients above their exact binomial chance threshold.

Usage: .venv/Scripts/python -I scripts/decodability.py hefmi_ich [--k 10]
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from hemishift.data.loaders import load_moabb_epochs, moabb_cohort
from hemishift.data.splits import calibration_split
from hemishift.eval.stats import chance_threshold
from hemishift.paths import RESULTS_DIR

# Reuse premise_check.py's covariances, re-centring and model, so R0 is identical in both.
_spec = importlib.util.spec_from_file_location("premise_check",
                                               Path(__file__).with_name("premise_check.py"))
premise = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(premise)

# Seconds relative to MI onset; 3.5 s long, inside a quiet period before the cue.
REST_WINDOWS = {
    "hefmi_ich": (-8.0, -4.5),  # 10 s pre-cue rest in each 37 s stored epoch (cue at -2 s)
}


def score(X, y, k):
    """Own-trials R0 accuracy and test size, or None if a class has too few trials."""
    try:
        calib, test = calibration_split(y, k, mode="first")
    except ValueError:
        return None
    reference = premise.recenter(premise.covariances(X), calib)
    model = premise.model().fit(reference[calib], y[calib])
    return 100 * np.mean(model.predict(reference[test]) == y[test]), test.size


def interleave(first, second):
    """Recording order for paired epochs, where first[i] happened just before second[i]."""
    n = len(first)
    X = np.concatenate([first, second])
    order = np.argsort(np.concatenate([np.arange(n)] * 2) * 2 + np.repeat([0, 1], n), kind="stable")
    return X[order], np.repeat([0, 1], n)[order]


def wang2026sf_rows(k, phases=("pre",)):
    """S3 (Tianjin): Task B = affected-hand MI vs the rest epoch just before it; Task SF =
    sixth-finger vs affected-hand MI, the paper's own task (~85 % with random 10-fold CV)."""
    from hemishift.data.loaders import load_wang2026sf_epochs, wang2026sf_affected_side

    rows = []
    for subject in sorted(wang2026sf_affected_side()):
        e = load_wang2026sf_epochs([subject], phases=phases)
        y = e.labels
        mi = np.flatnonzero(y == "affected_mi")
        paired = mi[(mi > 0) & (y[mi - 1] == "rest")]  # rest epoch right before each MI
        X_b, y_b = interleave(e.X[paired - 1], e.X[paired])
        task_sf = np.isin(y, ["affected_mi", "sixth_finger_mi"])
        row = {"subject": subject, "group": "stroke"}
        for task, (X, labels) in {"task_b": (X_b, y_b), "task_sf": (e.X[task_sf], y[task_sf])}.items():
            result = score(X, np.asarray(labels), k)
            if result:
                row[task], row[f"n_test_{task[5:]}"] = result
        rows.append(row)
        print(f"[wang2026sf] subject {subject}: " +
              "  ".join(f"{t} {row[t]:.0f}%" for t in ("task_b", "task_sf") if t in row))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Within-patient decodability per cohort.")
    parser.add_argument("name", help="cohort key: hefmi_ich, liu2024, physionet or wang2026sf")
    parser.add_argument("--k", type=int, default=10, help="calibration trials per class")
    args = parser.parse_args()

    if args.name == "wang2026sf":
        report(pd.DataFrame(wang2026sf_rows(args.k)), args)
        return
    dataset, affected_side, group = moabb_cohort(args.name)
    rows = []
    for subject in dataset.subject_list:
        side = affected_side(subject) if callable(affected_side) else affected_side.get(subject)
        if side is None:
            continue
        mi = load_moabb_epochs(dataset, {subject: side}, args.name, "?", subjects=[subject])
        row = {"subject": subject, "group": group(subject) if callable(group) else group}
        result = score(mi.X, mi.labels, args.k)
        if result:
            row["task_a"], row["n_test_a"] = result
        if args.name in REST_WINDOWS:
            rest = load_moabb_epochs(dataset, {subject: side}, args.name, "?", subjects=[subject],
                                     window=REST_WINDOWS[args.name])
            affected = mi.labels == "right_hand"
            # Interleave in recording order: each trial's rest window precedes its MI.
            X = np.concatenate([rest.X[affected], mi.X[affected]])
            y = np.array(["rest"] * affected.sum() + ["mi"] * affected.sum())
            order = np.argsort(np.concatenate([np.arange(affected.sum())] * 2) * 2
                               + np.repeat([0, 1], affected.sum()), kind="stable")
            result = score(X[order], y[order], args.k)
            if result:
                row["task_b"], row["n_test_b"] = result
        rows.append(row)
        print(f"[{args.name}] subject {subject} ({row['group']}): " +
              "  ".join(f"{t} {row[t]:.0f}%" for t in ("task_a", "task_b") if t in row))
    report(pd.DataFrame(rows), args)


def report(table, args):
    out = RESULTS_DIR / "summary" / f"decodability_{args.name}_k{args.k}.csv"
    table.to_csv(out, index=False)
    for task in [c for c in table.columns if c.startswith("task_")]:
        for grp, part in table.dropna(subset=[task]).groupby("group"):
            above = sum(acc >= 100 * chance_threshold(int(n))
                        for acc, n in zip(part[task], part[f"n_test_{task[5:]}"]))
            print(f"{task} {grp:8s}: mean {part[task].mean():.1f} %, median {part[task].median():.1f} %, "
                  f"above chance {above}/{len(part)}")
    print(f"results in {out}")


if __name__ == "__main__":
    main()
