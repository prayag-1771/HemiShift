"""Week-2 gap diagnosis: how does stroke MI-EEG differ from healthy MI-EEG? (PLAN.md §4.0)

Per subject, for affected-hand imagery (after lesion-side normalisation that is the right hand,
with the lesion in the left hemisphere; healthy subjects use their right hand):
  erd_<band>_<ch>  ERD % at C3 (ipsilesional), Cz, C4 (contralesional), MI 0.5-4.0 s vs the
                   reference 1.5-0.5 s before MI onset
  li_<band>        laterality index C3 vs C4 (+1 = healthy contralateral pattern)
  var_<band>_C3    trial-to-trial spread of log power change at C3
  slow_asym        log ratio of (1-8 Hz)/(8-13 Hz) power, left over right, in the reference window
Writes results/summary/gap_<cohort>.csv. With two or more cohorts, compares the stroke and
healthy groups (Mann-Whitney U, Holm-corrected). All of this is exploratory, not pre-registered.

Usage: .venv/Scripts/python -I scripts/diagnose_gap.py liu2024 physionet [--subjects 10]
"""

import argparse

import numpy as np
import pandas as pd
from moabb.paradigms import MotorImagery
from scipy import stats

from hemishift.data.channels import MOTOR_TEMPLATE, hemisphere
from hemishift.data.erd import BETA_HZ, MU_HZ, band_power, erd_percent, laterality_index
from hemishift.data.harmonise import TARGET_SFREQ, harmonise
from hemishift.data.loaders import moabb_cohort
from hemishift.eval.stats import holm
from hemishift.paths import RESULTS_DIR

TMIN, TMAX = -2.0, 4.0
REFERENCE = (-1.5, -0.5)
TASK = (0.5, 4.0)
BANDS = {"mu": MU_HZ, "beta": BETA_HZ}


def window(times, bounds):
    return (times >= bounds[0]) & (times < bounds[1])


def subject_measures(X, times):
    """X: affected-hand trials (n_trials, template channels, n_times), lesion on the left."""
    ref, task = window(times, REFERENCE), window(times, TASK)
    c3, cz, c4 = (MOTOR_TEMPLATE.index(ch) for ch in ("C3", "Cz", "C4"))
    row = {}
    for name, band in BANDS.items():
        power = band_power(X, TARGET_SFREQ, band)
        ref_power, task_power = power[..., ref].mean(-1), power[..., task].mean(-1)
        erd = erd_percent(task_power.mean(0), ref_power.mean(0))  # trial-averaged, per channel
        row[f"erd_{name}_C3"], row[f"erd_{name}_Cz"], row[f"erd_{name}_C4"] = erd[[c3, cz, c4]]
        row[f"li_{name}"] = float(laterality_index(erd[c3], erd[c4]))
        row[f"var_{name}_C3"] = float(np.std(np.log(task_power[:, c3] / ref_power[:, c3])))
    slow = band_power(X, TARGET_SFREQ, (1.0, 8.0))[..., ref].mean((0, -1))
    alpha = band_power(X, TARGET_SFREQ, MU_HZ)[..., ref].mean((0, -1))
    left = [i for i, ch in enumerate(MOTOR_TEMPLATE) if hemisphere(ch) == "left"]
    right = [i for i, ch in enumerate(MOTOR_TEMPLATE) if hemisphere(ch) == "right"]
    row["slow_asym"] = float(np.log((slow[left] / alpha[left]).mean() /
                                    (slow[right] / alpha[right]).mean()))
    return row


def diagnose(name, n_subjects):
    dataset, affected_side, group = moabb_cohort(name)
    subjects = dataset.subject_list[:n_subjects] if n_subjects else dataset.subject_list
    paradigm = MotorImagery(n_classes=2, events=["left_hand", "right_hand"], fmin=1, fmax=40,
                            tmin=TMIN, tmax=TMAX, resample=TARGET_SFREQ)
    rows = []
    for subject in subjects:  # one subject at a time keeps memory flat
        side = affected_side(subject) if callable(affected_side) else affected_side.get(subject)
        if side is None:
            print(f"[{name}] subject {subject}: no defined affected side, skipped")
            continue
        epochs, labels, _ = paradigm.get_data(dataset, subjects=[subject], return_epochs=True)
        X, labels = harmonise(epochs.get_data(copy=False) * 1e6, epochs.ch_names, labels, side)
        affected = np.asarray(labels) == "right_hand"
        times = epochs.times + TMIN - epochs.times[0]
        rows.append({"cohort": name, "group": group(subject) if callable(group) else group,
                     "subject": subject, "affected_side": side, "n_trials": int(affected.sum()),
                     **subject_measures(X[affected], times)})
        print(f"[{name}] subject {subject}: {rows[-1]['n_trials']} affected-hand trials")
    table = pd.DataFrame(rows)
    out = RESULTS_DIR / "summary" / f"gap_{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    return table


def compare(tables, tag):
    data = pd.concat(tables, ignore_index=True)
    measures = [c for c in data.columns if c.startswith(("erd_", "li_", "var_", "slow_"))]
    stroke, healthy = data[data["group"] == "stroke"], data[data["group"] == "healthy"]
    rows = []
    for m in measures:
        a, b = stroke[m].dropna(), healthy[m].dropna()
        test = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append({"measure": m, "stroke_median": a.median(), "healthy_median": b.median(),
                     "rank_biserial": 1 - 2 * test.statistic / (a.size * b.size), "p": test.pvalue})
    result = pd.DataFrame(rows)
    result["p_holm"] = holm(result["p"])
    print(result.round(3).to_string(index=False))
    result.to_csv(RESULTS_DIR / "summary" / f"gap_comparison_{tag}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Stroke-vs-healthy MI-EEG gap diagnosis.")
    parser.add_argument("names", nargs="+", help="MOABB cohort keys, e.g. liu2024 physionet")
    parser.add_argument("--subjects", type=int, default=None, help="first N subjects per cohort")
    args = parser.parse_args()
    tables = [diagnose(name, args.subjects) for name in args.names]
    if pd.concat(tables)["group"].nunique() > 1:  # across cohorts, or within one (HEFMI-ICH)
        compare(tables, "_".join(args.names))


if __name__ == "__main__":
    main()
