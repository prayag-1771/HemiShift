"""Pre-registered confirmation of the combination study on the S3 locked test patients
(ACHIEVEMENTS.md, 2026-11-01). Runs once.

The combination decoders (scripts/combo_explore.py, rounds 1 and 2) were developed on the 16
S3 dev patients and the 27 S4 patients only. None of them has been run on the 8 locked-test
patients. Those patients were used before in two declared pre-session analyses (the pooling
confirmation, the ICA sensitivity check) and, descriptively, in s3_neuro (own TS + LR,
blocked CV, all 24 patients).

Setting: the 8 locked patients' pre, post and follow-up sessions; donors are the 16 dev
patients' same session; a returning patient's earlier sessions are its own. Protocol P3: the
first k (rest, MI) pairs calibrate, k = 5 primary and 10 descriptive. Every decoder of rounds 1
and 2 runs unchanged; only the hypotheses below are tested.

Hypotheses: one-sided Wilcoxon signed-rank over patients, k = 5; later sessions use each
patient's mean over post and follow-up. Fixed sequence at alpha = 0.05: each is tested only if
every earlier one was confirmed.
  H-C1  bma > own, pre session
  H-C2  bma > own, later sessions
  H-C3  prevoth_unsup_bank28 > own, later sessions (no labels after calibration)
  H-C4  prevzero_unsup_bank > own, later sessions (no labels at all in the later session)
  H-C5  bma > own_sup, pre session (beyond the known co-adaptive decoder)
Results: results/summary/s3_locked_combo.csv.

Usage: .venv/Scripts/python -I scripts/s3_locked_combo.py [--dry-run] [--stats-only]
  --dry-run decodes dev patients 2 and 3 with the other 14 dev patients as donors (a check of
  the code path that loads no locked-test data).
"""

import argparse
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from hemishift.eval.stats import clean_differences, signed_rank_p
from hemishift.paths import RESULTS_DIR


def _script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


combo = _script("combo_explore")
ALPHA = 0.05
HYPOTHESES = (  # (name, decoder, baseline, setting)
    ("H-C1", "bma", "own", "pre"),
    ("H-C2", "bma", "own", "later"),
    ("H-C3", "prevoth_unsup_bank28", "own", "later"),
    ("H-C4", "prevzero_unsup_bank", "own", "later"),
    ("H-C5", "bma", "own_sup", "pre"),
)
OUT = RESULTS_DIR / "summary" / "s3_locked_combo.csv"


def decode(jobs, targets):
    first, _ = combo.run("s3", jobs, combo.run_patient, targets)
    second, _ = combo.run("s3", jobs, combo.run_patient_r2, targets)
    return first.merge(second, on=["session", "subject", "k"], validate="one_to_one")


def describe(t, decoder, baseline):
    diff = clean_differences(t[decoder] - t[baseline])
    return (f"{decoder} {t[decoder].mean():.1f} % vs {baseline} {t[baseline].mean():.1f} % "
            f"(mean diff {diff.mean():+.1f}, median {np.median(diff):+.1f}, helped {(diff > 0).sum()}, "
            f"harmed {(diff < 0).sum()})")


def report(table):
    print("\n=== pre-registered hypotheses (k = 5; one-sided Wilcoxon; fixed sequence) ===")
    going = True
    for name, decoder, baseline, setting in HYPOTHESES:
        t = combo.per_setting(table, setting, 5)
        p = signed_rank_p(t[decoder] - t[baseline], "greater")
        if going:
            verdict = "CONFIRMED" if p < ALPHA else "not confirmed"
            going = p < ALPHA
        else:
            verdict = "not tested (an earlier hypothesis was not confirmed); p is descriptive"
        print(f"  {name} ({setting}, {len(t)} patients): {describe(t, decoder, baseline)}; "
              f"p = {p:.4f} -> {verdict}")
    print("\n=== descriptive: every decoder vs own ===")
    for setting in ("pre", "later"):
        for k in combo.KS:
            t = combo.per_setting(table, setting, k)
            print(f"{setting}, k = {k} ({len(t)} patients), own {t['own'].mean():.1f} %")
            for n in combo.candidates(t):
                print(f"  {describe(t, n, 'own')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--jobs", type=int, default=7)
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    if args.stats_only:
        table = pd.read_csv(OUT)
    elif args.dry_run:
        table = decode(args.jobs, [2, 3])
    else:
        table = decode(args.jobs, "s3locked")
        table.to_csv(OUT, index=False)
    report(table)


if __name__ == "__main__":
    main()
