"""Leakage and overfitting audit of the combination study (ACHIEVEMENTS.md, 2026-11-01).

  pairs    every decoder rerun with pair-wise updates (block = 2) on S3 dev and on the S3 locked
           patients (raw and ICA-cleaned), compared with the per-trial runs; the six locked-test
           hypotheses re-tested
  permute  labels shuffled within every patient-session (donors and earlier sessions included);
           every decoder must fall to chance. S3 dev (pair-wise updates) and S4
  chrono   S4 with strictly chronological calibration: every trial up to the one completing
           k per class calibrates
  verify   split hashes, cached patient sets, every confirmatory p-value recomputed from the
           saved per-patient results, and patients possibly shared between S3 and S4
Results: results/summary/audit_<check>_<data>.csv.

Usage: .venv/Scripts/python -I scripts/audit_combo.py pairs|permute|chrono|verify [--jobs 7]
"""

import argparse
import hashlib
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.exceptions import ConvergenceWarning

from hemishift.eval.stats import clean_differences, signed_rank_p
from hemishift.paths import CONFIG_DIR, HARMONISED_DIR, RAW_DIR, RESULTS_DIR


def _script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


combo = _script("combo_explore")
locked = _script("s3_locked_combo")
SUMMARY = RESULTS_DIR / "summary"
KEY = ("own", "pooled_bank", "pooled_unsup_bank", "own_sup", "pooled_sup_bank", "bma", "prevoth_bank28",
       "prevoth_unsup_bank28", "prevzero_unsup_bank", "prevoth_sup_bank28")


def both_rounds(donors, jobs, tag=None, **options):
    """Rounds 1 and 2 under the audit's options. tag: each patient-session is saved under
    results/combo/checkpoints/audit_<tag>_r<round> as it finishes, so a rerun resumes."""
    saved = {r: None if tag is None else RESULTS_DIR / "combo" / "checkpoints" / f"audit_{tag}_r{r}"
             for r in (1, 2)}
    first, _ = combo.run(donors, jobs, combo.run_patient, checkpoint=saved[1], **options)
    second, _ = combo.run(donors, jobs, combo.run_patient_r2, checkpoint=saved[2], **options)
    return first.merge(second, on=["session", "subject", "k"], validate="one_to_one")


def compare(name, before, after):
    """Mean accuracy of the key decoders, per-trial vs the audited run, k = 5."""
    print(f"\n--- {name}: mean accuracy, k = 5 (original -> audited) ---")
    for setting in ("pre", "later"):
        a, b = combo.per_setting(before, setting, 5), combo.per_setting(after, setting, 5)
        cells = []
        for n in KEY:
            if n in a and n in b and a[n].notna().all() and b[n].notna().all():
                cells.append(f"{n} {a[n].mean():.1f}->{b[n].mean():.1f}")
        print(f"  {setting:5s} " + " | ".join(cells))


def pairs(jobs):
    runs = {
        "s3dev": (("s3", {}), combo.load_tables("s3")),
        "locked": (("s3", {"targets": "s3locked"}), pd.read_csv(SUMMARY / "s3_locked_combo.csv")),
        "locked_clean": (("s3clean", {"targets": "s3cleanlocked"}),
                         pd.read_csv(SUMMARY / "s3_locked_combo_clean.csv")),
    }
    for name, ((donors, options), before) in runs.items():
        out = SUMMARY / f"audit_pairs_{name}.csv"
        if out.exists():
            after = pd.read_csv(out)
        else:
            after = both_rounds(donors, jobs, block=2, **options)
            after.to_csv(out, index=False)
        compare(f"pair-wise updates, {name}", before, after)
        if name.startswith("locked"):
            locked.report(after, f"{name}, pair-wise updates")


def permute(jobs):
    for cohort, block in (("s3", 2), ("s4", 1)):
        out = SUMMARY / f"audit_permute_{cohort}.csv"
        if out.exists():
            table = pd.read_csv(out)
        else:
            table = both_rounds(cohort, jobs, block=block, permute=0)
            table.to_csv(out, index=False)
        print(f"\n--- label permutation, {cohort} (block {block}): every decoder should be near 50 % ---")
        for setting in ("pre", "later"):
            for k in combo.KS:
                t = combo.per_setting(table, setting, k)
                names = combo.candidates(t) + ["own"]
                means = {n: t[n].mean() for n in names}
                worst = max(means, key=means.get)
                p = signed_rank_p(t[worst] - 50.0)
                print(f"  {setting:5s} k={k:2d}: mean over decoders {np.mean(list(means.values())):.1f} %, "
                      f"range {min(means.values()):.1f}-{max(means.values()):.1f} % (highest: {worst}, "
                      f"two-sided p vs 50 % = {p:.3f}); bma {means.get('bma', np.nan):.1f} %")


def chrono(jobs):
    out = SUMMARY / "audit_chrono_s4.csv"
    if out.exists():
        after = pd.read_csv(out)
    else:
        after = both_rounds("s4", jobs, tag="chrono_s4", chrono=True)
        after.to_csv(out, index=False)
    compare("strictly chronological calibration, S4", combo.load_tables("s4"), after)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:10]


def verify():
    print("--- split files ---")
    for name in ("wang2026sf_locked_test.json", "liu2024_locked_test.json"):
        split = json.loads((CONFIG_DIR / "splits" / name).read_text())
        overlap = set(split["dev"]) & set(split["test"])
        print(f"  {name}: sha256 {file_hash(CONFIG_DIR / 'splits' / name)}, dev {len(split['dev'])}, "
              f"test {len(split['test'])}, overlap {sorted(overlap) or 'none'}")
    split = json.loads((CONFIG_DIR / "splits" / "wang2026sf_locked_test.json").read_text())
    print("--- cached S3 data hold only the intended patients ---")
    for path in sorted(HARMONISED_DIR.glob("combo_s3*.npz")):
        ids = {int(k.split("|")[1]) for k in np.load(path).files if k.startswith("y|")}
        group = "test" if "locked" in path.name else "dev"
        print(f"  {path.name}: {len(ids)} patients, all in {group}: {ids <= set(split[group])}, "
              f"any in the other group: {bool(ids & set(split['dev' if group == 'test' else 'test']))}")
    print("--- confirmatory p-values recomputed (one-sided exact Wilcoxon, ties rounded) ---")
    checks = []
    s3 = pd.read_csv(SUMMARY / "s3_locked_test.csv")
    s3 = s3[s3.k == 5]
    checks.append(("S3 pooling locked test", s3["pooled"] - s3["own"], 0.047))
    s4 = pd.read_csv(SUMMARY / "s4_replication.csv")
    pool = s4[(s4.analysis == "pooling") & (s4.k == 5)]
    later = s4[(s4.analysis == "cross_session") & (s4.k == 5)].groupby("subject").mean(numeric_only=True)
    checks += [("S4 H-R1", pool["pooled"] - pool["own"], 0.019),
               ("S4 H-R2 (raw p)", pool["pooled_fb"] - pool["pooled"], None),
               ("S4 H-R3 (raw p)", later["previous"] - later["own"], None),
               ("S4 H-R4 (raw p, margin 2)", later["previous_zero"] - later["own"] + 2, None)]
    for version in ("", "_clean"):
        table = pd.read_csv(SUMMARY / f"s3_locked_combo{version}.csv")
        for name, decoder, baseline, setting in locked.HYPOTHESES:
            t = combo.per_setting(table, setting, 5)
            checks.append((f"{name}{version}", t[decoder] - t[baseline], None))
    for name, diff, logged in checks:
        d = clean_differences(diff)
        d = d[d != 0]
        exact = sps.wilcoxon(d, alternative="greater", method="exact").pvalue if len(d) <= 25 else np.nan
        mine = signed_rank_p(diff, "greater")
        note = "" if logged is None else f" (logged {logged})"
        print(f"  {name:28s} n {len(d):2d}, helped {(d > 0).sum():2d}, harmed {(d < 0).sum():2d}: "
              f"p {mine:.4f}, exact {exact:.4f}{note}")
    print("--- patients possibly in both S3 and S4 (same institution) ---")
    s3p = pd.read_csv(RAW_DIR / "wang2026sf" / "participants.tsv", sep="\t")
    s4p = pd.read_csv(RAW_DIR / "liu2025_meta" / "participants.tsv", sep="\t")
    print(f"  S3 columns: {list(s3p.columns)}")
    print(f"  S4 columns: {list(s4p.columns)}")
    return s3p, s4p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("check", choices=("pairs", "permute", "chrono", "verify"))
    parser.add_argument("--jobs", type=int, default=7)
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    if args.check == "verify":
        verify()
    else:
        {"pairs": pairs, "permute": permute, "chrono": chrono}[args.check](args.jobs)


if __name__ == "__main__":
    main()
