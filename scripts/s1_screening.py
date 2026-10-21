"""S1 step 3i: can calibration data alone tell who can use a left/right BCI? (dev patients)

Protocol P3 (chronological): the first 20 trials (10 per class) calibrate and the remaining 19
test. Two candidate decoders from step 3e: base (MI 0.5-4 s, 8-30 Hz TS + LR) and ens16.
Each is scored by leave-one-pair-out CV on the 20 calibration trials. The SafeTransfer gate
(`hemishift.eval.safetransfer.choose`) then picks one, or abstains when the best calibration
score does not beat chance (Bonferroni over the two). Reported on the 19 test trials:
  - accuracy of always-base, always-ens16, and the gate (abstaining patients fall back to base)
  - accuracy of the patients the gate passes vs those it abstains on
  - Spearman correlation between calibration-CV and test accuracy
  - exploratory: test accuracy against NIHSS, days since stroke and the rhythm peak
Results: results/summary/s1_screening_dev.csv.

Usage: .venv/Scripts/python -I scripts/s1_screening.py
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from hemishift.data.liu2024_raw import EXCLUDED, participants
from hemishift.eval.safetransfer import choose
from hemishift.paths import CONFIG_DIR, RESULTS_DIR


def _script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


classical = _script("s1_classical")
locked = _script("s1_locked_test")  # its load(): the 12 covariance sets, first 39 trials

N_CAL = 20
OUT = RESULTS_DIR / "summary" / "s1_screening_dev.csv"


def decoders(data, y, fit_on, predict_on):
    p = lambda c: classical.candidate_proba(data, y, c, fit_on, predict_on)  # noqa: E731
    return {"base": p(("MI 0.5-4", ("8-30",))),
            "ens16": np.mean([p(c) for c in classical.CANDIDATES], axis=0)}


def main():
    split = json.loads((CONFIG_DIR / "splits" / "liu2024_locked_test.json").read_text())
    peaks = RESULTS_DIR / "summary" / "s1_individual_dev.csv"  # rhythm peaks, step 3g
    audit = pd.read_csv(peaks).set_index("subject") if peaks.exists() else pd.DataFrame()
    clinical = participants()
    rows = []
    for subject in split["dev"]:
        if subject in EXCLUDED:
            continue
        data, y = locked.load(subject)
        cal, test = np.arange(N_CAL), np.arange(N_CAL, len(y))
        correct = {"base": 0, "ens16": 0}
        for pair in range(N_CAL // 2):
            held = cal[2 * pair:2 * pair + 2]
            for name, p in decoders(data, y, np.setdiff1d(cal, held), held).items():
                correct[name] += int(((p > 0.5).astype(int) == y[held]).sum())
        choice, _ = choose(correct, N_CAL)
        final = {name: 100 * np.mean((p > 0.5).astype(int) == y[test])
                 for name, p in decoders(data, y, cal, test).items()}
        rows.append({"subject": subject, **final, "cal_base": 100 * correct["base"] / N_CAL,
                     "cal_ens16": 100 * correct["ens16"] / N_CAL, "choice": choice or "abstain",
                     "gate": final[choice or "base"], "nihss": clinical.loc[subject, "NIHSS"],
                     "days": clinical.loc[subject, "Duration"],
                     "peak_hz": audit.loc[subject, "peak_hz"] if subject in audit.index else np.nan})
        print(f"sub {subject:2d}: calibration base {correct['base']}/20 ens16 {correct['ens16']}/20 -> "
              f"{choice or 'ABSTAIN':7s} | test base {final['base']:.0f} ens16 {final['ens16']:.0f}")
    table = pd.DataFrame(rows)
    table.to_csv(OUT, index=False)

    n_test = len(y) - N_CAL
    print(f"\n{len(table)} dev patients; calibrate on {N_CAL}, test on the next {n_test} trials")
    for name in ("base", "ens16", "gate"):
        print(f"  always-{name:5s} mean {table[name].mean():5.1f} %")
    passed = table["choice"] != "abstain"
    print(f"  gate passed {passed.sum()}/{len(table)}: test accuracy {table.loc[passed, 'gate'].mean():.1f} % "
          f"vs abstained {table.loc[~passed, 'gate'].mean():.1f} %")
    if passed.any() and (~passed).any():
        u = stats.mannwhitneyu(table.loc[passed, "gate"], table.loc[~passed, "gate"], alternative="greater")
        print(f"  passed > abstained: Mann-Whitney p = {u.pvalue:.3f}")
    cal_best = table[["cal_base", "cal_ens16"]].max(axis=1)
    rho = stats.spearmanr(cal_best, table["gate"])
    print(f"  calibration-CV vs test accuracy: Spearman rho {rho.statistic:.2f} (p {rho.pvalue:.3f})")
    for column in ("nihss", "days", "peak_hz"):
        ok = table[column].notna()
        r = stats.spearmanr(table.loc[ok, column], table.loc[ok, "ens16"])
        print(f"  ens16 test accuracy vs {column}: rho {r.statistic:+.2f} (p {r.pvalue:.3f}, n {ok.sum()})")
    print(f"results in {OUT}")


if __name__ == "__main__":
    main()
