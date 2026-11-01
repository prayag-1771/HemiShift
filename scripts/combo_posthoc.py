"""Second round of the combination study (exploratory; ACHIEVEMENTS.md, 2026-11-01).

Combinations computed from the per-trial probabilities that combo_explore.py saved, so no
decoder is re-run:
  <decoder>+bias  unsupervised bias adaptation (Vidaurre et al. 2011): before each test trial
                  the decoder's logit is shifted by the median logit of the earlier test
                  trials, from the fifth test trial on, so that over time half of the trials
                  are called imagery (the paradigms are balanced). No labels.
  avg_static      mean probability of pooled_bank, pooled_bank28 and pooled_bank_crops
                  (later sessions: prevoth_bank and prevoth_bank28)
  avg_online      mean probability of pooled_sup_bank, pooled_sup_bank28 and prior_sup_bank
                  (later sessions: prevoth_sup_bank and prevoth_sup_bank28)
  bma_all         Bayesian model averaging over every decoder of the setting (rounds 1 and 2)
The candidates are fixed before this script is first run. Selection without hindsight is done
as in combo_explore.report, over the first round's decoders and these together.

Usage: .venv/Scripts/python -I scripts/combo_posthoc.py
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

from hemishift.paths import RESULTS_DIR


def _script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


combo = _script("combo_explore")
START = 4  # test trials seen before the bias is adapted
BIASED = ("pooled_bank", "pooled_bank28", "pooled_bank_crops", "prior_bank", "prevoth_bank",
          "prevoth_bank28", "prevzero_bank", "prevzero_unsup_bank", "pooled_unsup_bank",
          "pooled_unsup_bank28", "prevoth_unsup_bank", "prevoth_unsup_bank28", "prevothzero_bank28",
          "prevothzero_unsup_bank28")
AVERAGES = {"pre": {"avg_static": ("pooled_bank", "pooled_bank28", "pooled_bank_crops"),
                    "avg_online": ("pooled_sup_bank", "pooled_sup_bank28", "prior_sup_bank")},
            "later": {"avg_static": ("prevoth_bank", "prevoth_bank28"),
                      "avg_online": ("prevoth_sup_bank", "prevoth_sup_bank28")}}


def bias_adapted(p):
    """Logits shifted by the median logit of the earlier test trials (from trial START on)."""
    z = logit(np.clip(p, 1e-6, 1 - 1e-6))
    shift = np.array([np.median(z[:i]) if i >= START else 0.0 for i in range(len(z))])
    return 1 / (1 + np.exp(-(z - shift)))


def second_round(cohort):
    table = combo.load_tables(cohort)
    probs = dict(np.load(RESULTS_DIR / "combo" / f"{cohort}_probs.npz"))
    second = RESULTS_DIR / "combo" / f"{cohort}_probs_r2.npz"
    if second.exists():
        probs.update(np.load(second))
    rows = []
    for _, r in table.iterrows():
        key = f"{r['session']}|{r['subject']}|{r['k']}"
        y = probs[f"{key}|y"]
        p = {n: probs[f"{key}|{n}"] for n in table.columns if f"{key}|{n}" in probs}
        setting = "pre" if r["session"] == "pre" else "later"
        new = {f"{n}+bias": bias_adapted(p[n]) for n in BIASED if n in p}
        for name, members in AVERAGES[setting].items():
            new[name] = np.mean([p[m] for m in members], axis=0)
        new["bma_all"] = combo.bma([q for n, q in p.items() if n != "bma"], y)
        rows.append({**r.to_dict(), **{n: 100 * np.mean((q > 0.5).astype(int) == y)
                                       for n, q in new.items()}})
    return pd.DataFrame(rows)


def main():
    tables = {cohort: second_round(cohort) for cohort in ("s3", "s4")}
    for cohort, table in tables.items():
        table.to_csv(RESULTS_DIR / "summary" / f"combo_{cohort}_posthoc.csv", index=False)
    combo.FAMILIES["no labels after calibration"] += tuple(f"{n}+bias" for n in BIASED) + ("avg_static",)
    combo.FAMILIES["cue labels during the session"] += ("avg_online", "bma_all")
    combo.report(tables)


if __name__ == "__main__":
    main()
