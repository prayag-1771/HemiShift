"""Recompute every reported Wilcoxon p-value with floating-point ties handled (2026-10-31).

Accuracy differences that are equal in exact arithmetic (e.g. +1/70 of the test trials for
four patients) came out of floating point a few ulps apart, so scipy ranked them apart instead
of as ties; true zeros between two-session means survived as +-1e-15. hemishift.eval.stats
now rounds differences to 1e-9 before ranking (clean_differences, signed_rank_p). This script
recomputes, from the saved per-patient results, the p-values that were logged before the fix
and prints the logged value next to the corrected one. No decoder is re-run.

Usage: .venv/Scripts/python -I scripts/recheck_stats.py
"""

import numpy as np
import pandas as pd

from hemishift.eval.stats import holm, paired_comparison, signed_rank_p
from hemishift.paths import RESULTS_DIR

S = RESULTS_DIR / "summary"


def read(name):
    return pd.read_csv(S / f"{name}.csv")


def line(label, logged, corrected):
    flag = "" if f"{logged}" == f"{corrected:.{len(str(logged).split('.')[-1])}f}" else "   <- changed"
    print(f"  {label:58s} logged {logged:<7} corrected {corrected:.4f}{flag}")


def main():
    print("Confirmatory (one-sided, pre-registered):")
    t = read("s3_locked_test")
    for k, logged in ((5, 0.035), (10, 0.14)):
        x = t[t.k == k]
        line(f"S3 locked test, pooled > own, k = {k}", logged, signed_rank_p(x.pooled - x.own, "greater"))
    t = read("s1_locked_test")
    line("S1 locked test, ens16 > base", 0.083, signed_rank_p(t.ens16 - t.base, "greater"))

    print("Sensitivity (one-sided, S3 locked test on the cleaned release):")
    t = read("s3_locked_sensitivity")
    for k, logged in ((5, (0.2891, 0.0312, 0.0547)), (10, (0.2188, 0.0312, 0.0156))):
        x = t[t.k == k]
        for (a, b), old in zip((("pooled_830", "own_830"), ("pooled_bank", "own_830"),
                                ("pooled_bank", "pooled_830")), logged):
            line(f"k = {k}: {a} > {b}", old, signed_rank_p(x[a] - x[b], "greater"))

    print("Development values quoted in the paper (two-sided):")
    t = read("s3_augment_dev")
    for k in (5, 10):
        x = t[t.k == k]
        names = ["+healthy", "+lesionsim", "+matched", "+patients"]
        ps = [paired_comparison(x[n] / 100, x["own"] / 100)["p"] for n in names]
        adjusted = holm(ps)
        logged = {5: (0.004, 0.018), 10: (0.16, None)}[k]
        line(f"S3 dev pooling (+patients vs own), k = {k}", logged[0], ps[3])
        if logged[1] is not None:
            line(f"  Holm over the four donor types, k = {k}", logged[1], adjusted[3])
    t = read("s1_classical_dev")
    variants = [c for c in t.columns if c not in ("subject", "n_trials", "base")]
    ps = [paired_comparison(t[v] / 100, t["base"] / 100)["p"] for v in variants]
    line("S1 dev ens16 vs base", 0.011, ps[variants.index("ens16")])
    line("  Holm over the 11 variants", 0.12, holm(ps)[variants.index("ens16")])
    t = read("s1_drift_dev")
    line("S1 dev rebound block vs global re-centring", 0.0205,
         paired_comparison(t["rebound 4-6 s, block"] / 100, t["rebound 4-6 s, global"] / 100)["p"])
    t = read("s1_deep_dev")
    line("S1 dev healthy-pretrained + fine-tuned vs own", 0.26,
         paired_comparison(t.finetune / 100, t.tslr / 100)["p"])
    line("S1 dev healthy-pretrained zero-shot vs own", 0.017,
         paired_comparison(t.zeroshot / 100, t.tslr / 100)["p"])

    t = read("s3_theta_search_dev").set_index("subject")
    settings = list(t.columns)
    identity = "1.0/0.0/0.0"
    nested = np.empty(len(t))
    for held in np.array_split(np.arange(len(t)), 4):
        rest = np.setdiff1d(np.arange(len(t)), held)
        best = max(settings, key=lambda st: (t[st].to_numpy()[rest].mean(), st == identity))
        nested[held] = t[best].to_numpy()[held]
    line("LesionSim nested choice vs identity", 0.80,
         paired_comparison(nested / 100, t[identity].to_numpy() / 100)["p"])

    t = read("s3_safetransfer_dev")
    for k, logged in ((5, 0.033), (10, 0.001)):
        x = t[t.k == k]
        line(f"SafeTransfer gate vs always transfer, k = {k} (one-sided)", logged,
             signed_rank_p(x.gate - x.transfer, "greater"))

    t = read("s3_neuro_longitudinal")
    wide = t.pivot(index="subject", columns="phase", values="accuracy")
    ps = {}
    for a, b, logged in (("pre", "post", 0.18), ("pre", "follow", 0.038), ("post", "follow", None)):
        both = wide[[a, b]].dropna()
        ps[(a, b)] = signed_rank_p(both[b] - both[a])
        if logged is not None:
            line(f"S3 longitudinal accuracy, {b} - {a}", logged, ps[(a, b)])
    line("  Holm over the three phase pairs, follow - pre", 0.11, holm(list(ps.values()))[1])


if __name__ == "__main__":
    main()
