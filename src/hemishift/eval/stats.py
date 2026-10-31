"""Statistics used across the evaluation (PLAN.md §6).

All comparisons are made across patients: per-patient scores are too noisy to compare one
patient at a time (±7–12 points with 16–40 test trials, LITERATURE.md §5).
"""

import numpy as np
from scipy import stats


def chance_threshold(n_trials, alpha=0.05, n_classes=2):
    """Smallest accuracy that beats chance for one subject tested on `n_trials` trials.

    Exact one-sided binomial test: the smallest k / n_trials with P(X >= k) <= alpha under
    X ~ Binomial(n_trials, 1 / n_classes). Returns NaN when no accuracy can reach significance
    (too few trials). E.g. 16 trials -> 0.75, 40 -> 0.65, 100 -> 0.59.
    """
    p_chance = 1 / n_classes
    for k in range(n_trials + 1):
        if stats.binom.sf(k - 1, n_trials, p_chance) <= alpha:  # sf(k - 1) = P(X >= k)
            return k / n_trials
    return float("nan")


def holm(p_values):
    """Holm-Bonferroni adjusted p-values, returned in the input order."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.empty_like(p)
    running_max = 0.0
    for rank, index in enumerate(np.argsort(p)):
        running_max = max(running_max, (p.size - rank) * p[index])
        adjusted[index] = min(1.0, running_max)
    return adjusted


def clean_differences(diff, decimals=9):
    """Paired differences with floating-point noise removed (rounded to 1e-9).

    Accuracies are fractions of a few dozen trials, so the same true difference can come out
    as 4.285714285714285 for one patient and 4.285714285714292 for another, and a true zero
    (e.g. between two-session means) as +-1e-15. Unrounded, the Wilcoxon test ranks such ties
    apart and keeps the zeros, which shifts its p-value (S3 locked test: 0.035 instead of the
    correct 0.047; ACHIEVEMENTS.md, 2026-10-31).
    """
    return np.round(np.asarray(diff, dtype=float), decimals)


def signed_rank_p(diff, alternative="two-sided"):
    """Wilcoxon signed-rank p-value over patients, zero differences dropped."""
    diff = clean_differences(diff)
    nonzero = diff[diff != 0]
    return float(stats.wilcoxon(nonzero, alternative=alternative).pvalue) if nonzero.size else 1.0


def paired_comparison(scores_a, scores_b, n_boot=10_000, seed=0):
    """Compare two methods on the same patients (a - b).

    Returns the median difference with a bootstrap 95 % CI over patients, the two-sided
    Wilcoxon signed-rank p-value, and the matched-pairs rank-biserial correlation (effect size,
    from -1 to 1; positive means a is better).
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("both methods must be scored on the same patients")
    diff = clean_differences(a - b)
    rng = np.random.default_rng(seed)
    boot_medians = np.median(rng.choice(diff, size=(n_boot, diff.size), replace=True), axis=1)
    ci_low, ci_high = np.percentile(boot_medians, [2.5, 97.5])

    nonzero = diff[diff != 0]
    if nonzero.size:
        p_value = signed_rank_p(nonzero)
        ranks = stats.rankdata(np.abs(nonzero))
        rank_biserial = (ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum()
    else:
        p_value, rank_biserial = 1.0, 0.0
    return {
        "n": int(diff.size),
        "median_diff": float(np.median(diff)),
        "ci95": (float(ci_low), float(ci_high)),
        "p": float(p_value),
        "rank_biserial": float(rank_biserial),
    }


def negative_transfer_rate(transfer_scores, baseline_scores, margin=0.02):
    """Share of patients whose transferred decoder is at least `margin` below their baseline.

    The 2-point margin is pre-registered (ACHIEVEMENTS.md) so noise-level dips don't count.
    """
    diff = np.asarray(transfer_scores, dtype=float) - np.asarray(baseline_scores, dtype=float)
    return float(np.mean(diff <= -margin + 1e-12))
