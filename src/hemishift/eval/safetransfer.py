"""SafeTransfer gate (PLAN.md §8): pick a decoder per patient from calibration data alone, or
abstain.

Every candidate decoder is scored by cross-validation on the patient's calibration trials only.
The best candidate is used if its calibration accuracy beats chance by the exact binomial test
(Bonferroni over the candidates, since the best of several is picked); otherwise the gate
abstains ("collect more calibration first"), because wrong feedback can teach the patient the
wrong thing. Ties go to the candidate listed first, so list the patient's own decoder first:
transfer has to earn its place.
"""

from hemishift.eval.stats import chance_threshold


def choose(correct, n_trials, alpha=0.05):
    """(name, calibration accuracy) of the best candidate, or (None, accuracy) to abstain.

    `correct` maps candidate name -> calibration trials it got right under cross-validation,
    in priority order (own decoder first).
    """
    best = max(correct, key=correct.get)  # the first of equal maxima, i.e. the higher priority
    accuracy = correct[best] / n_trials
    if accuracy < chance_threshold(n_trials, alpha / len(correct)):
        return None, accuracy
    return best, accuracy
