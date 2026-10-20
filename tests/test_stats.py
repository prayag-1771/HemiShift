import math

import numpy as np
import pytest

from hemishift.eval.stats import chance_threshold, holm, negative_transfer_rate, paired_comparison


@pytest.mark.parametrize(
    "n_trials, expected",
    [(16, 12 / 16), (20, 15 / 20), (24, 17 / 24), (40, 26 / 40), (100, 59 / 100)],
)
def test_chance_threshold_matches_exact_binomial(n_trials, expected):
    assert chance_threshold(n_trials) == pytest.approx(expected)


def test_chance_threshold_is_nan_when_unreachable():
    assert math.isnan(chance_threshold(4))  # even 4/4 has p = 0.0625 > 0.05


def test_holm_known_example():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    np.testing.assert_allclose(holm([0.5, 0.9]), [1.0, 1.0])  # capped at 1


def test_paired_comparison_detects_a_consistent_gain():
    rng = np.random.default_rng(1)
    baseline = rng.uniform(0.5, 0.8, size=30)
    result = paired_comparison(baseline + 0.05, baseline)
    assert result["median_diff"] == pytest.approx(0.05)
    assert result["p"] < 0.001
    assert result["rank_biserial"] == pytest.approx(1.0)
    assert result["ci95"][0] == pytest.approx(0.05)


def test_paired_comparison_identical_scores():
    result = paired_comparison([0.6, 0.7], [0.6, 0.7])
    assert result["p"] == 1.0 and result["median_diff"] == 0.0


def test_negative_transfer_rate_uses_two_point_margin():
    transfer = [0.58, 0.59, 0.70, 0.60]
    baseline = [0.60, 0.60, 0.60, 0.60]
    # -2 points counts as harmed, -1 point is noise, gains and ties are not harmed
    assert negative_transfer_rate(transfer, baseline) == pytest.approx(0.25)
