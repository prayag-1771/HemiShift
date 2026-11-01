import numpy as np
from sklearn.linear_model import LogisticRegression

from hemishift.models.combo import (bma, crop_features, crops_static, donor_decoder, map_lr,
                                    map_proba, online, population_prior, stack, static,
                                    tangent_all)


def _spd(rng, n, c=4, shift=0.0):
    """n random SPD matrices; `shift` inflates the first channel's variance."""
    A = rng.standard_normal((n, c, 3 * c))
    C = A @ A.transpose(0, 2, 1) / (3 * c)
    C[:, 0, 0] += shift
    return C


def _patient(rng, n=40, c=4):
    y = np.tile([0, 1], n // 2)
    C = np.concatenate([_spd(rng, 1, c, 1.5 * label) for label in y])
    return C, y


def test_stack_gives_each_donor_group_the_patients_weight():
    T = np.zeros((6, 2))
    groups = [(np.zeros((30, 2)), np.zeros(30)), (np.zeros((12, 2)), np.ones(12))]
    _, Y, w = stack(T, np.zeros(6), groups)
    assert len(Y) == 48
    assert np.isclose(w[:6].sum(), 6) and np.isclose(w[6:36].sum(), 6) and np.isclose(w[36:].sum(), 6)


def test_map_lr_tends_to_lr_with_a_flat_prior_and_to_the_prior_mean_with_a_tight_one():
    rng = np.random.default_rng(0)
    T = rng.standard_normal((60, 3))
    y = (T[:, 0] + 0.5 * rng.standard_normal(60) > 0).astype(int)
    mean = np.array([0.3, -0.2, 0.1, 0.05])
    flat = map_lr(T, y, (mean, np.full(4, 1e6)))
    ref = LogisticRegression(C=1e6, max_iter=5000).fit(T, y)
    assert np.allclose(flat, np.append(ref.coef_[0], ref.intercept_[0]), atol=1e-2)
    tight = map_lr(T, y, (mean, np.full(4, 1e-8)))
    assert np.allclose(tight, mean, atol=1e-4)
    assert np.allclose(map_proba(tight, T), 1 / (1 + np.exp(-(T @ mean[:-1] + mean[-1]))))


def test_population_prior_is_the_mean_and_variance_of_the_other_patients():
    W = np.array([[1.0, 0.0], [3.0, 2.0], [2.0, 4.0]])
    mean, var = population_prior(W, floor=0.0)
    assert np.allclose(mean, [2.0, 2.0]) and np.allclose(var, [1.0, 4.0])


def test_online_first_trial_equals_the_static_decoder():
    rng = np.random.default_rng(1)
    C, y = _patient(rng)
    cov = {"a": C}
    cal, test = np.arange(10), np.arange(10, 40)
    p_static = static(cov, y, cal, test, ("a",))
    p_online = online(cov, y, cal, test, ("a",))
    assert np.isclose(p_online[0], p_static[0])
    zero = (tangent_all(_patient(rng)[0]), y)
    assert np.isclose(online(cov, y, cal, test, ("a",), zero=zero)[0],
                      static(cov, y, cal, test, ("a",), zero=zero)[0])


def test_online_uses_only_earlier_labels():
    rng = np.random.default_rng(2)
    C, y = _patient(rng)
    cov = {"a": C}
    cal, test = np.arange(10), np.arange(10, 40)
    p = online(cov, y, cal, test, ("a",))
    flipped = y.copy()
    flipped[25:] = 1 - flipped[25:]  # labels of trial 25 onward change; decisions up to 25 must not
    q = online(cov, flipped, cal, test, ("a",))
    assert np.allclose(p[:16], q[:16])  # test[15] is trial 25


def test_online_decisions_ignore_future_epochs():
    rng = np.random.default_rng(5)
    C, y = _patient(rng)
    cal, test = np.arange(10), np.arange(10, 40)
    D, y_d = _patient(rng, n=60)
    groups = [(tangent_all(D), y_d)]
    p = online({"a": C}, y, cal, test, ("a",), groups=groups)
    changed = C.copy()
    changed[25:] = _patient(rng)[0][25:]  # replace every epoch from trial 25 on
    q = online({"a": changed}, y, cal, test, ("a",), groups=groups)
    assert np.allclose(p[:15], q[:15])  # test[14] is trial 24
    unsup = online({"a": C}, y, cal, test, ("a",), labels="cal", groups=groups)
    unsup_changed = online({"a": changed}, y, cal, test, ("a",), labels="cal", groups=groups)
    assert np.allclose(unsup[:15], unsup_changed[:15])


def test_pairwise_updates_use_only_complete_earlier_pairs():
    rng = np.random.default_rng(6)
    C, y = _patient(rng)
    cal, test = np.arange(10), np.arange(10, 40)
    p = online({"a": C}, y, cal, test, ("a",), block=2)
    changed, flipped = C.copy(), y.copy()
    changed[20] = _patient(rng)[0][20]  # test[10], the first epoch of a pair
    flipped[20] = 1 - flipped[20]
    q = online({"a": changed}, flipped, cal, test, ("a",), block=2)
    assert np.allclose(p[:10], q[:10])  # earlier decisions never saw trial 20
    assert np.isclose(p[11], q[11])  # nor did its pair partner (test[11])
    assert not np.isclose(p[12], q[12])  # the next pair does
    assert np.allclose(online({"a": C}, y, cal, test, ("a",), block=1),
                       online({"a": C}, y, cal, test, ("a",)))


def test_stack_weights_and_fixed_donor_total():
    T, y = np.zeros((4, 2)), np.array([0, 1, 0, 1])
    _, _, w = stack(T, y, [(np.zeros((8, 2)), np.zeros(8))], weights=[0.25, 0.5, 1, 1])
    assert np.isclose(w[4:].sum(), 2.75)  # donors weigh as much as the weighted patient rows
    _, _, w = stack(T, y, [(np.zeros((8, 2)), np.zeros(8))], donor_total=10)
    assert np.isclose(w[4:].sum(), 10) and np.allclose(w[:4], 1)


def test_forgetting_with_a_huge_half_life_changes_nothing():
    rng = np.random.default_rng(7)
    C, y = _patient(rng)
    cal, test = np.arange(10), np.arange(10, 40)
    D, y_d = _patient(rng, n=60)
    groups = [(tangent_all(D), y_d)]
    base = online({"a": C}, y, cal, test, ("a",), groups=groups)
    slow = online({"a": C}, y, cal, test, ("a",), groups=groups, half_life=1e9)
    assert np.allclose(base, slow, atol=1e-4)


def test_bma_blocks_update_weights_only_between_blocks():
    y = np.tile([0, 1], 10)
    good = np.where(y == 1, 0.9, 0.1)
    bad = np.where(y == 1, 0.2, 0.8)
    out = bma([good, bad], y, block=2)
    assert np.isclose(out[0], out[0]) and np.isclose(out[1], (good[1] + bad[1]) / 2)
    flipped = y.copy()
    flipped[2] = 1 - flipped[2]
    assert np.isclose(bma([good, bad], flipped, block=2)[3], out[3])  # trial 2's label not used for trial 3


def test_one_full_window_crop_equals_the_pooled_static_decoder():
    rng = np.random.default_rng(3)
    C, y = _patient(rng)
    D, y_d = _patient(rng, n=60)
    cal, test = np.arange(10), np.arange(10, 40)
    donors = crop_features([D[:, None]])
    p_crops = crops_static([C[:, None]], y, cal, test, (donors, y_d))
    p_static = static({"a": C}, y, cal, test, ("a",), [(tangent_all(D), y_d)])
    assert np.allclose(p_crops, p_static, atol=1e-6)


def test_donor_decoder_weighs_groups_equally_and_one_group_is_plain_lr():
    rng = np.random.default_rng(4)
    T = rng.standard_normal((50, 3))
    y = (T[:, 0] > 0).astype(int)
    one = donor_decoder((T, y))
    ref = LogisticRegression(C=1.0, max_iter=1000).fit(T, y)
    assert np.allclose(one.coef_, ref.coef_, atol=1e-5)
    # two groups of 50 and 20 rows: each weighs as much as the average group (35 rows)
    two = donor_decoder([(T, y), (T[:20], y[:20])])
    weights = np.concatenate([np.full(50, 35 / 50), np.full(20, 35 / 20)])
    ref = LogisticRegression(C=1.0, max_iter=1000).fit(np.vstack([T, T[:20]]),
                                                       np.concatenate([y, y[:20]]),
                                                       sample_weight=weights)
    assert np.allclose(two.coef_, ref.coef_, atol=1e-5)


def test_bma_starts_from_the_average_and_moves_to_the_better_decoder():
    y = np.tile([0, 1], 20)
    good = np.where(y == 1, 0.9, 0.1)
    bad = np.full(len(y), 0.5)
    out = bma([good, bad], y)
    assert np.isclose(out[0], (good[0] + bad[0]) / 2)
    assert abs(out[-1] - good[-1]) < 1e-3
