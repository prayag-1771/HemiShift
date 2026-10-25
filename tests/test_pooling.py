import importlib.util
from pathlib import Path

import numpy as np

from hemishift.models.pooling import OwnTSLR, PooledTSLR, recentre_by_domain

ROOT = Path(__file__).resolve().parents[1]


def _covariances(rng, n, scale):
    A = rng.normal(size=(n, 6, 200))
    A[:, 2] *= scale[:, None]  # class 1 has less power on channel 2 (an "ERD")
    return np.einsum("nct,ndt->ncd", A, A) / 200


def _data(seed=0):
    rng = np.random.default_rng(seed)
    y = np.tile([0, 1], 20)
    C = _covariances(rng, 40, np.where(y == 1, 0.6, 1.0))
    donor_y = np.tile([0, 1], 150)
    donor_C = _covariances(rng, 300, np.where(donor_y == 1, 0.6, 1.0)) * 3.0  # another "device"
    domains = np.repeat(np.arange(3), 100)
    return C, y, recentre_by_domain(donor_C, domains), donor_y


def test_recentred_domains_have_identity_mean():
    from pyriemann.utils.mean import mean_riemann

    C, _, donor_C, _ = _data()
    for d in range(3):
        np.testing.assert_allclose(mean_riemann(donor_C[d * 100:(d + 1) * 100]), np.eye(6), atol=1e-6)


def test_pooled_decoder_matches_the_confirmed_experiment_code():
    spec = importlib.util.spec_from_file_location("s3_augment", ROOT / "scripts" / "s3_augment.py")
    augment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(augment)
    C, y, donor_C, donor_y = _data()
    expected = augment.pooled_proba(C[:10], y[:10], C[10:], donor_C, donor_y)
    got = PooledTSLR(donor_C, donor_y).fit(C[:10], y[:10]).predict_proba(C[10:])
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-9)


def test_own_decoder_matches_the_experiment_code():
    spec = importlib.util.spec_from_file_location("premise_check", ROOT / "scripts" / "premise_check.py")
    premise = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(premise)
    C, y, _, _ = _data()
    cal, test = np.arange(10), np.arange(10, 40)
    A = premise.recenter(C, cal)
    expected = premise.model().fit(A[cal], y[cal]).predict_proba(A[test])[:, 1]
    np.testing.assert_allclose(OwnTSLR().fit(C[cal], y[cal]).predict_proba(C[test]), expected,
                               rtol=1e-6, atol=1e-9)


def test_pooled_decoder_separates_the_classes():
    C, y, donor_C, donor_y = _data(seed=1)
    p = PooledTSLR(donor_C, donor_y).fit(C[:10], y[:10]).predict_proba(C[10:])
    assert np.mean((p > 0.5) == y[10:]) > 0.9
