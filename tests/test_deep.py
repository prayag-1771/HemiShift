import numpy as np
import pytest

from hemishift.models.deep import euclidean_alignment, make_eegnet, predict_proba, random_crops, train


def test_euclidean_alignment_whitens_the_reference_set():
    rng = np.random.default_rng(0)
    mixing = rng.normal(size=(4, 4))
    X = np.einsum("cd,ndt->nct", mixing, rng.normal(size=(30, 4, 200)))
    aligned = euclidean_alignment(X, np.arange(30))
    mean_cov = np.einsum("nct,ndt->ncd", aligned, aligned).mean(0) / 200
    np.testing.assert_allclose(mean_cov, np.eye(4), atol=1e-4)


def test_random_crops_have_the_requested_length():
    X = np.zeros((5, 3, 100), dtype=np.float32)
    assert random_crops(X, 64, np.random.default_rng(0)).shape == (5, 3, 64)


def test_eegnet_learns_an_easy_separable_task():
    rng = np.random.default_rng(1)
    n, chans, times = 120, 4, 128
    y = rng.integers(0, 2, size=n)
    t = np.arange(times) / 128
    X = rng.normal(size=(n, chans, times)).astype(np.float32)
    # Class 1 carries a strong 10 Hz rhythm on channel 0. Power, unlike the sign of a sine,
    # survives random cropping, which shifts the phase.
    X[:, 0] += y[:, None] * 3.0 * np.sin(2 * np.pi * 10 * t)
    model = train(make_eegnet(chans, 96), X[:80], y[:80], crop=96, epochs=30, lr=3e-3, batch=16)
    accuracy = np.mean(predict_proba(model, X[80:], crop=96).argmax(1) == y[80:])
    assert accuracy > 0.8
