import numpy as np
import pytest

from hemishift.augment.lesionsim import (
    MU_HZ,
    attenuate_erd,
    band_split,
    ipsilesional,
    jitter_erd,
    shift_to_contralesional,
    vary_erd,
)

SFREQ = 128.0
CHANNELS = ["C3", "Cz", "C4"]  # C3 is ipsilesional after harmonisation
N = int(3.5 * SFREQ)
T = np.arange(N) / SFREQ
REST = np.array([2.0, 2.0, 2.0])  # mu amplitude at rest, per channel


def _trial(c3_amplitude, c4_amplitude=2.0):
    """One trial of pure 10 Hz rhythm; amplitudes may be arrays over time (ERD shapes)."""
    carrier = np.sin(2 * np.pi * 10.0 * T)
    return np.stack([c3_amplitude * carrier, 2.0 * carrier, c4_amplitude * carrier])[None]


def _mu_amplitude(X, channel):
    middle = slice(int(0.75 * SFREQ), N - int(0.75 * SFREQ))  # away from the edges
    return band_split(X, SFREQ, MU_HZ)[1][0, channel, middle]


def test_ipsilesional_channels_are_the_left_hemisphere():
    assert ipsilesional(["F3", "Fz", "F4", "C3", "Cz", "C4"]) == [0, 3]


def test_band_split_reconstruction_is_exact():
    X = np.random.default_rng(0).normal(size=(2, 3, N))
    component, amplitude, phase = band_split(X, SFREQ, MU_HZ)
    np.testing.assert_allclose(amplitude * np.cos(phase), component, atol=1e-10)


@pytest.mark.parametrize(
    "op, kwargs",
    [(attenuate_erd, {"alpha": 1.0}), (shift_to_contralesional, {"beta": 0.0}),
     (jitter_erd, {"shift_s": 0.0})],
)
def test_neutral_settings_return_the_input(op, kwargs):
    X = np.random.default_rng(1).normal(size=(2, 3, N))
    np.testing.assert_allclose(op(X, SFREQ, CHANNELS, REST, **kwargs), X, atol=1e-10)


def test_attenuate_erd_moves_amplitude_toward_rest():
    X = _trial(c3_amplitude=1.0)  # ERD: 1 instead of the rest level 2
    half = attenuate_erd(X, SFREQ, CHANNELS, REST, alpha=0.5)
    gone = attenuate_erd(X, SFREQ, CHANNELS, REST, alpha=0.0)
    assert _mu_amplitude(half, 0).mean() == pytest.approx(1.5, rel=0.03)
    assert _mu_amplitude(gone, 0).mean() == pytest.approx(2.0, rel=0.03)
    np.testing.assert_allclose(gone[0, 1:], X[0, 1:], atol=1e-10)  # Cz, C4 untouched


def test_attenuate_erd_accepts_one_gain_per_trial():
    X = np.concatenate([_trial(c3_amplitude=1.0)] * 2)  # two identical ERD trials
    out = attenuate_erd(X, SFREQ, CHANNELS, REST, alpha=np.array([1.0, 0.0]))
    np.testing.assert_allclose(out[0], X[0], atol=1e-10)  # gain 1: unchanged
    middle = slice(int(0.75 * SFREQ), N - int(0.75 * SFREQ))
    second = band_split(out[1:], SFREQ, MU_HZ)[1][0, 0, middle]
    assert second.mean() == pytest.approx(2.0, rel=0.03)  # gain 0: ERD removed


def test_vary_erd_widens_trial_to_trial_spread():
    rng = np.random.default_rng(0)
    X = np.concatenate([_trial(c3_amplitude=1.0)] * 40)
    middle = slice(int(0.75 * SFREQ), N - int(0.75 * SFREQ))

    def spread(data):
        power = band_split(data, SFREQ, MU_HZ)[1][:, 0, middle].mean(axis=1) ** 2
        return np.std(np.log(power))

    varied = vary_erd(X, SFREQ, CHANNELS, REST, spread=0.5, rng=rng, band=MU_HZ)
    assert spread(X) < 1e-6 < 0.1 < spread(varied)
    np.testing.assert_allclose(vary_erd(X, SFREQ, CHANNELS, REST, spread=0.0, rng=rng, band=MU_HZ),
                               X, atol=1e-10)


def test_shift_to_contralesional_copies_relative_erd_to_the_mirror():
    X = _trial(c3_amplitude=1.0, c4_amplitude=2.0)  # C3 has -50 % ERD, C4 none
    out = shift_to_contralesional(X, SFREQ, CHANNELS, REST, beta=1.0)
    assert _mu_amplitude(out, 2).mean() == pytest.approx(1.0, rel=0.03)  # C4 now -50 %
    np.testing.assert_allclose(out[0, :2], X[0, :2], atol=1e-10)  # C3, Cz untouched


def test_jitter_erd_delays_the_onset():
    step = np.where(T >= 1.0, 1.0, 2.0)  # ERD begins 1.0 s into the window
    out = jitter_erd(_trial(c3_amplitude=step), SFREQ, CHANNELS, REST, shift_s=0.5)
    amplitude = band_split(out, SFREQ, MU_HZ)[1][0, 0]
    assert amplitude[int(1.25 * SFREQ)] == pytest.approx(2.0, rel=0.08)  # not yet desynchronised
    assert amplitude[int(2.25 * SFREQ)] == pytest.approx(1.0, rel=0.08)  # delayed ERD has arrived
