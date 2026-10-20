import numpy as np
import pytest

from hemishift.data.erd import MU_HZ, band_power, erd_percent, laterality_index

SFREQ = 128.0


def _alpha(amplitude, seconds=4.0):
    t = np.arange(int(seconds * SFREQ)) / SFREQ
    return amplitude * np.sin(2 * np.pi * 10.0 * t)


def test_band_power_of_a_sinusoid_is_amplitude_squared():
    power = band_power(_alpha(2.0), SFREQ, MU_HZ)
    middle = power[128:-128]  # ignore filter edge effects
    assert middle.mean() == pytest.approx(4.0, rel=0.02)


def test_erd_of_halved_amplitude_is_minus_75_percent():
    rest = band_power(_alpha(2.0), SFREQ, MU_HZ)[128:-128].mean()
    task = band_power(_alpha(1.0), SFREQ, MU_HZ)[128:-128].mean()
    assert erd_percent(task, rest) == pytest.approx(-75.0, abs=1.0)


@pytest.mark.parametrize(
    "contra, ipsi, expected",
    [(-50.0, 0.0, 1.0), (0.0, -50.0, -1.0), (-30.0, -30.0, 0.0), (-30.0, 20.0, 1.0)],
)
def test_laterality_index(contra, ipsi, expected):
    assert laterality_index(contra, ipsi) == pytest.approx(expected)


def test_laterality_index_is_nan_without_desynchronisation():
    assert np.isnan(laterality_index(10.0, 5.0))
