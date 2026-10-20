import mne
import numpy as np

from hemishift.preprocess.quality import aperiodic_peak, channel_quality

SFREQ = 250.0
# S1's real 29-channel layout, so that neighbours are as dense as on a real recording.
NAMES = ["Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FCz", "FC3", "FC4", "FT7", "FT8", "Cz",
         "C3", "C4", "T7", "T8", "CP3", "CP4", "TP7", "TP8", "Pz", "P3", "P4", "P7", "P8", "Oz",
         "O1", "O2"]
RIGHT = [ch for ch in NAMES if ch[-1].isdigit() and int(ch[-1]) % 2 == 0]


def _raw(rng):
    """Correlated 'brain' signal on all channels plus a little independent noise."""
    t = np.arange(int(60 * SFREQ)) / SFREQ
    shared = np.cumsum(rng.normal(size=t.size)) * 0.05 + np.sin(2 * np.pi * 10 * t)
    data = np.array([shared + 0.2 * rng.normal(size=t.size) for _ in NAMES]) * 10e-6
    raw = mne.io.RawArray(data, mne.create_info(NAMES, SFREQ, "eeg"), verbose="error")
    raw.set_montage("standard_1020", verbose="error")
    return raw, t


def test_clean_recording_has_nothing_to_interpolate():
    raw, _ = _raw(np.random.default_rng(0))
    assert not channel_quality(raw.filter(1, 40, verbose="error"))["interpolate"].any()


def _with_fault(channels, make_fault, seed):
    rng = np.random.default_rng(seed)
    raw, t = _raw(rng)
    data = raw.get_data()
    for channel in [channels] if isinstance(channels, str) else channels:
        index = NAMES.index(channel)
        data[index] = make_fault(data[index], t, rng)
    raw = mne.io.RawArray(data, raw.info, verbose="error").filter(1, 40, verbose="error")
    return channel_quality(raw).set_index("channel")


def test_flat_channel_is_caught():
    table = _with_fault("F3", lambda x, t, rng: 1e-8 * rng.normal(size=t.size), seed=1)
    assert table.loc["F3", "flat"] and set(table.index[table["interpolate"]]) == {"F3"}


def test_noisy_channel_is_caught():
    table = _with_fault("C3", lambda x, t, rng: x + 40e-6 * np.sin(2 * np.pi * 35 * t), seed=2)
    assert table.loc["C3", "noisy"] and set(table.index[table["interpolate"]]) == {"C3"}


def test_decorrelated_channel_is_caught():
    table = _with_fault("P4", lambda x, t, rng: 10e-6 * rng.normal(size=t.size), seed=3)
    assert table.loc["P4", "decorrelated"] and set(table.index[table["interpolate"]]) == {"P4"}


def test_regional_slowing_is_flagged_but_not_interpolated():
    def slow(x, t, rng):  # a shared 2 Hz wave over one hemisphere, like lesion slowing
        return x + 60e-6 * np.sin(2 * np.pi * 2 * t)
    table = _with_fault(RIGHT, slow, seed=4)
    assert table.loc[["F4", "C4", "P4"], "slow"].all()
    assert not table["interpolate"].any()  # pathology is kept, not repaired


def test_aperiodic_peak_finds_a_real_peak_and_rejects_a_pure_slope():
    freqs = np.arange(1.0, 41.0, 0.5)
    background = 1.0 / freqs
    with_peak = background * (1 + 1.5 * np.exp(-0.5 * ((freqs - 8.5) / 1.0) ** 2))
    assert aperiodic_peak(freqs, with_peak) == 8.5
    assert np.isnan(aperiodic_peak(freqs, background))
