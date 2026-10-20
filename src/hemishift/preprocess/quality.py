"""Channel quality and rhythm-peak estimates that separate broken electrodes from pathology.

A stroke can make a channel over the lesion look abnormal, with more slow-wave power or a
different spectrum, and that is real brain activity to keep. Technical faults look different:
flat signals, broadband muscle or electrode noise, or a channel that no longer correlates with
its neighbours. Only those are marked for interpolation.
"""

import numpy as np
import pandas as pd
from scipy.signal import welch


def robust_z(values):
    values = np.asarray(values, dtype=float)
    mad = np.median(np.abs(values - np.median(values))) * 1.4826
    return (values - np.median(values)) / (mad if mad > 0 else 1.0)


def neighbour_correlation(data, positions, n_neighbours=4):
    """Mean absolute correlation of each channel with its nearest neighbours (by position)."""
    corr = np.abs(np.corrcoef(data))
    distance = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    np.fill_diagonal(distance, np.inf)
    nearest = np.argsort(distance, axis=1)[:, :n_neighbours]
    return np.array([corr[i, nearest[i]].mean() for i in range(len(data))])


def channel_quality(raw, threshold=3.5, min_neighbour_r=0.4):
    """One row per EEG channel: metrics, flags and whether it should be interpolated.

    Expects `raw` with EEG positions set and band-limited to about 1-40 Hz for the
    correlation metric. Relative criteria (robust z) get absolute floors, because robust z
    explodes when clean channels are nearly identical (tiny MAD), as in the PREP pipeline:
      flat          standard deviation below 0.5 microvolts
      noisy         30-40 Hz share of 1-40 Hz power: robust z > threshold and > 2 x median
      decorrelated  mean |r| with the 4 nearest neighbours < min_neighbour_r
      slow          1-6 Hz share: robust z > threshold and > 1.5 x median. This flags
                    pathology and is not repaired; lesion slowing is regional, so neighbours
                    stay correlated
    interpolate = flat | noisy | decorrelated.
    """
    picks = raw.copy().pick("eeg")
    data = picks.get_data()
    positions = np.array([ch["loc"][:3] for ch in picks.info["chs"]])
    freqs, psd = welch(data, fs=picks.info["sfreq"], nperseg=int(2 * picks.info["sfreq"]))
    total = psd[:, (freqs >= 1) & (freqs < 40)].sum(1)
    table = pd.DataFrame({
        "channel": picks.ch_names,
        "std_uv": data.std(1) * 1e6,
        "hf_share": psd[:, (freqs >= 30) & (freqs < 40)].sum(1) / total,
        "slow_share": psd[:, (freqs >= 1) & (freqs < 6)].sum(1) / total,
        "neighbour_r": neighbour_correlation(data, positions),
    })
    table["flat"] = table["std_uv"] < 0.5
    table["noisy"] = ((robust_z(table["hf_share"]) > threshold)
                      & (table["hf_share"] > 2 * table["hf_share"].median()))
    table["decorrelated"] = table["neighbour_r"] < min_neighbour_r
    table["slow"] = ((robust_z(table["slow_share"]) > threshold)
                     & (table["slow_share"] > 1.5 * table["slow_share"].median()))
    table["interpolate"] = table["flat"] | table["noisy"] | table["decorrelated"]
    return table


def aperiodic_peak(freqs, power, search=(6.0, 14.0), fit=(2.0, 40.0), min_height=0.1):
    """Rhythm peak above the 1/f background, or NaN if there is none.

    Fits log10(power) = a - b * log10(f) over `fit`, leaving out `search`, then takes the
    largest residual inside `search`. A peak must rise `min_height` log10 units (~26 %)
    above the background; otherwise there is no peak, rather than an edge value.
    """
    freqs, power = np.asarray(freqs), np.asarray(power)
    in_fit = (freqs >= fit[0]) & (freqs <= fit[1]) & ~((freqs >= search[0]) & (freqs <= search[1]))
    slope, intercept = np.polyfit(np.log10(freqs[in_fit]), np.log10(power[in_fit]), 1)
    in_search = (freqs >= search[0]) & (freqs <= search[1])
    residual = np.log10(power[in_search]) - (intercept + slope * np.log10(freqs[in_search]))
    best = int(np.argmax(residual))
    interior = 0 < best < residual.size - 1  # a maximum at the edge is a slope, not a peak
    return float(freqs[in_search][best]) if residual[best] >= min_height and interior else float("nan")
