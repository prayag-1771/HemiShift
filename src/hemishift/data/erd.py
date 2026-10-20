"""ERD measures for diagnosing how stroke EEG differs from healthy EEG (PLAN.md §4.0, rule 1).

ERD (event-related desynchronisation) is the drop in mu/beta power while a person imagines
moving. Its size per hemisphere, and the balance between hemispheres, is what LesionSim has
to reproduce, so these measures come first.
"""

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

MU_HZ = (8.0, 13.0)
BETA_HZ = (13.0, 30.0)


def band_power(X, sfreq, band, order=4):
    """Instantaneous power in `band` (squared Hilbert envelope), same shape as X."""
    sos = butter(order, band, btype="bandpass", fs=sfreq, output="sos")
    return np.abs(hilbert(sosfiltfilt(sos, X, axis=-1), axis=-1)) ** 2


def erd_percent(task_power, rest_power):
    """100 * (task - rest) / rest. Negative values mean desynchronisation (the MI signal)."""
    return 100.0 * (np.asarray(task_power) - np.asarray(rest_power)) / np.asarray(rest_power)


def laterality_index(erd_contra, erd_ipsi):
    """Balance of desynchronisation between hemispheres, in [-1, 1].

    +1: all desynchronisation is contralateral to the imagined hand (the healthy pattern);
    -1: all of it is ipsilateral (the signal has shifted hemisphere). NaN when neither side
    desynchronises.
    """
    contra = np.maximum(0.0, -np.asarray(erd_contra, dtype=float))
    ipsi = np.maximum(0.0, -np.asarray(erd_ipsi, dtype=float))
    total = contra + ipsi
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, (contra - ipsi) / total, np.nan)
