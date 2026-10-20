"""Common format for every cohort (PLAN.md §3).

Cohort loaders deliver epochs that are already band-passed, resampled and cut to the common
window (constants below). This module does the cohort-independent part on arrays:
motor-template channels in a fixed order, common-average reference, and lesion-side
normalisation so that the lesion is always in the LEFT hemisphere and the affected hand is
always the RIGHT hand.
"""

import numpy as np

from hemishift.data.channels import MOTOR_TEMPLATE, canonical, mirror_data

TARGET_SFREQ = 128.0  # UET175's native rate, the lowest among the cohorts
BAND_HZ = (1.0, 40.0)  # stored band; models choose their own sub-band
MI_WINDOW_S = (0.5, 4.0)  # seconds after MI onset

_SWAP_HANDS = {"left_hand": "right_hand", "right_hand": "left_hand"}
# After lesion-side normalisation the affected hand is always the right hand (Task A labels).
TASK_A_LABELS = {"right_hand": "affected", "left_hand": "unaffected"}


def select_template(X, ch_names, template=MOTOR_TEMPLATE):
    """Template channels, in template order, from a (..., n_channels, n_times) array."""
    names = [canonical(ch) for ch in ch_names]
    missing = [ch for ch in template if ch not in names]
    if missing:
        raise ValueError(f"cohort lacks template channels {missing}")
    return np.asarray(X)[..., [names.index(ch) for ch in template], :]


def common_average(X):
    """Re-reference to the mean over channels (axis -2)."""
    X = np.asarray(X, dtype=float)
    return X - X.mean(axis=-2, keepdims=True)


def normalise_lesion_side(X, labels, ch_names, affected_side):
    """Mirror left-side paralysis so every lesion sits in the left hemisphere.

    `affected_side` is the paralysed side, 'L' or 'R'. Left paralysis means a right-hemisphere
    lesion, so the data are mirrored and the left/right hand labels swap; afterwards the
    affected hand is always 'right_hand'. 'R' passes through unchanged (healthy subjects are
    given a virtual side by the caller).
    """
    if affected_side == "R":
        return np.asarray(X), list(labels)
    if affected_side != "L":
        raise ValueError(f"affected_side must be 'L' or 'R', got {affected_side!r}")
    return mirror_data(X, ch_names), [_SWAP_HANDS.get(label, label) for label in labels]


def harmonise(X, ch_names, labels, affected_side, template=MOTOR_TEMPLATE):
    """Template channels -> common average -> lesion on the left. Returns (X, labels)."""
    X = common_average(select_template(X, ch_names, template))
    return normalise_lesion_side(X, labels, template, affected_side)
