"""EEG channel names: one spelling, hemisphere membership, and left-right mirroring.

Cohorts spell channels differently. S1 uses the old 10-20 names T3/T4/T5/T6 where others use
T7/T8/P7/P8, and capitalisation varies. Everything downstream works on canonical 10-10 names,
so harmonisation and lesion-side mirroring (PLAN.md §3) can be written once.
"""

import re

import numpy as np

# Old 10-20 names that the 10-10 system renamed.
OLD_TO_NEW = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

# Motor-strip template (PLAN.md §3). CPz is left out: it is the reference in S1 and S3.
MOTOR_TEMPLATE = (
    "F3", "Fz", "F4", "FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CP4", "P3", "Pz", "P4",
)

_POSITION = re.compile(r"^([A-Z]+?)(\d+|Z)$")


def canonical(name):
    """Canonical 10-10 spelling: 'fc3' -> 'FC3', 'CZ' -> 'Cz', 'FPZ' -> 'Fpz', 'T3' -> 'T7'.

    Recording-system decorations are stripped ('EEG C3-REF' -> 'C3', 'Fc5.' -> 'FC5').
    Names that are not scalp positions (EOG, STI, ...) come back stripped but otherwise unchanged.
    """
    cleaned = name.strip().rstrip(".")
    cleaned = re.sub(r"^EEG[\s_-]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[-_](REF|LE|AVG|A1|A2|M1|M2)$", "", cleaned, flags=re.IGNORECASE)
    match = _POSITION.match(cleaned.upper())
    if not match:
        return cleaned
    letters, suffix = match.groups()
    if letters.startswith("FP"):
        letters = "Fp" + letters[2:]
    position = letters + ("z" if suffix == "Z" else suffix)
    return OLD_TO_NEW.get(position, position)


def _split(name):
    match = re.match(r"^([A-Za-z]+?)(\d+|z)$", canonical(name))
    if not match:
        raise ValueError(f"not a 10-10 scalp position: {name!r}")
    return match.groups()


def hemisphere(name):
    """'left' for odd numbers, 'right' for even, 'midline' for z; None for non-scalp channels."""
    try:
        _, suffix = _split(name)
    except ValueError:
        return None
    if suffix == "z":
        return "midline"
    return "left" if int(suffix) % 2 else "right"


def mirror(name):
    """Left-right counterpart: 'C3' -> 'C4', 'FC6' -> 'FC5', 'Cz' -> 'Cz'."""
    letters, suffix = _split(name)
    if suffix == "z":
        return letters + "z"
    number = int(suffix)
    return f"{letters}{number + 1 if number % 2 else number - 1}"


def mirror_data(data, ch_names):
    """Swap hemispheres in a (..., n_channels, n_times) array whose channels are `ch_names`.

    Used to put every patient's lesion on the same side. The montage must be left-right
    symmetric: each channel's mirror has to be present.
    """
    names = [canonical(ch) for ch in ch_names]
    try:
        source = [names.index(mirror(ch)) for ch in names]
    except ValueError as exc:
        raise ValueError(f"montage is not left-right symmetric: {exc}") from None
    return np.asarray(data)[..., source, :]
