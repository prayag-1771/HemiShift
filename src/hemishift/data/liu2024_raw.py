"""S1 (Liu et al. 2024) as a continuous MNE Raw, from the EDF (default) or the raw .mat.

The two sources match with r >= 0.990 per channel after a 4-30 Hz filter (checked
2026-10-21), and decode identically. The EDF is the default because the raw .mat files of
patients 5, 13, 15 and 49 are corrupted: neighbouring channels correlate at r <= 0.08, which
real EEG never does. Patient 13 is corrupted in the EDF too and is excluded (EXCLUDED).

.mat layout: 33 rows = 29 EEG, CPz (the reference, all zeros), HEOL and VEOR (EOG), and the
trigger (1 instruction, 2 MI onset, 3 break). The 40 trials of 8 s were cut back to back, so
stacking them restores the continuous 320 s recording.
"""

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat

from hemishift.data.channels import canonical
from hemishift.paths import RAW_DIR

SFREQ = 500.0
_EEG = ["FP1", "FP2", "Fz", "F3", "F4", "F7", "F8", "FCz", "FC3", "FC4", "FT7", "FT8", "Cz", "C3",
        "C4", "T3", "T4", "CPz", "CP3", "CP4", "TP7", "TP8", "Pz", "P3", "P4", "T5", "T6", "Oz",
        "O1", "O2"]
ROWS = _EEG + ["HEOL", "VEOR", "TRIGGER"]
LABELS = {1: "left_hand", 2: "right_hand"}
# Label-free quality rule, applied to all 50 patients alike: exclude a recording whose median
# correlation between neighbouring channels (1-40 Hz) is below 0.3.
EXCLUDED = {13: "neighbouring channels uncorrelated (median r 0.08) in both the EDF and the .mat"}


def mat_path(subject):
    return (RAW_DIR / "liu2024_raw" / "sourcedata" / "sourcedata" / f"sub-{subject:02d}" /
            f"sub-{subject:02d}_task-motor-imagery_eeg.mat")


def load_raw(subject, source="edf"):
    """Continuous recording: 29 EEG (canonical 10-10 names, standard positions), 2 EOG.

    CPz is dropped (it is the all-zero reference). MI onsets become annotations
    'left_hand' / 'right_hand'. Units are volts, as MNE expects.
    """
    if source == "edf":
        return _load_edf(subject)
    eeg = loadmat(mat_path(subject), squeeze_me=True, struct_as_record=False)["eeg"]
    stacked = np.concatenate(np.asarray(eeg.rawdata, dtype=float), axis=1)  # (33, 160000)
    keep = [i for i, name in enumerate(ROWS) if name not in ("CPz", "TRIGGER")]
    names = [canonical(ROWS[i]) for i in keep]
    types = ["eog" if ROWS[i] in ("HEOL", "VEOR") else "eeg" for i in keep]
    raw = mne.io.RawArray(stacked[keep] * 1e-6, mne.create_info(names, SFREQ, types),
                          verbose="error")
    raw.set_montage("standard_1005", on_missing="ignore", verbose="error")

    onsets = np.flatnonzero(stacked[ROWS.index("TRIGGER")] == 2)
    labels = [LABELS[int(v)] for v in np.asarray(eeg.label)]
    raw.set_annotations(mne.Annotations(onsets / SFREQ, 4.0, labels))
    return raw


def _load_edf(subject):
    from moabb.datasets import Liu2024

    from hemishift.data.loaders import configure_moabb

    configure_moabb()
    raw = Liu2024().get_data(subjects=[subject])[subject]["0"]["0"]
    events = mne.find_events(raw, shortest_event=0, verbose="error")
    events = events[np.isin(events[:, 2], list(LABELS))]
    raw = raw.pick(["eeg", "eog"]).load_data(verbose="error")
    raw.rename_channels({ch: canonical(ch) for ch in raw.ch_names if canonical(ch) != ch})
    raw.set_montage("standard_1005", on_missing="ignore", verbose="error")
    raw.set_annotations(mne.Annotations(events[:, 0] / raw.info["sfreq"], 4.0,
                                        [LABELS[int(code)] for code in events[:, 2]]))
    return raw


def participants():
    """The original clinical table (NEMAR's copy is broken), indexed by subject number."""
    table = pd.read_csv(RAW_DIR / "liu2024_meta" / "participants.tsv", sep="\t")
    table.index = table["Participant_ID"].str.split("-").str[1].astype(int)
    return table
