"""Load cohorts into the common epoch format (PLAN.md §3).

Every loader returns an EpochSet: template channels, common average, 1-40 Hz, 128 Hz, the
0.5-4.0 s window after MI onset, microvolts, and the lesion on the left (affected hand =
'right_hand'). One metadata row per trial, in recording order within each patient, so
calibration trials can be taken chronologically (splits.calibration_split).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hemishift.data.channels import MOTOR_TEMPLATE
from hemishift.data.harmonise import BAND_HZ, MI_WINDOW_S, TARGET_SFREQ, fix_length, harmonise
from hemishift.paths import RAW_DIR


@dataclass
class EpochSet:
    X: np.ndarray  # (n_trials, n_channels, n_times), microvolts
    labels: np.ndarray  # after lesion-side normalisation: 'right_hand' is the affected hand
    meta: pd.DataFrame  # cohort, patient, session, run, trial, affected_side, group
    sfreq: float = TARGET_SFREQ
    ch_names: tuple = MOTOR_TEMPLATE


def healthy_side(subject):
    """Healthy subjects get a virtual left-hemisphere lesion: affected hand = right hand."""
    return "R"


def liu2024_affected_side():
    """Paralysed side per Liu2024 subject from the original figshare table.

    NEMAR's participants.tsv, which MOABB loads, has no clinical columns (LITERATURE.md §3).
    """
    table = pd.read_csv(RAW_DIR / "liu2024_meta" / "participants.tsv", sep="\t")
    sides = table.set_index("Participant_ID")["ParalysisSide"].str[0].str.upper()
    return {int(pid.split("-")[1]): side for pid, side in sides.items()}


def configure_moabb():
    """Store MOABB data under data/raw/moabb, fetched from each dataset's own host.

    MOABB 1.7.2 otherwise prefers NEMAR mirrors. Their downloader writes files in place and
    later trusts whatever exists, so an interrupted transfer can leave a truncated file in
    use. Upstream hosts go through pooch (temp file, then rename), and they are also the
    copies MOABB's loaders read (e.g. Liu2024's figshare zip).
    """
    import moabb
    from moabb.utils import set_download_dir

    set_download_dir(str(RAW_DIR / "moabb"))
    moabb.set_download_provider("upstream")


def hefmi_affected_side():
    """Paralysed side per HEFMI-ICH patient (MOABB subjects 18-37), from configs/.

    Transcribed from Shi et al. 2025 (Sci Data 12:1816), Table 1, which gives lesion location
    only. Affected side = opposite the lesion (all lesions but one are supratentorial). The
    brainstem patient has no defined side and is left out. UNVERIFIED: that the paper's
    patient IDs 1-20 map to MOABB subjects 18-37 in order (MOABB groups figshare files by
    upload order).
    """
    from hemishift.paths import CONFIG_DIR

    table = pd.read_csv(CONFIG_DIR / "hefmi_ich_patients.csv").dropna(subset=["affected_side"])
    return dict(zip(table["moabb_subject"].astype(int), table["affected_side"]))


def moabb_cohort(name):
    """(dataset, affected_side, group) for the MOABB cohorts whose sides are known.

    `affected_side` is a dict or callable subject -> 'L'/'R' (None to skip the subject);
    `group` is a string or callable subject -> 'stroke'/'healthy'.
    """
    import moabb.datasets as moabb_datasets

    from hemishift.data.registry import COHORTS

    configure_moabb()
    dataset = getattr(moabb_datasets, COHORTS[name]["source"])()
    if name == "liu2024":
        return dataset, liu2024_affected_side(), "stroke"
    if COHORTS[name]["role"] == "healthy":
        return dataset, healthy_side, "healthy"
    if name == "hefmi_ich":  # subjects 1-17 healthy controls, 18-37 ICH patients
        patients = hefmi_affected_side()
        return (dataset,
                lambda s: healthy_side(s) if s <= 17 else patients.get(s),
                lambda s: "healthy" if s <= 17 else "stroke")
    raise KeyError(f"no affected-side source for {name!r} yet")


def load_moabb_epochs(dataset, affected_side, cohort, group, events=("left_hand", "right_hand"),
                      subjects=None):
    """Epochs from a MOABB dataset object, harmonised patient by patient.

    `affected_side` maps a MOABB subject number to 'L' or 'R'. Stroke cohorts pass their
    clinical table; healthy cohorts pass `healthy_side`.
    """
    from moabb.paradigms import MotorImagery

    paradigm = MotorImagery(
        n_classes=len(events), events=list(events), fmin=BAND_HZ[0], fmax=BAND_HZ[1],
        tmin=MI_WINDOW_S[0], tmax=MI_WINDOW_S[1], resample=TARGET_SFREQ,
    )
    epochs, labels, meta = paradigm.get_data(dataset, subjects=subjects, return_epochs=True)
    data = fix_length(epochs.get_data(copy=False)) * 1e6  # volts -> microvolts
    labels = np.asarray(labels)

    parts, part_labels, part_meta = [], [], []
    for subject in pd.unique(meta["subject"]):
        rows = np.flatnonzero(meta["subject"].to_numpy() == subject)
        side = affected_side(subject) if callable(affected_side) else affected_side[subject]
        X, subject_labels = harmonise(data[rows], epochs.ch_names, labels[rows], side)
        parts.append(X)
        part_labels.extend(subject_labels)
        part_meta.append(pd.DataFrame({
            "cohort": cohort,
            "patient": f"{cohort}:{subject}",
            "session": meta["session"].to_numpy()[rows],
            "run": meta["run"].to_numpy()[rows],
            "trial": np.arange(rows.size),  # recording order within the patient
            "affected_side": side,
            "group": group,
        }))
    return EpochSet(
        X=np.concatenate(parts).astype(np.float32),
        labels=np.asarray(part_labels),
        meta=pd.concat(part_meta, ignore_index=True),
    )
