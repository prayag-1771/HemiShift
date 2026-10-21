"""Configurable preprocessing for continuous recordings, ending in harmonised MI epochs.

Each option can be switched off, so its contribution can be measured on its own
(scripts/s1_preprocessing.py). Every step is label-free, so it can run on any patient,
including locked test patients at evaluation time, without leaking class information.
"""

from dataclasses import dataclass

import mne
import numpy as np

from hemishift.data.channels import CORE_TEMPLATE, canonical
from hemishift.data.harmonise import MI_WINDOW_S, TARGET_SFREQ, fix_length
from hemishift.preprocess.quality import channel_quality


@dataclass(frozen=True)
class Preprocessing:
    template: tuple = CORE_TEMPLATE
    interpolate: bool = False  # repair flat / noisy / decorrelated channels (spherical spline)
    eog_regression: bool = False  # regress HEOL/VEOR out of the EEG
    reference: str = "template"  # "template" (CAR over the picked channels), "all" (CAR over
    #                              every EEG channel before picking) or "laplacian" (CSD)
    band: tuple = (1.0, 40.0)
    window: tuple = MI_WINDOW_S


def prepare_epochs(raw, config, events=("left_hand", "right_hand")):
    """(X in microvolts, labels, info) for MI epochs of one recording.

    `raw` needs EEG positions and annotations named after `events`. `info` reports what was
    repaired, so per-patient decisions are on record.
    """
    raw = raw.copy().load_data(verbose="error").filter(*config.band, verbose="error")
    repaired = []
    if config.interpolate:
        quality = channel_quality(raw)
        repaired = quality.loc[quality["interpolate"], "channel"].tolist()
        if repaired:
            raw.info["bads"] = repaired
            raw.interpolate_bads(reset_bads=True, verbose="error")
    if config.eog_regression and "eog" in raw.get_channel_types():
        # MNE's regression needs an average reference first (Gratton-style practice).
        raw.set_eeg_reference("average", projection=True, verbose="error")
        regression = mne.preprocessing.EOGRegression(picks="eeg", picks_artifact="eog")
        raw = regression.fit(raw).apply(raw).apply_proj(verbose="error")
    if config.reference == "all":
        raw.set_eeg_reference("average", verbose="error")
    elif config.reference == "laplacian":
        raw = mne.preprocessing.compute_current_source_density(raw.pick("eeg"), verbose="error")

    event_id = {name: code for name, code in
                mne.events_from_annotations(raw, verbose="error")[1].items() if name in events}
    found = mne.events_from_annotations(raw, event_id=event_id, verbose="error")[0]
    names = [ch for ch in raw.ch_names if canonical(ch) in config.template]
    epochs = mne.Epochs(raw, found, event_id, tmin=config.window[0], tmax=config.window[1],
                        baseline=None, picks=names, preload=True, verbose="error")
    epochs = epochs.reorder_channels(sorted(names, key=lambda ch: config.template.index(canonical(ch))))
    epochs = epochs.resample(TARGET_SFREQ, verbose="error")
    X = fix_length(epochs.get_data(copy=False),
                   int(round((config.window[1] - config.window[0]) * TARGET_SFREQ)))
    X = X * (1.0 if config.reference == "laplacian" else 1e6)  # CSD units differ; scale-free models
    if config.reference == "template":
        X = X - X.mean(axis=1, keepdims=True)
    labels = np.array([{code: name for name, code in event_id.items()}[c] for c in epochs.events[:, 2]])
    return X, labels, {"repaired": repaired}
