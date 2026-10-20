"""LesionSim operators: make healthy motor-imagery EEG look like stroke EEG (PLAN.md §4.1).

Data come harmonised: the (virtual) lesion is in the LEFT hemisphere, so ipsilesional
channels are the odd-numbered ones and contralesional channels their mirrors.

Every operator works on band envelopes. A band component b(t) is split into amplitude A(t)
and phase φ(t) with the Hilbert transform; only A(t) is edited, and b'(t) = A'(t)·cos φ(t)
replaces b(t). Re(analytic signal) is b(t) itself, so leaving A unchanged reproduces the
input exactly. That identity is what the unit tests pin down.

`rest_amplitude` is each channel's mean envelope at rest for one subject,
shape (n_channels,), in the same band. ERD is A(t) falling below it.
"""

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from hemishift.data.channels import hemisphere, mirror

MU_HZ = (8.0, 13.0)
BETA_HZ = (13.0, 30.0)


def band_split(X, sfreq, band, pad_s=0.5):
    """Band component, its amplitude and its phase; reflection padding limits edge effects."""
    X = np.asarray(X, dtype=float)
    pad = min(int(pad_s * sfreq), X.shape[-1] - 1)
    widths = [(0, 0)] * (X.ndim - 1) + [(pad, pad)]
    sos = butter(4, band, btype="bandpass", fs=sfreq, output="sos")
    component = sosfiltfilt(sos, np.pad(X, widths, mode="reflect"), axis=-1)
    analytic = hilbert(component, axis=-1)[..., pad:X.shape[-1] + pad]
    return analytic.real, np.abs(analytic), np.angle(analytic)


def _rebuild(X, component, amplitude, phase, rows):
    out = np.array(X, dtype=float, copy=True)
    out[..., rows, :] += np.maximum(amplitude, 0.0) * np.cos(phase) - component
    return out


def ipsilesional(ch_names):
    """Indices of left-hemisphere (ipsilesional after harmonisation) channels."""
    return [i for i, ch in enumerate(ch_names) if hemisphere(ch) == "left"]


def attenuate_erd(X, sfreq, ch_names, rest_amplitude, alpha, band=MU_HZ):
    """O1, weaker ERD on the lesioned side: A' = A0 + alpha·(A − A0) on ipsilesional channels.

    alpha = 1 leaves the data unchanged; alpha = 0 removes the band's ERD there entirely.
    alpha may also be one value per trial, shape (n_trials,), for X of shape
    (n_trials, n_channels, n_times).
    """
    rows = ipsilesional(ch_names)
    component, amplitude, phase = band_split(X, sfreq, band)
    a0 = np.asarray(rest_amplitude, dtype=float)[rows, None]
    gain = np.asarray(alpha, dtype=float)
    if gain.ndim:  # one gain per trial
        gain = gain.reshape(gain.shape + (1, 1))
    edited = a0 + gain * (amplitude[..., rows, :] - a0)
    return _rebuild(X, component[..., rows, :], edited, phase[..., rows, :], rows)


def vary_erd(X, sfreq, ch_names, rest_amplitude, spread, rng, band=BETA_HZ):
    """O4b, less consistent ERD across trials, the one stroke-vs-healthy difference that
    survived correction in the gap diagnosis (ACHIEVEMENTS.md).

    Each trial's ipsilesional ERD is scaled by its own log-normal gain (median 1, log-sd
    `spread`), which widens the trial-to-trial spread without shifting the typical trial.
    spread = 0 leaves the data unchanged.
    """
    gains = np.exp(rng.normal(0.0, spread, size=np.asarray(X).shape[0]))
    return attenuate_erd(X, sfreq, ch_names, rest_amplitude, gains, band=band)


def shift_to_contralesional(X, sfreq, ch_names, rest_amplitude, beta, band=MU_HZ):
    """O2, the healthy hemisphere takes over: add a share `beta` of each ipsilesional
    channel's relative ERD to its mirror channel. beta = 0 leaves the data unchanged."""
    names = list(ch_names)
    ipsi = ipsilesional(names)
    contra = [names.index(mirror(names[i])) for i in ipsi]
    component, amplitude, phase = band_split(X, sfreq, band)
    a0 = np.asarray(rest_amplitude, dtype=float)
    relative_erd = (amplitude[..., ipsi, :] - a0[ipsi, None]) / a0[ipsi, None]
    edited = amplitude[..., contra, :] + beta * relative_erd * a0[contra, None]
    return _rebuild(X, component[..., contra, :], edited, phase[..., contra, :], contra)


def jitter_erd(X, sfreq, ch_names, rest_amplitude, shift_s, band=MU_HZ):
    """O4, looser timing: delay (shift_s > 0) or advance the ipsilesional ERD time course.

    Samples shifted in from outside the window take the rest level, i.e. no ERD there.
    """
    rows = ipsilesional(ch_names)
    component, amplitude, phase = band_split(X, sfreq, band)
    a0 = np.asarray(rest_amplitude, dtype=float)[rows, None]
    modulation = amplitude[..., rows, :] - a0
    shift = int(round(shift_s * sfreq))
    shifted = np.zeros_like(modulation)
    if shift >= 0:
        shifted[..., shift:] = modulation[..., : modulation.shape[-1] - shift]
    else:
        shifted[..., :shift] = modulation[..., -shift:]
    return _rebuild(X, component[..., rows, :], a0 + shifted, phase[..., rows, :], rows)
