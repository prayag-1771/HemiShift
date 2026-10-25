"""Online calibration and decoding of Task B (affected-hand MI vs rest) for one patient.

The streaming counterpart of protocol P3: the first k (rest, MI) pairs calibrate, then every
later epoch is decoded as soon as its window has arrived. Two decoders run side by side, so
the effect of pooling shows trial by trial:
  own      TS + LR on the calibration epochs only
  pooled   calibration + other patients' epochs (`PooledTSLR`, confirmed on the S3 locked test)
After calibration, the SafeTransfer gate (leave-one-pair-out CV on the calibration pairs)
reports whether the decoders can be trusted for this patient.
Scoring follows the offline protocol: a rest epoch counts only when the next cue is
affected-hand MI (a pair); sixth-finger trials are not decoded.
"""

from fractions import Fraction

import numpy as np
from pyriemann.estimation import Covariances
from scipy.signal import butter, resample_poly, sosfilt, sosfilt_zi, sosfiltfilt

from hemishift.data.channels import EXTENDED_TEMPLATE, mirror_data
from hemishift.data.harmonise import (
    BAND_HZ,
    MI_WINDOW_S,
    N_TIMES,
    TARGET_SFREQ,
    common_average,
    fix_length,
    select_template,
)
from hemishift.eval.safetransfer import choose
from hemishift.models.pooling import OwnTSLR, PooledTSLR

LABELS = {"rest": 0, "affected_mi": 1}


def covariances(X, band=(8.0, 30.0)):
    """8-30 Hz OAS covariances of 128 Hz epochs, exactly as in the experiments."""
    sos = butter(4, band, btype="bandpass", fs=TARGET_SFREQ, output="sos")
    return Covariances("oas").transform(sosfiltfilt(sos, np.asarray(X, dtype=float), axis=-1))


class CausalBandpass:
    """The 1-40 Hz band-pass, run on a stream chunk by chunk (the filter state is kept)."""

    def __init__(self, sfreq, band=BAND_HZ, order=4):
        self.sos = butter(order, band, btype="bandpass", fs=sfreq, output="sos")
        self.state = None

    def __call__(self, chunk):
        """Filter a (n_channels, n_samples) chunk that follows the previous one."""
        chunk = np.asarray(chunk, dtype=float)
        if self.state is None:  # start in steady state at the first sample, not at zero
            self.state = sosfilt_zi(self.sos)[:, None, :] * chunk[None, :, :1]
        out, self.state = sosfilt(self.sos, chunk, axis=-1, zi=self.state)
        return out


def harmonised_epoch(window, ch_names, sfreq, affected_side, template=EXTENDED_TEMPLATE):
    """A filtered window (all channels, the MI_WINDOW_S span after a cue, at `sfreq`, µV) in
    the format the decoders were built on: 128 Hz, template channels, common average over
    them, lesion in the left hemisphere."""
    ratio = Fraction(TARGET_SFREQ / sfreq).limit_denominator(1000)
    x = resample_poly(np.asarray(window, dtype=float), ratio.numerator, ratio.denominator, axis=-1)
    x = common_average(select_template(fix_length(x, N_TIMES), ch_names, template))
    return mirror_data(x, template) if affected_side == "L" else x


def window_samples(sfreq):
    """Samples from a cue to the start of its window, and the window length, at `sfreq`."""
    return int(round(MI_WINDOW_S[0] * sfreq)), int(round((MI_WINDOW_S[1] - MI_WINDOW_S[0]) * sfreq))


class Session:
    """Feed epochs in recording order with `add`; it returns events describing what happened."""

    def __init__(self, donor_C, donor_y, k=5):
        self.k = k
        self.donor_C, self.donor_y = donor_C, donor_y
        self.cal_C, self.cal_y = [], []
        self.pending_rest = None  # the last rest epoch, until the next cue says if it is paired
        self.decoders = None
        self.hits = {"own": [], "pooled": []}

    @property
    def calibrated(self):
        return self.decoders is not None

    def add(self, C, label, trial):
        """Process one epoch's covariance; `label` is 'rest', 'affected_mi' or 'sixth_finger_mi'."""
        events = []
        if label == "rest":
            decision = self._decide(C, "rest", trial) if self.calibrated else None
            if decision:
                events.append(decision)
            self.pending_rest = (C, decision)
            return events
        rest, self.pending_rest = self.pending_rest, None
        if label != "affected_mi":  # sixth-finger trial: its preceding rest is not a pair
            if rest and rest[1]:
                events.append({"type": "unscored", "trial": rest[1]["trial"]})
            return events
        if not self.calibrated:
            if rest is not None:
                self.cal_C += [rest[0], C]
                self.cal_y += [0, 1]
                events.append({"type": "calibration", "pairs": len(self.cal_y) // 2, "k": self.k})
                if len(self.cal_y) == 2 * self.k:
                    events.append(self._calibrate())
            return events
        decision = self._decide(C, "affected_mi", trial)
        if rest and rest[1]:
            events.append(self._score(rest[1]))
        events.append(decision)
        events.append(self._score(decision))
        return events

    def _calibrate(self):
        C, y = np.array(self.cal_C), np.array(self.cal_y)
        correct = {"own": 0, "pooled": 0}
        for pair in range(len(y) // 2):  # leave-one-pair-out, calibration data only
            held = np.arange(2 * pair, 2 * pair + 2)
            fit_on = np.setdiff1d(np.arange(len(y)), held)
            for name, decoder in self._new_decoders().items():
                p = decoder.fit(C[fit_on], y[fit_on]).predict_proba(C[held])
                correct[name] += int(((p > 0.5).astype(int) == y[held]).sum())
        self.decoders = {name: d.fit(C, y) for name, d in self._new_decoders().items()}
        choice, accuracy = choose(correct, len(y))
        return {"type": "calibrated", "k": self.k, "cv_correct": correct, "n_cal": len(y),
                "gate": choice or "abstain", "gate_accuracy": accuracy}

    def _new_decoders(self):
        return {"own": OwnTSLR(), "pooled": PooledTSLR(self.donor_C, self.donor_y)}

    def _decide(self, C, label, trial):
        p = {name: float(d.predict_proba(C[None])[0]) for name, d in self.decoders.items()}
        return {"type": "decision", "trial": trial, "label": label, "p_mi": p,
                "correct": {name: (v > 0.5) == (LABELS[label] == 1) for name, v in p.items()}}

    def _score(self, decision):
        for name, ok in decision["correct"].items():
            self.hits[name].append(bool(ok))
        return {"type": "score", "trial": decision["trial"], "n": len(self.hits["own"]),
                "accuracy": {name: 100 * float(np.mean(h)) for name, h in self.hits.items()}}
