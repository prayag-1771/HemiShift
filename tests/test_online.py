import numpy as np
from scipy.signal import sosfilt, sosfilt_zi

from hemishift.data.channels import EXTENDED_TEMPLATE
from hemishift.models.pooling import recentre_by_domain
from hemishift.online.lsl import StreamBuffer
from hemishift.online.session import CausalBandpass, Session, harmonised_epoch, window_samples


def test_stream_buffer_returns_windows_across_compaction():
    buffer = StreamBuffer(n_channels=2, keep=50)
    stamps = np.arange(200) / 10.0
    data = np.column_stack([np.arange(200), -np.arange(200)]).astype(float)
    for start in range(0, 200, 7):  # odd chunk sizes force several compactions
        buffer.extend(data[start:start + 7], stamps[start:start + 7])
    window = buffer.window(17.0, 20)  # samples 170..189
    np.testing.assert_array_equal(window[0], np.arange(170, 190))
    assert buffer.window(19.5, 10) is None  # not all arrived yet


def test_causal_bandpass_in_chunks_equals_one_pass():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 1000))
    band = CausalBandpass(sfreq=250.0)
    chunked = np.hstack([band(x[:, i:i + 37]) for i in range(0, 1000, 37)])
    zi = sosfilt_zi(band.sos)[:, None, :] * x[None, :, :1]
    np.testing.assert_allclose(chunked, sosfilt(band.sos, x, axis=-1, zi=zi)[0], atol=1e-10)


def test_harmonised_epoch_has_the_decoder_format():
    names = list(EXTENDED_TEMPLATE) + ["O1", "O2"]
    offset, length = window_samples(250.0)
    assert (offset, length) == (125, 875)
    window = np.random.default_rng(1).normal(size=(len(names), length))
    window[names.index("C3")] += 5.0  # mark the left hemisphere
    right = harmonised_epoch(window, names, 250.0, "R")
    left = harmonised_epoch(window, names, 250.0, "L")
    assert right.shape == (len(EXTENDED_TEMPLATE), 448)
    np.testing.assert_allclose(right.sum(axis=0), 0.0, atol=1e-9)  # common average
    c3, c4 = EXTENDED_TEMPLATE.index("C3"), EXTENDED_TEMPLATE.index("C4")
    np.testing.assert_allclose(left[c4], right[c3])  # left paralysis is mirrored


def _covariance(rng, mi):
    A = rng.normal(size=(5, 300))
    A[1] *= 0.5 if mi else 1.0
    return A @ A.T / 300


def test_session_calibrates_then_scores_only_pairs():
    rng = np.random.default_rng(2)
    donor_y = np.tile([0, 1], 50)
    donor_C = recentre_by_domain(np.array([_covariance(rng, y) for y in donor_y]), np.zeros(100))
    session = Session(donor_C, donor_y, k=3)
    labels = ["rest", "affected_mi", "rest", "sixth_finger_mi"] * 8
    events = []
    for trial, label in enumerate(labels, start=1):
        C = None if label == "sixth_finger_mi" else _covariance(rng, label == "affected_mi")
        events += session.add(C, label, trial)
    kinds = [e["type"] for e in events]
    assert kinds.count("calibration") == 3 and kinds.count("calibrated") == 1
    calibrated = next(e for e in events if e["type"] == "calibrated")
    assert calibrated["n_cal"] == 6 and set(calibrated["cv_correct"]) == {"own", "pooled"}
    # 8 pairs in total, 3 calibrate; the remaining 5 pairs give 10 scored epochs.
    last_score = [e for e in events if e["type"] == "score"][-1]
    assert last_score["n"] == 10
    # Rests before sixth-finger trials are decoded but never scored: one in block 3 (right after
    # calibration ends) and one in each of blocks 4-8.
    assert kinds.count("unscored") == 6
    assert last_score["accuracy"]["pooled"] > 80
