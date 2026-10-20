import numpy as np
import pytest
from moabb.datasets.fake import FakeDataset

from hemishift.data.channels import MOTOR_TEMPLATE
from hemishift.data.harmonise import N_TIMES
from hemishift.data.loaders import healthy_side, load_moabb_epochs


@pytest.fixture(scope="module")
def fake():
    # Template channels in a different order plus an extra channel. (Spelling variants are
    # covered in test_channels; MOABB's fake montage needs standard spellings.)
    channels = tuple(reversed(MOTOR_TEMPLATE)) + ("O1",)
    return FakeDataset(event_list=("left_hand", "right_hand"), n_sessions=1, n_runs=1,
                       n_subjects=2, channels=channels, sfreq=256, duration=60, seed=0)


def test_common_format(fake):
    epochs = load_moabb_epochs(fake, healthy_side, cohort="fake", group="healthy")
    assert N_TIMES == 448  # 3.5 s at 128 Hz
    assert epochs.X.shape[1:] == (len(MOTOR_TEMPLATE), N_TIMES)
    assert epochs.X.dtype == np.float32
    np.testing.assert_allclose(epochs.X.mean(axis=1), 0.0, atol=1e-3)  # common average
    assert set(epochs.labels) == {"left_hand", "right_hand"}
    assert list(epochs.meta["patient"].unique()) == ["fake:1", "fake:2"]


def test_trial_index_is_recording_order_per_patient(fake):
    meta = load_moabb_epochs(fake, healthy_side, cohort="fake", group="healthy").meta
    for _, rows in meta.groupby("patient"):
        np.testing.assert_array_equal(rows["trial"], np.arange(len(rows)))


def test_left_paralysis_swaps_hand_labels(fake):
    right = load_moabb_epochs(fake, {1: "R", 2: "R"}, cohort="fake", group="stroke")
    left = load_moabb_epochs(fake, {1: "L", 2: "L"}, cohort="fake", group="stroke")
    swap = {"left_hand": "right_hand", "right_hand": "left_hand"}
    assert list(left.labels) == [swap[label] for label in right.labels]
    c3, c4 = MOTOR_TEMPLATE.index("C3"), MOTOR_TEMPLATE.index("C4")
    np.testing.assert_allclose(left.X[:, c3], right.X[:, c4])  # hemispheres swapped
