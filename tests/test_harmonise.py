import numpy as np
import pytest

from hemishift.data.channels import MOTOR_TEMPLATE
from hemishift.data.harmonise import (
    N_TIMES,
    common_average,
    fix_length,
    harmonise,
    normalise_lesion_side,
    select_template,
)


def _recording():
    """One trial; each template channel holds a distinct constant so positions are traceable.

    Channels arrive reversed, upper-cased and with an extra EOG channel, as real cohorts do.
    """
    values = {ch: float(i + 1) for i, ch in enumerate(MOTOR_TEMPLATE)}
    source = list(reversed(MOTOR_TEMPLATE))
    names = [ch.upper() for ch in source] + ["HEOL"]
    rows = [values[ch] for ch in source] + [99.0]
    return np.tile(np.array(rows)[:, None], (1, 8))[None], names, values


def test_select_template_orders_channels_and_drops_extras():
    X, names, values = _recording()
    out = select_template(X, names)
    assert out.shape == (1, len(MOTOR_TEMPLATE), 8)
    np.testing.assert_array_equal(out[0, :, 0], [values[ch] for ch in MOTOR_TEMPLATE])


def test_select_template_reports_missing_channels():
    with pytest.raises(ValueError, match="lacks template channels"):
        select_template(np.zeros((1, 2, 4)), ["C3", "C4"])


def test_common_average_has_zero_channel_mean():
    X = np.random.default_rng(0).normal(size=(3, 5, 10))
    np.testing.assert_allclose(common_average(X).mean(axis=1), 0.0, atol=1e-12)


def test_fix_length_crops_448_and_449_to_the_same_length():
    for n in (N_TIMES, N_TIMES + 1):
        assert fix_length(np.zeros((2, 3, n))).shape == (2, 3, N_TIMES)
    with pytest.raises(ValueError, match="need 448"):
        fix_length(np.zeros((2, 3, N_TIMES - 1)))


def test_left_paralysis_is_mirrored_and_hand_labels_swap():
    names = ["C3", "Cz", "C4"]
    X = np.array([[[1.0], [2.0], [3.0]]])
    out, labels = normalise_lesion_side(X, ["left_hand", "rest"], names, "L")
    np.testing.assert_array_equal(out[0, :, 0], [3.0, 2.0, 1.0])
    assert labels == ["right_hand", "rest"]  # affected (left) hand is now the right hand


def test_right_paralysis_passes_through():
    X = np.ones((1, 3, 2))
    out, labels = normalise_lesion_side(X, ["left_hand"], ["C3", "Cz", "C4"], "R")
    np.testing.assert_array_equal(out, X)
    assert labels == ["left_hand"]


def test_unknown_side_is_rejected():
    with pytest.raises(ValueError, match="affected_side"):
        normalise_lesion_side(np.ones((1, 3, 2)), [], ["C3", "Cz", "C4"], "left")


def test_harmonise_end_to_end_left_paralysis():
    X, names, values = _recording()
    out, labels = harmonise(X, names, ["left_hand"], "L")
    template_values = np.array([values[ch] for ch in MOTOR_TEMPLATE])
    centred = template_values - template_values.mean()
    mirrored_order = [MOTOR_TEMPLATE.index(ch) for ch in
                      ("F4", "Fz", "F3", "FC4", "FCz", "FC3", "C4", "Cz", "C3",
                       "CP4", "CP3", "P4", "Pz", "P3")]
    np.testing.assert_allclose(out[0, :, 0], centred[mirrored_order])
    assert labels == ["right_hand"]
