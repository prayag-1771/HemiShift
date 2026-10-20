import numpy as np
import pytest

from hemishift.data.channels import (
    CORE_TEMPLATE,
    EXTENDED_TEMPLATE,
    canonical,
    hemisphere,
    mirror,
    mirror_data,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("fc3", "FC3"), ("CZ", "Cz"), ("FPZ", "Fpz"), ("Fp1", "Fp1"), ("AFz", "AFz"),
        ("T3", "T7"), ("T4", "T8"), ("T5", "P7"), ("T6", "P8"),  # old 10-20 names (S1)
        ("EEG C3-REF", "C3"), ("Fc5.", "FC5"), ("CPZ", "CPz"),
        ("HEOL", "HEOL"), ("STI 014", "STI 014"),  # non-scalp channels pass through
    ],
)
def test_canonical(raw, expected):
    assert canonical(raw) == expected


def test_hemisphere():
    assert hemisphere("C3") == "left"
    assert hemisphere("FC4") == "right"
    assert hemisphere("Cz") == "midline"
    assert hemisphere("T3") == "left"  # T3 is T7
    assert hemisphere("VEOR") is None


@pytest.mark.parametrize(
    "name, expected", [("C3", "C4"), ("FC6", "FC5"), ("Cz", "Cz"), ("Fp1", "Fp2"), ("T3", "T8")]
)
def test_mirror(name, expected):
    assert mirror(name) == expected


@pytest.mark.parametrize("template", [CORE_TEMPLATE, EXTENDED_TEMPLATE])
def test_mirror_is_an_involution_on_the_templates(template):
    for name in template:
        assert mirror(mirror(name)) == name
        assert mirror(name) in template  # each template is left-right symmetric


def test_core_template_is_inside_the_extended_one():
    assert set(CORE_TEMPLATE) <= set(EXTENDED_TEMPLATE)


def test_mirror_data_swaps_hemispheres():
    names = ["C3", "Cz", "C4"]
    data = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])[None]  # (trials, channels, times)
    flipped = mirror_data(data, names)
    np.testing.assert_array_equal(flipped[0, :, 0], [3.0, 2.0, 1.0])
    np.testing.assert_array_equal(mirror_data(flipped, names), data)  # flipping twice restores


def test_mirror_data_rejects_asymmetric_montage():
    with pytest.raises(ValueError, match="not left-right symmetric"):
        mirror_data(np.zeros((2, 10)), ["C3", "Cz"])
