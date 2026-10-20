import numpy as np
import pytest

from hemishift.data.splits import calibration_split, leave_site_out, patient_folds, split_hash


def test_patient_folds_are_balanced_and_reproducible():
    patients = [f"p{i}" for i in range(23)]
    folds = patient_folds(patients, n_folds=5, seed=3)
    assert set(folds) == set(patients)
    sizes = np.bincount(list(folds.values()))
    assert sizes.max() - sizes.min() <= 1
    assert folds == patient_folds(list(reversed(patients)), n_folds=5, seed=3)


def test_leave_site_out_holds_each_site_out_once():
    splits = list(leave_site_out(["beijing", "tianjin", "hefmi", "beijing"]))
    assert [held for held, _ in splits] == ["beijing", "hefmi", "tianjin"]
    for held, train in splits:
        assert held not in train and len(train) == 2


def test_calibration_takes_first_k_trials_per_class():
    labels = ["L", "R", "L", "R", "L", "R", "L", "R"]
    calibration, test = calibration_split(labels, k_per_class=2)
    np.testing.assert_array_equal(calibration, [0, 1, 2, 3])
    np.testing.assert_array_equal(test, [4, 5, 6, 7])


def test_calibration_random_mode_is_reproducible_and_disjoint():
    labels = np.repeat(["L", "R"], 20)
    a = calibration_split(labels, 5, mode="random", seed=7)
    b = calibration_split(labels, 5, mode="random", seed=7)
    np.testing.assert_array_equal(a[0], b[0])
    assert not set(a[0]) & set(a[1]) and len(a[0]) + len(a[1]) == 40


def test_calibration_needs_trials_left_for_testing():
    with pytest.raises(ValueError, match="need more than"):
        calibration_split(["L", "L", "R", "R"], k_per_class=2)


def test_split_hash_is_stable_and_key_order_independent():
    assert split_hash({"a": [1, 2], "b": 3}) == split_hash({"b": 3, "a": [1, 2]})
    assert split_hash({"a": [1, 2]}) != split_hash({"a": [2, 1]})
