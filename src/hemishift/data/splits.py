"""Leakage-safe splits (PLAN.md §7): patients, sites and calibration trials never cross."""

import hashlib
import json

import numpy as np


def patient_folds(patient_ids, n_folds=5, seed=0):
    """Map each patient to a fold in 0..n_folds-1 (shuffled, sizes differ by at most one)."""
    ids = sorted(set(patient_ids))
    order = np.random.default_rng(seed).permutation(len(ids))
    return {ids[i]: int(rank % n_folds) for rank, i in enumerate(order)}


def leave_site_out(sites):
    """Yield (held_out_site, training_sites) once per site (protocol P4)."""
    unique = sorted(set(sites))
    for held_out in unique:
        yield held_out, [site for site in unique if site != held_out]


def calibration_split(labels, k_per_class, mode="first", seed=0):
    """Split one patient's trials into k calibration trials per class and the rest for testing.

    `labels` must be in recording order. mode="first" takes the first k trials of each class,
    as a real calibration session would; this is the primary protocol, because randomly drawn
    calibration trials sit next to test trials in time and flatter the decoder.
    mode="random" draws them at random, for sensitivity analysis only.
    Returns (calibration_indices, test_indices), both sorted.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    calibration = []
    for cls in np.unique(labels):
        indices = np.flatnonzero(labels == cls)
        if indices.size <= k_per_class:
            raise ValueError(f"class {cls!r} has {indices.size} trials; need more than k={k_per_class}")
        if mode == "first":
            calibration.extend(indices[:k_per_class])
        elif mode == "random":
            calibration.extend(rng.choice(indices, size=k_per_class, replace=False))
        else:
            raise ValueError(f"mode must be 'first' or 'random', got {mode!r}")
    calibration = np.sort(np.asarray(calibration, dtype=int))
    return calibration, np.setdiff1d(np.arange(labels.size), calibration)


def locked_test_split(table, id_col, strata_cols, n_test, seed=0):
    """Hold out `n_test` subjects for a final test that is never used during development.

    Strata are filled proportionally (largest remainder), so each held-out patient is drawn
    from a stratum (e.g. paralysed side x severity band) at its share of the cohort. Returns
    (dev_ids, test_ids), both sorted.
    """
    rng = np.random.default_rng(seed)
    strata = table.groupby(list(strata_cols))[id_col].apply(list)
    quota = strata.map(len) / len(table) * n_test
    take = np.floor(quota).astype(int)
    for key in (quota - take).sort_values(ascending=False).index[: n_test - take.sum()]:
        take[key] += 1
    test = [pid for key, ids in strata.items() for pid in rng.choice(ids, size=take[key], replace=False)]
    dev = sorted(set(table[id_col]) - set(test))
    return dev, sorted(test)


def split_hash(split):
    """Short, stable fingerprint of a split; recorded next to every reported number."""
    payload = json.dumps(split, sort_keys=True, default=str).encode()
    return hashlib.sha1(payload).hexdigest()[:10]
