"""The donor pool for the demo: the 16 S3 development patients' covariances, re-centred per
patient, exactly as in the confirmed locked-test evaluation. Cached after the first build."""

import importlib.util

import numpy as np

from hemishift.models.pooling import recentre_by_domain
from hemishift.paths import RESULTS_DIR, ROOT

CACHE = RESULTS_DIR / "models" / "demo_donors.npz"


def load_donors():
    """(donor covariances, labels): rest 0, affected-hand MI 1."""
    if CACHE.exists():
        cached = np.load(CACHE)
        return cached["C"], cached["y"]
    spec = importlib.util.spec_from_file_location("s3_transfer", ROOT / "scripts" / "s3_transfer.py")
    s3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s3)
    patients = s3.load_patients("dev")
    C = np.concatenate([recentre_by_domain(s3.premise.covariances(X), np.zeros(len(X)))
                        for X, _ in patients.values()])
    y = np.concatenate([y for _, y in patients.values()])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, C=C, y=y)
    return C, y
