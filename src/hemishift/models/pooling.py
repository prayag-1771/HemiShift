"""Pooled Riemannian decoder: a patient's few calibration epochs plus other patients' data.

The method confirmed on the S3 locked test (ACHIEVEMENTS.md, 2026-10-24): with 5 calibration
pairs it beat the patient's own decoder by 4.3 points on never-seen patients.
- Every domain (each donor patient, and the target patient's calibration) is re-centred on
  its own Riemannian mean, so all of them sit at the identity.
- Tangent vectors are taken at the identity, and one logistic regression is fitted on
  everything.
- All donor epochs together weigh as much as the calibration epochs, so donors guide the
  decision boundary without drowning the patient.
Same computation as `scripts/s3_augment.py` (`pooled_proba`), split into fit and predict so an
online system can calibrate once and then decode trial by trial.
"""

import numpy as np
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import tangent_space
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def recentre_by_domain(C, domains):
    """Each domain's covariances whitened by that domain's own Riemannian mean."""
    out = np.empty_like(C)
    for d in np.unique(domains):
        rows = domains == d
        W = invsqrtm(mean_riemann(C[rows]))
        out[rows] = W @ C[rows] @ W
    return out


class OwnTSLR:
    """The patient's own decoder: calibration-mean re-centring, tangent space, LR (as in the
    experiments' `own_proba`)."""

    def fit(self, C_cal, y_cal):
        self.W_ = invsqrtm(mean_riemann(C_cal))
        self.model_ = make_pipeline(TangentSpace(metric="riemann"),
                                    LogisticRegression(C=1.0, max_iter=1000))
        self.model_.fit(self.W_ @ C_cal @ self.W_, y_cal)
        return self

    def predict_proba(self, C):
        return self.model_.predict_proba(self.W_ @ np.asarray(C) @ self.W_)[:, 1]


class PooledTSLR:
    """Tangent-space logistic regression on calibration + re-centred donor covariances."""

    def __init__(self, donor_C, donor_y):
        self.donor_C = np.asarray(donor_C)  # already re-centred per donor domain
        self.donor_y = np.asarray(donor_y)

    def fit(self, C_cal, y_cal):
        self.W_ = invsqrtm(mean_riemann(C_cal))
        identity = np.eye(C_cal.shape[-1])
        X = tangent_space(np.concatenate([self.W_ @ C_cal @ self.W_, self.donor_C]), identity,
                          metric="riemann")
        y = np.concatenate([y_cal, self.donor_y])
        w = np.concatenate([np.ones(len(y_cal)),
                            np.full(len(self.donor_y), len(y_cal) / len(self.donor_y))])
        self.model_ = LogisticRegression(C=1.0, max_iter=1000).fit(X, y, sample_weight=w)
        return self

    def predict_proba(self, C):
        """P(class 1) per covariance matrix, re-centred with the calibration mean."""
        C = np.asarray(C)
        identity = np.eye(C.shape[-1])
        return self.model_.predict_proba(
            tangent_space(self.W_ @ C @ self.W_, identity, metric="riemann"))[:, 1]
