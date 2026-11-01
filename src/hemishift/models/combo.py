"""Decoders of the combination study (scripts/combo_explore.py; exploratory).

All of them are re-centred Riemannian tangent space + logistic regression, as in
`hemishift.models.pooling`:
- `static`: fitted once on the calibration, with optional donor groups pooled in (each group
  together weighs as much as the patient's labelled trials), a population prior, or a
  donor-only decoder that uses no patient labels.
- `online`: refitted before every test trial on the calibration and every earlier test
  trial (their cue labels, or the calibration labels only), re-centred on all of them.
- `map_lr`: logistic regression with a Gaussian prior on its weights. The study takes the
  prior from the other patients' own decoders (empirical Bayes), instead of pooling their
  trials.
- `crops_static`: covariances of sub-windows as extra samples.
- `bma`: Bayesian model averaging over decoders, weighted by their likelihood on the trials
  decoded so far.
`own_exact` and `pooled_exact` reproduce the confirmed baselines (s3_transfer.own_proba and
s3_augment.pooled_proba) from 8-30 Hz covariances.
"""

import numpy as np
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import tangent_space
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def lr(warm=False):
    return LogisticRegression(C=1.0, max_iter=1000, warm_start=warm)


def centred(C):
    """C whitened by its own Riemannian mean (a donor domain)."""
    W = invsqrtm(mean_riemann(C))
    return W @ C @ W


def tangent_all(C):
    """Tangent vectors at the identity of a domain re-centred on its own mean."""
    return tangent_space(centred(C), np.eye(C.shape[-1]), metric="riemann")


def recentred(C, reference, init=None):
    """Tangent vectors at the identity after whitening by the mean of C[reference]; also
    returns that mean."""
    M = mean_riemann(C[reference], init=init)
    W = invsqrtm(M)
    return tangent_space(W @ C @ W, np.eye(C.shape[-1]), metric="riemann"), M


def stack(T, y, groups):
    """Patient rows weigh 1; each donor group together weighs as much as the patient rows."""
    X, Y, w = [T], [y], [np.ones(len(y))]
    for T_d, y_d in groups:
        X.append(T_d)
        Y.append(y_d)
        w.append(np.full(len(y_d), len(y) / len(y_d)))
    return np.vstack(X), np.concatenate(Y), np.concatenate(w)


def map_lr(T, y, prior, start=None):
    """Logistic regression with a Gaussian prior N(mean, diag(var)) on [weights, intercept];
    returns the MAP [weights, intercept]."""
    mean, var = prior
    X = np.hstack([T, np.ones((len(T), 1))])

    def objective(w):
        z = X @ w
        return (np.sum(np.logaddexp(0, z) - y * z) + 0.5 * np.sum((w - mean) ** 2 / var),
                X.T @ (expit(z) - y) + (w - mean) / var)

    return minimize(objective, mean if start is None else start, jac=True, method="L-BFGS-B").x


def map_proba(w, T):
    return expit(T @ w[:-1] + w[-1])


def population_prior(weights, floor=1e-4):
    """Mean and per-coordinate variance (plus a floor) of other patients' [weights, intercept]."""
    W = np.asarray(weights)
    return W.mean(axis=0), W.var(axis=0, ddof=1) + floor


def static(cov, y, cal, test, views, groups=(), prior=None, zero=None):
    """P(MI) of the test trials from a decoder fixed after calibration. cov: view -> (n, c, c);
    the views' tangent vectors sit side by side. groups: donor groups (T, y) pooled with the
    calibration; prior: MAP with this prior instead; zero: a decoder trained on these donors
    (T, y) only, no patient labels used."""
    T = np.hstack([recentred(cov[v], cal)[0] for v in views])
    if zero is not None:
        return lr().fit(*zero).predict_proba(T[test])[:, 1]
    if prior is not None:
        return map_proba(map_lr(T[cal], y[cal], prior), T[test])
    X, Y, w = stack(T[cal], y[cal], groups)
    return lr().fit(X, Y, sample_weight=w).predict_proba(T[test])[:, 1]


def own_exact(C, y, cal, test):
    """s3_transfer.own_proba from 8-30 Hz covariances."""
    W = invsqrtm(mean_riemann(C[cal]))
    A = W @ C @ W
    model = make_pipeline(TangentSpace(metric="riemann"), LogisticRegression(C=1.0, max_iter=1000))
    return model.fit(A[cal], y[cal]).predict_proba(A[test])[:, 1]


def pooled_exact(C, y, cal, test, donor_C, donor_y):
    """s3_augment.pooled_proba: donor_C already re-centred per donor."""
    W = invsqrtm(mean_riemann(C[cal]))
    identity = np.eye(C.shape[-1])
    X = tangent_space(np.concatenate([W @ C[cal] @ W, donor_C]), identity, metric="riemann")
    Y = np.concatenate([y[cal], donor_y])
    w = np.concatenate([np.ones(len(cal)), np.full(len(donor_y), len(cal) / len(donor_y))])
    model = LogisticRegression(C=1.0, max_iter=1000).fit(X, Y, sample_weight=w)
    return model.predict_proba(tangent_space(W @ C[test] @ W, identity, metric="riemann"))[:, 1]


def crop_features(crops, reference=None):
    """crops: one list per band of (n, m, c, c) sub-window covariances. Each band is re-centred
    on the mean over the reference trials' sub-windows (all trials if None); returns
    (n, m, features)."""
    blocks = []
    for C in crops:
        n, m = C.shape[:2]
        rows = np.arange(n) if reference is None else reference
        W = invsqrtm(mean_riemann(C[rows].reshape(len(rows) * m, *C.shape[2:])))
        flat = C.reshape(n * m, *C.shape[2:])
        blocks.append(tangent_space(W @ flat @ W, np.eye(C.shape[-1]),
                                    metric="riemann").reshape(n, m, -1))
    return np.concatenate(blocks, axis=2)


def crops_static(crops, y, cal, test, donors):
    """Pooled decoder on sub-window covariances: every sub-window of a calibration epoch weighs
    1/m, the donors (T (n, m, f), y) together as much as the calibration; the test
    probabilities are averaged over each epoch's sub-windows."""
    T = crop_features(crops, cal)
    m = T.shape[1]
    T_d, y_d = donors
    X = np.vstack([T[cal].reshape(len(cal) * m, -1), T_d.reshape(len(T_d) * m, -1)])
    Y = np.concatenate([np.repeat(y[cal], m), np.repeat(y_d, m)])
    w = np.concatenate([np.full(len(cal) * m, 1 / m),
                        np.full(len(y_d) * m, len(cal) / (len(y_d) * m))])
    model = lr().fit(X, Y, sample_weight=w)
    p = model.predict_proba(T[test].reshape(len(test) * m, -1))[:, 1]
    return p.reshape(len(test), m).mean(axis=1)


def online(cov, y, cal, test, views, labels="all", groups=(), prior=None, zero=None):
    """P(MI) of each test trial in recording order, from a decoder refitted just before it.
    labels: "all" (the calibration and every earlier test trial, by their cue labels) or
    "cal" (the calibration only). The patient is re-centred on the calibration and every
    earlier test trial. Donor groups weigh as much as the labelled trials. zero: a fixed
    decoder trained on these donors (T, y); only the re-centring adapts."""
    fixed = lr().fit(*zero) if zero is not None else None
    model, w_map = lr(warm=True), None
    means = [None] * len(views)
    p = np.empty(len(test))
    for i, trial in enumerate(test):
        ref = np.concatenate([cal, test[:i]])
        rows = np.append(ref, trial)
        blocks = []
        for j, v in enumerate(views):
            means[j] = mean_riemann(cov[v][ref], init=means[j])
            W = invsqrtm(means[j])
            blocks.append(tangent_space(W @ cov[v][rows] @ W, np.eye(cov[v].shape[-1]),
                                        metric="riemann"))
        T = np.hstack(blocks)
        if fixed is not None:
            p[i] = fixed.predict_proba(T[-1:])[0, 1]
            continue
        lab = np.arange(len(ref)) if labels == "all" else np.arange(len(cal))
        if prior is not None:
            w_map = map_lr(T[lab], y[ref][lab], prior, w_map)
            p[i] = map_proba(w_map, T[-1:])[0]
            continue
        X, Y, w = stack(T[lab], y[ref][lab], groups)
        p[i] = model.fit(X, Y, sample_weight=w).predict_proba(T[-1:])[0, 1]
    return p


def bma(probs, y):
    """Bayesian model averaging: each test trial's P(MI) is the decoders' mean weighted by
    their likelihood on the earlier test trials (whose labels arrive after each decision)."""
    P = np.clip(np.vstack(probs), 1e-6, 1 - 1e-6)
    loglik = np.zeros(len(P))
    out = np.empty(P.shape[1])
    for i in range(P.shape[1]):
        weights = np.exp(loglik - loglik.max())
        out[i] = weights @ P[:, i] / weights.sum()
        loglik += np.log(P[:, i] if y[i] == 1 else 1 - P[:, i])
    return out
