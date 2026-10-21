"""Compact deep decoders: EEGNet with Euclidean alignment and cropped training.

- Euclidean alignment (He & Wu 2020) whitens each subject's trials by the mean covariance of
  a reference set, so data from different people share one coordinate frame.
- Cropped training: every epoch draws a random `crop`-sample window from each trial. That
  multiplies the effective data, and prediction averages over sliding windows.
"""

import numpy as np
import torch
from torch import nn

from hemishift.paths import device


def inverse_sqrt(cov, rtol=1e-6):
    """Inverse matrix square root that leaves out (near-)null directions.

    Common-average-referenced data has a direction with no variance (the channel sum). A plain
    inverse square root scales it by about 1e6, which turns float rounding, or any later edit
    that breaks the zero sum (e.g. a LesionSim copy), into huge values. Directions with variance
    below `rtol` times the largest are set to zero instead.
    """
    values, vectors = np.linalg.eigh(cov)
    keep = values > rtol * values.max()
    scale = np.zeros_like(values)
    scale[keep] = 1.0 / np.sqrt(values[keep])
    return (vectors * scale) @ vectors.T


def euclidean_alignment(X, reference):
    """Whiten every trial of X (n, channels, times) by the reference trials' mean covariance."""
    Xr = np.asarray(X[reference], dtype=float)  # float64, so the null direction stays at ~0
    covs = np.einsum("nct,ndt->ncd", Xr, Xr) / X.shape[-1]
    return np.einsum("cd,ndt->nct", inverse_sqrt(covs.mean(axis=0)), X).astype(np.float32)


def make_eegnet(n_chans, n_times, n_outputs=2, seed=0):
    from braindecode.models import EEGNet

    torch.manual_seed(seed)
    return EEGNet(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times)


def random_crops(X, crop, rng):
    starts = rng.integers(0, X.shape[-1] - crop + 1, size=len(X))
    return np.stack([trial[:, s:s + crop] for trial, s in zip(X, starts)])


def train(model, X, y, crop, epochs, lr, batch=64, weight_decay=1e-2, seed=0):
    """Train in place on random crops; y holds integer class ids."""
    run_on = device()
    model.to(run_on).train()
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    targets = torch.as_tensor(y, dtype=torch.long)
    for _ in range(epochs):
        order = rng.permutation(len(X))
        for start in range(0, len(X), batch):
            idx = order[start:start + batch]
            inputs = torch.as_tensor(random_crops(X[idx], crop, rng)).to(run_on)
            loss = loss_fn(model(inputs), targets[idx].to(run_on))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    return model


@torch.no_grad()
def predict_proba(model, X, crop, stride=16):
    """Class probabilities per trial, averaged over sliding `crop`-sample windows."""
    run_on = device()
    model.to(run_on).eval()
    starts = range(0, X.shape[-1] - crop + 1, stride)
    total = 0.0
    for s in starts:
        window = torch.as_tensor(np.ascontiguousarray(X[..., s:s + crop])).to(run_on)
        total = total + torch.softmax(model(window), dim=1).cpu().numpy()
    return total / len(starts)
