# SPDX-License-Identifier: Apache-2.0
"""BatchNorm folding (backlog E-203), per spec §2 (`pred_proj`: Lin→BN→GELU→Lin).

Fold a **frozen** `BatchNorm1d` into the preceding `Linear` so the proven graph is
pure `Linear`s (no separate BN op). With `y = BN(W·x + b)` and frozen stats,

    s  = gamma / sqrt(running_var + eps)
    W' = W * s[:, None]
    b' = (b - running_mean) * s + beta

gives `BN(W·x + b) = W'·x + b'` exactly. Pure NumPy, deterministic — runs and is
unit-tested without torch (the torch entry point converts and calls in here).
"""
from __future__ import annotations

import numpy as np


def fold_linear_bn(weight, bias, gamma, beta, running_mean, running_var, eps: float = 1e-5):
    """Return `(W', b')` folding a frozen BatchNorm1d into the preceding Linear.

    `weight` is `[out, in]`, `bias` is `[out]`, and the BN stats are `[out]`.
    """
    weight = np.asarray(weight, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    running_mean = np.asarray(running_mean, dtype=np.float64)
    running_var = np.asarray(running_var, dtype=np.float64)

    s = gamma / np.sqrt(running_var + eps)
    w2 = weight * s[:, None]
    b2 = (bias - running_mean) * s + beta
    return w2, b2


def linear(x, weight, bias):
    """Reference `x·Wᵀ + b` (`x` is `[batch, in]` or `[in]`)."""
    return np.asarray(x, dtype=np.float64) @ np.asarray(weight, dtype=np.float64).T + np.asarray(
        bias, dtype=np.float64
    )


def linear_then_bn(x, weight, bias, gamma, beta, running_mean, running_var, eps: float = 1e-5):
    """Reference `BN(Linear(x))` with frozen stats — what the fold must reproduce."""
    y = linear(x, weight, bias)
    gamma = np.asarray(gamma, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    running_mean = np.asarray(running_mean, dtype=np.float64)
    running_var = np.asarray(running_var, dtype=np.float64)
    return gamma * (y - running_mean) / np.sqrt(running_var + eps) + beta
