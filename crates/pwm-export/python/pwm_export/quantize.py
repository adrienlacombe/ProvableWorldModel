# SPDX-License-Identifier: Apache-2.0
"""Quantization pass (backlog E-202), per spec §3.

Per-tensor **symmetric** quantization with a **power-of-two** scale: a real value
`x` maps to `q = clamp(round(x / 2**log2), -qmax-1, qmax)` so the dequantized
value is `q * 2**log2`. int8 weights/activations, int32 accumulators; the int8
MLP MAC bound `2048·127·127 = 33,032,192 < 2³¹−1` keeps fc1/fc2 in one int32.

Pure NumPy — no torch — so it runs and is unit-tested without a checkpoint. The
torch entry point in [`pwm_export.export`] just converts tensors to arrays and
calls in here.
"""
from __future__ import annotations

import math

import numpy as np

#: int32 accumulator ceiling (exclusive); accumulators must stay below this.
INT32_MAX = 2**31 - 1


def pow2_log2(absmax: float, qmax: int = 127) -> int:
    """The smallest `log2` such that `absmax / 2**log2 <= qmax` (0 for all-zero)."""
    if absmax == 0.0:
        return 0
    return math.ceil(math.log2(absmax / qmax))


def quantize_array(w, qmax: int = 127) -> tuple[list[int], int]:
    """Quantize an array to `(q_int_list, log2)` with a power-of-two scale.

    `q` is row-major flattened and clamped to the signed `[-qmax-1, qmax]` storage
    range (int8 when `qmax=127`). The dequantized value is `q * 2**log2`.
    """
    w = np.asarray(w, dtype=np.float64)
    absmax = float(np.abs(w).max()) if w.size else 0.0
    log2 = pow2_log2(absmax, qmax)
    scale = 2.0**log2
    q = np.clip(np.round(w / scale), -qmax - 1, qmax).astype(np.int64)
    return q.flatten().tolist(), log2


def dequantize(q: list[int], log2: int) -> np.ndarray:
    """Inverse of [`quantize_array`] (for error checking / golden vectors)."""
    return np.asarray(q, dtype=np.float64) * (2.0**log2)


def mac_bound(inner: int, qmax: int = 127) -> int:
    """Worst-case |accumulator| for an `inner`-length int8·int8 dot product."""
    return inner * qmax * qmax


def mac_fits_int32(inner: int, qmax: int = 127) -> bool:
    """Whether an `inner`-length int8 MAC stays within the int32 accumulator."""
    return mac_bound(inner, qmax) < INT32_MAX
