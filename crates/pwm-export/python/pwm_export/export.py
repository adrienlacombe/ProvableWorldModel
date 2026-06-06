# SPDX-License-Identifier: Apache-2.0
"""le-wm checkpoint → quantized integer graph → committed manifest.

Backlog E-201 (ingest), E-202 (quantize), E-203 (BatchNorm fold). This module
**requires PyTorch and a le-wm checkpoint to run**; it is the trusted, offline
export step. The canonical-encoding / commitment side it depends on
([`pwm_export.canonical`]) has *no* torch dependency and is unit-tested for
byte-identical parity with the Rust verifier, so a real export here produces a
manifest whose commitments the Rust prover and verifier reproduce exactly.

Number policy (specs.md §3): int8 weights, power-of-two per-tensor symmetric
scales, int32 accumulators; BatchNorm folded into the preceding linear.
"""
from __future__ import annotations

import math

from . import canonical as c


def load_checkpoint(path: str):
    """E-201: load a le-wm `JEPA` checkpoint (the `_object.ckpt` form) on CPU and
    put it in eval mode (deterministic; no dropout / BN updates)."""
    import torch  # local import: torch is only needed to run the pipeline

    model = torch.load(path, map_location="cpu", weights_only=False)
    model.eval()
    return model


def quantize_tensor(w, qmax: int = 127):
    """E-202: per-tensor symmetric quantization with a power-of-two scale.

    Returns `(q_int_list, log2)` where the real value is `q * 2**log2` and `q` is
    clamped to `[-qmax-1, qmax]`. `log2` is chosen so the largest magnitude fits.
    """
    import torch

    absmax = float(w.abs().max().item())
    log2 = 0 if absmax == 0.0 else math.ceil(math.log2(absmax / qmax))
    scale = 2.0**log2
    q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax).to(torch.int64)
    return q.flatten().tolist(), log2


def fold_batchnorm(weight, bias, bn):
    """E-203: fold a (frozen) BatchNorm1d into the preceding linear's weight/bias.

    `y = bn(W x + b)` with frozen stats becomes an affine `W' x + b'`:
        s = gamma / sqrt(running_var + eps)
        W' = W * s[:, None];  b' = (b - running_mean) * s + beta
    """
    import torch

    gamma = bn.weight if getattr(bn, "affine", False) else torch.ones_like(bn.running_mean)
    beta = bn.bias if getattr(bn, "affine", False) else torch.zeros_like(bn.running_mean)
    s = gamma / torch.sqrt(bn.running_var + bn.eps)
    w2 = weight * s.unsqueeze(1)
    b2 = (bias - bn.running_mean) * s + beta
    return w2, b2


def tensor_dict(tensor_id: int, scale_id: int, q_list: list[int], shape: list[int]) -> dict:
    """Build the canonical tensor dict (cells bounded to the int8 storage range)."""
    return {
        "tensor_id": tensor_id,
        "scale_id": scale_id,
        "shape": shape,
        "data": [(v, -128, 127) for v in q_list],
    }


def build_manifest(ops: list[dict], weights: list[dict], tables: list[dict], scales: list[dict]):
    """E-205: assemble the committed manifest bytes + commitments via the canonical
    bridge (byte-identical to the Rust verifier)."""
    return {
        "model_commitment": c.model_commitment(ops, weights, 1, 1).hex(),
        "quantization_commitment": c.quantization_commitment(scales, tables).hex(),
        "graph_commitment": c.graph_commitment(ops).hex(),
        "weights_root": c.weights_root(weights).hex(),
    }


# A full `export_lewm(checkpoint_path) -> manifest + weights + golden vectors`
# driver wires load_checkpoint → per-layer quantize_tensor / fold_batchnorm →
# build_manifest, and runs the le-wm predictor in fixed point to emit golden
# vectors for the Rust parity gate (E-207). It is omitted here because it needs a
# real checkpoint to produce meaningful output; the pieces above are its building
# blocks and the canonical bridge is independently tested.
