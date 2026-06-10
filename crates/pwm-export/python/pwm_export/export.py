# SPDX-License-Identifier: Apache-2.0
"""le-wm checkpoint → quantized integer graph → committed manifest (driver).

Ties the pipeline together: ingest (E-201, [`pwm_export.ingest`]) → BatchNorm fold
(E-203, [`pwm_export.fold`]) → quantization (E-202, [`pwm_export.quantize`]) →
manifest commitments (E-205, via the torch-free [`pwm_export.canonical`] bridge,
byte-identical to the Rust verifier).

Everything except `load_checkpoint`/`quantize_torch` runs on plain NumPy arrays —
so `export_graph` is exercised end-to-end on a synthetic le-wm-shaped graph in the
tests, with no torch or checkpoint required. A real export only swaps the array
source (a loaded checkpoint) for the synthetic one.
"""
from __future__ import annotations

from typing import Any

from . import canonical as c


def tensor_dict(tensor_id: int, scale_id: int, q_list: list[int], shape: list[int]) -> dict:
    """Build the canonical tensor dict (cells bounded to the int8 storage range)."""
    return {
        "tensor_id": tensor_id,
        "scale_id": scale_id,
        "shape": shape,
        "data": [(v, -128, 127) for v in q_list],
    }


def predictor_weight_dicts(
    blocks: list[dict], d: int, heads: int, dim_head: int, mlp: int
) -> list[dict]:
    """The predictor-bundle weight tensors in the Rust prover's canonical scheme.

    Mirrors `lewm_predictor.rs` exactly: per block `i` the tensor ids are
    `1000 + 5*i ..` in registration order `adaln, qkv, out, fc1, fc2`.
    Weight tensor `k` uses `scale_id = 10 + k`, matching the Rust builder and
    binding the per-tensor `log2` interpretation into the quantization
    commitment. `weights_root` over this list is the commitment the bundle
    carries and the prover must reproduce bit-for-bit (pinned cross-language in
    the canonical-parity tests).
    """
    inner = heads * dim_head
    out = []
    for i, bk in enumerate(blocks):
        base = 1000 + 5 * i
        for off, (key, shape) in enumerate((
            ("adaln", [6 * d, d]),
            ("qkv", [3 * inner, d]),
            ("out", [d, inner]),
            ("fc1", [mlp, d]),
            ("fc2", [d, mlp]),
        )):
            k = 5 * i + off
            out.append(tensor_dict(base + off, 10 + k, bk[key], shape))
    return out


def quantize_torch(w, qmax: int = 127) -> tuple[list[int], int]:
    """Torch entry point for E-202: convert a tensor to an array and quantize."""
    import numpy as np

    from .quantize import quantize_array

    return quantize_array(np.asarray(w.detach().cpu().numpy()), qmax)


def fold_torch(weight, bias, bn):
    """Torch entry point for E-203: pull frozen BN stats and fold (NumPy core)."""
    import torch

    from .fold import fold_linear_bn

    affine = getattr(bn, "affine", False)
    gamma = bn.weight if affine else torch.ones_like(bn.running_mean)
    beta = bn.bias if affine else torch.zeros_like(bn.running_mean)
    return fold_linear_bn(
        weight.detach().cpu().numpy(),
        bias.detach().cpu().numpy(),
        gamma.detach().cpu().numpy(),
        beta.detach().cpu().numpy(),
        bn.running_mean.detach().cpu().numpy(),
        bn.running_var.detach().cpu().numpy(),
        float(bn.eps),
    )


def quantize_linear(tensor_id: int, scale_id: int, weight) -> tuple[dict, dict]:
    """Quantize one `[out, in]` Linear weight → `(tensor_dict, scale_dict)`.

    Asserts the int8 MAC for this layer fits the int32 accumulator (spec §3).
    """
    import numpy as np

    from .quantize import mac_fits_int32, quantize_array

    weight = np.asarray(weight, dtype=np.float64)
    out, inner = weight.shape
    if not mac_fits_int32(inner):
        raise ValueError(f"layer {tensor_id}: inner={inner} overflows int32 MAC")
    q, log2 = quantize_array(weight)
    tensor = tensor_dict(tensor_id, scale_id, q, [out, inner])
    scale = {"scale_id": scale_id, "log2": log2, "dtype": "i8"}
    return tensor, scale


def build_manifest(ops: list[dict], weights: list[dict], tables: list[dict], scales: list[dict]):
    """E-205: assemble the committed manifest commitments via the canonical bridge."""
    return {
        "model_commitment": c.model_commitment(ops, weights, 1, 1).hex(),
        "quantization_commitment": c.quantization_commitment(scales, tables).hex(),
        "graph_commitment": c.graph_commitment(ops).hex(),
        "weights_root": c.weights_root(weights).hex(),
    }


def export_graph(named_weights: list[tuple[int, str, Any]], tables: list[dict]) -> dict:
    """Quantize a list of `(op_id, name, weight)` Linears into a committed manifest.

    Returns `{"manifest", "weights", "ops", "scales"}` — the manifest carries the
    same commitments the Rust prover/verifier reproduce (E-205/E-207).
    """
    import numpy as np

    ops: list[dict] = []
    weights: list[dict] = []
    scales: list[dict] = []
    for idx, (op_id, _name, w) in enumerate(named_weights):
        out, inner = np.asarray(w).shape
        tensor, scale = quantize_linear(tensor_id=idx, scale_id=idx, weight=w)
        weights.append(tensor)
        scales.append(scale)
        ops.append(
            {
                "kind": "linear",
                "op_id": op_id,
                "weight_id": idx,
                "bias_id": None,
                "rows": int(out),
                "cols": int(inner),
            }
        )
    return {
        "manifest": build_manifest(ops, weights, tables, scales),
        "weights": weights,
        "ops": ops,
        "scales": scales,
    }


# A real `export_lewm(path)` = `ingest.load_checkpoint` → `ingest.state_arrays_from_torch`
# → `ingest.extract_v0_subgraph` → `fold_torch` (pred_proj BN) → `export_graph`.
# Each piece is tested on synthetic NumPy arrays; only the checkpoint read needs
# torch + the real file.
