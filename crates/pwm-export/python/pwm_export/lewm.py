# SPDX-License-Identifier: Apache-2.0
"""le-wm-specific ingest adapter (backlog E-201, real model).

Maps the **actual** le-wm module layout (upstream `module.py`: `ARPredictor` →
`Transformer` → `ConditionalBlock`, `Embedder`, `MLP` pred_proj) to the V0
subgraph this project proves, and validates the spec §2 dims. Generic shape
validation lives in [`pwm_export.ingest`]; this file knows the real `named_*` keys.

Operates on a `{name: numpy array}` map (the caller converts tensors via
`ingest.state_arrays_from_torch`), so it has no torch dependency itself.
"""
from __future__ import annotations

import numpy as np

from .ingest import V0_DIMS, V0Dims


def conditional_block_linears(dims: V0Dims, i: int) -> dict[str, tuple[int, int]]:
    """Expected `[out, in]` shapes of the Freivalds linears in predictor block `i`
    (`ConditionalBlock`)."""
    latent, inner, mlp = dims.latent_dim, dims.attn_inner, dims.mlp_dim
    p = f"predictor.transformer.layers.{i}"
    return {
        f"{p}.attn.to_qkv.weight": (3 * inner, latent),   # fused Q,K,V
        f"{p}.attn.to_out.0.weight": (latent, inner),     # attention out-proj
        f"{p}.mlp.net.1.weight": (mlp, latent),           # FFN fc1
        f"{p}.mlp.net.4.weight": (latent, mlp),           # FFN fc2
        f"{p}.adaLN_modulation.1.weight": (6 * latent, latent),  # AdaLN-zero
    }


def v0_linear_shapes(dims: V0Dims = V0_DIMS) -> dict[str, tuple[int, int]]:
    """Every Freivalds-linear weight of the real le-wm V0 subgraph, with shapes."""
    latent, mlp = dims.latent_dim, dims.mlp_dim
    shapes: dict[str, tuple[int, int]] = {}
    # action encoder (Embedder): SiLU MLP after the kernel-1 conv.
    shapes["action_encoder.embed.0.weight"] = (4 * latent, 10)
    shapes["action_encoder.embed.2.weight"] = (latent, 4 * latent)
    # predictor blocks.
    for i in range(dims.depth):
        shapes.update(conditional_block_linears(dims, i))
    # pred_proj MLP (Linear → BatchNorm1d → GELU → Linear).
    shapes["pred_proj.net.0.weight"] = (mlp, latent)
    shapes["pred_proj.net.3.weight"] = (latent, mlp)
    return shapes


def extract_v0_linears(named_params: dict, dims: V0Dims = V0_DIMS) -> list[tuple[str, np.ndarray]]:
    """Validate the real le-wm V0 dims and return its Freivalds-linear weights.

    `named_params` is `{name: array}` from a loaded le-wm checkpoint. Raises on a
    missing weight or a shape that disagrees with the spec §2 reference dims
    (192/3/6/16/64/2048). The returned list is the input to
    [`pwm_export.export.export_graph`].
    """
    expected = v0_linear_shapes(dims)
    out: list[tuple[str, np.ndarray]] = []
    for name, want in expected.items():
        if name not in named_params:
            raise KeyError(f"le-wm checkpoint missing V0 weight: {name}")
        arr = np.asarray(named_params[name])
        if arr.shape != want:
            raise ValueError(f"{name}: expected {want}, got {tuple(arr.shape)}")
        out.append((name, arr))
    return out
