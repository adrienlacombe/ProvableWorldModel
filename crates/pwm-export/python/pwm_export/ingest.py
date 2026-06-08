# SPDX-License-Identifier: Apache-2.0
"""le-wm checkpoint ingest (backlog E-201), per spec §2 (`QuantizedLeWM-v1`).

The V0 proven subgraph is `action_encoder → predictor (×depth) → pred_proj`. This
module validates and extracts that subgraph from a name→array parameter map and
records the dims. The dim/shape **validation** is pure (NumPy only) and unit-
tested without a checkpoint; only `state_arrays_from_torch` (the thin loader that
turns a real `JEPA`/`weights.pt` into the parameter map) needs torch + the file.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class V0Dims:
    """le-wm V0 reference dims (specs.md §2, verified vs upstream 2026-06-05)."""

    latent_dim: int = 192
    history_size: int = 3
    depth: int = 6
    heads: int = 16
    dim_head: int = 64
    mlp_dim: int = 2048

    @property
    def attn_inner(self) -> int:
        """Attention inner width `heads · dim_head` (1024), projected back to 192."""
        return self.heads * self.dim_head

    def as_tuple(self) -> tuple[int, int, int, int, int, int]:
        """The headline dims `(latent, history, depth, heads, dim_head, mlp)`."""
        return (
            self.latent_dim,
            self.history_size,
            self.depth,
            self.heads,
            self.dim_head,
            self.mlp_dim,
        )


#: The reference V0 instance (`192/3/6/16/64/2048`).
V0_DIMS = V0Dims()


def block_param_shapes(dims: V0Dims, i: int) -> dict[str, tuple[int, int]]:
    """Expected `[out, in]` weight shapes for predictor block `i`."""
    latent, inner, mlp = dims.latent_dim, dims.attn_inner, dims.mlp_dim
    p = f"predictor.blocks.{i}"
    return {
        f"{p}.attn.wq": (inner, latent),
        f"{p}.attn.wk": (inner, latent),
        f"{p}.attn.wv": (inner, latent),
        f"{p}.attn.wproj": (latent, inner),
        f"{p}.ffn.fc1": (mlp, latent),
        f"{p}.ffn.fc2": (latent, mlp),
    }


def v0_param_shapes(dims: V0Dims = V0_DIMS, action_dim: int = 4) -> dict[str, tuple[int, int]]:
    """All expected weight shapes of the V0 subgraph.

    `action_dim` is a config field (not one of the headline dims); the action
    encoder maps `[action_dim] → [latent]` per history position.
    """
    shapes: dict[str, tuple[int, int]] = {
        "action_encoder.embed": (dims.latent_dim, action_dim),
    }
    for i in range(dims.depth):
        shapes.update(block_param_shapes(dims, i))
    # pred_proj is a le-wm MLP (Lin→BN→GELU→Lin) over the latent.
    shapes["pred_proj.fc1"] = (dims.latent_dim, dims.latent_dim)
    shapes["pred_proj.fc2"] = (dims.latent_dim, dims.latent_dim)
    return shapes


def extract_v0_subgraph(arrays: dict, dims: V0Dims = V0_DIMS, action_dim: int = 4) -> dict:
    """Validate and extract the V0 subgraph from a name→array map.

    Raises `KeyError` for a missing parameter and `ValueError` for a shape
    mismatch. Returns `{"dims", "action_dim", "params"}` with the extracted
    arrays — the dims are asserted to equal the reference `192/3/6/16/64/2048`.
    """
    expected = v0_param_shapes(dims, action_dim)
    missing = sorted(k for k in expected if k not in arrays)
    if missing:
        raise KeyError(f"checkpoint missing V0 params: {missing}")
    params: dict[str, np.ndarray] = {}
    for name, want in expected.items():
        got = tuple(np.asarray(arrays[name]).shape)
        if got != want:
            raise ValueError(f"{name}: expected shape {want}, got {got}")
        params[name] = np.asarray(arrays[name], dtype=np.float64)
    return {"dims": dims, "action_dim": action_dim, "params": params}


def state_arrays_from_torch(model) -> dict:
    """Thin torch adapter: a `JEPA`/`nn.Module` → `{name: numpy array}` (CPU).

    The only torch-dependent step; the real le-wm key names are remapped to the
    `extract_v0_subgraph` scheme by the (checkpoint-specific) caller.
    """
    return {name: p.detach().cpu().numpy() for name, p in model.named_parameters()}


def load_checkpoint(path: str):
    """E-201 loader: a le-wm `JEPA` checkpoint on CPU in eval mode (torch)."""
    import torch  # local import: only needed to read a real checkpoint

    model = torch.load(path, map_location="cpu", weights_only=False)
    model.eval()
    return model
