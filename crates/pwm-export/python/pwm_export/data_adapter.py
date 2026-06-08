# SPDX-License-Identifier: Apache-2.0
"""stable-worldmodel data adapter (backlog E-208).

Produces the inputs a proof binds to: `(z_history, action, z_next)` transitions,
goal latents, and the fixed candidate action set for P2 planning. The
stable-worldmodel loaders + frozen encoder (`load_transitions`/`encode_latents`/
`candidate_set`) need swm + a checkpoint to run, but the bundle assembly
(`quantize_latents`/`build_p2_bundle`) is pure NumPy and unit-tested — it turns
already-encoded latents into the committed, prover-consumable public-input bundle.
The encoder is frozen, so latents are computed once, offline; the prover/verifier
never touch this module.
"""
from __future__ import annotations

import numpy as np

from .quantize import quantize_array


def load_transitions(dataset_path: str, num_steps: int = 2, frameskip: int = 1):
    """Load `(obs, action, next_obs)` clips via stable-worldmodel.

    `num_steps=2` yields one transition per item: `pixels` is `(2, C, H, W)` and
    `action` is `(2, A)` (see the stable-worldmodel dataset contract).
    """
    import stable_worldmodel as swm  # local import; only needed to run

    return swm.data.load_dataset(
        dataset_path,
        num_steps=num_steps,
        frameskip=frameskip,
        keys_to_load=["pixels", "action", "proprio"],
    )


def encode_latents(model, info: dict) -> dict:
    """Encode observations (and actions) to latents with the frozen le-wm encoder.

    Returns `info` with `emb` (and `act_emb`) added; the encoder is `.eval()` and
    detached, so this is deterministic and is the committed latent input.
    """
    return model.encode(info)


def candidate_set(model, info: dict, candidate_actions):
    """Score a fixed candidate action set — the exact `get_cost` call (B, S) that a
    P2 planning proof attests. `candidate_actions` is `(B, S, T, action_dim)`.
    """
    return model.get_cost(info, candidate_actions)


def quantize_latents(latents, qmax: int = 127) -> tuple[list[int], int]:
    """Quantize encoded latents to the int8 input scale (same policy as weights)."""
    return quantize_array(latents, qmax)


def build_p2_bundle(z_history, action, z_next, goal, candidate_actions) -> dict:
    """Assemble the committed P2 public-input bundle from encoded latents.

    `z_history` is `[history_size, latent]`, `action` `[history_size, action_dim]`,
    `z_next`/`goal` `[latent]`, and `candidate_actions` `[S, T, action_dim]` — the
    **fixed candidate set** P2 attests. Latents are quantized to int8; the bundle
    is consumable by `pwm_prover::prove_planning` (shapes recorded, values bounded).
    """
    zh = np.asarray(z_history, dtype=np.float64)
    cand = np.asarray(candidate_actions, dtype=np.float64)
    zh_q, zh_log2 = quantize_latents(zh)
    zn_q, zn_log2 = quantize_latents(z_next)
    goal_q, goal_log2 = quantize_latents(goal)
    return {
        "history_size": int(zh.shape[0]),
        "latent_dim": int(zh.shape[1]),
        "num_candidates": int(cand.shape[0]),
        "horizon": int(cand.shape[1]),
        "z_history": {"data": zh_q, "log2": zh_log2},
        "z_next": {"data": zn_q, "log2": zn_log2},
        "goal": {"data": goal_q, "log2": goal_log2},
        "action": np.asarray(action, dtype=np.float64).flatten().tolist(),
        "candidate_actions": cand.reshape(cand.shape[0], -1).tolist(),
    }


# A `prepare_p2_inputs(dataset, model, num_candidates)` driver = `load_transitions`
# → `encode_latents` → `candidate_set` → `build_p2_bundle`; only the first three
# need swm + a checkpoint. `build_p2_bundle` is exercised on synthetic latents in
# the tests and emits the committed public inputs for `pwm_prover::prove_planning`.
