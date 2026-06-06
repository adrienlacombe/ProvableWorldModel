# SPDX-License-Identifier: Apache-2.0
"""stable-worldmodel data adapter (backlog E-208).

Produces the inputs a proof binds to: `(z_history, action, z_next)` transitions,
goal latents, and the fixed candidate action set for P2 planning. **Requires
`stable-worldmodel` + a trained le-wm checkpoint to run.** The encoder is frozen,
so latents are computed once, offline, and become committed public inputs; the
prover/verifier never touch this module.
"""
from __future__ import annotations


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


# A `prepare_p2_inputs(dataset, model, num_candidates) -> (z_history, goal,
# candidate_actions)` driver collects an initial history + goal latent and a fixed
# candidate set, quantizes the latents to the manifest's input scale, and writes
# them as committed public inputs for `pwm_prover::prove_planning`. It is omitted
# here because it needs the swm dataset + checkpoint to produce real tensors.
