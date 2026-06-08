# ProvableWorldModel

A **commit-and-audit** proof system for deterministic, quantized inference of a
JEPA-style world model (**LeWorldModel**). It adapts the
[CommitLLM](https://github.com/lambdaclass/CommitLLM) scheme — Freivalds checks
for the large linear layers, exact integer re-execution for everything else, all
bound by Merkle commitments and a Fiat-Shamir transcript — to prove that a
committed quantized predictor, rolled out over a fixed set of candidate action
sequences, produced the claimed latent trajectories, costs, and selected action.

It proves a precise **arithmetic relation**. It does **not** claim floating-point
PyTorch equivalence, physical truth of predictions, or zero-knowledge privacy.

> **Pivot (2026-06-05).** This project previously targeted a custom Circle-STARK
> arithmetization over the Mersenne-31 field using a vendored Stwo prover. It now
> uses the CommitLLM commit-and-audit scheme: no proving circuit, no
> arithmetization, **stable Rust toolchain** (no nightly), and a `no_std`,
> float-free verifier. The proof *target* (the quantized LeWorldModel predictor,
> rollout, and fixed-candidate planner) is unchanged. See [roadmap.md](roadmap.md).

## Start here

- [roadmap.md](roadmap.md) — the plan: pivot rationale, what carries over, phases, sequencing.
- [specs.md](specs.md) — the normative specification of the commit-and-audit design.
- [backlog.md](backlog.md) — the actionable issue backlog (milestones M0–M8).
- [docs/legacy-stark/](docs/legacy-stark/) — the archived pre-pivot STARK corpus (superseded).

## What it is

| Crate | Role |
|---|---|
| `pwm-core` | Value field (M31) + audit field (Fp61), fixed-point reference, tensors, manifest types, Merkle commitments, Fiat-Shamir transcript, Freivalds check, trace model. `no_std`, no proving substrate. |
| `pwm-export` | le-wm checkpoint → quantized integer graph → manifest + golden vectors; the Rust integer reference; the stable-worldmodel data adapter. |
| `pwm-prover` | Run the integer reference inference, build + Merkle-commit the trace, derive challenges, emit the `AuditArtifact`. |
| `pwm-verifier` | CPU, `no_std`, float-free: Freivalds-check the linear layers, exactly recompute the rest, check rollout / cost / argmin. |
| `pwm-testkit` | Golden vectors, accept/reject, and mutation-test harness. |

## How it verifies

The prover runs the model normally and commits to its execution trace. A CPU
verifier audits it:

- **Linear / matmul (fixed weights):** Freivalds — `v = rᵀW` precomputed once per
  weight matrix, then `v·x == r·z` per instance. Information-theoretically sound
  (error ≤ `N/p`, `p = 2⁶¹−1`).
- **Requant, residual, attention inner products, softmax / GELU / SiLU /
  LayerNorm tables, MSE, argmin:** exact integer recomputation — no tolerance.

Because the model runs in **exact integer fixed-point** (not bf16 on a GPU), the
attention that CommitLLM cannot verify for LLMs is here recomputed exactly: there
is no residual attention hole.

## Scope tiers

| Statement | What it proves |
|---|---|
| P0 | One quantized predictor step. |
| P1 | Autoregressive latent rollout over a horizon. |
| P2 (V0) | Fixed-candidate planning: roll out all candidates, score by goal MSE, select the argmin. |
| P3 | Full CEM planner (deferred). |
| P4 | Pixel-to-plan end to end, including the ViT encoder (deferred). |

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
