# ProvableWorldModel

[![CI](https://github.com/AbdelStark/ProvableWorldModel/actions/workflows/ci.yml/badge.svg)](https://github.com/AbdelStark/ProvableWorldModel/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-stable%20(MSRV%201.85)-orange.svg)](rust-toolchain.toml)
[![verifier](https://img.shields.io/badge/verifier-no__std%20%C2%B7%20float--free-5eead4.svg)](crates/pwm-verifier)

Cryptographic proof that a quantized world model planned exactly as claimed.

ProvableWorldModel lets anyone verify, on a CPU with no floating point, that a
committed quantized [LeWorldModel](https://github.com/lucas-maes/le-wm) (an
action-conditioned JEPA world model) was run exactly as specified: rolling out
candidate action plans, scoring them against a goal, and picking the best. It
Freivalds-checks the large matmuls and exactly re-executes everything else, all
bound by Merkle commitments and a Fiat-Shamir transcript. No proving circuit, no
arithmetization, stable Rust.

It adapts the [CommitLLM](https://github.com/lambdaclass/CommitLLM)
commit-and-audit scheme from language models to a world model, and because the
model runs in exact integer fixed point rather than bf16 on a GPU, it closes the
one honest hole CommitLLM leaves open: non-reproducible attention.

**Interactive explainer:** https://abdelstark.github.io/ProvableWorldModel/

### What it does not claim

It proves a precise exact arithmetic relation. It does not claim floating point or
PyTorch equivalence, not physical truth of the predictions, not zero knowledge.
Public claims must be a subset of the proven statement.

## Quickstart

Run the whole scheme as a two-party game. A prover runs real integer inference and
writes a proof; a verifier accepts it, then a forged matmul gets rejected.

```bash
git clone https://github.com/AbdelStark/ProvableWorldModel
cd ProvableWorldModel
docker compose up --build
```

```
prover-1    | ✓ prover ran inference and wrote a 3417-byte proof to /shared/artifact.bin
verifier-1  |   ✓ verifier drew a secret challenge r (2 vectors); checked v·x == r·z per linear op
verifier-1  |   ✓ ACCEPT  output [6, -2]
verifier-1  |   • forged the accumulator of linear op 100 (a fake matmul result)
verifier-1  |   ✗ REJECT  FreivaldsCheckFailed { op_id: 100 }
```

Without Docker:

```bash
cargo run -p pwm-testkit --bin pwm --release          # the full story in one process
cargo run -p pwm-testkit --bin pwm --release -- --json  # machine-readable
cargo test --workspace                                  # the accept and reject suites
```

See [demo/README.md](demo/README.md) for what each step shows.

## How it works

```text
  le-wm checkpoint
        |
        v
  [pwm-export]   quantize to an exact integer graph, fold BatchNorm,
        |        emit a canonical manifest + committed lookup tables
        v
  quantized graph + committed inputs (latent history, goal, candidate actions)
        |
        v
  [pwm-prover]   run the exact integer reference inference, record the trace,
        |        Merkle-commit it, squeeze the Freivalds challenge via Fiat-Shamir
        v
  AuditArtifact  (commitments + trace + claimed outputs)
        |
        v
  [pwm-verifier] CPU, no_std, float-free:
        |          - Freivalds-check every fixed-weight matmul:  v·x == r·z
        |          - exactly recompute requant, attention, tables, LayerNorm, cost, argmin
        |          - check the rollout recurrence and the selection
        v
  accept  (Ok)   or   reject  (a specific, typed VerifyError)
```

The prover runs the model normally and commits to its execution trace. The
verifier never re-runs the model. For each fixed weight matrix it precomputes
`v = rᵀW` once and checks `v·x == r·z` per use, which holds because
`rᵀ(Wx) = (rᵀW)x`. Everything cheap and deterministic, attention dot products,
requant, the nonlinear table reads, the cost and the argmin, is recomputed
exactly from committed data.

### Soundness

A wrong accumulator `z ≠ Wx` passes one random Freivalds check with probability at
most `1/p`, with `p = 2⁶¹ − 1`. A union bound over `N ≈ 10⁵` checked instances
stays around `2⁻⁴⁴`. The same `W` is reused across every candidate, rollout step,
and transformer block, so `v = rᵀW` is computed once per weight matrix and reused
about `S × horizon` times: that reuse is the whole point.

Mutating any load-bearing value, a weight, a scale, the rounding mode, a table, the
op order, the planner config, a public input, a claimed output, or any trace cell,
changes a commitment or fails an exact check, and the proof is rejected.

## What gets proven

| Tier | Claim | Status |
|---|---|---|
| **P0** | One predictor step: `z_next = PredProj(ARPredictor(z_hist, ActEnc(actions)))`. | shipped |
| **P1** | Autoregressive rollout: each step feeds the next; the recurrence wiring is checked. | shipped |
| **P2** | Fixed-candidate planning (V0): roll out all `S` candidates, score by goal MSE, prove the selected is the argmin. | shipped |
| P3 | Full CEM planner (sampling, elites, distribution updates). | deferred |
| P4 | Pixel to plan, including the ViT encoder. | deferred |

All `S` candidate costs must be proven, not only the winner: proving only the
selected candidate would be unsound.

## Architecture

Five small crates, one trust anchor. The verifier depends on neither the exporter
nor any Python or float runtime; it verifies arithmetic only, sharing one
Freivalds and trace implementation with the prover through `pwm-core`.

| Crate | Role |
|---|---|
| [`pwm-core`](crates/pwm-core) | Fields (M31 value, Fp61 audit), fixed-point reference, tensors, Merkle commitments, Fiat-Shamir transcript, the Freivalds check, the trace model. `no_std`. |
| [`pwm-export`](crates/pwm-export) | le-wm checkpoint to quantized integer graph, manifest, golden vectors, the Rust integer reference, the data adapter. |
| [`pwm-prover`](crates/pwm-prover) | Run the reference inference, commit the trace, derive challenges, emit the `AuditArtifact`. |
| [`pwm-verifier`](crates/pwm-verifier) | CPU, `no_std`, float-free. Freivalds-check the linears, recompute the rest, check rollout, cost, argmin. |
| [`pwm-testkit`](crates/pwm-testkit) | Golden vectors, the accept and reject suites, the mutation harness, the `pwm` demo CLI. |

```text
pwm-prover --> pwm-export --> pwm-core <-- pwm-verifier
                                  ^
                           pwm-testkit
```

The argmin uniqueness check and the Freivalds probability bound are also formally
verified in Lean 4 (no `sorry`); see [lean/](lean).

## The model

LeWorldModel is a JEPA-style action-conditioned world model: it predicts the next
latent, not pixels. The proven V0 subgraph is `action_encoder -> predictor ->
pred_proj` at `latent_dim = 192`, `history_size = 3`, depth `6`, `16` heads,
`dim_head = 64`, `mlp_dim = 2048`, which is bit-deterministic in eval mode. The
pixel encoder (ViT-Tiny/14) is deferred to P4; V0 takes latents as inputs.

## Documentation

- [Interactive explainer](https://abdelstark.github.io/ProvableWorldModel/) the visual walkthrough.
- [specs.md](specs.md) the normative specification of the commit-and-audit design.
- [roadmap.md](roadmap.md) the plan, the pivot rationale, and the sequencing.
- [demo/README.md](demo/README.md) the local demo.
- [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
