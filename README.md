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

It proves an exact arithmetic relation. It does not claim floating-point or PyTorch
equivalence, the physical truth of the predictions, or zero knowledge. Public
claims must be a subset of the proven statement.

## Quickstart

Run the whole scheme as a two-party game. A prover runs a real world-model
predictor step in exact integer arithmetic and writes a proof; a verifier accepts
it, then a forged matmul gets rejected.

```bash
git clone https://github.com/AbdelStark/ProvableWorldModel
cd ProvableWorldModel
docker compose up --build
```

```
prover-1    | [prover] model   le-wm action-conditioned predictor block (self-attention + GELU FFN + residuals)
prover-1    | [prover] infer   exact integer forward pass in 0.027 ms
prover-1    | [prover]   z_history [1, 0, 0, 1]  action [1, 0]
prover-1    | [prover]   z_next    [3905, 1802]  (predicted next latent)
prover-1    | [prover] trace   15 ops, block_root 99ba60b3...
verifier-1  | [verifier] challenge  replayed the Fiat-Shamir transcript, derived Freivalds r for 7 linear ops
verifier-1  | [verifier] ACCEPT     in 0.048 ms   z_next [3905, 1802]
verifier-1  | [verifier] tamper     forged the output of matmul op 4 (a fake projection result)
verifier-1  | [verifier] REJECT     FreivaldsCheckFailed { op_id: 4 }
```

The demo proves the le-wm predictor architecture (attention, action conditioning,
GELU feed-forward, residuals) as a compact instance. See [demo/README.md](demo/README.md).

Without Docker:

```bash
cargo run -p pwm-testkit --bin pwm --release          # the full story in one process
cargo run -p pwm-testkit --bin pwm --release -- --json  # machine-readable
cargo test --workspace                                  # the accept and reject suites
```

See [demo/README.md](demo/README.md) for what each step shows.

### Prove the real pretrained checkpoint

The exporter ingests the real [`quentinll/lewm-pusht`](https://huggingface.co/quentinll/lewm-pusht)
checkpoint, quantizes the full 192-dim V0 subgraph (the action encoder, the six
predictor blocks, and `pred_proj`), folds the BatchNorm, and commits the manifest.
It writes two prover bundles: the `pred_proj` head and the full predictor.

```bash
pip install torch numpy
# download weights.pt + config.json from the model page above, then
LEWM_WEIGHTS=weights.pt python crates/pwm-export/python/scripts/export_lewm_v0.py \
  /tmp/lewm_pred_proj.json /tmp/lewm_predictor.json
```

The full **6-block, 16-head attention predictor** with the real quantized weights
proves and verifies in the Rust prover:

```bash
cargo run -p pwm-testkit --bin pwm --release -- prove-predictor /tmp/lewm_predictor.json
```

```
[prover] model   le-wm V0 predictor (6 blocks, 16 heads), REAL quantized checkpoint weights
[prover] config  dim=192, history=3, heads=16, dim_head=64, mlp=2048, depth=6  (self-attention + AdaLN + GELU FFN + residuals)
[prover] graph   2437 ops, 30 weight tensors
[prover] infer   exact integer forward pass in 22.911 ms
[prover]   z_next[..6] [37, 0, -7, 55, -39, -17]  (predicted next-latent head)
[verifier] ACCEPT  in 23.876 ms
[verifier] tamper  forged matmul op Some(2) -> REJECT FreivaldsCheckFailed { op_id: 2 }
```

The predictor is built as the real le-wm architecture over the named-buffer block
DAG: per ConditionalBlock, AdaLN-zero conditioning (`SiLU(c) -> Linear -> chunk6`)
modulates a 16-head self-attention sub-block (per head: `Q.Kᵀ -> softmax ->
prob.V`) and a GELU feed-forward, each with a gated residual, with the multi-head
reshapes expressed as Slice/Concat. The verifier Freivalds-checks every projection
and exactly recomputes the attention, softmax, GELU, LayerNorm, and residuals.
`prove-lewm` proves the smaller `pred_proj` head on its own.

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
`rᵀ(Wx) = (rᵀW)x`. The cheap, deterministic ops are recomputed exactly from
committed data: attention dot products, requant, the nonlinear table reads, the
cost, and the argmin.

### Soundness

A wrong accumulator `z ≠ Wx` passes one random Freivalds check with probability at
most `1/p`, with `p = 2⁶¹ − 1`. A union bound over `N ≈ 10⁵` checked instances
stays around `2⁻⁴⁴`. The same `W` is reused across every candidate, rollout step,
and transformer block, so `v = rᵀW` is computed once per weight matrix and reused
about `S × horizon` times: that reuse is the whole point.

Mutating any load-bearing value changes a commitment or fails an exact check, and
the proof is rejected. That covers weights, scales, the rounding mode, tables, op
order, planner config, public inputs, claimed outputs, and every trace cell.

## What gets proven

| Tier | Claim | Status |
|---|---|---|
| **P0** | One predictor step: `z_next = PredProj(ARPredictor(z_hist, ActEnc(actions)))`. | implemented and tested; the full 192-dim 6-block 16-head predictor proves and verifies with real checkpoint weights |
| **P1** | Autoregressive rollout: each step feeds the next; the recurrence wiring is checked. | implemented and tested |
| **P2** | Fixed-candidate planning (V0): roll out all `S` candidates, score by goal MSE, prove the selected is the argmin. | implemented and tested |
| P3 | Full CEM planner (sampling, elites, distribution updates). | deferred |
| P4 | Pixel to plan, including the ViT encoder. | deferred |

What is implemented and tested today (170 tests): the commit-and-audit protocol
(Freivalds, exact replay, Merkle commitments, Fiat-Shamir), the full predictor op
vocabulary (attention, AdaLN, GELU and SiLU tables, LayerNorm, residuals,
softmax), the rollout recurrence, the MSE cost, and the argmin with tie-break,
each with accept and reject tests. The exporter ingests the real
`quentinll/lewm-pusht` checkpoint and quantizes the full 192-dim V0 subgraph, and
the Rust prover proves and verifies the full 6-block, 16-head, 192-dim predictor
with the real quantized weights (`pwm prove-predictor`), plus the `pred_proj` head
on its own (`pwm prove-lewm`). The proof attests the exact integer (quantized)
relation; per-tensor activation-scale calibration for float-faithful outputs is a
further refinement. For P2, all `S` candidate costs must be proven, not only the
winner: proving only the selected candidate would be unsound.

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
latent, not pixels. The target V0 subgraph the exporter ingests is
`action_encoder -> predictor -> pred_proj` at `latent_dim = 192`,
`history_size = 3`, depth `6`, `16` heads, `dim_head = 64`, `mlp_dim = 2048`, which
is bit-deterministic in eval mode. The full 192-dim predictor proves and verifies
in the Rust prover with the real quantized weights (`pwm prove-predictor`). The
pixel encoder (ViT-Tiny/14) is deferred to P4; V0 takes latents as inputs.

## Documentation

- [Interactive explainer](https://abdelstark.github.io/ProvableWorldModel/): the visual walkthrough.
- [specs.md](specs.md): the normative specification of the commit-and-audit design.
- [roadmap.md](roadmap.md): the plan, the pivot rationale, and the sequencing.
- [demo/README.md](demo/README.md): the local demo.
- [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
