# ProvableWorldModel

A succinct-proof system for deterministic, quantized inference of a JEPA-style
world model (LeWorldModel), built as custom Circle-STARK arithmetization (AIR)
over the Mersenne-31 field using a vendored Stwo prover.

The system proves a precise arithmetic relation: given a committed quantized
predictor, a latent history, a goal latent, and a fixed set of candidate action
sequences, the claimed predicted latent trajectories, costs, and the selected
action are exactly the output of the specified fixed-point inference and planning
algorithm. It does **not** claim to prove floating-point PyTorch equivalence,
physical truth of predictions, or zero-knowledge privacy unless a specific
relation and audit establish those properties.

## Status

Specification phase. The canonical specification corpus is complete; the
implementation is decomposed into tracked GitHub issues. There is no released
code yet.

- Start here: [SPEC.md](SPEC.md) — the specification index.
- Scope and the first-release proof statement: [docs/spec/00-overview.md](docs/spec/00-overview.md).
- Architecture: [docs/spec/01-architecture.md](docs/spec/01-architecture.md).
- Decisions: [docs/rfcs/](docs/rfcs/).
- Implementation tracker: [docs/roadmap/IMPLEMENTATION.md](docs/roadmap/IMPLEMENTATION.md).

## What it is

| Layer | Crate | Role |
|---|---|---|
| Core | `pwm-core` | Field and fixed-point types, tensors, manifest types, canonical serialization, transcript |
| Export | `pwm-export` | PyTorch checkpoint to quantized graph to manifest, weights, and golden vectors |
| AIR | `pwm-air` | Circle-STARK components: range-check, tensor memory, linear, matmul, requantize, activations, attention, predictor, rollout, cost, argmin |
| Circuits | `pwm-circuits` | Low-level gates and adapters over vendored stwo-circuits |
| Prover | `pwm-prover` | Trace building and proof generation |
| Verifier | `pwm-verifier` | Verification (no PyTorch, `no_std`-capable) |

Stwo and stwo-circuits are vendored under `third_party/` at pinned revisions; see
[RFC-0015](docs/rfcs/RFC-0015-third-party-vendoring-and-pinning.md).

## Scope tiers

The proof statements form a hierarchy. The first release (V0) targets P2.

| Statement | What it proves |
|---|---|
| P0 | One quantized predictor step |
| P1 | Autoregressive latent rollout over a horizon |
| P2 | Fixed-candidate planning: roll out all candidates, score by goal MSE, select the argmin (V0) |
| P3 | Full CEM planner (deferred) |
| P4 | Pixel-to-plan end to end, including the encoder (deferred) |

## Contributing and security

- [CONTRIBUTING.md](CONTRIBUTING.md) — spec-driven workflow, code standards, CI gates.
- [SECURITY.md](SECURITY.md) — private vulnerability reporting; soundness is the top severity class.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and
[NOTICE](NOTICE).
