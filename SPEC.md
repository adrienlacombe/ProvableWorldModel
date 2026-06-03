# ProvableWorldModel — Specification Index

Status: living specification. This file is the entry point and index for the
canonical specification corpus. It is intentionally short: the normative detail
lives in `docs/spec/` (specification) and `docs/rfcs/` (decisions). Where this
index and a corpus document disagree, the corpus document governs.

## Thesis

ProvableWorldModel produces succinct proofs that a committed, quantized,
JEPA-style world model (LeWorldModel) was executed exactly as specified. The
system arithmetizes deterministic fixed-point inference of the model's latent
predictor, rollout, and fixed-candidate planner as custom Circle-STARK AIR over
the Mersenne-31 field (M31), using a vendored Stwo prover. A verifier checks the
proof without running PyTorch and without trusting the prover's host.

The proof establishes an arithmetic relation, not empirical truth. It proves that
the claimed predicted latents, costs, and selected action are the output of the
specified algorithm on the committed model. It does not prove that the model's
predictions match the world, that the model is calibrated, that a floating-point
PyTorch run would produce the same bits, or that the proof is zero-knowledge.
Those are separate claims and are not implied.

## What the first release (V0) proves

> Given a committed quantized LeWorldModel predictor manifest, an initial latent
> history, a goal latent, and S fixed candidate action sequences, the proof
> verifies that every candidate was rolled out through the committed quantized
> predictor, every final-latent cost was computed as specified, and the selected
> candidate has minimum cost under deterministic tie-breaking.

See [docs/spec/00-overview.md](docs/spec/00-overview.md) for the full scope,
the proof-statement tiers (P0–P4), the release versions (V0–V3), and the
feasibility verdict.

## Specification corpus

Normative specification (`docs/spec/`):

| Document | Concern |
|---|---|
| [00-overview.md](docs/spec/00-overview.md) | Thesis, goals, non-goals, success criteria, scope tiers, feasibility verdict |
| [01-architecture.md](docs/spec/01-architecture.md) | Crates, module boundaries, data flow, AIR strategy, trace model |
| [02-public-api.md](docs/spec/02-public-api.md) | Public Rust API, CLI shapes, artifact formats, stability policy |
| [03-data-model.md](docs/spec/03-data-model.md) | Manifest schema, canonical types, invariants, schema versioning |
| [04-error-model.md](docs/spec/04-error-model.md) | Error taxonomy, failure modes, verifier rejections, recovery |
| [05-observability.md](docs/spec/05-observability.md) | Logging, metrics, tracing, redaction rules |
| [06-security.md](docs/spec/06-security.md) | Threat model, trust boundaries, soundness, privacy/ZK boundary, disclosure |
| [07-testing-strategy.md](docs/spec/07-testing-strategy.md) | Test pyramid, golden vectors, negative/differential/mutation tests, CI gates |
| [08-performance-budget.md](docs/spec/08-performance-budget.md) | Proving cost model, targets, scaling, profiling, perf gates |
| [09-release-and-versioning.md](docs/spec/09-release-and-versioning.md) | Semver, relation_id immutability, deprecation, changelog, MSRV, license |
| [10-glossary.md](docs/spec/10-glossary.md) | Canonical terms |

Decisions (`docs/rfcs/`). Status and target milestone shown; Draft/Future RFCs
are specified to full quality but deferred beyond V0:

| RFC | Title | Status | Milestone |
|---|---|---|---|
| [RFC-0000](docs/rfcs/RFC-0000-security-model-and-statement-taxonomy.md) | Security model and statement taxonomy | Accepted | v0.1 |
| [RFC-0001](docs/rfcs/RFC-0001-model-manifest-and-export-pipeline.md) | Model manifest and export pipeline | Accepted | v0.1 |
| [RFC-0002](docs/rfcs/RFC-0002-fixed-point-arithmetic-over-m31.md) | Fixed-point arithmetic over M31 | Accepted | v0.1 |
| [RFC-0003](docs/rfcs/RFC-0003-range-check-and-lookup-infrastructure.md) | Range-check and lookup infrastructure | Accepted | v0.1 |
| [RFC-0004](docs/rfcs/RFC-0004-tensor-memory-and-wiring-air.md) | Tensor memory and wiring AIR | Accepted | v0.1 |
| [RFC-0005](docs/rfcs/RFC-0005-linear-matmul-and-requantization-components.md) | Linear, matmul, and requantization components | Accepted | v0.1 |
| [RFC-0006](docs/rfcs/RFC-0006-nonlinear-primitive-components.md) | Nonlinear primitive components | Accepted | v0.2 |
| [RFC-0007](docs/rfcs/RFC-0007-leworldmodel-predictor-air.md) | LeWorldModel predictor AIR | Accepted | v0.2 |
| [RFC-0008](docs/rfcs/RFC-0008-rollout-air.md) | Rollout AIR | Accepted | v0.2 |
| [RFC-0009](docs/rfcs/RFC-0009-fixed-candidate-planner-proof.md) | Fixed-candidate planner proof | Accepted | v1.0 |
| [RFC-0010](docs/rfcs/RFC-0010-cem-planner-proof.md) | CEM planner proof | Draft | Future |
| [RFC-0011](docs/rfcs/RFC-0011-pixel-encoder-proof.md) | Pixel encoder proof | Draft | Future |
| [RFC-0012](docs/rfcs/RFC-0012-recursive-aggregated-verification.md) | Recursive / aggregated verification | Draft | Future |
| [RFC-0013](docs/rfcs/RFC-0013-testing-fuzzing-and-audit-strategy.md) | Testing, fuzzing, and audit strategy | Accepted | v0.1 |
| [RFC-0014](docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md) | Canonical serialization, public-input binding, transcript | Accepted | v0.1 |
| [RFC-0015](docs/rfcs/RFC-0015-third-party-vendoring-and-pinning.md) | Third-party vendoring, dependency pinning, audit boundary | Accepted | v0.1 |
| [RFC-0016](docs/rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md) | Prover/verifier CLI, artifact bundle, reproducibility | Accepted | v0.2 |

## Release plan

| Version | Milestone | Delivers |
|---|---|---|
| V0 | v0.1 → v0.2 → v1.0 | P0 step, P1 rollout, P2 fixed-candidate planning, end to end |
| V1 | post-1.0 | Batched/efficient P2, optional audited ZK mode, recursive verification |
| V2 | Future | P3 CEM planner proof |
| V3 | Future | P4 pixel-to-plan end-to-end proof |

## Origin

The project began as the feasibility study preserved at
[docs/feasibility-study.md](docs/feasibility-study.md). That document is historical
rationale; the corpus above is the maintained, verified specification.

## Implementation

The implementation work decomposed from this corpus is tracked in
[docs/roadmap/IMPLEMENTATION.md](docs/roadmap/IMPLEMENTATION.md) and in the
project's GitHub issues.

## Project policies

- License: Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
- Security and disclosure: see [docs/spec/06-security.md](docs/spec/06-security.md).
- Contributing: see [CONTRIBUTING.md](CONTRIBUTING.md).
