# ProvableWorldModel — Specification Index

Status: living specification. Post-pivot (2026-06-05) the canonical specification
is the top-level [specs.md](specs.md), backed by [roadmap.md](roadmap.md) (plan)
and [backlog.md](backlog.md) (work). The pre-pivot STARK corpus is archived under
[docs/legacy-stark/](docs/legacy-stark/) and is no longer normative.

## Thesis

ProvableWorldModel produces **commit-and-audit** proofs that a committed,
quantized, JEPA-style world model (LeWorldModel) was executed exactly as
specified. The prover runs deterministic integer fixed-point inference and
commits to the execution trace; a CPU verifier audits it with Freivalds checks
for the large linear layers and exact integer re-execution for everything else,
all bound by Merkle commitments and a Fiat-Shamir transcript.

The proof establishes an arithmetic relation, not empirical truth. It does not
prove floating-point PyTorch equivalence, physical truth of predictions, or
zero-knowledge privacy.

## What the first release (V0) proves

> Given a committed quantized LeWorldModel predictor manifest, an initial latent
> history, a goal latent, and S fixed candidate action sequences, the proof
> verifies that every candidate was rolled out through the committed quantized
> predictor, every final-latent cost was computed as specified, and the selected
> candidate has minimum cost under deterministic tie-breaking.

## Where to read

| Concern | Document |
|---|---|
| Plan, pivot rationale, phases, sequencing | [roadmap.md](roadmap.md) |
| Scope, number representation, the audit protocol, Freivalds, exact replay, crate layout, API, threat model | [specs.md](specs.md) |
| Issue backlog (M0–M8) and critical path | [backlog.md](backlog.md) |
| Archived pre-pivot STARK corpus (RFCs, AIR specs, feasibility study) | [docs/legacy-stark/](docs/legacy-stark/) |

## Project policies

- License: Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
- Security and disclosure: see [SECURITY.md](SECURITY.md).
- Contributing: see [CONTRIBUTING.md](CONTRIBUTING.md).
