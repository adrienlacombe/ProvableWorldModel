# Legacy STARK corpus (archived, superseded)

This directory preserves the **pre-pivot** ProvableWorldModel specification: a
custom Circle-STARK arithmetization (AIR) over the Mersenne-31 field using a
vendored Stwo prover. As of the 2026-06-05 pivot it is **archived and no longer
normative**. Nothing here is built or maintained; it is kept for provenance and
because parts of it (the model decomposition, the fixed-point semantics, the
manifest/serialization design) informed the current commit-and-audit design.

The authoritative specification is now:

- [`../../roadmap.md`](../../roadmap.md) — plan and pivot rationale.
- [`../../specs.md`](../../specs.md) — the commit-and-audit specification.
- [`../../backlog.md`](../../backlog.md) — the work backlog.

## Why the pivot

The STARK approach required arithmetizing every operation as custom AIR with
range-checks and LogUp lookups (the dominant engineering risk), and the vendored
Stwo + stwo-circuits required a **nightly** toolchain that conflicted with the
workspace's stable MSRV. The project moved to the
[CommitLLM](https://github.com/lambdaclass/CommitLLM) commit-and-audit scheme:
Freivalds checks for the large linear layers, exact integer re-execution for the
rest, Merkle commitments, a stable toolchain, and a `no_std` float-free verifier.

## What was carried forward into the live tree (not here)

- The bit-exact integer fixed-point reference (`pwm-core::fixed_point`).
- The Blake2s Merkle commitments and binding structs (`pwm-core::commit`).
- The canonical serialization and manifest types (`pwm-core::{serialize, manifest}`).
- The Fiat-Shamir transcript (re-implemented natively, no Stwo channel).
- The LeWorldModel predictor decomposition and quantization policy (now in `specs.md`).

## Contents

- `spec/` — the 11-document normative spec corpus (00-overview … 10-glossary).
- `rfcs/` — RFC-0000 … RFC-0016.
- `security/` — the soundness/binding checklist.
- `roadmap/` — the original implementation tracker.
- `feasibility-study.md` — the founding analysis.

> Cross-references inside these files (to `docs/spec/...`, `third_party/stwo`,
> the `pwm-air`/`pwm-circuits` crates, etc.) point at paths that no longer exist
> in the live tree. They are accurate *as of the archive* and are intentionally
> left unmodified.
