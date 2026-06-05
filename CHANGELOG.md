# Changelog

All notable changes to ProvableWorldModel are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) under the `0.x`
convention. See [specs.md](specs.md) and [roadmap.md](roadmap.md).

This is a single workspace changelog; per-crate sections may be split out before
the first `1.0` release. Relation and schema-version events are first-class
entries: a new `relation_id`, `manifest_version`, or `artifact_version` is an
`Added` line, and any change that narrows verifier acceptance is a `Security`
entry.

## [Unreleased]

### Changed

- **Pivot (2026-06-05): cryptographic backend changed from Circle-STARK (vendored
  Stwo) to the CommitLLM commit-and-audit scheme.** The proof target (quantized
  LeWorldModel predictor → rollout → fixed-candidate planner) is unchanged.
  Removed the `pwm-air` and `pwm-circuits` crates and the vendored `third_party/stwo`
  and `third_party/stwo-circuits` trees; the workspace is now the five crates
  `pwm-core`, `pwm-export`, `pwm-prover`, `pwm-verifier`, `pwm-testkit`. Replaced
  the nightly toolchain with **stable Rust** (the nightly-only Stwo dependency is
  gone). `pwm-core` gains a native `M31`, the audit field `Fp61`, a native
  Fiat-Shamir transcript, the `freivalds` check, and the `trace` model; the
  pre-pivot spec corpus is archived under `docs/legacy-stark/`. See
  [roadmap.md](roadmap.md), [specs.md](specs.md), [backlog.md](backlog.md).

### Added

- Cargo workspace and the first-party crates `pwm-core`, `pwm-export`,
  `pwm-prover`, `pwm-verifier`, `pwm-testkit`; `pwm-verifier` builds `no_std`,
  float-free, and does not depend on `pwm-export` (#23).
- CI merge-gate pipeline: `cargo fmt`, `clippy -D warnings`, tests on MSRV and
  stable, the `no_std` verifier build, the documentation build with an internal
  link check, the license/SPDX gate, and a constraint-mutation gate (#73, #74).
- `pwm-core::obs`: the structured-logging schema, the `Level` policy, the
  type-level redaction guard (`LoggableValue`), the named metric set, and the
  span-tree scaffolding (#66).
- `pwm-core::relation::StatementType`, the P0–P4 statement discriminant (#66).
- `pwm-testkit`: the accept/reject (dual-test) harness, the golden-vector
  loader, and the constraint-mutation runner skeleton (#77).
- `docs/security/soundness-binding-checklist.md`, mapping each binding and
  range-safety requirement to its enforcing RFC, issue, rejection, and test
  (#80).
- Release automation: centralized `[workspace.dependencies]` versioning, this
  `CHANGELOG.md`, the changelog/semver lint (`ci/check-changelog.py`), the
  release-preparation helper (`ci/prepare-release.py`), and the tagged release
  workflow (#76).

[Unreleased]: https://github.com/AbdelStark/ProvableWorldModel/commits/main
