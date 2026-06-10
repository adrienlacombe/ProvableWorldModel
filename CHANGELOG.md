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

### Security

- **Per-buffer range enforcement on the exact-recompute path** (#180): the
  verifier now rejects — with the typed `VerifyError::BufferRange` /
  `VerifyError::InputRange` — any prover-supplied value outside the single-M31
  envelope `[-P_HALF, P_HALF]` before computing on it: every exactly-recomputed
  op's claimed output (flat trace and block paths), every seeded block input
  buffer, and every proof-carried quantization parameter (`zero_point`,
  `clamp_lo/hi`, `one`). The data-dependent kernels (`predictor::matmul`,
  `layernorm` variance, `softmax` exp sum and `e·one` products) now accumulate
  in `i128` with explicit envelope bound checks (mirroring
  `fixed_point::finish`), and `ActivationTable::eval` computes its index in
  `i128`, so an adversarial buffer or table can no longer wrap `i64` arithmetic
  into a fail-closed **panic** (under `overflow-checks`) — it is rejected
  cleanly. Narrows verifier acceptance; no honest proof regresses (full suite
  plus the real-dims predictor test stay green).

- **The predictor bundle is now commitment-chained to the export** (#188): the
  Python export emits, in `lewm_predictor.json`, a predictor-scoped
  `weights_root` computed over exactly the 30 proven block tensors in the Rust
  prover's canonical `tensor_id` order (`pwm_export.export.predictor_weight_dicts`),
  and the prover binds that *carried* value into the model commitment
  (`prove_predictor_with_weights_root`), so `verify_predictor` rejects with
  `CommitmentMismatch(Model)` unless the proven weights reproduce the export's
  commitment bit-for-bit. The artifact alone now certifies which weight set was
  proven (bundle ⇄ export bound; checkpoint ⇄ export remains trusted
  preprocessing — documented in README and specs §13). The Python↔Rust
  canonical-encoding parity gate is now two-way: the multi-leaf `weights_root`
  vector and the full predictor weight scheme are pinned on both sides.

### Changed

- Polished the explainer site against the Impeccable design guidelines: removed
  the AI-default scaffolding (per-section uppercase eyebrow kickers, side-stripe
  accent borders, hero metric boxes, decorative card chips), swapped the body
  face from Inter to Archivo, and fixed real defects found while verifying in a
  browser — grid blowouts that caused horizontal page scroll (`.start`, `.arch`),
  a tamper switch that could not be toggled from the keyboard, the closed mobile
  menu remaining in the tab order, a focus ring that reshaped pill buttons, and
  missing arrow-key navigation on the tier tabs.

- **The `prove-predictor` demo path is now commitment-bound** (#187): the full
  6-block predictor (synthetic and `--profile real`) routes through
  `pwm_prover::prove_predictor` → `PredictorArtifact` and is verified by
  `pwm_verifier::verify_predictor`, so the demo's ACCEPT now asserts the model /
  quantization / planner / input / output commitments in the artifact's public
  input, not just the arithmetic. `lewm_predictor::prove` returns
  `Result<PredictorArtifact, ProveError>` (was a bare `Block` with an internal
  `expect`), `verify` takes the artifact, and the builders return a
  `PredictorCircuit` struct (block, weights, tables, scales, inputs) instead of a
  4-tuple. The CLI VERIFY stage narration states the commitment checks, and the
  report carries the quantization / input / output commitments.

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
- Decomposed the CLI `prove-predictor` command (a ~340-line match arm that
  interleaved proving and rendering) into a single `PredictorReport` computation
  plus pure `print_predictor_report_{json,human}` reporters; output is unchanged.
- Locked in a curated, cast-noise-free slice of clippy style lints in
  `[workspace.lints.clippy]` (`unreadable_literal`, `redundant_closure_for_method_calls`,
  `map_unwrap_or`, `explicit_iter_loop`, `uninlined_format_args`,
  `semicolon_if_nothing_returned`) and applied the resulting idiom fixes, so the
  gate prevents regressions without the noise a blanket `pedantic` would add.
- Replaced stringly-typed rejection sub-discriminants with typed enums:
  `VerifyError::CommitmentMismatch(CommitmentKind)` and `MissingBinding(BindingKind)`
  (was `&'static str`), and split `BlockError::MissingBinding` into `MissingWeight`,
  `MissingTable`, and `InvalidRounding{op_id,discriminant}`. Rejection codes are now
  compiler-checked and stable for downstream matchers.
- `pwm-testkit::lewm::prove` returns `Result<AuditArtifact, ProveError>` instead of
  panicking, so a malformed bundle whose input escapes the M31 range fails cleanly.
- Routed the predictor block prover's linear ops through the shared
  `pwm_core::predictor::linear` kernel (was an inlined loop), removing prover /
  reference / verifier drift; narrowed `pwm_core::freivalds::dot_fp_i64` to
  `pub(crate)`.

### Fixed

- Closed the remaining block-path test-coverage gaps (#181): `Requant` was the
  one `BlockOp` variant without a tampered-output reject test (all 12 now
  reject a bumped claimed output with the exact typed error and `op_id`); the
  small-dims predictor accept tests now pin the exact integer output vector
  (not just `is_ok` + float tolerance), locking the quantized semantics
  end-to-end; and the soundness mutation campaign gained a
  `range_check.modp_predictor` mutant — the mod-`p` accumulator-aliasing
  forgery on a `BatchedLinear` accumulator — so the predictor/block path's
  Freivalds range guard is merge-gated like the flat demo path's.
- The soundness mutation campaign registered no mutant for the `tensor_memory`
  (op-to-op wiring) component, so its merge gate passed vacuously; it now exercises
  a `WiringMismatch` mutant. The committed `requantize` golden fixture is now run
  through the real primitive (it was only parsed), and `verify_planning_batched`
  gained the missing argmin/cost reject tests.
- Corrected stale documentation: `pwm-export` is consumed by `pwm-prover` and
  `pwm-testkit` (not a dependency-DAG leaf; the verifier trust root depends on it
  only as a dev-dependency); the `commit` hash rationale (Blake2s is the transcript
  sponge, not a vendored-Stwo match); the dependency DAGs in `specs.md`/`README`; and
  three release-tooling docstrings pointing at the relocated spec path.

### Added

- Cargo workspace and the first-party crates `pwm-core`, `pwm-export`,
  `pwm-prover`, `pwm-verifier`, `pwm-testkit`; `pwm-verifier` builds `no_std`,
  float-free, and does not depend on `pwm-export` (#23).
- The commit-and-audit protocol: Blake2s Merkle commitments, the native
  Fiat-Shamir transcript, the Freivalds linear check, and the exact-recompute
  trace model (`pwm-core`).
- The P0 quantized feed-forward statement (`pwm.lewm.mlp.v1`), the P1 rollout, and
  the P2 fixed-candidate planning proof, with `prove_*`/`verify_*` (`pwm-prover`,
  `pwm-verifier`).
- The full named-buffer le-wm predictor (AdaLN-zero, multi-head attention, GELU
  FFN, gated residuals) as a block DAG, proven by `prove_block` and audited by
  `verify_block`; the exporter ingests the real `quentinll/lewm-pusht` checkpoint.
- A first-class **commitment-bound** predictor relation `pwm.lewm.predictor_step.v1`
  (`RELATION_PREDICTOR`): `prove_predictor` / `verify_predictor` recompute and check
  the model, quantization, input, and output commitments (PRED-02, #179).
- The `pwm` demo CLI with a staged, observable pipeline (LOAD → INFER → COMMIT →
  VERIFY → TAMPER, op histogram, MAC count, latency/throughput), the Docker compose
  demo (synthetic and `--profile real` paths), and the interactive explainer site.
- CI merge-gate pipeline: `cargo fmt`, `clippy -D warnings`, tests on MSRV and
  stable, the `no_std` verifier build, the documentation build with an internal
  link check, the license/SPDX + supply-chain (`cargo deny check`) gate, and a real
  constraint-mutation gate (#73, #74).
- `pwm-core::obs`: the structured-logging schema, the `Level` policy, the
  type-level redaction guard (`LoggableValue`), the named metric set, and the
  span-tree scaffolding, behind the off-by-default `obs` feature so the verifier
  trust root excludes it (#66).
- `pwm-core::relation::StatementType`, the P0–P4 statement discriminant (#66).
- `pwm-testkit`: the accept/reject (dual-test) harness, the golden-vector
  loader, and a real soundness mutation campaign over the live verifier (#77).
- `docs/security/soundness-binding-checklist.md`, mapping each binding and
  range-safety requirement to its enforcing RFC, issue, rejection, and test
  (#80).
- Release automation: centralized `[workspace.dependencies]` versioning, this
  `CHANGELOG.md`, the changelog/semver lint (`ci/check-changelog.py`), the
  release-preparation helper (`ci/prepare-release.py`), and the tagged release
  workflow (#76).

### Security

- **Freivalds accumulator range guard.** The mod-`p` Freivalds check now rejects
  any input, bias, or claimed accumulator outside the single-M31 envelope (and any
  layer too wide for the soundness margin), closing a mod-`p` accumulator-aliasing
  forgery where a prover could add a multiple of `p` to an accumulator and have the
  verifier accept an attacker-chosen output. Regression-locked by an executed `+p`
  reject test and a soundness mutation campaign.
- **Overflow-safe planning and recompute.** `mse_cost` accumulates in checked
  `i128` (the 192-dim cost no longer overflows `i64` into a wrapped argmin);
  requant/rescale shifts are validated before use (`InvalidShift`); and
  `overflow-checks` is enabled in the release profile so any integer wrap on the
  recompute path traps instead of silently producing a wrong-but-matching value.

[Unreleased]: https://github.com/AbdelStark/ProvableWorldModel/commits/main
