# Implementation Tracker — 2026-06-03

Generated from the specification corpus on branch `spec/bootstrap-2026-06-03` (see the corpus PR linked in the repository). Every implementable unit of work in the specification is filed below as a GitHub issue. Each issue is independently shippable; cross-issue dependencies are noted inline and in each issue's Dependencies section.

Statement scope: the issue set covers the V0 deliverable (P0 predictor step, P1 rollout, P2 fixed-candidate planning) across milestones v0.1 -> v0.2 -> v1.0. Work beyond V0 (CEM/P3, pixel encoder/P4, recursive verification, audited ZK) is represented by tracking issues under the Future milestone and is intentionally not decomposed yet.

## Milestone: v0.1 — Foundations

| # | Title | Area | Priority | Effort | RFC | Status |
|---|-------|------|----------|--------|-----|--------|
| #23 | build: create Cargo workspace and six-crate skeleton | core | p0 | s | RFC-0015 | Open |
| #24 | vendoring: vendor and pin Stwo under third_party/stwo | core | p0 | m | RFC-0015 | Open |
| #25 | vendoring: vendor and pin stwo-circuits under third_party/stwo-circuits | core | p0 | m | RFC-0015 | Open |
| #26 | vendoring: modification/re-audit workflow and dependency inventory | core | p1 | s | RFC-0015 | Open |
| #27 | core: M31/QM31 re-export and centered signed-integer encoding | core | p0 | s | RFC-0002 | Open |
| #28 | core: BoundedInt type with overflow=reject add/sub/mul | core | p0 | m | RFC-0002 | Open |
| #29 | core: requantization with nearest-ties-to-even and truncate-toward-zero | core | p0 | m | RFC-0002 | Open |
| #30 | core: Tensor type with shape/scale invariants | core | p0 | s | RFC-0002 RFC-0001 | Open |
| #31 | core: limb-decomposed accumulators beyond the safe M31 interval | core | p1 | m | RFC-0002 | Open |
| #32 | core: byte-canonical serialization for manifest, public input, and artifact | core | p0 | m | RFC-0014 | Open |
| #33 | core: public-input digest binding all public fields | core | p0 | s | RFC-0014 RFC-0000 | Open |
| #34 | core: Fiat-Shamir transcript with documented channel ordering | core | p0 | m | RFC-0014 | Open |
| #35 | core: model/quantization/planner commitments | core | p0 | m | RFC-0014 RFC-0001 | Open |
| #36 | export: manifest types, (de)serialization, and schema validation | export | p0 | m | RFC-0001 | Open |
| #37 | export: PyTorch checkpoint to deterministic quantized graph | export | p0 | l | RFC-0001 RFC-0002 | Open |
| #38 | export: write manifest, weights, and per-op commitments (byte-identical) | export | p0 | m | RFC-0001 RFC-0014 | Open |
| #39 | export: Python fixed-point reference and golden-vector emitter | export | p0 | m | RFC-0001 RFC-0013 | Open |
| #40 | export: Python<->Rust fixed-point parity harness | export | p1 | m | RFC-0001 RFC-0013 | Open |
| #41 | cli: pwm manifest verify subcommand | export | p1 | s | RFC-0001 RFC-0016 | Open |
| #42 | air: LogUp lookup relation and multiplicity checking | air | p0 | l | RFC-0003 | Open |
| #43 | air: range-check component (u8/i8/u16/bounded-limb) | air | p0 | m | RFC-0003 RFC-0002 | Open |
| #44 | air: committed activation-table lookup relation | air | p1 | m | RFC-0003 RFC-0006 | Open |
| #45 | air: TensorCell read/write consistency component | air | p0 | l | RFC-0004 | Open |
| #46 | air: broadcast/reshape/transpose/concat/slice wiring rules | air | p1 | m | RFC-0004 | Open |
| #47 | air: linear AIR component | air | p0 | l | RFC-0005 RFC-0002 | Open |
| #48 | air: batched matmul over candidate/step/seq/head/channel axes | air | p1 | l | RFC-0005 | Open |
| #49 | air: in-circuit requantization component | air | p0 | m | RFC-0005 RFC-0002 | Open |
| #73 | ci: pipeline with fmt, clippy, tests, no_std verifier, docs, license gates | ci | p0 | m | RFC-0013 | Open |
| #74 | docs: documentation build and internal-link check | docs | p1 | m | RFC-0013 | Open |
| #77 | testing: accept/reject harness, golden loader, mutation runner | testing | p0 | m | RFC-0013 | Open |
| #80 | security: soundness/binding checklist and enforcement mapping | security | p0 | m | RFC-0000 | Open |

## Milestone: v0.2 — Predictor & Rollout

| # | Title | Area | Priority | Effort | RFC | Status |
|---|-------|------|----------|--------|-----|--------|
| #50 | air: GELU_Q lookup/polynomial component | air | p0 | m | RFC-0006 | Open |
| #51 | air: Softmax_Q approximation component | air | p0 | l | RFC-0006 | Open |
| #52 | air: LayerNorm_Q with bounded reciprocal-sqrt | air | p0 | l | RFC-0006 | Open |
| #53 | air: AdaLN_Q (AdaLN-zero, SiLU) conditional modulation | air | p0 | m | RFC-0006 RFC-0007 | Open |
| #54 | air: ActionEncoder_Q component | air | p1 | m | RFC-0007 | Open |
| #55 | air: MLP/FeedForward_Q component (192->2048->192, GELU) | air | p0 | m | RFC-0007 RFC-0005 RFC-0006 | Open |
| #56 | air: Attention_Q component (16 heads, dim_head 64) | air | p0 | l | RFC-0007 RFC-0006 | Open |
| #57 | air: ConditionalBlock_Q (AdaLN + attention + MLP) | air | p0 | l | RFC-0007 | Open |
| #58 | air: ARPredictor_Q (depth-6 stack, positional embeddings, pred_proj) | air | p0 | l | RFC-0007 | Open |
| #59 | prover: P0 proof — one predictor step end-to-end | prover | p0 | l | RFC-0007 RFC-0000 | Open |
| #60 | air: autoregressive windowing recurrence component | air | p0 | l | RFC-0008 | Open |
| #61 | prover: P1 proof — configurable-horizon rollout | prover | p0 | l | RFC-0008 RFC-0000 | Open |
| #62 | prover: trace builder (reference inference -> traces) | prover | p0 | l | RFC-0016 RFC-0005 | Open |
| #63 | verifier: verify() with full rejection taxonomy | verifier | p0 | l | RFC-0000 RFC-0014 | Open |
| #64 | cli: pwm prove and pwm verify with the ProofArtifact bundle | prover | p0 | m | RFC-0016 | Open |
| #65 | prover: determinism digest and reproducibility test | prover | p1 | m | RFC-0016 | Open |
| #66 | observability: structured logging and redaction guard | core | p1 | m | RFC-0016 | Open |
| #67 | observability: prover/verifier metrics and audit record | prover | p1 | m | RFC-0016 | Open |
| #75 | ci: performance-budget regression gate | ci | p1 | m | RFC-0013 | Open |

## Milestone: v1.0 — Fixed-Candidate Planning Proof

| # | Title | Area | Priority | Effort | RFC | Status |
|---|-------|------|----------|--------|-----|--------|
| #68 | air: Cost_Q (goal-latent MSE) component | planner | p0 | m | RFC-0009 | Open |
| #69 | air: Argmin component with deterministic tie-break | planner | p0 | m | RFC-0009 | Open |
| #70 | planner: batched candidate rollouts (all candidates proven) | planner | p0 | l | RFC-0009 RFC-0008 | Open |
| #71 | prover: P2 proof — fixed-candidate planning (V0 deliverable) | planner | p0 | l | RFC-0009 RFC-0000 | Open |
| #72 | cli: pwm prove --statement p2 and planner_config handling | planner | p1 | m | RFC-0009 RFC-0016 | Open |
| #76 | ci: release automation (versioning, changelog, packaging) | ci | p1 | m | RFC-0016 | Open |
| #78 | testing: complete negative-test suite and mutation coverage | testing | p0 | l | RFC-0013 | Open |
| #79 | security: P2 relation security review gate | security | p0 | m | RFC-0000 RFC-0013 | Open |
| #81 | docs: V0 user guide, model card, and claim-boundary document | docs | p1 | m | RFC-0001 | Open |

## Tracking issues

- #2 [Tracking] Third-party vendoring and dependency pinning (v0.1 — Foundations)
- #3 [Tracking] pwm-core fixed-point arithmetic over M31 (v0.1 — Foundations)
- #4 [Tracking] Canonical serialization, commitments, and transcript (v0.1 — Foundations)
- #5 [Tracking] Model manifest and export pipeline (v0.1 — Foundations)
- #6 [Tracking] Range-check and lookup infrastructure (v0.1 — Foundations)
- #7 [Tracking] Tensor memory and wiring AIR (v0.1 — Foundations)
- #8 [Tracking] Linear, matmul, and requantization components (v0.1 — Foundations)
- #9 [Tracking] Nonlinear primitive components (v0.2 — Predictor & Rollout)
- #10 [Tracking] LeWorldModel predictor AIR (v0.2 — Predictor & Rollout)
- #11 [Tracking] Rollout AIR (v0.2 — Predictor & Rollout)
- #12 [Tracking] Prover/verifier CLI, artifact bundle, reproducibility (v0.2 — Predictor & Rollout)
- #13 [Tracking] Fixed-candidate planner proof (P2, the V0 deliverable) (v1.0 — Fixed-Candidate Planning Proof)
- #14 [Tracking] Testing, fuzzing, and audit (v0.1 — Foundations)
- #15 [Tracking] Observability (v0.2 — Predictor & Rollout)
- #16 [Tracking] Security and soundness hardening (v0.1 — Foundations)
- #17 [Tracking] CI, build, and release automation (v0.1 — Foundations)
- #18 [Tracking] Documentation (v1.0 — Fixed-Candidate Planning Proof)
- #19 [Tracking] Future: CEM planner proof (P3 / V2) (Future)
- #20 [Tracking] Future: pixel encoder proof (P4 / V3) (Future)
- #21 [Tracking] Future: recursive / aggregated verification (Future)
- #22 [Tracking] Future: audited zero-knowledge mode (Future)

## Cross-cutting dependencies

- #24 (vendor-stwo) depends on #23
- #25 (vendor-stwo-circuits) depends on #23
- #26 (vendoring-audit-policy) depends on #24, #25
- #27 (core-field-encoding) depends on #24
- #28 (core-bounded-int) depends on #27
- #29 (core-requantize) depends on #28
- #30 (core-tensor) depends on #28
- #31 (core-limb-accumulator) depends on #28
- #32 (core-canonical-serialization) depends on #30
- #33 (core-public-input-digest) depends on #32
- #34 (core-fiat-shamir-transcript) depends on #24, #32
- #35 (core-commitment-scheme) depends on #32
- #36 (export-manifest-schema) depends on #32
- #37 (export-quantizer) depends on #36, #29
- #38 (export-manifest-writer) depends on #37, #35
- #39 (export-golden-vectors) depends on #37
- #40 (export-parity-harness) depends on #39, #29
- #41 (cli-manifest-verify) depends on #38
- #42 (air-logup-core) depends on #24
- #43 (air-range-check) depends on #42
- #44 (air-activation-table) depends on #42
- #45 (air-tensor-memory) depends on #42, #30
- #46 (air-wiring-ops) depends on #45
- #47 (air-linear) depends on #45, #43, #49
- #48 (air-matmul-batched) depends on #47, #31
- #49 (air-requant-component) depends on #43, #29
- #50 (nonlinear-gelu) depends on #44
- #51 (nonlinear-softmax) depends on #44, #49
- #52 (nonlinear-layernorm) depends on #44, #49
- #53 (nonlinear-adaln) depends on #52, #44
- #54 (predictor-action-encoder) depends on #47, #44
- #55 (predictor-mlp) depends on #48, #50
- #56 (predictor-attention) depends on #48, #51
- #57 (predictor-conditional-block) depends on #53, #56, #55
- #58 (predictor-arpredictor) depends on #57, #54
- #59 (predictor-p0-proof) depends on #58, #62, #63
- #60 (rollout-windowing) depends on #58
- #61 (rollout-p1-proof) depends on #60, #59
- #62 (prover-trace-builder) depends on #47, #34
- #63 (verifier-core) depends on #33, #35
- #64 (cli-prove-verify) depends on #62, #63
- #65 (reproducibility-digest) depends on #64
- #66 (observability-core) depends on #23
- #67 (observability-prover-metrics) depends on #66, #62
- #68 (planner-cost-mse) depends on #48, #43
- #69 (planner-argmin) depends on #68
- #70 (planner-batched-rollouts) depends on #61, #68
- #71 (planner-p2-proof) depends on #70, #69
- #72 (planner-cli) depends on #71, #64
- #73 (ci-pipeline) depends on #23
- #74 (docs-build) depends on #73
- #75 (ci-perf-gate) depends on #67, #64
- #76 (release-automation) depends on #73
- #77 (testing-harness-skeleton) depends on #23
- #78 (testing-negative-suite) depends on #71, #77
- #79 (security-review-gate) depends on #71, #80
- #80 (security-soundness-checklist) depends on #23
- #81 (docs-userguide-modelcard) depends on #71, #64
