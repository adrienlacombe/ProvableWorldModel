# 00 — Overview: Thesis, Goals, Non-Goals, Success Criteria, Scope

Status: Normative. Role: the entry point and scope-defining document for the ProvableWorldModel specification corpus. Every other normative document refines a concern this file names; where they conflict, the precise document governs that concern and this file governs scope and intent.

This document defines what ProvableWorldModel proves, what it deliberately does not prove, the tiered proof statements (P0–P4), the release versions (V0–V3), the feasibility verdict, and the verbatim V0 public statement. It is binding on contributors and reviewers. It does not define types (see [03-data-model.md](03-data-model.md#model-manifest)), the threat model (see [06-security.md](06-security.md#threat-model)), or the architecture (see [01-architecture.md](01-architecture.md#crate-layout)).

---

## Thesis
<a id="thesis"></a>

ProvableWorldModel is a succinct-proof system that proves **deterministic, quantized inference of a JEPA-style world model (LeWorldModel)** as an exact arithmetic relation, using a custom Circle-STARK arithmetization (AIR) over the Mersenne-31 field (M31), built on a vendored Stwo prover.

The single load-bearing claim is precise and narrow:

> Given a committed quantized LeWorldModel predictor manifest, an initial latent history, a goal latent, and S fixed candidate action sequences, a verified proof establishes that every candidate was rolled out through the committed quantized predictor, every final-latent cost was computed as specified, and the selected candidate has minimum cost under deterministic tie-breaking.

The proof is over a fully specified fixed-point computation, not over floating-point PyTorch behavior. Three boundaries are non-negotiable and are restated wherever the temptation to overclaim recurs:

| Boundary | The system proves | The system does NOT prove |
| --- | --- | --- |
| Computation vs. truth | The committed quantized model produced the claimed latents/costs/selection. | That the predicted future is real, or that the model is correct about the world. |
| Fixed-point vs. float | An exact integer fixed-point relation defined by the manifest. | That floating-point PyTorch on a GPU in bf16 would produce the same bits. |
| Validity vs. privacy | A succinct validity statement about the relation. | Zero-knowledge / hiding of weights or inputs (unless a separate hiding audit is passed). |

This narrowing is the design. The source feasibility analysis is explicit that the project is "technically feasible only if the proof statement is made exact, deterministic, and quantized" (see [feasibility-study.md](../feasibility-study.md) §0). Any statement that drops one of "exact", "deterministic", or "quantized" is out of scope for the proof system and must not appear in a public claim. The mechanism by which the manifest makes the relation exact is specified in [03-data-model.md](03-data-model.md#model-manifest); the soundness obligations that keep it honest are in [06-security.md](06-security.md#soundness-requirements).

Stwo is an appropriate base because it exposes a custom-AIR-oriented Circle STARK stack over M31: an M31 field module (`P = 2^31 - 1`), CM31 and QM31 extensions, FRI and a polynomial commitment scheme, Fiat-Shamir channels (Blake2s / Keccak256 / Poseidon252 backends), a LogUp constraint framework, and a verifier path that is `no_std`-compatible (verified against upstream at 2026-06-03). The vendoring policy, pinned revisions, and audit boundary are locked in [RFC-0015](../rfcs/RFC-0015-third-party-vendoring-and-pinning.md).

### Invariants asserted by this document

| Invariant | Statement |
| --- | --- |
| INV-OVERVIEW-01 | A proof asserts only the exact, deterministic, quantized arithmetic relation named by its `relation_id`; it asserts nothing about physical truth, float equivalence, or privacy. |
| INV-OVERVIEW-02 | A proof is valid only for its declared `relation_id`. `relation_id`s are immutable; any semantic change to a statement mints a new id (`pwm.lewm.<statement>.v<N>`). Cross-relation reuse of a proof is rejected. See [06-security.md](06-security.md#soundness-requirements) and [RFC-0000](../rfcs/RFC-0000-security-model-and-statement-taxonomy.md). |
| INV-OVERVIEW-03 | Everything that determines the relation's value is bound by `model_commitment` and `quantization_commitment`. Anything unbound is mutable by the prover and therefore unsound; it must not exist in V0. See [03-data-model.md](03-data-model.md#model-manifest). |
| INV-OVERVIEW-04 | The verifier never executes PyTorch, Python, ONNX runtime, or floating-point inference. It checks the arithmetic relation only. Python lives exclusively in the export and reference-inference pipeline. See [01-architecture.md](01-architecture.md#data-flow). |
| INV-OVERVIEW-05 | The term "zero-knowledge" is never applied to V0. V0 is a succinct validity proof. ZK is an optional future mode gated on a hiding audit. See [06-security.md](06-security.md#privacy-and-zk). |

---

## Goals
<a id="goals"></a>

Goals are stated so each is independently testable. The owning document for the test is named.

| ID | Goal | How it is tested | Owner doc |
| --- | --- | --- | --- |
| G1 | Prove an exact fixed-point relation, not float behavior. | Rust fixed-point inference matches the Python fixed-point reference bit-for-bit on golden vectors. | [07-testing-strategy.md](07-testing-strategy.md#golden-vectors) |
| G2 | Bind the entire model + quantization definition into the public commitments. | Mutating any bound field (a weight, a scale, a rounding mode, a lookup table) changes a commitment and the verifier rejects. | [06-security.md](06-security.md#binding-requirements) |
| G3 | Range-safe arithmetic: no value relied on as an integer may silently wrap mod p. | A witness that wraps the field but breaks integer semantics is rejected by a range/limb check. | [RFC-0002](../rfcs/RFC-0002-fixed-point-arithmetic-over-m31.md), [06-security.md](06-security.md#soundness-requirements) |
| G4 | Prove the LeWorldModel predictor step (P0) over the exported quantized graph. | A P0 proof verifies; changing the action, the latent history, or the claimed output is rejected. | [RFC-0007](../rfcs/RFC-0007-leworldmodel-predictor-air.md) |
| G5 | Prove autoregressive latent rollout over a configurable horizon (P1). | A P1 proof verifies for horizon = 5; mutating any intermediate predicted latent is rejected. | [RFC-0008](../rfcs/RFC-0008-rollout-air.md) |
| G6 | Prove fixed-candidate planning end-to-end (P2 = V0): all rollouts, all costs, argmin with deterministic tie-break. | The headline P2 proof verifies; a lower-cost unselected candidate, an out-of-range index, a mismatched `selected_cost`, or a tie-break violation are each rejected. | [RFC-0009](../rfcs/RFC-0009-fixed-candidate-planner-proof.md), [07-testing-strategy.md](07-testing-strategy.md#negative-tests) |
| G7 | Deterministic, byte-identical export. | Exporting the same checkpoint twice yields a byte-identical manifest and weight tensors; `canonical_json_hash` is stable. | [RFC-0001](../rfcs/RFC-0001-model-manifest-and-export-pipeline.md) |
| G8 | A canonical Fiat-Shamir transcript and public-input digest shared bit-for-bit by prover and verifier. | Prover and verifier derive identical challenges in identical channel order; a reordered transcript is rejected. | [RFC-0014](../rfcs/RFC-0014-canonical-serialization-and-transcript.md) |
| G9 | Every shipped component has both accepting and rejecting tests. | CI gate blocks merge of any AIR component lacking a negative test. | [07-testing-strategy.md](07-testing-strategy.md#ci-gates), [RFC-0013](../rfcs/RFC-0013-testing-fuzzing-and-audit-strategy.md) |
| G10 | A vendored, pinned, auditable proving substrate. | `third_party/stwo/REVISION` records the exact pinned rev; the build uses the vendored copy, not upstream crates. | [RFC-0015](../rfcs/RFC-0015-third-party-vendoring-and-pinning.md) |
| G11 | Reproducible proof artifacts. | Same inputs produce the same proof inputs and the same claimed outputs (the bit-for-bit reproducibility contract). | [RFC-0016](../rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md) |

Goals G1–G3 and G7–G11 are V0 infrastructure goals (milestone `v0.1`); G4–G6 are the V0 proof goals (milestones `v0.2` and `v1.0`).

---

## Non-Goals
<a id="non-goals"></a>

Each non-goal is excluded with a reason and the version (if any) where it may re-enter scope. Exclusion is a soundness and tractability decision, not an oversight.

| ID | Non-goal (NOT in V0) | Reason for exclusion | Re-enters at |
| --- | --- | --- | --- |
| NG1 | Floating-point / bf16 / GPU PyTorch equivalence. | A float kernel is not a clean arithmetic relation; proving it would require formally specifying every kernel, accumulation order, and rounding rule. The risk register rates this Critical. | Never as stated; only the exact fixed-point relation is proven. |
| NG2 | Physical-truth claims (the predicted future is real; the action is environment-optimal; the model is calibrated). | These are not properties of the computation and cannot be established by a STARK over the inference relation. | Never (separate, non-cryptographic claims). |
| NG3 | Zero-knowledge / weight or input hiding. | A validity proof is not automatically hiding; trace commitments, lookup tables, and public outputs may leak. Calling V0 "ZK" would be an overclaim. | V1, only after a hiding audit. See [06-security.md](06-security.md#privacy-and-zk). |
| NG4 | The CEM sampling/search loop (seeded sampling, top-k, mean/var updates, iteration recurrence). | Proving the full optimizer is materially harder than proving fixed-candidate scoring; Gaussian sampling and square roots are expensive in-field. V0 proves scoring + selection over a fixed candidate set. | V2 (P3). See [RFC-0010](../rfcs/RFC-0010-cem-planner-proof.md). |
| NG5 | The pixel encoder (ViT over 224x224 images, patch 14). | ViT patch/token attention from pixels dominates proving cost and is far larger than the latent predictor. V0 assumes latents are valid inputs. | V3 (P4). See [RFC-0011](../rfcs/RFC-0011-pixel-encoder-proof.md). |
| NG6 | Training, the SIGReg loss (Sketch Isotropic Gaussian Regularizer, weight 0.09), AdamW, dataloading, and the config framework. | The proof concerns inference of a fixed checkpoint, not how it was trained. Training artifacts are context only. | Never (out of proving scope by definition). |
| NG7 | Recursive / aggregated verification. | A monolithic trace may suffice for V0 scale; decomposition and aggregation need separate design and benchmarking. | V1/V2. See [RFC-0012](../rfcs/RFC-0012-recursive-aggregated-verification.md). |
| NG8 | Dependence on upstream canonical Stwo / stwo-circuits crates. | Reproducibility and audit require pinned, vendored, modifiable copies under `third_party/`. | Never; vendoring is the policy. See [RFC-0015](../rfcs/RFC-0015-third-party-vendoring-and-pinning.md). |
| NG9 | Proving only the selected candidate's rollout (a "fast path" planner proof). | Proving the winner alone does not prove it is the winner; the prover could pick favorable candidates off-circuit. All candidates must be scored. | Never; this is a soundness rule, not a feature. See [06-security.md](06-security.md#soundness-requirements). |
| NG10 | Off-circuit nonlinearities (prover-supplied softmax probabilities, unconstrained reciprocal/inverse-sqrt, float helper values). | Any value not constrained by the AIR is a soundness hole. | Never; rejected by design in [RFC-0006](../rfcs/RFC-0006-nonlinear-primitive-components.md). |

OPEN QUESTION (owner: maintainers, area:air): for V0 the predictor's nonlinear primitives may be implemented as committed in-field approximations of the original LeWorldModel architecture (lookup/rational, retaining the checkpoint) or as a proof-native distilled predictor that replaces LayerNorm/softmax with cheaper proof-friendly operators. Both are sound; they differ in which model the proof is about. Resolution path: [RFC-0006](../rfcs/RFC-0006-nonlinear-primitive-components.md) and [RFC-0007](../rfcs/RFC-0007-leworldmodel-predictor-air.md), to be locked before milestone `v0.2`. Whichever is chosen, the proven artifact is named `QuantizedLeWM-v1`, not "the exact PyTorch model".

---

## Success Criteria
<a id="success-criteria"></a>

V0 (= P2, milestone `v1.0`) is "done" when, and only when, all of the following hold. Each criterion is objective and maps to a test owner.

| ID | Success criterion | Verification | Owner doc |
| --- | --- | --- | --- |
| SC1 | The headline P2 proof works end-to-end: prove fixed-candidate planning, verify acceptance. | A `prove_planning` run over S candidates produces a `ProofArtifact`; `verify(&artifact)` returns `Ok(())`. | [RFC-0009](../rfcs/RFC-0009-fixed-candidate-planner-proof.md), [02-public-api.md](02-public-api.md#rust-public-api) |
| SC2 | Every negative test rejects. The full negative-test suite (wrong commitment, wrong action, wrong latent, wrong intermediate activation, wrong accumulator, wrong rounding remainder, wrong lookup value, wrong cost, wrong argmin, wrong tie-break, wrong relation_id) yields a `VerifyError`. | Each negative fixture returns `Err(VerifyError::…)`; none verifies. | [07-testing-strategy.md](07-testing-strategy.md#negative-tests), [04-error-model.md](04-error-model.md#verifier-rejections) |
| SC3 | Export is byte-identical on re-export. | Re-exporting the same checkpoint with the same config yields byte-identical manifest + weight tensors; `model_commitment` and `quantization_commitment` are unchanged. | [RFC-0001](../rfcs/RFC-0001-model-manifest-and-export-pipeline.md) |
| SC4 | Rust fixed-point inference matches the Python fixed-point reference bit-for-bit. | For every golden vector, the Rust reference and the Python reference produce identical integer outputs for every op, latent, cost, and the selected index. | [07-testing-strategy.md](07-testing-strategy.md#differential-tests), [RFC-0013](../rfcs/RFC-0013-testing-fuzzing-and-audit-strategy.md) |

V0 is NOT done if any negative test verifies (a soundness failure outranks all functional progress), if export is non-deterministic, or if the two references diverge on any bit. The performance budget (proving time, trace size, memory) for V0 is a separate gate tracked in [08-performance-budget.md](08-performance-budget.md#targets); a slow-but-correct V0 satisfies the soundness criteria above and is acceptable for the first deliverable.

INV-OVERVIEW-06: a release tagged V0 must satisfy SC1–SC4 simultaneously. Partial satisfaction (e.g. acceptance works but a negative test verifies) is a release blocker, not a known limitation.

---

## Scope and Statement Tiers
<a id="scope-and-statement-tiers"></a>

The proof system is organized into five immutable statement tiers. The tier identifiers (P0–P4) and the `StatementType` enum that names them are fixed; see the canonical type in [03-data-model.md](03-data-model.md#public-input). What each tier *binds* is what its proof actually establishes.

| Tier | `StatementType` | `relation_id` family | What it binds (the proven claim) | Public inputs (beyond the four commitments) |
| --- | --- | --- | --- | --- |
| P0 | `P0Step` | `pwm.lewm.predictor_step.v1` | One predictor step: `claimed_z_next == PredProj_Q(ARPredictor_Q(z_history, ActionEncoder_Q(actions)))` over the committed quantized graph. | `latent_history`, action(s), `claimed_output_commitment`. |
| P1 | `P1Rollout` | `pwm.lewm.rollout.v1` | Autoregressive rollout: for each step `t`, `z_{t+1} == Predict_Q(window(z), window(a))`, and the trajectory is consistent with the recurrence. | initial `latent_history`, `action_sequence`, claimed trajectory root. |
| P2 (V0) | `P2FixedCandidatePlanning` | `pwm.lewm.fixed_candidate_planning.v1` | For all S candidates: `traj_s == Rollout_Q(...)`, `cost_s == MSE_Q(final_s, goal)`, `selected_cost == cost[selected_index]`, `selected_cost <= cost_s` for all s, and smallest-index tie-break. | `latent_history`, `goal_latent`, `candidate_actions`, `selected_index`, `selected_cost`. |
| P3 | `P3Cem` | `pwm.lewm.cem.v1` | The full CEM planner: seeded sampling, candidate clipping, all costs, top-k, mean/variance updates, iteration recurrence, deterministic final selection. | `planner_config_commitment` (binds CEM params), seed, selected sequence. |
| P4 | `P4PixelToPlan` | `pwm.lewm.pixel_to_plan.v1` | Encoder + predictor + planner end-to-end: `z_history == Encoder_Q(pixels)`, `z_goal == Encoder_Q(goal)`, selection == `Planner_Q(z_history, z_goal)`. | pixel observation history, pixel goal, selected sequence. |

P2 composes P0 (one step) and P1 (rollout) with MSE cost and argmin; it is the smallest statement that is useful as a planning proof and the cleanest one that is fully sound. P0 and P1 are not separately shipped as public deliverables — they are the building blocks proven internally on the way to P2 (milestones `v0.2`), and their proofs are exercised by the test suite.

### Versions

Versions are release bundles; tiers are statements. The mapping is fixed:

| Version | Tier | Adds over prior version | Milestone(s) |
| --- | --- | --- | --- |
| V0 | P2 | First public deliverable: fixed-candidate planning, single-rollout-at-a-time correctness, full negative suite. | `v0.1` (foundations) + `v0.2` (predictor & rollout) + `v1.0` (planning proof) |
| V1 | P2 (batched/efficient) | Batched candidate rollouts at scale, optional ZK after a hiding audit, recursion/aggregation. | Future |
| V2 | P3 | The CEM planner proof (seeded sampling, top-k, distribution updates). | Future |
| V3 | P4 | The pixel encoder (ViT) and full pixel-to-plan composition. | Future |

INV-OVERVIEW-07: a `StatementType` and its `relation_id` family are immutable identifiers. A proof generated for one tier MUST be rejected if submitted as another (e.g. a P0 proof submitted as P1). This is enforced by the verifier checking `relation_id` first; see [04-error-model.md](04-error-model.md#verifier-rejections).

### V0 model scope (the QuantizedLeWM-v1 surface)

The V0 reference configuration targets the LeWorldModel checkpoint with `latent_dim = embed_dim = 192`, `history_size = 3`, predictor `depth = 6`, `heads = 16`, `dim_head = 64`, `mlp_dim = 2048`, `num_preds = 1`, approximately 15M trainable parameters (verified against upstream at 2026-06-03; le-wm is MIT-licensed, consumed as a checkpoint + config, not vendored). The predictor is `ARPredictor` (autoregressive) with learned positional embeddings, `ConditionalBlock` using AdaLN-zero modulation, a FeedForward/MLP, and `F.scaled_dot_product_attention` with action conditioning. Activation functions are not uniform: the predictor FFN/MLP uses **GELU**, while AdaLN modulation and the action `Embedder` use **SiLU**; each AIR component states its own activation and the manifest binds it per op. The full dimensional and architectural detail belongs in [01-architecture.md](01-architecture.md#component-model) and [03-data-model.md](03-data-model.md#model-manifest); this overview fixes only the scope boundary.

V0 supports, as the proven `QuantizedLeWM-v1` relation:

```text
action_encoder      (Embedder, SiLU)
predictor           (ARPredictor: ConditionalBlock x6, attention, FFN with GELU, AdaLN with SiLU)
pred_proj           (prediction projection)
rollout recurrence  (autoregressive windowing, horizon = 5, action_block = 5)
goal-latent MSE     (F.mse_loss equivalent over the final predicted latent)
argmin over fixed candidates (deterministic smallest-index tie-break)
```

V0 excludes (re-stated as a scope fence; see [Non-Goals](#non-goals)): the pixel encoder, training loss, SIGReg, PyTorch dataloading, the config framework, and the CEM sampling/update loop. The CEM solver values that V0 does NOT prove but that V2 will (`num_samples = 300`, `n_steps = 30`, `topk = 30`, `var_scale = 1.0`, `batch_size = 1` from the solver config; `horizon = 5`, `receding_horizon = 5`, `action_block = 5` from the task `plan_config`) are recorded here for scope clarity and consumed in [RFC-0010](../rfcs/RFC-0010-cem-planner-proof.md) and [08-performance-budget.md](08-performance-budget.md#scaling).

The defensible recommended build order (source §0, §2, §15) is: prove one fixed-point predictor step, then one rollout, then batch rollouts, then candidate selection, then — only later — the CEM optimizer and the pixel encoder.

---

## Feasibility Verdict
<a id="feasibility-verdict"></a>

The project is feasible **as the exact, deterministic, quantized relation defined above** and infeasible (or unsound) as a float-equivalence or whole-truth claim. The dominant engineering problem is not STARK soundness but arithmetizing a transformer-like predictor economically: a single predictor call already contains millions of multiply-accumulate operations before rollout horizon, candidate count, or CEM iterations multiply it. A worst-case length-2048 signed int8 dot product has magnitude `2048 * 127 * 127 = 33,032,192 < 2^31 - 1`, so int8 MLP accumulations fit in one signed M31 value; wider ranges force limb decomposition. The fixed-point and limb policy is locked in [RFC-0002](../rfcs/RFC-0002-fixed-point-arithmetic-over-m31.md) and the linear/matmul accumulator strategy in [RFC-0005](../rfcs/RFC-0005-linear-matmul-and-requantization-components.md).

The verdict table reproduces and tightens source §2. "Feasibility" is engineering tractability; "Soundness status" is whether the relation can be made sound; "Recommendation" is the version gate.

| Scope | Feasibility | Soundness status | Recommendation |
| --- | --- | --- | --- |
| One quantized linear / MLP layer | High | Sound with range checks and exact fixed-point semantics ([RFC-0002](../rfcs/RFC-0002-fixed-point-arithmetic-over-m31.md), [RFC-0005](../rfcs/RFC-0005-linear-matmul-and-requantization-components.md)). | Build first (`v0.1`). |
| One quantized predictor step (P0) | Medium | Sound only once attention, LayerNorm/AdaLN, GELU (FFN) / SiLU (AdaLN, Embedder), and action conditioning are formalized as committed in-field operators ([RFC-0006](../rfcs/RFC-0006-nonlinear-primitive-components.md), [RFC-0007](../rfcs/RFC-0007-leworldmodel-predictor-air.md)). | V0 core (`v0.2`). |
| Latent rollout for one action sequence (P1) | Medium | Sound once the autoregressive recurrence and rounding are fixed and the trajectory is committed ([RFC-0008](../rfcs/RFC-0008-rollout-air.md)). | V0 (`v0.2`). |
| Fixed-candidate planning with argmin (P2) | Medium | Sound; cost scales linearly in candidates x horizon. Proving only the winner is unsound — all candidates must be scored ([RFC-0009](../rfcs/RFC-0009-fixed-candidate-planner-proof.md)). | V0 / V1. |
| Full CEM proof (P3) | Low-to-medium | Possible but heavy: must prove seeded sampling, clipping, all costs, top-k, and distribution updates; Gaussian sampling and square roots are expensive in-field ([RFC-0010](../rfcs/RFC-0010-cem-planner-proof.md)). | V2 only. |
| Pixel encoder proof (P4) | Low-to-medium | Possible but expensive: ViT patch/token attention over 224x224, patch 14, dominates cost ([RFC-0011](../rfcs/RFC-0011-pixel-encoder-proof.md)). | V3. |
| Floating-point PyTorch equivalence | Low | Not sound unless every kernel and rounding rule is formally specified; out of scope by design. | Do not claim ([Non-Goals](#non-goals) NG1). |
| Zero-knowledge weight privacy | Unresolved until audited | A validity proof is not automatically hiding; requires a separate hiding audit. | Optional security mode, V1+ ([06-security.md](06-security.md#privacy-and-zk)). |

INV-OVERVIEW-08: feasibility is conditional on every nonlinear primitive being a committed, in-field, range-bounded operator. An off-circuit or unconstrained nonlinearity changes the verdict from "sound" to "unsound" regardless of engineering effort. This condition is enforced by the manifest binding (INV-OVERVIEW-03) and the approximation requirements in [06-security.md](06-security.md#soundness-requirements).

The condensed risk picture (full register in [06-security.md](06-security.md#threat-model)): the Critical risks are floating-point ambiguity (mitigated by the exported fixed-point model), field wraparound (mitigated by range-checking every integer interpretation), unsound planner claims (mitigated by proving all candidates), ZK overclaim (mitigated by never calling V0 ZK), and model-commitment gaps (mitigated by binding the full manifest). The High-cost-but-tractable risks are softmax, LayerNorm, CEM explosion, and the encoder — all deferred or approximated, never silently dropped.

---

## V0 Statement
<a id="v0-statement"></a>

The following is the canonical V0 public statement. It is reproduced verbatim from the feasibility analysis (source §14). Public materials describing V0 MUST use this statement or a strict subset of it, and MUST NOT add claims it does not contain.

```text
This proof verifies quantized LeWorldModel latent planning for a fixed candidate set.

Given:
  - a committed QuantizedLeWM predictor manifest,
  - an initial latent history,
  - a goal latent,
  - S candidate action sequences,

the proof verifies that:
  - every candidate was rolled out through the committed quantized predictor,
  - every final latent cost was computed as specified,
  - the selected candidate has minimum cost under deterministic tie-breaking.
```

V0 public materials MUST NOT claim any of the following unless the corresponding relation is actually included in the shipped proof:

```text
proves the original PyTorch model
proves the true future
proves full CEM
proves zero-knowledge privacy
proves end-to-end pixel planning
```

INV-OVERVIEW-09: any public claim about V0 is a subset of the verbatim V0 statement above. A claim that exceeds it (for example, asserting float equivalence or ZK) is a specification violation and a disclosure-relevant misrepresentation; see [06-security.md](06-security.md#disclosure).

---

## Where to read next

| If you need… | Read |
| --- | --- |
| Crate layout, trace model, direct-AIR vs circuit-lowering | [01-architecture.md](01-architecture.md#crate-layout) |
| Public Rust APIs, CLI, artifact bundle | [02-public-api.md](02-public-api.md#rust-public-api) |
| Manifest schema, `PublicInput`, `Witness`, `BoundedInt`, `Tensor`, schema versioning | [03-data-model.md](03-data-model.md#model-manifest) |
| Error taxonomy and verifier rejection codes | [04-error-model.md](04-error-model.md#error-taxonomy) |
| Logging, metrics, redaction of private witnesses | [05-observability.md](05-observability.md#redaction) |
| Threat model, soundness, binding, range-safety, the ZK boundary | [06-security.md](06-security.md#threat-model) |
| Test pyramid, golden vectors, negative/differential/mutation tests, CI gates | [07-testing-strategy.md](07-testing-strategy.md#test-pyramid) |
| Proving cost model, trace sizes, scaling with candidates x horizon | [08-performance-budget.md](08-performance-budget.md#cost-model) |
| Semver, relation_id immutability, MSRV, license, deprecation | [09-release-and-versioning.md](09-release-and-versioning.md#relation-versioning) |
| Canonical term definitions | [10-glossary.md](10-glossary.md) |
| The full RFC set (RFC-0000 … RFC-0016) | [docs/rfcs/](../rfcs/) — start with [RFC-0000](../rfcs/RFC-0000-security-model-and-statement-taxonomy.md) (statement taxonomy) |
| The founding analysis | [docs/feasibility-study.md](../feasibility-study.md) |
