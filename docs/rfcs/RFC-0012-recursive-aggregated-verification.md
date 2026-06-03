# RFC-0012: Recursive / aggregated verification

- Status: Draft
- Authors: ProvableWorldModel maintainers
- Created: 2026-06-03
- Target milestone: Future

## Summary

A complete fixed-candidate planning proof (P2) for the V0 reference LeWorldModel
configuration is a single monolithic trace covering `S` candidate rollouts, each
of horizon `5`, through a 6-block autoregressive predictor, plus per-candidate
MSE cost and a global argmin. As `S` grows, that single trace becomes the
dominant cost and the limiting factor on memory and proving wall-clock. This RFC
locks the canonical **decomposition and aggregation shape** the project will use
when it eventually splits and recursively combines proofs, and the **canonical
recursive public-input schema** that binds the pieces together, without
attempting to implement recursion in V0. The decomposition is fixed as four proof
classes: proof A (model and weight validity), proof B_s (one candidate rollout),
proof C (cost and argmin over a committed candidate-cost root), and proof D
(recursive aggregation that verifies A, every B_s, and C, and re-exposes the P2
public input). This RFC is deferred beyond V0; it is gated on the V0 monolithic
P2 proof from `docs/rfcs/RFC-0009-fixed-candidate-planner-proof.md` shipping and
on a recursive-verifier cost study. It cites stwo-cairo (Apache-2.0), which ships
both a Rust verifier and a Cairo-language verifier for Circle STARK proofs, as
the prior art for the recursive/on-chain verification path.

## Motivation

The founding analysis flags monolithic-trace size as the reason to consider
splitting proofs and aggregating them
(`docs/feasibility-study.md`, source RFC-012, "A complete planning proof may be
too large as one monolithic trace; split proofs can be aggregated"). The same
analysis establishes that proving only the selected rollout is unsound: a sound
planner proof must prove costs for *all* candidates, or prove a committed
candidate-cost table plus a correct minimum argument
(`docs/feasibility-study.md` §9.5; `docs/spec/06-security.md#soundness-requirements`).
Decomposition that drops candidates, or that lets the prover choose which B_s to
include, would reintroduce exactly that unsoundness. The aggregation shape
therefore is not a free engineering choice: it is constrained by the planner
soundness rule. Locking the shape now — before any recursive code exists —
prevents two failure modes: (1) an ad hoc split that proves a subset of
candidates and silently weakens P2, and (2) an aggregate public input that fails
to re-bind the same commitments P2 binds, allowing a valid aggregate to attest to
a different model, quantization, or planner config than its children.

Concrete scenarios this RFC governs:

- **Scaling candidate count.** A caller wants P2 over `S = 256` candidates.
  Proving 256 horizon-5 rollouts in one trace exceeds the memory target in
  `docs/spec/08-performance-budget.md#scaling`. Decomposition lets each B_s prove
  independently (and in parallel), with proof D binding them.
- **On-chain / external verification.** A downstream consumer wants a single
  small proof verifiable by a constrained verifier (a Cairo-language verifier, an
  L1 contract, or a `no_std` embedded verifier). A recursive proof D that verifies
  many child proofs and exposes one P2 public input is the standard shape for that
  consumer; stwo-cairo demonstrates the Circle STARK verifier-in-Cairo path
  (`docs/feasibility-study.md` §8, §11 RFC-012, verified against upstream at
  2026-06-03).
- **Incremental re-proving.** Changing one candidate action sequence should
  require re-proving one B_s plus C and D, not the entire batch. The decomposition
  makes B_s the unit of incremental work.

This RFC depends on, and must remain consistent with, the statement taxonomy in
`docs/rfcs/RFC-0000-security-model-and-statement-taxonomy.md`, the manifest
binding in `docs/rfcs/RFC-0001-model-manifest-and-export-pipeline.md`, the
fixed-candidate planner in
`docs/rfcs/RFC-0009-fixed-candidate-planner-proof.md`, the serialization and
Fiat-Shamir transcript in
`docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md`, and the vendoring
boundary in `docs/rfcs/RFC-0015-third-party-vendoring-and-pinning.md`.

## Goals

- Lock the four-proof decomposition (A, B_s, C, D) and define, for each, its exact
  statement, public input, witness, and committed outputs.
- Lock the canonical recursive public-input schema (`RecursivePublicInput`) that
  proof D consumes and the aggregate `PublicInput` it re-exposes, so an aggregate
  proof binds the identical model, quantization, planner-config, and
  relation-version commitments that a monolithic P2 proof binds.
- Lock the rule that an aggregate proof D is valid **iff** proof A verifies, *every*
  B_s for `s in 0..S` verifies against the committed candidate-trajectory root, and
  proof C verifies against the committed candidate-cost root — partial aggregation
  is unsound and is rejected.
- Define the relation-id minting rule for recursive statements
  (`pwm.lewm.recursive_aggregation.v<N>`) under the immutability rule of
  `docs/spec/09-release-and-versioning.md#relation-versioning`.
- Enumerate every failure mode of the decomposition/aggregation path with the
  verifier or builder response, cross-referencing
  `docs/spec/04-error-model.md#verifier-rejections`.
- Specify accepting and rejecting tests, deferred to the milestone, that prove the
  aggregate is sound and partial/forged aggregation is rejected
  (`docs/spec/07-testing-strategy.md#negative-tests`).
- State explicitly that V0 ships the monolithic P2 proof only, and define the
  gating dependency and decision owner for un-deferring this RFC.

## Non-Goals

- This RFC does not implement recursion in V0. V0 ships the monolithic P2 proof
  from `docs/rfcs/RFC-0009-fixed-candidate-planner-proof.md`; this RFC is the
  forward-compatible target for a later milestone.
- It does not specify zero-knowledge or hiding for the aggregate. The aggregate is
  a succinct validity proof, identical in privacy posture to V0
  (`docs/spec/06-security.md#privacy-and-zk`). ZK is owned by a separate future
  RFC.
- It does not decide the concrete recursion mechanism inside Stwo (verify-in-AIR
  of a Circle STARK proof vs. a wrapping circuit vs. a Cairo-program verifier
  compiled to an AIR). That is an OPEN QUESTION below with an owner and a
  resolution path; this RFC fixes the decomposition shape and public-input schema
  that any chosen mechanism must satisfy.
- It does not cover CEM (P3) or pixel-encoder (P4) decomposition. Those compose
  with this shape but are owned by
  `docs/rfcs/RFC-0010-cem-planner-proof.md` and
  `docs/rfcs/RFC-0011-pixel-encoder-proof.md`.
- It does not vendor stwo-cairo. stwo-cairo is cited as prior art only; the V0
  vendoring inventory (`docs/rfcs/RFC-0015-third-party-vendoring-and-pinning.md`)
  vendors `stwo` and `stwo-circuits` only.

## Proposed Design

### Decomposition (locked)

The monolithic P2 statement is decomposed into four proof classes. The names A,
B_s, C, D are stable identifiers, mirroring the founding analysis (source
RFC-012). Each class is a self-contained STARK with its own `relation_id`, its own
`PublicInput`, and a committed output that the next class consumes.

```text
Proof A   : model + weight validity
            relation_id = pwm.lewm.weight_validity.v1
            Proves: QuantizedWeights commitment == manifest weights.root, and
            tensor shapes/scales/op order match model_commitment +
            quantization_commitment. Output: A binds (model_commitment,
            quantization_commitment) and exposes nothing else private.

Proof B_s : one candidate rollout (per candidate s, 0 <= s < S)
            relation_id = pwm.lewm.rollout.v1   (same as RFC-0008 P1)
            Proves: trajectory_s == Rollout_Q(latent_history, candidate_actions_s;
            committed weights) for candidate index s, under the committed
            predictor. Output: a per-candidate trajectory commitment
            traj_root_s = commit(final_latent_s) bound to candidate index s and to
            (model_commitment, quantization_commitment).

Proof C   : cost + argmin over committed candidate-cost root
            relation_id = pwm.lewm.cost_argmin.v1
            Proves: for the committed candidate_cost_root C_root over costs
            cost_0..cost_{S-1}, where each cost_s == MSE_Q(final_latent_s,
            goal_latent), selected_cost == cost_{selected_index} and
            cost_s - selected_cost >= 0 for all s, with the smallest-index
            tie-break enforced by cost_s - selected_cost - 1 >= 0 for all
            s < selected_index. Output: (selected_index, selected_cost, C_root).

Proof D   : recursive aggregation
            relation_id = pwm.lewm.recursive_aggregation.v1
            Proves: A verifies; B_s verifies for EVERY s in 0..S and each
            traj_root_s is the s-th leaf of the committed candidate-trajectory
            root T_root; C verifies and each cost_s in C_root is computed from the
            final latent committed in traj_root_s and the public goal_latent.
            Re-exposes a P2-shaped PublicInput.
```

Locked soundness rule (INV-RFC0012-01, below): proof D is valid **iff** A and
*all S* of the B_s and C verify and their committed roots chain. Proof D MUST NOT
accept a subset of B_s. The number of candidates `S` is bound into D's public
input; D's relation enforces that exactly `S` distinct B_s child proofs, with
candidate indices `0..S-1` each present exactly once, are aggregated. This is the
direct translation of the planner soundness rule
(`docs/spec/06-security.md#soundness-requirements`,
`docs/feasibility-study.md` §9.5) into the recursive setting.

### Data flow and lifecycle

```text
manifest + weights ─► Proof A ──► (model_commitment, quantization_commitment)
                                          │
latent_history, candidate_actions_s ─► Proof B_s ──► traj_root_s  (per s)
                                          │                 │
                                          ▼                 ▼
                          T_root = MerkleCommit(traj_root_0..traj_root_{S-1})
                                          │
goal_latent + trajectory finals ─► Proof C ──► (selected_index, selected_cost, C_root)
                                          │
            A, {B_s}, C, T_root, C_root ─► Proof D ──► aggregate Proof + P2 PublicInput
```

Prose statement of the diagram: weights are validated once in A; each candidate
is rolled out independently in B_s, producing a trajectory-final commitment;
those commitments are Merkle-aggregated into a single candidate-trajectory root
T_root; cost and argmin run in C against the candidate-cost root C_root, whose
leaves are derived from the same trajectory finals; proof D verifies A, all B_s,
and C, checks that T_root and C_root chain consistently, and emits one aggregate
proof whose public input is byte-identical in shape to a monolithic P2 public
input.

The Merkle commitment scheme for T_root and C_root is the manifest
`weights.commitment_scheme` already bound by `model_commitment`
(`docs/spec/03-data-model.md#model-manifest`); reusing it avoids introducing a
second unbound commitment scheme. Leaf encoding for T_root and C_root is the
canonical serialization defined in
`docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md`; leaf `s` is
`canonical_bytes(candidate_index=s || traj_root_s)` for T_root and
`canonical_bytes(candidate_index=s || cost_s)` for C_root. Fixing leaf encoding
prevents a second-preimage or index-substitution attack on the aggregation tree.

### Canonical recursive public-input schema (locked)

The aggregate proof D consumes a `RecursivePublicInput` and re-exposes a P2-shaped
`PublicInput` (canonical type from
`docs/spec/03-data-model.md#public-input`, signatures from the data model). The
recursive schema is added to `pwm-core` and is the single source of truth for
recursion; the data-model doc formats and explains it.

```rust
/// Verification key handle for a child relation. Binds the relation_id and the
/// preprocessed-trace / parameter commitments a verifier needs to check a child
/// proof. Opaque bytes; schema-versioned alongside Proof.
/// See docs/rfcs/RFC-0015-third-party-vendoring-and-pinning.md for the Stwo
/// parameter set bound here.
pub struct ChildVerifyingKey {
    pub relation_id: [u8; 32],   // child relation, e.g. pwm.lewm.rollout.v1
    pub vk_commitment: [u8; 32], // commitment to preprocessed columns + STARK params
}

/// Public-input digest of a child proof, as produced by the canonical
/// public-input digest in RFC-0014. Proof D binds the digest, not the full
/// child public input, to keep D's trace bounded.
pub struct ChildPublicInputDigest([u8; 32]);

/// One aggregated child reference inside proof D.
pub struct AggregatedChild {
    pub vk: ChildVerifyingKey,
    pub public_input_digest: ChildPublicInputDigest,
    pub candidate_index: Option<u32>,   // Some(s) for B_s; None for A and C
}

/// Public input consumed by proof D. The aggregate exposes the P2 PublicInput in
/// `exposed`; `children` plus the two roots are what D's relation actually
/// constrains.
pub struct RecursivePublicInput {
    pub relation_id: [u8; 32],                 // pwm.lewm.recursive_aggregation.v1
    pub aggregate_artifact_version: u32,        // schema version of this struct
    pub candidate_count: u32,                   // S; D enforces exactly S B_s
    pub model_validity_child: AggregatedChild,  // proof A
    pub rollout_children: Vec<AggregatedChild>, // proof B_0..B_{S-1}, len == S
    pub cost_argmin_child: AggregatedChild,     // proof C
    pub candidate_trajectory_root: [u8; 32],    // T_root
    pub candidate_cost_root: [u8; 32],          // C_root
    pub exposed: PublicInput,                   // P2-shaped, statement_type = P2FixedCandidatePlanning
}
```

The re-exposed `PublicInput.exposed` MUST satisfy:

- `exposed.statement_type == StatementType::P2FixedCandidatePlanning`.
- `exposed.relation_id` equals the *P2* relation id
  `pwm.lewm.fixed_candidate_planning.v1` (NOT the aggregation relation id), so a
  consumer that accepts a monolithic P2 proof accepts an aggregate D proof for the
  same statement without code changes — the aggregate is observationally
  equivalent at the P2 boundary.
- `exposed.model_commitment`, `exposed.quantization_commitment`, and
  `exposed.planner_config_commitment` equal the commitments bound by proof A and
  by every B_s and by C; D's relation enforces this equality across children.
- `exposed.selected_index` and `exposed.selected_cost` equal the (`selected_index`,
  `selected_cost`) output by proof C.
- `exposed.candidate_actions_commitment` equals the candidate-action commitment
  whose leaves index the B_s by `candidate_index`.

The aggregate is delivered as a `ProofArtifact`
(`docs/spec/03-data-model.md#public-input`; canonical type in the data model),
with `artifact.public_input = exposed` and an aggregate `Proof`. The verifier
entry is the existing one — no new public verifier function:

```rust
// Unchanged public surface; an aggregate artifact verifies through the same path.
pub fn verify(artifact: &ProofArtifact) -> Result<(), VerifyError>;
```

`verify` dispatches on `artifact.public_input.relation_id`. For the P2 relation id
it accepts both a monolithic P2 proof and an aggregate D proof whose internal
relation is `pwm.lewm.recursive_aggregation.v1`; the aggregate proof's bytes carry
the recursion, and the P2-shaped public input is what the caller sees. This keeps
`docs/spec/02-public-api.md#rust-public-api` stable across the V0 → recursion
transition.

### Named invariants

| Invariant | Statement | Enforced by |
| --- | --- | --- |
| INV-RFC0012-01 (total aggregation) | Proof D verifies iff proof A, every B_s for `s in 0..candidate_count`, and proof C each verify; candidate indices `0..S-1` each appear exactly once in `rollout_children`. No subset is accepted. | proof D relation + `verify` dispatch; `docs/spec/06-security.md#soundness-requirements` |
| INV-RFC0012-02 (commitment chaining) | Every child's `model_commitment` and `quantization_commitment` equal `exposed.model_commitment` / `exposed.quantization_commitment`; `traj_root_s` is leaf `s` of `candidate_trajectory_root`; each `cost_s` leaf of `candidate_cost_root` derives from the trajectory final committed in `traj_root_s`. | proof D relation |
| INV-RFC0012-03 (P2 equivalence) | `exposed` is a well-formed P2 `PublicInput` whose `relation_id` is `pwm.lewm.fixed_candidate_planning.v1`; an aggregate D proof and a monolithic P2 proof are interchangeable at the `verify` boundary. | `verify`; `docs/spec/02-public-api.md#stability-policy` |
| INV-RFC0012-04 (vk binding) | Each `ChildVerifyingKey.vk_commitment` binds the child's preprocessed-trace and STARK parameter set; D rejects a child verified under a different vk than the one named in the relation. | proof D relation; `docs/rfcs/RFC-0015-third-party-vendoring-and-pinning.md` |
| INV-RFC0012-05 (relation immutability) | Recursive relation ids (`pwm.lewm.weight_validity.v1`, `pwm.lewm.cost_argmin.v1`, `pwm.lewm.recursive_aggregation.v1`) are immutable; any semantic change mints `.v2`. | `docs/spec/09-release-and-versioning.md#relation-versioning` |
| INV-RFC0012-06 (digest canonicality) | `ChildPublicInputDigest` is the canonical public-input digest from RFC-0014; D binds digests, not raw child public inputs, and the digest function is shared prover/verifier. | `docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md` |

### Failure modes and system response

| Failure mode | System response |
| --- | --- |
| `rollout_children.len() != candidate_count` | `verify` returns `VerifyError` (aggregation arity); `docs/spec/04-error-model.md#verifier-rejections`. INV-RFC0012-01. |
| A candidate index in `0..S` missing, duplicated, or out of range | reject (incomplete/duplicate aggregation); INV-RFC0012-01. |
| A child proof B_s fails to verify under its `vk` | reject; the aggregate is invalid even if all other children pass. INV-RFC0012-01. |
| Proof A absent or fails | reject; weights not bound, the aggregate proves nothing about the model. INV-RFC0012-01. |
| Proof C absent or fails, or `selected_cost != cost_{selected_index}` | reject (cost/argmin failure); same taxonomy as RFC-0009. |
| `traj_root_s` is not leaf `s` of `candidate_trajectory_root` | reject (trajectory-root chaining); INV-RFC0012-02. |
| A child's `model_commitment` / `quantization_commitment` differs from `exposed` | reject (commitment mismatch); INV-RFC0012-02. This blocks aggregating B_s proofs computed under different models. |
| `exposed.relation_id != pwm.lewm.fixed_candidate_planning.v1` | reject (relation mismatch); INV-RFC0012-03; `docs/spec/04-error-model.md#verifier-rejections`. |
| Aggregate proof submitted with an unknown `aggregate_artifact_version` | reject (unsupported schema version); `docs/spec/04-error-model.md#failure-modes`. |
| `ChildVerifyingKey.vk_commitment` does not match the relation D expects | reject (vk binding); INV-RFC0012-04. |
| Two leaves of `candidate_cost_root` collide (second-preimage attempt) | reject; canonical leaf encoding binds `candidate_index`; INV-RFC0012-02, RFC-0014. |
| Prover requests aggregation but recursion is not yet implemented (V0) | builder returns a "feature not enabled" error and points to this RFC's milestone; recursion is behind a compile-time feature flag (see Migration / Rollout). |

### Determinism and concurrency

- B_s proofs are independent and MAY be produced in parallel; the aggregation
  order is fixed by `candidate_index`, not by completion order, so the aggregate
  is bit-for-bit reproducible regardless of scheduling
  (`docs/rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md`).
- The Fiat-Shamir transcript for proof D commits children in `candidate_index`
  order (A first, then B_0..B_{S-1}, then C), following the canonical channel
  ordering of `docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md`. Any
  other order produces a different, rejected transcript.
- All roots (T_root, C_root, vk commitments, public-input digests) are derived by
  the canonical, deterministic functions in RFC-0014; no nondeterministic input
  enters the aggregate.

## Alternatives Considered

**Alternative 1 — Ship recursion in V0 instead of the monolithic P2 proof.**
What it is: make the four-proof decomposition the V0 deliverable, skipping the
monolithic P2 trace. Why considered: it would scale to large `S` immediately and
match the "split proofs can be aggregated" framing of the founding analysis. Why
rejected: V0's purpose is to land the headline P2 statement end-to-end with a full
negative-test suite (`docs/spec/00-overview.md#v0-statement`, milestone `v1.0`).
Recursion adds a verify-in-proof relation (or a Cairo-verifier integration) whose
soundness and cost are unstudied; doing it before the monolithic proof exists
means debugging arithmetization and recursion simultaneously, with no monolithic
reference to differential-test the aggregate against. The monolithic P2 proof is
also the ground-truth oracle for INV-RFC0012-03 (P2 equivalence) — without it
there is nothing to assert equivalence against. Deferring recursion is the
honest, sequenced path the analysis itself recommends
(`docs/feasibility-study.md` §0, §14).

**Alternative 2 — Prove only the selected candidate and a committed candidate set,
no per-candidate B_s proofs.** What it is: a single proof that the selected
candidate's rollout and cost are correct, plus a commitment to the full candidate
set, trusting the argmin off-circuit. Why considered: it is dramatically cheaper —
one rollout instead of `S`. Why rejected: it is unsound. The founding analysis is
explicit that proving only the selected rollout does not prove planning; the
prover can choose a favorable candidate off-circuit
(`docs/feasibility-study.md` §9.5;
`docs/spec/06-security.md#soundness-requirements`). The decomposition exists
precisely to keep "all candidates proven" (INV-RFC0012-01) while splitting the
work; abandoning that constraint defeats the purpose. This alternative is recorded
as a rejected unsound design, not a future option.

**Alternative 3 — Flat batched aggregation without recursion (one verifier checks
S+2 independent proofs directly).** What it is: ship A, B_s, C as independent
artifacts and have the verifier check all of them plus the chaining, with no proof
D and no verify-in-proof. Why considered: it avoids the hardest piece (recursive
verification) entirely and still gives parallel B_s proving. Why rejected: it does
not deliver the on-chain / constrained-verifier use case (a single small proof),
which is the main external motivation; the consumer would have to ship and verify
`S + 2` proofs and re-implement the chaining checks, re-exposing the same
soundness surface in every consumer. It is, however, a legitimate intermediate
deliverable: this RFC permits a flat-aggregation builder as a stepping stone
toward proof D (see Migration / Rollout), because flat aggregation exercises the
chaining invariants (INV-RFC0012-02) without the recursion mechanism. It is
rejected only as the *final* shape.

## Drawbacks

- **Added relation surface.** Three new relation ids
  (`pwm.lewm.weight_validity.v1`, `pwm.lewm.cost_argmin.v1`,
  `pwm.lewm.recursive_aggregation.v1`) must be maintained immutably forever under
  `docs/spec/09-release-and-versioning.md#relation-versioning`. More relations
  means more golden vectors and more negative tests.
- **Recursion overhead.** Verify-in-proof (or a Cairo-verifier AIR) adds proving
  cost per child; for small `S`, the monolithic P2 proof is cheaper than A + S·B_s
  + C + D. The crossover `S` is unknown until the cost study (OPEN QUESTION below).
  Recursion is a win only above that crossover.
- **Larger trusted arithmetization.** The aggregate's soundness now also depends on
  the correctness of the verify-in-proof relation, which is among the most
  error-prone components in any recursive STARK system. This is precisely why it is
  deferred and gated on an audit.
- **Two commitment roots to maintain.** T_root and C_root add commitment plumbing
  that the monolithic proof does not have; a bug in leaf encoding is a soundness
  bug. Mitigated by reusing the manifest commitment scheme and RFC-0014 canonical
  leaf encoding (INV-RFC0012-02).
- **Deferred value.** Because this RFC is `Future`, none of its benefit is realized
  in V0 or V1; callers needing large `S` in the interim must either accept the
  monolithic proof's memory cost or wait.

## Migration / Rollout

- **Feature flag.** Recursion is gated behind a `pwm-verifier`/`pwm-prover`
  Cargo feature `recursion` (default off) and a prover-CLI flag
  `--aggregate` (see `docs/spec/02-public-api.md#cli`). With the feature off, the
  builder returns the "feature not enabled" error above; V0 and V1 ship with it
  off.
- **Relation versioning.** The aggregate exposes `pwm.lewm.fixed_candidate_planning.v1`
  (INV-RFC0012-03), so adding recursion does **not** mint a new P2 relation id and
  does not invalidate existing P2 proofs. The three recursive relation ids are new
  and additive. A consumer pinned to the P2 relation id transparently accepts
  aggregate proofs once `verify` ships recursion support; until then it accepts
  monolithic proofs only.
- **Schema versioning.** `RecursivePublicInput.aggregate_artifact_version` and
  `Proof`'s schema version gate forward compatibility; an unknown version is a hard
  reject, never a best-effort parse
  (`docs/spec/09-release-and-versioning.md#semver`).
- **Staged landing.** (1) Land A, B_s, C as independent relations and a flat
  builder that emits and chains `S + 2` artifacts (Alternative 3, exercises
  INV-RFC0012-02); (2) land proof D recursion behind `recursion`; (3) flip
  `verify` to accept aggregate D proofs at the P2 boundary once differential tests
  against the monolithic proof pass (INV-RFC0012-03). Each stage is independently
  releasable and reversible by toggling the feature flag.
- **No deprecation of the monolithic proof.** The monolithic P2 path remains
  supported indefinitely; recursion is an addition, not a replacement. Both must
  produce the same P2 public input for the same inputs, enforced by a differential
  test (below).

## Testing Strategy

All tests below are deferred to the recursion milestone and run under the
`recursion` feature. They extend the suites in
`docs/spec/07-testing-strategy.md` and use the rejection taxonomy of
`docs/spec/04-error-model.md#verifier-rejections`.

Accepting tests:

- `test_aggregate_accepts_all_children_valid` — A, B_0..B_{S-1}, C all valid and
  chained; `verify(aggregate_artifact)` returns `Ok(())`. Covers INV-RFC0012-01.
  (`docs/spec/07-testing-strategy.md#golden-vectors`)
- `test_aggregate_p2_equivalence_differential` — for the same manifest, latent
  history, goal latent, and candidate set, the monolithic P2 proof and the
  aggregate D proof expose a byte-identical `PublicInput` and both verify. Covers
  INV-RFC0012-03. (`docs/spec/07-testing-strategy.md#differential-tests`)
- `test_aggregate_parallel_order_reproducible` — B_s built in two different
  completion orders yield a bit-identical aggregate proof and public input. Covers
  determinism / INV-RFC0012-06. (`docs/spec/07-testing-strategy.md#golden-vectors`)
- `test_aggregate_verifier_no_std` — the aggregate verifies on the `no_std`
  verifier path (Stwo's `ensure-verifier-no_std` posture, verified against upstream
  at 2026-06-03), demonstrating the constrained-verifier use case.

Rejecting / negative tests
(`docs/spec/07-testing-strategy.md#negative-tests`):

- `test_reject_missing_candidate` — drop B_{S-1}; `verify` rejects (arity).
  INV-RFC0012-01.
- `test_reject_duplicate_candidate_index` — submit B_2 twice and omit B_3; reject
  (duplicate/missing index). INV-RFC0012-01.
- `test_reject_lower_cost_candidate_excluded` — a candidate with strictly lower
  cost exists but its B_s is omitted; reject (incomplete aggregation reproduces the
  RFC-0009 lower-cost rejection). INV-RFC0012-01.
- `test_reject_child_model_commitment_mismatch` — one B_s proven under a different
  `model_commitment`; reject (commitment chaining). INV-RFC0012-02.
- `test_reject_trajectory_root_mismatch` — `traj_root_2` not equal to leaf 2 of
  `candidate_trajectory_root`; reject. INV-RFC0012-02.
- `test_reject_cost_root_leaf_forgery` — substitute a `cost_s` leaf inconsistent
  with the committed trajectory final; reject. INV-RFC0012-02.
- `test_reject_wrong_exposed_relation_id` — `exposed.relation_id` set to the
  aggregation id instead of the P2 id; reject. INV-RFC0012-03.
- `test_reject_unknown_aggregate_artifact_version` — bump
  `aggregate_artifact_version` to an unsupported value; reject (schema version).
- `test_reject_child_vk_substitution` — verify a child under a `vk` other than the
  one named in D's relation; reject. INV-RFC0012-04.
- `test_reject_argmin_violation_in_C` — proof C with `selected_cost` not minimal;
  C fails, D rejects (reuses RFC-0009 argmin negative vectors).
- Constraint-mutation tests (`docs/spec/07-testing-strategy.md#mutation-tests`):
  mutate one constraint in the verify-in-proof relation of D and confirm at least
  one negative test flips from reject to accept (mutation caught), enforcing that
  D's relation is not vacuous.

## Open Questions

- OPEN QUESTION: The concrete recursion mechanism inside the vendored Stwo stack —
  verify-a-Circle-STARK-proof inside an AIR vs. a Cairo-program verifier compiled
  to an AIR (the stwo-cairo path) vs. a wrapping circuit — is not decided here.
  Owner role: prover/verifier lead (`area:verifier`). Resolution path: a
  recursive-verifier cost-and-soundness study landing as a successor RFC before
  this RFC moves from `Draft` to `Accepted`; the study must report the crossover
  `S` above which recursion beats the monolithic proof
  (`docs/spec/08-performance-budget.md#scaling`).
- OPEN QUESTION: Whether to vendor any part of stwo-cairo (Apache-2.0) for the
  Cairo-verifier path, or keep it reference-only. Owner role: core maintainer
  (`area:core`). Resolution path: amend
  `docs/rfcs/RFC-0015-third-party-vendoring-and-pinning.md` if the cost study
  selects the Cairo-verifier mechanism; otherwise it stays reference-only as stated
  in Non-Goals.
- OPEN QUESTION: The trigger to un-defer this RFC. Owner role: project lead.
  Resolution path: open the work when the monolithic P2 proof from
  `docs/rfcs/RFC-0009-fixed-candidate-planner-proof.md` has shipped at milestone
  `v1.0` and a caller requires `S` above the (to-be-measured) crossover; tracked
  under the `Future` milestone.

## References

- `docs/feasibility-study.md` — source RFC-012 (decomposition shape), §0
  (sequencing verdict), §9.5 (planner soundness), §8 and §11 (stwo-cairo /
  recursion prior art), §14 (V0 statement scope).
- `docs/spec/00-overview.md#v0-statement`, `#scope-and-statement-tiers`,
  `#feasibility-verdict` — V0 is the monolithic P2 proof; recursion is deferred.
- `docs/spec/02-public-api.md#rust-public-api`, `#cli`, `#stability-policy` — the
  unchanged `verify` surface and the `--aggregate` CLI flag.
- `docs/spec/03-data-model.md#public-input`, `#model-manifest` —
  `PublicInput`/`ProofArtifact` canonical types, manifest commitment scheme.
- `docs/spec/04-error-model.md#verifier-rejections`, `#failure-modes` — rejection
  taxonomy for aggregation failures.
- `docs/spec/06-security.md#soundness-requirements`, `#privacy-and-zk` — planner
  soundness rule and the no-ZK posture inherited by the aggregate.
- `docs/spec/07-testing-strategy.md#golden-vectors`, `#negative-tests`,
  `#differential-tests`, `#mutation-tests` — test suites the recursion tests
  extend.
- `docs/spec/08-performance-budget.md#scaling`, `#cost-model` — the crossover-`S`
  study and trace-size scaling with candidates.
- `docs/spec/09-release-and-versioning.md#relation-versioning`, `#semver` —
  relation immutability and schema versioning of the recursive ids.
- `docs/rfcs/RFC-0000-security-model-and-statement-taxonomy.md` — P0–P4 taxonomy
  and the exact-relation-id rule.
- `docs/rfcs/RFC-0008-rollout-air.md` — the B_s rollout relation
  (`pwm.lewm.rollout.v1`).
- `docs/rfcs/RFC-0009-fixed-candidate-planner-proof.md` — the monolithic P2 proof
  this RFC decomposes; the cost/argmin relation reused by proof C.
- `docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md` — canonical
  public-input digest, leaf encoding, and Fiat-Shamir channel ordering for proof D.
- `docs/rfcs/RFC-0015-third-party-vendoring-and-pinning.md` — vendored `stwo` /
  `stwo-circuits` boundary and Stwo parameter set bound by `ChildVerifyingKey`.
- `docs/rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md` — bit-for-bit
  reproducibility of the aggregate across parallel B_s scheduling.
- stwo-cairo (github.com/starkware-libs/stwo-cairo, Apache-2.0; verified against
  upstream at 2026-06-03) — a Circle STARK prover/verifier for the Cairo CPU
  architecture providing both a Rust verifier and a Cairo-language verifier; prior
  art for recursive / on-chain verification. Reference only; not vendored for V0.
- `stwo` (github.com/starkware-libs/stwo, Apache-2.0, workspace v2.2.0; verified
  against upstream at 2026-06-03) — `no_std` verifier path and Fiat-Shamir channels
  used by the recursive verifier.
