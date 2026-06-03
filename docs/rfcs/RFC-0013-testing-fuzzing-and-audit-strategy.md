# RFC-0013: Testing, fuzzing, and audit strategy

- Status: Accepted
- Authors: ProvableWorldModel maintainers
- Created: 2026-06-03
- Target milestone: v0.1

## Summary

This RFC locks the ProvableWorldModel verification stack: a fixed ten-layer test
hierarchy, the canonical negative-test list every AIR component must satisfy,
the use of differential testing (Python fixed-point reference versus Rust
fixed-point reference versus the proven AIR) and constraint-mutation testing as
mandatory soundness checks, and one hard ship rule — **no component, gate,
manifest field, or proof statement merges to a release branch without BOTH an
accepting test (a valid witness proves and verifies) and a rejecting test (a
named tampered witness is rejected with the documented error)**. It further locks
a **security-review gate**: no public soundness claim about any relation_id may
be made until that relation has passed the layer-10 review recorded in
`docs/spec/06-security.md#disclosure`. The decision is permanent; this document
defines the spec section `docs/spec/07-testing-strategy.md` elaborates, and the
two are kept consistent by the changelog discipline in
`docs/spec/09-release-and-versioning.md#changelog`.

## Motivation

The founding analysis is unambiguous that the dominant failure mode of a
proof-of-ML system is not a broken STARK but a constraint set that *accepts a
witness that violates the intended integer semantics* — a finite field wraps
where integer inference must not (`docs/feasibility-study.md` §9.3), an
accumulator overflows yet lands on a satisfying field value
(`docs/feasibility-study.md` §5.2), an off-circuit softmax probability is
supplied unconstrained (`docs/feasibility-study.md` §6.3), or the prover proves
only the selected planning candidate
(`docs/feasibility-study.md` §9.5). These are *silent* soundness holes:
positive tests pass, demos work, and the system is unsound. The source RFC-013
(`docs/feasibility-study.md` §13) enumerates a ten-layer stack and a list of
required negative tests and asserts "No component is complete unless it has
both accepting tests for valid witnesses and rejecting tests for invalid
witnesses." This RFC makes that rule binding and operational.

Concrete scenarios this strategy must catch before any V0 public claim:

- A linear-component constraint that forgets to range-check the accumulator;
  the field value is correct but the integer overflowed (P0 unsound).
- A requantization gadget whose remainder constraint is `0 <= rem` but not
  `rem < 2^r`; rounding is forgeable (P0/P1 unsound).
- An argmin component that constrains `selected_cost <= cost[selected_index]`
  but never proves the other candidates' costs (P2 unsound — the central claim
  of the V0 statement, `docs/spec/00-overview.md#v0-statement`).
- A serialization change that alters the public-input digest without minting a
  new relation_id, letting a P0 proof verify as P1
  (`docs/spec/06-security.md#binding-requirements`).

These map to the error taxonomy and verifier-rejection set in
`docs/spec/04-error-model.md#error-taxonomy` and
`docs/spec/04-error-model.md#verifier-rejections`; this RFC is the testing
counterpart that proves each rejection actually fires.

## Goals

- Lock a fixed, numbered ten-layer test stack with a stated owner area and CI
  gate for each layer, aligned with `docs/spec/07-testing-strategy.md#test-pyramid`.
- Lock the canonical negative-test list (the source §13 list, mapped to specific
  `VerifyError` / export error variants from
  `docs/spec/04-error-model.md#verifier-rejections`), and require every AIR
  component to instantiate the subset that applies to it.
- Lock differential testing as a required layer: the three references
  (Python float64 oracle, Python fixed-point, Rust fixed-point) and the AIR must
  agree bit-for-bit on the integer outputs, per
  `docs/spec/07-testing-strategy.md#differential-tests`.
- Lock constraint-mutation testing as a required soundness layer: a curated set
  of mutated AIR constraints and mutated witnesses must cause rejection;
  surviving mutants fail CI, per
  `docs/spec/07-testing-strategy.md#mutation-tests`.
- Lock the both-tests ship rule and make it a mechanically checkable CI gate
  (`INV-TEST-01`), aligned with `docs/spec/07-testing-strategy.md#ci-gates`.
- Lock the security-review gate (`INV-TEST-08`) preventing public soundness
  claims before layer-10 sign-off, aligned with
  `docs/spec/06-security.md#disclosure`.
- Define the golden-vector format, location, regeneration command, and the
  byte-identical regeneration check (`INV-TEST-02`), aligned with
  `docs/spec/07-testing-strategy.md#golden-vectors`.

## Non-Goals

- This RFC does not define the error taxonomy itself; it consumes the variants
  owned by `docs/spec/04-error-model.md#error-taxonomy`.
- It does not define performance regression gates; those are owned by
  `docs/spec/08-performance-budget.md#perf-gates`. Layer-9 below references but
  does not specify them.
- It does not define the security threat model or the disclosure policy text;
  those are owned by `docs/spec/06-security.md#threat-model` and
  `docs/spec/06-security.md#disclosure`. This RFC defines only the *gate* that
  blocks a claim on the review.
- It does not specify CEM (P3) or pixel-encoder (P4) tests beyond requiring that
  when those statements land, they inherit this same stack. Their statement-
  specific negative tests are owned by RFC-0010 and RFC-0011.
- It does not select a property-testing or fuzzing crate brand as a normative
  dependency; it specifies the required behavior and leaves the crate choice to
  RFC-0015's dependency inventory (`OPEN QUESTION` below).

## Proposed Design

### The locked ten-layer test stack

The stack is immutable in ordering and intent. Layers are numbered L1..L10;
each layer's outputs feed the next. A component is "tested" only when every
layer that applies to it is green. The mapping to crates uses the area labels
from the contract (`area:core`, `area:export`, `area:air`, `area:circuits`,
`area:prover`, `area:verifier`, `area:testing`, `area:security`).

```text
L1  Python float64 reference        area:export    semantic oracle, not bound
L2  Python fixed-point reference    area:export    bit-exact, manifest-bound
L3  Rust fixed-point reference      area:core      bit-exact, manifest-bound
L4  AIR witness-generation tests    area:air       witness builder + trace shape
L5  Prover/verifier ACCEPT tests    area:prover    valid witness -> Ok(())
L6  Negative REJECT tests           area:verifier  tampered witness -> Err(...)
L7  Differential tests (seeds)      area:testing   L2==L3==AIR over random seeds
L8  Constraint-mutation tests       area:testing   mutated constraint must reject
L9  Manifest serialization tests    area:core      canonical + round-trip + perf gate ref
L10 Security review                 area:security  GATE before any public claim
```

Prose statement of the same stack: L1 is the floating-point PyTorch-side oracle
used only to bound quantization error during export design — it is never bound
by a commitment and never authoritative for proof correctness. L2 is the Python
fixed-point reference; it is the bit-exact definition of the quantized relation
and is bound by `model_commitment` + `quantization_commitment`
(`docs/spec/03-data-model.md#model-manifest`). L3 is the Rust fixed-point
reference inside `pwm-core`, which must equal L2 bit-for-bit on every golden
vector. L4 checks that the witness builder in `pwm-circuits` produces traces of
the declared shape with all required auxiliary columns
(range/lookup witnesses per `docs/spec/03-data-model.md#witness`). L5 proves a
valid witness and runs `verify` to `Ok(())`. L6 takes the same valid witness,
applies one named tamper, and asserts the documented `VerifyError`. L7 runs L2,
L3, and a proof over many random seeds and asserts identical integer outputs.
L8 mutates constraints and witnesses and asserts rejection (surviving mutants
are bugs). L9 asserts canonical serialization is deterministic and round-trips
and references the proving-cost perf gate. L10 is the human security review that
gates any public soundness claim.

### Layer-by-layer contracts

#### L1 — Python float64 reference (`area:export`)

Purpose: a non-authoritative oracle that runs the LeWorldModel reference graph
in float64 to bound the quantization error of the exported integer graph during
manifest authoring. Per the verified facts, the predictor FFN/MLP uses GELU
while AdaLN modulation and the action `Embedder` use SiLU
(verified against upstream at 2026-06-03); the float64 oracle must mirror these
per-module activations exactly so that the int8/int16 error envelope is measured
against the true graph, not a uniform-activation stand-in. Cost path is MSE on
the final predicted latent versus the goal latent (`F.mse_loss`, verified
against upstream at 2026-06-03). L1 outputs an error report consumed by RFC-0001
export design; it is **not** bound and **not** a CI gate for proof correctness.

#### L2 — Python fixed-point reference (`area:export`)

Purpose: the authoritative bit-exact definition of the quantized relation. L2
implements the integer semantics of `docs/spec/03-data-model.md#bounded-integers`
exactly: signed centered encoding into M31, per-tensor bounds `[lo, hi]`,
`Rounding::NearestTiesToEven` default with manifest override to
`TruncateTowardZero` (exactly one active mode per manifest, per contract §3), and
`OverflowPolicy::Reject`. L2 emits golden vectors (see format below). L2 must be
deterministic across machines: no NumPy thread-order reductions in the integer
path; all accumulations are specified left-to-right.

#### L3 — Rust fixed-point reference (`area:core`)

Purpose: the prover-side reference inside `pwm-core` used to build traces. L3
must equal L2 bit-for-bit on every golden vector. The invariant:

- `INV-TEST-03` (reference parity): for every golden vector, the Rust
  fixed-point reference output equals the Python fixed-point reference output
  byte-for-byte, including every intermediate activation tensor recorded in the
  vector. A mismatch fails CI and blocks merge.

L3 uses the canonical types verbatim:

```rust
// pwm-core, used by L3 and the trace builder.
pub struct M31(u32);            // canonical value in [0, p), p = 2^31 - 1
pub struct BoundedInt { pub value: i64, pub lo: i64, pub hi: i64 }
pub enum Rounding { NearestTiesToEven, TruncateTowardZero }
pub enum OverflowPolicy { Reject }
```

#### L4 — AIR witness-generation tests (`area:air`, `area:circuits`)

Purpose: verify that the witness builder produces a `Witness`
(`docs/spec/03-data-model.md#witness`) whose traces have the declared shape and
whose auxiliary columns are present and consistent before any proving:

```rust
pub struct Witness {
    pub model_weights: Option<QuantizedWeights>,
    pub latent_history: Tensor,
    pub goal_latent: Option<Tensor>,
    pub candidate_actions: Option<Tensor>,
    pub action_embeddings: Tensor,
    pub predictor_activations: Vec<Tensor>,
    pub rollout_trajectory: Tensor,
    pub costs: Option<Vec<BoundedInt>>,
    pub argmin_witness: Option<ArgminWitness>,
    pub range_witnesses: Vec<RangeWitness>,
    pub lookup_witnesses: Vec<LookupWitness>,
}
```

L4 asserts: (a) every `BoundedInt` column has a matching entry in
`range_witnesses`; (b) every activation/normalization lookup has a matching
`lookup_witnesses` entry with correct multiplicity; (c) `TensorCell` writes are
unique per `(tensor_id, index, time)` and every read resolves to a prior write
(`docs/spec/03-data-model.md#tensor-memory-cells`). L4 does not run the prover;
it is the fast pre-prove structural gate.

```rust
pub struct TensorCell {
    pub tensor_id: u32,
    pub index: [u32; 4],   // up to 4 dims; unused dims = 0
    pub value: M31,
    pub scale_id: u32,
    pub time: u32,         // write ordering for read/write consistency
}
```

#### L5 — Prover/verifier ACCEPT tests (`area:prover`, `area:verifier`)

Purpose: a valid witness proves and verifies. The accept harness:

```rust
// L5 accept harness shape (pwm-prover test support).
fn assert_accepts(stmt: StatementType, fixture: &Fixture) {
    let artifact: ProofArtifact = prove(stmt, &fixture.public_input, &fixture.witness);
    assert!(verify(&artifact).is_ok());
}

// canonical verifier entry, used by L5 and L6.
pub fn verify(artifact: &ProofArtifact) -> Result<(), VerifyError>;
```

Required L5 fixtures by statement (each named, checked in under
`crates/<crate>/tests/golden/`):

| Fixture name                        | Statement | Notes |
| ----------------------------------- | --------- | ----- |
| `accept_p0_predictor_step`          | P0Step    | one ARPredictor_Q step, depth 6, heads 16 |
| `accept_p1_rollout_h5`              | P1Rollout | horizon 5, action_block 5 (verified eval config) |
| `accept_p2_planning_S_small`        | P2FixedCandidatePlanning | small fixed S; exercises full argmin + tie-break |

#### L6 — Negative REJECT tests (`area:verifier`)

Purpose: the soundness backbone. Each negative test starts from an L5 accept
fixture, applies exactly one named tamper, and asserts the documented
`VerifyError` variant (`docs/spec/04-error-model.md#verifier-rejections`). The
**locked canonical negative-test list** (from `docs/feasibility-study.md` §13,
each now bound to a verifier rejection):

| #  | Negative test name                  | Tamper applied                                              | Expected rejection (see error model) |
| -- | ----------------------------------- | ---------------------------------------------------------- | ------------------------------------- |
| N1 | `reject_wrong_model_commitment`     | flip one byte of `model_commitment`                        | commitment-mismatch rejection         |
| N2 | `reject_wrong_quant_commitment`     | flip one byte of `quantization_commitment`                 | commitment-mismatch rejection         |
| N3 | `reject_wrong_action`               | mutate one action in `candidate_actions`                   | relation-unsatisfied rejection        |
| N4 | `reject_wrong_latent`               | mutate one value in `latent_history`                       | relation-unsatisfied rejection        |
| N5 | `reject_wrong_intermediate_activation` | mutate one tensor in `predictor_activations`            | relation-unsatisfied rejection        |
| N6 | `reject_wrong_accumulator`          | replace an accumulator with an in-field but overflowed value | range-check rejection                 |
| N7 | `reject_wrong_rounding_remainder`   | set requant `rem >= 2^r` (or wrong ties-to-even adjust)    | range-check / requant rejection       |
| N8 | `reject_wrong_activation_lookup`    | use a `(x,y)` pair absent from the committed table         | lookup-membership rejection           |
| N9 | `reject_wrong_layernorm_reciprocal` | mutate the AdaLN/LayerNorm inverse-sqrt witness            | relation-unsatisfied rejection        |
| N10| `reject_wrong_attention_probability`| supply an off-circuit softmax probability                  | relation-unsatisfied / lookup rejection |
| N11| `reject_wrong_candidate_cost`       | mutate one `costs[s]` away from `MSE_Q` output             | relation-unsatisfied rejection        |
| N12| `reject_wrong_argmin`               | claim a `selected_index` whose cost is not minimal         | argmin / range-check rejection        |
| N13| `reject_wrong_tie_break`            | pick a larger index at a tie (violate smallest-index rule) | argmin / range-check rejection        |

Two additional binding-layer negatives are added (they protect the relation_id
and statement-type invariants of `docs/spec/06-security.md#binding-requirements`,
which the source list assumed but did not enumerate):

| #  | Negative test name                  | Tamper applied                                  | Expected rejection |
| -- | ----------------------------------- | ----------------------------------------------- | ------------------ |
| N14| `reject_p0_proof_as_p1`             | submit a P0 proof under a P1 `relation_id`      | relation_id mismatch rejection |
| N15| `reject_partial_candidate_set`      | omit one candidate's rollout/cost from the trace | relation-unsatisfied (planner soundness) |

N15 directly enforces the planner-soundness rule of RFC-0009
(`docs/feasibility-study.md` §9.5): proving only the selected candidate is
unsound, so the test omits a non-selected candidate's cost and requires
rejection.

Component applicability rule (`INV-TEST-04`): each AIR component must instantiate
every negative test in the list whose tamper targets a column the component
owns. The mapping is recorded in `docs/spec/07-testing-strategy.md#negative-tests`:
`linear`/`matmul`/`requant` own N6, N7; `range_check`/`tensor_memory` own N6, and
the read/write-consistency negatives; `activation_lookup` owns N8;
`layernorm`/`attention`/`mlp` own N9, N10; `predictor`/`rollout` own N3, N4, N5;
`cost`/`argmin` own N11, N12, N13, N15; the public-input layer owns N1, N2, N14.

#### L7 — Differential tests across random seeds (`area:testing`)

Purpose: catch reference drift and AIR-vs-reference divergence on inputs no
human picked. For each statement and a seeded RNG, generate random in-range
inputs, run L2 (Python fixed-point), L3 (Rust fixed-point), and a full
prove/verify, and assert the integer outputs (claimed latents, costs, selected
index) are identical across all three. Seeds are fixed and recorded so failures
are reproducible.

- `INV-TEST-05` (tri-reference agreement): for every differential seed, L2, L3,
  and the AIR-proven outputs agree bit-for-bit on all public integer outputs.
  Any disagreement fails CI; the offending seed is added to the golden corpus.

L7 also runs a property-based generator over `BoundedInt` arithmetic
(add/sub/mul/accumulate/requantize/clamp/compare, per RFC-0002) asserting the
`OverflowPolicy::Reject` invariant: any input whose true integer accumulator
would exceed its declared `[lo, hi]` must be rejected by the reference, never
silently wrapped.

#### L8 — Constraint-mutation tests (`area:testing`)

Purpose: prove the constraints are *load-bearing*. A curated mutant set is
maintained per component; each mutant is a single deliberate weakening of a
constraint or a single tampered witness value. The harness asserts the mutant is
**killed** (the verifier rejects, or proving fails). The required mutant
families per component:

```text
drop-a-range-check          remove one range constraint   -> must be killed by L6 N6/N7
weaken-remainder-bound      change rem<2^r to rem<2^(r+1)  -> killed by N7
swap-accumulator-init       acc_0 = 0 instead of bias      -> killed by N5/N11
omit-tie-break-strictness   use >=0 instead of -1>=0       -> killed by N13
skip-candidate-cost-link    unbound costs[s]               -> killed by N11/N15
forge-lookup-multiplicity   inflate a lookup multiplicity  -> killed by N8
```

- `INV-TEST-06` (no surviving mutants): every mutant in the curated set must be
  killed by at least one negative test. A surviving mutant is a CI failure and
  blocks the release; the gap it exposes is closed by adding a negative test,
  not by removing the mutant. The mutant catalogue lives at
  `crates/pwm-air/tests/mutants/` and is referenced from
  `docs/spec/07-testing-strategy.md#mutation-tests`.

L8 is distinct from L6: L6 tampers *witnesses* against fixed constraints; L8
tampers *constraints* (or removes them) and confirms a negative test still
catches the resulting unsoundness. Together they prove both directions:
tampered data is rejected, and the constraint that rejects it is necessary.

#### L9 — Manifest and serialization tests (`area:core`)

Purpose: the binding surface. Tests assert canonical serialization is
deterministic and round-trips, that the public-input digest is stable, and that
the manifest commitment binds every field listed in
`docs/spec/03-data-model.md#model-manifest`.

- `INV-TEST-02` (golden regeneration is byte-identical): regenerating golden
  vectors and manifests from the same checkpoint and config yields byte-identical
  files; the CI gate diffs regenerated artifacts against the committed ones.
- `INV-TEST-07` (commitment coverage): a parameterized test mutates each bound
  manifest field in turn (one weight, one bias, one quantization scale, one
  rounding mode, one lookup-table entry, one tensor shape, the planner config,
  the relation version, the serialization version) and asserts the corresponding
  commitment changes; a field whose mutation does not change a commitment is an
  unbound-field bug (this is the L9 counterpart of RFC-0001's binding
  requirement and `docs/spec/06-security.md#binding-requirements`).

L9 references the perf gate in `docs/spec/08-performance-budget.md#perf-gates`
but does not define it.

#### L10 — Security review gate (`area:security`)

Purpose: the human gate. No public soundness claim about a given relation_id may
be published until that relation has passed a recorded security review covering:
the binding requirements (`docs/spec/06-security.md#binding-requirements`), the
soundness requirements (`docs/spec/06-security.md#soundness-requirements`), the
completeness of the L6 negative list for every component touched, the L8 mutant
report (zero survivors), and the L7 differential corpus.

- `INV-TEST-08` (review gate): for each `relation_id`, a public soundness claim
  is permitted only after a layer-10 review is recorded against that exact
  relation_id in the security log (`docs/spec/06-security.md#disclosure`). A new
  relation_id (any semantic change, per contract §4) requires a fresh review;
  the prior review does not transfer.

### The both-tests ship rule (the locked core decision)

- `INV-TEST-01` (both-tests rule): a component, gate, manifest field, or proof
  statement may merge to a release branch only if it has at least one accepting
  test (L5: valid witness -> `Ok(())`) AND at least one rejecting test (L6: a
  named tampered witness -> the documented `VerifyError`). The CI gate enforces
  this by requiring, for every file under `crates/pwm-air/components/`, a
  co-located test module that registers both an accept fixture and at least one
  negative from the L6 list applicable to that component. A component with only
  accept tests fails the gate. This rule is permanent and is not waivable per
  component; it may only be changed by a superseding RFC.

### CI gate wiring

The CI gates, in order, aligned with `docs/spec/07-testing-strategy.md#ci-gates`:

```text
gate-1  L3==L2 reference parity (INV-TEST-03)            blocks merge
gate-2  L4 witness structural checks                     blocks merge
gate-3  L5 accept fixtures green (all statements in tier) blocks merge
gate-4  L6 negative list complete + green (INV-TEST-01/04) blocks merge
gate-5  L7 differential corpus green (INV-TEST-05)       blocks merge
gate-6  L8 zero surviving mutants (INV-TEST-06)          blocks merge
gate-7  L9 byte-identical regeneration + coverage (INV-TEST-02/07) blocks merge
gate-8  L10 review recorded for the relation_id (INV-TEST-08) blocks public claim
```

gate-1..gate-7 block merge to a release branch; gate-8 blocks publication of a
soundness claim, not the merge, because a relation may be merged as
experimental/unclaimed before review.

### Golden-vector format

Golden vectors are checked in under `crates/pwm-export/tests/golden/<relation_id>/`
and mirrored for Rust consumption. Each vector is a canonical-JSON file plus a
hash sidecar:

```json
{
  "relation_id": "pwm.lewm.fixed_candidate_planning.v1",
  "manifest_canonical_json_hash": "0x...",
  "rounding": "nearest_ties_to_even",
  "inputs": { "latent_history": [...], "goal_latent": [...], "candidate_actions": [...] },
  "intermediates": { "action_embeddings": [...], "predictor_activations": [[...]], "rollout_trajectory": [...] },
  "outputs": { "costs": [...], "selected_index": 0, "selected_cost": ... }
}
```

All values are mathematical integers (the `BoundedInt::value` field), never field
residues, so the file is human-auditable. The sidecar hash is the canonical-JSON
hash; `INV-TEST-02` diffs both the file and the hash on regeneration.

### Failure-mode enumeration and system response

| Failure mode                                    | Detected by | System response |
| ----------------------------------------------- | ----------- | --------------- |
| Rust reference diverges from Python reference   | L3/gate-1   | CI fails; merge blocked; offending vector reported |
| Witness builder emits wrong-shape trace         | L4/gate-2   | CI fails; merge blocked |
| Valid witness fails to verify                   | L5/gate-3   | CI fails; completeness bug; merge blocked |
| Tampered witness verifies (missing constraint)  | L6/gate-4   | CI fails; soundness bug; merge blocked; new negative added |
| Component shipped with only accept tests         | gate-4/INV-TEST-01 | CI fails; merge blocked |
| Reference/AIR disagree on a random seed         | L7/gate-5   | CI fails; seed added to golden corpus; merge blocked |
| A constraint mutant survives                    | L8/gate-6   | CI fails; merge blocked; negative test added to kill it |
| Manifest regeneration not byte-identical        | L9/gate-7   | CI fails; merge blocked |
| A manifest field is unbound by any commitment   | L9/INV-TEST-07 | CI fails; binding bug; merge blocked |
| Public claim attempted before layer-10 review   | gate-8/INV-TEST-08 | Claim blocked; release notes may not assert soundness for that relation_id |

## Alternatives Considered

### Alternative A: positive-only golden-vector testing (rejected)

Test only that valid witnesses prove and verify, relying on the STARK's
soundness for the rest. Rejected: STARK soundness guarantees the *constraints*
are satisfied, not that the constraints *encode the intended integer relation*.
The dominant failure mode (`docs/feasibility-study.md` §9.1–9.3) is a missing or
weakened constraint that admits a wrong witness; a positive-only suite is green
on every such bug. The whole point of L6/L8 is that constraints must be shown to
be load-bearing. This alternative is the exact trap the source RFC-013 warns
against.

### Alternative B: rely on an external audit instead of an in-repo mutation suite (rejected)

Defer soundness assurance entirely to a one-time third-party security audit and
skip L8 constraint-mutation testing. Rejected: an audit is a snapshot; every
subsequent commit to `pwm-air` can reintroduce an unsoundness the audit cleared.
Constraint-mutation testing is the *continuous* counterpart that keeps the audit
valid between reviews. We keep L10 (the review gate) precisely because it is
complementary to, not a substitute for, L8 — and we require L8's zero-survivors
report as an input to L10, so the audit reviews a system that already proves its
own constraints are necessary.

### Alternative C: differential testing against PyTorch float output (rejected)

Make the AIR's success criterion "matches PyTorch float inference within a
tolerance." Rejected on the founding principle (`docs/feasibility-study.md` §0,
`docs/spec/00-overview.md#non-goals`): the proof relation is the *quantized
integer* graph, not the float model. A tolerance-based oracle would make the
test suite accept witnesses that are merely *close* to float, which is exactly
the unsound "proves the PyTorch model" claim the project forbids. L1 keeps a
float oracle but only to bound quantization error during export design; it is
never an accept/reject authority. The authoritative differential is the
bit-exact tri-reference agreement of L7 (`INV-TEST-05`).

## Drawbacks

- The both-tests rule (`INV-TEST-01`) and the per-component negative-test
  applicability rule (`INV-TEST-04`) roughly double the test-authoring cost of
  every AIR component and slow initial development. This is accepted: the cost is
  paid once per component and bought back the first time it catches a silent
  soundness hole.
- Constraint-mutation testing (L8) requires maintaining a mutant catalogue in
  lockstep with the constraints, which is extra surface that can rot. Mitigated
  by `INV-TEST-06` failing CI on surviving mutants, which forces the catalogue
  to stay coupled to the constraints.
- The tri-reference requirement (L2/L3/AIR all bit-identical) makes any
  intentional change to integer semantics a three-place edit plus a golden
  regeneration. This friction is intentional: a one-place edit to integer
  semantics is precisely the change that must be hard.
- The layer-10 gate can become a release bottleneck if reviewers are scarce.
  Mitigated by allowing experimental relations to merge unclaimed (gate-8 blocks
  the *claim*, not the merge).

## Migration / Rollout

- **Feature flags / staging.** The CI gates land incrementally per milestone:
  gates 1–4 and 7 are enforced from `v0.1` (foundations: core arithmetic,
  range/lookup, tensor memory, linear/matmul/requant, manifest/export). Gates 5
  (differential corpus) and 6 (mutation) become blocking at `v0.1` for the
  components that exist and expand automatically as components are added — a new
  component cannot reduce coverage because `INV-TEST-04` requires its applicable
  negatives. Gate 8 (review) is enforced at `v1.0`, the first milestone that
  produces a public P2 soundness claim, and retroactively for any earlier public
  claim.
- **Relation/schema versioning.** Negative-test fixtures and golden vectors are
  keyed by `relation_id` (`pwm.lewm.<statement>.v<N>`). A new relation_id mints a
  new golden directory and inherits the full L6 list; the old directory is kept
  for regression until the relation is deprecated per
  `docs/spec/09-release-and-versioning.md#deprecation`. Golden-vector and
  manifest schema bumps follow `docs/spec/03-data-model.md#schema-versioning`;
  `artifact_version` bumps on `ProofArtifact` are gated by an accept+reject
  fixture for the new version under `INV-TEST-01`.
- **Deprecation.** When a relation_id is deprecated, its accept/reject fixtures
  move to a `regression/` tier that still runs but is not required for new
  components; they are removed only when the relation_id leaves the support
  window (`docs/spec/09-release-and-versioning.md#deprecation`).
- **Backfill.** Any component merged before this RFC that lacks a rejecting test
  is treated as failing gate-4 and must be backfilled before the next release
  branch is cut; no waiver.

## Testing Strategy

This RFC *is* the testing strategy; the tests below validate the strategy's own
machinery and cross-reference `docs/spec/07-testing-strategy.md` and
`docs/spec/04-error-model.md#verifier-rejections`.

Accepting tests (named):

- `accept_p0_predictor_step`, `accept_p1_rollout_h5`,
  `accept_p2_planning_S_small` — the L5 fixtures above; each proves and verifies
  to `Ok(())`. `accept_p1_rollout_h5` uses the verified eval config horizon=5,
  action_block=5.
- `accept_reference_parity_golden` — L3 reproduces every L2 golden vector
  bit-for-bit (`INV-TEST-03`).
- `accept_golden_regeneration_byte_identical` — regenerating golden vectors and
  the manifest yields byte-identical files (`INV-TEST-02`).
- `accept_tri_reference_seeds` — L2, L3, and the AIR agree on the integer
  outputs for the fixed differential seed corpus (`INV-TEST-05`).

Rejecting / negative tests (named): the full L6 list N1–N15 above, each asserting
its documented `VerifyError` from
`docs/spec/04-error-model.md#verifier-rejections`. Additionally:

- `reject_unbound_manifest_field_*` — the L9 parameterized coverage test
  (`INV-TEST-07`): each bound manifest field mutated in turn must change a
  commitment; a field that does not is a failure.
- `reject_surviving_mutant_*` — the L8 catalogue
  (`drop-a-range-check`, `weaken-remainder-bound`, `swap-accumulator-init`,
  `omit-tie-break-strictness`, `skip-candidate-cost-link`,
  `forge-lookup-multiplicity`); each must be killed by a named negative
  (`INV-TEST-06`).
- `reject_component_without_rejecting_test` — a CI meta-test (`INV-TEST-01`):
  scans `crates/pwm-air/components/` and fails if any component module lacks a
  co-located negative test from its applicable L6 subset.

ML-specific and differential tests: the tri-reference agreement (`INV-TEST-05`)
and the per-module activation fidelity of L1/L2 (GELU in the predictor FFN/MLP;
SiLU in AdaLN modulation and the action `Embedder`, verified against upstream at
2026-06-03) are the ML-specific layer; they ensure the integer graph and its
float oracle both model the real LeWorldModel module structure rather than a
simplified stand-in.

## Open Questions

- OPEN QUESTION (owner: `area:testing` maintainer; resolution: RFC-0015): which
  Rust property-testing and fuzzing crates are adopted as the normative
  dependencies for the L7 generator and the `BoundedInt` arithmetic property
  tests. This RFC fixes the required behavior (seeded, reproducible, in-range
  generation; reject-on-overflow property) and defers the crate selection and
  version pinning to the dependency inventory in RFC-0015
  (`docs/rfcs/RFC-0015-third-party-vendoring-and-pinning.md`).
- OPEN QUESTION (owner: `area:security` maintainer; resolution: milestone
  `v1.0`): whether the layer-10 review for the V0 P2 relation is performed by an
  external auditor or by an internal reviewer distinct from the implementer.
  Either satisfies `INV-TEST-08`; the choice is recorded in the security log
  (`docs/spec/06-security.md#disclosure`) before the first public claim.

## References

- Founding analysis: `docs/feasibility-study.md` §13 (test layers, negative-test
  list, both-tests rule), §9.1–§9.5 (binding/determinism/range/approximation/
  planner soundness), §0 (the exact-relation principle).
- Testing spec: `docs/spec/07-testing-strategy.md#test-pyramid`,
  `#golden-vectors`, `#negative-tests`, `#differential-tests`,
  `#mutation-tests`, `#ci-gates`.
- Error model: `docs/spec/04-error-model.md#error-taxonomy`,
  `#failure-modes`, `#verifier-rejections`, `#recovery`.
- Security: `docs/spec/06-security.md#binding-requirements`,
  `#soundness-requirements`, `#threat-model`, `#disclosure`.
- Data model: `docs/spec/03-data-model.md#model-manifest`,
  `#witness`, `#bounded-integers`, `#tensor-memory-cells`, `#schema-versioning`.
- Overview: `docs/spec/00-overview.md#v0-statement`, `#non-goals`.
- Performance: `docs/spec/08-performance-budget.md#perf-gates`.
- Release: `docs/spec/09-release-and-versioning.md#deprecation`, `#changelog`.
- Related RFCs: RFC-0001 (export pipeline / byte-identical re-export),
  RFC-0002 (fixed-point semantics under test in L7 properties),
  RFC-0009 (fixed-candidate planner soundness enforced by N15),
  RFC-0010 / RFC-0011 (deferred statements that inherit this stack),
  RFC-0015 (dependency pinning / property-test crate selection).
- Verified external facts (verified against upstream at 2026-06-03): LeWorldModel
  per-module activations (GELU in predictor FFN/MLP; SiLU in AdaLN and action
  `Embedder`), MSE goal cost, eval config horizon=5 / action_block=5.
