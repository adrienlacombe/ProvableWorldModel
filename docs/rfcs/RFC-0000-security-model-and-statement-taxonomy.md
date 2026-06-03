# RFC-0000: Security model and statement taxonomy

- Status: Accepted
- Authors: ProvableWorldModel maintainers
- Created: 2026-06-03
- Target milestone: v0.1

## Summary

This RFC fixes the meaning of "proved" for ProvableWorldModel. It defines the
exact set of proof statements the system supports (P0 through P4), states for
each statement what its proof binds and what it deliberately does not, and locks
the rule that a proof is valid only for the exact `relation_id` it declares. The
locked consequence: a proof minted for one statement (for example P0, one
predictor step) and submitted under another statement's `relation_id` (for
example P1, rollout) MUST be rejected by the verifier, because the digest the
verifier reconstructs from public inputs no longer matches the digest the prover
bound into the Fiat-Shamir transcript. This RFC is the security charter the rest
of the corpus references; it is the source of the statement-tier taxonomy in
`docs/spec/00-overview.md#scope-and-statement-tiers` and the binding/soundness
requirements in `docs/spec/06-security.md#soundness-requirements`.

## Motivation

The single largest soundness hazard for an ML proving system is ambiguity about
what the cryptographic guarantee actually covers. The founding analysis
(`docs/feasibility-study.md`, §0, §1.1) is explicit: the project proves an exact
arithmetic relation over a committed quantized model, not floating-point
equivalence, not physical truth, not zero-knowledge. Without a frozen statement
taxonomy, "proved world model" degrades into a marketing phrase, and a prover can
exploit any gap between what an observer believes was proved and what the AIR
constrains.

Two concrete attack scenarios motivate the locked decision:

1. Statement substitution. A prover holds a valid P0 proof (one predictor step).
   A relying party expects evidence of a full rollout (P1) before trusting a
   downstream plan. If the verifier accepted any structurally valid Stwo proof
   regardless of which statement it attests, the prover could pass the P0 proof
   off as P1 and the relying party would believe a multi-step trajectory was
   verified when only a single transition was. The fix is to bind `relation_id`
   and `statement_type` into the public-input digest and reject on mismatch.

2. Silent semantics drift. A prover re-quantizes the model, swaps an activation
   approximation table, or changes the planner tie-break, then reuses an old
   proof. Anything not bound by the public-input digest is mutable by the prover
   and therefore unsound (`docs/feasibility-study.md` §4.2, §9.1). Each statement
   must enumerate exactly which commitments it binds.

The feasibility study sketches the taxonomy in §1.2 and §11 (source RFC-000) and
the soundness requirements in §9. This RFC lifts, sharpens, and makes those
permanent, and adds the precise rejection semantics that the source left
implicit.

## Goals

- Define an immutable, totally ordered set of five proof statements P0–P4, each
  with a stable `relation_id` and a stable `StatementType` discriminant.
- For each statement, enumerate the public inputs, the private witness, the
  commitments bound, the output bound, the accepted tensor shapes, the security
  tier, and the privacy mode, as typed fields of `PublicInput`
  (`docs/spec/03-data-model.md#public-input`).
- Lock the rule "a proof is valid only for its declared `relation_id`", and make
  it testable: a proof for statement X submitted under statement Y's
  `relation_id` MUST be rejected.
- Specify exactly which `PublicInput` and `Witness` fields are required, optional,
  or forbidden per statement, so a verifier can reject malformed statements before
  invoking the Stwo verifier.
- Name every binding invariant and every rejection failure mode with the
  verifier's response, cross-referenced to
  `docs/spec/04-error-model.md#verifier-rejections`.
- Establish the security/soundness/privacy vocabulary used corpus-wide: "succinct
  validity proof" for V0, never "zero-knowledge".

## Non-Goals

- This RFC does not specify the AIR constraints that enforce each relation. Those
  live in the component RFCs: RFC-0007 (predictor), RFC-0008 (rollout), RFC-0009
  (fixed-candidate planner), RFC-0010 (CEM), RFC-0011 (pixel encoder).
- It does not define the byte-level public-input serialization or the
  Fiat-Shamir channel ordering; those are locked in RFC-0014
  (`docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md`). This RFC
  states the binding requirement; RFC-0014 states the bytes.
- It does not define the manifest schema or commitment scheme; those are locked in
  RFC-0001 and `docs/spec/03-data-model.md#model-manifest`.
- It does not establish zero-knowledge or hiding. ZK is an explicit future mode
  gated on a hiding audit (`docs/spec/06-security.md#privacy-and-zk`).
- It does not prove that the quantized relation approximates the original
  floating-point LeWorldModel; approximation error is bounded and committed per
  RFC-0006, not proved by the inference proof.

## Proposed Design

### 1. The statement taxonomy is closed and immutable

The supported statements are exactly the five members of `StatementType`
(`docs/spec/03-data-model.md#public-input`, canonical signature reproduced here
verbatim):

```rust
pub enum StatementType { P0Step, P1Rollout, P2FixedCandidatePlanning,
                         P3Cem, P4PixelToPlan }
```

Each `StatementType` discriminant is paired with exactly one base `relation_id`
string under the scheme `pwm.lewm.<statement>.v<N>` (contract naming §4):

| StatementType            | Base relation_id (v1)                  | Composes        | Version | Milestone |
| ------------------------ | -------------------------------------- | --------------- | ------- | --------- |
| `P0Step`                 | `pwm.lewm.predictor_step.v1`           | (base)          | V0..    | v0.2      |
| `P1Rollout`              | `pwm.lewm.rollout.v1`                  | P0              | V0..    | v0.2      |
| `P2FixedCandidatePlanning` | `pwm.lewm.fixed_candidate_planning.v1` | P1 + cost + argmin | V0 (headline) | v1.0 |
| `P3Cem`                  | `pwm.lewm.cem_planning.v1`             | P2 + sampling   | V2      | Future    |
| `P4PixelToPlan`          | `pwm.lewm.pixel_to_plan.v1`            | encoder + P3/P2 | V3      | Future    |

`relation_id` strings are immutable. Any semantic change (a new rounding mode, a
new activation approximation, an added bound check, a changed tie-break) mints a
new id by incrementing `<N>`; it never mutates an existing id
(`docs/spec/09-release-and-versioning.md#relation-versioning`). The
`PublicInput.relation_id` field is the BLAKE-domain hash of the chosen
`relation_id` string (the 32-byte digest, not the UTF-8 bytes); the mapping from
string to digest is fixed by RFC-0014.

INV-SEC-01 (closed taxonomy): the verifier supports proofs only for
`relation_id` digests in a compiled-in allowlist derived from the table above for
the verifier's build. A `relation_id` outside the allowlist is rejected with
`VerifyError::UnknownRelation` before any proof work. Adding a statement or a new
version requires a code change and a release (`docs/spec/09-release-and-versioning.md#semver`),
never a runtime flag.

### 2. What each statement binds

Every statement binds, unconditionally, the four core commitments. These are
non-`Option` fields of `PublicInput` and are always present:

```rust
pub struct PublicInput {
    pub relation_id: [u8; 32],
    pub model_commitment: [u8; 32],
    pub quantization_commitment: [u8; 32],
    pub planner_config_commitment: [u8; 32],
    pub statement_type: StatementType,
    pub latent_history_commitment: Option<[u8; 32]>,
    pub latent_history_public: Option<Vec<M31>>,
    pub goal_latent_commitment: Option<[u8; 32]>,
    pub goal_latent_public: Option<Vec<M31>>,
    pub candidate_actions_commitment: Option<[u8; 32]>,
    pub candidate_actions_public: Option<Vec<M31>>,
    pub claimed_output_commitment: [u8; 32],
    pub selected_index: Option<u32>,
    pub selected_cost: Option<BoundedInt>,
}
```

`model_commitment` and `quantization_commitment` together bind the full model
manifest: architecture, weights, biases, quantization scales, the single active
rounding mode, lookup/activation/normalization approximation tables, tensor
shapes, the relation version, and the serialization version
(`docs/spec/03-data-model.md#model-manifest`, contract §6.6). The LeWorldModel
reference target binds a predictor with `latent_dim = 192`, `history_size = 3`,
`depth = 6`, `heads = 16`, `dim_head = 64`, `mlp_dim = 2048` (the V0 reference
configuration; ~15M parameters). Per-module activation functions are part of the
bound architecture and are not uniform: the predictor FFN/MLP uses GELU, while
AdaLN modulation and the action `Embedder` use SiLU (verified against upstream at
2026-06-03). `planner_config_commitment` binds the planner rule: for P2 it binds
the cost function (final-latent MSE to goal) and the deterministic argmin
tie-break (smallest index attaining the minimum); for P3 it additionally binds
the CEM parameters `num_samples = 300`, `n_steps = 30`, `topk = 30`,
`var_scale = 1.0`, and the task `horizon = 5`, `action_block = 5` (verified
against upstream at 2026-06-03; see RFC-0010).

The statement-specific binding profile is the following. "pub" means the value
itself is in the public input (the `_public` variant carries it); "commit" means
only a 32-byte commitment is public (the `_commitment` variant); "req" required,
"opt" caller's choice of pub-or-commit, "forbid" must be absent (`None`):

| Field / statement                 | P0   | P1   | P2   | P3   | P4   |
| ---------------------------------- | ---- | ---- | ---- | ---- | ---- |
| `relation_id`                      | req  | req  | req  | req  | req  |
| `model_commitment`                 | req  | req  | req  | req  | req  |
| `quantization_commitment`          | req  | req  | req  | req  | req  |
| `planner_config_commitment`        | req\*| req\*| req  | req  | req  |
| `statement_type`                   | req  | req  | req  | req  | req  |
| `latent_history_*` (pub or commit) | req  | req  | req  | req  | forbid (derived from pixels) |
| `goal_latent_*` (pub or commit)    | forbid | forbid | req | req | forbid (derived from pixels) |
| `candidate_actions_*`              | req (single action history) | req (one sequence) | req (S sequences) | forbid (sampled in-circuit) | req |
| `claimed_output_commitment`        | req (z_next) | req (trajectory root) | req (selected plan + cost) | req | req |
| `selected_index`                   | forbid | forbid | req | req | req |
| `selected_cost`                    | forbid | forbid | req | req | req |

\* For P0/P1 `planner_config_commitment` binds the empty planner config (a fixed,
manifest-declared "no planner" sentinel commitment), so the same digest machinery
applies uniformly. It is `req` with that sentinel value, never absent.

Pixel inputs (P4 only) are not yet `PublicInput` fields. P4 is `Future`; when it
lands, RFC-0011 extends `PublicInput` with `pixel_history_commitment` and
`pixel_goal_commitment` under a new artifact version and a new `relation_id`
version, per `docs/spec/09-release-and-versioning.md#relation-versioning`. Until
then the verifier rejects any artifact whose `statement_type` is `P4PixelToPlan`
with `VerifyError::UnsupportedStatement`.

INV-SEC-02 (full binding): no semantically load-bearing value is left unbound. If
a value can change the relation's truth (a weight, a scale, a rounding mode, an
approximation table, the planner rule, an input tensor, the claimed output, the
selected index, the selected cost), it is either in the public-input digest
directly or bound through a commitment that is in the digest. This is the
restatement of the source's "anything unbound is mutable by the prover and
therefore unsound" (`docs/feasibility-study.md` §9.1).

INV-SEC-03 (input determinism): for a given statement, the choice between the
`_public` and `_commitment` variant of an optional input is itself bound: the
public-input digest covers a one-byte presence/variant tag per optional field
(defined in RFC-0014), so a prover cannot swap a committed input for a public one
without changing the digest.

### 3. The relation_id binding rule (the locked decision)

The decision this RFC makes permanent:

> A proof is valid only for the exact `relation_id` it declares. The verifier
> reconstructs the public-input digest from the supplied `PublicInput` (including
> `relation_id` and `statement_type`) and checks it against the digest the prover
> absorbed into the Fiat-Shamir transcript before the first commitment. A proof
> generated for one statement and presented under a different statement's
> `relation_id` MUST be rejected.

Mechanism (the byte-level details are RFC-0014's; the requirement is this RFC's):

1. The prover, before committing any trace, absorbs the canonical public-input
   digest `H_pi = digest(PublicInput)` into the Stwo Fiat-Shamir channel. `H_pi`
   is computed over all `PublicInput` fields, `relation_id` and `statement_type`
   first. The challenges that drive FRI and the LogUp argument therefore depend on
   `relation_id`.
2. The verifier recomputes `H_pi` from the `PublicInput` carried in the artifact
   and absorbs it in the identical position before deriving the same challenges
   (`docs/feasibility-study.md` §10.3, §10.4).
3. If the artifact's `relation_id`/`statement_type` differ in any byte from what
   the prover absorbed, the verifier's challenges diverge from the prover's, the
   FRI/LogUp checks fail, and verification rejects. There is no code path that
   verifies a proof against a `relation_id` other than the one bound at proving
   time.
4. As defense in depth and for precise error reporting, the verifier first checks
   `statement_type` against the allowlist and against the structural profile in
   the table of §2, rejecting early with a specific `VerifyError` rather than a
   generic FRI failure.

INV-SEC-04 (relation binding): `verify(artifact)` returns `Ok(())` only if the
artifact's declared `relation_id` equals the `relation_id` whose digest was bound
into the transcript at proving time. Equivalently: substituting any other
`relation_id` (including a different version `vN` of the same statement, or a
different statement's id) causes rejection. This makes statement substitution and
version substitution cryptographically infeasible, not merely policy-prohibited.

INV-SEC-05 (no partial planning): for P2/P3, the proof binds costs for ALL
candidates and the full argmin/selection, never only the selected candidate
(`docs/feasibility-study.md` §9.5). `selected_index` and `selected_cost` are
public, and the AIR (RFC-0009/RFC-0010) constrains `cost_s - selected_cost >= 0`
for every candidate `s` and the strict `cost_s - selected_cost - 1 >= 0` for
every `s < selected_index` (contract §3 tie-break). A proof that constrains only
the selected rollout is not a valid P2/P3 proof; it cannot mint
`pwm.lewm.fixed_candidate_planning.v1`.

### 4. Verifier entry and structural pre-checks

The canonical entry point (contract §6.5, verbatim):

```rust
pub fn verify(artifact: &ProofArtifact) -> Result<(), VerifyError>;
```

`verify` performs, in order:

```text
1. Reject if artifact.public_input.relation_id digest not in allowlist
     -> VerifyError::UnknownRelation
2. Reject if statement_type maps to a relation_id whose statement differs
     from the declared relation_id's statement
     -> VerifyError::StatementRelationMismatch
3. Reject if statement_type is P4PixelToPlan in a build without P4 support
     -> VerifyError::UnsupportedStatement
4. Reject if the per-statement field profile (table in §2) is violated:
     a required field is None, a forbidden field is Some, or an input that
     must be pub-or-commit is neither
     -> VerifyError::MalformedPublicInput
5. Recompute H_pi = digest(public_input); absorb into transcript (RFC-0014).
6. Run the Stwo verifier with the reconstructed transcript. Any FRI/LogUp/
     constraint failure (including a relation_id that does not match the bound
     digest) -> VerifyError::ProofInvalid
7. Decode claimed_output_commitment / claimed_outputs; enforce application
     checks: selected_index < S, tensor shapes match the manifest, output
     commitment matches canonical serialization of claimed_outputs
     -> VerifyError::OutputBindingFailure on mismatch
```

The verifier never executes PyTorch or any reference inference; it checks the
arithmetic relation only (`docs/feasibility-study.md` §10.4). The full
`VerifyError` taxonomy is owned by `docs/spec/04-error-model.md#verifier-rejections`;
this RFC contributes the four security-critical variants
`UnknownRelation`, `StatementRelationMismatch`, `UnsupportedStatement`,
`MalformedPublicInput`, plus `ProofInvalid` and `OutputBindingFailure`.

### 5. Security tiers and privacy mode

| Statement | Security tier (what is cryptographically guaranteed) | Privacy mode (V0) |
| --------- | ---------------------------------------------------- | ----------------- |
| P0–P4     | Succinct validity: the claimed outputs are exactly the result of the committed quantized relation under the bound manifest, with M31/QM31 soundness. | Validity only; NOT zero-knowledge. Weights may be `private_committed` (only a commitment is public) but the proof is not proven hiding. |

INV-SEC-06 (no ZK overclaim): no artifact, log line, error string, document, or
public statement describes a V0 proof as "zero-knowledge" or "private". Weight
privacy via `private_committed` visibility binds a commitment, not a hiding proof;
the trace is not proven to leak nothing. ZK is a future mode gated on the hiding
audit in `docs/spec/06-security.md#privacy-and-zk`. The correct term is "succinct
validity proof".

### 6. Trust boundaries

```text
+-----------------------------+        +-----------------------------+
| Untrusted: prover           |        | Trusted: verifier build     |
|  - chooses witness          |        |  - relation_id allowlist    |
|  - chooses model/quant only |  proof |  - reconstructs H_pi        |
|    within the committed      | =====> |  - Stwo verifier + FRI/LogUp|
|    manifest (else digest     |        |  - structural pre-checks    |
|    changes -> reject)        |        |  - NO PyTorch, NO Python    |
+-----------------------------+        +-----------------------------+
        public inputs (relation_id, four commitments, statement-specific
        inputs/commitments, claimed output) cross the boundary in the clear.
```

The prover is fully untrusted: its only leverage is choosing a witness consistent
with the public inputs. Because every load-bearing value is bound (INV-SEC-02),
the prover cannot change the model, quantization, planner rule, inputs, or
outputs without producing a different `relation_id`-bound digest and thus a proof
that fails to verify. This is the threat model elaborated in
`docs/spec/06-security.md#threat-model` and `#trust-boundaries`.

## Alternatives Considered

1. Single universal relation with statement encoded only as a public flag.
   A single `relation_id` (`pwm.lewm.inference.v1`) with `statement_type` as an
   ordinary, possibly-unbound public field. Considered because it minimizes the
   allowlist and lets one verifier circuit cover everything. Rejected: if
   `statement_type` is not bound into the digest, statement substitution is
   trivial (attack 1 in Motivation); if it is bound but shares one circuit, the
   circuit must constrain every statement's relation simultaneously, which is
   strictly more expensive and harder to audit than per-statement relations, and
   a bug in one statement's constraints weakens all of them. Per-statement
   `relation_id`s give independent audit surfaces and let V0 ship P2 without the
   P3/P4 circuitry existing.

2. Statement validity enforced only by the verifier's structural pre-checks
   (steps 1–4 of §4), not by transcript binding. Considered because structural
   checks are cheap and easy to reason about. Rejected: structural checks alone do
   not bind `relation_id` cryptographically. A prover who controls the artifact
   could present a P0 proof with a `PublicInput` hand-edited to claim P1; the
   structural checks (which only inspect field presence) would pass, and if the
   underlying Stwo proof did not also depend on `relation_id`, verification would
   succeed. Binding `H_pi` (including `relation_id`) into the Fiat-Shamir
   transcript is what makes substitution infeasible; the structural checks are
   retained only as early, precise error reporting (defense in depth), not as the
   security boundary.

3. Binding the model/planner via the witness instead of the public input.
   Put `model_commitment` and `planner_config_commitment` only in the witness and
   constrain them in-circuit. Considered because it shrinks the public input.
   Rejected: a relying party must see the model and planner identity to decide
   whether the proof is about the model they care about. If those commitments are
   witness-only, two different models could each produce a valid proof and a
   verifier could not tell them apart from the public input. Commitments that a
   relying party must inspect belong in `PublicInput` and the digest, per
   INV-SEC-02.

## Drawbacks

- Adding any new statement or any semantic revision requires minting a new
  `relation_id`, a code change to the verifier allowlist, and a release. This is
  intentional friction (it is what prevents silent semantics drift) but it makes
  experimentation slower; experimental relations must use a clearly non-production
  `vN` and a separate verifier build.
- Per-statement relations mean P2's headline proof, although it composes P0 + P1
  + cost + argmin, has its own `relation_id` and is not interchangeable with a P1
  proof even over the same trajectory. A relying party that wants both a P1
  attestation and a P2 attestation needs two proofs (or recursion, deferred to
  RFC-0012). This is the correct trade: it prevents a P2 proof from being passed
  off as a standalone rollout attestation and vice versa.
- The taxonomy is closed to exactly five statements. A genuinely new proof shape
  (for example, a reward-model proof) is out of scope and would require a new RFC,
  not a new `relation_id` version. This RFC deliberately does not leave an
  open-ended extension point in the `StatementType` enum.

## Migration / Rollout

- V0 ships P2 (`pwm.lewm.fixed_candidate_planning.v1`) as the first public
  relation, with P0/P1 relations available as building blocks at v0.2. P3 and P4
  relations are reserved (their `StatementType` discriminants exist) but their
  verifier support is `Future`; a v1.0 verifier rejects them with
  `VerifyError::UnsupportedStatement` (step 3 of §4).
- Versioning: relation revisions follow
  `docs/spec/09-release-and-versioning.md#relation-versioning`. A new `vN`
  coexists with the old one; the verifier allowlist may carry both during a
  deprecation window. Old `relation_id`s are deprecated, not deleted, until the
  window closes; removal is a breaking change per
  `docs/spec/09-release-and-versioning.md#semver`.
- The `ProofArtifact.artifact_version` field (contract §6.5) gates structural
  changes to `PublicInput` (for example, the P4 pixel-commitment fields). A
  verifier rejects an artifact whose `artifact_version` it does not understand
  before inspecting `relation_id`.
- Feature flags: P3/P4 verifier support is behind compile-time Cargo features
  (`statement-cem`, `statement-pixel`). The allowlist (INV-SEC-01) is assembled
  from the enabled features, so a default v1.0 build cannot accept a P3 proof even
  if one is presented. This makes the supported-statement set a build-time fact,
  not a runtime configuration.

## Testing Strategy

Cross-references: `docs/spec/07-testing-strategy.md#negative-tests`,
`#golden-vectors`, `#ci-gates`, and the rejection taxonomy in
`docs/spec/04-error-model.md#verifier-rejections`.

Accepting tests (each is a named CI test):

- `accept_p0_step_roundtrip`: a valid P0 proof over the reference predictor
  verifies; `relation_id == pwm.lewm.predictor_step.v1`.
- `accept_p1_rollout_horizon5`: a valid P1 proof over a horizon-5 rollout
  verifies; trajectory root matches `claimed_output_commitment`.
- `accept_p2_planning_min_selected`: a valid P2 proof with the genuine argmin
  selection verifies; `selected_cost == cost[selected_index]`.
- `accept_p2_committed_weights`: P2 with `weights.visibility = private_committed`
  verifies; only `model_commitment` is public.

Rejecting / negative tests (the locked decision and INV-SEC enforcement):

- `reject_p0_submitted_as_p1`: take a valid P0 artifact, rewrite its
  `relation_id` and `statement_type` to P1, leave the proof bytes unchanged.
  MUST reject. Expected first failure: `StatementRelationMismatch` (step 2) if
  only `relation_id` is changed; `ProofInvalid` (step 6) once the transcript
  digest diverges. This is the canonical test for INV-SEC-04 and the locked
  decision.
- `reject_unknown_relation`: artifact with `relation_id = pwm.lewm.predictor_step.v2`
  on a verifier whose allowlist has only `v1`. MUST reject with
  `UnknownRelation`.
- `reject_version_substitution`: a valid `...v1` proof presented under a `...v2`
  `relation_id`. MUST reject (INV-SEC-04 covers version substitution, not only
  statement substitution).
- `reject_changed_model_commitment`: flip one byte of `model_commitment`. MUST
  reject with `ProofInvalid` (digest mismatch).
- `reject_changed_quantization_commitment`: flip one byte of
  `quantization_commitment`. MUST reject.
- `reject_changed_planner_config_commitment`: flip one byte of
  `planner_config_commitment` on a P2 artifact. MUST reject.
- `reject_p2_missing_selected_index`: P2 artifact with `selected_index = None`.
  MUST reject with `MalformedPublicInput` (step 4, required field absent).
- `reject_p0_with_goal_latent`: P0 artifact with `goal_latent_public = Some(..)`.
  MUST reject with `MalformedPublicInput` (forbidden field present).
- `reject_p4_on_v1_build`: P4 artifact on a v1.0 build. MUST reject with
  `UnsupportedStatement`.
- `reject_p2_partial_planning`: a crafted P2 proof that constrains only the
  selected candidate's rollout/cost (omitting other candidates' cost
  constraints). MUST reject (INV-SEC-05); such a witness cannot satisfy the
  P2 AIR (RFC-0009) and cannot mint the P2 `relation_id`.
- `reject_p2_nonmin_selection`: P2 with `selected_index` not attaining the
  minimum cost. MUST reject.
- `reject_p2_tiebreak_violation`: P2 with two candidates tied at the minimum and
  `selected_index` not the smallest such index. MUST reject (tie-break, contract
  §3).
- `reject_input_variant_swap`: take a valid P2 proof whose `latent_history` was
  bound as `_commitment`, present it with `latent_history_public = Some(..)`.
  MUST reject (INV-SEC-03, presence/variant tag in digest).

Differential test:

- `diff_relation_id_string_to_digest`: assert the Rust `relation_id` string ->
  32-byte digest mapping matches the Python export reference (RFC-0001), so prover
  and verifier agree on every allowlist entry. Drift here would silently break
  INV-SEC-04.

CI gate: per RFC-0013, no statement is considered shippable without both an
`accept_*` and the full set of `reject_*` tests above passing
(`docs/spec/07-testing-strategy.md#ci-gates`).

## Open Questions

- OPEN QUESTION (owner: maintainers / security area): should the verifier
  allowlist additionally pin a minimum `manifest_version` per `relation_id`, to
  reject proofs that bind an obsolete manifest schema even when the `relation_id`
  is current? Resolution path: decide in RFC-0001 (manifest schema) before v0.1
  freeze; default if unresolved is no minimum (the `relation_id` version is the
  sole gate).
- OPEN QUESTION (owner: maintainers / verifier area): for P2/P3, do we expose the
  per-candidate cost vector in `PublicInput`, or only `selected_index` /
  `selected_cost` plus the in-circuit all-candidate constraint? Resolution path:
  RFC-0009 (fixed-candidate planner) at v1.0; this RFC requires only that all
  costs be bound, not that they be public.

## References

- Founding analysis: `docs/feasibility-study.md` §0, §1.1, §1.2, §8, §9, §10,
  §11 (source RFC-000), §14 (recommended V0 statement).
- `docs/spec/00-overview.md#scope-and-statement-tiers`, `#v0-statement`,
  `#feasibility-verdict`.
- `docs/spec/03-data-model.md#public-input`, `#witness`, `#model-manifest`,
  `#relation-id`, `#bounded-integers`.
- `docs/spec/04-error-model.md#verifier-rejections`, `#failure-modes`.
- `docs/spec/06-security.md#threat-model`, `#trust-boundaries`,
  `#soundness-requirements`, `#binding-requirements`, `#privacy-and-zk`.
- `docs/spec/07-testing-strategy.md#negative-tests`, `#golden-vectors`,
  `#ci-gates`.
- `docs/spec/09-release-and-versioning.md#relation-versioning`, `#semver`,
  `#deprecation`.
- RFC-0001 (manifest and export), RFC-0009 (fixed-candidate planner), RFC-0010
  (CEM), RFC-0011 (pixel encoder), RFC-0014 (canonical serialization and
  transcript), RFC-0013 (testing/audit strategy).
- Prior art: Circle STARK over M31, FRI/PCS, LogUp lookups (Stwo, verified
  against upstream at 2026-06-03). JEPA / LeWorldModel (`github.com/lucas-maes/le-wm`,
  MIT; reference only, not vendored).
