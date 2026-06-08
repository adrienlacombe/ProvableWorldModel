# Error model: taxonomy, failure modes, recovery, and verifier rejections

Status: Normative. This document is the single source of truth for how ProvableWorldModel classifies, signals, and recovers from every error across the pipeline (export/quantization, manifest validation, trace building, proving, verification). It defines the typed error enums each crate returns, the failure-mode catalogue with triggers and responses, the boundary between soundness failures (which must reject and never recover) and operational errors (which abort safely with an actionable message), the recovery policy, and the named invariants `INV-ERR-NN` that bind the whole system to "abort, emit nothing, never silently degrade."

This document is normative for `pwm-export`, `pwm-core`, `pwm-prover`, and `pwm-verifier`. Types referenced here are owned by `docs/spec/03-data-model.md`; the verifier entry point is owned by `docs/spec/02-public-api.md`. Soundness rationale lives in `docs/spec/06-security.md#soundness-requirements`; the tests that exercise every rejecting path live in `docs/spec/07-testing-strategy.md#negative-tests`.

---

## 0. Design principles for the error model

The proof system is a soundness boundary, not a best-effort service. The error model is built around four principles, each enforced by an invariant below.

1. **Reject over repair.** No stage attempts to "fix up" data that violates an arithmetic or commitment contract. A field wraparound, an accumulator that exceeds its declared bound, or a commitment mismatch is a hard stop, not a warning. This is the difference between this system and a numerical library: a numerical library that clamps is helpful; a proof system that clamps outside a declared range is unsound.
2. **No partial proof.** A proof is an all-or-nothing artifact. The prover either produces a complete `ProofArtifact` (see `docs/spec/02-public-api.md#artifact-formats`) that a conformant verifier accepts, or it emits nothing and exits non-zero. There is no "proof of the first 4 of 5 candidates."
3. **Typed errors, not strings.** Every fallible boundary returns a typed Rust `enum`, not an opaque string or panic. Strings are for the human-readable `Display` impl only. A caller can match on the variant and route programmatically.
4. **The two-class rule.** Every error is exactly one of two classes: a **soundness failure** (the witness or proof is arithmetically/cryptographically invalid; rejection is mandatory and is a *correct* outcome of the protocol) or an **operational error** (a configuration, I/O, resource, or schema problem that prevents the system from running, independent of whether the underlying claim is true). The distinction drives recovery policy: soundness failures are never retried or worked around; operational errors abort cleanly and may be retried after the operator fixes the cause.

> ASCII overview of where each error enum is raised:
>
> ```text
>   Python/Rust export        manifest load        trace build         prove           verify
>   ┌───────────────┐   ┌──────────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐
>   │ ExportError   │──▶│ ManifestError    │──▶│ TraceError │──▶│ ProveError │   │ VerifyError│
>   └───────────────┘   └──────────────────┘   └────────────┘   └────────────┘   └────────────┘
>        producer side (pwm-export, pwm-prover) — never runs in the verifier        consumer side
> ```
>
> In prose: export and quantization errors surface first; the manifest is validated and committed; the trace builder turns the committed manifest plus witness into AIR traces; the prover commits and produces a proof; and only the verifier (which never runs Python or PyTorch, per `docs/spec/02-public-api.md#rust-public-api`) ever returns a `VerifyError`. An honest prover that detects a soundness condition aborts with `ProveError` *before* emitting anything; a dishonest prover that forces a bad witness through produces a proof the verifier rejects with `VerifyError`.

---

<a id="error-taxonomy"></a>

## 1. Error taxonomy
Errors are organized by pipeline stage. Each stage owns exactly one top-level enum. Variants carry structured context (ids, indices, bounds) so a caller can act without parsing the `Display` string. The enums below are the canonical signatures for this document; `pwm-core` re-exports the shared ones.

### 1.1 Class legend

| Class | Meaning | Recovery |
| --- | --- | --- |
| **S** (soundness) | The witness/proof is arithmetically or cryptographically invalid. Rejection is the protocol working as designed. | Never repaired, never retried, never silently degraded. See [§4 Recovery policy](#recovery). |
| **O** (operational) | A config, I/O, resource, or schema problem prevents running, independent of claim validity. | Safe abort with actionable message; operator fixes cause; may re-run. No partial output. |

Every variant below is tagged **S** or **O**.

### 1.2 `ExportError` — export and quantization (`pwm-export`)

Raised by the Python+Rust export/quantization pipeline (PyTorch checkpoint → quantized graph → manifest → golden vectors). See `docs/rfcs/RFC-0001-model-manifest-and-export-pipeline.md` for the pipeline and `docs/rfcs/RFC-0002-fixed-point-arithmetic-over-m31.md` for the arithmetic semantics these errors guard.

```rust
/// Errors from quantizing a checkpoint and emitting a manifest + golden vectors.
/// Producer-side only; never reaches the verifier.
pub enum ExportError {
    /// O: checkpoint/config file missing, unreadable, or wrong format.
    CheckpointIo { path: String, source: String },
    /// O: the exported graph contains an op the quantizer has no rule for.
    UnsupportedOp { op_id: String, op_kind: String },
    /// O: an architecture field (depth/heads/dim_head/mlp_dim/latent_dim) read
    /// from the checkpoint disagrees with the target reference configuration.
    ArchitectureMismatch { field: String, found: String, expected: String },
    /// S: a quantized weight/bias/scale falls outside the declared per-tensor
    /// range [lo, hi]; embedding it into M31 would be ambiguous (RFC-0002).
    QuantizedValueOutOfRange { tensor_id: u32, index: u64, value: i64, lo: i64, hi: i64 },
    /// S: chosen scale would let a worst-case accumulator exceed its declared
    /// int32/limb bound for this op (overflow is unsound; RFC-0002 §accumulators).
    ScaleOverflowsAccumulator { op_id: String, worst_case_abs: i128, accumulator_bound: i128 },
    /// S: a requantization shift/zero-point is inconsistent with declared scales,
    /// so Python and AIR rounding could diverge.
    RequantParamsInconsistent { op_id: String, detail: String },
    /// S: an activation/normalization approximation table has a domain gap or a
    /// duplicate/non-canonical entry (would make the LogUp lookup ill-defined).
    ApproximationTableInvalid { table_id: String, detail: String },
    /// O: re-export of the same checkpoint produced different bytes
    /// (violates byte-identical re-export; RFC-0001 acceptance criteria).
    NonDeterministicExport { canonical_json_hash_a: String, canonical_json_hash_b: String },
    /// S: the Python fixed-point reference and the Rust fixed-point reference
    /// disagree bit-for-bit on a golden vector (parity gate; RFC-0001/RFC-0013).
    GoldenVectorParityFailure { vector_id: String, op_id: String, first_diff_index: u64 },
    /// O: declared activation/weight dtype not in the supported set
    /// (int8 weights; int8/int16 activations; int32 biases).
    UnsupportedDtype { tensor_id: u32, dtype: String },
}
```

Note the split: `QuantizedValueOutOfRange`, `ScaleOverflowsAccumulator`, `RequantParamsInconsistent`, `ApproximationTableInvalid`, and `GoldenVectorParityFailure` are **soundness (S)** — they describe states where, if the export were allowed to proceed, the resulting proof would attest to an arithmetic relation that does not hold over the integers. Export must refuse to emit a manifest in those cases. The rest are **operational (O)**.

### 1.3 `ManifestError` — manifest load and commitment (`pwm-core`)

Raised when loading and validating a manifest and recomputing its commitments. The manifest schema and the binding requirements are owned by `docs/spec/03-data-model.md#model-manifest`; serialization and digest rules by `docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md`.

```rust
/// Errors from parsing, validating, and committing a model manifest.
pub enum ManifestError {
    /// O: file missing or unreadable.
    Io { path: String, source: String },
    /// O: YAML does not parse, or a required key is absent / wrong type.
    Schema { key: String, detail: String },
    /// O: manifest_version is not a version this build understands.
    UnsupportedManifestVersion { found: String, supported: Vec<String> },
    /// O: relation_id string is malformed (not `pwm.lewm.<statement>.v<N>`).
    MalformedRelationId { found: String },
    /// S: recomputed canonical_json_hash != the manifest's declared hash
    /// (the manifest was edited after signing/committing).
    SerializationHashMismatch { declared: String, recomputed: String },
    /// S: recomputed model commitment != weights.root in the manifest.
    ModelCommitmentMismatch { declared: String, recomputed: String },
    /// S: recomputed quantization commitment != declared.
    QuantizationCommitmentMismatch { declared: String, recomputed: String },
    /// S: an op references a scale_id / tensor_id / table_id not present in the
    /// manifest tables (an unbound reference is prover-mutable, thus unsound).
    DanglingReference { op_id: String, kind: String, id: String },
    /// O: overflow_policy is anything other than `reject` (V0 supports reject only).
    UnsupportedOverflowPolicy { found: String },
    /// O: more than one active rounding mode is declared (must be exactly one).
    AmbiguousRoundingMode { found: Vec<String> },
}
```

### 1.4 `TraceError` — trace building and reference inference (`pwm-prover`)

Raised by the trace builder while executing the deterministic fixed-point reference and laying out AIR traces. This is where the prover *honestly detects* that the supplied witness cannot satisfy the relation and aborts before producing anything. Arithmetic semantics are `docs/rfcs/RFC-0002-fixed-point-arithmetic-over-m31.md`; tensor wiring is `docs/rfcs/RFC-0004-tensor-memory-and-wiring-air.md`; lookups are `docs/rfcs/RFC-0003-range-check-and-lookup-infrastructure.md`.

```rust
/// Errors from reference inference + AIR trace construction.
pub enum TraceError {
    /// O: a public-input/witness tensor shape disagrees with the manifest op shape.
    ShapeMismatch { op_id: String, expected: Vec<u32>, found: Vec<u32> },
    /// S: a value would wrap mod p if encoded directly; integer semantics broken.
    FieldWraparound { op_id: String, index: u64, value: i128 },
    /// S: an accumulator exceeds its declared int32/limb bound (overflow=reject).
    AccumulatorOverflow { op_id: String, value: i128, bound: i128 },
    /// S: requantization remainder not in [0, 2^r) or violates the declared
    /// rounding mode (nearest_ties_to_even | truncate_toward_zero).
    InvalidRoundingRemainder { op_id: String, n: i128, shift: u32, rem: i128 },
    /// S: a comparison/difference witness is negative where the relation requires
    /// it nonnegative (would wrap into a valid-looking field value).
    ComparisonWraparound { op_id: String, lhs: i128, rhs: i128 },
    /// S: an activation/softmax/inv-sqrt input falls outside the committed table
    /// domain — no valid lookup row exists.
    OutOfDomainLookup { table_id: String, input: i128, domain_lo: i128, domain_hi: i128 },
    /// S: an op reads a tensor cell that was never written (no producing write),
    /// violating read/write consistency (RFC-0004).
    MissingTensorWrite { tensor_id: u32, index: [u32; 4] },
    /// S: a read used a different scale_id than the producing write.
    WrongScale { tensor_id: u32, index: [u32; 4], read_scale: u32, write_scale: u32 },
    /// S: a broadcast/reshape/transpose/concat/slice was performed that the
    /// manifest does not authorize for this op.
    UnpermittedBroadcast { op_id: String, detail: String },
    /// S: the argmin tie-break or minimality witness is internally inconsistent
    /// (some cost_s - selected_cost < 0, or selected_index not minimal).
    ArgminWitnessInconsistent { detail: String },
    /// O: a required witness field for the statement_type is absent
    /// (e.g. goal_latent missing for P2).
    MissingWitnessField { field: String, statement_type: String },
}
```

`ShapeMismatch`, `MissingWitnessField` are **operational** (the caller wired up inputs wrong). Everything else is **soundness**: the witness is arithmetically inconsistent with the committed relation, and the honest prover refuses to continue.

### 1.5 `ProveError` — proving (`pwm-prover`)

Raised by the prover after a valid trace exists. Wraps trace errors and adds proving-stage failures. The CLI and artifact bundle are owned by `docs/rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md`.

```rust
/// Errors from the proving stage. On ANY variant, the prover emits no artifact.
pub enum ProveError {
    /// Propagated soundness/operational failure detected during reference inference
    /// and trace layout. Class follows the inner TraceError.
    Trace(TraceError),
    /// Propagated manifest validation failure. Class follows the inner ManifestError.
    Manifest(ManifestError),
    /// S: a built AIR constraint is not satisfied by the witness trace
    /// (internal consistency check before committing; should be unreachable for an
    /// honest witness and indicates a witness or trace-builder bug).
    ConstraintUnsatisfied { component: String, constraint: String, row: u64 },
    /// S: a LogUp interaction (range/lookup/memory) claimed-sum is nonzero, i.e.
    /// multiplicities do not balance (RFC-0003).
    LookupImbalance { relation: String, claimed_sum_nonzero: bool },
    /// O: insufficient memory/time to build or commit the trace at the requested size.
    ResourceExhausted { stage: String, detail: String },
    /// O: failure inside the vendored Stwo prover (FRI/PCS/commitment) not
    /// attributable to a constraint (I/O, internal invariant). Pinned rev in
    /// third_party/stwo/REVISION (RFC-0015).
    Backend { stage: String, detail: String },
    /// O: could not write the ProofArtifact bundle to disk.
    ArtifactIo { path: String, source: String },
}
```

`ConstraintUnsatisfied` and `LookupImbalance` are **soundness** signals encountered prover-side: an honest prover that hits them has either been handed an invalid witness or has a bug, and in both cases it must abort and emit nothing rather than ship a proof that would fail verification (or, worse, an unsound proof).

### 1.6 `VerifyError` — verification (`pwm-verifier`)

The consumer-side enum. `verify(artifact: &ProofArtifact) -> Result<(), VerifyError>` (signature owned by `docs/spec/02-public-api.md#rust-public-api`) returns `Ok(())` only when every check below passes. Every variant is a **rejection**; the verifier never accepts and never repairs. Variants are fully specified in [§3 Verifier rejection taxonomy](#verifier-rejections).

```rust
pub enum VerifyError {
    RelationIdUnsupported { found: [u8; 32] },
    ArtifactVersionUnsupported { found: u32, supported: Vec<u32> },
    ModelCommitmentMismatch { expected: [u8; 32], found: [u8; 32] },
    QuantizationCommitmentMismatch { expected: [u8; 32], found: [u8; 32] },
    PlannerConfigMismatch { expected: [u8; 32], found: [u8; 32] },
    PublicInputDigestMismatch { recomputed: [u8; 32], in_transcript: [u8; 32] },
    ShapeMismatch { field: String, expected: Vec<u32>, found: Vec<u32> },
    SelectedIndexOutOfRange { selected_index: u32, num_candidates: u32 },
    SelectedCostMismatch { selected_index: u32, detail: String },
    TieBreakViolation { offending_index: u32, selected_index: u32 },
    OutOfDomainLookup { relation: String },
    RangeViolation { column: String },
    AccumulatorOverflow { component: String },
    ProofInvalid { stage: String, detail: String },
}
```

---

<a id="failure-modes"></a>

## 2. Failure modes
Each row names a concrete failure, the precise condition that triggers it, and the system's response (which enum/variant fires and at which stage). The class column (**S**/**O**) determines recovery per [§4](#recovery). Every soundness row corresponds to at least one negative test in `docs/spec/07-testing-strategy.md#negative-tests`.

### 2.1 Arithmetic and field-safety failures

| # | Failure | Trigger | System response | Class |
| --- | --- | --- | --- | --- |
| F1 | Field wraparound | An integer value whose magnitude is `>= p = 2^31 - 1` is encoded into M31, so distinct integers map to the same field element. | Honest prover: `TraceError::FieldWraparound` → `ProveError::Trace`, abort, emit nothing. Dishonest prover forcing it through: a range relation has no valid table row → `VerifyError::RangeViolation`. | S |
| F2 | Accumulator overflow | A dot-product/accumulator chain exceeds its declared int32/limb bound, even if `acc mod p` looks valid. | Honest prover: `TraceError::AccumulatorOverflow` (overflow_policy=reject). Verifier: the accumulator range column fails → `VerifyError::AccumulatorOverflow`. | S |
| F3 | Invalid rounding remainder | `n = q*2^r + rem` with `rem` not in `[0, 2^r)`, or the tie case is resolved against the manifest's declared rounding mode. | Honest prover: `TraceError::InvalidRoundingRemainder`. Verifier: requant constraint + remainder range fail → `VerifyError::RangeViolation` (or `ProofInvalid` if the FRI check catches the broken constraint). | S |
| F4 | Comparison wraparound | A claimed-nonnegative difference (e.g. `cost_s - selected_cost`) is actually negative; `value mod p` is a large positive field element that passes a naive equality but fails the range proof. | Honest prover: `TraceError::ComparisonWraparound`. Verifier: difference range column fails → `VerifyError::RangeViolation`; for the planner specifically → `VerifyError::TieBreakViolation` or `SelectedCostMismatch`. | S |
| F5 | Out-of-domain activation lookup | An activation/softmax/inverse-sqrt input lies outside the committed table domain, so no lookup row exists. | Honest prover: `TraceError::OutOfDomainLookup`. Verifier: LogUp lookup has no matching multiplicity → `VerifyError::OutOfDomainLookup` (claimed-sum imbalance manifests as `ProofInvalid` if forced). | S |
| F6 | Wrong scale | A read uses a different `scale_id` than the producing write for the same `TensorCell`. | Honest prover: `TraceError::WrongScale`. Verifier: the `scale_id` field is part of the tensor-memory permutation tuple, so the multiset argument fails → `VerifyError::ProofInvalid` (memory relation) or `RangeViolation`. | S |

### 2.2 Wiring and memory failures

| # | Failure | Trigger | System response | Class |
| --- | --- | --- | --- | --- |
| F7 | Missing tensor write | An op reads a `TensorCell` (`tensor_id`,`index`) that no prior op wrote and no manifest constant supplies. | Honest prover: `TraceError::MissingTensorWrite`. Verifier: read tuple has no matching write in the permutation/multiset argument → `VerifyError::ProofInvalid` (tensor-memory relation). | S |
| F8 | Broadcast without manifest permission | An op broadcasts/reshapes/transposes/concats/slices in a way the manifest's op definition does not authorize. | Honest prover: `TraceError::UnpermittedBroadcast`. Verifier: the unauthorized read does not match a sanctioned constant/write pattern → `VerifyError::ProofInvalid` (wiring relation). | S |

### 2.3 Commitment and binding failures

| # | Failure | Trigger | System response | Class |
| --- | --- | --- | --- | --- |
| F9 | Model commitment change | Weights/biases/architecture/shapes differ from what `model_commitment` binds. | Prover load: `ManifestError::ModelCommitmentMismatch`. Verifier: `PublicInput.model_commitment` is bound into the transcript; any change shifts the digest → `VerifyError::ModelCommitmentMismatch` (or `PublicInputDigestMismatch` if the prover lied about the digest). | S |
| F10 | Quantization commitment change | Scales/rounding/clamp ranges/approximation tables differ from `quantization_commitment`. | Prover load: `ManifestError::QuantizationCommitmentMismatch`. Verifier: `VerifyError::QuantizationCommitmentMismatch`. | S |
| F11 | Planner commitment change | Candidate count / cost rule / tie-break policy differ from `planner_config_commitment`. | Verifier: `VerifyError::PlannerConfigMismatch`. | S |
| F12 | Serialization tamper | The manifest body was edited but `canonical_json_hash` was not recomputed. | Prover load: `ManifestError::SerializationHashMismatch`. Verifier: digest divergence → `VerifyError::PublicInputDigestMismatch`. | S |

### 2.4 Operational failures

| # | Failure | Trigger | System response | Class |
| --- | --- | --- | --- | --- |
| F13 | Unsupported op | The exported graph has an op the quantizer cannot lower. | `ExportError::UnsupportedOp`, abort export; operator extends the quantizer or removes the op. | O |
| F14 | Schema/version mismatch | Manifest key missing/wrong-typed, or `manifest_version`/`artifact_version` unknown to this build. | `ManifestError::Schema` / `UnsupportedManifestVersion`; verifier: `VerifyError::ArtifactVersionUnsupported`. | O |
| F15 | Shape mismatch | A public-input/witness tensor shape disagrees with the manifest op shape. | Prover: `TraceError::ShapeMismatch`. Verifier: `VerifyError::ShapeMismatch`. | O |
| F16 | Resource exhaustion | Trace too large to build/commit in available memory/time. | `ProveError::ResourceExhausted`, abort, emit nothing; operator reduces candidates×horizon or provisions more resources. | O |
| F17 | Backend / I/O failure | Vendored Stwo internal failure or disk error writing the artifact. | `ProveError::Backend` / `ProveError::ArtifactIo`, abort, emit nothing. | O |
| F18 | Non-deterministic export | Re-exporting the same checkpoint yields different canonical bytes. | `ExportError::NonDeterministicExport`, abort; this is a reproducibility bug (RFC-0001), not a claim about validity. | O |

---

<a id="verifier-rejections"></a>

## 3. Verifier rejection taxonomy
The verifier is the soundness gate. It runs a fixed, ordered sequence of checks and returns the *first* failing one as a typed `VerifyError`. It never runs Python or PyTorch and never re-quantizes — it verifies the committed arithmetic relation only (`docs/spec/02-public-api.md#rust-public-api`). Ordering matters: cheap, structural checks run before the expensive cryptographic proof check so that a malformed artifact fails fast and predictably.

### 3.1 Check order

```text
 1. RelationIdUnsupported            ← is this build allowed to verify this relation?
 2. ArtifactVersionUnsupported       ← can this build parse this bundle layout?
 3. ModelCommitmentMismatch          ← does the bound model match the expected one?
 4. QuantizationCommitmentMismatch   ← does the bound quantization match?
 5. PlannerConfigMismatch            ← does the bound planner config match?
 6. PublicInputDigestMismatch        ← does recomputed PI digest match the transcript?
 7. ShapeMismatch                    ← do public tensor shapes match the manifest?
 8. SelectedIndexOutOfRange          ← (P2+) is selected_index < num_candidates?
 9. <Stwo proof verification>        ← FRI/PCS + all AIR constraints + LogUp sums
      ├─ RangeViolation              ← a bounded column left its range
      ├─ AccumulatorOverflow         ← accumulator range column failed
      ├─ OutOfDomainLookup           ← an activation/table lookup had no row
      └─ ProofInvalid                ← any other constraint / FRI / PCS failure
10. SelectedCostMismatch             ← (P2+) selected_cost == cost[selected_index]?
11. TieBreakViolation                ← (P2+) selected_index is the smallest minimizer?
```

Checks 1–8 are structural and run before the proof is verified. Check 9 is the Stwo proof verification; constraint/lookup failures inside it are surfaced as the specific variants `RangeViolation`, `AccumulatorOverflow`, `OutOfDomainLookup`, or the catch-all `ProofInvalid`. Checks 10–11 are application-level planner checks decoded from the verified public output and re-validated against the proof's committed costs. A planner proof binds all candidate costs into the relation, so checks 10–11 cannot be satisfied by a prover that scored only a favorable subset (see `docs/spec/06-security.md#soundness-requirements` and RFC-0009).

### 3.2 Variant reference

| Variant | Meaning (one line) | Triggered by |
| --- | --- | --- |
| `RelationIdUnsupported` | The artifact's `relation_id` is not one this verifier build is configured to accept (e.g. a P0 proof submitted to a P2 verifier, or an unminted/typo'd id). | F-relation; RFC-0000 |
| `ArtifactVersionUnsupported` | `artifact_version` is outside the set this build can parse. | F14 |
| `ModelCommitmentMismatch` | `model_commitment` differs from the expected committed model. | F9 |
| `QuantizationCommitmentMismatch` | `quantization_commitment` differs from expected. | F10 |
| `PlannerConfigMismatch` | `planner_config_commitment` differs from expected. | F11 |
| `PublicInputDigestMismatch` | The digest recomputed from the canonical public-input serialization differs from the digest bound into the Fiat-Shamir transcript. | F12; RFC-0014 |
| `ShapeMismatch` | A public tensor (latent history, goal, candidate actions, claimed output) has a shape inconsistent with the manifest. | F15 |
| `SelectedIndexOutOfRange` | `selected_index >= num_candidates`. | planner; RFC-0009 |
| `SelectedCostMismatch` | `selected_cost != cost[selected_index]` as bound in the proof. | planner; RFC-0009 |
| `TieBreakViolation` | `selected_index` is not the smallest index attaining the minimum cost (some `s < selected_index` has `cost_s - selected_cost - 1 >= 0` failing, i.e. `cost_s <= selected_cost`). | F4; RFC-0009 |
| `OutOfDomainLookup` | A LogUp lookup (activation/softmax/inv-sqrt/range table) referenced a value with no committed table row. | F5; RFC-0003/RFC-0006 |
| `RangeViolation` | A bounded-integer column took a value outside its declared `[lo, hi]` range. | F1, F3, F4; RFC-0002/RFC-0003 |
| `AccumulatorOverflow` | An accumulator range column exceeded its declared int32/limb bound. | F2; RFC-0002/RFC-0005 |
| `ProofInvalid` | Any other Stwo proof failure: a non-range AIR constraint, a LogUp claimed-sum imbalance (including tensor-memory permutation / wiring), or a FRI/PCS check failure. | F6, F7, F8; RFC-0004 |

### 3.3 Why these and only these

The variant set is closed by design. Adding a "soft" outcome (a warning, a `MaybeValid`, a numeric tolerance) would breach `INV-ERR-05`. The verifier's result type is `Result<(), VerifyError>`: success is the unit type (there is nothing to "trust but verify"), and every failure is one of the variants above. There is no degraded-accept path.

---

<a id="recovery"></a>

## 4. Recovery policy
### 4.1 Prover recovery policy

The prover follows a strict abort-and-emit-nothing policy. The pseudocode below is normative for `pwm-prover`; the concrete CLI is RFC-0016.

```text
fn run_prove(manifest_path, witness):
    manifest = load_manifest(manifest_path)?         // ManifestError → abort, exit nonzero, no artifact
    reference = run_fixed_point_reference(manifest, witness)?  // TraceError(S) → abort
    trace     = build_traces(manifest, witness, reference)?    // TraceError → abort
    // Self-check BEFORE any commitment leaves the process:
    assert_constraints_satisfied(trace)?             // ProveError::ConstraintUnsatisfied → abort
    assert_lookup_sums_balance(trace)?               // ProveError::LookupImbalance → abort
    proof = stwo_prove(trace)?                       // ProveError::Backend/ResourceExhausted → abort
    write_artifact(proof, public_input)?             // ProveError::ArtifactIo → abort
    // Only here does an artifact exist on disk.
```

Rules:

- **On any soundness condition** (F1–F12), the prover aborts. It does not clamp the offending value into range, does not silently widen a declared bound, does not skip a candidate, and does not emit a "partial" proof. `overflow_policy=reject` is the only supported policy in V0 (`docs/spec/03-data-model.md#bounded-integers`; RFC-0002).
- **On any operational error** (F13–F18), the prover aborts with an actionable `Display` message naming the offending op/tensor/field and exits non-zero. No artifact is written. The operator fixes the cause (extend the quantizer, fix shapes, provision resources) and re-runs from scratch.
- **No silent clamping outside declared clamp ranges.** A manifest may declare explicit per-tensor clamp ranges (`clamp_policy: explicit`); clamping inside those declared ranges is part of the *specified* relation and is committed. Clamping a value the manifest did not authorize is a soundness violation, not a recovery.
- **The self-check before committing** (`ConstraintUnsatisfied`, `LookupImbalance`) guarantees an honest prover never ships a proof that a conformant verifier would reject for an internal-consistency reason. If the self-check fires, it is either a malformed witness or a trace-builder bug; either way, abort, never ship.

### 4.2 Verifier recovery policy

The verifier has no recovery path by construction. It returns `Ok(())` or a typed `VerifyError`. It never:

- accepts with a warning,
- applies a numeric tolerance to a commitment or a range,
- re-runs the model to "double-check" (it cannot — no Python/PyTorch in the verifier),
- or downgrades a rejection to a softer status.

A `VerifyError` is a *correct, final* outcome of the protocol when the proof is invalid. It is not an exceptional condition to be retried; retrying the same artifact yields the same `VerifyError` deterministically (RFC-0016 reproducibility contract).

### 4.3 Soundness failures vs operational errors (the decision table)

| Property | Soundness failure (S) | Operational error (O) |
| --- | --- | --- |
| What it means | The witness/proof is arithmetically or cryptographically invalid. | A config/I/O/resource/schema problem blocks running. |
| Correct outcome | Reject (prover aborts; verifier returns the matching `VerifyError`). | Safe abort with actionable message. |
| May the system repair it? | No. Never clamp, never widen a bound, never skip. | No automatic repair, but the operator may fix the cause and re-run. |
| May it be retried as-is? | Yes, but the result is identical (deterministic reject). | Only after the operator removes the cause. |
| May partial output be emitted? | No. | No. |
| Is it a negative-test condition? | Yes — every S failure has a rejecting test (`docs/spec/07-testing-strategy.md#negative-tests`). | No — these are exercised by integration/operational tests, not soundness negatives. |
| Example variants | `FieldWraparound`, `AccumulatorOverflow`, `OutOfDomainLookup`, `TieBreakViolation`, `ModelCommitmentMismatch`, `ProofInvalid`. | `UnsupportedOp`, `Schema`, `ShapeMismatch`, `ResourceExhausted`, `ArtifactIo`. |

The dividing test is simple: *if the underlying claim is true, would this error still occur?* If yes, it is operational (the environment is wrong, not the math). If the error can only occur because the claim is false or the witness is inconsistent with the committed relation, it is a soundness failure and must reject.

---

## 5. Invariants

These are the named invariants this document owns. Each is a hard, testable property. Violation is a defect, not a tuning knob. Cross-referenced by `docs/spec/06-security.md#soundness-requirements` and exercised by `docs/spec/07-testing-strategy.md#negative-tests`.

| Invariant | Statement |
| --- | --- |
| **INV-ERR-01** | Every fallible boundary returns a typed enum from `{ExportError, ManifestError, TraceError, ProveError, VerifyError}`. No public fallible API returns a bare string error or relies on `panic!`/`unwrap` for an expected failure. |
| **INV-ERR-02** | Every error is classified exactly once as soundness (**S**) or operational (**O**), per the tables in [§1](#error-taxonomy) and [§2](#failure-modes). No error is both, and none is unclassified. |
| **INV-ERR-03** | On any soundness failure (F1–F12), the prover emits no `ProofArtifact`. There is no partial, degraded, or best-effort proof. The process exits non-zero. |
| **INV-ERR-04** | `overflow_policy = reject` is enforced end-to-end: a value that would wrap mod p, or an accumulator exceeding its declared bound, causes a hard reject at export, trace build, *and* verification. No silent wrap and no silent saturation ever occur. (RFC-0002) |
| **INV-ERR-05** | The verifier returns `Result<(), VerifyError>` and nothing else. There is no "accept with warning," no numeric tolerance on commitments or ranges, and no degraded-accept path. A rejection is final and deterministic. |
| **INV-ERR-06** | No value is clamped outside a clamp range explicitly declared in the manifest and bound by `quantization_commitment`. Any clamp the manifest does not authorize is a soundness violation, not a recovery. |
| **INV-ERR-07** | Every commitment-binding failure (model, quantization, planner config, public-input digest, serialization hash) rejects: the prover refuses to load a tampered manifest, and the verifier rejects a mismatched commitment. Anything left unbound is treated as prover-mutable and therefore unsound (RFC-0000, RFC-0014). |
| **INV-ERR-08** | The verifier runs its checks in the fixed order of [§3.1](#verifier-rejections) and returns the *first* failing check. Reordering that changes which `VerifyError` is returned for a given artifact is a defect. |
| **INV-ERR-09** | Every soundness failure mode (each **S** row in [§2](#failure-modes)) has at least one corresponding rejecting (negative) test, and no AIR component ships without both an accepting and a rejecting test (RFC-0013). |
| **INV-ERR-10** | An honest prover's pre-commitment self-check (`ConstraintUnsatisfied`, `LookupImbalance`) guarantees that any artifact it *does* emit passes verification under the same build and pinned Stwo revision. If the self-check fires, the prover aborts and emits nothing. |

---

## 6. Open questions

- **OPEN QUESTION (owner: pwm-verifier maintainer):** Should `OutOfDomainLookup` and `RangeViolation` always be distinguishable from the generic `ProofInvalid` at verification time, or are some lookup/range failures only observable as an aggregate LogUp claimed-sum imbalance (hence reported as `ProofInvalid`)? The answer depends on whether per-relation claimed sums are exposed individually by the vendored Stwo verifier path. Resolution path: RFC-0003 (range-check and lookup infrastructure) decides the granularity of the surfaced relation identifiers; until then, treat `RangeViolation`/`AccumulatorOverflow`/`OutOfDomainLookup` as best-effort refinements of `ProofInvalid` and rely on `ProofInvalid` as the guaranteed catch-all.
- **OPEN QUESTION (owner: pwm-prover maintainer):** Whether `ProveError::ConstraintUnsatisfied` and `ProveError::LookupImbalance` should be compiled out (debug-only assertions) in release builds for performance, or always run. Keeping them always-on costs prover time but upholds INV-ERR-10 unconditionally. Resolution path: RFC-0016 (CLI, artifact bundle, and reproducibility) decides the release-build policy; the default until decided is always-on.

---

## 7. References

- Founding analysis: `docs/feasibility-study.md` (§5 overflow=reject and arithmetic semantics; §9.3 range-safety; §9.5 planner soundness; §10.4 verifier flow; §13 risk register).
- Data model and types: `docs/spec/03-data-model.md#bounded-integers`, `docs/spec/03-data-model.md#model-manifest`, `docs/spec/03-data-model.md#public-input`, `docs/spec/03-data-model.md#witness`, `docs/spec/03-data-model.md#tensor-memory-cells`.
- Public API and verifier entry: `docs/spec/02-public-api.md#rust-public-api`, `docs/spec/02-public-api.md#artifact-formats`.
- Security and soundness: `docs/spec/06-security.md#soundness-requirements`, `docs/spec/06-security.md#binding-requirements`.
- Testing: `docs/spec/07-testing-strategy.md#negative-tests`, `docs/spec/07-testing-strategy.md#mutation-tests`.
- RFC-0000 Security model and statement taxonomy: `docs/rfcs/RFC-0000-security-model-and-statement-taxonomy.md`.
- RFC-0001 Model manifest and export pipeline: `docs/rfcs/RFC-0001-model-manifest-and-export-pipeline.md`.
- RFC-0002 Fixed-point arithmetic over M31: `docs/rfcs/RFC-0002-fixed-point-arithmetic-over-m31.md`.
- RFC-0003 Range-check and lookup infrastructure: `docs/rfcs/RFC-0003-range-check-and-lookup-infrastructure.md`.
- RFC-0004 Tensor memory and wiring AIR: `docs/rfcs/RFC-0004-tensor-memory-and-wiring-air.md`.
- RFC-0009 Fixed-candidate planner proof: `docs/rfcs/RFC-0009-fixed-candidate-planner-proof.md`.
- RFC-0013 Testing, fuzzing, and audit strategy: `docs/rfcs/RFC-0013-testing-fuzzing-and-audit-strategy.md`.
- RFC-0014 Canonical serialization and transcript: `docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md`.
- RFC-0015 Third-party vendoring and pinning: `docs/rfcs/RFC-0015-third-party-vendoring-and-pinning.md`.
- RFC-0016 CLI, artifact bundle, and reproducibility: `docs/rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md`.
