# Public API: Rust Surface, CLI, Artifact Formats, Stability Policy

> Status: Normative (v0.1 design freeze). Role: defines the supported public surface of
> ProvableWorldModel — the Rust APIs third parties may build against, the `pwm` CLI command
> shapes, the on-disk artifact formats, and the stability guarantees that bind all of them.
> This document is authoritative for *what is public*; the exact type definitions live in
> [docs/spec/03-data-model.md](03-data-model.md#public-input), and the load-bearing decisions
> are locked in [docs/rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md](../rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md)
> and [docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md](../rfcs/RFC-0014-canonical-serialization-and-transcript.md).

The public surface is deliberately narrow. ProvableWorldModel is a soundness-critical system:
the smaller the verifier-facing API, the smaller the audit. Everything not enumerated here is
internal and may change without notice. The verifier entry point in particular is the most
important contract in the entire project, and its guarantees (§[API invariants](#api-invariants))
are the things an integrator is entitled to rely on.

---

## Table of contents

- [Surface overview and crate visibility](#surface-overview-and-crate-visibility)
- [Rust public API](#rust-public-api)
  - [Re-exported types](#re-exported-types)
  - [Verifier entry point](#verifier-entry-point)
  - [Prover entry points](#prover-entry-points)
  - [Error surface](#error-surface)
- [CLI](#cli)
  - [`pwm export`](#pwm-export)
  - [`pwm manifest verify`](#pwm-manifest-verify)
  - [`pwm prove`](#pwm-prove)
  - [`pwm verify`](#pwm-verify)
  - [Global flags and exit codes](#global-flags-and-exit-codes)
- [Artifact formats](#artifact-formats)
  - [The ProofArtifact bundle](#the-proofartifact-bundle)
  - [The model manifest file](#the-model-manifest-file)
  - [The weights file](#the-weights-file)
  - [The golden-vector file](#the-golden-vector-file)
  - [Canonical serialization](#canonical-serialization)
- [Stability policy](#stability-policy)
- [API invariants](#api-invariants)
- [Failure modes](#failure-modes)
- [Open questions](#open-questions)

---

## Surface overview and crate visibility

The workspace has six crates (see [docs/spec/01-architecture.md#crate-layout](01-architecture.md#crate-layout)).
Only a subset is *public surface*. "Public" means: covered by semver (§[stability policy](#stability-policy)),
documented for external use, and stable across patch and minor releases per the rules below.

| Crate          | Public surface?              | What is exposed                                                                                          | What is internal                                                              |
| -------------- | ---------------------------- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `pwm-core`     | Public (types only)          | `M31`, `QM31`, `BoundedInt`, `Rounding`, `OverflowPolicy`, `Tensor`, `QuantizedWeights`, `TensorCell`, `StatementType`, `PublicInput`, `Witness`, `ArgminWitness`, `RangeWitness`, `LookupWitness`, `Proof`, `ProofArtifact`, manifest types, `relation_id` helpers | Internal transcript implementation details, trace column layout, field arithmetic internals |
| `pwm-verifier` | Public (the primary surface) | `verify()`, `VerifyError`, public-input digest helpers                                                   | Recursive verifier internals, preprocessed-trace recomputation internals      |
| `pwm-prover`   | Public (entry points only)   | `prove_rollout()`, `prove_planning()`, `ProveError`, the `pwm` binary                                    | `trace_builder`, witness assembly, Stwo channel wiring                        |
| `pwm-air`      | Internal                     | nothing                                                                                                  | all AIR components (linear, matmul, attention, rollout, cost, argmin, …)      |
| `pwm-circuits` | Internal                     | nothing                                                                                                  | low-level gates, `stwo-circuits` adapters, witness builder                    |
| `pwm-export`   | Internal + CLI               | the `pwm export` subcommand only; no stable Rust API                                                     | Python/Rust quantization pipeline, manifest writer, parity harness            |

Three properties of this table are normative and enforced by [API invariants](#api-invariants):

1. **`pwm-verifier` does not depend on `pwm-export`, `pwm-air`, `pwm-circuits` builder code, or
   PyTorch.** The verifier is a self-contained arithmetic checker. See
   [INV-API-03](#api-invariants) and [docs/spec/01-architecture.md#module-boundaries](01-architecture.md#module-boundaries).
2. **`pwm-air` and `pwm-circuits` expose no public items.** They are implementation substrate.
   AIR layout is *bound by the proof and the manifest*, not by the API; changing a constraint
   layout that does not change the proven relation is a patch-level change.
3. **`pwm-export` exposes no stable Rust API**, only the `pwm export` CLI. The export pipeline is
   Python-driven (see [decided choice: language split](#stability-policy)); its Rust internals are
   free to churn. The *outputs* of export (manifest, weights, golden vectors) are stable artifacts
   and are versioned independently (§[artifact formats](#artifact-formats)).

---

## Rust public API

All signatures below are reproduced **verbatim** from the canonical type registry in
[docs/spec/03-data-model.md](03-data-model.md#public-input) and authoring contract §6. This
document does not re-define them; it states their *API contracts* — preconditions, postconditions,
errors, and determinism. The data-model document owns the field-by-field schema and invariants.

### Re-exported types

The public types are defined in `pwm-core` and re-exported from `pwm-verifier` and `pwm-prover` so
that an integrator can depend on a single crate.

```rust
// pwm-core: field and fixed-point (see docs/spec/03-data-model.md#bounded-integers)
pub struct M31(u32);            // canonical value in [0, p), p = 2^31 - 1
pub struct QM31([M31; 4]);      // degree-4 "secure field"; challenges/soundness only

pub struct BoundedInt {
    pub value: i64,             // mathematical signed value
    pub lo: i64,                // inclusive lower bound (declared)
    pub hi: i64,                // inclusive upper bound (declared)
}

pub enum Rounding { NearestTiesToEven, TruncateTowardZero }
pub enum OverflowPolicy { Reject }     // V0: reject only; wrapping is unsound

// pwm-core: tensors and weights (see docs/spec/03-data-model.md#tensor-types)
pub struct Tensor {
    pub tensor_id: u32,
    pub shape: Vec<u32>,        // row-major
    pub scale_id: u32,          // index into the manifest scale table
    pub data: Vec<BoundedInt>,  // len == product(shape)
}

pub struct QuantizedWeights {
    pub commitment: [u8; 32],   // matches manifest weights.root
    pub tensors: Vec<Tensor>,   // per-op weight + bias tensors
}

// pwm-core: public input, witness, statement type (see docs/spec/03-data-model.md#public-input)
pub enum StatementType { P0Step, P1Rollout, P2FixedCandidatePlanning,
                         P3Cem, P4PixelToPlan }

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

pub struct Witness {
    pub model_weights: Option<QuantizedWeights>,  // None when public/committed-only
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

pub struct ArgminWitness {
    pub selected_index: u32,
    pub diffs: Vec<BoundedInt>,   // cost_s - selected_cost (>= 0), per candidate
}

// pwm-core: proof and on-disk bundle (see docs/spec/03-data-model.md#public-input
// and RFC-0016 for the bundle layout)
pub struct Proof { /* opaque Stwo proof bytes + metadata; schema-versioned */ }

pub struct ProofArtifact {     // the on-disk bundle
    pub artifact_version: u32,
    pub public_input: PublicInput,
    pub proof: Proof,
    pub claimed_outputs: Option<Vec<Tensor>>,
}
```

The `Proof` type is **opaque on purpose**: it wraps Stwo proof bytes plus a schema version tag.
Integrators must treat it as a blob to be passed to `verify()` or serialized via the bundle. Its
internal layout is not public and changes with the Stwo vendoring revision
(see [docs/rfcs/RFC-0015-third-party-vendoring-and-pinning.md](../rfcs/RFC-0015-third-party-vendoring-and-pinning.md)).
A `Proof` is only meaningful relative to the `PublicInput` it was produced with; the two must
travel together, which is exactly why `ProofArtifact` bundles them.

`RangeWitness` and `LookupWitness` appear in `Witness` and are defined in
[docs/spec/03-data-model.md#witness](03-data-model.md#witness). They are public types only so that
`Witness` is constructible by callers building proofs through the library path; they carry no
behavior.

### Verifier entry point

`pwm-verifier::verify` is the single most important public function in the system. Everything an
integrator trusts reduces to its contract.

```rust
pub enum VerifyError { /* see docs/spec/04-error-model.md#verifier-rejections */ }

/// Verify a proof artifact against the relation it declares.
///
/// PRECONDITIONS (caller's responsibility, but never cause a panic — see postconditions):
///   - `artifact` is a fully-deserialized `ProofArtifact`; the bytes-to-struct step has
///     already happened and is NOT performed here (use `ProofArtifact::from_bundle` for that).
///   - The caller decides, out of band, whether `artifact.public_input.relation_id` is a
///     relation it is willing to accept. `verify` checks that the relation_id is one this
///     build *implements*; it does not encode the caller's policy about which relations are
///     acceptable for the caller's use case.
///
/// POSTCONDITIONS:
///   - Returns `Ok(())` IFF the proof is a valid Circle-STARK proof for the AIR identified by
///     `artifact.public_input.relation_id`, AND every public-input field is consistent with
///     the proof's bound public values (recomputed digest matches), AND all application-level
///     checks pass (selected_index in range, claimed-output commitment matches claimed_outputs
///     if present, tensor shapes consistent with the statement type).
///   - Returns `Err(VerifyError::...)` for every other input. The error variant identifies the
///     first failed check in a fixed, documented order (see
///     docs/spec/04-error-model.md#verifier-rejections).
///   - NEVER panics, aborts, or runs unboundedly on any input, including adversarially crafted
///     ones (INV-API-01, INV-API-02).
///   - Is a pure function of `artifact`: no I/O, no clock, no RNG, no global state, no threads
///     that affect the result, no dependency on environment or filesystem (INV-API-04).
///   - Does not link `pwm-export` or PyTorch (INV-API-03).
///
/// ERRORS: see VerifyError taxonomy in docs/spec/04-error-model.md#verifier-rejections.
///   Representative variants: UnsupportedRelation, MalformedProof, PublicInputDigestMismatch,
///   StarkVerificationFailed, OutputCommitmentMismatch, SelectedIndexOutOfRange,
///   ShapeMismatch, ArtifactVersionUnsupported.
pub fn verify(artifact: &ProofArtifact) -> Result<(), VerifyError>;
```

Two deserialization helpers accompany `verify`. They are part of the public surface because the
boundary between "untrusted bytes" and "typed artifact" is itself security-critical and must be
explicit:

```rust
impl ProofArtifact {
    /// Parse a bundle directory or single-file bundle into a typed artifact.
    /// Performs structural validation only (lengths, version tags, well-formedness).
    /// Does NOT verify the proof. NEVER panics on malformed bytes (INV-API-01).
    pub fn from_bundle(path: &std::path::Path) -> Result<ProofArtifact, VerifyError>;

    /// Serialize the artifact to a bundle at `path` using the canonical scheme
    /// (RFC-0014). Deterministic: same artifact -> same bytes (INV-API-05).
    pub fn to_bundle(&self, path: &std::path::Path) -> Result<(), std::io::Error>;
}
```

`from_bundle` returning `VerifyError` (not a separate parse-error type) is deliberate: from the
integrator's standpoint, "these bytes are not a verifiable artifact" and "this artifact does not
verify" are the same outcome — rejection — and collapsing them keeps the trust decision a single
`Result`. The specific malformed-input variants are enumerated in
[docs/spec/04-error-model.md#verifier-rejections](04-error-model.md#verifier-rejections).

The canonical full-verification pattern an integrator should copy:

```rust
use pwm_verifier::{verify, VerifyError};
use pwm_core::ProofArtifact;

fn accept(path: &std::path::Path, expected_relation: [u8; 32]) -> Result<(), VerifyError> {
    let artifact = ProofArtifact::from_bundle(path)?;
    // The caller's policy check: is this the relation I asked for?
    if artifact.public_input.relation_id != expected_relation {
        return Err(VerifyError::UnsupportedRelation);
    }
    verify(&artifact)
}
```

### Prover entry points

The prover library exposes exactly two statement-specific entry points for V0. P0 (one step) is
exercised through `prove_rollout` with `horizon = 1`; it has no separate entry point in V0 to keep
the surface minimal. P3 (CEM) and P4 (pixel) have no entry points until their milestones
(`Future`; see [docs/spec/00-overview.md#scope-and-statement-tiers](00-overview.md#scope-and-statement-tiers)).

```rust
pub enum ProveError { /* see docs/spec/04-error-model.md#error-taxonomy */ }

/// Build a P1 rollout proof (and, with horizon == 1, a P0 single-step proof).
///
/// PRECONDITIONS:
///   - `manifest` has already been loaded and its canonical-JSON hash equals
///     `public_input.model_commitment` AND `public_input.quantization_commitment`. If not,
///     returns ProveError::CommitmentMismatch; the prover does NOT silently re-commit.
///   - `witness.model_weights`, if `Some`, has `.commitment == manifest.weights.root`.
///   - `public_input.statement_type` is `P1Rollout` (or `P0Step` when horizon == 1).
///   - `witness` is internally consistent with `public_input` (history/action lengths,
///     declared bounds). Inconsistency yields a typed ProveError, never a panic.
///   - `horizon` matches the manifest planner/rollout configuration. The V0 reference
///     configuration targets horizon = 5 (see docs/spec/08-performance-budget.md#scaling).
///
/// POSTCONDITIONS:
///   - On success, the returned `ProofArtifact` satisfies `verify(&artifact) == Ok(())`
///     (INV-API-06: prover/verifier agreement — every artifact the prover emits verifies).
///   - `artifact.public_input == public_input` (the prover does not mutate the public input).
///   - `artifact.artifact_version` is the current bundle version this build emits.
///   - The proof is produced by executing the Rust fixed-point reference inference, building
///     operator/range/lookup traces, and running the vendored Stwo prover with the canonical
///     Fiat-Shamir channel order (RFC-0014). Bit-for-bit reproducible from identical inputs
///     (INV-API-05; RFC-0016 reproducibility contract).
///
/// ERRORS: CommitmentMismatch, WitnessInconsistent, RangeViolation (a value exceeds its
///   declared bound — overflow policy is Reject, never wrap), HorizonMismatch,
///   UnsupportedStatement, ProverInternal (a bug; reported, never a panic). See
///   docs/spec/04-error-model.md#error-taxonomy.
pub fn prove_rollout(
    manifest: &Manifest,
    public_input: &PublicInput,
    witness: &Witness,
    horizon: u32,
) -> Result<ProofArtifact, ProveError>;

/// Build a P2 fixed-candidate planning proof: rollout of ALL candidates, MSE cost of every
/// final latent against the goal, and argmin with deterministic tie-break. This is the V0
/// headline statement.
///
/// PRECONDITIONS:
///   - All of `prove_rollout`'s manifest/weights/commitment preconditions.
///   - `public_input.statement_type == P2FixedCandidatePlanning`.
///   - `public_input.goal_latent_public` or `.goal_latent_commitment` is `Some` (a goal is
///     required to compute cost); `witness.goal_latent` is `Some`.
///   - `witness.candidate_actions` is `Some` and describes S candidate sequences; `public_input`
///     binds them via `candidate_actions_public` or `candidate_actions_commitment`.
///   - `public_input.selected_index` and `public_input.selected_cost` MAY be pre-filled by the
///     caller; if so the prover proves they are correct, and if they are wrong returns
///     ProveError::SelectionInconsistent rather than silently "fixing" them. If `None`, the
///     prover computes the deterministic argmin and fills them in the returned artifact.
///
/// POSTCONDITIONS:
///   - Proves, for every candidate s in 0..S: trajectory_s = Rollout_Q(history, actions_s) and
///     cost_s = MSE_Q(final_s, goal). Proves selected_cost == cost[selected_index] and
///     selected_cost <= cost_s for all s, with smallest-index tie-break enforced via
///     cost_s - selected_cost - 1 >= 0 range checks for all s < selected_index (see
///     docs/spec/06-security.md#soundness-requirements). PARTIAL proofs are unsound and are not
///     produced (RFC-0009).
///   - On success, `verify(&artifact) == Ok(())` (INV-API-06).
///   - Bit-for-bit reproducible (INV-API-05).
///
/// ERRORS: as `prove_rollout`, plus SelectionInconsistent, GoalMissing, CandidatesMissing.
pub fn prove_planning(
    manifest: &Manifest,
    public_input: &PublicInput,
    witness: &Witness,
) -> Result<ProofArtifact, ProveError>;
```

`Manifest` is the parsed model manifest type defined in
[docs/spec/03-data-model.md#model-manifest](03-data-model.md#model-manifest). It is public because
both prover entry points take it by reference; it is constructed via `Manifest::load` (loads and
validates a manifest file, computing its commitment), which is the only manifest constructor on the
public surface.

```rust
impl Manifest {
    /// Load and structurally validate a manifest file, computing model_commitment and
    /// quantization_commitment from its canonical-JSON hash (RFC-0014). NEVER panics on
    /// malformed input; returns ProveError::ManifestInvalid (see
    /// docs/spec/04-error-model.md#error-taxonomy).
    pub fn load(path: &std::path::Path) -> Result<Manifest, ProveError>;

    pub fn model_commitment(&self) -> [u8; 32];
    pub fn quantization_commitment(&self) -> [u8; 32];
}
```

### Error surface

The public surface exposes exactly three error enums. Their variants are owned by the error-model
document; this document references them and never re-enumerates them, so the two cannot drift.

| Error enum    | Returned by                                              | Variant taxonomy owner                                                                 |
| ------------- | -------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `VerifyError` | `verify`, `ProofArtifact::from_bundle`                   | [docs/spec/04-error-model.md#verifier-rejections](04-error-model.md#verifier-rejections) |
| `ProveError`  | `prove_rollout`, `prove_planning`, `Manifest::load`      | [docs/spec/04-error-model.md#error-taxonomy](04-error-model.md#error-taxonomy)          |
| `ExportError` | export pipeline (CLI-only; not a stable Rust API)        | [docs/spec/04-error-model.md#error-taxonomy](04-error-model.md#error-taxonomy)          |

All three implement `std::error::Error` and `Display`. None contains a private witness, raw weight
value, or unredacted intermediate in its `Display` output; redaction rules are normative in
[docs/spec/05-observability.md#redaction](05-observability.md#redaction). This is part of the
not-leaking contract ([INV-API-07](#api-invariants)).

---

## CLI

A single binary, `pwm`, ships from `pwm-prover` (it links the prover; the verifier subcommand also
links `pwm-verifier`). The CLI is the supported automation surface and is covered by the same
stability policy as the Rust API. Command shapes, the bundle layout, and the reproducibility
contract are locked in
[docs/rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md](../rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md).

The four subcommands map onto the pipeline stages: `export` (Python-backed; produces artifacts),
`manifest verify` (checks an artifact against itself), `prove` (produces a proof bundle), `verify`
(checks a proof bundle). Each subcommand reads typed inputs and writes typed outputs; none reads
hidden state from the environment beyond explicitly documented variables (§[global flags](#global-flags-and-exit-codes)).

### `pwm export`

```text
USAGE:
    pwm export --checkpoint <FILE> --config <FILE> --quant <FILE> --out <DIR> [OPTIONS]

DESCRIPTION:
    Run the export/quantization pipeline. Consumes a LeWorldModel PyTorch checkpoint plus its
    config and a quantization spec; emits a manifest, a weights file, and golden vectors. This is
    the only subcommand that invokes Python (see "language split" in #stability-policy). le-wm is
    MIT-licensed and is NOT vendored; the pipeline consumes a checkpoint and config only
    (RFC-0001; verified against upstream at 2026-06-03).

REQUIRED:
    --checkpoint <FILE>   PyTorch checkpoint (.pt/.pth) of the LeWorldModel predictor.
    --config <FILE>       Architecture config for the checkpoint (latent_dim, depth, heads, ...).
    --quant <FILE>        Quantization spec: per-tensor dtypes, scales, rounding, clamp policy.
    --out <DIR>           Output directory for the artifact set (created if absent).

OPTIONS:
    --relation-id <ID>    relation_id to stamp (e.g. pwm.lewm.fixed_candidate_planning.v1).
                          Default: derived from --quant + --config statement scope.
    --rounding <MODE>     nearest_ties_to_even (default) | truncate_toward_zero. Overrides the
                          manifest default only if the quant spec does not pin it; exactly one
                          active rounding mode per manifest (decided choice).
    --weights-visibility <V>   public (default) | private_committed.
    --check-parity        After export, run Python-vs-Rust fixed-point parity on golden vectors
                          and fail if any vector differs (RFC-0001 acceptance criterion).

OUTPUTS (into --out, see #artifact-formats):
    manifest.yaml         the canonical model manifest.
    weights.pwmw          the quantized weights file.
    golden.pwmg           golden test vectors (Python fixed-point reference outputs).

EXIT CODES: see #global-flags-and-exit-codes. 3 on parity mismatch under --check-parity.
```

### `pwm manifest verify`

```text
USAGE:
    pwm manifest verify --manifest <FILE> [--weights <FILE>] [--golden <FILE>]

DESCRIPTION:
    Validate an exported artifact set WITHOUT producing or checking a proof. Confirms the manifest
    is structurally valid, recomputes model_commitment/quantization_commitment from the manifest's
    canonical-JSON hash, and (if --weights is given) confirms weights.root matches the embedded
    weight commitment. With --golden, replays the Rust fixed-point reference against the golden
    vectors and fails on any bit-level difference (this is the export parity gate, runnable
    independently of `pwm export --check-parity`).

REQUIRED:
    --manifest <FILE>     manifest.yaml to validate.

OPTIONS:
    --weights <FILE>      weights.pwmw; if present, verifies weights.commitment == manifest root.
    --golden <FILE>       golden.pwmg; if present, runs Rust fixed-point parity (no Python).
    --print-commitments   Print model_commitment and quantization_commitment as hex and exit 0
                          if the manifest is valid.

OUTPUTS: structured report on stdout (text or, with --json, machine-readable). No files written.

EXIT CODES: 0 valid; 2 structurally invalid manifest; 3 commitment or parity mismatch.
```

### `pwm prove`

```text
USAGE:
    pwm prove --statement <S> --manifest <FILE> --weights <FILE> --input <FILE> --out <DIR>
              [OPTIONS]

DESCRIPTION:
    Produce a proof bundle. Loads and re-validates the manifest (commitment check), loads weights,
    builds the witness from --input, executes the Rust fixed-point reference, builds traces, and
    runs the vendored Stwo prover. The emitted bundle satisfies `pwm verify` by construction
    (INV-API-06). Bit-for-bit reproducible from identical inputs (RFC-0016).

REQUIRED:
    --statement <S>       p0 | p1 | p2. (p3/p4 are Future and rejected with exit 2 in V0.)
    --manifest <FILE>     manifest.yaml. Its commitment MUST match the proof's public input.
    --weights <FILE>      weights.pwmw.
    --input <FILE>        public+private inputs: latent history, actions/candidates, goal (for p2).
    --out <DIR>           Output directory for the ProofArtifact bundle.

OPTIONS:
    --horizon <N>         Rollout horizon for p1. Must match the manifest. V0 reference targets 5.
    --selected-index <I>  For p2: assert the prover proves this index is the argmin. If omitted,
                          the prover computes the deterministic argmin and records it.
    --no-claimed-outputs  Omit claimed_outputs from the bundle (commitment still binds them).
    --threads <N>         Prover worker threads. Does NOT affect proof bytes (INV-API-05).

OUTPUTS (into --out, see #the-proofartifact-bundle):
    public_input.bin, proof.bin, claimed_outputs.bin (unless --no-claimed-outputs),
    artifact.json (the bundle index with artifact_version).

EXIT CODES: 0 success; 2 bad arguments / unsupported statement; 3 commitment mismatch;
    4 witness inconsistent or range violation (overflow=reject); 70 prover internal error.
```

### `pwm verify`

```text
USAGE:
    pwm verify --artifact <DIR|FILE> [--expect-relation <ID>] [OPTIONS]

DESCRIPTION:
    Verify a proof bundle. Deserializes the bundle (structural validation), then runs the pure
    verifier (verify()). Does NOT load the manifest, weights, Python, or any model artifact: the
    proof and public input are self-contained (INV-API-03, INV-API-04). The verifier checks the
    relation it implements; --expect-relation adds the caller's policy check that the bundle's
    relation_id is the one the caller asked for.

REQUIRED:
    --artifact <DIR|FILE> The ProofArtifact bundle (directory) or single-file bundle.

OPTIONS:
    --expect-relation <ID>  Reject (exit 5) unless public_input.relation_id equals this exactly.
    --json                  Emit a machine-readable accept/reject report on stdout.

OUTPUTS: on accept, exit 0 and (with --json) {"verified": true, "relation_id": "..."}; on reject,
    nonzero exit and the VerifyError variant name (never a private witness — INV-API-07).

EXIT CODES: 0 verified; 1 verification failed (proof invalid for its relation); 2 bad arguments;
    5 relation mismatch under --expect-relation; 6 malformed/unsupported bundle (from_bundle
    failed, including artifact_version unsupported).
```

### Global flags and exit codes

```text
GLOBAL FLAGS (accepted by every subcommand):
    -v, --verbose         Increase log verbosity. Logging follows docs/spec/05-observability.md;
                          private witnesses and weights are redacted at every level (INV-API-07).
    -q, --quiet           Suppress non-error output.
    --json                Machine-readable output where the subcommand supports it.
    --version             Print pwm version, artifact_version, manifest_version supported, and the
                          vendored Stwo REVISION (third_party/stwo/REVISION; RFC-0015).
    -h, --help            Help for the binary or a subcommand.

ENVIRONMENT (the only env vars that influence behavior; none affect proof or verify RESULT):
    PWM_LOG               Log filter (e.g. info, debug). Observability only.
    PWM_THREADS           Default worker thread count for proving. Does not affect proof bytes.
```

Exit codes are uniform across subcommands so scripts can branch reliably:

| Code | Meaning                                                                                  |
| ---- | ---------------------------------------------------------------------------------------- |
| 0    | Success (exported / valid / proved / verified).                                          |
| 1    | Verification failed: the proof is invalid for its declared relation (`pwm verify` only). |
| 2    | Usage error: bad arguments, unknown subcommand, unsupported statement in V0.             |
| 3    | Commitment or parity mismatch (manifest/weights/golden inconsistency).                   |
| 4    | Witness inconsistent or a range/overflow violation during proving (overflow=reject).     |
| 5    | Relation mismatch under `--expect-relation` (`pwm verify`).                              |
| 6    | Malformed or unsupported bundle / `artifact_version` (`from_bundle` failed).             |
| 70   | Internal error (a bug). Reported, never a panic-abort; see #failure-modes.               |

Exit-code stability is part of the CLI contract under [stability policy](#stability-policy): the
meaning of a code is stable within a major version; new codes may be added only at the high end and
only for previously-unreachable conditions.

---

## Artifact formats

Four on-disk artifacts cross the trust boundary. Their formats are versioned independently because
they evolve on different schedules: the bundle (`artifact_version`), the manifest
(`manifest_version`), the weights file, and the golden-vector file. The exact byte-canonical
serialization scheme for all of them is locked in
[docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md](../rfcs/RFC-0014-canonical-serialization-and-transcript.md);
this section specifies the *layout and versioning contract*, and defers the byte grammar to
RFC-0014 rather than duplicating it.

### The ProofArtifact bundle

A bundle is a directory (the canonical form) holding the serialized `ProofArtifact`. A single-file
form (a concatenation with a fixed header) is also accepted by `from_bundle` for transport; both
deserialize to the identical typed `ProofArtifact`.

```text
<bundle>/                         (directory form)
  artifact.json        index: { "artifact_version": <u32>, "files": [ ... in fixed order ... ],
                                "relation_id": "<hex>", "statement_type": "<P0..P4>" }
  public_input.bin     canonical serialization of PublicInput (RFC-0014)
  proof.bin            opaque Stwo proof bytes + schema tag (the `Proof` blob)
  claimed_outputs.bin  canonical serialization of Option<Vec<Tensor>> (present unless omitted)
```

Layout and ordering contract:

| Field             | Type   | Contract                                                                                                    |
| ----------------- | ------ | ----------------------------------------------------------------------------------------------------------- |
| `artifact_version`| `u32`  | The bundle schema version. `verify`/`from_bundle` reject an unknown version with `ArtifactVersionUnsupported` (exit 6). V0 emits `artifact_version = 1`. |
| `files` order     | fixed  | `public_input.bin`, `proof.bin`, `claimed_outputs.bin`. The order is canonical so the bundle index hashes deterministically (INV-API-05). |
| `proof.bin`       | opaque | Tagged with the `Proof` schema version, which tracks the vendored Stwo revision (RFC-0015). Not independently public. |
| `claimed_outputs` | option | If present, MUST hash to `public_input.claimed_output_commitment` (verified by `verify`); if absent, the commitment still binds the outputs and the verifier accepts an outputs-free bundle. |

The bundle is self-contained: it carries everything `verify` needs and nothing it does not. In
particular it does **not** embed the manifest or weights — `verify` does not consult them
(INV-API-03). An integrator who wants to associate a bundle with a specific model does so by
checking `public_input.model_commitment` against a manifest they hold, *outside* `verify`.

### The model manifest file

`manifest.yaml`, written by `pwm export`. YAML on disk; bound by a canonical-JSON hash, not by its
YAML bytes, so reformatting does not change the commitment (RFC-0014). The full schema, field
semantics, and `manifest_version` rules are owned by
[docs/spec/03-data-model.md#model-manifest](03-data-model.md#model-manifest) and locked by
[docs/rfcs/RFC-0001-model-manifest-and-export-pipeline.md](../rfcs/RFC-0001-model-manifest-and-export-pipeline.md).
The public-API-relevant facts:

- Top-level keys (see the data-model doc for the full schema): `manifest_version, model_family,
  relation_id, field{...}, quantization{...}, architecture{...}, weights{visibility,
  commitment_scheme, root}, ops[]{...}, serialization{canonical_json_hash}`.
- The manifest's canonical-JSON hash yields **two** public commitments consumed by the API:
  `model_commitment` (binds architecture, weights root, shapes, planner config, relation version,
  serialization version) and `quantization_commitment` (binds scales, rounding, overflow/clamp
  policy, activation/normalization approximation tables). Anything unbound is mutable by the prover
  and therefore unsound (see [docs/spec/06-security.md#binding-requirements](06-security.md#binding-requirements)).
- Exactly one active rounding mode per manifest (decided choice): `nearest_ties_to_even` by default,
  or `truncate_toward_zero` if the manifest overrides it. The same mode is enforced bit-for-bit by
  the Python reference, the Rust reference, and the AIR.
- The V0 reference configuration targets `latent_dim = 192`, `history_size = 3`, predictor
  `depth = 6`, `heads = 16`, `dim_head = 64`, `mlp_dim = 2048`, ~15M parameters (verified against
  upstream at 2026-06-03). Activation functions are not uniform: the predictor FFN/MLP uses GELU,
  while AdaLN modulation and the action Embedder use SiLU — the manifest records this per module,
  and the API treats activation identity as part of the bound model.

### The weights file

`weights.pwmw`, written by `pwm export`. A canonical binary container of quantized weight and bias
tensors plus a 32-byte commitment.

| Element       | Type                     | Contract                                                                                          |
| ------------- | ------------------------ | ------------------------------------------------------------------------------------------------- |
| header        | version tag + tensor count | Structural version of the weights container; independent of `manifest_version` and `artifact_version`. |
| `commitment`  | `[u8; 32]`               | MUST equal `manifest.weights.root`. `pwm manifest verify --weights` checks this (exit 3 on mismatch). It also equals `QuantizedWeights::commitment` when loaded into the typed form. |
| tensors       | `Vec<Tensor>`            | Per-op weight and bias tensors, each `BoundedInt`-valued with a declared range and `scale_id`. Deserializing maps directly to `QuantizedWeights` (§[re-exported types](#re-exported-types)). |

Visibility (`public` vs `private_committed`, from the manifest) governs whether weight values
appear in the public input. In `private_committed` mode the weights file is a private witness input
to the prover and never travels in the bundle; only the 32-byte commitment is public. The API does
not itself provide weight hiding beyond commitment binding — full hiding (zero-knowledge) is a
`Future` mode gated on a hiding audit and is never claimed for V0 (decided choice; see
[docs/spec/06-security.md#privacy-and-zk](06-security.md#privacy-and-zk)).

### The golden-vector file

`golden.pwmg`, written by `pwm export`. The Python fixed-point reference's input/output pairs for
the exported model, used by `pwm manifest verify --golden` and by the CI parity gate to prove the
Rust reference matches the Python reference bit-for-bit (RFC-0001 acceptance criterion). It is a
**test artifact**, not consumed by `prove` or `verify`, but it ships with the artifact set because
it is the evidence that the two reference implementations agree.

| Element        | Contract                                                                                              |
| -------------- | ----------------------------------------------------------------------------------------------------- |
| header         | version tag + bound `relation_id` (golden vectors are valid only for the manifest they were exported with). |
| vectors        | a list of `{ inputs, expected_outputs }` where outputs are the Python fixed-point reference results. |
| determinism    | regenerating from the same checkpoint + config + quant spec yields a byte-identical file (RFC-0001). |

### Canonical serialization

All four artifacts use a single byte-canonical serialization scheme so that "same inputs produce
the same bytes" is a property and not an aspiration. The scheme — field encodings, integer
endianness, length prefixes, map-key ordering, the canonical-JSON form used for the manifest hash,
and the public-input digest — is specified in full in
[docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md](../rfcs/RFC-0014-canonical-serialization-and-transcript.md).
This document does **not** restate the byte grammar; it asserts the contract that follows from it:

- Serialization is total and deterministic: a given typed value serializes to exactly one byte
  string, and that string deserializes back to an equal value (round-trip), or `from_bundle` /
  `Manifest::load` rejects it with a typed error.
- The public-input digest is computed from the canonical `PublicInput` bytes and is what the proof
  binds; `verify` recomputes it and compares (`PublicInputDigestMismatch` on failure). This is the
  hinge of [INV-API-08](#api-invariants).
- The Fiat-Shamir transcript channel order shared by prover and verifier is part of RFC-0014, not
  the public Rust API; it is internal to `prove_*`/`verify` and is mentioned here only because its
  stability is what makes proofs portable across builds at the same relation_id.

---

## Stability policy

The public surface is the union of: the re-exported `pwm-core` types, `pwm-verifier::{verify,
VerifyError}` and `ProofArtifact::{from_bundle, to_bundle}`, `pwm-prover::{prove_rollout,
prove_planning, ProveError, Manifest::{load, model_commitment, quantization_commitment}}`, the
`pwm` CLI command shapes and exit codes, and the four artifact formats. Everything else is internal.

This policy is the API-facing view of the project-wide rules in
[docs/spec/09-release-and-versioning.md](09-release-and-versioning.md#semver), which is
authoritative for semver, MSRV, deprecation windows, and changelog discipline. The two must not
disagree; where this document states a rule, it is restating 09's rule for the API surface.

What semver covers (a breaking change requires a **major** bump):

| Surface                         | Breaking change examples                                                                 |
| ------------------------------- | ---------------------------------------------------------------------------------------- |
| Public Rust types & signatures  | Removing/renaming a field of `PublicInput`; changing a function signature; removing a `VerifyError` variant that callers match on (variants are `#[non_exhaustive]`, so *adding* one is not breaking). |
| `verify` semantics              | Any change that causes a previously-accepted artifact to be rejected, or vice versa, at the same `relation_id`. (This is the strongest guarantee in the system — see INV-API-08/09.) |
| CLI command shapes & exit codes | Removing a subcommand or flag; changing an exit code's meaning; making an optional flag required. |
| Artifact formats                | A bundle/manifest/weights/golden change that an in-range reader cannot parse.            |

What is explicitly **not** covered by semver (may change in any release, including patch):

| Internal surface                                            | Why it can churn                                                              |
| ----------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `pwm-air`, `pwm-circuits` items (none are public)           | AIR layout is bound by the proof + manifest, not the API; relayout that does not change the proven relation is invisible to callers. |
| `pwm-prover` internals (`trace_builder`, witness assembly)  | Implementation of `prove_*`; only the entry-point contracts are stable.       |
| `pwm-export` Rust internals                                 | Python-driven pipeline; only the `pwm export` CLI and its output artifacts are stable. |
| `Proof` internal bytes                                      | Tracks the vendored Stwo revision (RFC-0015); opaque to callers.             |
| Log/trace formats, span names, metric names                | Observability surface; governed by [docs/spec/05-observability.md](05-observability.md#logging), evolves freely. |

Two surfaces have stability rules **stronger** than semver, because they are soundness-critical:

1. **`relation_id` is immutable.** A proof is valid only for its exact `relation_id`
   (`pwm.lewm.<statement>.v<N>`). Any semantic change to a relation — a different rounding rule, a
   changed approximation, a reshaped manifest binding — mints a **new** `relation_id`; it never
   silently changes the meaning of an existing one. This is locked in
   [docs/rfcs/RFC-0000-security-model-and-statement-taxonomy.md](../rfcs/RFC-0000-security-model-and-statement-taxonomy.md)
   and detailed in [docs/spec/09-release-and-versioning.md#relation-versioning](09-release-and-versioning.md#relation-versioning).
2. **Artifact format versions never reuse a number with new meaning.** `artifact_version`,
   `manifest_version`, and the weights/golden header versions are monotonic; a reader that supports
   version `k` may always assume `k` means exactly what it meant when `k` shipped.

Stability begins at the first tagged release. Pre-1.0 (`v0.x`) the surface is explicitly unstable
per [docs/spec/09-release-and-versioning.md#semver](09-release-and-versioning.md#semver): minor
bumps may break it, and the V0 milestone exists precisely to freeze it. The CLI and bundle shapes
are locked at design time by
[docs/rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md](../rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md)
so that the V1.0 freeze ratifies an already-stable shape rather than inventing one.

---

## API invariants

Each invariant is named, testable, and enforced by tests in
[docs/spec/07-testing-strategy.md](07-testing-strategy.md#negative-tests). They are the properties
an integrator is entitled to rely on.

| ID         | Invariant                                                                                                                                                   |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| INV-API-01 | `verify` and `ProofArtifact::from_bundle` NEVER panic, abort, or trigger undefined behavior on ANY input, including adversarially crafted bytes. Malformed input yields `Err`. |
| INV-API-02 | `verify` terminates in time bounded by a polynomial in the artifact size; there is no input that makes it loop or allocate unboundedly.                       |
| INV-API-03 | `verify` (and the `pwm verify` path) does NOT link or invoke `pwm-export`, PyTorch, the manifest loader, or the weights loader. The proof and public input are self-contained. |
| INV-API-04 | `verify` is a pure, deterministic function of its `&ProofArtifact` argument: no I/O, clock, RNG, environment, filesystem, or thread-count dependence affects the result.       |
| INV-API-05 | Proving is bit-for-bit reproducible: identical (`manifest`, `public_input`, `witness`) inputs produce a byte-identical `ProofArtifact`, regardless of `--threads`/`PWM_THREADS` (RFC-0016). |
| INV-API-06 | Prover/verifier agreement: every `ProofArtifact` returned `Ok` by `prove_rollout`/`prove_planning` satisfies `verify(&artifact) == Ok(())`.                   |
| INV-API-07 | No public error `Display`, log line, or CLI reject message exposes a private witness value, raw weight, or unredacted intermediate (redaction per docs/spec/05-observability.md#redaction). |
| INV-API-08 | `verify` accepts IFF the proof is valid for `public_input.relation_id` AND the recomputed public-input digest matches the proof's bound public values. Binding follows the manifest commitments (docs/spec/06-security.md#binding-requirements). |
| INV-API-09 | `verify` rejects a proof submitted under the wrong `StatementType`/`relation_id` (e.g. a P0 proof presented as P1), an altered `model_commitment`, `quantization_commitment`, or `planner_config_commitment`. (RFC-0000 acceptance criteria.) |
| INV-API-10 | `prove_*` never produces a partial planning proof: P2 proves ALL candidate rollouts, ALL costs, the argmin, and the tie-break, or it returns `Err` (RFC-0009; docs/spec/06-security.md#soundness-requirements). |

---

## Failure modes

Every public entry point's failure modes and the system's response. Verifier rejection variants are
owned by [docs/spec/04-error-model.md#verifier-rejections](04-error-model.md#verifier-rejections);
this table maps API-surface failures to the response a caller observes.

| Failure mode                                                  | Entry point                | System response                                                                                  |
| ------------------------------------------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------ |
| Malformed bundle bytes (truncated, wrong magic, bad lengths)  | `from_bundle` / `pwm verify` | `Err(VerifyError::MalformedProof)` / exit 6. No panic (INV-API-01).                              |
| `artifact_version` unknown to this build                      | `from_bundle` / `pwm verify` | `Err(VerifyError::ArtifactVersionUnsupported)` / exit 6.                                          |
| `relation_id` not implemented by this build                   | `verify` / `pwm verify`    | `Err(VerifyError::UnsupportedRelation)` / exit 1.                                                 |
| `relation_id` valid but not the one the caller asked for      | `pwm verify --expect-relation` | exit 5. (The library path leaves this policy check to the caller, per §verifier entry point.)   |
| Proof cryptographically invalid for its relation              | `verify` / `pwm verify`    | `Err(VerifyError::StarkVerificationFailed)` / exit 1.                                             |
| Public-input digest does not match the proof's bound values   | `verify`                   | `Err(VerifyError::PublicInputDigestMismatch)` / exit 1. (INV-API-08.)                            |
| `claimed_outputs` present but do not hash to the commitment   | `verify`                   | `Err(VerifyError::OutputCommitmentMismatch)` / exit 1.                                            |
| `selected_index` out of range / wrong statement shape         | `verify`                   | `Err(VerifyError::SelectedIndexOutOfRange)` or `ShapeMismatch` / exit 1.                          |
| Manifest commitment ≠ public input commitment (prove)         | `prove_*` / `pwm prove`    | `Err(ProveError::CommitmentMismatch)` / exit 3. The prover never silently re-commits.            |
| Weights commitment ≠ manifest root                            | `prove_*` / `pwm manifest verify` | `Err(ProveError::CommitmentMismatch)` / exit 3.                                                   |
| Witness inconsistent with public input (lengths, bounds)      | `prove_*` / `pwm prove`    | `Err(ProveError::WitnessInconsistent)` / exit 4.                                                  |
| A value exceeds its declared bound during proving             | `prove_*` / `pwm prove`    | `Err(ProveError::RangeViolation)` / exit 4. Overflow policy is `Reject`; the field never wraps silently. |
| Caller-supplied `selected_index`/`selected_cost` are wrong    | `prove_planning` / `pwm prove --selected-index` | `Err(ProveError::SelectionInconsistent)` / exit 4. The prover proves the claim or rejects it; it does not "fix" a wrong selection. |
| Unsupported statement in V0 (`p3`, `p4`)                      | `pwm prove --statement`    | exit 2 (`UnsupportedStatement`). No P3/P4 entry points exist in V0.                              |
| Malformed manifest                                            | `Manifest::load` / `pwm manifest verify` | `Err(ProveError::ManifestInvalid)` / exit 2.                                                      |
| Golden-vector parity mismatch (Rust vs Python)                | `pwm manifest verify --golden` / `pwm export --check-parity` | exit 3. Surfaces a reference-implementation divergence; blocks release (RFC-0001).               |
| Internal prover bug                                           | `prove_*` / `pwm prove`    | `Err(ProveError::ProverInternal)` / exit 70. Reported as a typed error, never an uncontrolled panic-abort across the FFI/CLI boundary. |

---

## Open questions

- OPEN QUESTION (owner: `area:prover` maintainer; resolution: RFC-0016 before V0.2 design freeze):
  whether the canonical bundle form is the directory layout or the single-file layout, and whether
  both remain accepted long-term or one is deprecated. Both are specified here as accepted; the
  default emitted by `pwm prove` is the directory form pending RFC-0016 ratification.
- OPEN QUESTION (owner: `area:core` maintainer; resolution: RFC-0014): whether the public-input
  digest and the bundle index hash use the same hash function as the manifest's canonical-JSON hash,
  or whether the manifest hash may differ (e.g. to match an upstream Stwo channel backend such as
  Blake2s/Keccak256/Poseidon252, verified against upstream at 2026-06-03). The API contract holds
  regardless; only the concrete function is pending RFC-0014.
- OPEN QUESTION (owner: `area:verifier` maintainer; resolution: milestone v1.0): whether a stable C
  ABI / FFI wrapper around `verify` is part of the v1.0 public surface or deferred to a later
  release. Until decided, the only supported verifier entry points are the Rust `verify` and the
  `pwm verify` CLI.
