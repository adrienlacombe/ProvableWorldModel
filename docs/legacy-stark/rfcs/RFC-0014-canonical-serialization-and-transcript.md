# RFC-0014: Canonical serialization, public-input binding, and Fiat-Shamir transcript

- Status: Accepted
- Authors: ProvableWorldModel maintainers
- Created: 2026-06-03
- Target milestone: v0.1

## Summary

This RFC freezes the byte-level foundation on which every ProvableWorldModel
soundness claim rests: (1) a single canonical byte serialization for the model
manifest, the `PublicInput`, the witness-side commitable artifacts, and the
on-disk `ProofArtifact`; (2) a deterministic public-input digest derived from
that serialization; (3) the V0 commitment scheme used for all 32-byte
commitments (`model_commitment`, `quantization_commitment`,
`planner_config_commitment`, the manifest `weights.root`, and all
`*_commitment` fields of `PublicInput`); and (4) the exact Fiat-Shamir channel
ordering that prover and verifier must replay bit-for-bit. A STARK is sound only
if prover and verifier absorb the same bytes in the same order into the same
channel; any divergence either breaks soundness (the prover can grind a
challenge) or breaks completeness (an honest proof fails to verify). This RFC
makes that ordering normative and testable. V0 uses the vendored Stwo Blake2s
channel and Blake2s-based commitments; Poseidon252 is deferred to recursion
(`docs/rfcs/RFC-0012-recursive-aggregated-verification.md`).

## Motivation

The founding analysis at `docs/feasibility-study.md` specifies the prover flow
(§10.3) and verifier flow (§10.4) at the level of "commit traces to Stwo
channel" and "derive challenges in canonical order", and §9.1 enumerates the
binding requirements, but it never pins the concrete bytes, the digest
function, or the channel absorption order. That gap is the single most
soundness-critical unspecified item in the corpus: it underlies every
`*_commitment` in `docs/spec/03-data-model.md#public-input`, every binding
requirement in `docs/spec/06-security.md#binding-requirements`, and the
reproducibility contract in
`docs/rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md`. RFC-0001
(`docs/rfcs/RFC-0001-model-manifest-and-export-pipeline.md`) requires
"byte-identical re-export" but defers the actual serialization to this RFC.

Concrete scenarios this RFC must make impossible:

- A prover serializes the manifest with map keys in a different order than the
  verifier expects, producing a different `model_commitment`, and silently
  proves a different model than the one the verifier believes was committed.
- A prover absorbs the public input into the channel after deriving the first
  challenge instead of before, gaining one degree of freedom to grind a FRI
  query challenge.
- A floating-point or platform-dependent number (NaN payload, `-0.0`,
  endianness) leaks into a hashed structure, so the same logical artifact hashes
  differently on two machines, breaking the reproducibility contract in
  `docs/spec/08-performance-budget.md#profiling` differential runs.
- Two distinct logical artifacts share a hash preimage because a length-prefix
  was omitted and concatenation is ambiguous (`"ab" || "c"` vs `"a" || "bc"`).

## Goals

- Define one canonical serialization (`canonical_bytes`) for manifest,
  `PublicInput`, commitable tensors, and `ProofArtifact`, deterministic across
  platforms, with no floating point and no implementation-defined ordering.
- Define the public-input digest `pid = Blake2s256(domain_tag || canonical_bytes(PublicInput))`
  and make it the value the verifier recomputes and the AIR binds.
- Fix the V0 commitment primitive (Blake2s-256, 32-byte output) for all scalar
  commitments and the Merkle scheme for `weights.root`, and state precisely how
  each `*_commitment` field is computed.
- Fix the exact Fiat-Shamir channel: the vendored Stwo `Blake2sChannel`, with a
  fully enumerated absorption order shared bit-for-bit by prover and verifier.
- Name every invariant and enumerate every failure mode with its system
  response, cross-referencing the error model.

## Non-Goals

- Zero-knowledge / hiding of witness bytes. V0 is a succinct validity proof
  (`docs/spec/06-security.md#privacy-and-zk`); this RFC defines serialization of
  public and commitable data, not masking. Hiding is deferred.
- The internal FRI/PCS commitment layout of the proof itself. That is the
  vendored Stwo prover's responsibility
  (`docs/rfcs/RFC-0015-third-party-vendoring-and-pinning.md`); this RFC binds
  the inputs/outputs of that prover, not its internals.
- The manifest field semantics (scales, op graph). Those are owned by
  `docs/spec/03-data-model.md#model-manifest` and RFC-0001; this RFC defines
  only how the manifest is reduced to canonical bytes and a commitment.
- Poseidon252 commitments and recursive transcript composition (deferred to
  `docs/rfcs/RFC-0012-recursive-aggregated-verification.md`).

## Proposed Design

### 1. Canonical serialization (`canonical_bytes`)

Serialization is a total function from a typed value to a byte string. It is
defined structurally; there is exactly one valid encoding per value, and
decoding rejects any non-canonical encoding (decode-then-reencode must be the
identity on bytes). The encoding is a length-prefixed, fixed-endian, tag-free
binary form. We do NOT hash YAML or JSON text directly; text formats permit
whitespace, key-order, and number-formatting freedom that destroys
determinism. Instead the manifest YAML is parsed into the typed
`ManifestModel`, validated, and re-encoded with `canonical_bytes`. The YAML on
disk additionally carries `serialization.canonical_json_hash` for human
auditing only; the binding value is `canonical_bytes`, not the text.

Primitive encodings (`pwm-core/src/serialize.rs`):

| Type | Encoding | Notes |
| --- | --- | --- |
| `u8`/`u16`/`u32`/`u64` | little-endian fixed width | no varint; fixed width removes ambiguity |
| `i64` (e.g. `BoundedInt.value`) | two's-complement little-endian, 8 bytes | mathematical signed value, NOT field-reduced |
| `bool` | `0x00` / `0x01` | any other byte rejected on decode |
| `M31` | `u32` little-endian of the canonical residue in `[0, p)` | non-canonical residue (`>= p`) rejected |
| `[u8; 32]` (commitment) | 32 raw bytes | |
| `enum` (e.g. `StatementType`, `Rounding`) | `u8` discriminant, table below | unknown discriminant rejected |
| `Option<T>` | `0x00` for `None`; `0x01 || canonical_bytes(T)` for `Some` | |
| `Vec<T>` / `String` (UTF-8) | `u32` length prefix, then elements in order | strings are NFC-normalized UTF-8; length is in bytes |
| `struct` | concatenation of fields in declared field order (table below) | no field names in the byte stream; order is the schema |
| map / object | forbidden in canonical bytes | all maps are lowered to ordered structs/vecs at parse time |

Floating point is forbidden in any structure reachable by `canonical_bytes`.
The manifest, `PublicInput`, and witness carry only integers, enums, and
commitments. Scales are declared as integer shift amounts and integer
multipliers in the scale table (`docs/spec/03-data-model.md#tensor-types`),
never as `f32`/`f64`.

Enum discriminants (immutable; appending new variants is allowed, renumbering is
a breaking change gated by `relation_id`):

```text
StatementType: P0Step=0, P1Rollout=1, P2FixedCandidatePlanning=2, P3Cem=3, P4PixelToPlan=4
Rounding:      NearestTiesToEven=0, TruncateTowardZero=1
OverflowPolicy: Reject=0
```

Field order for the public input is its declaration order in the canonical
signature in `docs/spec/03-data-model.md#public-input`, reproduced here as the
serialization schedule:

```text
canonical_bytes(PublicInput) =
    relation_id                       // [u8;32]
 || model_commitment                  // [u8;32]
 || quantization_commitment           // [u8;32]
 || planner_config_commitment         // [u8;32]
 || u8(statement_type discriminant)
 || enc(latent_history_commitment)    // Option<[u8;32]>
 || enc(latent_history_public)        // Option<Vec<M31>>
 || enc(goal_latent_commitment)       // Option<[u8;32]>
 || enc(goal_latent_public)           // Option<Vec<M31>>
 || enc(candidate_actions_commitment) // Option<[u8;32]>
 || enc(candidate_actions_public)     // Option<Vec<M31>>
 || claimed_output_commitment         // [u8;32]
 || enc(selected_index)               // Option<u32>
 || enc(selected_cost)                // Option<BoundedInt> = i64 value || i64 lo || i64 hi
```

`BoundedInt` serializes as `value (i64 LE) || lo (i64 LE) || hi (i64 LE)`. The
declared bounds are part of the canonical bytes: two values with the same
`value` but different `[lo, hi]` are distinct artifacts, because the bounds are
load-bearing for range-check soundness
(`docs/spec/06-security.md#soundness-requirements`).

Tensors that are committed (latent history, goal latent, candidate actions,
claimed outputs) serialize as:

```text
canonical_bytes(Tensor) =
    u32(tensor_id) || u32(scale_id)
 || u32(rank) || rank * u32(shape[k])     // row-major dims
 || u32(len)  || len  * canonical_bytes(BoundedInt(data[i]))   // len == product(shape)
```

The public-vector forms in `PublicInput` (`*_public: Option<Vec<M31>>`) carry
already-field-reduced `M31` residues and are the encoding actually used when the
input is public; the commitment form (`*_commitment`) is used when the input is
committed-only. Exactly one of the two is `Some` for each input role; both
`Some` or both `None` for a role required by the statement type is a structural
error (INV-SER-08).

The on-disk bundle:

```text
canonical_bytes(ProofArtifact) =
    u32(artifact_version)
 || canonical_bytes(public_input)
 || u32(proof_len) || proof_bytes        // opaque Stwo proof; see note below
 || enc(claimed_outputs)                 // Option<Vec<Tensor>>
```

The `proof_bytes` block is the vendored Stwo proof serialized by Stwo's own
serializer. This RFC does NOT canonicalize Stwo's internal proof bytes; it pins
the Stwo revision (RFC-0015) so the serializer is fixed, and it length-prefixes
the block so the surrounding bundle stays unambiguous. `proof_bytes` are never
absorbed into the channel during verification (the verifier reconstructs the
transcript from public data and the proof's own commitment phases, per §4).

### 2. Public-input digest

```rust
// pwm-core/src/transcript.rs
pub const PID_DOMAIN_TAG: &[u8; 16] = b"pwm.pid.v1\0\0\0\0\0\0";

pub fn public_input_digest(pi: &PublicInput) -> [u8; 32] {
    blake2s256(&[PID_DOMAIN_TAG.as_slice(), &canonical_bytes(pi)].concat())
}
```

`pid` is a deterministic function of the public input alone. The verifier in
`docs/spec/04-error-model.md#recovery` step "Recompute public input digest"
computes `pid` and the AIR binds it: a dedicated preprocessed/public column
carries `pid` decomposed into `M31` limbs, and a boundary constraint forces the
trace's bound digest to equal the verifier-supplied `pid`. This is the
mechanism by which "the proof is valid only for this exact public input" is
enforced inside the constraint system rather than only at the application layer.
`relation_id` is the first 32 bytes hashed into `pid`, so a P0 proof submitted
as P1 yields a different `pid` and a different absorbed transcript (the
accepting/rejecting tests in §"Testing Strategy" exercise this).

### 3. Commitment scheme (Blake-based for V0)

V0 fixes one hash primitive for all commitments: Blake2s with 256-bit output
(`blake2s256`), matching the vendored Stwo `Blake2sChannel` and Blake2s Merkle
backend (verified against upstream at 2026-06-03;
`crates/stwo/src/core/channel/`). Rationale: using the same primitive for
commitments and for the Fiat-Shamir channel keeps the V0 hashing surface to a
single audited function, and Blake2s is the channel backend Stwo's V0 path is
built on. Poseidon252 (also present upstream) is reserved for the recursive
verifier (RFC-0012) where an arithmetization-friendly hash matters; V0 does not
use it.

Scalar commitments (domain-separated, length-prefixed):

```rust
// pwm-core/src/commit.rs
pub fn commit(domain_tag: &[u8; 16], payload: &[u8]) -> [u8; 32] {
    // domain_tag distinguishes commitment kinds so a manifest digest can never
    // collide with a public-input digest or a config digest.
    blake2s256(&[domain_tag.as_slice(), &(payload.len() as u64).to_le_bytes(), payload].concat())
}

pub const TAG_MODEL:   &[u8;16] = b"pwm.model.v1\0\0\0\0";
pub const TAG_QUANT:   &[u8;16] = b"pwm.quant.v1\0\0\0\0";
pub const TAG_PLANNER: &[u8;16] = b"pwm.plan.v1\0\0\0\0\0";
pub const TAG_OUTPUT:  &[u8;16] = b"pwm.out.v1\0\0\0\0\0\0";
pub const TAG_TENSOR:  &[u8;16] = b"pwm.tensor.v1\0\0\0";
```

| `PublicInput` field | Definition |
| --- | --- |
| `model_commitment` | `commit(TAG_MODEL, canonical_bytes(model_binding))` where `model_binding` is the manifest's architecture+ops+weights.root+shapes+relation_version+serialization_version sub-structure (the full list in `docs/spec/06-security.md#binding-requirements`). |
| `quantization_commitment` | `commit(TAG_QUANT, canonical_bytes(quant_binding))` over `arithmetic, default_rounding, overflow_policy, clamp_policy, scale table, activation_tables_commitment`. |
| `planner_config_commitment` | `commit(TAG_PLANNER, canonical_bytes(planner_binding))` over `horizon, action_block, candidate count S, tie-break rule id`. |
| `claimed_output_commitment` | `commit(TAG_OUTPUT, canonical_bytes(claimed_outputs))` over the committed output tensors in declared order. |
| `latent_history_commitment` etc. | `commit(TAG_TENSOR, canonical_bytes(Tensor))` for each committed input tensor. |
| `weights.root` (manifest) | Merkle root over the per-tensor `canonical_bytes(Tensor)` leaves; see below. |

Weight Merkle tree (`commitment_scheme: blake2s_merkle_v1` in the manifest):

```text
leaf_i      = blake2s256(b"pwm.wleaf.v1\0\0\0\0" || u32(tensor_id) || canonical_bytes(Tensor_i))
node(a,b)   = blake2s256(b"pwm.wnode.v1\0\0\0\0" || a || b)
root        = fold over leaves sorted by ascending tensor_id; odd level duplicates the last node
```

Sorting leaves by `tensor_id` removes ordering freedom; the manifest declares
tensor ids, so the tree shape is fully determined by the manifest. The V0
`commitment_scheme` value is `blake2s_merkle_v1`; the `blake3` / `poseidon`
options sketched in the feasibility study §4.2 are NOT V0 options and are
rejected for V0 (see Alternatives).

### 4. Fiat-Shamir transcript (the load-bearing ordering)

V0 uses the vendored Stwo `Blake2sChannel`
(`crates/stwo/src/core/channel/`, verified against upstream at 2026-06-03).
Prover and verifier instantiate the channel identically and absorb in exactly
the order below. The ordering is the soundness boundary: every value that the
prover could otherwise choose adaptively must be absorbed before the challenge
that depends on it is drawn.

```text
TRANSCRIPT SCHEDULE  (channel = Blake2sChannel; same on prover and verifier)

T0  channel := Blake2sChannel::default()                  // fixed initial state
T1  mix_u64(channel, TRANSCRIPT_PROTOCOL_VERSION)         // = 1 for v0.1; bumps invalidate old proofs
T2  mix_bytes(channel, public_input_digest(public_input)) // the 32-byte pid from §2
T3  mix_log_sizes(channel, component_log_sizes)           // ordered per the component schedule, §4.1
    --- preprocessed/constant phase ---
T4  mix_bytes(channel, preprocessed_trace_commitment)     // Stwo Merkle root of fixed columns
    --- main (witness) trace phase ---
T5  mix_bytes(channel, main_trace_commitment)             // Stwo Merkle root of main columns
T6  alpha   := draw_secure_felt(channel)                  // LogUp batching challenge(s)
    (draw as many QM31 challenges as the LogUp relation count requires, in
     component-schedule order; see §4.1)
    --- interaction (LogUp) trace phase ---
T7  mix_bytes(channel, interaction_trace_commitment)      // Stwo Merkle root of LogUp columns
T8  mix_secure_felt(channel, logup_claimed_sum)           // the global LogUp sum, absorbed before OODS
    --- constraint composition / DEEP-ALI ---
T9  z       := draw_secure_felt(channel)                  // OODS point
T10 mix_secure_felts(channel, oods_evaluations)           // all trace+composition evals at z and shifts
T11 beta    := draw_secure_felt(channel)                  // DEEP combination challenge
    --- FRI ---
T12 for each FRI layer L in 0..n_layers:
        mix_bytes(channel, fri_layer_commitment[L])
        fold_challenge[L] := draw_secure_felt(channel)
T13 mix_secure_felts(channel, fri_last_layer_poly)
T14 pow_nonce := draw / verify proof-of-work nonce against channel (grinding bits per security params)
T15 query_indices := draw_query_positions(channel, n_queries, domain_log_size)
```

T1 through T15 are produced by Stwo's `CommitmentSchemeProver` / `prove` and
consumed by `CommitmentSchemeVerifier` / `verify`. The ProvableWorldModel
contribution that this RFC freezes is T1–T3 and T8: the protocol-version mix,
the public-input digest mix, the component log-size mix, and the explicit
absorption of the LogUp claimed sum BEFORE the OODS point is drawn. T1–T3 occur
before any prover-chosen commitment, so the prover cannot adapt the public input
or the protocol version to a later challenge. T8 binds the LogUp claimed sum
into the transcript before `z`, closing the standard adaptive-soundness gap for
lookup arguments.

```rust
// pwm-core/src/transcript.rs — the single shared entry point
pub const TRANSCRIPT_PROTOCOL_VERSION: u64 = 1;

/// Identical call on prover (pwm-prover) and verifier (pwm-verifier).
/// Performs T0..T3 then returns the channel to the Stwo commitment scheme,
/// which performs T4..T15. There is exactly one implementation; both sides
/// call this function so divergence is impossible by construction.
pub fn init_channel(public_input: &PublicInput,
                    component_log_sizes: &[u32]) -> Blake2sChannel { /* T0..T3 */ }
```

#### 4.1 Component and relation schedule

`component_log_sizes` (T3) and the LogUp challenge draws (T6) are ordered by the
fixed component schedule. The schedule is the topological component order
declared by the AIR for the active `statement_type`, frozen as a constant table
in `pwm-air` and cross-referenced from
`docs/spec/01-architecture.md#component-model`. For V0/P2 the order is:

```text
0 range_check   1 tensor_memory   2 linear   3 matmul   4 requant
5 activation_lookup   6 layernorm   7 attention   8 mlp
9 predictor  10 rollout  11 cost  12 argmin
```

LogUp challenges at T6 are drawn one block per relation in this same order
(range relations, then activation-table relations, then tensor-memory
permutation), so the challenge-to-relation binding is positional and identical
on both sides. The schedule is part of the relation and is bound transitively
through `relation_id` (any reorder mints a new `relation_id`).

### 5. Invariants

- INV-SER-01 (canonical roundtrip): `decode(canonical_bytes(x)) == x` and
  `canonical_bytes(decode(b)) == b` for every accepted `b`. Non-canonical bytes
  are rejected on decode.
- INV-SER-02 (no floats): no value reachable by `canonical_bytes` is a
  floating-point number; scales are integer (shift, multiplier) pairs.
- INV-SER-03 (field canonicality): every `M31` byte field is a residue in
  `[0, p)`; `>= p` is rejected.
- INV-SER-04 (length-prefixing): every variable-length component is length
  prefixed, so concatenation is unambiguous and no two distinct values share a
  preimage by reparse.
- INV-SER-05 (domain separation): every hash call is prefixed by a distinct
  16-byte domain tag; no two commitment kinds share a tag.
- INV-SER-06 (bounds are bound): `BoundedInt` serialization includes `lo`/`hi`;
  changing declared bounds changes the artifact bytes and every dependent
  commitment.
- INV-TR-01 (digest binds public input): the AIR binds `pid` to a boundary
  column; a proof verifies only against the public input whose `pid` matches.
- INV-TR-02 (single transcript impl): prover and verifier call the same
  `init_channel`; there is no second code path. Enforced by the differential
  test in `docs/spec/07-testing-strategy.md#differential-tests`.
- INV-TR-03 (absorb-before-challenge): every prover-chosen value is mixed into
  the channel before any challenge that may depend on it is drawn; in particular
  `pid` and the protocol version precede T4, and `logup_claimed_sum` precedes
  the OODS point `z`.
- INV-TR-04 (version monotonicity): `TRANSCRIPT_PROTOCOL_VERSION` is absorbed at
  T1; incrementing it makes all prior proofs fail verification by construction.
- INV-CM-01 (commitment determinism): a commitment is a pure function of
  `canonical_bytes` of its payload and its domain tag; identical payloads on any
  platform yield identical 32-byte commitments.
- INV-CM-02 (weight-tree determinism): the weight Merkle tree shape is fixed by
  the manifest's tensor-id set; leaves are ordered by ascending `tensor_id`.

### 6. Failure modes and system responses

| # | Failure | Detected by | System response |
| --- | --- | --- | --- |
| F1 | Non-canonical encoding (e.g. `M31 >= p`, bad bool byte, trailing bytes) | decoder, on load | `VerifyError::Deserialization` / export `ManifestError::NonCanonical`; reject before any hashing (`docs/spec/04-error-model.md#error-taxonomy`) |
| F2 | `pid` recomputed by verifier differs from the digest bound in the proof | T2 vs boundary column at verify | `VerifyError::PublicInputMismatch`; `docs/spec/04-error-model.md#verifier-rejections` |
| F3 | `relation_id` not in the supported set | verifier step 2 (`docs/spec/04-error-model.md#recovery`) | `VerifyError::UnsupportedRelation` |
| F4 | `model_commitment` / `quantization_commitment` / `planner_config_commitment` mismatch vs manifest re-commit | recompute commitment, compare | `VerifyError::CommitmentMismatch` |
| F5 | Channel divergence (prover mixed in a different order or different bytes) | FRI/query checks fail because challenges differ | `VerifyError::ProofInvalid`; INV-TR-02/INV-TR-03 prevent this for honest impls |
| F6 | `artifact_version` unknown | bundle parse | `VerifyError::ArtifactVersion`; refuse to parse, no fallback guessing |
| F7 | Both `*_public` and `*_commitment` set (or both unset) for a required input role | structural validation | `VerifyError::PublicInputMismatch` (INV-SER-08) |
| F8 | Weight Merkle leaves not strictly ordered / duplicate `tensor_id` | tree builder | `ManifestError::NonCanonical`; export aborts |
| F9 | `TRANSCRIPT_PROTOCOL_VERSION` mismatch (old proof, new verifier) | T1 mix yields different challenges | `VerifyError::ProofInvalid`; surfaced as version-gated rejection in observability (`docs/spec/05-observability.md#logging`) |

## Alternatives Considered

- Hash the canonical YAML/JSON text directly (the feasibility study's
  `serialization.canonical_json_hash`, §4.2). Considered because it is simple
  and human-auditable. Rejected as the binding value: text formats admit
  whitespace, key-ordering, Unicode-normalization, and number-formatting freedom
  (`1.0` vs `1`, `+0` vs `0`), so two semantically identical manifests can hash
  differently across YAML libraries and language runtimes, violating INV-SER-01.
  We keep `canonical_json_hash` as a non-binding human audit aid and bind
  `canonical_bytes`.
- Use Poseidon252 (also vendored in Stwo) for all V0 commitments. Considered
  because an arithmetization-friendly hash makes in-circuit recomputation cheap,
  which matters for recursion. Rejected for V0: it doubles the audited hashing
  surface (Poseidon for commitments, Blake2s for the V0 channel), adds field
  parameter choices to the threat model, and V0 has no in-circuit hash
  recomputation requirement. Poseidon252 is the intended primitive for RFC-0012
  recursion, where in-circuit verification makes it worthwhile; the
  `commitment_scheme` manifest field and the domain-tag scheme are designed so a
  `poseidon252_merkle_v1` variant can be added without changing the
  serialization layer.
- A self-describing serialization (CBOR, Protobuf, MessagePack with canonical
  mode). Considered for ecosystem tooling. Rejected because canonical modes are
  underspecified or optional across implementations (canonical CBOR ordering,
  Protobuf's explicit lack of a canonical form), reintroducing the cross-impl
  determinism risk we are trying to eliminate; a tag-free fixed-order binary
  encoding has exactly one valid byte string per value and is trivial to audit.
- Draw the OODS point `z` before absorbing the LogUp claimed sum (omit T8).
  Considered because it is one fewer absorb. Rejected: it lets a malicious
  prover choose the claimed sum after seeing `z`, which is a known
  adaptive-soundness hole for lookup arguments; INV-TR-03 forbids it.

## Drawbacks

- The binary `canonical_bytes` format is not human-readable; debugging a
  mismatch requires a hexdump-and-decode tool. We mitigate with
  `pwm-core` providing a `canonical_decode_debug` pretty-printer and by keeping
  the human-readable manifest YAML alongside.
- Committing to Blake2s for V0 means the V0 proof is not efficiently
  verifiable inside another STARK; recursion (RFC-0012) will require either a
  Blake2s-in-circuit verifier or a re-proof under Poseidon252. This is an
  accepted V0 cost given recursion is `Future`.
- Fixing the component schedule (§4.1) as part of the transcript means any AIR
  component reordering mints a new `relation_id` and invalidates prior proofs.
  This is intentional (it is what makes the schedule sound) but it raises the
  cost of AIR refactors; mitigated by relation versioning (Migration / Rollout).
- A single shared `init_channel` is a coupling point between `pwm-prover` and
  `pwm-verifier` through `pwm-core`; a bug there breaks both sides. This is by
  design (INV-TR-02): one implementation is auditable, two are not.

## Migration / Rollout

- All formats are versioned with explicit integers, not inferred:
  `artifact_version` (bundle), `TRANSCRIPT_PROTOCOL_VERSION` (channel),
  `manifest_version` (manifest), and `serialization_version` (bound into
  `model_commitment`). The v0.1 values are `artifact_version = 1`,
  `TRANSCRIPT_PROTOCOL_VERSION = 1`, `manifest_version = "pwm-model-manifest-v1"`,
  `serialization_version = 1`.
- Any change to `canonical_bytes`, the digest, the commitment scheme, or the
  transcript schedule is a breaking change: it bumps the relevant version
  constant AND mints a new `relation_id` (`pwm.lewm.<statement>.v<N+1>`) per
  `docs/spec/09-release-and-versioning.md#relation-versioning`. Old proofs
  continue to verify only under their original constants; verifiers refuse
  unknown `artifact_version` (F6) rather than guessing.
- Feature flag `commitment_scheme` in the manifest selects the commitment
  primitive; V0 accepts only `blake2s_merkle_v1`. A future
  `poseidon252_merkle_v1` is added behind the same field with its own domain
  tags, so the serialization layer is untouched and old artifacts remain valid.
- The Stwo channel implementation is pinned via
  `docs/rfcs/RFC-0015-third-party-vendoring-and-pinning.md`
  (`third_party/stwo/REVISION`); bumping the vendored revision in a way that
  changes channel byte behavior is treated as a `TRANSCRIPT_PROTOCOL_VERSION`
  bump and is covered by the transcript golden vectors.
- Deprecation: when a transcript version is retired, the verifier keeps the
  ability to reject (not silently accept) the old version for at least one
  minor release, logging F9 with the observed version
  (`docs/spec/05-observability.md#logging`).

## Testing Strategy

Cross-references `docs/spec/07-testing-strategy.md` and the rejection taxonomy in
`docs/spec/04-error-model.md#verifier-rejections`.

Accepting tests:

- `test_canonical_roundtrip_manifest`: parse the V0 reference manifest, encode,
  decode, re-encode; assert byte-identical and structurally equal (INV-SER-01),
  cross-ref `docs/spec/07-testing-strategy.md#golden-vectors`.
- `test_public_input_digest_golden`: fixed `PublicInput` fixture hashes to a
  checked-in 32-byte golden `pid`; guards against accidental schedule changes.
- `test_commitment_golden_model_quant_planner`: each commitment over a fixture
  matches a checked-in golden 32-byte value (INV-CM-01).
- `test_weight_merkle_root_golden`: weight tree over a 3-tensor fixture matches a
  checked-in root; reordering input tensors does not change the root (leaves
  sorted by `tensor_id`, INV-CM-02).
- `test_transcript_replay_matches`: prover-side `init_channel` and verifier-side
  `init_channel` on the same `PublicInput` and `component_log_sizes` produce
  byte-identical channel state after T3 (INV-TR-02); a differential test per
  `docs/spec/07-testing-strategy.md#differential-tests`.
- `test_end_to_end_p2_accept`: a valid P2 `ProofArtifact` verifies and the bound
  `pid` equals the recomputed `pid` (INV-TR-01).
- `test_cross_platform_determinism`: encode + commit + `pid` computed on two
  targets (x86_64, aarch64) yield identical bytes (INV-SER-02, INV-CM-01),
  cross-ref `docs/spec/08-performance-budget.md#profiling`.

Rejecting / negative tests (each asserts the exact `VerifyError`, cross-ref
`docs/spec/04-error-model.md#verifier-rejections` and
`docs/spec/07-testing-strategy.md#negative-tests`):

- `reject_noncanonical_m31`: an `M31` byte field `>= p` -> F1
  `VerifyError::Deserialization`.
- `reject_trailing_bytes`: extra bytes after a decoded `ProofArtifact` -> F1.
- `reject_pid_mismatch`: flip one byte of the public input after the proof is
  made -> F2 `VerifyError::PublicInputMismatch`.
- `reject_relation_id_swap`: submit a P0 proof under the P1 `relation_id` -> F2
  (different `pid`) then F3 `VerifyError::UnsupportedRelation` if the id is
  unknown, exercising the "P0 submitted as P1 is rejected" requirement from
  `docs/rfcs/RFC-0000-security-model-and-statement-taxonomy.md`.
- `reject_model_commitment_tamper`: mutate one weight, recompute manifest, keep
  old `model_commitment` -> F4 `VerifyError::CommitmentMismatch`.
- `reject_quant_commitment_tamper` and `reject_planner_commitment_tamper`:
  analogous -> F4.
- `reject_unknown_artifact_version`: set `artifact_version = 999` -> F6
  `VerifyError::ArtifactVersion`.
- `reject_both_public_and_committed`: set both `goal_latent_public` and
  `goal_latent_commitment` -> F7 (INV-SER-08).
- `reject_duplicate_weight_tensor_id`: two leaves with the same `tensor_id` ->
  F8 `ManifestError::NonCanonical`.
- `reject_transcript_version_skew`: verify a `TRANSCRIPT_PROTOCOL_VERSION = 1`
  proof with a verifier built for version 2 -> F9 `VerifyError::ProofInvalid`.
- `reject_logup_sum_after_oods` (mutation test,
  `docs/spec/07-testing-strategy.md#mutation-tests`): a prover variant that
  draws `z` before mixing `logup_claimed_sum` must fail an honest verifier,
  demonstrating INV-TR-03 is load-bearing.

## Open Questions

- OPEN QUESTION (owner: maintainers / area:security): exact proof-of-work
  grinding bit count at T14 and `n_queries` at T15 for the V0 target soundness
  level. These are Stwo security parameters, not serialization choices;
  resolution path: fixed in `docs/spec/06-security.md#soundness-requirements`
  and consumed here by reference before v0.2.
- OPEN QUESTION (owner: maintainers / area:core): whether the preprocessed-trace
  commitment at T4 is recomputed by the verifier from the manifest or carried as
  a committed value the verifier checks against the manifest. Resolution path:
  decided in `docs/rfcs/RFC-0004-tensor-memory-and-wiring-air.md` and
  `docs/spec/01-architecture.md#trace-model`; this RFC's schedule is unaffected
  either way because T4 absorbs the resulting root regardless of provenance.

## References

- `docs/feasibility-study.md` §9.1 (binding requirements), §9.2 (determinism),
  §10.1 (public input schema), §10.3 (prover flow), §10.4 (verifier flow).
- `docs/spec/03-data-model.md#public-input`, `#tensor-types`,
  `#bounded-integers`, `#model-manifest`, `#schema-versioning`.
- `docs/spec/04-error-model.md#error-taxonomy`, `#verifier-rejections`,
  `#recovery`.
- `docs/spec/05-observability.md#logging`, `#redaction`.
- `docs/spec/06-security.md#binding-requirements`, `#soundness-requirements`,
  `#privacy-and-zk`.
- `docs/spec/07-testing-strategy.md#golden-vectors`, `#negative-tests`,
  `#differential-tests`, `#mutation-tests`.
- `docs/spec/08-performance-budget.md#profiling`.
- `docs/spec/09-release-and-versioning.md#relation-versioning`.
- `docs/spec/01-architecture.md#component-model`, `#trace-model`.
- `docs/rfcs/RFC-0000-security-model-and-statement-taxonomy.md`,
  `docs/rfcs/RFC-0001-model-manifest-and-export-pipeline.md`,
  `docs/rfcs/RFC-0004-tensor-memory-and-wiring-air.md`,
  `docs/rfcs/RFC-0012-recursive-aggregated-verification.md`,
  `docs/rfcs/RFC-0015-third-party-vendoring-and-pinning.md`,
  `docs/rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md`.
- Stwo Fiat-Shamir channels (Blake2s/Keccak256/Poseidon252),
  `crates/stwo/src/core/channel/`, and LogUp,
  `crates/stwo/src/core/.../constraint-framework/logup.rs`; verified against
  upstream at 2026-06-03.
