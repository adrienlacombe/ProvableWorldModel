# RFC-0001: Model manifest and export pipeline

- Status: Accepted
- Authors: ProvableWorldModel maintainers
- Created: 2026-06-03
- Target milestone: v0.1

## Summary

This RFC fixes the canonical artifact that a ProvableWorldModel proof is bound to:
the **model manifest** (a canonical-JSON-hashed YAML document) and its companion
**quantized weights blob** and **golden vectors**, all produced by a deterministic
five-stage **export pipeline** that consumes a frozen PyTorch LeWorldModel
checkpoint and emits the immutable inputs the prover and verifier agree on. The
decision locked here: there is exactly one manifest schema (the keys of
`docs/spec/03-data-model.md#model-manifest`, mirroring contract §6.6), exactly one
pipeline shape (`PyTorch checkpoint -> canonical graph -> quantized graph ->
manifest -> weights -> golden vectors`), and two binding invariants —
**re-exporting the same checkpoint with the same export config produces byte-identical
artifacts**, and **any change to a weight, bias, scale, rounding rule, shape, op
order, lookup table, or planner config changes a commitment**. The manifest's
`model_commitment` and `quantization_commitment` are what the V0 statement P2
(`docs/spec/00-overview.md#v0-statement`) is proven against; an unbound field is a
field the prover can silently change, which is unsound. le-wm is MIT-licensed; the
exporter reads a checkpoint and config and does not vendor le-wm source.

## Motivation

The founding analysis is explicit that the proof proves *model computation, not
physical truth*, and that this is only sound if the exact quantized relation is
frozen and committed: "A proof without this binding is not sound, because the
prover could silently change the model, quantization, activation approximation, or
planner rule" (`docs/feasibility-study.md` §4.2). The same source designates the
exporter and manifest as required-for-V0 (`docs/feasibility-study.md` §11,
RFC-001) and lists the four acceptance criteria this RFC promotes to invariants
(byte-identical re-export; weight change -> changed `model_commitment`; scale
change -> changed `quantization_commitment`; Rust fixed-point inference matches the
Python fixed-point reference bit-for-bit).

Concrete scenarios this RFC must make safe:

1. **Adversarial prover swaps a weight after committing.** A prover claims a P2
   proof against `model_commitment = C`, then rolls out candidates with a tampered
   weight that produces a more convenient argmin. If the weight is not bound by `C`,
   the verifier cannot detect this. The manifest's Merkle weights `root`, recomputed
   over the canonical weights blob, makes the swap change `model_commitment` and the
   proof invalid for `C` (`docs/spec/06-security.md#binding-requirements`).
2. **Prover silently changes the rounding mode.** The reference uses
   `nearest_ties_to_even`; the prover quantizes with `truncate_toward_zero` to nudge
   a cost. `quantization.default_rounding` is bound by `quantization_commitment`, and
   the AIR enforces exactly one active rounding mode per manifest
   (`docs/feasibility-study.md` §5.4; contract §3).
3. **Reproducibility dispute.** Two parties export the published checkpoint and
   disagree on the commitment. Byte-identical re-export (INV-MANIFEST-01) makes this
   a deterministic, diffable artifact, not a debate.

The architecture and data-model docs depend on this artifact:
`docs/spec/01-architecture.md#data-flow` routes the manifest from `pwm-export`
through `pwm-prover` and `pwm-verifier`; `docs/spec/03-data-model.md#model-manifest`
owns the schema this RFC fixes; `docs/spec/02-public-api.md#artifact-formats`
specifies the on-disk encodings.

## Goals

- **G1.** Define one canonical manifest schema (contract §6.6 keys), with every
  field typed, and the rule that the verifier reads only the manifest and the
  committed blobs — never PyTorch, never live Python.
- **G2.** Define the five-stage export pipeline as a sequence of pure functions with
  typed inputs/outputs, each stage independently testable and each stage's output a
  canonical, hashable artifact.
- **G3.** Lock the commitment derivation: which bytes feed `model_commitment`,
  `quantization_commitment`, and `weights.root`, and the canonicalization that makes
  re-export byte-identical (INV-MANIFEST-01, INV-MANIFEST-02).
- **G4.** Specify the LeWorldModel operator-graph extraction (V0 supported ops:
  action_encoder + ARPredictor + pred_proj + rollout + goal-MSE + argmin) and the
  per-module activation function recording (predictor FFN GELU; AdaLN/Embedder SiLU),
  so the manifest faithfully records the graph the AIR must match.
- **G5.** Specify the golden-vector format and the Python<->Rust parity contract
  (INV-MANIFEST-05) that gates the rest of the proving stack.
- **G6.** Specify manifest schema versioning, `relation_id` binding, and how the
  artifact bundle evolves without breaking existing proofs.

## Non-Goals

- **NG1.** Defining fixed-point arithmetic semantics (add/mul/requantize/clamp/round
  numerics). Owned by `docs/rfcs/RFC-0002-fixed-point-arithmetic-over-m31.md`. This
  RFC records *which* scales/rounding a tensor uses; RFC-0002 defines *what they mean*.
- **NG2.** Defining AIR component layouts (linear, attention, requant). Owned by
  RFC-0005 / RFC-0006 / RFC-0007. This RFC defines the manifest those components read.
- **NG3.** Choosing quantization-calibration algorithms (how int8 scales are
  *picked*). The pipeline records the chosen scales; an OPEN QUESTION below tracks the
  calibration policy. V0 accepts caller-supplied or power-of-two scales.
- **NG4.** Defining the proof-native nonlinear approximations (GELU/Softmax/LayerNorm
  tables themselves). RFC-0006 defines them; the manifest only *commits* their tables
  via `quantization.activation_tables_commitment` and per-op `lookup_table` ids.
- **NG5.** The pixel encoder export (PatchEmbed/ViTEncoder/Projector). Deferred to
  RFC-0011 (V3); the schema reserves `architecture` keys for it but V0 export rejects
  encoder ops.
- **NG6.** The CEM planner config proving. The manifest carries
  `planner_config_commitment`; CEM semantics are RFC-0010 (Future).

## Proposed Design

### 1. Artifact set (the export output)

A single export run with a fixed export config produces an **export bundle**:

```text
<bundle>/
  manifest.yaml          # canonical YAML; the human/tooling-facing schema
  manifest.canonical.json# canonical-JSON serialization actually hashed
  weights.bin            # canonical packed quantized weights + biases
  weights.merkle         # Merkle layer hashes (verifier-side recomputation aid)
  golden/
    vectors.jsonl        # one record per golden case (see §6)
    INPUTS/              # input tensors per case (canonical packed)
  EXPORT.lock            # exporter version + checkpoint hash + export-config hash
```

`manifest.yaml` is authored/read by humans and tooling; `manifest.canonical.json`
is the byte sequence whose Blake3 hash is `serialization.canonical_json_hash`. The
two are equivalent under the canonicalization in §3; CI checks they round-trip.

### 2. Manifest schema (canonical; mirrors contract §6.6, owned by `docs/spec/03-data-model.md#model-manifest`)

Top-level keys, in canonical (sorted) order. Types use Rust notation for the
in-memory representation (`pwm-core::manifest`); the on-disk form is YAML/JSON.

```text
manifest_version: u32                  # schema version; V0 = 1
model_family: String                   # "lewm"
relation_id: String                    # "pwm.lewm.fixed_candidate_planning.v1" (V0)
field:
  base: "M31"                          # p = 2^31 - 1 (contract §3)
  extension: "QM31"                    # degree-4 secure field; challenges only
  signed_encoding: "centered_mod_p"    # BoundedInt centered encoding (contract §6.1)
  max_abs_value: i64                   # global B such that |value| <= B holds
quantization:
  arithmetic: "fixed_point"
  default_rounding: "nearest_ties_to_even" | "truncate_toward_zero"  # exactly one
  overflow_policy: "reject"            # V0: reject only (contract §3)
  clamp_policy: "explicit"             # per-tensor clamp ranges declared in ops[]
  activation_tables_commitment: [u8; 32]  # Blake3 over all lookup-table bytes
architecture:
  latent_dim: u32                      # V0 reference targets 192
  history_size: u32                    # V0 reference targets 3
  predictor:
    type: "ar_transformer"
    depth: u32                         # V0 reference targets 6
    heads: u32                         # 16
    dim_head: u32                      # 64
    mlp_dim: u32                       # 2048
  action_encoder: { type: "embedder" }
  pred_proj: { type: "mlp_or_affine" }
weights:
  visibility: "public" | "private_committed"
  commitment_scheme: "blake3_merkle"   # V0 fixes one scheme (see §3)
  root: [u8; 32]                       # Merkle root over weights.bin canonical layout
ops:                                   # ordered list; index is the operator-graph order
  - id: String                         # e.g. "predictor.block0.attn.qkv"
    op: String                         # "linear" | "matmul" | "requant" | "gelu_q" | ...
    inputs: Vec<String>                # producer op ids / manifest-declared tensor ids
    outputs: Vec<String>
    weight_tensor_id: Option<u32>      # index into weights.bin tensor table
    bias_tensor_id: Option<u32>
    input_scale_id: u32                # index into the scale table (§4)
    weight_scale_id: Option<u32>
    output_scale_id: u32
    rounding: Option<String>           # per-op override; absent => default_rounding
    clamp: Option<{lo: i64, hi: i64}>  # required when clamp_policy=explicit & op clamps
    activation_fn: Option<String>      # "gelu" | "silu"  (per-module; see §5)
    lookup_table: Option<String>       # table id for nonlinear ops (RFC-0006)
    commitment: [u8; 32]               # Blake3 over this op's canonicalized record
scale_table:                           # the per-tensor scales referenced by scale_id
  - { scale_id: u32, kind: "pow2" | "rational", shift: Option<i32>,
      num: Option<i64>, den: Option<i64> }
planner:
  statement: "P2_fixed_candidate"      # V0
  num_candidates: u32                  # S
  horizon: u32                         # V0 reference targets 5
  action_block: u32                    # V0 reference targets 5
  cost: "mse_goal_latent"
  tie_break: "smallest_index"
serialization:
  canonical_json_hash: [u8; 32]        # Blake3 of manifest.canonical.json with this
                                       # field zeroed during hashing (self-reference)
```

The `pwm-core` typed view:

```rust
// pwm-core::manifest
pub struct Manifest {
    pub manifest_version: u32,
    pub model_family: String,
    pub relation_id: String,
    pub field: FieldSpec,
    pub quantization: QuantizationSpec,
    pub architecture: ArchitectureSpec,
    pub weights: WeightsSpec,        // visibility, commitment_scheme, root: [u8;32]
    pub ops: Vec<OpSpec>,            // ordered operator graph
    pub scale_table: Vec<ScaleSpec>,
    pub planner: PlannerSpec,
    pub serialization: SerializationSpec, // canonical_json_hash: [u8;32]
}

pub struct ManifestCommitments {     // derived, not stored in the manifest body
    pub model_commitment: [u8; 32],
    pub quantization_commitment: [u8; 32],
    pub planner_config_commitment: [u8; 32],
}
```

These three derived commitments are exactly the fields the `PublicInput` carries
(contract §6.3): `model_commitment`, `quantization_commitment`,
`planner_config_commitment`. The verifier recomputes them from the manifest it is
given and checks them against the proof's public input
(`docs/spec/02-public-api.md#rust-public-api`).

### 3. Commitment derivation and canonicalization (locks INV-MANIFEST-01/02/03)

**Canonical JSON.** `manifest.canonical.json` is produced by: (a) UTF-8, no BOM;
(b) object keys sorted lexicographically by Unicode code point; (c) no insignificant
whitespace; (d) integers as bare decimal, no leading zeros, no `+`; (e) byte arrays
as lowercase hex strings; (f) arrays preserve declared order (`ops` order is
semantically load-bearing — it is the operator-graph order). This is the same
discipline RFC-0014 (`docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md`)
applies to the public-input digest; the manifest uses the RFC-0014 canonical encoder
so prover and verifier never disagree on bytes.

**Hash function.** Blake3 for all manifest/weights commitments. Rationale in
Alternatives. (This is distinct from the Fiat-Shamir channel hash, which RFC-0014 /
the vendored Stwo channel selects independently; see contract §9b: Stwo provides
Blake2s/Keccak256/Poseidon252 channel backends.)

**`weights.root`.** `weights.bin` packs tensors in `weights.bin` tensor-table order;
each tensor is canonically serialized (`Tensor` per contract §6.2: row-major
`BoundedInt` values, leaf-encoded as fixed-width little-endian `i64` plus declared
`lo`/`hi`). The Merkle tree is a fixed-arity (binary, duplicate-last-leaf padding)
Blake3 tree over fixed-size chunks of `weights.bin`; `weights.root` is its root. A
single changed weight changes one leaf and therefore the root (INV-MANIFEST-02).

**`model_commitment`** = `Blake3( canonical_json_of( {manifest_version, model_family,
relation_id, field, architecture, weights, ops, scale_table, serialization} ) )`.
It binds architecture, the weights root, the full op graph (including order, scales,
rounding overrides, clamps, activation fns, lookup-table ids, and each op's own
`commitment`), and the scale table. Anything in this set, if changed, changes
`model_commitment`.

**`quantization_commitment`** = `Blake3( canonical_json_of( {quantization,
scale_table, per-op (rounding, clamp, input/weight/output scale ids)} ) )`. It binds
the rounding mode, overflow/clamp policy, activation-table commitment, and the full
scale assignment. A changed scale changes `quantization_commitment`
(INV-MANIFEST-03).

**`planner_config_commitment`** = `Blake3( canonical_json_of( planner ) )`. Binds
statement, `num_candidates`, `horizon`, `action_block`, cost rule, tie-break.

The overlap (`scale_table` feeds both `model_commitment` and
`quantization_commitment`) is intentional and not a soundness gap: both commitments
appear in the public input and both are checked, so a divergence in either is caught.

### 4. The five-stage export pipeline

Each stage is a pure function (deterministic given its declared inputs) in
`pwm-export`. Stages 1–3 are Python (PyTorch dependency); stage 4 and the Rust parity
arm of stage 5 are Rust. Python never runs in the verifier (contract §3).

```text
Stage 1  load_checkpoint   : (ckpt_path, le_wm_config) -> TorchModel (eval mode)
Stage 2  to_canonical_graph: TorchModel -> CanonicalGraph
Stage 3  quantize          : (CanonicalGraph, ExportConfig) -> QuantizedGraph
Stage 4  emit_manifest      : QuantizedGraph -> (Manifest, weights.bin, ManifestCommitments)
Stage 5  emit_golden        : (QuantizedGraph, Manifest, GoldenSpec) -> GoldenVectors
                              + Rust parity replay over the same vectors
```

**Stage 1 — load_checkpoint.** Loads the published LeWorldModel checkpoint with
`module.eval()` (dropout disabled; BatchNorm folded — see RFC-0006 NG handling). The
exporter consumes the checkpoint and the le-wm config; it does not vendor le-wm
source. `EXPORT.lock` records `checkpoint_sha256` and the resolved le-wm config hash.
**Failure: checkpoint not found / hash mismatch ->** `ExportError::CheckpointUnreadable`
(`docs/spec/04-error-model.md#failure-modes`), export aborts, no partial bundle.

**Stage 2 — to_canonical_graph.** Walks the LeWorldModel modules and produces a
`CanonicalGraph`: an ordered list of typed nodes with explicit tensor edges, derived
by structural traversal of the module tree, **not** by tracing dynamic Python
execution (the relation must not depend on a particular forward call). V0 extracts
exactly: `action_encoder` (Embedder, SiLU), the depth-6 `ARPredictor` of
`ConditionalBlock`s (each: AdaLN modulation with SiLU, multi-head attention via
scaled-dot-product, FeedForward/MLP with GELU), learned positional embeddings, and
`pred_proj`. Encoder/projector nodes, if present in the checkpoint, are recorded as
`unsupported_in_v0` and cause a typed rejection when V0 export is requested.
**Failure: unsupported op (e.g. encoder requested in V0) ->**
`ExportError::UnsupportedOp { op_id }`; **graph references a tensor with no producer
->** `ExportError::DanglingTensor { tensor_id }`.

**Stage 3 — quantize.** Applies the `ExportConfig` (dtype choices: weights int8;
activations int8 or int16 per-tensor; biases int32; accumulators bounded int32 or
limb; scales power-of-two where possible — contract §3) to produce a `QuantizedGraph`
where every node carries its scale ids, rounding, clamp ranges, and (for nonlinear
ops) its lookup-table id. **Per-module activation functions are recorded as found,
not assumed uniform** (verified against upstream at 2026-06-03): the predictor
FFN/MLP records `activation_fn: "gelu"`; AdaLN modulation and the action `Embedder`
record `activation_fn: "silu"`. Accumulator bound checks run here: for an int8 dot
product of length 2048, worst case `2048 * 127 * 127 = 33,032,192 < 2^31 - 1`, so it
fits a single signed M31 value; wider tensors are marked limb-decomposed.
**Failure: a declared scale/dtype lets an accumulator exceed its declared bound
without a limb decomposition ->** `ExportError::AccumulatorBoundViolation { op_id }`,
export aborts (this is `overflow_policy: reject` enforced at export time, before any
proof exists). **Manifest declares a value outside `field.max_abs_value` ->**
`ExportError::RangeDeclarationInfeasible`.

**Stage 4 — emit_manifest.** Serializes the `QuantizedGraph` to `manifest.yaml` +
`manifest.canonical.json`, packs `weights.bin`, computes `weights.root`, the three
derived commitments (§3), and `serialization.canonical_json_hash`. This stage is the
sole writer of commitments. **Failure: canonicalization produces non-round-tripping
bytes (yaml<->json mismatch) ->** `ExportError::CanonicalizationMismatch` (a bug
guard, must never fire in CI).

**Stage 5 — emit_golden + parity.** Produces golden vectors (§6) by running the
Python fixed-point reference over `GoldenSpec` inputs, then replays the identical
inputs through the Rust fixed-point reference (`pwm-core`) and asserts bit-for-bit
equality of every recorded intermediate and output. **Failure: any Python/Rust
divergence ->** `ExportError::ParityMismatch { op_id, index }` with the first
differing cell; export marks the bundle `parity_failed` and CI blocks release.

**Pipeline determinism.** All stages are seeded/seed-free and pinned to the
`EXPORT.lock` exporter version. INV-MANIFEST-01 (byte-identical re-export) is the
end-to-end determinism guarantee.

### 5. Per-module fidelity record (LeWorldModel specifics)

The manifest must record the LeWorldModel graph as it actually is, because the AIR
(RFC-0007) is verified to match the *exported quantized graph*, not live PyTorch:

| LeWorldModel module      | op record(s)                               | activation_fn | notes |
| ------------------------ | ------------------------------------------ | ------------- | ----- |
| `action_encoder` Embedder| `linear` + activation                      | `silu`        | action conditioning |
| `ConditionalBlock` AdaLN | modulation `linear` + activation           | `silu`        | AdaLN-zero modulation |
| `ConditionalBlock` attn  | qkv `linear`, `matmul`, `softmax_q`, `matmul` | (none)     | scaled-dot-product; softmax via table (RFC-0006) |
| `ConditionalBlock` MLP   | `linear`, activation, `linear`             | `gelu`        | FeedForward |
| learned pos. emb.        | preprocessed/static tensor                 | (none)        | public/fixed |
| `pred_proj`              | `mlp_or_affine` (`linear[+activation]`)    | per-config    | prediction projection |

This table is recorded structurally in `ops`; it is reproduced here as prose to
satisfy the no-undocumented-diagram rule. The activation split is a verified
correction (contract §9b): do not assume a uniform activation across modules.

### 6. Golden-vector format (locks INV-MANIFEST-05)

`golden/vectors.jsonl` — one JSON record per line:

```text
{
  "case_id": String,                 // stable id, e.g. "predictor_step.case0"
  "relation_id": String,             // must equal manifest.relation_id
  "statement": "P0Step"|"P1Rollout"|"P2FixedCandidatePlanning",
  "inputs_ref": String,              // path under golden/INPUTS/ (canonical packed)
  "expected": {
     "intermediates": [{op_id, output_ref}],   // per-op output tensors (packed)
     "claimed_output": output_ref,             // final latent / selected index+cost
     "selected_index": Option<u32>,            // P2 only
     "selected_cost": Option<i64>              // P2 only, BoundedInt.value
  },
  "input_commitments": { latent_history: [u8;32], goal_latent: Option<[u8;32]>,
                         candidate_actions: Option<[u8;32]> }
}
```

Golden vectors are the differential oracle: the same record is the accepting fixture
for `pwm-prover` integration tests and the input to the Python<->Rust parity check.
V0 ships golden vectors for at least P0 (one predictor step), P1 (a horizon-5
rollout), and P2 (an S-candidate plan with a deliberate tie to exercise tie-break).

### 7. Lifecycle and data flow

```text
checkpoint + le_wm config + ExportConfig
        | (pwm-export, Python: stages 1-3)
        v
QuantizedGraph
        | (pwm-export, stage 4)
        v
manifest.yaml / manifest.canonical.json / weights.bin / weights.merkle  + ManifestCommitments
        | (pwm-export, stage 5: Python emit + Rust parity replay)
        v
golden/vectors.jsonl  (parity-checked)
        |
        +--> pwm-prover  : reads manifest + weights, recomputes commitments, builds trace,
        |                  emits ProofArtifact whose PublicInput carries the three commitments
        +--> pwm-verifier: reads manifest, recomputes the three commitments, checks them
                           against ProofArtifact.public_input; never reads PyTorch
```

Prose mirror of the diagram: export is the only producer of commitments; both prover
and verifier independently recompute commitments from the manifest they are given and
compare to the public input, so the manifest is the single binding artifact across
the trust boundary (`docs/spec/06-security.md#trust-boundaries`).

### 8. Named invariants

- **INV-MANIFEST-01 (deterministic re-export).** Given the same checkpoint bytes,
  the same le-wm config, and the same `ExportConfig`, the pipeline emits a
  byte-identical `manifest.canonical.json`, `weights.bin`, and `golden/vectors.jsonl`.
  Enforced by `test_reexport_byte_identical`.
- **INV-MANIFEST-02 (weight binding).** Changing any single weight or bias leaf
  changes `weights.root` and therefore `model_commitment`. Enforced by
  `test_weight_flip_changes_model_commitment`.
- **INV-MANIFEST-03 (quantization binding).** Changing any scale, rounding mode,
  clamp range, or `activation_tables_commitment` changes `quantization_commitment`.
  Enforced by `test_scale_change_changes_quant_commitment`.
- **INV-MANIFEST-04 (graph-order binding).** Reordering `ops`, or changing any op's
  `inputs`/`outputs`/`op`/`activation_fn`/`lookup_table`, changes `model_commitment`.
  Enforced by `test_op_reorder_changes_model_commitment`.
- **INV-MANIFEST-05 (Python<->Rust parity).** For every golden case, the Python
  fixed-point reference and the Rust fixed-point reference (`pwm-core`) agree
  bit-for-bit on every recorded intermediate and the claimed output. Enforced by
  `test_python_rust_parity` (a differential test, see
  `docs/spec/07-testing-strategy.md#differential-tests`).
- **INV-MANIFEST-06 (self-consistent canonicalization).** `manifest.yaml` and
  `manifest.canonical.json` round-trip; `serialization.canonical_json_hash` equals the
  Blake3 of `manifest.canonical.json` with that field zeroed during hashing. Enforced
  by `test_manifest_roundtrip_and_self_hash`.
- **INV-MANIFEST-07 (relation binding).** The manifest's `relation_id` is immutable
  for a given semantics; the V0 manifest uses
  `pwm.lewm.fixed_candidate_planning.v1`. A proof is valid only for the manifest's
  declared `relation_id` (contract §4). Enforced by
  `test_relation_id_mismatch_rejected`.

### 9. Failure-mode enumeration (export side; verifier-side rejections in RFC-0009)

| Failure mode                                   | Stage | Typed error (`docs/spec/04-error-model.md#error-taxonomy`) | System response |
| ---------------------------------------------- | ----- | ---------------------------------------------------------- | --------------- |
| Checkpoint missing / hash mismatch             | 1     | `ExportError::CheckpointUnreadable`                        | abort, no bundle |
| Encoder/CEM op requested in V0                 | 2     | `ExportError::UnsupportedOp`                               | abort, list offending op ids |
| Tensor edge with no producer                   | 2     | `ExportError::DanglingTensor`                              | abort |
| Accumulator exceeds bound, no limb decomp.     | 3     | `ExportError::AccumulatorBoundViolation`                   | abort (reject policy) |
| Declared value outside `field.max_abs_value`   | 3     | `ExportError::RangeDeclarationInfeasible`                  | abort |
| yaml<->json non-round-trip                     | 4     | `ExportError::CanonicalizationMismatch`                    | abort (internal-bug guard) |
| Python/Rust intermediate divergence            | 5     | `ExportError::ParityMismatch`                              | mark `parity_failed`, block release |
| Manifest references unknown `scale_id`/`table` | 4     | `ExportError::DanglingReference`                           | abort |

## Alternatives Considered

**A1. Trace-based graph extraction (`torch.fx` / `torch.jit.trace` / ONNX export).**
Extract the operator graph by tracing a forward pass and emit ONNX, then quantize the
ONNX graph. *Considered* because it is the standard ML-export path and `pwm-export`
already names an `onnx_import/` area in the source layout
(`docs/feasibility-study.md` §3.1). *Rejected as the binding source* because a traced
graph depends on the specific inputs and control flow of the trace call; the proof
relation must be a fixed function of the module structure, not of a sampled forward.
ARPredictor's autoregressive rollout has data-dependent windowing that a single trace
would freeze incorrectly. We use structural traversal (Stage 2) as the source of
truth; ONNX import is retained only as an optional cross-check artifact, not as the
committed graph. OPEN QUESTION below tracks whether ONNX cross-check ships in V0.

**A2. Commit weights directly with the Stwo/Poseidon channel hash instead of a
separate Blake3 Merkle tree.** *Considered* because the prover already runs a
Fiat-Shamir channel (Poseidon252 is available in vendored Stwo, contract §9b) and
reusing it avoids a second hash primitive. *Rejected* because the manifest commitment
must be computable by tooling and humans **outside** the prover (CI diffing two
exports, a reviewer recomputing `weights.root` from `weights.bin`), and must be stable
even if the proof system's channel hash changes across Stwo revisions
(RFC-0015 pins, but may re-pin). Decoupling the artifact commitment (Blake3) from the
in-proof channel hash keeps the manifest a stable, audit-friendly artifact. The AIR
still binds the public input (which carries these commitments) through the channel, so
soundness is preserved.

**A3. One monolithic `model_commitment` covering quantization and planner too.**
*Considered* for simplicity — one hash to check. *Rejected* because the public input
(contract §6.3) deliberately carries three separate commitments so that the security
doc (`docs/spec/06-security.md#binding-requirements`) can reason about each binding
independently, and so a future relation can reuse a model under a different planner
config without re-committing weights. Three commitments with a documented overlap
(scale_table) is the locked choice.

## Drawbacks

- **D1.** Structural graph traversal (A1 rejection) means the exporter carries a
  hand-maintained mapping from LeWorldModel module classes to canonical op records; a
  new le-wm module type requires an exporter change rather than being picked up
  automatically by a tracer. Mitigated by the `UnsupportedOp` hard failure (never
  silently mis-export).
- **D2.** Two hash domains (Blake3 for artifacts, channel hash for the proof) is one
  more primitive to audit (A2 rejection). Accepted for the audit/stability benefit.
- **D3.** The parity check (Stage 5) requires maintaining two fixed-point references
  (Python and Rust) in lockstep forever; divergence is a release blocker. This cost is
  intentional: it is the only mechanism that catches a reference bug before it becomes
  a "sound proof of the wrong computation."
- **D4.** Canonical JSON with sorted keys plus an order-significant `ops` array is a
  subtle rule (keys sorted, arrays not) that contributors must respect. Mitigated by
  routing all serialization through the RFC-0014 canonical encoder.

## Migration / Rollout

- **Schema versioning.** `manifest_version` starts at `1`. Additive, optional fields
  may be introduced under the same version only if absent-field semantics are defined
  and the canonical hash of a manifest that omits them is unchanged; any field that
  changes the committed semantics requires bumping `manifest_version` and is governed
  by `docs/spec/09-release-and-versioning.md#schema-versioning`. The loader rejects an
  unknown `manifest_version` with `ExportError`/`VerifyError::UnsupportedManifestVersion`.
- **Relation versioning.** A change to the proven semantics (e.g. a different rounding
  default, a new supported op) mints a new `relation_id`
  (`pwm.lewm.fixed_candidate_planning.v2`); the old `relation_id` and any proofs
  against it remain valid. relation_ids are immutable (contract §4;
  `docs/spec/09-release-and-versioning.md#relation-versioning`).
- **Bundle versioning.** The on-disk bundle layout is versioned by `EXPORT.lock`'s
  exporter version; the `ProofArtifact.artifact_version` (contract §6.5) is bumped
  independently when the bundle layout changes.
- **Feature flags.** V0 export is gated behind a `--statement p2` selector; encoder
  (`--statement p4`) and CEM (`--statement p3`) export paths exist as stubs that
  return `UnsupportedOp` until RFC-0011 / RFC-0010 land. No flag silently changes
  commitment derivation.
- **Deprecation.** Deprecating a manifest field follows the standard window in
  `docs/spec/09-release-and-versioning.md#deprecation`: mark deprecated for one minor
  cycle (loader warns, value still bound), then remove with a `manifest_version` bump.

## Testing Strategy

Cross-references `docs/spec/07-testing-strategy.md` (test pyramid) and
`docs/spec/04-error-model.md` (the typed errors below). RFC-0013
(`docs/rfcs/RFC-0013-testing-fuzzing-and-audit-strategy.md`) requires both accepting
and rejecting tests per component.

**Accepting tests:**

- `test_reexport_byte_identical` — export the reference checkpoint twice; assert
  identical `manifest.canonical.json`, `weights.bin`, `golden/vectors.jsonl`
  (INV-MANIFEST-01).
- `test_manifest_roundtrip_and_self_hash` — `manifest.yaml` <-> `manifest.canonical.json`
  round-trip; recomputed `serialization.canonical_json_hash` matches
  (INV-MANIFEST-06).
- `test_python_rust_parity` — every golden case agrees Python<->Rust bit-for-bit on
  all intermediates and outputs (INV-MANIFEST-05;
  `docs/spec/07-testing-strategy.md#differential-tests`).
- `test_commitments_recomputable_by_verifier` — `pwm-verifier` recomputes
  `model_commitment`/`quantization_commitment`/`planner_config_commitment` from the
  manifest alone and they equal the values in the matching `ProofArtifact.public_input`.
- `test_golden_p0_p1_p2_present` — golden vectors exist for one predictor step, a
  horizon-5 rollout, and an S-candidate plan including a tie
  (`docs/spec/07-testing-strategy.md#golden-vectors`).

**Rejecting / negative tests** (`docs/spec/07-testing-strategy.md#negative-tests`):

- `test_weight_flip_changes_model_commitment` — flip one weight leaf; assert
  `model_commitment` changes (INV-MANIFEST-02).
- `test_scale_change_changes_quant_commitment` — change one `scale_table` entry;
  assert `quantization_commitment` changes (INV-MANIFEST-03).
- `test_op_reorder_changes_model_commitment` — swap two `ops`; assert
  `model_commitment` changes (INV-MANIFEST-04).
- `test_rounding_override_changes_quant_commitment` — flip an op's `rounding`;
  assert `quantization_commitment` changes.
- `test_relation_id_mismatch_rejected` — manifest `relation_id` disagreeing with the
  proof's `relation_id` yields `VerifyError` (INV-MANIFEST-07).
- `test_export_unsupported_op_rejected` — request P4 (encoder) under V0; assert
  `ExportError::UnsupportedOp`.
- `test_export_accumulator_bound_violation` — craft an `ExportConfig` whose scales let
  a length-N int8 accumulator exceed M31 without limb decomposition; assert
  `ExportError::AccumulatorBoundViolation` (verifies the `2048*127*127 < 2^31-1` bound
  logic from §4 by crossing it).
- `test_export_dangling_reference` — manifest op references an unknown `scale_id`;
  assert `ExportError::DanglingReference`.
- `test_parity_mismatch_blocks_release` — inject a one-LSB Rust-reference divergence;
  assert `ExportError::ParityMismatch` and that CI marks the bundle `parity_failed`.

**Mutation tests** (`docs/spec/07-testing-strategy.md#mutation-tests`): mutate a byte
in `weights.bin` and assert `weights.root` recomputation detects it (Merkle leaf
sensitivity).

**CI gates** (`docs/spec/07-testing-strategy.md#ci-gates`): the `area:export` gate
runs all the above; INV-MANIFEST-01 and INV-MANIFEST-05 are release-blocking.

## Open Questions

- **OQ1.** Quantization-calibration policy (how int8/int16 scales are *chosen* from
  the checkpoint) is out of scope here (NG3). *Owner: export maintainer. Resolution:
  a follow-up RFC scoped to v0.2, or accept caller-supplied/power-of-two scales for
  V0 and defer automatic calibration to RFC-0002's numeric framing.*
- **OQ2.** Whether the optional ONNX cross-check artifact (A1) ships in V0 or is
  deferred. *Owner: export maintainer. Resolution: v0.1 design-freeze decision; if not
  resolved by freeze it defaults to deferred (structural traversal remains the sole
  source of truth regardless).*
- **OQ3.** Whether `weights.merkle` (the layer-hash aid) is shipped in the bundle or
  recomputed on demand by the verifier. *Owner: verifier maintainer. Resolution:
  RFC-0016 (`docs/rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md`), which
  owns the on-disk bundle layout.*

## References

- Founding analysis: `docs/feasibility-study.md` §0 (verdict), §4.1–§4.2 (model
  formalization and manifest), §9.1 (binding requirements), §11 (RFC-001), §12
  (milestone 0 / design freeze).
- `docs/spec/00-overview.md#v0-statement`, `#scope-and-statement-tiers`,
  `#feasibility-verdict`.
- `docs/spec/01-architecture.md#crate-layout`, `#data-flow`, `#module-boundaries`.
- `docs/spec/02-public-api.md#artifact-formats`, `#rust-public-api`.
- `docs/spec/03-data-model.md#model-manifest`, `#schema-versioning`,
  `#bounded-integers`, `#tensor-types`.
- `docs/spec/04-error-model.md#error-taxonomy`, `#failure-modes`.
- `docs/spec/06-security.md#binding-requirements`, `#trust-boundaries`.
- `docs/spec/07-testing-strategy.md#golden-vectors`, `#negative-tests`,
  `#differential-tests`, `#mutation-tests`, `#ci-gates`.
- `docs/spec/09-release-and-versioning.md#schema-versioning`,
  `#relation-versioning`, `#deprecation`, `#license`.
- Related RFCs: RFC-0000 (statement taxonomy), RFC-0002 (fixed-point numerics),
  RFC-0006 (nonlinear approximations and their committed tables), RFC-0007 (predictor
  AIR matches the exported graph), RFC-0009 (planner verifier rejections), RFC-0014
  (canonical serialization / transcript), RFC-0015 (vendoring / pinning), RFC-0016
  (artifact bundle / reproducibility).
- LeWorldModel (github.com/lucas-maes/le-wm, MIT), verified against upstream at
  2026-06-03: embed_dim/latent_dim=192, depth=6, heads=16, dim_head=64, mlp_dim=2048,
  history_size=3, ~15M params; ARPredictor with ConditionalBlock/AdaLN-zero;
  predictor FFN uses GELU, AdaLN modulation and action Embedder use SiLU; rollout
  concatenates predicted embeddings and next actions, cost is final-latent MSE to goal.
  Apache-2.0 vendored Stwo provides the M31 field module, QM31, FRI/PCS, LogUp, and
  Blake2s/Keccak256/Poseidon252 channels (contract §9b).
