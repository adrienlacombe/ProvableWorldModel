# Data Model: Manifest Schema, Types, Invariants, Schema Versioning

Status: Normative. This document is the single source of truth for every canonical
type, schema, and persisted artifact in ProvableWorldModel. All other corpus
documents reference these definitions by anchor; they do not redefine them. The
canonical Rust types reproduced here are lifted verbatim from the authoring
contract section 6 and are binding on `pwm-core`, `pwm-air`, `pwm-prover`, and
`pwm-verifier`.

This document covers, in order: the model manifest schema and its commitment
binding (`#model-manifest`), the scale table and `scale_id` semantics, the
canonical field and fixed-point types (`#bounded-integers`), tensor and weight
types (`#tensor-types`), the public input (`#public-input`), the witness
(`#witness`), the tensor memory cell (`#tensor-memory-cells`), the `relation_id`
format and immutability rule, schema versioning rules (`#schema-versioning`), and
the full set of named data-model invariants (`INV-DM-NN`).

Conventions used throughout:

- `p = 2^31 - 1` is the M31 prime. "The field" means M31 unless stated otherwise.
- A "commitment" is a 32-byte digest produced by the manifest commitment scheme
  (Blake3 / Poseidon / Merkle, declared in the manifest; see `#model-manifest`).
  The exact byte serialization that feeds every commitment is fixed by
  RFC-0014 ([docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md](../rfcs/RFC-0014-canonical-serialization-and-transcript.md)).
- "Bound by `<commitment>`" means the field's canonical bytes are part of the
  preimage hashed into that 32-byte digest, so changing the field changes the
  digest and therefore invalidates any proof carrying the old digest in its
  `PublicInput`.
- All integer arithmetic in the model is over mathematical integers `Z`; the
  field is only the carrier. The AIR proves the integer relation and
  range-checks every value so that field wraparound cannot stand in for an
  out-of-range integer. This is the load-bearing soundness property of the entire
  data model (see `#bounded-integers`, INV-DM-06, INV-DM-07).

---

## model-manifest

The manifest is the canonical, human-readable, machine-checkable artifact that
defines the exact model being proven. It is a YAML document. Its canonical byte
serialization (RFC-0014) is hashed to produce the commitments carried in every
`PublicInput`. The prover and verifier both bind to it; neither runs PyTorch.
The full export pipeline that produces the manifest is specified in RFC-0001
([docs/rfcs/RFC-0001-model-manifest-and-export-pipeline.md](../rfcs/RFC-0001-model-manifest-and-export-pipeline.md)).

The V0 reference target is the LeWorldModel predictor configuration: the V0
reference configuration targets `latent_dim/embed_dim = 192`, `history_size = 3`,
predictor `depth = 6`, `heads = 16`, `dim_head = 64`, `mlp_dim = 2048`, roughly
15M trainable parameters (verified against upstream `lucas-maes/le-wm` at
2026-06-03). V0 binds `action_encoder`, `predictor`, and `pred_proj`; it excludes
the pixel encoder, training loss, SIGReg, dataloading, the config framework, and
the CEM sampling loop. The architecture block carries enough to fully describe
those V0 components plus declared (but unproven-in-V0) placeholders for the
encoder, which are bound but not exercised by P0/P1/P2.

### Canonical manifest schema (YAML)

```yaml
# --- identity and versioning ---
manifest_version: pwm-model-manifest-v1   # schema version of THIS document
model_family: lewm                        # model family identifier
relation_id: pwm.lewm.fixed_candidate_planning.v1  # see #relation-id

# --- field and signed-integer encoding ---
field:
  base: M31                               # p = 2^31 - 1
  extension: QM31                         # degree-4 secure field (challenges only)
  signed_encoding: centered_mod_p         # x -> x mod p; range-checked separately
  max_abs_value: per_tensor               # global cap fallback; real bounds per tensor

# --- quantization policy (one active mode) ---
quantization:
  arithmetic: fixed_point
  default_rounding: nearest_ties_to_even  # or truncate_toward_zero (manifest MAY override)
  overflow_policy: reject                 # V0: reject only; wrapping is unsound
  clamp_policy: explicit                  # every clamp range is declared per op
  activation_tables_commitment: "0x..."   # 32-byte commitment over ALL lookup tables

# --- scale table (powers of two; indexed by scale_id) ---
scales:
  - { scale_id: 0, log2: 0,  dtype: i8  }  # value = q * 2^0
  - { scale_id: 1, log2: -7, dtype: i8  }  # value = q * 2^-7
  - { scale_id: 2, log2: -12, dtype: i16 }
  - { scale_id: 3, log2: -16, dtype: i32 } # accumulators / costs

# --- architecture (bound; describes the exported graph) ---
architecture:
  latent_dim: 192
  history_size: 3
  predictor:
    type: ar_transformer                  # ARPredictor (autoregressive)
    depth: 6
    heads: 16
    dim_head: 64
    mlp_dim: 2048
    block:
      modulation: adaln_zero              # ConditionalBlock, AdaLN-zero
      modulation_activation: silu         # AdaLN modulation uses SiLU
      attention: scaled_dot_product
      ffn_activation: gelu                # predictor FFN/MLP uses GELU
      positional_embedding: learned
  action_encoder:
    type: embedder
    embedder_activation: silu             # action Embedder uses SiLU
  pred_proj:
    type: mlp_or_affine
  encoder:                                # declared, bound, NOT proven in V0
    type: vit_tiny
    hidden: 192
    image_size: 224
    patch_size: 14
    proven_in: P4                         # see RFC-0011

# --- weights (visibility + commitment) ---
weights:
  visibility: public                      # public | private_committed
  commitment_scheme: blake3               # blake3 | poseidon | merkle
  root: "0x..."                           # 32-byte weights.root == QuantizedWeights.commitment

# --- planner config (bound; V0 binds the fixed-candidate rule) ---
planner:
  kind: fixed_candidate                   # fixed_candidate | cem (cem = P3/V2)
  cost: mse_goal_latent                   # MSE of final predicted latent vs goal
  tie_break: smallest_index               # smallest index attaining minimum cost
  horizon: 5                              # rollout steps (le-wm pusht plan_config)
  action_block: 5                         # actions applied per step
  # CEM-only fields (absent for fixed_candidate); see RFC-0010:
  # cem: { num_samples: 300, n_steps: 30, topk: 30, var_scale: 1.0, seed: <u64> }

# --- ops list (ordered; each op binds its scales + per-op weight commitment) ---
ops:
  - id: action_encoder.embed
    op: linear
    in_scale_id: 1
    weight_scale_id: 0
    acc_scale_id: 3
    out_scale_id: 1
    activation: silu_lookup_v1
    weight_commitment: "0x..."
  - id: predictor.block0.attn.qkv
    op: linear
    in_scale_id: 1
    weight_scale_id: 0
    acc_scale_id: 3
    out_scale_id: 1
    weight_commitment: "0x..."
  - id: predictor.block0.attn.softmax
    op: softmax_approx_v1
    in_scale_id: 1
    out_scale_id: 1
    lookup_table: softmax_table_v1
    error_bound: "<declared max abs error in out-scale ULPs>"
  - id: predictor.block0.adaln
    op: adaln_q_v1
    modulation_activation: silu_lookup_v1
    inv_std_method: newton_bounded_v1
    in_scale_id: 1
    out_scale_id: 1
  - id: predictor.block0.ffn.fc1
    op: linear
    in_scale_id: 1
    weight_scale_id: 0
    acc_scale_id: 3
    out_scale_id: 1
    activation: gelu_lookup_v1
    weight_commitment: "0x..."
  - id: pred_proj.fc
    op: linear
    in_scale_id: 1
    weight_scale_id: 0
    acc_scale_id: 3
    out_scale_id: 1
    weight_commitment: "0x..."

# --- serialization self-hash ---
serialization:
  serialization_version: pwm-serialization-v1   # RFC-0014 byte format version
  canonical_json_hash: "0x..."          # hash of the canonical serialization of THIS file
```

### Field-by-field semantics and commitment binding

The table below documents every top-level key and its load-bearing children.
The "Bound by" column states which of the three commitments carried in
`PublicInput` (`model_commitment`, `quantization_commitment`,
`planner_config_commitment`) covers the field. A field bound by no commitment is
mutable by the prover and therefore unsound; the contract forbids such fields, so
every row below names a binding commitment.

| Key | Type | Semantics | Bound by |
| --- | --- | --- | --- |
| `manifest_version` | string enum | Schema version of the manifest document itself. Bump rules in `#schema-versioning`. | `model_commitment` |
| `model_family` | string | Family identifier (`lewm`). Distinguishes export adapters. | `model_commitment` |
| `relation_id` | string `pwm.lewm.<statement>.v<N>` | The exact relation this manifest can be proven under. Hashed to the 32-byte `PublicInput.relation_id`. See `#relation-id`. | `model_commitment` |
| `field.base` | enum `M31` | Carrier field, `p = 2^31 - 1`. | `model_commitment` |
| `field.extension` | enum `QM31` | Secure field for Fiat-Shamir challenges only; never carries tensor values (INV-DM-12). | `model_commitment` |
| `field.signed_encoding` | enum `centered_mod_p` | Signed `x` is embedded as `x mod p`; the integer range is enforced by a separate range-check witness, not by the encoding. | `model_commitment` |
| `field.max_abs_value` | enum/int | Global fallback magnitude cap; per-tensor `BoundedInt` bounds always take precedence and must be tighter. | `model_commitment` |
| `quantization.arithmetic` | enum `fixed_point` | The only V0 arithmetic mode. | `quantization_commitment` |
| `quantization.default_rounding` | enum | `nearest_ties_to_even` (canonical default) or `truncate_toward_zero`. Exactly one active mode per manifest (INV-DM-09). | `quantization_commitment` |
| `quantization.overflow_policy` | enum `reject` | V0 rejects on overflow; never wraps. Maps to `OverflowPolicy::Reject`. | `quantization_commitment` |
| `quantization.clamp_policy` | enum `explicit` | Every clamp range is declared per op; no implicit clamps. | `quantization_commitment` |
| `quantization.activation_tables_commitment` | `[u8;32]` | Commitment over the concatenation of all lookup tables (GELU/SiLU/softmax/inv-sqrt) in canonical order. Binds every approximation. | `quantization_commitment` |
| `scales[]` | list | The scale table. See `#scale-table`. Each entry declares `scale_id`, `log2`, `dtype`. | `quantization_commitment` |
| `architecture.latent_dim` | u32 | `192` (V0 target). Width of latent vectors. | `model_commitment` |
| `architecture.history_size` | u32 | `3` (V0 target). Number of past latents in the predictor window. | `model_commitment` |
| `architecture.predictor.*` | object | Predictor topology: `type`, `depth=6`, `heads=16`, `dim_head=64`, `mlp_dim=2048`, block modulation/activation details. `ffn_activation: gelu`, `modulation_activation: silu` (these differ; see correction note below). | `model_commitment` |
| `architecture.action_encoder.*` | object | `type: embedder`, `embedder_activation: silu`. | `model_commitment` |
| `architecture.pred_proj.*` | object | `type: mlp_or_affine`. Projects predictor output back to latent space. | `model_commitment` |
| `architecture.encoder.*` | object | ViT-tiny encoder; declared and bound but not proven until P4 (RFC-0011). | `model_commitment` |
| `weights.visibility` | enum | `public` (weights are in the artifact) or `private_committed` (only the root is public; weights are a private witness). | `model_commitment` |
| `weights.commitment_scheme` | enum | `blake3 | poseidon | merkle`. Defines how `weights.root` and all 32-byte commitments are computed. | `model_commitment` |
| `weights.root` | `[u8;32]` | Commitment over all weight/bias tensors in canonical op order. Equals `QuantizedWeights.commitment` (INV-DM-04). | `model_commitment` |
| `planner.kind` | enum | `fixed_candidate` (V0) or `cem` (P3/V2). | `planner_config_commitment` |
| `planner.cost` | enum | `mse_goal_latent`: cost is the MSE of the final predicted latent against the goal latent. | `planner_config_commitment` |
| `planner.tie_break` | enum `smallest_index` | The smallest index attaining the minimum cost wins (deterministic). | `planner_config_commitment` |
| `planner.horizon` | u32 | Rollout length, `5` for the le-wm PushT task. | `planner_config_commitment` |
| `planner.action_block` | u32 | Actions applied per rollout step, `5`. | `planner_config_commitment` |
| `planner.cem.*` | object | CEM-only; absent for `fixed_candidate`. `num_samples=300`, `n_steps=30`, `topk=30`, `var_scale=1.0`, `seed`. Used by RFC-0010 only. | `planner_config_commitment` |
| `ops[].id` | string | Stable operator identifier; defines trace ordering and weight lookup keys. | `model_commitment` |
| `ops[].op` | enum | Operator kind (`linear`, `softmax_approx_v1`, `adaln_q_v1`, `gelu_lookup_v1`, ...). | `model_commitment` |
| `ops[].*_scale_id` | u32 | `in_scale_id`, `weight_scale_id`, `acc_scale_id`, `out_scale_id` index the scale table. | `quantization_commitment` |
| `ops[].activation` / `ops[].lookup_table` / `ops[].error_bound` | string/int | Per-op approximation identifier, table name, and declared maximum error (in output-scale ULPs). | `quantization_commitment` |
| `ops[].weight_commitment` | `[u8;32]` | Per-op subtree commitment; the union over ops reproduces `weights.root`. | `model_commitment` |
| `serialization.serialization_version` | string enum | RFC-0014 byte-format version. | `model_commitment` |
| `serialization.canonical_json_hash` | `[u8;32]` | Self-hash of the manifest's own canonical serialization (excluding this field). Detects in-place tampering before commitments are even recomputed. | self / `model_commitment` |

Correction note (verified against upstream at 2026-06-03): LeWorldModel
activation functions are NOT uniform. The predictor FFN/MLP uses GELU; the AdaLN
modulation and the action Embedder use SiLU. The manifest states the activation
per module (`predictor.block.ffn_activation: gelu`,
`predictor.block.modulation_activation: silu`,
`action_encoder.embedder_activation: silu`) and per op (`ops[].activation`). An
exporter that emits a single uniform activation is rejected by the parity test in
RFC-0001.

### The model commitment must bind

The contract is explicit: anything not bound by a commitment is mutable by the
prover and therefore unsound. The `model_commitment` MUST bind the full following
list. Items split across `quantization_commitment` and
`planner_config_commitment` are noted; the union of the three commitments covers
everything, and a `PublicInput` carries all three.

```text
architecture                  -> model_commitment
weights                       -> model_commitment (weights.root)
biases                        -> model_commitment (carried in weight tensors)
quantization scales           -> quantization_commitment (scales[])
rounding rules                -> quantization_commitment (default_rounding)
lookup tables                 -> quantization_commitment (activation_tables_commitment)
activation approximations     -> quantization_commitment (per-op + tables)
normalization approximations  -> quantization_commitment (adaln/inv_std method + tables)
tensor shapes                 -> model_commitment (architecture + ops + weight tensor shapes)
planner config                -> planner_config_commitment
relation version              -> model_commitment (relation_id)
serialization version         -> model_commitment (serialization.serialization_version)
```

Splitting the binding across three commitments lets the verifier reject with a
precise reason (wrong model vs wrong quantization vs wrong planner config; see
[docs/spec/04-error-model.md#verifier-rejections](04-error-model.md#verifier-rejections))
and lets a planner-config change be made without re-hashing the entire weight
set. The three commitments are independent inputs to the Fiat-Shamir transcript
(RFC-0014); a proof is valid only if all three match the manifest the verifier
holds.

FAILURE MODES (manifest):

| Condition | System response |
| --- | --- |
| `serialization.canonical_json_hash` does not match the recomputed self-hash | Manifest load fails before any proving; `ManifestError::SelfHashMismatch` ([docs/spec/04-error-model.md#error-taxonomy](04-error-model.md#error-taxonomy)). |
| `relation_id` not in the verifier's supported set | Verifier rejects: `VerifyError::UnsupportedRelation`. |
| A field bound by `model_commitment` differs from the prover's manifest | `model_commitment` mismatch; verifier rejects. |
| A `scale_id` referenced by an op is absent from `scales[]` | Manifest validation fails (INV-DM-11); `ManifestError::UndeclaredScale`. |
| `weights.root` does not equal the recomputed `QuantizedWeights.commitment` | Load fails (INV-DM-04); `ManifestError::WeightRootMismatch`. |
| `planner.kind: cem` under a V0 relation_id | Rejected at export; CEM requires a P3 relation_id (RFC-0010). |

---

## scale-table

The scale table is the declared list of quantization scales, indexed by
`scale_id`. Every `Tensor` and every op references scales only by `scale_id`; raw
floating-point scale factors never appear in any proven artifact, because they are
not field-representable and would reintroduce floating-point ambiguity.

Scales are powers of two wherever possible. A power-of-two scale makes
requantization a right shift, which the AIR proves with exact quotient/remainder
constraints (RFC-0002,
[docs/rfcs/RFC-0002-fixed-point-arithmetic-over-m31.md](../rfcs/RFC-0002-fixed-point-arithmetic-over-m31.md)),
avoiding any field division.

```text
scale entry:
  scale_id : u32        index used everywhere
  log2     : i32        value = quantized_integer * 2^log2  (the real scale = 2^log2)
  dtype    : enum       i8 | i16 | i32  (the integer container the values live in)
```

Semantics:

- The real number represented by an integer `q` at scale `scale_id` is
  `q * 2^(scales[scale_id].log2)`. Because `log2` is typically negative
  (fractional scales), larger `scale_id` accumulator scales (e.g. `log2 = -16`)
  carry more fractional precision.
- `dtype` declares the integer container and therefore the default range that the
  associated `BoundedInt` bounds must lie within: `i8` is `[-128, 127]`, `i16` is
  `[-32768, 32767]`, `i32` is `[-2^31, 2^31 - 1]`. The per-tensor `BoundedInt.lo`
  and `BoundedInt.hi` MAY be tighter than the `dtype` range but MUST NOT be wider
  (INV-DM-05, INV-DM-11).
- Requantization between scales `s_in -> s_out` is a shift by
  `r = scales[s_in].log2 - scales[s_out].log2` (when `r >= 0`, a right shift by
  `r`; the exporter is required to keep `r >= 0` for all requant ops so the AIR
  only ever proves right shifts). The shift amount is derived from the table, not
  stored per op, which keeps the manifest minimal and makes the shift verifiable.
- A non-power-of-two scale MAY appear only if `op` explicitly declares a
  multiply-then-shift requant table; such scales set `log2` to the nearest
  representable shift and record the multiplier in the op. V0 prefers
  power-of-two scales and treats multiplier-based requant as an RFC-0002 opt-in.

The entire `scales[]` block is bound by `quantization_commitment`. Changing any
`log2`, `dtype`, or the ordering changes the commitment.

FAILURE MODES (scales):

| Condition | System response |
| --- | --- |
| Op references `scale_id` not present in `scales[]` | `ManifestError::UndeclaredScale` (INV-DM-11). |
| A tensor's `BoundedInt` bound exceeds its scale's `dtype` range | Export/manifest validation fails (INV-DM-05). |
| A requant op implies a negative shift (`r < 0`) | `ManifestError::NegativeRequantShift`; exporter must reorder scales. |

---

## bounded-integers

This section presents the canonical field and fixed-point types verbatim from
contract section 6.1. These types live in `pwm-core` and are the foundation of
every value in the system. The arithmetic semantics (add/sub/mul/accumulate/
requantize/clamp/compare) are specified in RFC-0002; this document defines the
types and their invariants.

```rust
/// Base field element, M31: integers mod p = 2^31 - 1.
pub struct M31(u32);            // canonical value in [0, p)
/// Degree-4 extension ("secure field") used for Fiat-Shamir challenges and
/// soundness; NOT used to represent quantized tensor values.
pub struct QM31([M31; 4]);

/// A bounded signed integer with a declared inclusive range, embedded into M31
/// via centered encoding. `value` is the mathematical integer; the AIR carries
/// `value mod p` plus a range-check witness proving lo <= value <= hi.
pub struct BoundedInt {
    pub value: i64,             // mathematical signed value
    pub lo: i64,                // inclusive lower bound (declared)
    pub hi: i64,                // inclusive upper bound (declared)
}

pub enum Rounding { NearestTiesToEven, TruncateTowardZero }
pub enum OverflowPolicy { Reject }     // V0: reject only; wrapping is unsound
```

Field semantics:

- `M31` holds a canonical residue in `[0, p)`, `p = 2^31 - 1` (verified against
  upstream stwo at 2026-06-03, `crates/stwo/src/core/fields/m31.rs`). The single
  `u32` is always reduced; non-canonical representations are a `pwm-core` bug,
  not a permitted state.
- `QM31` is the degree-4 secure-field extension used for Fiat-Shamir challenges
  and soundness amplification only. It NEVER carries a quantized tensor value
  (INV-DM-12). Tensor values are `M31` (carrying a `BoundedInt`'s `value mod p`);
  the secure field exists purely for the proof system, not the model arithmetic.

`BoundedInt` semantics:

- `value` is the mathematical signed integer the model computes with. It is the
  source of truth; the field element carried in the trace is `value mod p` under
  centered encoding (a negative `value` maps to `p + value`).
- `lo` and `hi` are the inclusive declared bounds. The AIR emits a range-check
  witness (`RangeWitness`, see `#witness`) proving `lo <= value <= hi`. The
  bounds are not advisory: the proof is unsound without them, because a field
  element that wrapped past `p` could satisfy an arithmetic constraint while
  representing an out-of-range integer (INV-DM-06, INV-DM-07).
- Bounds are declared by the exporter from the op's scale `dtype` and the static
  worst-case magnitude analysis (RFC-0002 / RFC-0005). They are tightened where
  static analysis permits, never loosened beyond the `dtype` range.

`Rounding`:

- `NearestTiesToEven` is the canonical default. `TruncateTowardZero` is the only
  permitted override. The manifest declares exactly one mode; both the Python and
  Rust references and the AIR enforce it bit-for-bit (INV-DM-09). The mode is
  bound by `quantization_commitment`.

`OverflowPolicy`:

- V0 supports only `Reject`. If any accumulator or intermediate would exceed its
  declared bound, the reference inference fails and no proof is produced. Wrapping
  is never permitted because field wraparound is the canonical unsoundness vector
  for integer ML in a finite field.

### The safe-accumulation bound and limb decomposition

A signed-`int8` dot product of length `N` has worst-case magnitude
`N * 127 * 127`. For the V0 MLP width `N = 2048`:

```text
2048 * 127 * 127 = 33,032,192 < 2^31 - 1 = 2,147,483,647
```

So an `int8 x int8` MLP accumulator of length 2048 fits in a single signed M31
value, and the accumulator scale (`scale_id = 3`, `dtype: i32`) holds it without
overflow. This is the basis for INV-DM-08: accumulators that provably stay within
the safe M31 interval are carried as a single `M31`; accumulators whose declared
bound can exceed that interval (wider activations, wider weights, long reductions,
or summed-square costs over int16 latents) MUST be limb-decomposed
(`acc = acc_lo + 2^k * acc_hi`) with an independent range check on each limb, per
RFC-0002 and RFC-0005
([docs/rfcs/RFC-0005-linear-matmul-and-requantization-components.md](../rfcs/RFC-0005-linear-matmul-and-requantization-components.md)).
The exporter computes the worst-case bound from tensor metadata and selects
single-cell vs limb representation; the choice is bound by the manifest (op
scales + shapes) and machine-checked, never left to the prover.

FAILURE MODES (bounded integers):

| Condition | System response |
| --- | --- |
| `value < lo` or `value > hi` for any `BoundedInt` | Reference inference fails (`OverflowPolicy::Reject`); the AIR's range check would also reject (INV-DM-06). |
| An accumulator exceeds `2^31 - 1` but its field residue collides with an in-range value | Range-check witness on the limb decomposition rejects (INV-DM-07, INV-DM-08). |
| A value declared as a single `M31` accumulator can statically exceed the safe interval | Export/AIR-build error; forces limb decomposition (INV-DM-08). |
| Manifest declares two rounding modes or none | `ManifestError::AmbiguousRounding` (INV-DM-09). |

---

## tensor-types

This section presents the canonical tensor and quantized-weight types verbatim
from contract section 6.2. They live in `pwm-core`.

```rust
/// Dense row-major tensor of bounded integers with a per-tensor scale id.
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
```

`Tensor` semantics:

- `tensor_id` is a stable identifier unique within a proof. It keys the tensor in
  the tensor-memory relation (`#tensor-memory-cells`) so that reads and writes can
  be matched by `(tensor_id, index)`.
- `shape` is row-major. Up to 4 dimensions are addressable in the tensor-memory
  cell (`TensorCell.index: [u32; 4]`); tensors with fewer dims pad trailing index
  slots with 0. Tensors with more than 4 dims are not representable in V0 and must
  be reshaped by the exporter (RFC-0004).
- `scale_id` indexes `scales[]`. Every element of `data` is interpreted at this
  scale (INV-DM-10). A single tensor has exactly one scale (per-tensor
  quantization, the V0 policy); per-channel quantization is out of V0 scope.
- `data` is the dense row-major payload. INV-DM-01 requires
  `len(data) == product(shape)`. Each element is a `BoundedInt` whose bounds must
  be consistent with the scale's `dtype` (INV-DM-05).

`QuantizedWeights` semantics:

- `commitment` is the 32-byte root over all weight and bias tensors in canonical
  op order. It MUST equal the manifest `weights.root` (INV-DM-04); the loader
  checks this before any trace is built.
- `tensors` holds one entry per weight and per bias tensor, in the op order
  declared by `ops[]`. The mapping from op to its weight/bias tensor is by
  position and `tensor_id`, both committed.
- When `weights.visibility == private_committed`, `QuantizedWeights` is a private
  witness (it lives in `Witness.model_weights`, never in the public artifact), and
  only `commitment` is public via `model_commitment`. When `visibility == public`,
  the tensors ship in the artifact and `Witness.model_weights` is `None`.

FAILURE MODES (tensors):

| Condition | System response |
| --- | --- |
| `len(data) != product(shape)` | `pwm-core` constructor rejects (INV-DM-01); never enters a trace. |
| Element bound wider than `scales[scale_id].dtype` | Validation fails (INV-DM-05). |
| `scale_id` not in `scales[]` | `ManifestError::UndeclaredScale` (INV-DM-11). |
| `QuantizedWeights.commitment != weights.root` | Loader rejects (INV-DM-04). |
| Tensor with `len(shape) > 4` | Export error; must be reshaped (RFC-0004). |

---

## public-input

This section presents the canonical public input verbatim from contract section
6.3. The `PublicInput` is the complete set of values the verifier sees and binds
to; it carries the three commitments and the statement-specific public data. Its
canonical digest (RFC-0014) is the public-input binding that anchors the entire
proof. The source feasibility study used `FieldElement` and `Vec<FieldElement>`;
the canonical type is `M31` (and `Vec<M31>`), and `selected_cost` is a
`BoundedInt`, per the corrected signatures below.

```rust
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
```

| Field | Type | Semantics | Required for |
| --- | --- | --- | --- |
| `relation_id` | `[u8;32]` | Hash of the manifest's `relation_id` string. The verifier rejects unless it supports this exact id (`#relation-id`). | all |
| `model_commitment` | `[u8;32]` | Binds architecture, weights, shapes, relation version, serialization version. See `#model-manifest`. | all |
| `quantization_commitment` | `[u8;32]` | Binds scales, rounding, overflow/clamp policy, lookup tables, approximations. | all |
| `planner_config_commitment` | `[u8;32]` | Binds planner kind, cost, tie-break, horizon, action block, and CEM params when present. | all (planner-free statements bind the empty/`fixed_candidate` config) |
| `statement_type` | `StatementType` | Discriminates P0/P1/P2/P3/P4. The verifier binds this; a P0 proof submitted as P1 is rejected (see RFC-0000). | all |
| `latent_history_commitment` | `Option<[u8;32]>` | Commitment to the latent history when it is private; `None` when public. Exactly one of this and `latent_history_public` is `Some` (INV-DM-13). | P0/P1/P2/P3 |
| `latent_history_public` | `Option<Vec<M31>>` | The latent history in the clear when public. | P0/P1/P2/P3 |
| `goal_latent_commitment` | `Option<[u8;32]>` | Commitment to the goal latent when private; `None` when public or when the statement has no goal (P0/P1). | P2/P3 |
| `goal_latent_public` | `Option<Vec<M31>>` | Goal latent in the clear when public. | P2/P3 |
| `candidate_actions_commitment` | `Option<[u8;32]>` | Commitment to the candidate action set when private. | P2 (private actions) |
| `candidate_actions_public` | `Option<Vec<M31>>` | Candidate actions in the clear when public. For P0/P1 this carries the single action sequence. | P0/P1/P2 |
| `claimed_output_commitment` | `[u8;32]` | Commitment to the claimed outputs: predicted next latent (P0), trajectory (P1), or trajectory root / selected result (P2). The proof binds the outputs to this digest. | all |
| `selected_index` | `Option<u32>` | The chosen candidate index (P2/P3 only). Must be `< S` (number of candidates), checked by the verifier. | P2/P3 |
| `selected_cost` | `Option<BoundedInt>` | The cost of the selected candidate (P2/P3 only). A `BoundedInt`, range-checked; equals `cost[selected_index]` and is `<=` every other cost (RFC-0009). | P2/P3 |

Commitment-vs-public duality (INV-DM-13): for each of latent history, goal latent,
and candidate actions, exactly one of the `_commitment` / `_public` pair is
populated. Public data is hashed directly into the public-input digest; private
data is represented only by its commitment, with the cleartext supplied in the
`Witness`. This is the privacy seam: V0 is a succinct validity proof, not
zero-knowledge, and making an input private via the commitment path does NOT by
itself hide the witness in the trace (see
[docs/spec/06-security.md#privacy-and-zk](06-security.md#privacy-and-zk)).

FAILURE MODES (public input):

| Condition | System response |
| --- | --- |
| Both `_commitment` and `_public` set, or neither, for the same input | Verifier rejects (INV-DM-13); malformed public input. |
| `statement_type` inconsistent with `relation_id` | Verifier rejects: `VerifyError::StatementRelationMismatch`. |
| `selected_index >= S` | Verifier rejects (out-of-range selection; RFC-0009). |
| `selected_cost.value` outside its declared `[lo, hi]` | Verifier rejects (range check fails). |
| Any of the three commitments differs from the verifier's manifest | Verifier rejects with the specific commitment-mismatch error ([docs/spec/04-error-model.md#verifier-rejections](04-error-model.md#verifier-rejections)). |

---

## witness

This section presents the canonical witness types verbatim from contract section
6.3. The `Witness` is the prover-side private data; it is never serialized into
the public `ProofArtifact`. The verifier never receives it.

```rust
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
```

| Field | Type | Semantics | Present for |
| --- | --- | --- | --- |
| `model_weights` | `Option<QuantizedWeights>` | The quantized weights when `private_committed`; `None` when weights are public (they are in the artifact, not the witness). | private-weight proofs |
| `latent_history` | `Tensor` | The latent history fed to the predictor. Its commitment or public form appears in `PublicInput`. | all |
| `goal_latent` | `Option<Tensor>` | The goal latent for cost computation. | P2/P3 |
| `candidate_actions` | `Option<Tensor>` | The candidate action sequences (P2) or the single action sequence (P0/P1). | P0/P1/P2 |
| `action_embeddings` | `Tensor` | Output of `ActionEncoder_Q` over the actions. Witnessed so the predictor trace can consume it. | all |
| `predictor_activations` | `Vec<Tensor>` | All intermediate activations of the predictor blocks (attention scores, post-AdaLN values, FFN intermediates) needed to fill the trace. | all |
| `rollout_trajectory` | `Tensor` | The autoregressive trajectory of predicted latents over the horizon. | P1/P2/P3 |
| `costs` | `Option<Vec<BoundedInt>>` | Per-candidate MSE costs (length `S`). Each is range-checked. | P2/P3 |
| `argmin_witness` | `Option<ArgminWitness>` | The selection witness; see below. | P2/P3 |
| `range_witnesses` | `Vec<RangeWitness>` | One per bounded value (or batched table) proving `lo <= value <= hi` via the LogUp range relation (RFC-0003). | all |
| `lookup_witnesses` | `Vec<LookupWitness>` | Activation/normalization table lookups (GELU/SiLU/softmax/inv-sqrt) with multiplicities (RFC-0003, RFC-0006). | all with nonlinearities |

`ArgminWitness` semantics:

- `selected_index` equals `PublicInput.selected_index` and indexes the chosen
  candidate.
- `diffs[s] = cost_s - selected_cost` for every candidate `s`. Each `diff` is a
  `BoundedInt` proven `>= 0`, which establishes `selected_cost <= cost_s` for all
  `s` (the minimum condition). The tie-break (smallest index attaining the
  minimum) is enforced separately for all `s < selected_index` via the
  `cost_s - selected_cost - 1 >= 0` range check (RFC-0009,
  [docs/rfcs/RFC-0009-fixed-candidate-planner-proof.md](../rfcs/RFC-0009-fixed-candidate-planner-proof.md)).
  Proving the minimum without the tie-break is unsound: a different index could
  attain the same minimum.

### Auxiliary witness types (RangeWitness, LookupWitness)

The canonical `Witness` references `RangeWitness` and `LookupWitness`. These are
owned in detail by the range/lookup infrastructure (RFC-0003,
[docs/rfcs/RFC-0003-range-check-and-lookup-infrastructure.md](../rfcs/RFC-0003-range-check-and-lookup-infrastructure.md));
their schemas are reproduced here so this document is self-contained.

```rust
/// Proof that a value lies in a declared bounded range, via the LogUp range
/// relation against a preprocessed range table. Multiplicities are checked.
pub struct RangeWitness {
    pub table_id: u32,          // which range table (u8, i8, u16, bounded-limb)
    pub value: M31,             // the value being range-checked (field residue)
    pub multiplicity: u32,      // LogUp multiplicity for this (table, value)
}

/// Proof that an (input, output) pair is a row of a committed activation /
/// normalization lookup table, via the LogUp lookup relation.
pub struct LookupWitness {
    pub table_id: u32,          // GELU_Q | SiLU_Q | Softmax_Q | InvSqrt_Q, etc.
    pub input: M31,             // quantized input at the table's in-scale
    pub output: M31,            // quantized output at the table's out-scale
    pub multiplicity: u32,      // LogUp multiplicity for this row
}
```

- `table_id` keys a preprocessed table whose contents are committed by
  `quantization_commitment` (range tables) or
  `activation_tables_commitment` (activation/normalization tables). A witness
  that references a table not in the commitment is rejected.
- `multiplicity` is the LogUp multiplicity: the number of times the row is used.
  The interaction-trace claimed sum reconciles multiplicities against the table;
  an inconsistent claimed sum is rejected (RFC-0003). LogUp is verified against
  upstream stwo at 2026-06-03 (`constraint-framework/src/logup.rs`).

FAILURE MODES (witness):

| Condition | System response |
| --- | --- |
| `model_weights` present but `weights.visibility == public` (or vice versa) | Trace builder rejects; the visibility flag and the witness presence must agree. |
| A `RangeWitness.value` outside its `table_id` range | Range relation rejects; verifier fails (INV-DM-06). |
| A `LookupWitness` row not in the committed table | Lookup relation rejects; verifier fails. |
| `argmin_witness.diffs[s] < 0` for any `s` | Range check on `diff` rejects; selected candidate is not minimal (RFC-0009). |
| Tie-break `cost_s - selected_cost - 1 < 0` for some `s < selected_index` | Tie-break range check rejects (RFC-0009). |
| `len(costs) != S` or `len(diffs) != S` | Trace builder rejects; planner witness malformed. |

---

## tensor-memory-cells

This section presents the canonical tensor-memory cell verbatim from contract
section 6.4. It lives in `pwm-air`. The cell is the unit of the tensor-memory
relation that wires operator outputs to subsequent operator inputs across the NN
graph; the relation itself (read/write consistency via permutation/multiset
argument, broadcast/reshape/transpose/concat/slice rules) is owned by RFC-0004
([docs/rfcs/RFC-0004-tensor-memory-and-wiring-air.md](../rfcs/RFC-0004-tensor-memory-and-wiring-air.md)).

```rust
pub struct TensorCell {
    pub tensor_id: u32,
    pub index: [u32; 4],   // up to 4 dims; unused dims = 0
    pub value: M31,
    pub scale_id: u32,
    pub time: u32,         // write ordering for read/write consistency
}
```

| Field | Type | Semantics |
| --- | --- | --- |
| `tensor_id` | u32 | Matches the producing `Tensor.tensor_id`; identifies which tensor this cell belongs to. |
| `index` | `[u32; 4]` | Row-major multi-index into the tensor. Tensors with `< 4` dims set trailing slots to 0. The pair `(tensor_id, index)` is the cell address. |
| `value` | `M31` | The field residue of the cell's `BoundedInt` value (centered encoding). The integer bound is enforced by the value's `RangeWitness`, not stored in the cell. |
| `scale_id` | u32 | The scale of this cell; equals the owning tensor's `scale_id`. A read that expects a different `scale_id` is rejected (INV-DM-14). Carrying the scale per cell lets the wiring relation reject scale mismatches without consulting the manifest mid-proof. |
| `time` | u32 | A monotonic write-ordering tag. Each cell is written exactly once at a `time`; reads reference the write whose `(tensor_id, index)` matches. The permutation/multiset argument uses `time` to enforce that every read corresponds to a unique prior write (RFC-0004). |

Relation summary (full rules in RFC-0004):

- Each `(tensor_id, index)` is written exactly once (one producer per cell).
- Every read of `(tensor_id, index)` must match a prior write or a committed
  static constant (broadcast constants are declared in the manifest/preprocessed
  trace, never invented by the prover).
- Reshape preserves the multiset of `value`s; transpose preserves the index
  mapping; concat/slice prove their index mappings. These are enforced by the
  wiring AIR, not by this type.

FAILURE MODES (tensor memory):

| Condition | System response |
| --- | --- |
| A read with no matching write or static constant | Permutation/multiset argument fails; verifier rejects (RFC-0004). |
| Two writes to the same `(tensor_id, index)` | Multi-producer violation; rejected. |
| Read `scale_id` differs from the cell's `scale_id` | Scale mismatch; rejected (INV-DM-14). |
| Broadcast without manifest permission | Rejected (no committed broadcast pattern). |

---

## relation-id

The `relation_id` is the immutable identifier of the exact arithmetic relation a
proof attests to. Its format is:

```text
pwm.lewm.<statement>.v<N>
```

- `pwm` is the project namespace.
- `lewm` is the model family (matches `model_family`).
- `<statement>` is the proof statement slug.
- `v<N>` is the relation version (monotonic, starts at `v1`).

Canonical examples (one per supported statement tier):

| `relation_id` | Statement | First release |
| --- | --- | --- |
| `pwm.lewm.predictor_step.v1` | P0: one latent predictor step | V0.2 |
| `pwm.lewm.rollout.v1` | P1: latent rollout | V0.2 |
| `pwm.lewm.fixed_candidate_planning.v1` | P2: fixed-candidate planner | V1.0 (headline) |
| `pwm.lewm.cem_planning.v1` | P3: full CEM planner | Future (V2) |
| `pwm.lewm.pixel_to_plan.v1` | P4: pixel-to-plan end-to-end | Future (V3) |

Immutability rule (load-bearing soundness property): a `relation_id` is
permanent. Any semantic change to the relation — a different rounding mode, a new
activation approximation, a changed argmin tie-break, an added or removed bound
input, a different field encoding, or any change to what the relation binds —
mints a new `relation_id` (typically by incrementing `<N>`). A proof is valid only
for its declared `relation_id`; the verifier rejects a proof whose `relation_id`
it does not support. This rule is locked by RFC-0000
([docs/rfcs/RFC-0000-security-model-and-statement-taxonomy.md](../rfcs/RFC-0000-security-model-and-statement-taxonomy.md))
and its release/versioning consequences are owned by
[docs/spec/09-release-and-versioning.md#relation-versioning](09-release-and-versioning.md#relation-versioning).
The mapping from `relation_id` to its definition is itself part of RFC-0000;
RFC-0000 is the authority for what each id means and binds.

INV-DM-15 makes this checkable: the manifest `relation_id` string, the hashed
32-byte `PublicInput.relation_id`, and the verifier's supported-relation set must
all agree.

FAILURE MODES (relation_id):

| Condition | System response |
| --- | --- |
| Verifier does not support the proof's `relation_id` | `VerifyError::UnsupportedRelation`; reject. |
| Manifest semantics changed but `relation_id` not bumped | Caught at export review / parity test; shipping it is a soundness defect (RFC-0000). |
| `relation_id` string and hashed `PublicInput.relation_id` disagree | Reject (INV-DM-15). |

---

## schema-versioning

ProvableWorldModel carries four independent version axes. Each governs a distinct
artifact and has its own bump rule. Conflating them is a common source of subtle
incompatibility, so they are kept separate. The release-level policy (semver,
deprecation windows, MSRV, support windows) is owned by
[docs/spec/09-release-and-versioning.md#semver](09-release-and-versioning.md#semver);
this section defines the data-model bump rules and compatibility semantics.

| Axis | Where declared | Governs | Bump rule |
| --- | --- | --- | --- |
| `manifest_version` | manifest top-level (`pwm-model-manifest-v1`) | The set and meaning of manifest keys | Bump on any non-backward-compatible change to manifest structure (new required key, removed key, changed key semantics). Adding an optional key with a safe default is a minor revision and does not require a new manifest_version if the canonical serialization and self-hash rules accommodate it; otherwise bump. |
| `artifact_version` | `ProofArtifact.artifact_version: u32` | The on-disk proof bundle layout | Bump on any change to the `ProofArtifact` byte layout or its field set. The verifier rejects an `artifact_version` it does not recognize. |
| `relation_id` version `v<N>` | manifest `relation_id` | The proven arithmetic relation | Bump (`v<N>` -> `v<N+1>`) on any semantic change to the relation. Immutable per id; see `#relation-id`. This is the strictest axis. |
| `serialization_version` | manifest `serialization.serialization_version` (`pwm-serialization-v1`) | The canonical byte serialization that feeds all hashes/commitments and the Fiat-Shamir transcript | Bump on any change to canonical byte ordering, encoding, or padding. Because every commitment depends on it, a serialization bump changes all commitments and therefore requires a relation_id bump as well. Owned by RFC-0014 ([docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md](../rfcs/RFC-0014-canonical-serialization-and-transcript.md)). |

Compatibility matrix and dependency ordering:

```text
serialization_version  --(determines bytes of)-->  all commitments
                        --(any change forces)----->  relation_id bump

relation_id            --(pins)------------------->  manifest semantics + serialization
                        --(must match)------------>  verifier supported set

manifest_version       --(structures)------------>  the manifest the commitments hash
artifact_version       --(wraps)----------------->  PublicInput + Proof on disk
```

Bump-rule consequences and what stays compatible:

- A `manifest_version` bump WITHOUT a relation semantic change is possible only
  when the new structure serializes to the identical canonical bytes for the same
  model (pure cosmetic/key-renaming with a compatibility shim). In practice almost
  any `manifest_version` bump that changes bytes forces a `serialization_version`
  bump and hence a `relation_id` bump. Treat manifest-version bumps as relation
  bumps unless a parity test proves byte identity.
- An `artifact_version` bump is the cheapest: it changes only how the bundle is
  framed on disk, not the proven relation. A verifier MAY support multiple
  `artifact_version`s simultaneously to ease migration; it MUST support only
  `relation_id`s it can verify.
- A `serialization_version` bump is the most expensive: it invalidates every
  prior commitment and proof for the affected models because the hashed bytes
  change. It MUST coincide with a `relation_id` bump (INV-DM-16).
- Old proofs remain verifiable as long as the verifier retains the corresponding
  `relation_id`, `serialization_version`, and `artifact_version` decoders.
  Dropping support for any of these is a breaking change governed by the
  deprecation policy in
  [docs/spec/09-release-and-versioning.md#deprecation](09-release-and-versioning.md#deprecation).

FAILURE MODES (versioning):

| Condition | System response |
| --- | --- |
| Verifier lacks a decoder for the artifact's `artifact_version` | `VerifyError::UnsupportedArtifactVersion`; reject. |
| `serialization_version` changed without a `relation_id` bump | INV-DM-16 violation; commitments silently change meaning. Caught by RFC-0014 parity tests; shipping it is a soundness defect. |
| `manifest_version` unknown to the loader | `ManifestError::UnsupportedManifestVersion`; load fails. |
| Two artifacts claim the same `relation_id` but different `serialization_version` | Reject; a relation pins exactly one serialization (INV-DM-16). |

---

## invariants

Every invariant below is checkable and is enforced at the stated point (exporter,
`pwm-core` constructor, AIR, or verifier). Cross-cutting failure responses are in
the per-section failure tables and in
[docs/spec/04-error-model.md#failure-modes](04-error-model.md#failure-modes).

| ID | Invariant | Enforced at |
| --- | --- | --- |
| INV-DM-01 | For every `Tensor`, `len(data) == product(shape)`. | `pwm-core` `Tensor` constructor |
| INV-DM-02 | Every `Tensor.tensor_id` is unique within a single proof's tensor set. | trace builder |
| INV-DM-03 | A `Tensor` addresses at most 4 dimensions; `len(shape) <= 4`. | `pwm-core` / exporter |
| INV-DM-04 | `QuantizedWeights.commitment == weights.root` (manifest) `== hash(canonical-serialization(tensors))` under the declared commitment scheme. | loader |
| INV-DM-05 | For every `BoundedInt` in a tensor, `[lo, hi]` is within the range of the tensor scale's `dtype` (e.g. `i8 -> [-128,127]`); never wider. | exporter / validation |
| INV-DM-06 | Every `BoundedInt` satisfies `lo <= value <= hi`, proven by a `RangeWitness`. | AIR range relation |
| INV-DM-07 | The field residue `value mod p` never substitutes for an out-of-range integer: every value interpreted as an integer carries a range check, so wraparound cannot satisfy a constraint. | AIR (range + wiring) |
| INV-DM-08 | Any accumulator whose declared bound can exceed the safe M31 interval (`2048 * 127 * 127 = 33,032,192 < 2^31 - 1` is the int8/len-2048 reference; wider configs exceed it) is limb-decomposed with a range check on each limb; accumulators provably inside the safe interval may be a single `M31`. | exporter / AIR (RFC-0002, RFC-0005) |
| INV-DM-09 | A manifest declares exactly one active rounding mode (`NearestTiesToEven` default, `TruncateTowardZero` override); Python ref, Rust ref, and AIR all enforce that single mode bit-for-bit. | manifest validation + parity tests |
| INV-DM-10 | Every element of a `Tensor` is interpreted at the tensor's single `scale_id` (per-tensor quantization). | `pwm-core` / AIR |
| INV-DM-11 | Every `scale_id` referenced by any tensor or op indexes a declared entry in `scales[]`. | manifest validation |
| INV-DM-12 | `QM31` is used only for Fiat-Shamir challenges and soundness; no quantized tensor value is ever a `QM31`. Tensor values are `M31`. | type system + review |
| INV-DM-13 | For each of latent history, goal latent, candidate actions: exactly one of the `_commitment` / `_public` pair in `PublicInput` is `Some`. | verifier |
| INV-DM-14 | A tensor-memory read's expected `scale_id` equals the matched `TensorCell.scale_id`. | AIR wiring relation (RFC-0004) |
| INV-DM-15 | The manifest `relation_id` string, the hashed `PublicInput.relation_id`, and the verifier's supported-relation set agree; a proof is valid only for its exact `relation_id`. | verifier (RFC-0000) |
| INV-DM-16 | A `serialization_version` change coincides with a `relation_id` bump; a single `relation_id` pins exactly one `serialization_version`. | RFC-0014 parity tests + verifier |
| INV-DM-17 | Every commitment (`model_commitment`, `quantization_commitment`, `planner_config_commitment`, `weights.root`, per-op `weight_commitment`, `activation_tables_commitment`) equals `hash(canonical-serialization(bound-data))` under the manifest's declared `commitment_scheme`. | loader / verifier (RFC-0014) |

OPEN QUESTION: whether `BoundedInt.lo`/`hi` should be elided from the per-element
serialization (carried once per tensor + scale instead of per element) to shrink
the witness without weakening INV-DM-05/INV-DM-06. Owner: `area:core`
maintainers. Resolution path: RFC-0002
([docs/rfcs/RFC-0002-fixed-point-arithmetic-over-m31.md](../rfcs/RFC-0002-fixed-point-arithmetic-over-m31.md))
during milestone `v0.1 — Foundations`.

OPEN QUESTION: whether per-channel (vs strictly per-tensor) quantization will be
admitted in a later relation version, which would relax INV-DM-10 to per-channel
scale indexing. Owner: `area:export` maintainers. Resolution path: a new
`relation_id` version under RFC-0001; explicitly out of V0 scope.

---

## References

- Founding analysis: [docs/feasibility-study.md](../feasibility-study.md) (sections 4.2, 5, 7.5, 10.1, 10.2).
- Overview and scope tiers: [docs/spec/00-overview.md#scope-and-statement-tiers](00-overview.md#scope-and-statement-tiers).
- Public API (where these types are exposed): [docs/spec/02-public-api.md#rust-public-api](02-public-api.md#rust-public-api), [docs/spec/02-public-api.md#artifact-formats](02-public-api.md#artifact-formats).
- Error taxonomy and verifier rejections: [docs/spec/04-error-model.md#error-taxonomy](04-error-model.md#error-taxonomy), [docs/spec/04-error-model.md#verifier-rejections](04-error-model.md#verifier-rejections).
- Security, binding, and privacy boundary: [docs/spec/06-security.md#binding-requirements](06-security.md#binding-requirements), [docs/spec/06-security.md#privacy-and-zk](06-security.md#privacy-and-zk).
- Release and versioning policy: [docs/spec/09-release-and-versioning.md#relation-versioning](09-release-and-versioning.md#relation-versioning), [docs/spec/09-release-and-versioning.md#semver](09-release-and-versioning.md#semver).
- RFC-0000 Security model and statement taxonomy: [docs/rfcs/RFC-0000-security-model-and-statement-taxonomy.md](../rfcs/RFC-0000-security-model-and-statement-taxonomy.md).
- RFC-0001 Model manifest and export pipeline: [docs/rfcs/RFC-0001-model-manifest-and-export-pipeline.md](../rfcs/RFC-0001-model-manifest-and-export-pipeline.md).
- RFC-0002 Fixed-point arithmetic over M31: [docs/rfcs/RFC-0002-fixed-point-arithmetic-over-m31.md](../rfcs/RFC-0002-fixed-point-arithmetic-over-m31.md).
- RFC-0003 Range-check and lookup infrastructure: [docs/rfcs/RFC-0003-range-check-and-lookup-infrastructure.md](../rfcs/RFC-0003-range-check-and-lookup-infrastructure.md).
- RFC-0004 Tensor memory and wiring AIR: [docs/rfcs/RFC-0004-tensor-memory-and-wiring-air.md](../rfcs/RFC-0004-tensor-memory-and-wiring-air.md).
- RFC-0005 Linear, matmul, and requantization components: [docs/rfcs/RFC-0005-linear-matmul-and-requantization-components.md](../rfcs/RFC-0005-linear-matmul-and-requantization-components.md).
- RFC-0009 Fixed-candidate planner proof: [docs/rfcs/RFC-0009-fixed-candidate-planner-proof.md](../rfcs/RFC-0009-fixed-candidate-planner-proof.md).
- RFC-0014 Canonical serialization, public-input binding, and Fiat-Shamir transcript: [docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md](../rfcs/RFC-0014-canonical-serialization-and-transcript.md).
