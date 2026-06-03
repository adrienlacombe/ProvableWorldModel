# RFC-0005: Linear, matmul, and requantization components

- Status: Accepted
- Authors: ProvableWorldModel maintainers
- Created: 2026-06-03
- Target milestone: v0.1

## Summary

This RFC locks the AIR column layout and constraint set for the three components
that carry the bulk of LeWorldModel proving cost: `linear` (affine projection +
bias + requantize), `matmul` (two witness operands, no static weight table), and
`requant` (the shared right-shift-with-rounding-and-clamp gadget). It fixes a
single shared trace schema with one accumulating dot-product row group per output
element, five named batching axes (candidate, rollout step, sequence position,
head, output channel) folded into a single linearized `batch_id`, the
public-vs-private weight policy (public weights flow through the preprocessed
`WeightTable`; private weights flow through a committed witness lookup), and the
chunked-limb accumulator strategy that bounds every partial sum inside the safe
signed M31 interval. Every integer column is range-checked
(docs/spec/06-security.md#soundness-requirements), the requantization is
quotient/remainder-constrained, and the components emit `TensorCell` reads and
writes (RFC-0004) for wiring. This is the cost-dominant arithmetic substrate; all
of P0/P1/P2 (docs/spec/00-overview.md#scope-and-statement-tiers) lower onto it.

## Motivation

Dense linear algebra dominates proving cost. The V0 reference configuration
targets predictor `depth = 6`, `heads = 16`, `dim_head = 64`, `mlp_dim = 2048`,
`latent_dim = 192` (docs/spec/03-data-model.md#model-manifest,
contract §6.7), so a single predictor step is millions of multiply-accumulates
before rollout horizon (`horizon = 5`) or candidate count `S` multiply it. The
founding analysis names this the central engineering problem: "the biggest
practical problem is not STARK soundness; it is arithmetizing a transformer-like
neural network economically" (docs/feasibility-study.md §2). Source RFC-005 and
§7.6 give the linear AIR sketch but leave four load-bearing choices open: the
exact column set, how batching axes map to trace rows, where public versus
private weights enter the trace, and when the accumulator must be split into
limbs. Each choice is soundness- or cost-critical, so each is decided here and
made immutable for `relation_id` `pwm.lewm.*.v1`.

Concrete scenario: proving P2 fixed-candidate planning requires
`S * horizon` predictor steps, each invoking the predictor's attention QKV/out
projections and two MLP linears. Without a shared, batched, limb-aware linear
component the trace either overflows M31 silently (unsound) or compiles every MAC
to a generic scalar gate (too expensive, per docs/spec/01-architecture.md#air-strategy
which selects direct AIR for dense hot paths). This RFC removes both failure
modes by fixing one schema that all dense ops reuse.

## Goals

- Lock one trace schema shared by `linear`, `matmul`, and `requant`, with named
  columns and constraints reproducible by a new contributor without the author.
- Lock the five batching axes and their deterministic linearization into a single
  `batch_id`, so batched proving is bit-identical to repeated single-instance
  proving.
- Lock the public-vs-private weight policy: public weights enter via the
  preprocessed `WeightTable`; private weights enter via a committed witness lookup
  bound to `model_commitment`.
- Lock the chunked-limb accumulator strategy, including the machine-checkable rule
  (derived from tensor metadata in the manifest) for deciding when a single M31
  accumulator is safe versus when limbs are required.
- Range-check every integer column and quotient/remainder-constrain every
  requantization, satisfying docs/spec/06-security.md#range-safety (every value
  interpreted as an integer is range-checked) and overflow policy `reject`
  (contract §3).
- Emit `TensorCell` reads/writes (RFC-0004) so wiring across operators is sound.

## Non-Goals

- Nonlinear primitives (GELU, softmax, LayerNorm, AdaLN). The requantize gadget is
  reused by them, but their semantics are RFC-0006.
- Convolution / patch embedding. Source RFC-005 listed convolution; it is deferred
  to RFC-0011 (pixel encoder, Future) and removed from this component's surface.
  This RFC covers `linear`, `matmul`, `requant` only.
- The predictor block wiring (which linears feed which) — that is RFC-0007.
- The LogUp range/lookup machinery itself (multiplicity argument, table
  construction) — that is RFC-0003. This RFC consumes those relations by id.
- Tensor memory consistency mechanics (permutation/multiset argument) — RFC-0004.
  This RFC only declares the cells it reads and writes.
- Cost model numbers (trace rows per op, total MAC counts) —
  docs/spec/08-performance-budget.md#cost-model. This RFC fixes the per-op row
  shape; the budget doc multiplies it out.

## Proposed Design

### Component taxonomy and the shared dot-product engine

Three components, one engine. All three are instances of an accumulating
dot-product over a sequence of `(operand_a, operand_b)` products, terminated by a
requantization. They differ only in where `operand_b` comes from and whether a
bias seeds the accumulator.

| Component  | `operand_a` | `operand_b`        | bias seed | output                    |
|------------|-------------|--------------------|-----------|---------------------------|
| `linear`   | activation  | weight (W table)   | yes       | requantized activation    |
| `matmul`   | activation  | activation         | no        | requantized activation    |
| `requant`  | n/a         | n/a (degenerate)   | seed only | requantized activation    |

`requant` is the degenerate single-row case (the accumulator is loaded directly
from one input cell and immediately requantized); it exists as its own component
so nonlinear primitives (RFC-0006) and the cost component (RFC-0009 lowering) can
invoke shift-with-rounding-and-clamp without instantiating a full linear. The
shared engine is named `DotProductEngine` in `crates/pwm-air/src/components/`
and is parameterized by an `OperandSource` enum.

```rust
// crates/pwm-air/src/components/linear.rs
pub enum OperandSource {
    /// operand_b read from the preprocessed WeightTable (public weights).
    PublicWeight { weight_tensor_id: u32 },
    /// operand_b read from a committed private-weight witness lookup,
    /// keyed by (weight_tensor_id, in_idx, out_idx); bound to model_commitment.
    PrivateWeight { weight_tensor_id: u32 },
    /// operand_b read from tensor memory (matmul: both operands are activations).
    Activation { rhs_tensor_id: u32 },
}

pub struct DotProductSpec {
    pub op_id: u32,            // index into manifest ops[]; binds scales + rounding
    pub n_terms: u32,          // contraction length N (e.g. in_features, dim_head)
    pub operand_b: OperandSource,
    pub has_bias: bool,        // linear: true; matmul: false; requant: seed-only
    pub acc_limbs: u8,         // 1 = single M31 acc; >1 = chunked limb acc (see below)
    pub limb_radix_bits: u8,   // k in acc = Σ acc_limb_j * 2^(k*j); only if acc_limbs > 1
    pub shift: u8,             // requantization right-shift r (see Requantization)
    pub rounding: Rounding,    // NearestTiesToEven | TruncateTowardZero (manifest-bound)
    pub out_clamp: (i64, i64), // inclusive output activation range [lo, hi]
}
```

`Rounding`, `BoundedInt`, `M31` are the canonical types (contract §6.1). There is
exactly one active `rounding` per manifest (contract §3); `DotProductSpec.rounding`
is copied from the op's manifest entry and is bound by `quantization_commitment`,
not chosen at trace time.

### Trace columns (main trace)

One row group per output element. A row group has `n_terms` accumulation rows
(`linear`/`matmul`) or one row (`requant`). The schema extends the source §7.6
sketch with the batching, limb, and operand-provenance columns the sketch omitted.

```text
Main-trace columns (one row = one accumulation step):
  op_id          u32   manifest op index; constant within a row group
  batch_id       u32   linearized batch coordinate (see Batching axes)
  out_idx        u32   output channel index j within this op+batch
  in_idx         u32   contraction index i (0..n_terms-1)
  x              M31    operand_a value (activation x_i), centered-encoded
  w              M31    operand_b value (weight w_ij or activation), centered-encoded
  prod           M31    x * w   (single-limb) OR prod_lo/prod_hi pair (limb mode)
  acc            M31    running accumulator before this step's add (limb-0 column)
  acc_next       M31    running accumulator after this step's add (limb-0 column)
  is_first       {0,1}  1 on the first accumulation row of the group
  is_last        {0,1}  1 on the last accumulation row of the group
  bias           M31    bias_j; meaningful only where is_first = 1 (else 0)
  q              M31    requantization quotient; meaningful where is_last = 1
  rem            M31    requantization remainder; meaningful where is_last = 1
  y              M31    requantized clamped output; meaningful where is_last = 1

Limb-mode extension columns (present iff acc_limbs > 1):
  acc_limb[1..L-1]      additional accumulator limbs (limb 0 reuses acc/acc_next)
  prod_hi               high half of x*w when a single product can exceed limb 0
  carry[0..L-2]         inter-limb carry witnesses, range-checked to {0, 1}
```

Centered encoding: a signed integer `v` is carried as `v mod p` with a range-check
witness proving `lo <= v <= hi` (contract §6.1, RFC-0002). Every `M31` column above
that represents a signed quantity (`x`, `w`, `prod`, `acc`, `acc_next`, `bias`,
`q`, `rem`, `y`, all limbs) is bound to a range relation from RFC-0003.

### Constraints

Let `1{·}` denote a boolean selector column. All equalities are over M31 unless a
range relation is named. Constraint identifiers are stable.

```text
[C-LIN-1  acc-seed]
    is_first * (acc - bias_seed) = 0
        where bias_seed = bias       if has_bias
                        = 0           if not has_bias

[C-LIN-2  product]   (single-limb)
    prod - x * w = 0
                     (limb mode: prod_lo + 2^k * prod_hi - x*w = 0, both halves
                      range-checked; see [C-LIN-6])

[C-LIN-3  accumulate]   (single-limb)
    acc_next - (acc + prod) = 0
                     (limb mode: per-limb add with carry; see [C-LIN-6])

[C-LIN-4  acc-chain]
    (1 - is_first) * (acc - acc_prev_next) = 0
        // this row's acc equals previous row's acc_next within the same group
        // (group identity = (op_id, batch_id, out_idx) unchanged and is_first = 0)

[C-LIN-5  requantize-decompose]
    is_last * (acc_next - (q * 2^shift + rem)) = 0
    rem range-checked:  0 <= rem < 2^shift          (RFC-0003 bounded-limb table)

[C-LIN-6  limb-bound]   (limb mode only)
    For limb decomposition acc = Σ_{j=0}^{L-1} acc_limb_j * 2^(limb_radix_bits*j):
      each acc_limb_j range-checked to [0, 2^limb_radix_bits)
      each carry_j range-checked to {0, 1}
      reconstruction equality binds the limbs to acc_next on is_last rows

[C-LIN-7  round-and-clamp]
    is_last * (y - RoundAndClamp(q, rem, shift, rounding, out_clamp)) = 0
        // RoundAndClamp is the requant gadget; see Requantization below

[C-LIN-8  output-range]
    y range-checked to out_clamp = [lo, hi]          (RFC-0003)
```

`acc_prev_next` in [C-LIN-4] is the `acc_next` of the immediately preceding row,
read via the standard adjacent-row constraint mechanism of the Stwo constraint
framework (next-row evaluation). Row groups are laid out contiguously so adjacency
holds; group boundaries are marked by `is_first`/`is_last`. The pair
(`is_first`, `is_last`) is constrained boolean and, for `requant`, both are 1 on
the single row.

### Requantization gadget (`RoundAndClamp`)

Requantization is right-shift by `r = shift` with explicit rounding then clamp.
Never informal division (docs/feasibility-study.md §5.4). For a signed
accumulator `n = acc_next`:

```text
n = q * 2^r + rem,   0 <= rem < 2^r          [C-LIN-5]
```

Rounding (one of two, manifest-bound, contract §3):

```text
NearestTiesToEven (canonical default):
    half = 2^(r-1)
    rem <  half           -> rounded = q
    rem >  half           -> rounded = q + 1     (q advances toward +inf; sign of n
                                                  is carried in q since n centered)
    rem == half           -> rounded = q + (q mod 2)   // round to even
TruncateTowardZero (manifest override):
    rounded = q
```

The tie comparison `rem < half`, `rem > half`, `rem == half` is decided with
range-check witnesses on `half - rem - 1` (strictly-less branch) and `rem - half`
(strictly-greater branch), mirroring the comparison method in
docs/feasibility-study.md §5.5; the equality branch is a zero-test. `q mod 2` is a
single bit witness range-checked to {0, 1}. After rounding, clamp:

```text
y = min(hi, max(lo, rounded))                 with [C-LIN-8] enforcing y in [lo, hi]
```

`requant` (the standalone component) is exactly one row group with `is_first =
is_last = 1`, `acc` seeded from one input cell (no products), and the gadget
applied. This is the shared shift gadget reused by RFC-0006 nonlinear outputs and
RFC-0009 cost requantization. `INV-AIR-LIN-04` below names its safety.

### Batching axes

The five batching axes (the locked decision) are folded into the single `batch_id`
column by a fixed, row-major linearization. The axes, in major-to-minor order:

```text
batch_id = ((((candidate * R + step) * P + pos) * Hd + head) * Co_blocks + co_block)
  candidate   c   in [0, S)         fixed candidate index (P2); 0 for P0/P1
  step        t   in [0, horizon)   rollout step;            0 for P0
  pos         p   in [0, seq_len)   sequence position within a predictor call
  head        h   in [0, heads)     attention head;          0 for non-attention linears
  co_block    b   in [0, Co_blocks) output-channel block (channels are the
                                    innermost loop INSIDE a row group, not a
                                    batch axis; co_block tiles large Co for layout)
```

The strides `R = horizon`, `P = seq_len`, `Hd = heads`, `Co_blocks` are taken
from the manifest op entry and the statement parameters; they are preprocessed
constants, not witness. Output channel `out_idx` (the `j` loop) is the innermost
iteration and is NOT part of `batch_id`; it indexes within a batch. This
linearization is the canonical one: it is bound by `planner_config_commitment`
(for `S`, `horizon`) and `model_commitment` (for `seq_len`, `heads`, `Co`), so the
prover cannot reorder batches to forge a different computation. Determinism of
batch ordering is `INV-AIR-LIN-05`.

INV: batched proving is observationally identical to repeated single-instance
proving. Formally, the multiset of `(op_id, batch_id, out_idx, in_idx, x, w)`
tuples emitted for a batched op equals the disjoint union of the tuples each
single instance would emit. This is `INV-AIR-LIN-06` and is the property the
differential test `diff_batched_equals_unbatched` checks.

### Public versus private weights

The locked policy:

```text
manifest weights.visibility = public            -> OperandSource::PublicWeight
manifest weights.visibility = private_committed  -> OperandSource::PrivateWeight
```

- Public weights are materialized as preprocessed `WeightTable` columns (fixed,
  agreed by prover and verifier before proving, per
  docs/spec/01-architecture.md#trace-model). The `w` column on each accumulation
  row is constrained equal to the preprocessed entry at `(weight_tensor_id, in_idx,
  out_idx)` via a static lookup (RFC-0003). No hiding; the verifier knows the
  weights. The `WeightTable` commitment is part of `model_commitment`.
- Private weights are NOT in the preprocessed trace. The `w` column reads from a
  committed private-weight witness via a LogUp lookup keyed by `(weight_tensor_id,
  in_idx, out_idx)`; the lookup table's multiset commitment is bound to
  `model_commitment` through `QuantizedWeights.commitment` (contract §6.2,
  matches `manifest weights.root`). The verifier checks the commitment, never the
  values. This is a validity proof, NOT zero-knowledge (contract §3,
  docs/spec/06-security.md#privacy-and-zk): the trace is not hiding, so private
  weights are confidential only insofar as they are absent from public inputs and
  the proof does not currently mask trace columns. ZK masking is RFC-0012/Future.

`matmul` has no weights: both `operand_a` and `operand_b` are activations read
from tensor memory (RFC-0004). Used for attention `score = Q Kᵀ` and `out = prob
V` (docs/feasibility-study.md §7.7), where both factors are runtime witness.

### Chunked / limb accumulator strategy

The locked rule. A single M31 accumulator is safe iff the maximum absolute partial
sum over the whole contraction fits strictly inside the centered safe interval.
From contract §3 (V0 quantization: weights int8, activations int8/int16) and the
worst-case bound in docs/feasibility-study.md §5.2:

```text
max_abs_acc(op) = |bias_max| + n_terms * x_abs_max * w_abs_max
SINGLE-LIMB SAFE  iff  max_abs_acc(op) < 2^30        // strict; leaves headroom
```

Worked V0 cases:

```text
int8 x int8, n_terms = 2048:  2048 * 127 * 127 = 33,032,192  < 2^30  -> single limb
int8 x int8, n_terms = 192:   192 * 127 * 127  =  3,096,768  < 2^30  -> single limb
int16 x int8, n_terms = 2048: 2048 * 32767 * 127 = 8.5e9     > 2^30  -> LIMBS REQUIRED
```

When `max_abs_acc(op) >= 2^30`, the component uses `acc_limbs = L` limbs in
base `2^limb_radix_bits` (`limb_radix_bits` chosen so each limb and each carry stay
in M31; default `limb_radix_bits = 16`, `L = ceil(bits(max_abs_acc)/16) + 1`). The
limb count `L` and radix are computed from manifest tensor metadata (the declared
`BoundedInt` ranges of the operands and bias) at trace-build time and are
machine-checked: `crates/pwm-air` recomputes `max_abs_acc` from
`docs/spec/03-data-model.md#bounded-integers` ranges and asserts the chosen
`acc_limbs` covers it. A manifest whose declared ranges imply an accumulator that
cannot be represented even with limbs is rejected at export (RFC-0001) and at
trace build with `TraceBuildError::AccumulatorUnrepresentable`
(docs/spec/04-error-model.md#failure-modes). This realizes
docs/feasibility-study.md §5.3 ("if `acc` may exceed the safe signed M31 interval,
use limb decomposition").

### Tensor memory interaction

Per RFC-0004, the component reads inputs and writes outputs as `TensorCell`
(contract §6.4):

```text
reads:   x  at TensorCell{ tensor_id = input_tensor_id, index = [batch dims.., in_idx], ... }
         w  at TensorCell (matmul rhs) OR WeightTable/private-weight lookup (linear)
writes:  y  at TensorCell{ tensor_id = output_tensor_id, index = [batch dims.., out_idx], ... }
```

The `time` field (contract §6.4) orders the write after all reads of its row group.
Read/write consistency (every read matches a prior write or a static constant) is
the RFC-0004 permutation argument; this component only declares the cells.

### Invariants

```text
INV-AIR-LIN-01  (product binding) For every accumulation row, prod is exactly the
                M31 product x*w (or its limb decomposition); enforced by [C-LIN-2].
INV-AIR-LIN-02  (accumulation chain) acc_next of row r equals acc of row r+1 within
                a group; the group's terminal acc_next equals bias + Σ_i x_i w_ij;
                enforced by [C-LIN-1], [C-LIN-3], [C-LIN-4].
INV-AIR-LIN-03  (requant exactness) acc_next = q*2^shift + rem with 0 <= rem <
                2^shift, and y = RoundAndClamp(q, rem, shift, rounding, out_clamp)
                for the single manifest-bound rounding; enforced by [C-LIN-5],
                [C-LIN-7].
INV-AIR-LIN-04  (no field wrap) Every signed column lies in its declared [lo, hi]
                via a range relation, and max_abs_acc(op) < representable limb
                capacity, so no partial sum, product, or output silently wraps mod
                p. (docs/spec/06-security.md#range-safety; overflow policy reject,
                contract §3.)
INV-AIR-LIN-05  (deterministic batch order) batch_id is the fixed major-to-minor
                linearization of (candidate, step, pos, head, co_block) with strides
                bound by model_commitment + planner_config_commitment; the prover
                cannot permute it.
INV-AIR-LIN-06  (batch = repetition) The multiset of emitted accumulation tuples for
                a batched op equals the disjoint union over single instances.
INV-AIR-LIN-07  (weight provenance) Public weights equal the preprocessed
                WeightTable entry (bound by model_commitment); private weights equal
                the committed private-weight lookup entry (bound by
                QuantizedWeights.commitment = weights.root). The w column has exactly
                one provenance per op_id, fixed by manifest visibility.
```

### Failure modes

| Failure mode | Detected by | System response |
|---|---|---|
| `prod != x*w` (mutated product) | [C-LIN-2] | Constraint unsatisfied -> `VerifyError::ConstraintViolation` (docs/spec/04-error-model.md#verifier-rejections) |
| Mutated intermediate accumulator (same final field value via wrap) | [C-LIN-3]+[C-LIN-4]+`INV-AIR-LIN-04` range checks | Range relation rejects out-of-bound acc -> `VerifyError::RangeCheckFailed` |
| `rem >= 2^shift` (bad requant remainder) | [C-LIN-5] range check | `VerifyError::RangeCheckFailed` |
| Wrong rounding (e.g. round-half-up under nearest-ties-to-even manifest) | [C-LIN-7] | `VerifyError::ConstraintViolation` |
| Output outside clamp range | [C-LIN-8] | `VerifyError::RangeCheckFailed` |
| Public weight not matching WeightTable | static lookup (RFC-0003) | `VerifyError::LookupMembershipFailed` |
| Private weight not in committed table | LogUp multiplicity (RFC-0003) | `VerifyError::LookupSumMismatch` |
| Accumulator unrepresentable even with limbs | trace builder precheck | `TraceBuildError::AccumulatorUnrepresentable`, prover aborts before proving (docs/spec/04-error-model.md#failure-modes) |
| Reordered batch / forged batch_id | `INV-AIR-LIN-05` strides bound in commitments | digest mismatch -> `VerifyError::PublicInputMismatch` |
| Manifest declares `acc_limbs` too small for ranges | trace builder machine-check | `TraceBuildError::AccumulatorUnrepresentable` (export-side mirror in RFC-0001) |

## Alternatives Considered

### A1. One product per trace row with a separate sum/permutation argument

Lay out each `x_i * w_ij` product on its own row with no running accumulator
column, then prove `acc = Σ prod` with a LogUp/permutation sum rather than an
adjacent-row recurrence. Considered because it decouples products from ordering
and could share one giant sum argument across all ops. Rejected: (1) it loses the
cheap adjacent-row accumulation constraint and replaces it with an interaction-trace
sum that costs an extra committed column and a challenge per op group, inflating
the trace for the cost-dominant path (docs/spec/08-performance-budget.md#cost-model);
(2) it complicates the limb-carry logic, which is naturally expressed as a
sequential recurrence; (3) bugs in the global sum argument are harder to localize
than a per-row `acc_next = acc + prod`. The recurrence layout in §7.6 of the source
is the better fit and is what this RFC locks.

### A2. Generic scalar-gate lowering via stwo-circuits

Compile every MAC to the stwo-circuits gate set (`Add`, `Sub`, `Mul`,
`PointwiseMul`, `Eq`, `TripleXor`, `M31ToU32`, `BlakeGGate`, `Permutation`,
`Output` — verified against upstream at 2026-06-03; there is no dedicated range or
bit-extraction gate, range checks come from `sub`/`mul`/`assert_bits` helper
constraints). Considered because it would reuse an existing, auditable circuit
substrate and avoid bespoke AIR. Rejected for the dense hot path: lowering a
2048-length dot product to scalar gates produces thousands of generic rows per
output element with no batching structure, which docs/spec/01-architecture.md#air-strategy
explicitly routes away from ("avoid generic scalar gates for dense matmul hot
paths"). The founding analysis reaches the same split (docs/feasibility-study.md
§3.2): direct AIR for dense linear/matmul/attention; circuits for hashing,
transcript, and small gadgets. We keep stwo-circuits for those, not for matmul.

### A3. Per-channel (per-output) scales instead of per-tensor, with no shared shift

Allow each output channel its own `shift`/scale rather than the per-tensor scale
(contract §6.2 `Tensor.scale_id`, one scale id per tensor). Considered because
per-channel quantization is common in production int8 and improves accuracy.
Rejected for V0: it multiplies the requantization constant set by `Co`, forces
`shift` into a per-row witness (weakening `INV-AIR-LIN-03`'s single-rounding
guarantee), and complicates the manifest commitment. V0 fixes per-tensor scales
(contract §3, scales powers-of-two where possible); per-channel scaling is a
candidate for V1 under a new `relation_id` if accuracy requires it. This keeps the
requant gadget a single shared shift.

## Drawbacks

- The fixed major-to-minor `batch_id` linearization (`INV-AIR-LIN-05`) is rigid: a
  future op with a different natural iteration order (e.g. a transposed matmul that
  prefers column-major) must either conform or mint a new `relation_id`.
- Per-tensor scales (rejecting A3) leave accuracy on the table for models that
  benefit from per-channel quantization; those models cannot be proven under
  `pwm.lewm.*.v1` without a new relation.
- The single-limb-vs-limbs decision is conservative (`< 2^30`, not `< p`), trading a
  small amount of capacity headroom for a uniform safety margin; a few ops that
  would just fit in a single limb under a tighter bound will pay for limb columns.
- Private weights are committed but the trace is not hiding (not ZK). Operators who
  need true weight confidentiality must wait for the RFC-0012/Future masking work;
  this RFC does not provide it and the docs must not call it ZK
  (docs/spec/06-security.md#privacy-and-zk).
- `matmul` carries both operands as witness with no static-weight shortcut, so
  attention projections are more expensive than weight-table linears; this is
  intrinsic to attention and is accounted for in
  docs/spec/08-performance-budget.md#scaling.

## Migration / Rollout

- This component ships in `v0.1 — Foundations` (contract §4) as part of
  "linear/matmul/requant". It has no predecessor to deprecate; it is a foundational
  lock.
- Schema versioning: the trace schema and constraint ids are tied to the AIR
  component version, which is bound into `model_commitment` via the manifest
  `serialization` version (contract §6.6) and surfaced in
  docs/spec/09-release-and-versioning.md#relation-versioning. Any change to a column
  set, a constraint, the batch linearization, the limb rule, or the requant gadget
  is a semantic change that mints a new `relation_id` (`pwm.lewm.*.v2`); the
  existing `v1` proofs remain valid only for `v1`.
- Feature flags: `acc_limbs > 1` (limb mode) is selected per-op from manifest
  metadata, not a global flag; single-limb and limb-mode ops coexist in one proof.
  Private weights are selected per-manifest by `weights.visibility`; both visibility
  modes coexist across manifests but a single manifest is uniform per the manifest
  schema.
- Export side (RFC-0001) must emit the same `max_abs_acc` precheck so that
  `TraceBuildError::AccumulatorUnrepresentable` cannot first surface at prove time;
  the two checks are differential-tested against each other
  (`diff_export_rust_acc_bounds`).
- Rollout order across milestones: this component lands first; RFC-0006 (nonlinear,
  v0.2) reuses `requant`; RFC-0007 (predictor, v0.2) wires multiple `linear`/`matmul`
  instances; RFC-0009 (planner, v1.0) reuses `requant` for cost. No interface break
  is anticipated across these because the `DotProductSpec` surface is fixed here.

## Testing Strategy

Cross-references: docs/spec/07-testing-strategy.md (#golden-vectors,
#negative-tests, #differential-tests, #mutation-tests),
docs/spec/04-error-model.md#verifier-rejections.

Accepting tests:

- `accept_linear_random_vectors` — random int8 weights/activations, single-limb,
  per-tensor scale; prover produces a proof, verifier accepts; output equals the
  Rust fixed-point reference (golden vector). (docs/spec/07-testing-strategy.md#golden-vectors)
- `accept_linear_with_bias_nearest_even` — bias seed plus nearest-ties-to-even
  requant exercising all three tie branches (`rem<half`, `rem>half`, `rem==half`).
- `accept_matmul_qk_and_probv` — two-activation-operand matmul shaped like attention
  `Q Kᵀ` and `prob V`; accepts and matches reference.
- `accept_requant_standalone` — degenerate single-row `requant` for both
  `NearestTiesToEven` and `TruncateTowardZero`; matches reference for each.
- `accept_linear_limb_mode_int16` — int16 activations, `n_terms = 2048` forcing
  `acc_limbs > 1`; limb reconstruction accepts and matches reference.
- `accept_linear_private_weights` — `weights.visibility = private_committed`;
  verifier accepts using only the commitment, weight values absent from public input.

Differential tests:

- `diff_batched_equals_unbatched` — batched op over (candidate, step, pos, head)
  produces the same per-instance outputs as repeated single-instance linears;
  checks `INV-AIR-LIN-06`. (docs/spec/07-testing-strategy.md#differential-tests)
- `diff_export_rust_acc_bounds` — Python export and Rust trace builder compute the
  same `max_abs_acc` and the same `acc_limbs` for a battery of manifests.
- `diff_rust_python_fixed_point` — Rust reference linear/matmul equals the Python
  fixed-point reference bit-for-bit across random seeds.

Rejecting (negative) tests — each must produce the mapped `VerifyError`:

- `reject_mutated_product` — flip one `prod` value -> `ConstraintViolation` ([C-LIN-2]).
- `reject_mutated_accumulator_wrap` — alter an intermediate `acc` to a value that
  collides mod p on the final field element but exceeds its range ->
  `RangeCheckFailed` (`INV-AIR-LIN-04`).
- `reject_bad_requant_remainder` — set `rem >= 2^shift` -> `RangeCheckFailed`
  ([C-LIN-5]).
- `reject_wrong_rounding` — apply round-half-up where the manifest declares
  nearest-ties-to-even -> `ConstraintViolation` ([C-LIN-7]).
- `reject_output_out_of_clamp` — output below `lo`/above `hi` -> `RangeCheckFailed`
  ([C-LIN-8]).
- `reject_changed_public_weight` — alter a `WeightTable` entry away from the
  committed value -> `LookupMembershipFailed` / `PublicInputMismatch`
  (`INV-AIR-LIN-07`).
- `reject_forged_private_weight` — supply a private weight not in the committed
  table -> `LookupSumMismatch`.
- `reject_reordered_batch` — permute `batch_id` assignment -> `PublicInputMismatch`
  (`INV-AIR-LIN-05`).

Constraint mutation tests (docs/spec/07-testing-strategy.md#mutation-tests): drop or
weaken each of [C-LIN-1]..[C-LIN-8] in turn and assert at least one negative test
above starts failing to be rejected (i.e. the constraint is load-bearing). No
component ships without both accepting and rejecting tests (RFC-0013).

## Open Questions

- OPEN QUESTION (owner: area:air maintainer; resolution: RFC-0006): the exact
  shared signature by which RFC-0006 nonlinear components invoke the `requant`
  gadget (whether they call `DotProductEngine` with `OperandSource` degenerate or a
  thinner gadget entry point). This RFC fixes the gadget semantics; the call surface
  is finalized when RFC-0006 lands in v0.2.
- OPEN QUESTION (owner: area:air maintainer; resolution: docs/spec/08-performance-budget.md#profiling,
  v0.2): the optimal default `limb_radix_bits` (16 assumed) and the `co_block`
  tile size for the trace layout, to be set from profiling once the predictor AIR
  (RFC-0007) provides realistic op shapes. Correctness does not depend on the
  values; only trace size does.

## References

- docs/feasibility-study.md — source RFC-005, §5.3 (linear layer semantics), §5.4
  (requantization), §7.6 (linear component AIR sketch), §5.2 (accumulator bounds).
- docs/spec/00-overview.md#scope-and-statement-tiers, #v0-statement — statement
  tiers P0/P1/P2 that lower onto these components.
- docs/spec/01-architecture.md#air-strategy, #component-model, #trace-model —
  direct-AIR selection, component model, preprocessed/main/interaction traces.
- docs/spec/03-data-model.md#bounded-integers, #tensor-types, #model-manifest —
  `BoundedInt`, `Tensor`, `QuantizedWeights`, op scale entries.
- docs/spec/04-error-model.md#verifier-rejections, #failure-modes — `VerifyError`
  and `TraceBuildError` variants referenced above.
- docs/spec/06-security.md#range-safety, #soundness-requirements, #privacy-and-zk —
  range-safety, overflow=reject, validity-not-ZK boundary.
- docs/spec/08-performance-budget.md#cost-model, #scaling, #profiling — per-op row
  shape feeds the cost model; limb radix / tile size profiling.
- docs/spec/09-release-and-versioning.md#relation-versioning — relation_id minting
  on semantic change.
- RFC-0002 (fixed-point arithmetic over M31) — signed encoding, requant semantics
  this component instantiates.
- RFC-0003 (range-check and lookup infrastructure) — the LogUp/static lookup
  relations consumed by every range check and weight lookup here.
- RFC-0004 (tensor memory and wiring AIR) — `TensorCell` reads/writes this
  component declares.
- RFC-0006 (nonlinear primitive components) — reuses the `requant` gadget.
- RFC-0007 (LeWorldModel predictor AIR) — wires multiple `linear`/`matmul` instances.
- RFC-0009 (fixed-candidate planner proof) — reuses `requant` for MSE cost.
- RFC-0013 (testing, fuzzing, and audit strategy) — accepting+rejecting test rule.
- contract §6.1/§6.2/§6.4 — `M31`, `BoundedInt`, `Rounding`, `Tensor`,
  `QuantizedWeights`, `TensorCell` canonical signatures.
