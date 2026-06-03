# RFC-0002: Fixed-point arithmetic over M31

- Status: Accepted
- Authors: ProvableWorldModel maintainers
- Created: 2026-06-03
- Target milestone: v0.1

## Summary

This RFC fixes the arithmetic semantics of every integer value the proof system
manipulates. ProvableWorldModel proves an exact relation over the Mersenne-31
field M31 (`p = 2^31 - 1`), but quantized LeWorldModel inference is integer
arithmetic that must never wrap. This RFC locks: a signed centered encoding of
mathematical integers into M31; per-tensor declared bounds carried by every
`BoundedInt`; the bit-exact semantics of add, sub, mul, accumulate, requantize,
clamp, and compare; the canonical rounding mode `NearestTiesToEven` with an
optional manifest override to `TruncateTowardZero`; an overflow policy of
`reject` (never wrap); and the limb-decomposition rule that activates when an
accumulator can leave the safe signed M31 interval. The safe-interval bound for
the V0 reference configuration is `2048 * 127 * 127 = 33,032,192 < 2^31 - 1`,
so an int8 dot product of length 2048 fits in a single signed M31 value, while
anything wider decomposes into range-checked limbs. These rules are normative
for `pwm-core` (`fixed_point.rs`), the Python reference in `pwm-export`, and
every `pwm-air` component that interprets a column as an integer.

## Motivation

Soundness of the V0 statement
(`docs/spec/00-overview.md#v0-statement`) depends on a single non-obvious fact:
a finite field wraps, integer ML inference must not. A prover that is allowed to
let an accumulator silently overflow `p` can satisfy every algebraic constraint
while computing a different integer result, and the verifier would accept. The
founding analysis identifies this as a Critical risk ("Field wraparound") and
devotes section 5 (`docs/feasibility-study.md`, "Arithmetic semantics") to the
remedy: bound every integer, range-check every interpretation, define division
and rounding exactly, and reject overflow rather than wrap. This RFC is the
binding expansion of that section.

Concrete scenarios this RFC must make impossible:

- A malicious prover supplies an accumulator value `v` and a "true" value
  `v + k*p` that are equal mod `p`; the dot-product constraint holds, but the
  requantized output differs. (Defeated by INV-FP-01 / INV-FP-06.)
- A prover rounds a requantization tie in whichever direction yields the
  candidate cost it wants. (Defeated by the single bound rounding mode,
  INV-FP-04, bound by `quantization_commitment`.)
- A prover claims `a <= b` by exploiting field wrap so that `b - a` is a small
  canonical residue while the integers actually satisfy `a > b`. (Defeated by
  the difference-witness comparison rule, INV-FP-07.)

The decided choices in the contract (`pwm-core` field types, V0 quantization,
rounding default, overflow policy, argmin tie-break) are inputs to this RFC;
this RFC turns them into typed semantics and named invariants the AIR enforces.

## Goals

- Define the centered signed encoding `enc: Z -> M31` and its inverse on the
  bounded domain, and state the unique-decoding invariant.
- Specify `BoundedInt` lifetime: how `lo`/`hi` are declared, propagated through
  each operation, and discharged by a range check.
- Give bit-exact pseudocode and AIR constraint shapes for add, sub, mul,
  accumulate, requantize (with both rounding modes), clamp, and compare.
- Lock the canonical rounding mode and the single-active-mode-per-manifest rule.
- Lock `overflow_policy = reject` and define the static and dynamic checks that
  realize it.
- Lock the limb-decomposition trigger and layout, with the
  `2048 * 127 * 127 < 2^31 - 1` safe-interval derivation.
- Enumerate every failure mode with the system response, and name every
  invariant.

## Non-Goals

- The LogUp range-check and lookup machinery itself. This RFC states which
  values must be range-checked and the bound semantics; the relation encoding,
  multiplicity argument, and table layout are
  `docs/rfcs/RFC-0003-range-check-and-lookup-infrastructure.md`.
- Tensor read/write wiring and `TensorCell` consistency. See
  `docs/rfcs/RFC-0004-tensor-memory-and-wiring-air.md`.
- The linear/matmul/requant component column layouts and batching axes. This RFC
  defines the scalar arithmetic those components instantiate; the component
  design is `docs/rfcs/RFC-0005-linear-matmul-and-requantization-components.md`.
- Nonlinear approximations (GELU_Q, Softmax_Q, LayerNorm_Q). Those reduce to the
  primitives defined here plus committed lookup tables; see
  `docs/rfcs/RFC-0006-nonlinear-primitive-components.md`.
- The manifest schema and commitment construction. See
  `docs/rfcs/RFC-0001-model-manifest-and-export-pipeline.md` and
  `docs/spec/03-data-model.md#model-manifest`.
- Zero-knowledge / hiding. V0 is a succinct validity proof; see
  `docs/spec/06-security.md#privacy-and-zk`.

## Proposed Design

### 1. Field and bounded-integer types

The canonical types are fixed in the contract and reproduced here verbatim; this
RFC does not redefine them, it gives them semantics. See
`docs/spec/03-data-model.md#bounded-integers`.

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

`QM31` is the secure field and is never used to encode a tensor value; it carries
Fiat-Shamir challenges only (`docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md`).
All quantized data lives in M31 via `BoundedInt`.

### 2. Signed centered encoding (locked)

Let `p = 2^31 - 1` and `P_HALF = (p - 1) / 2 = 1073741823`. Define the
**centered representable interval**

```text
M31_SIGNED = [ -P_HALF, +P_HALF ]      (= [-1073741823, 1073741823])
```

Encoding and decoding:

```text
enc(x)  = x mod p                       for x in M31_SIGNED  -> M31 in [0, p)
dec(f)  = f               if f <= P_HALF
        = f - p           if f >  P_HALF                      -> i64 in M31_SIGNED
```

`enc` maps negative `x` to the upper half of `[0, p)`: `enc(-1) = p - 1`,
`enc(-P_HALF) = P_HALF + 1`. `dec` is the unique inverse on `M31_SIGNED`. The
field value `0` is its own decode; there is exactly one residue (`p`, which is
never canonical) that has no signed preimage, and canonical `M31` excludes it by
construction (`M31(u32)` is always in `[0, p)`).

- **INV-FP-01 (unique signed decode).** Every `BoundedInt` carried in the AIR
  satisfies `lo <= value <= hi` with `[lo, hi] ⊆ M31_SIGNED`. Therefore
  `dec(enc(value)) == value`. The range check that discharges `[lo, hi]`
  (RFC-0003) is what makes this hold in-circuit; without it the prover could
  present any residue.

The contract field block (`field.signed_encoding`) records this as
`centered_mod_p`; the manifest binds it (`docs/spec/03-data-model.md#model-manifest`).

### 3. BoundedInt lifecycle and the bound algebra

Every `BoundedInt` enters the system with `[lo, hi]` declared either by the
manifest (`field.max_abs_value` per tensor, weight/activation ranges) or by the
bound-propagation rules below for derived values. The pair `[lo, hi]` is **not**
advisory: it is the exact interval the AIR range-checks, and it is the input to
the static overflow analysis (section 7).

Bound propagation (computed at trace-build time, before any proving):

```text
add(a, b)        : lo = a.lo + b.lo,            hi = a.hi + b.hi
sub(a, b)        : lo = a.lo - b.hi,            hi = a.hi - b.lo
mul(a, b)        : lo = min(a.lo*b.lo, a.lo*b.hi, a.hi*b.lo, a.hi*b.hi)
                   hi = max(a.lo*b.lo, a.lo*b.hi, a.hi*b.lo, a.hi*b.hi)
accumulate(acc, terms) : lo = acc.lo + Σ term_i.lo, hi = acc.hi + Σ term_i.hi
clamp(x, c_lo, c_hi)   : lo = max(x.lo, c_lo),  hi = min(x.hi, c_hi)
```

- **INV-FP-02 (declared-bound containment).** For every operation, the computed
  result `value` lies in the propagated `[lo, hi]`, and `[lo, hi] ⊆ M31_SIGNED`
  unless the value is limb-decomposed (section 7). Trace building computes the
  propagated bound from declared input bounds; if `[lo, hi] ⊄ M31_SIGNED` and no
  limb decomposition is configured, trace building aborts with
  `Error::AccumulatorRange` (see `docs/spec/04-error-model.md#failure-modes`)
  before any proof is attempted. This is the static half of overflow=reject.

### 4. Scalar arithmetic semantics (locked)

All operations are defined on mathematical integers; the AIR enforces the same
relation on `enc(value)` in M31. Because every operand is range-checked into
`M31_SIGNED` and results are bound-propagated and re-checked, the field relation
and the integer relation coincide (INV-FP-06).

#### 4.1 Add / Sub

```text
add(a, b) = a + b                 enc: a_f + b_f       (mod p)
sub(a, b) = a - b                 enc: a_f - b_f       (mod p)
```

AIR constraint (one row per op, or fused into a component):

```text
result_f - (a_f + s * b_f) = 0     where s = +1 for add, -1 for sub
range_check(result, result.lo, result.hi)
```

- **INV-FP-03 (no implicit reduction).** Add/sub never reduce or requantize. The
  result bound is the propagated bound; if it exceeds `M31_SIGNED` the value must
  be a limb (section 7) or trace building rejects.

#### 4.2 Mul

```text
mul(a, b) = a * b                 enc: a_f * b_f       (mod p)
```

AIR constraint:

```text
prod_f - a_f * b_f = 0
range_check(prod, prod.lo, prod.hi)
```

For the int8 hot path `a, b ∈ [-127, 127]` (symmetric int8; the asymmetric
endpoint `-128` is excluded by the export quantizer per
`docs/rfcs/RFC-0001-model-manifest-and-export-pipeline.md` to keep products
symmetric), so `prod ∈ [-16129, 16129] ⊂ M31_SIGNED`.

#### 4.3 Accumulate

Accumulation is a left fold and is the operation most likely to leave
`M31_SIGNED`. It is defined as a recurrence so the AIR can chain it with a single
running column:

```text
acc_0     = bias                       (bias is a declared BoundedInt)
acc_{i+1} = acc_i + prod_i             for i in 0..N-1
result    = acc_N
```

AIR constraints (the linear-component instantiation lives in RFC-0005):

```text
is_first * (acc - bias) = 0
acc_next - (acc + prod) = 0
is_last  * (result - acc_next) = 0
range_check(acc_i, acc.lo, acc.hi)     for the running bound, OR limb-checked
```

- **INV-FP-08 (accumulator never wraps).** Either the propagated accumulator
  bound is `⊆ M31_SIGNED` and the running accumulator column is range-checked to
  that bound, or the accumulator is limb-decomposed (section 7) and each limb is
  range-checked. There is no third option; an un-bounded accumulator column is a
  trace-build error.

#### 4.4 Requantize (locked rounding semantics)

Requantization rescales an accumulator to an output scale by an arithmetic right
shift of `r` bits, with an exact quotient/remainder split — never informal
division.

```text
requantize(n, r, zero_point, c_lo, c_hi, mode):
    # exact euclidean-style split with NONNEGATIVE remainder
    q   = n >> r          (arithmetic, floor toward -inf)
    rem = n - q * 2^r     so that 0 <= rem < 2^r
    rounded = round(q, rem, r, mode)
    shifted = rounded + zero_point
    return clamp(shifted, c_lo, c_hi)
```

Rounding, mode `NearestTiesToEven` (canonical default):

```text
half = 2^(r-1)
if rem <  half:                rounded = q
if rem >  half:                rounded = q + 1
if rem == half:                rounded = q + (q & 1)     # bump to even
```

Rounding, mode `TruncateTowardZero` (optional manifest override):

```text
# truncate toward zero on the ORIGINAL signed n, not floor toward -inf
if n >= 0:                     rounded = q
else:                          rounded = q + (1 if rem != 0 else 0)
```

AIR constraints for the quotient/remainder split:

```text
acc_f - (q_f * pow2_r + rem_f) = 0
range_check(rem, 0, 2^r - 1)          # 0 <= rem < 2^r  (RFC-0003)
range_check(q,   q.lo, q.hi)
# rounding is a selector-driven affine relation on (q, rem):
#   ntte: rounded = q + ntte_bump(rem, q),  ntte_bump ∈ {0,1} via lookup/eq gates
#   ttz : rounded = q + ttz_bump(rem, sign(n))
range_check(shifted, shifted.lo, shifted.hi)
```

- **INV-FP-04 (single bound rounding mode).** Exactly one `Rounding` value is
  active per manifest. The active mode is a field of the manifest
  (`quantization.default_rounding`) and is bound by `quantization_commitment`
  (`docs/spec/03-data-model.md#model-manifest`). The Python reference, the Rust
  reference, and the AIR all read the same mode and produce bit-identical output;
  the differential test in
  `docs/spec/07-testing-strategy.md#differential-tests` enforces this. A manifest
  that declares no mode is rejected at load with `Error::ManifestRounding`.
- **INV-FP-05 (exact remainder split).** `0 <= rem < 2^r` and
  `n == q * 2^r + rem` hold as integer identities, enforced by the range check on
  `rem` and the algebraic split constraint. Floor-toward-`-inf` `q` makes the
  split unique; `TruncateTowardZero` derives its rounding from `n`'s sign on top
  of that unique split, so both modes share one split constraint.

#### 4.5 Clamp

```text
clamp(x, c_lo, c_hi) = c_lo  if x < c_lo
                     = c_hi  if x > c_hi
                     = x     otherwise
```

Clamp is **explicit per tensor**: `c_lo`/`c_hi` come from the manifest op entry,
never inferred. AIR realizes clamp with two nonnegative difference witnesses and
a selector choosing which of `{c_lo, x, c_hi}` is the output, all range-checked
(this reuses the compare gadget in 4.6).

- **INV-FP-09 (clamp is bound-narrowing, not wrap-fixing).** Clamp may only be
  emitted where the manifest declares a clamp; it must not be used to "repair" an
  out-of-range accumulator. The post-clamp bound is
  `[max(x.lo, c_lo), min(x.hi, c_hi)]`, and `x`'s pre-clamp bound must already be
  `⊆ M31_SIGNED` (or limb-checked). An attempt to clamp a value whose pre-clamp
  bound escapes `M31_SIGNED` is `Error::AccumulatorRange` at trace build.

#### 4.6 Compare

Comparisons are not native field operations. `a <= b` is proven by a nonnegative
difference witness:

```text
d = b - a
range_check(d, 0, D_max)     where D_max = b.hi - a.lo  (and D_max <= P_HALF)
```

Strict `a < b` is `b - a - 1 >= 0`, i.e. `range_check(b - a - 1, 0, D_max - 1)`.

This is the exact mechanism the locked argmin tie-break uses: for the selected
index `i*` and every `s < i*`, the AIR checks
`cost_s - selected_cost - 1 >= 0`, and for every `s`,
`cost_s - selected_cost >= 0` (contract §3 argmin tie-break;
`docs/rfcs/RFC-0009-fixed-candidate-planner-proof.md`).

- **INV-FP-07 (comparison cannot wrap).** Every comparison difference is
  range-checked into `[0, D_max]` with `D_max <= P_HALF`. Because the difference
  itself is bounded well inside `M31_SIGNED`, a wrapped difference (which would
  be a large residue near `p`) fails the range check. This closes the
  "comparison wraparound" attack.

### 5. The master soundness invariant

- **INV-FP-06 (field/integer agreement).** For any straight-line composition of
  the operations in section 4 over operands that are range-checked into
  `M31_SIGNED`, with every intermediate result either bounded in `M31_SIGNED` or
  limb-decomposed, the M31 relation enforced by the AIR holds **iff** the
  integer relation holds. Proof sketch: each primitive's field constraint is the
  reduction mod `p` of its integer identity; range checks confine every operand
  and result to `M31_SIGNED` (or to limbs with their own checks), an interval of
  width `< p`; on an interval of width `< p`, `enc` is injective and the
  reduction mod `p` is the identity on representatives. Hence no two distinct
  in-range integer assignments share a field assignment, and the field relation
  determines the integer relation uniquely. This invariant is the arithmetic
  core of `docs/spec/06-security.md#soundness-requirements`.

### 6. Overflow policy: reject (locked)

`OverflowPolicy::Reject` is the only V0 policy. Wrapping is unsound (it breaks
INV-FP-06) and is never permitted. Reject is realized in two layers:

- **Static (trace build, `pwm-core`/`pwm-prover`).** Bound propagation
  (section 3) computes `[lo, hi]` for every value. If any non-limb value's bound
  escapes `M31_SIGNED`, trace building aborts with `Error::AccumulatorRange`
  before proving. This is a prover-side guard: a well-formed manifest never trips
  it, and a malformed one fails fast with a precise diagnostic.
- **Dynamic (in-circuit).** Every value carries a range check (RFC-0003) to its
  declared bound. A witness whose actual value leaves its declared bound makes
  the LogUp multiplicity argument inconsistent, and `verify`
  (`docs/spec/02-public-api.md#rust-public-api`) returns
  `VerifyError::RangeCheckFailed` (`docs/spec/04-error-model.md#verifier-rejections`).

There is no `Wrap` or `Saturate` variant in the enum; adding one is a new
relation version (`docs/spec/09-release-and-versioning.md#relation-versioning`),
not a flag.

### 7. Limb decomposition (locked trigger and layout)

The safe interval for a single signed M31 value is `M31_SIGNED`, width
`2*P_HALF = p - 1 ≈ 2^31`. The decision rule:

```text
SAFE_HI = P_HALF = 1073741823          (= (2^31 - 2) / 2)
if propagated |bound| <= SAFE_HI:  keep as a single M31 value
else:                              limb-decompose
```

**Safe-interval derivation for the V0 reference configuration.** The widest V0
dot product is the predictor MLP with `mlp_dim = 2048`
(`docs/spec/00-overview.md#scope-and-statement-tiers`, contract §6.7). With
symmetric int8 operands the worst-case accumulator magnitude is

```text
2048 * 127 * 127 = 33,032,192 < 2^31 - 1 = 2,147,483,647
```

so a length-2048 int8 dot product fits in **one** signed M31 value with about
65x headroom, and no limb decomposition is required for the V0 int8 MLP/linear
path. Bias is int32-bounded and added once: `33,032,192 + 2^31` would exceed
`M31_SIGNED`, but the V0 manifest bounds biases so that `acc_0 = bias` and the
running `acc_N` stay within `M31_SIGNED` (the export quantizer enforces this;
RFC-0001). The static check (section 6) verifies it per op rather than assuming.

When a value's bound exceeds `SAFE_HI` (int16 activations into long
accumulators, or any future wider quantization), it is represented as two limbs:

```text
value = lo + 2^K * hi          with K chosen so both limbs fit M31_SIGNED
lo ∈ [0, 2^K - 1]              (unsigned low limb, range-checked)
hi ∈ [hi_lo, hi_hi]            (signed high limb carrying the sign, range-checked)
```

`K` is a per-op manifest field (default `K = 16`) so prover and verifier agree on
the split, and `K` is bound by `quantization_commitment`. Arithmetic on limbs
(add of two limb values, the carry into a non-limb value at requantize time) is
defined by the schoolbook identities with explicit carry witnesses, each
range-checked; the full limb-arithmetic AIR layout is specified alongside the
linear component in
`docs/rfcs/RFC-0005-linear-matmul-and-requantization-components.md` since that is
where wide accumulators arise. This RFC fixes the *trigger* (`> SAFE_HI`), the
*two-limb shape*, and the rule that each limb is independently range-checked.

- **INV-FP-10 (limb completeness and tightness).** A limb-decomposed value
  reconstructs exactly (`value == lo + 2^K * hi`, enforced by an algebraic
  constraint), each limb is range-checked, and `2^K * hi.hi + (2^K - 1)` fits in
  `M31_SIGNED` (so the reconstruction itself never wraps). Trace build chooses
  the minimum number of limbs (V0: at most two) such that this holds; a value
  needing more than two limbs in V0 is `Error::AccumulatorRange` and signals a
  manifest that violates the V0 quantization envelope.

### 8. Reference and AIR co-equality

Three implementations realize section 4 and must agree bit-for-bit:

1. Python fixed-point reference (`pwm-export`) — the oracle that produces golden
   vectors.
2. Rust fixed-point reference (`pwm-core::fixed_point`) — runs in the prover's
   trace builder and in CI parity checks; never runs in the verifier.
3. The AIR constraints (`pwm-air`) — the relation the proof actually enforces.

- **INV-FP-11 (tri-reference parity).** For the same manifest and inputs, the
  Python reference, the Rust reference, and the witness implied by the AIR
  produce identical `M31` outputs for every primitive. The verifier never runs
  either reference (`docs/feasibility-study.md` section 10.4); parity is a
  build-time and CI guarantee, not a runtime one.

### 9. Data flow

```text
manifest (rounding, clamp, scales, bounds, K)   inputs (BoundedInt tensors)
        |                                                |
        v                                                v
   bound propagation  -----------------------------> trace builder
        |  (static reject if bound ⊄ M31_SIGNED)        |
        |                                                v
        |                                   per-op witness rows + range/lookup
        |                                                |
        v                                                v
   AIR constraints (add/sub/mul/acc/requant/clamp/compare) over enc(value)
        |                                                |
        +----------------------> Stwo prove -------------+--> Proof
```

Prose: bounds and modes come only from the commitment-bound manifest; the trace
builder runs the Rust reference to fill witness columns and emits a range/lookup
obligation for every integer value; the AIR re-checks the same identities in
M31; overflow is rejected statically (bad bound) or dynamically (failed range
check).

## Alternatives Considered

**A1. Unsigned offset-binary encoding (store `x + B` as an unsigned value).**
Maps a signed range `[-B, B]` to `[0, 2B]` and works with unsigned range tables.
Rejected because subtraction and accumulation then require tracking and undoing
the offset (`(x+B) + (y+B) = (x+y) + 2B`), which multiplies bookkeeping across
the dot-product hot path and makes the bias term position-dependent. Centered
`x mod p` keeps add/sub/mul as the field's own operations with no offset algebra,
at the cost of needing a signed range check — which RFC-0003 provides natively.

**A2. Represent quantized values directly in QM31 to get a wider safe interval.**
QM31 (the degree-4 secure field) has far more headroom, so accumulators would
rarely need limbs. Rejected for two reasons: (1) QM31 is the soundness field for
Fiat-Shamir challenges (contract §6.1); using it for data conflates the
soundness budget with data representation and complicates the security argument
in `docs/spec/06-security.md#soundness-requirements`; (2) QM31 columns are 4x the
trace width of M31 columns, so the dense matmul hot path would pay 4x trace cost
to avoid an occasional two-limb split. Single-M31 representation with limbs only
when `> SAFE_HI` is strictly cheaper for the int8-dominated V0 workload.

**A3. Floor/truncation-only rounding (drop NearestTiesToEven entirely).**
Truncation toward zero has the simplest AIR (no tie logic). Rejected as the
*default* because LeWorldModel quantized inference accumulates rounding bias over
6 predictor blocks x horizon-5 rollout, and a systematic floor bias degrades the
final-latent MSE used by the planner cost, which can flip the argmin selection
and thus the proven decision. NearestTiesToEven is the unbiased default; we keep
TruncateTowardZero as an opt-in manifest override (INV-FP-04) for models exported
under a truncating quantizer, so the system can prove either, but never both in
one manifest.

**A4. Saturating overflow (clamp accumulators to the field instead of
rejecting).** Saturation would let proving proceed on any manifest. Rejected
because saturation is a silent semantic change: the proven relation would no
longer be the declared integer computation, breaking INV-FP-06 and the V0
statement. Overflow must be a hard reject so that a manifest outside the safe
envelope fails loudly rather than proving a different function.

## Drawbacks

- Every integer value carries a range-check obligation, which is the dominant
  source of LogUp multiplicity rows (`docs/spec/08-performance-budget.md#cost-model`).
  This is intrinsic to range-safety, not incidental.
- The two rounding modes double the rounding-path test matrix and the AIR's
  rounding selector logic. We accept this to support both
  nearest-ties-to-even-exported and truncation-exported checkpoints.
- Limb decomposition adds reconstruction constraints and carry witnesses for
  wide accumulators; V0's int8 path avoids it, but int16 activations (a manifest
  option) pay the cost.
- Bound propagation must be exact and conservative; a too-loose declared bound
  wastes range-table rows, a too-tight one is a (correctly) rejecting bug. The
  export pipeline owns producing tight, correct bounds (RFC-0001).

## Migration / Rollout

- **Relation versioning.** These semantics are part of `pwm.lewm.*.v1`. Any
  change to encoding, a rounding rule, the overflow policy, the `SAFE_HI`
  threshold, or the limb layout mints a new `relation_id` (e.g. `...v2`) per
  `docs/spec/09-release-and-versioning.md#relation-versioning`; a proof is valid
  only for its declared `relation_id` (contract §4).
- **Manifest binding.** `field.signed_encoding = centered_mod_p`,
  `quantization.default_rounding`, `quantization.overflow_policy = reject`,
  per-tensor `clamp` bounds, per-op limb `K`, and `field.max_abs_value` are all
  bound by `model_commitment` / `quantization_commitment`
  (`docs/spec/03-data-model.md#model-manifest`). A manifest that omits a required
  field is rejected at load (`Error::ManifestRounding`, `Error::ManifestField`).
- **Feature flag.** `TruncateTowardZero` ships behind a manifest field, not a
  compile-time flag; both modes are in the v0.1 binary. There is no flag for
  wrapping or saturation — those are simply absent from `OverflowPolicy`.
- **Limb rollout.** V0 ships the two-limb path in `pwm-core`/`pwm-air` but the
  V0 reference manifest (int8) never triggers it; int16-activation manifests
  exercise it. This lets the limb code land and be tested in v0.1 without being
  on the V0 critical path.

## Testing Strategy

Cross-references: `docs/spec/07-testing-strategy.md`,
`docs/spec/04-error-model.md`. Every primitive ships with both accepting and
rejecting tests (contract §7 RFC-0013 rule).

Accepting tests (golden vectors, `docs/spec/07-testing-strategy.md#golden-vectors`):

- `fp_encode_roundtrip_accept`: for a sweep over `M31_SIGNED` extremes and
  random values, `dec(enc(x)) == x` (INV-FP-01).
- `fp_mul_int8_accept`: int8 x int8 products land in `[-16129, 16129]` and match
  the Python reference (INV-FP-11).
- `fp_accumulate_mlp2048_accept`: a length-2048 int8 dot product fits one M31
  value, equals `33,032,192` at the worst-case all-`127` vector, and the AIR
  accepts (INV-FP-08, safe-interval derivation).
- `fp_requantize_ntte_accept` / `fp_requantize_ttz_accept`: requantize golden
  vectors for both modes, including the tie cases `rem == 2^(r-1)` with even and
  odd `q`, and negative `n` truncation (INV-FP-04, INV-FP-05).
- `fp_clamp_explicit_accept`: clamp at lower, interior, and upper points matches
  the reference (INV-FP-09).
- `fp_compare_argmin_tiebreak_accept`: the locked tie-break (`cost_s -
  selected_cost - 1 >= 0` for `s < i*`) accepts a valid argmin witness
  (INV-FP-07; composes into RFC-0009).
- `fp_limb_int16_accumulate_accept`: an int16 accumulator that exceeds `SAFE_HI`
  decomposes into two range-checked limbs, reconstructs exactly, and the AIR
  accepts (INV-FP-10).

Rejecting / negative tests (`docs/spec/07-testing-strategy.md#negative-tests`,
mapped to `docs/spec/04-error-model.md#verifier-rejections`):

- `fp_overflow_same_field_value_reject`: present an accumulator equal mod `p` to
  the true value but out of `[lo, hi]` (the classic `v` vs `v + p` attack);
  expect `VerifyError::RangeCheckFailed` (INV-FP-06, INV-FP-08).
- `fp_accumulator_mutation_reject`: mutate one intermediate accumulator;
  expect `VerifyError::RangeCheckFailed` or `VerifyError::ConstraintUnsatisfied`.
- `fp_requantize_bad_remainder_reject`: supply `rem >= 2^r` or `rem < 0`;
  expect `VerifyError::RangeCheckFailed` (INV-FP-05).
- `fp_requantize_wrong_tie_direction_reject`: round a tie up when `q` is even
  (violating ties-to-even); expect `VerifyError::ConstraintUnsatisfied`
  (INV-FP-04).
- `fp_compare_wraparound_reject`: claim `a <= b` where the integer order is `a >
  b` by exploiting a large residue difference; expect
  `VerifyError::RangeCheckFailed` (INV-FP-07).
- `fp_clamp_repairs_overflow_reject` (trace-build negative): a manifest that
  tries to clamp an out-of-`M31_SIGNED` value; expect `Error::AccumulatorRange`
  at trace build, never a proof (INV-FP-09).
- `fp_manifest_no_rounding_reject`: manifest with no `default_rounding`; expect
  `Error::ManifestRounding` at load (INV-FP-04).
- `fp_three_limb_needed_reject` (trace-build negative): a manifest whose op
  exceeds the two-limb envelope; expect `Error::AccumulatorRange` (INV-FP-10).

Differential and mutation tests:

- `fp_python_rust_parity_diff` (`docs/spec/07-testing-strategy.md#differential-tests`):
  randomized inputs through all primitives in both modes; Python and Rust
  references must agree bit-for-bit (INV-FP-11).
- `fp_constraint_mutation` (`docs/spec/07-testing-strategy.md#mutation-tests`):
  drop or weaken each rounding/range/split constraint in turn; at least one
  negative test above must flip from reject to accept, proving the constraint is
  load-bearing.

## Open Questions

- OPEN QUESTION: the precise int16-activation accumulator profile (which V0 ops,
  if any, the export pipeline emits at int16 versus int8) is owned by the export
  maintainers and resolves in
  `docs/rfcs/RFC-0001-model-manifest-and-export-pipeline.md` during v0.1; until
  it lands, the two-limb path is implemented and tested but not on the int8 V0
  critical path. The `2048 * 127 * 127` int8 envelope is settled regardless.
- OPEN QUESTION: whether the limb base `K` should be a single global constant or
  per-op is owned by the AIR maintainers and resolves with the wide-accumulator
  layout in
  `docs/rfcs/RFC-0005-linear-matmul-and-requantization-components.md` (v0.1);
  this RFC fixes the default `K = 16` and that `K` is commitment-bound either way.

## References

- Founding analysis, arithmetic semantics: `docs/feasibility-study.md`
  (sections 5, 9.2, 9.3; risk register section 13 "Field wraparound").
- V0 statement and scope: `docs/spec/00-overview.md#v0-statement`,
  `docs/spec/00-overview.md#scope-and-statement-tiers`.
- Canonical types and manifest: `docs/spec/03-data-model.md#bounded-integers`,
  `docs/spec/03-data-model.md#tensor-types`,
  `docs/spec/03-data-model.md#model-manifest`,
  `docs/spec/03-data-model.md#invariants`.
- Public API and verifier entry: `docs/spec/02-public-api.md#rust-public-api`.
- Error and rejection taxonomy: `docs/spec/04-error-model.md#failure-modes`,
  `docs/spec/04-error-model.md#verifier-rejections`.
- Soundness requirements: `docs/spec/06-security.md#soundness-requirements`,
  `docs/spec/06-security.md#privacy-and-zk`.
- Testing: `docs/spec/07-testing-strategy.md#golden-vectors`,
  `docs/spec/07-testing-strategy.md#negative-tests`,
  `docs/spec/07-testing-strategy.md#differential-tests`,
  `docs/spec/07-testing-strategy.md#mutation-tests`.
- Performance: `docs/spec/08-performance-budget.md#cost-model`.
- Versioning: `docs/spec/09-release-and-versioning.md#relation-versioning`.
- Related RFCs: `docs/rfcs/RFC-0001-model-manifest-and-export-pipeline.md`,
  `docs/rfcs/RFC-0003-range-check-and-lookup-infrastructure.md`,
  `docs/rfcs/RFC-0004-tensor-memory-and-wiring-air.md`,
  `docs/rfcs/RFC-0005-linear-matmul-and-requantization-components.md`,
  `docs/rfcs/RFC-0006-nonlinear-primitive-components.md`,
  `docs/rfcs/RFC-0009-fixed-candidate-planner-proof.md`,
  `docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md`.
