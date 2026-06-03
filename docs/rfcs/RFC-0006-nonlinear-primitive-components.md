# RFC-0006: Nonlinear primitive components (GELU, softmax, LayerNorm, AdaLN)

- Status: Accepted
- Authors: ProvableWorldModel maintainers
- Created: 2026-06-03
- Target milestone: v0.2

## Summary

This RFC locks the proof-native arithmetization of the four nonlinear primitives
the LeWorldModel predictor requires: `GELU_Q`, `Softmax_Q`, `LayerNorm_Q`, and
`AdaLN_Q`. Each is defined as an exact fixed-point relation over centered
integers embedded in M31 (see
[docs/rfcs/RFC-0002-fixed-point-arithmetic-over-m31.md](RFC-0002-fixed-point-arithmetic-over-m31.md)),
realized through exactly one of four committed mechanisms: LogUp table lookup,
piecewise-polynomial evaluation, bounded Newton reciprocal-square-root, or folded
affine. The RFC fixes the allowed mechanism per primitive, the explicitly
rejected mechanisms, and the rule that every approximation parameter (id, input
domain, output domain, scale ids, table or polynomial commitment, rounding mode,
maximum tabulated error, and golden test vectors) is bound by the model manifest
via `quantization.activation_tables_commitment` and the per-op `commitment`
field. A nonlinear op whose approximation is not committed is unsound because the
prover could silently substitute a different curve. This matters because these
primitives are the highest-risk part of the system
([docs/feasibility-study.md](../feasibility-study.md), §6) and the only place
where the proof relation can diverge from the exported reference graph without
detection if left unbound. The predictor FFN uses GELU; AdaLN modulation and the
action `Embedder` use SiLU; both activations are specified here.

## Motivation

The LeWorldModel predictor is an autoregressive transformer
(`ARPredictor` of `ConditionalBlock`s) rather than an MLP. Verified against
upstream at 2026-06-03, each block uses AdaLN-zero modulation, scaled
dot-product attention, and a FeedForward, and the model conditions on action
embeddings produced by an `Embedder`. The feasibility study identifies these
nonlinearities as the dominant semantic risk
([docs/feasibility-study.md](../feasibility-study.md), §6): attention softmax,
LayerNorm/AdaLN normalization (with its reciprocal square root), and the GELU
inside the FFN cannot be expressed as native M31 field operations. Any
implementation that lets the prover supply softmax probabilities, a reciprocal,
or an inverse square root off-circuit is unsound — the field wraps, comparisons
are not native, and division is not a field-respecting operation, so an
unconstrained helper value is a free parameter the prover can choose to make a
false statement verify.

The matching binding requirement is already stated as an obligation in
[docs/spec/06-security.md#soundness-requirements](../spec/06-security.md#soundness-requirements)
(approximation requirements) and
[docs/spec/03-data-model.md#model-manifest](../spec/03-data-model.md#model-manifest):
for each approximate primitive the manifest must include an approximation
identifier, domain, scale, table or polynomial commitment, rounding mode, maximum
error, and test vectors. This RFC turns that obligation into concrete typed AIR
component layouts, manifest fields, and witness shapes, and decides which
mechanism each primitive uses so a contributor can implement the
`activation_lookup.rs`, `layernorm.rs`, and `attention.rs` components named in
[docs/spec/01-architecture.md#component-model](../spec/01-architecture.md#component-model)
without further design work.

Concrete scenarios this design must defend against:

- A prover claims a candidate rollout produces a low cost by feeding the
  attention block softmax probabilities that do not sum to the committed
  denominator. The design must constrain the denominator, the per-element
  numerator lookup, and the normalization division.
- A prover swaps the GELU table for a SiLU table (or vice versa) to match a
  different exported model. The per-op `commitment` and
  `activation_tables_commitment` must bind the exact table identity.
- A prover supplies an inverse-standard-deviation value for LayerNorm that is
  larger than the true value, shrinking the normalized residual. The reciprocal
  square root must be constrained by a checked back-multiplication, not asserted.

## Goals

- Define `GELU_Q`, `Softmax_Q`, `LayerNorm_Q`, `AdaLN_Q` as exact, deterministic
  fixed-point relations with named invariants and enumerated failure modes.
- Lock the implementation mechanism per primitive (allowed list) and enumerate
  rejected mechanisms with reasons.
- Specify per-module activation selection: predictor FFN/MLP uses GELU; AdaLN
  modulation and the action `Embedder` use SiLU.
- Specify the manifest fields and commitment binding for every approximation,
  extending
  [docs/spec/03-data-model.md#model-manifest](../spec/03-data-model.md#model-manifest).
- Specify the AIR column layout, preprocessed/main/interaction trace usage, and
  range/lookup wiring for each component, consistent with
  [docs/spec/01-architecture.md#trace-model](../spec/01-architecture.md#trace-model).
- Give specific accepting and rejecting tests cross-referencing
  [docs/spec/07-testing-strategy.md](../spec/07-testing-strategy.md) and the
  rejection codes in
  [docs/spec/04-error-model.md#verifier-rejections](../spec/04-error-model.md#verifier-rejections).

## Non-Goals

- Proving that any quantized approximation is numerically close to the original
  PyTorch float curve. The proof enforces the committed approximation exactly; a
  separate error theorem or exhaustive tabulation argument (out of scope here,
  owned by export) bounds approximation error. See Open Questions.
- The linear/matmul/requantization machinery that surrounds these primitives;
  that is
  [docs/rfcs/RFC-0005-linear-matmul-and-requantization-components.md](RFC-0005-linear-matmul-and-requantization-components.md).
  This RFC consumes its requantized integer inputs and produces requantized
  integer outputs.
- The full predictor block wiring (Q/K/V projections, residual structure, block
  ordering); that is
  [docs/rfcs/RFC-0007-leworldmodel-predictor-air.md](RFC-0007-leworldmodel-predictor-air.md).
  This RFC defines the nonlinear leaves it calls.
- Pixel-encoder LayerNorm/softmax at ViT scale, deferred to
  [docs/rfcs/RFC-0011-pixel-encoder-proof.md](RFC-0011-pixel-encoder-proof.md)
  (V3). The primitives here are reused there but cost is out of v0.2 scope.
- Choosing whether to prove the original checkpoint or a distilled proof-native
  predictor. This RFC supports both — folded affine and lookup mechanisms cover
  the distilled and faithful paths respectively — but the model-identity decision
  belongs to
  [docs/rfcs/RFC-0001-model-manifest-and-export-pipeline.md](RFC-0001-model-manifest-and-export-pipeline.md).

## Proposed Design

### Common contract: requantized integer in, requantized integer out

Every nonlinear primitive operates on `BoundedInt` values (the canonical signed
encoding from
[docs/spec/03-data-model.md#bounded-integers](../spec/03-data-model.md#bounded-integers)):

```rust
pub struct BoundedInt {
    pub value: i64,             // mathematical signed value
    pub lo: i64,                // inclusive lower bound (declared)
    pub hi: i64,                // inclusive upper bound (declared)
}
pub enum Rounding { NearestTiesToEven, TruncateTowardZero }
```

Inputs arrive already requantized to a declared `input_scale_id`; outputs are
emitted at a declared `output_scale_id`. The active rounding mode is the single
mode declared in `quantization.default_rounding` (manifest-wide; one active mode
per manifest, per
[docs/spec/03-data-model.md#model-manifest](../spec/03-data-model.md#model-manifest)).
All intermediate columns are range-checked through the LogUp infrastructure of
[docs/rfcs/RFC-0003-range-check-and-lookup-infrastructure.md](RFC-0003-range-check-and-lookup-infrastructure.md);
all tensor reads/writes flow through `TensorCell`
([docs/spec/03-data-model.md#tensor-memory-cells](../spec/03-data-model.md#tensor-memory-cells)).

The four committed mechanisms, fixed for this RFC:

| Mechanism | Definition | Used by |
| --- | --- | --- |
| `lookup` | LogUp membership in a committed table of `(x, y)` pairs over a finite quantized input domain | `GELU_Q`, `SiLU_Q`, `Softmax_Q` exponential, `LayerNorm_Q`/`AdaLN_Q` reciprocal-sqrt seed |
| `piecewise_poly` | Evaluate one of `K` committed polynomials selected by a domain segment, with checked segment selection | `GELU_Q`, `SiLU_Q` alternative for wide domains |
| `newton_rsqrt` | Bounded fixed-iteration Newton refinement of `1/sqrt(v+eps)`, each step a checked back-multiplication | `LayerNorm_Q`, `AdaLN_Q` |
| `folded_affine` | Compile-time-folded affine `y = a*x + b` (eval-mode BatchNorm, distilled RMSNorm-as-affine) | `LayerNorm_Q` distilled path only |

Each op declares exactly one mechanism in the manifest. The mechanism choice is
part of the model commitment (INV-RFC0006-08).

### Manifest binding (extends docs/spec/03-data-model.md#model-manifest)

Each nonlinear op carries an `approx` block. The aggregate of all `approx`
blocks is committed under `quantization.activation_tables_commitment`; each op's
`approx` is additionally bound by its own `ops[].commitment`. The verifier never
reconstructs the curve, but it does check that the table/polynomial columns used
in the preprocessed trace hash to these commitments
([docs/spec/01-architecture.md#trace-model](../spec/01-architecture.md#trace-model),
verifier step "verify preprocessed trace commitments").

```yaml
ops:
  - id: predictor.block0.mlp.act        # FFN activation
    op: gelu                            # predictor FFN uses GELU
    mechanism: lookup                   # one of: lookup|piecewise_poly|newton_rsqrt|folded_affine
    approx:
      approx_id: gelu_q.lookup.v1
      input_scale_id: 12
      output_scale_id: 12
      input_domain: { lo: -512, hi: 511 }   # inclusive, in input-scale integer units
      output_domain: { lo: -64, hi: 511 }
      rounding: nearest_ties_to_even         # MUST equal quantization.default_rounding
      table_commitment: 0x...                # blake3 over canonical (x,y) rows
      max_tabulated_error: 1                 # units of output scale; 0 = exact tabulation
      golden_vectors_commitment: 0x...

  - id: predictor.action_embedder.act    # action Embedder activation
    op: silu                             # Embedder uses SiLU (NOT GELU)
    mechanism: lookup
    approx: { approx_id: silu_q.lookup.v1, ... }

  - id: predictor.block0.adaln.act       # AdaLN modulation MLP activation
    op: silu                             # AdaLN modulation uses SiLU (NOT GELU)
    mechanism: lookup
    approx: { approx_id: silu_q.lookup.v1, ... }

  - id: predictor.block0.attn.softmax
    op: softmax
    mechanism: lookup
    approx:
      approx_id: softmax_q.lookup.v1
      input_scale_id: 14                 # score scale after QK^T requant
      output_scale_id: 15                # probability scale (fixed-point, sums to unit)
      input_domain: { lo: -255, hi: 0 }  # shifted by row max -> non-positive
      exp_table_commitment: 0x...
      recip_mechanism: newton_rsqrt_free # softmax uses checked reciprocal, not rsqrt; see below
      recip_commitment: 0x...
      rounding: nearest_ties_to_even
      max_tabulated_error: 1
      golden_vectors_commitment: 0x...

  - id: predictor.block0.norm            # LayerNorm or AdaLN normalization
    op: layernorm                        # or: adaln
    mechanism: newton_rsqrt
    approx:
      approx_id: layernorm_q.newton.v1
      input_scale_id: 12
      output_scale_id: 12
      eps: 1                             # fixed-point epsilon in variance scale units
      newton_iters: 3                    # bounded, compile-time constant
      rsqrt_seed_table_commitment: 0x... # initial approximation table
      gamma_scale_id: 16
      beta_scale_id: 16
      rounding: nearest_ties_to_even
      max_tabulated_error: 2
      golden_vectors_commitment: 0x...
```

INV-RFC0006-01 (rounding agreement): every `approx.rounding` MUST equal
`quantization.default_rounding`. The manifest writer rejects a mismatch; the
verifier rejects a manifest whose op rounding diverges from the global mode
(there is exactly one active rounding mode per manifest, per §3 of the contract).

### Per-module activation selection (locked)

| Module | Activation | Rationale (verified against upstream at 2026-06-03) |
| --- | --- | --- |
| `predictor.block*.mlp` (FeedForward) | GELU (`gelu_q`) | predictor FFN/MLP uses GELU |
| `predictor.block*.adaln` (AdaLN-zero modulation MLP) | SiLU (`silu_q`) | AdaLN modulation uses SiLU |
| `predictor.action_embedder` (`Embedder`) | SiLU (`silu_q`) | action Embedder uses SiLU |

GELU and SiLU share one component (`activation_lookup.rs`) parameterized by
table/polynomial identity; the distinction is purely the committed approximation,
not the AIR layout. The activation functions are NOT uniform across the model;
stating the wrong activation per module produces a manifest whose
`activation_tables_commitment` does not match the exported reference and the
export parity test (RFC-0001) fails before any proof is attempted.

### GELU_Q / SiLU_Q component (`activation_lookup.rs`)

Definition for input `x` at `input_scale_id`, output `y` at `output_scale_id`:

```text
y = clamp(Table[approx_id](x), output_domain.lo, output_domain.hi)
```

`lookup` mechanism main-trace columns (one row per evaluated element):

```text
op_id        // selector binding this row to a specific ops[].id
x            // input element (BoundedInt value, range-checked to input_domain)
y            // output element (BoundedInt value, range-checked to output_domain)
```

Constraints:

```text
(x, y) ∈ ActivationTable[approx_id]          // LogUp membership (RFC-0003)
x ∈ [input_domain.lo, input_domain.hi]       // range check (RFC-0003)
y ∈ [output_domain.lo, output_domain.hi]     // range check
```

`piecewise_poly` mechanism (for wide domains where a full table is too large):

```text
seg          // segment index in [0, K)
c0..c_d      // committed coefficients for segment seg (preprocessed, selected by seg)
y_raw        // Horner evaluation = (((c_d * x + c_{d-1}) * x + ...) + c_0)
y            // requantize(y_raw, poly_scale -> output_scale), then clamp
```

Constraints:

```text
seg correctly selects x's segment:  seg_lo[seg] <= x <= seg_hi[seg]   // range-checked bracket
coefficients read from preprocessed poly table at row seg            // lookup membership
y_raw = Horner(c, x)                                                 // arithmetic, fully constrained
(y, q, rem) satisfy requantization with one active rounding mode      // RFC-0005 requant relation
y ∈ output_domain                                                    // range check
```

INV-RFC0006-02 (domain totality): the union of `[seg_lo[k], seg_hi[k]]` over all
`k` MUST cover `input_domain` with no gaps and no overlaps. Verified by the
manifest writer; out-of-domain inputs at proving time trigger
`PWM-TRACE-ACTIVATION-OUT-OF-DOMAIN` (see Failure modes).

### Softmax_Q component (in `attention.rs`)

Softmax over a row of `n` scores `s_0..s_{n-1}` (post-mask, post-scale). Masked
positions are set to a committed `NEG_INF_Q` sentinel before this primitive. The
relation is decomposed into committed, checked steps so no probability is
free-form:

```text
m       = max_j s_j                          // row max, witnessed
e_j     = ExpTable[approx_id](s_j - m)       // shifted exp via lookup; s_j - m <= 0
Z       = Σ_j e_j                            // denominator accumulator
p_j     = requantize(e_j * R, recip_scale)   // R = checked reciprocal of Z
```

Main-trace columns per `(row, j)`:

```text
op_id  row  j  s   shifted  e   is_max  Z_partial  R  p
```

Constraints:

```text
INV-RFC0006-03 (row max): exactly one j per row has is_max = 1, and
    for all j: s_max - s_j >= 0   (range-checked nonnegative difference)
shifted_j = s_j - s_max                       // shifted <= 0, range-checked
(shifted_j, e_j) ∈ ExpTable[approx_id]         // LogUp membership
Z = Σ_j e_j   via running accumulator Z_partial (RFC-0005 accumulate semantics)
INV-RFC0006-04 (checked reciprocal): R satisfies
    Z * R = ONE_Q + r_recip,  0 <= r_recip < Z   // exact division with remainder
p_j = requantize(e_j * R, recip_scale -> output_scale)   // one active rounding mode
INV-RFC0006-05 (probability sum): Σ_j p_j = ONE_Q + sum_slack,
    |sum_slack| <= n * rounding_ulp             // bounded by per-element rounding only
all of s, shifted, e, Z, R, p range-checked to their declared bounds
```

The reciprocal is NOT a lookup and NOT free: it is the checked back-multiplication
INV-RFC0006-04. This is the same shape as the Newton final back-check but applied
to plain reciprocal (`newton_rsqrt_free` in the manifest denotes "checked
reciprocal via back-multiplication, zero Newton iterations"). The exponential is
the only tabulated part, and its domain is restricted to non-positive shifted
scores so the table is bounded.

### LayerNorm_Q / AdaLN_Q component (`layernorm.rs`)

For a feature vector `x_0..x_{d-1}` with affine parameters `gamma`, `beta`:

```text
mean    = Σ x_i / d                          // exact integer mean with remainder
var     = Σ (x_i - mean)^2 / d               // exact, with remainder
inv_std ≈ 1 / sqrt(var + eps)                // newton_rsqrt mechanism
y_i     = requantize(gamma_i * (x_i - mean) * inv_std + beta_i)
```

`AdaLN_Q` is `LayerNorm_Q` with `gamma`, `beta` (and, for AdaLN-zero, a gating
scale) produced at runtime by a SiLU-activated modulation MLP rather than read
from static weights. The normalization core (mean, variance, `inv_std`) is
identical; only the source of the affine parameters differs. AdaLN therefore
reuses this component and additionally reads `gamma`/`beta`/`gate` from
`TensorCell`s produced by the modulation MLP (a `linear` + `silu` chain, defined
by RFC-0005 and the `silu_q` activation above).

Main-trace columns per feature row plus per-element rows:

```text
// per row:
op_id  row  sum_x  mean  q_mean  rem_mean  sum_sq  var  q_var  rem_var
       v(=var+eps)  y0  inv_std  newton_0 .. newton_{k}  back_check
// per element i:
x_i  centered_i(=x_i-mean)  gamma_i  beta_i  gate_i  prod_i  y_i
```

Constraints:

```text
sum_x = Σ x_i                                              // accumulate
mean: sum_x = mean * d + rem_mean, 0 <= rem_mean < d       // exact integer division
centered_i = x_i - mean
sum_sq = Σ centered_i^2                                    // accumulate of squares
var: sum_sq = var * d + rem_var, 0 <= rem_var < d
v = var + eps
INV-RFC0006-06 (rsqrt seed): y0 = RsqrtSeedTable[approx_id](v)   // LogUp membership
Newton refinement, k = newton_iters fixed (compile-time):
    for t in 0..k:  newton_{t+1} = requantize( newton_t * (3*ONE_Q - v * newton_t^2) / 2 )
    newton_0 = y0;  inv_std = newton_k
INV-RFC0006-07 (rsqrt back-check): inv_std^2 * v = ONE_Q + back_check,
    |back_check| <= rsqrt_tolerance         // tolerance declared in manifest, range-checked
y_i = requantize(gamma_i * centered_i * inv_std + beta_i)  // (gate_i applied for AdaLN-zero)
all intermediates range-checked
```

INV-RFC0006-07 is the soundness anchor: even though `inv_std` is computed by a
bounded Newton recurrence, its correctness does not depend on the recurrence
converging — it is independently pinned by the back-check `inv_std^2 * (var+eps)
≈ 1` within a committed tolerance. A prover supplying a wrong `inv_std` fails the
back-check regardless of how it was produced. The Newton iterations are present
only to make a valid `inv_std` cheap to witness; soundness comes from the
back-multiplication, not the iteration.

`folded_affine` mechanism (distilled path) replaces the entire core with
`y_i = requantize(a_i * x_i + b_i)` where `a`, `b` are committed folded constants.
This is mathematically valid only when the normalization is eval-mode affine
(folded BatchNorm) or a distilled RMSNorm-as-affine; the manifest writer asserts
the source op is foldable and records `mechanism: folded_affine`. Choosing this
mechanism means the proof is for the distilled model, bound by a distinct
`model_commitment`.

### Named invariants

| Invariant | Statement |
| --- | --- |
| INV-RFC0006-01 | Every op `approx.rounding` equals `quantization.default_rounding`; one active rounding mode per manifest. |
| INV-RFC0006-02 | Piecewise-poly segments tile `input_domain` with no gap or overlap. |
| INV-RFC0006-03 | Softmax row max: exactly one `is_max=1` per row; all `s_max - s_j >= 0`. |
| INV-RFC0006-04 | Softmax denominator reciprocal checked by `Z*R = ONE_Q + r`, `0 <= r < Z`. |
| INV-RFC0006-05 | Softmax probabilities sum to `ONE_Q` within `n * rounding_ulp`. |
| INV-RFC0006-06 | LayerNorm/AdaLN reciprocal-sqrt seed read from the committed seed table. |
| INV-RFC0006-07 | `inv_std^2 * (var+eps) = ONE_Q + back_check`, `|back_check| <= rsqrt_tolerance`. |
| INV-RFC0006-08 | Each op's `mechanism` and full `approx` block are bound by `model_commitment` and `quantization.activation_tables_commitment`; unbound approximations are rejected at manifest load. |

### Failure modes and system response

| Code | Trigger | System response |
| --- | --- | --- |
| `PWM-MANIFEST-APPROX-UNBOUND` | An op of type gelu/silu/softmax/layernorm/adaln lacks an `approx` block or its mechanism is not one of the four allowed | Manifest load fails in `pwm-export` and `pwm-core`; never proves. Maps to manifest error in [docs/spec/04-error-model.md#error-taxonomy](../spec/04-error-model.md#error-taxonomy). |
| `PWM-MANIFEST-ROUNDING-MISMATCH` | `approx.rounding != quantization.default_rounding` (INV-RFC0006-01) | Manifest load fails; export aborts. |
| `PWM-MANIFEST-POLY-DOMAIN-GAP` | Piecewise-poly segments do not tile `input_domain` (INV-RFC0006-02) | Manifest writer aborts export. |
| `PWM-TRACE-ACTIVATION-OUT-OF-DOMAIN` | A witnessed `x` lies outside the declared `input_domain` | Trace build fails in `pwm-prover` (no proof emitted); if forced, lookup membership fails and verifier returns `VerifyError::LookupConstraint`. |
| `PWM-VERIFY-LOOKUP` | Activation/exp/seed lookup membership unsatisfied at verify time | `verify` returns `VerifyError` lookup variant, [docs/spec/04-error-model.md#verifier-rejections](../spec/04-error-model.md#verifier-rejections). |
| `PWM-VERIFY-SOFTMAX-DENOM` | INV-RFC0006-04 unsatisfied (bad reciprocal) | `verify` rejects with a constraint-violation variant. |
| `PWM-VERIFY-SOFTMAX-SUM` | INV-RFC0006-05 unsatisfied (probabilities do not sum) | `verify` rejects. |
| `PWM-VERIFY-RSQRT-BACKCHECK` | INV-RFC0006-07 unsatisfied (`inv_std` not pinned) | `verify` rejects. |
| `PWM-VERIFY-RANGE` | Any nonlinear intermediate outside its declared bound | `verify` rejects with the range-check variant. |
| `PWM-VERIFY-TABLE-COMMITMENT` | Preprocessed table columns do not hash to the committed `*_commitment` | `verify` rejects before constraint evaluation. |

All verify-time codes are surfaced as the `VerifyError` enum
([docs/spec/02-public-api.md#rust-public-api](../spec/02-public-api.md#rust-public-api),
canonical signature `pub fn verify(artifact: &ProofArtifact) -> Result<(), VerifyError>`).

### Trace placement

Per [docs/spec/01-architecture.md#trace-model](../spec/01-architecture.md#trace-model):
table columns (`ActivationTable`, `ExpTable`, `RsqrtSeedTable`, piecewise-poly
coefficient rows, segment brackets) live in the preprocessed trace and are
committed; element values, accumulators, `inv_std`, reciprocal, probabilities,
and Newton intermediates live in the main trace; LogUp membership and range
checks live in the interaction trace with multiplicity columns per RFC-0003.

## Alternatives Considered

### Off-circuit nonlinearities (prover-supplied softmax / reciprocal / inv-sqrt)

What it is: the prover computes softmax probabilities, the LayerNorm reciprocal,
and GELU outputs in plain floating point or high-precision integer arithmetic
off-circuit and supplies them directly as witness values, with only the
surrounding linear algebra constrained. Why considered: it is the cheapest
possible AIR — no tables, no Newton, no back-checks. Why rejected: it is unsound
(feasibility study §6, §9.4). An unconstrained probability, reciprocal, or
inverse square root is a free parameter; a malicious prover sets it to whatever
makes a false `cost_s`/`selected_index` verify. This is the single most important
thing this RFC exists to forbid; it is on the rejected list permanently.

### Replace all nonlinearities with proof-native ones via mandatory distillation

What it is: require a distilled predictor in which softmax attention is replaced
by a linear/kernel attention, LayerNorm by RMSNorm-as-affine, and GELU by a cheap
polynomial — then only ever prove the distilled model. Why considered: it
minimizes trace size and avoids the exp table and Newton recurrence entirely; the
`folded_affine` mechanism here is exactly this path for normalization. Why
rejected as the *only* path: it forecloses proving the actual LeWorldModel
checkpoint, which is a stated capability the manifest must support
([docs/spec/03-data-model.md#model-manifest](../spec/03-data-model.md#model-manifest)
binds the real architecture). We keep distillation available as one mechanism
(`folded_affine`, and lighter activation tables) but do not mandate it; the
model-identity decision is RFC-0001's, not this RFC's.

### Generic scalar-gate lowering of every nonlinearity via stwo-circuits gates

What it is: express GELU/softmax/LayerNorm by lowering to the low-level
stwo-circuits gate set rather than dedicated AIR components. Verified against
upstream at 2026-06-03, that gate set is `Add, Sub, Mul, PointwiseMul, Eq,
TripleXor, M31ToU32, BlakeGGate, Permutation, Output` — there is no range or
bit-extraction gate (range checking is built from `sub`/`mul`/`assert_bits`
constraints, not a dedicated gate). Why considered: maximal reuse of the vendored
circuit substrate and its prover/verifier. Why rejected for the hot path: the
feasibility study (§3.2) and
[docs/spec/01-architecture.md#air-strategy](../spec/01-architecture.md#air-strategy)
fix direct custom AIR for dense/regular operators because scalar-gate lowering of
a per-row exp lookup, a denominator reduction, and a Newton recurrence is far
more expensive than a tabular LogUp + a handful of polynomial constraints. The
circuit gates remain appropriate for hashing/transcript/glue (RFC-0014/RFC-0015),
not for these primitives.

### Single shared activation table for the whole model

What it is: commit one activation table and use it for every nonlinearity,
assuming a uniform activation. Why considered: one commitment, one table. Why
rejected: the activations are not uniform — predictor FFN uses GELU while AdaLN
modulation and the action Embedder use SiLU (verified against upstream at
2026-06-03). A single shared table would either misrepresent SiLU modules as GELU
(parity failure) or be ambiguous about which curve applies where. Per-op
`mechanism`/`approx` with per-op `commitment` is required.

## Drawbacks

- Trace cost. Softmax adds a per-element exp lookup, a row-reduction, a checked
  reciprocal, and a per-element requant; LayerNorm/AdaLN add two exact integer
  divisions, a seed lookup, `newton_iters` refinement rows, and a back-check.
  These are heavier than the linear components they sit beside and dominate the
  per-block nonlinear cost analyzed in
  [docs/spec/08-performance-budget.md#cost-model](../spec/08-performance-budget.md#cost-model).
- Approximation, not float-exactness. The committed curve is what is proven; the
  proof says nothing about its distance from PyTorch float unless an external
  error bound is supplied. The `max_tabulated_error` and golden vectors document
  intent but are not a soundness guarantee about the original model.
- Table-size pressure. Lookup mechanisms bound the input domain tightly to keep
  tables small, which constrains the dynamic range the quantizer may use. Wide
  domains must use `piecewise_poly`, adding segment-selection constraints.
- Per-module commitments increase manifest surface and the number of
  commitments the verifier checks against the preprocessed trace.

## Migration / Rollout

- Relation versioning. These primitives participate in
  `pwm.lewm.predictor_step.v1` and downstream relations. Any change to a
  mechanism choice, table semantics, Newton iteration count, or rounding behavior
  is a semantic change that mints a new `relation_id` (immutable id rule,
  [docs/spec/09-release-and-versioning.md#relation-versioning](../spec/09-release-and-versioning.md#relation-versioning)).
  Approximation `approx_id`s are versioned (`gelu_q.lookup.v1`) and never mutated
  in place.
- Schema versioning. The `approx` block and per-module activation fields are
  introduced under `manifest_version: pwm-model-manifest-v1`; the additive
  `mechanism`/`approx` fields are gated by manifest schema version per
  [docs/spec/03-data-model.md#schema-versioning](../spec/03-data-model.md#schema-versioning).
- Feature flags. Component availability is staged by Cargo feature in `pwm-air`:
  `nl-activation` (GELU/SiLU lookup + piecewise-poly), `nl-softmax`,
  `nl-layernorm` (Newton path), `nl-folded-affine`. v0.2 ships `nl-activation`,
  `nl-softmax`, `nl-layernorm`; `nl-folded-affine` ships behind the same flag set
  but is only selectable when the manifest marks the source op foldable.
- Deprecation. Replacing a mechanism for an op (for example, moving a wide-domain
  GELU from `lookup` to `piecewise_poly`) is done by minting a new `approx_id`
  and a new `relation_id`; the old proof remains valid only for its declared id.
  No in-place table edits.

## Testing Strategy

Cross-references [docs/spec/07-testing-strategy.md](../spec/07-testing-strategy.md)
and [docs/spec/04-error-model.md#verifier-rejections](../spec/04-error-model.md#verifier-rejections).
No nonlinear component ships without both accepting and rejecting tests
(RFC-0013).

Accepting tests (golden vectors, [#golden-vectors](../spec/07-testing-strategy.md#golden-vectors)):

- `gelu_q_lookup_golden`: Rust `GELU_Q` lookup output matches the Python
  fixed-point reference bit-for-bit across the committed input domain.
- `silu_q_lookup_golden`: same for SiLU on the AdaLN-modulation and action-
  Embedder tables; asserts the GELU and SiLU tables are distinct commitments.
- `softmax_q_row_golden`: full softmax row (max-shift, exp lookup, denominator,
  reciprocal, probabilities) matches the reference; INV-RFC0006-05 sum holds.
- `layernorm_q_newton_golden`: LayerNorm `inv_std` from 3 Newton iterations
  satisfies INV-RFC0006-07; full normalized output matches the reference.
- `adaln_q_golden`: AdaLN-zero with runtime SiLU-MLP `gamma`/`beta`/`gate` matches
  the reference and reuses the LayerNorm core.
- `folded_affine_layernorm_golden`: distilled affine path equals
  `a*x+b` reference under a distinct `model_commitment`.
- `nonlinear_accept_proof`: a one-block predictor proof containing all four
  primitives verifies via `verify(artifact)` returning `Ok(())`.

Rejecting / negative tests ([#negative-tests](../spec/07-testing-strategy.md#negative-tests)),
each asserting the specific failure code:

- `reject_offcircuit_softmax`: probabilities supplied not equal to
  `e_j * R` -> `PWM-VERIFY-SOFTMAX-SUM` / `PWM-VERIFY-LOOKUP`.
- `reject_softmax_bad_reciprocal`: `R` not satisfying `Z*R = ONE_Q + r` ->
  `PWM-VERIFY-SOFTMAX-DENOM`.
- `reject_layernorm_inflated_invstd`: `inv_std` larger than true value but Newton
  rows internally consistent -> `PWM-VERIFY-RSQRT-BACKCHECK`.
- `reject_activation_out_of_domain`: input one ULP past `input_domain.hi` ->
  `PWM-TRACE-ACTIVATION-OUT-OF-DOMAIN` (build) / `PWM-VERIFY-RANGE` (forced).
- `reject_wrong_activation_table`: prove with SiLU table where manifest binds
  GELU -> `PWM-VERIFY-TABLE-COMMITMENT`.
- `reject_unbound_approx`: manifest with a softmax op lacking `approx` ->
  `PWM-MANIFEST-APPROX-UNBOUND` at load.
- `reject_rounding_mismatch`: op rounding diverges from global ->
  `PWM-MANIFEST-ROUNDING-MISMATCH`.
- `reject_poly_domain_gap`: piecewise-poly segments leave a gap ->
  `PWM-MANIFEST-POLY-DOMAIN-GAP`.

Differential tests ([#differential-tests](../spec/07-testing-strategy.md#differential-tests)):

- `diff_python_rust_nonlinear`: Python and Rust fixed-point references agree
  bit-for-bit on random inputs across all four primitives and both activations.

Constraint mutation tests ([#mutation-tests](../spec/07-testing-strategy.md#mutation-tests)):

- `mutate_drop_softmax_sum_constraint`: removing INV-RFC0006-05 must make
  `reject_offcircuit_softmax` pass (proving the constraint is load-bearing).
- `mutate_drop_rsqrt_backcheck`: removing INV-RFC0006-07 must make
  `reject_layernorm_inflated_invstd` pass.
- `mutate_drop_activation_range`: removing the output range check must make
  `reject_activation_out_of_domain` pass.

CI gates ([#ci-gates](../spec/07-testing-strategy.md#ci-gates)): all accepting,
rejecting, differential, and mutation tests above gate merges to `pwm-air`
nonlinear components.

## Open Questions

- OPEN QUESTION (owner: area:export maintainer; resolution: RFC-0001 at v0.2):
  whether `max_tabulated_error` is computed against the original float curve at
  export time and whether an exhaustive-tabulation error proof is attached. This
  RFC enforces the committed curve regardless; the error-vs-float relationship is
  export's to certify.
- OPEN QUESTION (owner: area:air maintainer; resolution: RFC-0007 at v0.2):
  the exact softmax score input domain bound for the V0 predictor sequence length
  (history-window attention is short, but the post-`QK^T` requant scale fixes the
  domain). Resolved jointly with the predictor block layout, which sets the score
  scale.
- OPEN QUESTION (owner: area:air maintainer; resolution:
  [docs/spec/08-performance-budget.md#cost-model](../spec/08-performance-budget.md#cost-model)
  during v0.2): whether `newton_iters = 3` is the minimum that satisfies
  INV-RFC0006-07 within `rsqrt_tolerance` for the V0 variance range, or whether 2
  suffices. The back-check makes any choice sound; this is a cost question only.

## References

- [docs/feasibility-study.md](../feasibility-study.md), §6 (nonlinear
  operations), §9.4 (approximation requirements), §3.2 (direct AIR vs circuit
  lowering), §13 (risk register: softmax cost, LayerNorm cost).
- [docs/spec/01-architecture.md#component-model](../spec/01-architecture.md#component-model),
  [#air-strategy](../spec/01-architecture.md#air-strategy),
  [#trace-model](../spec/01-architecture.md#trace-model).
- [docs/spec/02-public-api.md#rust-public-api](../spec/02-public-api.md#rust-public-api)
  (`verify` / `VerifyError`).
- [docs/spec/03-data-model.md#model-manifest](../spec/03-data-model.md#model-manifest),
  [#bounded-integers](../spec/03-data-model.md#bounded-integers),
  [#tensor-memory-cells](../spec/03-data-model.md#tensor-memory-cells),
  [#schema-versioning](../spec/03-data-model.md#schema-versioning).
- [docs/spec/04-error-model.md#error-taxonomy](../spec/04-error-model.md#error-taxonomy),
  [#verifier-rejections](../spec/04-error-model.md#verifier-rejections).
- [docs/spec/06-security.md#soundness-requirements](../spec/06-security.md#soundness-requirements).
- [docs/spec/07-testing-strategy.md#golden-vectors](../spec/07-testing-strategy.md#golden-vectors),
  [#negative-tests](../spec/07-testing-strategy.md#negative-tests),
  [#differential-tests](../spec/07-testing-strategy.md#differential-tests),
  [#mutation-tests](../spec/07-testing-strategy.md#mutation-tests),
  [#ci-gates](../spec/07-testing-strategy.md#ci-gates).
- [docs/spec/08-performance-budget.md#cost-model](../spec/08-performance-budget.md#cost-model).
- [docs/spec/09-release-and-versioning.md#relation-versioning](../spec/09-release-and-versioning.md#relation-versioning).
- [docs/rfcs/RFC-0001-model-manifest-and-export-pipeline.md](RFC-0001-model-manifest-and-export-pipeline.md),
  [docs/rfcs/RFC-0002-fixed-point-arithmetic-over-m31.md](RFC-0002-fixed-point-arithmetic-over-m31.md),
  [docs/rfcs/RFC-0003-range-check-and-lookup-infrastructure.md](RFC-0003-range-check-and-lookup-infrastructure.md),
  [docs/rfcs/RFC-0005-linear-matmul-and-requantization-components.md](RFC-0005-linear-matmul-and-requantization-components.md),
  [docs/rfcs/RFC-0007-leworldmodel-predictor-air.md](RFC-0007-leworldmodel-predictor-air.md),
  [docs/rfcs/RFC-0011-pixel-encoder-proof.md](RFC-0011-pixel-encoder-proof.md),
  [docs/rfcs/RFC-0013-testing-fuzzing-and-audit-strategy.md](RFC-0013-testing-fuzzing-and-audit-strategy.md),
  [docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md](RFC-0014-canonical-serialization-and-transcript.md),
  [docs/rfcs/RFC-0015-third-party-vendoring-and-pinning.md](RFC-0015-third-party-vendoring-and-pinning.md).
- LeWorldModel (github.com/lucas-maes/le-wm, MIT): `ARPredictor`,
  `ConditionalBlock` (AdaLN-zero), `FeedForward` (GELU), `Embedder` (SiLU),
  `F.scaled_dot_product_attention`; verified against upstream at 2026-06-03.
- LogUp lookups in the vendored Stwo constraint framework
  (`constraint-framework/src/logup.rs`); verified against upstream at 2026-06-03.
