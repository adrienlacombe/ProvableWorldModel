# RFC-0003: Range-check and lookup infrastructure

- Status: Accepted
- Authors: ProvableWorldModel maintainers
- Created: 2026-06-03
- Target milestone: v0.1

## Summary

This RFC locks the reusable LogUp-based lookup infrastructure on which every
quantized arithmetic relation in ProvableWorldModel depends. The decision: range
safety and table membership are enforced by a single, shared family of LogUp
relations carried on the interaction trace, not by per-component ad hoc
constraints and not by a dedicated prover gate. We define a fixed catalogue of
preprocessed lookup tables (`u8`, `i8`, `u16`, a parameterized bounded-signed-limb
table, an activation table, and optional `requant`/`softmax`/`inv_sqrt` tables),
a uniform `RangeCheck` API (`pwm-air`) that any component calls to bind a column
to a table, and a multiplicity-balance discipline that the verifier checks via the
LogUp claimed-sum being zero. Critically, the vendored `stwo-circuits` workspace
exposes no range/bit-extraction gate (verified against upstream at 2026-06-03):
its first-class gate set is `Add, Sub, Mul, PointwiseMul, Eq, TripleXor, M31ToU32,
BlakeGGate, Permutation, Output`, and `extract_bits()` is a helper built from
`sub`/`mul`/`assert_bits` constraints. Range checking in this project is therefore
a LogUp/constraint concern owned by `pwm-air`, layered on Stwo's
`constraint-framework` LogUp, never delegated to a circuit gate.

## Motivation

Field arithmetic over M31 wraps; integer ML inference must not. Every value the
AIR interprets as a bounded integer must be proven to lie in its declared range,
or a malicious prover can choose a field element that satisfies the algebraic
constraints while representing a different integer (the field-wraparound attack;
see `docs/spec/06-security.md#soundness-requirements` and the risk register in
`docs/feasibility-study.md` §13). The founding analysis enumerates the values that
require range checks (`docs/feasibility-study.md` §9.3): input latents, actions,
weights, biases, products, accumulators, requantization quotients and remainders,
activation outputs, attention scores, softmax probabilities, LayerNorm
intermediates, cost differences, and argmin comparison witnesses. Nonlinear
primitives additionally require table-driven evaluation: GELU, softmax, and
inverse-sqrt are evaluated by lookup against committed tables
(`docs/feasibility-study.md` §6.1, §6.3, §7.4). Tensor read/write consistency
(RFC-0004) and weight-table membership (RFC-0005) are also lookup relations.

All of these reduce to the same primitive: "prove that a column of witness values
is a multiset-subset of a known table, with checked multiplicities." Rather than
re-deriving this in `linear.rs`, `requant.rs`, `activation_lookup.rs`,
`layernorm.rs`, `attention.rs`, `cost.rs`, and `argmin.rs`, this RFC factors it
into one audited subsystem. This is the v0.1 dependency that unblocks RFC-0002
(fixed-point arithmetic — `docs/rfcs/RFC-0002-fixed-point-arithmetic-over-m31.md`),
RFC-0004 (tensor memory), and RFC-0005 (linear/matmul/requant). It expands source
RFC-003 and `docs/feasibility-study.md` §7.4.

Concrete system scenario: a `LinearComponent` accumulates a length-2048 int8 dot
product. The accumulator column `acc` must be proven in `[-B_acc, B_acc]`. The
component does not hand-roll a bit decomposition; it calls
`RangeCheck::bound(acc, table = I8_LIMB(width))`, which appends `acc` (and any
limb decomposition the table requires) to the relevant LogUp relation. At proving
time the multiplicities of every table entry are tallied; at verification time the
LogUp claimed sum for that relation must be zero. One subsystem, every component,
one audit surface.

## Goals

- Define a fixed, schema-versioned catalogue of lookup tables with explicit
  domains, encodings, and table-construction rules, committed under
  `activation_tables_commitment` in the manifest where applicable.
- Provide a single typed `pwm-air` API (`RangeCheck`, `LookupBus`) that any AIR
  component uses to bind a column to a table; no component implements its own
  range argument.
- Specify the LogUp relation layout (preprocessed table columns, main-trace value
  columns, interaction-trace fraction columns, multiplicity columns) and the
  exact balance equation the verifier checks.
- Specify the multiplicity-accounting rule and the claimed-sum-equals-zero
  verifier check that makes membership sound, and name the invariants that make
  it so.
- Specify the bounded-signed-limb decomposition so that values exceeding a single
  small table's domain (accumulators, squared costs) are range-checked compositionally.
- Enumerate every failure mode (out-of-domain witness, multiplicity mismatch,
  wrong table, uncommitted table, malformed limb decomposition, non-zero claimed
  sum) with the system's response and the named verifier rejection.

## Non-Goals

- This RFC does not define the per-tensor numeric bounds themselves (`BoundedInt.lo`,
  `BoundedInt.hi`) or the add/sub/mul/requantize semantics — that is RFC-0002
  (`docs/rfcs/RFC-0002-fixed-point-arithmetic-over-m31.md`). This RFC consumes
  those bounds and proves membership.
- It does not define the activation/softmax/inv-sqrt approximations (domain,
  polynomial, error bound) — that is RFC-0006
  (`docs/rfcs/RFC-0006-nonlinear-primitive-components.md`). This RFC defines only
  the lookup mechanism those tables ride on, and the rule that the table bytes are
  committed.
- It does not define the tensor-memory read/write multiset argument
  (RFC-0004) or the weight-table membership argument (RFC-0005); those are
  consumers of the `LookupBus` defined here, with their own RFCs for relation
  shape.
- It does not define the canonical Fiat-Shamir transcript or the LogUp challenge
  derivation order — that is RFC-0014
  (`docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md`). This RFC
  states which challenges it consumes and in what order relative to that channel.
- It does not implement CEM top-k membership lookups; those are deferred to
  RFC-0010 (Future) and reuse this infrastructure.

## Proposed Design

### Decision (locked)

ProvableWorldModel enforces all range and table-membership constraints through a
single LogUp-based lookup subsystem in `pwm-air`, built on the vendored Stwo
`stwo-constraint-framework` LogUp (`constraint-framework/src/logup.rs`, verified
against upstream at 2026-06-03). The subsystem provides a fixed catalogue of
tables and one uniform binding API. There is no separate range gate: the vendored
`stwo-circuits` first-class gate set is `Add, Sub, Mul, PointwiseMul, Eq,
TripleXor, M31ToU32, BlakeGGate, Permutation, Output`, with `extract_bits()` a
helper over `sub`/`mul`/`assert_bits` (verified against upstream at 2026-06-03).
Range checking is a LogUp/constraint relation owned by `pwm-air`, never a
circuit-gate concern; `pwm-circuits` lookups (RFC-0015) are reserved for hashing
and transcript glue.

### Table catalogue (locked)

The catalogue is a closed enum. Adding a table is a schema change governed by
`docs/spec/09-release-and-versioning.md#relation-versioning`. Tables are either
**static** (domain known a priori, committed implicitly by the relation_id and
the AIR source) or **manifest-committed** (bytes bound by
`activation_tables_commitment`, per `docs/spec/03-data-model.md#model-manifest`).

| Table id | Kind | Domain | Encoding | Committed by | Used by |
|----------|------|--------|----------|--------------|---------|
| `U8` | static | `[0, 255]` | one column, value = entry | relation_id + AIR | byte limbs, masks |
| `I8` | static | `[-128, 127]` | centered: stores `v + 128` in `[0,255]` | relation_id + AIR | int8 weights/activations |
| `U16` | static | `[0, 65535]` | one column, value = entry | relation_id + AIR | u16 activations, 16-bit limbs |
| `I8_LIMB(width w)` | static (parameterized) | signed value in `[-(2^w-1)·K, (2^w-1)·K]` via `w`-bit limbs | limb columns each bound to `U8`/`U16`; sign column in `{0,1}` | relation_id + AIR | accumulators, products, squared diffs, costs |
| `ACTIVATION(id)` | manifest-committed | declared in RFC-0006 op spec | `(x, y)` pair columns | `activation_tables_commitment` | GELU_Q FFN/MLP, SiLU for AdaLN + action Embedder |
| `REQUANT(id)` (optional, v0.2) | manifest-committed | `(acc_high_bits, q)` | pair columns | `activation_tables_commitment` | requant fast path (RFC-0005) |
| `SOFTMAX(id)` (optional, v0.2) | manifest-committed | declared in RFC-0006 | `(score_q, exp_q)` pair columns | `activation_tables_commitment` | attention softmax (RFC-0006) |
| `INV_SQRT(id)` (optional, v0.2) | manifest-committed | declared in RFC-0006 | `(var_q, inv_std_q)` pair columns | `activation_tables_commitment` | LayerNorm/AdaLN (RFC-0006) |

Notes that are load-bearing and must not be lost:

- `I8` uses a centered encoding (`stored = v + 128`) so the static table column is
  `U8`-shaped; the consuming component proves `value == stored - 128` with one
  linear constraint. This mirrors the centered-into-M31 encoding decided in
  RFC-0002 and §3 of the contract.
- Per-module activation correction: the predictor FFN/MLP uses GELU, while AdaLN
  modulation and the action `Embedder` use SiLU (verified against upstream at
  2026-06-03). The `ACTIVATION(id)` table therefore carries a per-op function
  identity; there is no single global activation table. RFC-0006 owns the
  per-module mapping; this RFC owns only that each is a distinct committed table.
- `REQUANT`, `SOFTMAX`, `INV_SQRT` are optional and gated to v0.2 (they support
  nonlinear primitives, RFC-0006, milestone v0.2). v0.1 ships `U8`, `I8`, `U16`,
  `I8_LIMB`, and the lookup-bus mechanism; the activation/softmax/inv-sqrt tables
  ride the same mechanism once RFC-0006 lands.

### LogUp relation layout

A lookup relation proves that a multiset of witness tuples is contained in a table
multiset. Following Stwo's LogUp (`constraint-framework/src/logup.rs`), each side
contributes rational fractions `m / (z - combine(tuple, alpha))` summed over rows,
where `z` and `alpha` are QM31 Fiat-Shamir challenges and `combine` is the random
linear combination of the tuple's field entries. The relation is sound when the
total sum over table rows (with multiplicities) minus the total sum over witness
rows (each with multiplicity 1) equals zero.

Column roles per relation:

```text
Preprocessed trace (fixed, agreed by prover and verifier):
  table_value[k]        // table entry columns (1 col for U8/U16, w cols for limb,
                        // 2 cols for ACTIVATION/SOFTMAX/INV_SQRT/REQUANT pairs)

Main trace (witness):
  looked_up_value[k]    // the value(s) a consuming component asserts are in-table
  multiplicity[t]       // per table row t: how many witness rows looked it up

Interaction trace (LogUp, QM31):
  table_fraction        // m_t / (z - combine(table_row_t, alpha))
  use_fraction          // 1   / (z - combine(witness_row,  alpha))
  running_sum           // cumulative LogUp sum; last value is the claimed sum
```

Balance equation (the soundness core), with `T` table rows and `W` witness rows:

```text
sum_{t in 0..T} multiplicity[t] / (z - combine(table_row[t], alpha))
  - sum_{w in 0..W} 1 / (z - combine(witness_row[w], alpha))
  == 0
```

The verifier checks `running_sum.last() == 0` for every relation. Because `z` and
`alpha` are unpredictable, equality of the two rational sums holds with
overwhelming probability only if the witness multiset is genuinely contained in
the table multiset with the claimed multiplicities (Schwartz–Zippel over QM31).

### Multiplicity accounting (locked rule)

For each lookup relation, the prover MUST populate `multiplicity[t]` as exactly
the count of witness rows whose `combine`d tuple equals table row `t`. The trace
builder computes these by tallying during witness construction; it does not trust
caller-supplied counts. The named invariant INV-RC-02 below makes a mismatch
detectable: if a prover under- or over-counts any `multiplicity[t]`, the two
rational sums differ and `running_sum.last() != 0`, so the verifier rejects.
Multiplicities are themselves range-checked into `U16` (or `U8_LIMB` for large
fan-out) to prevent a prover from encoding a negative or wrapped count.

### Public `pwm-air` interface (locked)

```rust
/// The set of lookup relations in a proof. One bus per proof; components register
/// their lookups against it; the trace builder finalizes it into preprocessed,
/// main, and interaction trace columns.
pub struct LookupBus {
    relations: Vec<LookupRelation>,
}

/// A closed catalogue of tables. Adding a variant is a schema change
/// (docs/spec/09-release-and-versioning.md#relation-versioning).
#[non_exhaustive]
pub enum Table {
    U8,
    I8,
    U16,
    /// `width` = bits per limb (8 or 16); `num_limbs` derived from the bound.
    I8Limb { width: u8, num_limbs: u8 },
    /// Manifest-committed; `table_id` indexes the op-specific table whose bytes
    /// are bound by `activation_tables_commitment`.
    Activation { table_id: u32 },
    Requant { table_id: u32 },     // v0.2
    Softmax { table_id: u32 },     // v0.2
    InvSqrt { table_id: u32 },     // v0.2
}

/// Uniform range/membership API. Components call these; they never write LogUp
/// columns directly.
pub struct RangeCheck<'a> { bus: &'a mut LookupBus }

impl<'a> RangeCheck<'a> {
    /// Bind a single bounded-integer column to its smallest fitting table,
    /// decomposing into limbs when `bound` exceeds the table domain. The bound is
    /// taken from the column's BoundedInt (docs/spec/03-data-model.md#bounded-integers).
    pub fn bound(&mut self, value: ColumnRef, bound: &BoundedInt) -> RangeWitness;

    /// Bind a (input, output) pair to a manifest-committed table.
    pub fn lookup_pair(&mut self, input: ColumnRef, output: ColumnRef,
                       table: Table) -> LookupWitness;

    /// Bind an explicit limb decomposition (used by linear/cost accumulators that
    /// already carry limbs in their own trace layout).
    pub fn bound_limbs(&mut self, limbs: &[ColumnRef], sign: ColumnRef,
                       width: u8) -> RangeWitness;
}
```

`RangeWitness` and `LookupWitness` are the per-relation witness records already
declared in the canonical `Witness` (contract §6.3,
`docs/spec/03-data-model.md#witness`):

```rust
// Carried in Witness.range_witnesses and Witness.lookup_witnesses.
pub struct RangeWitness {
    pub relation: u32,            // which LogUp relation this row contributes to
    pub table: TableTag,          // discriminant of Table
    pub limbs: Vec<M31>,          // limb decomposition (empty for single-column tables)
    pub sign: M31,                // 0 or 1; 0 for unsigned tables
}

pub struct LookupWitness {
    pub relation: u32,
    pub table_id: u32,            // manifest table index (Activation/Requant/Softmax/InvSqrt)
    pub input: M31,
    pub output: M31,
}
```

These signatures do not contradict contract §6; they expand `RangeWitness` and
`LookupWitness`, which §6 names but leaves to the owning subsystem (this RFC) and
to `docs/spec/03-data-model.md#witness` to format.

### Bounded-signed-limb decomposition

A value `x` with declared bound `|x| <= B` that exceeds a single small table's
domain is range-checked compositionally. For limb width `w` and `L` limbs:

```text
x = sign * sum_{l in 0..L} limb[l] * 2^(w*l)
sign in {0, 1}                         // 0 encodes nonnegative, 1 encodes negative
limb[l] in [0, 2^w - 1]                // each limb bound to U8 (w=8) or U16 (w=16)
sum_{l} limb[l] * 2^(w*l) <= B         // top-limb range narrowed so the recomposed
                                       // magnitude cannot exceed B
```

Constraints added by `bound_limbs`:

```text
recompose:  magnitude == sum_l limb[l] * 2^(w*l)
sign_bool:  sign * (sign - 1) == 0
value_eq:   x == (1 - 2*sign) * magnitude   // value carried by the calling component
each limb[l] looked up in U8/U16 via the bus (multiplicity-counted)
```

`L` is derived from `B`: `L = ceil(log2(B+1) / w)`, with the most-significant limb's
table choice narrowed when `B` is not a clean power of `2^w` (a `bound`-specific
sub-table or an additional `cost_s - B - 1 >= 0`-style range check on the top limb,
mirroring the argmin comparison pattern in RFC-0002/RFC-0009). This composition is
how accumulators (RFC-0005), squared MSE diffs and running costs (RFC-0009 cost
component), and argmin comparison witnesses (RFC-0009) are range-checked without
any wide native table.

### Data flow and lifecycle

```text
1. Component (linear, requant, cost, ...) calls RangeCheck::bound / lookup_pair /
   bound_limbs during trace construction, naming a column and its BoundedInt bound.
2. RangeCheck appends the value (and limbs/sign) to the relevant LookupRelation and
   records a RangeWitness/LookupWitness into the Witness.
3. trace_builder (pwm-prover) finalizes the LookupBus:
     - emits preprocessed table columns for static tables,
     - loads manifest-committed table columns for Activation/Requant/Softmax/InvSqrt
       and checks their bytes against activation_tables_commitment,
     - tallies multiplicity[t] for every relation,
     - range-checks the multiplicity columns themselves.
4. After committing main + preprocessed traces, the prover draws z, alpha from the
   Fiat-Shamir channel in the order fixed by RFC-0014, then builds the LogUp
   interaction trace fractions and running sums.
5. Verifier redraws z, alpha in the identical order and checks running_sum.last()
   == 0 for every relation, plus the algebraic constraints (recompose, sign_bool,
   value_eq, centered-encoding linear constraints).
```

Determinism: limb decomposition is canonical (least-significant-first, fixed `w`
per table, `L` derived deterministically from `B`); multiplicity tallying is over a
deterministic row order. Two runs on identical inputs produce identical lookup
traces, satisfying the reproducibility contract in RFC-0016
(`docs/rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md`).

### Invariants

- **INV-RC-01 (total range coverage).** Every main-trace column whose
  `BoundedInt` declares a finite `[lo, hi]` is bound to exactly one range relation
  via `RangeCheck`. No bounded-integer column reaches the constraint evaluator
  without a registered `RangeWitness`. Enforced by a trace-builder assertion that
  the set of declared bounded columns equals the set of range-checked columns.
- **INV-RC-02 (multiplicity faithfulness).** For every lookup relation,
  `multiplicity[t]` equals the exact count of witness rows mapping to table row
  `t`; the LogUp claimed sum is zero iff this holds. The trace builder computes
  multiplicities; it never accepts them from a component.
- **INV-RC-03 (domain closure).** A `lookup_pair` against a manifest-committed
  table is valid only if the `(input, output)` tuple is a row of the table whose
  bytes hash to `activation_tables_commitment`. An out-of-domain or off-table
  tuple makes the claimed sum non-zero.
- **INV-RC-04 (table commitment binding).** Static table contents are fixed by the
  AIR source and the `relation_id`; manifest-committed table contents are bound by
  `activation_tables_commitment`. No table column is prover-mutable at proving
  time. (See `docs/spec/06-security.md#binding-requirements`.)
- **INV-RC-05 (limb soundness).** A limb decomposition is accepted only if
  `recompose`, `sign_bool`, and `value_eq` hold and every limb is in its base
  table; the most-significant limb is additionally narrowed so the recomposed
  magnitude cannot exceed the declared bound `B`. This prevents a prover from
  representing an out-of-bound integer with in-range limbs.
- **INV-RC-06 (multiplicity range safety).** Every `multiplicity[t]` column is
  itself range-checked into `U16` (or limb-decomposed for larger fan-out), so a
  count cannot wrap mod p to forge a balance.

### Failure modes and system response

| # | Failure | Where detected | System response | Rejection |
|---|---------|----------------|-----------------|-----------|
| F1 | Witness value outside its table domain | Verifier: LogUp claimed sum != 0 | Reject proof | `VerifyError::LookupClaimedSumNonZero` (docs/spec/04-error-model.md#verifier-rejections) |
| F2 | Prover miscounts a `multiplicity[t]` | Verifier: claimed sum != 0 | Reject proof | `VerifyError::LookupClaimedSumNonZero` |
| F3 | Bounded column never registered with a range relation | Prover: INV-RC-01 trace-builder assertion | Abort proving; emit `TraceError::UnboundedColumn` | n/a (caught pre-proof) |
| F4 | Manifest-committed table bytes do not hash to `activation_tables_commitment` | Prover: step 3 commitment check; Verifier: preprocessed commitment mismatch | Abort proving / reject proof | `VerifyError::TableCommitmentMismatch` |
| F5 | Malformed limb decomposition (recompose/sign/value_eq fails) | Verifier: algebraic constraint | Reject proof | `VerifyError::ConstraintUnsatisfied` |
| F6 | Limbs in-range but recomposed magnitude exceeds `B` | Verifier: top-limb narrowing constraint (INV-RC-05) | Reject proof | `VerifyError::ConstraintUnsatisfied` |
| F7 | Multiplicity column value wrapped/negative | Verifier: U16 range relation on multiplicities (INV-RC-06) | Reject proof | `VerifyError::LookupClaimedSumNonZero` |
| F8 | Lookup against a table not in the catalogue / wrong `relation` id | Prover: `Table` is a closed enum; unknown id rejected by builder | Abort proving | `TraceError::UnknownTable` |
| F9 | `z`/`alpha` drawn in the wrong order vs RFC-0014 | Verifier: redrawn challenges differ; sum check fails | Reject proof | `VerifyError::TranscriptMismatch` |

All `VerifyError` variants are owned by `docs/spec/04-error-model.md#verifier-rejections`;
`TraceError` variants by `docs/spec/04-error-model.md#failure-modes`. This RFC
contributes the lookup-specific variants listed above.

## Alternatives Considered

### A. A dedicated range/bit-extraction gate in `pwm-circuits`

Express each range check as a circuit gate analogous to a hypothetical
`stwo-circuits` range primitive, lowering bounded columns to bit-extraction
gadgets in the circuit IR. Rejected for two reasons. First, factually: the
vendored `stwo-circuits` workspace has no such gate (verified against upstream at
2026-06-03 — the first-class set is `Add, Sub, Mul, PointwiseMul, Eq, TripleXor,
M31ToU32, BlakeGGate, Permutation, Output`; `extract_bits()` is a helper over
`sub`/`mul`/`assert_bits`, not a gate). Adopting it would mean inventing and
auditing a new gate, contradicting RFC-0015's pin-and-minimally-modify policy.
Second, economically: bit-extracting every bounded value in a length-2048
accumulator dominates the trace, where a single shared LogUp table amortizes the
cost across all consumers. Direct AIR + LogUp is the route the founding analysis
recommends for hot paths (`docs/feasibility-study.md` §3.2).

### B. Per-component bespoke range constraints (no shared subsystem)

Let each component (`linear`, `requant`, `cost`, `argmin`) implement its own range
argument inline. Rejected: it multiplies the soundness-critical surface by the
number of components, makes the "every bounded column is range-checked" invariant
(INV-RC-01) unverifiable centrally, and guarantees subtle divergence (one
component forgetting the multiplicity check, another using a different limb width).
A field-wraparound bug in any single component is a total soundness break
(`docs/feasibility-study.md` §13, severity Critical). One audited subsystem with
one API is the controllable surface.

### C. Permutation-argument range checks instead of LogUp

Enforce membership with a multiset/permutation argument (sorted-column or
grand-product) rather than LogUp fractions. Rejected for v0.1: LogUp is already
the primitive Stwo's `constraint-framework` provides and is shared with the
tensor-memory consistency argument (RFC-0004) and weight-table membership
(RFC-0005), so standardizing on it minimizes the audited cryptographic surface and
challenge-derivation complexity. A sorted-column permutation range check also
forces an additional sortedness constraint and a contiguity argument, which is
more trace and more places to be wrong than a table lookup with multiplicities.
Permutation arguments remain the right tool for read/write ordering in RFC-0004,
where the relation is genuinely a permutation, not a subset.

## Drawbacks

- The LogUp subsystem adds an interaction trace and two QM31 challenges, increasing
  proof size and prover work beyond the main trace. This is the standard cost of
  range safety; it is not optional given field wraparound.
- Manifest-committed tables (`ACTIVATION`/`SOFTMAX`/`INV_SQRT`) couple this
  subsystem to RFC-0006's approximation decisions; a change to an activation table
  remints `activation_tables_commitment` and therefore the model commitment. This
  is intended binding, but it means table edits are not free.
- Wide accumulators force limb decomposition, which adds columns and constraints
  proportional to limb count. For the int8/length-2048 path the worst-case dot
  product `2048 * 127 * 127 = 33,032,192 < 2^31 - 1` fits a single M31 and needs no
  limbs (`docs/feasibility-study.md` §5.2), but int16 activations or accumulated
  costs will pay the limb cost.
- A closed `Table` enum means new table kinds require a schema/relation version
  bump rather than a drop-in addition. This is deliberate (binding > convenience).

## Migration / Rollout

This is a v0.1 foundational subsystem with no prior version to migrate from; the
rollout discipline governs future change.

- **Feature gating.** v0.1 ships `U8`, `I8`, `U16`, `I8Limb`, and the
  `LookupBus`/`RangeCheck` API. The `Requant`, `Softmax`, and `InvSqrt` variants
  are present in the `Table` enum but gated behind a `nonlinear-tables` cargo
  feature (off by default) until RFC-0006 lands in v0.2; constructing a gated
  variant without the feature is a compile error in `pwm-air`.
- **Relation versioning.** The lookup-table catalogue and the LogUp column layout
  are part of every `relation_id` (`pwm.lewm.<statement>.v<N>`,
  contract §4). Any change to a static table's contents, the limb-decomposition
  rule, or the catalogue is a semantic change that mints a new `relation_id`
  per `docs/spec/09-release-and-versioning.md#relation-versioning`; old proofs
  remain valid only against their original id.
- **Schema versioning.** `RangeWitness`/`LookupWitness` are versioned with the
  rest of the `Witness` under `manifest_version` and the artifact
  `artifact_version` (`docs/spec/03-data-model.md#schema-versioning`).
- **Deprecation.** A superseded table kind is marked deprecated in the catalogue
  for one minor release with a changelog entry
  (`docs/spec/09-release-and-versioning.md#deprecation`,
  `docs/spec/09-release-and-versioning.md#changelog`) before removal; removal
  remints the relation_id.
- **Vendoring boundary.** The LogUp primitive comes from vendored
  `stwo-constraint-framework` at the revision pinned in
  `third_party/stwo/REVISION` (RFC-0015,
  `docs/rfcs/RFC-0015-third-party-vendoring-and-pinning.md`). Bumping that revision
  is a separate, reviewed change with its own re-audit.

## Testing Strategy

Cross-references `docs/spec/07-testing-strategy.md` and
`docs/spec/04-error-model.md#verifier-rejections`. No component ships without both
accepting and rejecting tests (RFC-0013).

Accepting tests (`#test-pyramid`, `#golden-vectors`):

- `range_u8_full_table_accepts` — a witness column hitting every value in `[0,255]`
  with assorted multiplicities verifies; claimed sum is zero.
- `range_i8_centered_accepts` — int8 values across `[-128,127]` verify via the
  `I8` centered encoding (`value == stored - 128`).
- `range_u16_accepts` — boundary values `{0, 1, 65534, 65535}` verify.
- `limb_decomposition_accepts` — a 32-bit signed accumulator near
  `±33,032,192` verifies through `I8Limb { width: 16, num_limbs: 2 }` with
  `recompose`/`sign_bool`/`value_eq` satisfied.
- `multiplicity_high_fanout_accepts` — a single table row looked up 4096 times
  tallies and balances (multiplicity range-checked into `U16`).
- `activation_pair_lookup_accepts` — a GELU `(x,y)` pair from the v0.2 committed
  table verifies and matches the manifest `activation_tables_commitment` (golden
  vector shared with RFC-0006).
- `lookup_bus_golden_vector` — a fixed multi-relation bus (range + activation +
  limb) reproduces a byte-identical interaction trace across two runs
  (reproducibility, RFC-0016).

Rejecting / negative tests (`#negative-tests`, `#mutation-tests`), each asserting
the named `VerifyError`:

- `reject_value_above_u8_domain` — witness value `256` against `U8`; expect
  `LookupClaimedSumNonZero` (F1).
- `reject_undercounted_multiplicity` — decrement one `multiplicity[t]` by 1; expect
  `LookupClaimedSumNonZero` (F2).
- `reject_overcounted_multiplicity` — increment one `multiplicity[t]` by 1; expect
  `LookupClaimedSumNonZero` (F2).
- `reject_unbounded_column_pre_proof` — register a bounded column without a range
  relation; expect `TraceError::UnboundedColumn` at trace-build time (F3,
  INV-RC-01).
- `reject_tampered_activation_table` — flip one byte of a committed activation
  table; expect `TableCommitmentMismatch` (F4, INV-RC-04).
- `reject_malformed_limb_decomposition` — set limbs that violate `value_eq`; expect
  `ConstraintUnsatisfied` (F5).
- `reject_in_range_limbs_over_bound` — limbs each in-range but recomposed magnitude
  `> B`; expect `ConstraintUnsatisfied` from top-limb narrowing (F6, INV-RC-05).
- `reject_wrapped_multiplicity` — set a `multiplicity[t]` to a value that wraps mod
  p; expect `LookupClaimedSumNonZero` via the `U16` multiplicity check (F7,
  INV-RC-06).
- `reject_unknown_table_tag` — request a `Table` discriminant outside the
  catalogue; expect `TraceError::UnknownTable` (F8).
- `reject_transcript_reorder` — draw `z`/`alpha` before committing the main trace
  (wrong order vs RFC-0014); expect `TranscriptMismatch` (F9).

Differential / mutation tests (`#differential-tests`, `#mutation-tests`):

- `mutation_drop_one_range_constraint` — a constraint-mutation harness removes the
  range check on one bounded column and asserts that at least one negative vector
  now (incorrectly) verifies, proving the check is load-bearing (kills the mutant
  by failing the mutation gate, per `#ci-gates`).
- `differential_multiplicity_random` — over random seeds, the Rust trace builder's
  multiplicity tally matches an independent reference tally bit-for-bit.

CI gates (`#ci-gates`): the lookup subsystem must pass all accepting tests, all
rejecting tests must produce the exact named `VerifyError`, and the
constraint-mutation gate must show no surviving mutants in the range/lookup
constraints before merge.

## Open Questions

- OPEN QUESTION (owner: area:air maintainer; resolution: RFC-0005 at v0.1): the
  exact limb width default (`w = 8` vs `w = 16`) and `num_limbs` policy for the
  linear/matmul accumulator hot path is finalized in
  `docs/rfcs/RFC-0005-linear-matmul-and-requantization-components.md` once the
  per-tensor accumulator bounds are profiled; this RFC fixes the decomposition
  mechanism, RFC-0005 fixes the width that minimizes trace area.
- OPEN QUESTION (owner: area:air maintainer; resolution: RFC-0006 at v0.2):
  whether the optional `REQUANT` table is worth its commitment cost versus the
  pure-constraint requant path in RFC-0005; the `Requant` variant ships gated and
  unused until RFC-0006/RFC-0005 benchmarks decide.

## References

- `docs/feasibility-study.md` §7.4 (interaction trace / LogUp relations), §6.1 and
  §6.3 (activation and softmax lookups), §9.3 (range-safety requirements), §5.2
  (int8 accumulator bound), §3.2 (direct AIR vs circuit lowering), §13 (risk
  register).
- `docs/rfcs/RFC-0002-fixed-point-arithmetic-over-m31.md` — bounded-integer
  semantics this RFC range-checks.
- `docs/rfcs/RFC-0004-tensor-memory-and-wiring-air.md` — consumer; tensor
  read/write consistency over the same bus.
- `docs/rfcs/RFC-0005-linear-matmul-and-requantization-components.md` — consumer;
  accumulator limb range checks and weight-table membership.
- `docs/rfcs/RFC-0006-nonlinear-primitive-components.md` — owns the
  activation/softmax/inv-sqrt table definitions that ride this infrastructure.
- `docs/rfcs/RFC-0009-fixed-candidate-planner-proof.md` — consumer; MSE cost and
  argmin comparison range checks.
- `docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md` — Fiat-Shamir
  challenge (`z`, `alpha`) derivation order.
- `docs/rfcs/RFC-0015-third-party-vendoring-and-pinning.md` — pinned vendored
  `stwo-constraint-framework` LogUp; `stwo-circuits` gate set (no range gate).
- `docs/rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md` — bit-for-bit
  trace reproducibility.
- `docs/spec/01-architecture.md#air-strategy`, `docs/spec/01-architecture.md#component-model`,
  `docs/spec/01-architecture.md#trace-model` — preprocessed/main/interaction trace
  model and direct-AIR strategy.
- `docs/spec/03-data-model.md#bounded-integers`, `docs/spec/03-data-model.md#witness`,
  `docs/spec/03-data-model.md#model-manifest`, `docs/spec/03-data-model.md#schema-versioning`.
- `docs/spec/04-error-model.md#verifier-rejections`, `docs/spec/04-error-model.md#failure-modes`.
- `docs/spec/06-security.md#soundness-requirements`, `docs/spec/06-security.md#binding-requirements`.
- `docs/spec/07-testing-strategy.md#test-pyramid`, `#golden-vectors`, `#negative-tests`,
  `#differential-tests`, `#mutation-tests`, `#ci-gates`.
- `docs/spec/09-release-and-versioning.md#relation-versioning`, `#deprecation`, `#changelog`.
- Stwo LogUp: vendored `stwo-constraint-framework` (`constraint-framework/src/logup.rs`),
  M31 `P = 2^31 - 1`, QM31 secure field (verified against upstream at 2026-06-03).
- stwo-circuits gate set `Add, Sub, Mul, PointwiseMul, Eq, TripleXor, M31ToU32,
  BlakeGGate, Permutation, Output`; `extract_bits()` helper, no range gate
  (verified against upstream at 2026-06-03).
