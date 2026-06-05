# RFC-0004: Tensor memory and wiring AIR

- Status: Accepted
- Authors: ProvableWorldModel maintainers
- Created: 2026-06-03
- Target milestone: v0.1

## Summary

This RFC fixes how tensor values flow between AIR components. It locks a single
canonical `TensorCell` relation (the typed record from
`docs/spec/03-data-model.md#tensor-memory-cells`) and a read/write-consistency
argument built on multiset equality (a LogUp permutation argument) so that every
value an operator reads is exactly a value some operator (or a committed static
constant) wrote. It then locks the arithmetization of the six structural
operations that move values without computing on them — broadcast, reshape,
transpose, concat, slice — and of static constants. Without this layer, each
component (`linear`, `matmul`, `attention`, `rollout`, `cost`, `argmin`) would
only constrain its own internal arithmetic; the prover would be free to feed a
component any values it likes at the wires between components. The wiring AIR is
the mechanism that makes the per-component proofs compose into a proof of the
whole graph.

## Motivation

The feasibility study isolates this concern in `docs/feasibility-study.md` §7.5
("Tensor memory model"): NN graphs reuse tensor values across many operators, so
"the proof needs a sound wiring model", and reads must "match a prior write or
static constant" enforced "via a permutation/multiset equality argument". Section
§9.1 (binding requirements) extends this: anything not bound is mutable by the
prover, and inter-operator wiring is exactly the surface a malicious prover would
exploit if it were unconstrained. `docs/spec/06-security.md#soundness-requirements`
restates this as a soundness obligation on component composition.

Concrete scenario the design must defeat: a prover produces a valid `linear`
trace whose output is a tensor `Y`, and a valid `attention` trace whose input is
a tensor `X`, but `X != Y` element-wise. Each component proof passes in
isolation; the composed claim "attention(linear(input)) == output" is false. The
predictor (`docs/spec/03-data-model.md#witness`,
`Witness.predictor_activations`) chains six `ConditionalBlock` layers plus
`ActionEncoder_Q` and `PredProj_Q` (RFC-0007); the rollout (RFC-0008) feeds each
step's output latent into the next step's input window. Every one of those wires
is a place the prover could lie. The `TensorCell` relation closes all of them
with one uniform argument rather than ad-hoc equality constraints per boundary.

A second scenario: structural ops. The predictor reshapes `[seq, heads*dim_head]`
to `[seq, heads, dim_head]` for attention and transposes for the score
`Q K^T` (source §7.7). A naive implementation that re-emits values into new
`TensorCell` writes under permuted indices must prove the permutation is exactly
the declared reshape/transpose, or the prover can scramble the data. This RFC
specifies those proofs so that structural ops cost lookups, not recomputation.

## Goals

- Lock the canonical `TensorCell` record (`docs/spec/03-data-model.md#tensor-memory-cells`,
  type from the contract; reproduced below) as the single inter-component wire
  format for V0.
- Lock read/write consistency as a multiset-equality (LogUp permutation)
  argument over `(tensor_id, index, value, scale_id)` keyed reads against writes,
  with explicit, named invariants for single-writer, scale agreement, and
  read-before-write ordering.
- Lock the arithmetization of broadcast, reshape, transpose, concat, and slice as
  structural ops that emit a derived tensor whose cells are proven equal (by
  index-mapped lookups) to source cells — never recomputed, never trusted.
- Lock how static constants (positional embeddings, masks, zero-points) enter the
  memory as committed writes from the preprocessed trace
  (`docs/spec/01-architecture.md#trace-model`).
- Make the wiring AIR component-agnostic: any arithmetic component reads its
  inputs and writes its outputs through the same interface, so adding a new
  component (RFC-0005 through RFC-0009) requires no change here.
- Enumerate every failure mode (mismatched read, forged write, scale confusion,
  illegal broadcast, structural-op index forgery) with the verifier's response,
  cross-referenced to `docs/spec/04-error-model.md#verifier-rejections`.

## Non-Goals

- Per-operator arithmetic constraints. `linear`/`matmul`/`requant` semantics are
  RFC-0005; nonlinear primitives are RFC-0006. This RFC only constrains the
  values crossing component boundaries, not how a component derives them.
- The range/lookup machinery itself (LogUp relation construction, multiplicity
  columns, the secure-field claimed sum). That is RFC-0003
  (`docs/rfcs/RFC-0003-range-check-and-lookup-infrastructure.md`); this RFC
  consumes RFC-0003's permutation primitive and does not redefine it.
- Fixed-point arithmetic semantics (encoding, requantization, rounding):
  RFC-0002 (`docs/rfcs/RFC-0002-fixed-point-arithmetic-over-m31.md`).
- A general dynamic-addressing RAM with prover-chosen indices. V0 tensor access
  patterns are static (known from the manifest's op graph and shapes), so the
  wiring is a compile-time-known set of reads and writes, not data-dependent
  addressing. Data-dependent gather/scatter is deferred (see Open Questions).
- Recursive aggregation of per-tensor sub-proofs: RFC-0012, milestone Future.

## Proposed Design

### Data model: the TensorCell record

The canonical wire record is fixed (contract §6.4;
`docs/spec/03-data-model.md#tensor-memory-cells`). It is reproduced here verbatim
and is normative for this RFC:

```rust
pub struct TensorCell {
    pub tensor_id: u32,
    pub index: [u32; 4],   // up to 4 dims; unused dims = 0
    pub value: M31,
    pub scale_id: u32,
    pub time: u32,         // write ordering for read/write consistency
}
```

Field semantics, locked:

| Field       | Meaning                                                                                          |
| ----------- | ------------------------------------------------------------------------------------------------ |
| `tensor_id` | Stable identifier of a logical tensor in the op graph (manifest `ops[].id` derived; see below).  |
| `index`     | Row-major coordinate, left-padded with the tensor's declared rank; trailing unused dims are `0`. |
| `value`     | The element, already a canonical `M31` (a `BoundedInt` reduced mod `p`; the integer bound lives in the range witness, not here). |
| `scale_id`  | Index into the manifest scale table (`docs/spec/03-data-model.md#model-manifest`); identifies the fixed-point scale of `value`. |
| `time`      | A monotone logical write timestamp assigned by the trace builder; used only to order writes before reads, never as data. |

Rank handling is unambiguous: a tensor of declared rank `r` (`r <= 4`) uses
`index[0..r]`; the trace builder MUST set `index[r..4] = 0` and the consistency
argument treats all four index lanes as key material, so a rank-3 tensor can
never collide with a rank-4 tensor that happens to share three coordinates,
because their `tensor_id`s differ. Tensors with rank `> 4` are rejected at trace
build (`E-TRACE-TENSOR-RANK`, `docs/spec/04-error-model.md#failure-modes`); V0
shapes (latent `[history_size=3, 192]`, attention `[seq, heads=16, dim_head=64]`
fold the head and dim_head lanes per the layout chosen in RFC-0005, MLP
`[seq, mlp_dim=2048]`) all fit within rank 4.

`tensor_id` allocation is deterministic and committed: the export pipeline
(RFC-0001) assigns `tensor_id`s in a canonical traversal of the op graph and
records the `(tensor_id -> op output / graph input / constant)` map inside the
material bound by `model_commitment`. Two runs of the same manifest therefore
produce identical `tensor_id`s, satisfying the reproducibility contract
(RFC-0016). The verifier never needs the map; it only needs that prover and
preprocessed trace agree, which the commitment enforces.

<a id="read-write-consistency"></a>

### The memory as two LogUp multisets
The wiring AIR is realized as one logical component, `tensor_memory`
(`pwm-air/components/tensor_memory.rs`, contract §4), that maintains two
multisets over the interaction trace (`docs/spec/01-architecture.md#trace-model`,
`#component-model`):

- `Writes`: every cell ever produced — outputs of arithmetic components, graph
  inputs, and static constants.
- `Reads`: every cell ever consumed — inputs to arithmetic components, plus the
  cells a structural op consumes from its source.

Each component does not own its own equality constraints to its neighbors.
Instead, when a component consumes value `v` at `(tensor_id=t, index=ix, scale=s)`
it emits a `Reads` entry, and when it produces a value it emits a `Writes` entry.
Consistency is the single relation:

```text
multiset(Writes keyed on (tensor_id, index, value, scale_id))
    ==
multiset(Reads  keyed on (tensor_id, index, value, scale_id), with multiplicity)
```

Mechanically this is RFC-0003's LogUp permutation argument over a single fused
key. The fused key is a random linear combination in the secure field `QM31`
(contract §6.1) drawn from the Fiat-Shamir channel (RFC-0014):

```text
key(cell) = tensor_id + alpha*index[0] + alpha^2*index[1] + alpha^3*index[2]
                      + alpha^4*index[3] + alpha^5*value + alpha^6*scale_id
```

where `alpha` is a `QM31` challenge bound after the main trace is committed
(channel ordering is RFC-0014's responsibility;
`docs/spec/01-architecture.md#data-flow`). Crucially, `time` is NOT in the key.
`time` participates only in the ordering invariant below; including it in the key
would make reads and writes of the same cell at different logical times fail to
cancel, which is the opposite of what is wanted.

Multiplicity: a written cell may be read more than once (the same latent feeds
several downstream ops). The argument is a multiplicity-weighted LogUp
(RFC-0003): each `Writes` entry carries a multiplicity column `m >= 1` equal to
the number of `Reads` of that exact cell, and the LogUp fractional sum balances:

```text
sum over distinct cells c of  m(c) / (X - key(c))      [from Writes]
    ==
sum over read events r of      1   / (X - key(r))      [from Reads]
```

The claimed sum equality is checked by the verifier (RFC-0003), so the prover
cannot read a cell that was never written (no `Writes` term to cancel it) nor
forge a write that is never consumed without paying a multiplicity it cannot
satisfy. This is the entire soundness core of inter-component wiring.

### Named invariants

- **INV-TM-01 (single writer).** Each `(tensor_id, index)` pair appears in
  `Writes` exactly once. Enforced by a dedicated uniqueness check: the trace
  builder writes cells in `time` order and the AIR proves `Writes` is sorted by
  `(tensor_id, index)` with strictly increasing key, rejecting any duplicate
  `(tensor_id, index)`. A tensor element has exactly one producer; reuse is
  expressed by multiple reads, never multiple writes.
- **INV-TM-02 (read matches write).** Every `Reads` entry's key equals some
  `Writes` entry's key. This is the LogUp balance above. A read whose
  `(tensor_id, index, value, scale_id)` does not correspond to a write makes the
  fractional sums unequal and the proof fails.
- **INV-TM-03 (scale agreement).** A read carries the same `scale_id` as the
  write it matches, because `scale_id` is part of the fused key. Reading a
  correct value at the wrong declared scale is therefore unsatisfiable — it is a
  distinct key with no matching write. This blocks the "right bits, wrong
  fixed-point scale" attack (source §4.2 binding list, scales).
- **INV-TM-04 (write-before-read ordering).** For every read event there exists a
  write of the same cell with `write.time < read.time`. Enforced by a separate
  LogUp range argument on `read.time - write.time - 1 >= 0` paired per matched
  cell (the comparison-as-range-check pattern of RFC-0002). This prevents a
  component from "reading" a value its producer has not yet computed within the
  same trace, which matters for the rollout recurrence (RFC-0008) where step
  `t+1` reads step `t`'s output.
- **INV-TM-05 (constants are committed writes).** Every static constant cell is a
  `Writes` entry materialized from the preprocessed trace
  (`docs/spec/01-architecture.md#trace-model`), with `time` set to the reserved
  pre-execution epoch `0`. Its `value` and `scale_id` are fixed columns the
  verifier recomputes/commits, so the prover cannot substitute a different
  constant.
- **INV-TM-06 (graph-boundary binding).** Cells whose `tensor_id` is a declared
  graph input (latent history, goal latent, candidate actions per
  `docs/spec/03-data-model.md#public-input`) are `Writes` at epoch `0` whose
  `value`s are bound to the public input (directly when `*_public`, or via the
  matching `*_commitment` opened in-circuit). The final claimed output tensor's
  cells are `Reads`/`Writes` bound to `claimed_output_commitment`. This is what
  ties the memory to `PublicInput` rather than letting it float.

### Structural operations

Structural ops move values without arithmetic. Each is arithmetized as: the op
consumes source cells as `Reads` and produces destination cells as `Writes`, and
an index-mapping constraint ties each destination index to its source index. The
*value* is copied verbatim — proven by the key including `value`, so a destination
write and its source read share the value lane and cancel under the same `alpha`
term. The op-specific work is proving the *index* relationship and the
*scale* relationship. All structural ops live as selectors/columns inside the
`tensor_memory` component; they do not get separate gates.

| Op          | Destination cell rule (locked)                                                                                                    | Scale rule                          | Index/shape constraint                                                                                                                                       |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `reshape`   | dst flat offset == src flat offset under both tensors' row-major strides.                                                          | `dst.scale_id == src.scale_id`.     | Prove `flatten(dst.index, dst.shape) == flatten(src.index, src.shape)`; declared `product(dst.shape) == product(src.shape)` checked at trace build (multiset of values preserved, source §7.5). |
| `transpose` | dst[perm(ix)] == src[ix] for a fixed permutation `perm` of axes, declared in the op.                                              | `dst.scale_id == src.scale_id`.     | Prove `dst.index == apply_perm(perm, src.index)` per matched cell; `perm` is a manifest-bound constant.                                                       |
| `concat`    | dst along `axis` is src_k offset by the sum of prior inputs' extents along `axis`.                                                | each segment keeps its `src.scale_id`; all inputs MUST share scale (checked at build, `E-TRACE-CONCAT-SCALE`). | Prove `dst.index[axis] == src_k.index[axis] + offset_k`, other lanes equal; `offset_k` from declared input shapes.                                            |
| `slice`     | dst == src restricted to `[start, start+len)` along each sliced axis (static `start`, `len`).                                     | `dst.scale_id == src.scale_id`.     | Prove `src.index[d] == dst.index[d] + start[d]` for sliced axes, equal otherwise; `start`/`len` manifest-bound.                                               |
| `broadcast` | dst[ix] == src[proj(ix)] where `proj` drops/zeros broadcast axes (extent 1 in src, >1 in dst).                                    | `dst.scale_id == src.scale_id`.     | Prove `src.index == proj(dst.index)`; the broadcast axis set is **explicitly declared** in the op and bound by `model_commitment`. See INV-TM-07.            |

- **INV-TM-07 (no implicit broadcast).** Broadcast is legal only when the op
  declares a broadcast-axis set in the manifest. A read of `src` at an index that
  `proj` could not produce, or a destination axis not in the declared set, is
  unsatisfiable. There is no NumPy/PyTorch-style automatic rank-aligning
  broadcast; the prover cannot manufacture a broadcast the manifest did not
  authorize (source §7.5: "broadcasting must be explicit"). The destination of a
  broadcast is materialized as real `Writes`, so a single source cell becomes
  multiple writes — its multiplicity is the broadcast fan-out, and INV-TM-01's
  single-writer rule applies to the *destination* cells (each distinct), while
  the *source* cell is read once per destination. To keep INV-TM-01 satisfied
  (one writer per destination index), broadcast emits its own writes rather than
  aliasing the source `tensor_id`.

Structural-op index maps are affine in the indices with manifest-constant
coefficients (`perm`, `offset_k`, `start`, `proj` axis mask). They are therefore
expressed as degree-1 AIR constraints on the index lanes plus the shared `value`
lane in the LogUp key; no division and no lookups beyond the consistency
permutation are required. This is why structural ops are cheap: a reshape of a
`[3,192]` tensor costs 576 read/write pairs and 576 affine index constraints, not
576 multiplications.

Reshape's value-multiset preservation (source §7.5 "reshapes must preserve
multiset of values") is a *consequence* of INV-TM-02 plus the bijective flat-offset
constraint, not a separate argument: a bijection on indices that copies values
preserves the value multiset by construction.

### Component interface

Every arithmetic component (RFC-0005+) takes its operands and emits its results
exclusively through this trait, so the wiring layer is the only code that touches
the `TensorCell` multisets:

```rust
/// Implemented by pwm-air's tensor_memory component; consumed by every other
/// component. `read`/`write` append to the Reads/Writes interaction multisets.
pub trait TensorMemory {
    /// Record that this op consumes `cell.value` at (tensor_id, index, scale_id).
    /// Adds a Reads entry; the caller has already constrained the value's role
    /// in its own arithmetic.
    fn read(&mut self, cell: TensorCell);

    /// Record that this op produces `cell`. Adds a Writes entry; enforces
    /// INV-TM-01 (single writer) at trace-build time and in-AIR.
    fn write(&mut self, cell: TensorCell);

    /// Emit a structural op (broadcast/reshape/transpose/concat/slice). Reads
    /// the source tensor and writes the derived tensor under the locked index
    /// map; `op` carries the manifest-bound parameters (perm/axis/start/len).
    fn structural(&mut self, op: StructuralOp, src: TensorId, dst: TensorId);
}

pub enum StructuralOp {
    Reshape { src_shape: [u32; 4], dst_shape: [u32; 4] },
    Transpose { perm: [u8; 4] },
    Concat { axis: u8, segment_offsets: Vec<u32> },
    Slice { start: [u32; 4], len: [u32; 4] },
    Broadcast { broadcast_axes: [bool; 4] },
}

pub type TensorId = u32;
```

`structural` is the only way to relate two `tensor_id`s without arithmetic; there
is no path by which a component can assert `value` equality across tensors except
through a `Writes`/`Reads` pair that the consistency argument checks. This is the
property that makes "wiring is sound" a single audit target.

### Data flow and lifecycle

```text
preprocessed trace ── static-constant writes (epoch 0) ──┐
public input ─────── graph-input writes (epoch 0) ───────┤
                                                          v
component_k.write(out_cells)  ──> Writes multiset  <── tensor_memory
component_k.read(in_cells)    ──> Reads  multiset       (LogUp permutation
structural(op,...)            ──> Reads + Writes          INV-TM-01..07)
                                                          |
                                                          v
                          claimed-output reads ── bound to claimed_output_commitment
```

In prose: before any arithmetic, the trace builder seeds the `Writes` multiset
with all static constants (epoch 0, from the preprocessed trace) and all graph
inputs (epoch 0, bound to the public input). Then components execute in graph
order; each `write` advances `time`, each `read` references an earlier-time cell,
and structural ops are interleaved as their position in the graph dictates. After
all components run, `tensor_memory` proves the single read/write balance plus the
six structural-index families and the ordering range checks. The verifier checks
the LogUp claimed sums and the public-input bindings; it never reconstructs the
multisets element-by-element.

### Determinism and concurrency

`time` assignment is a deterministic function of the canonical graph traversal
(RFC-0001 / RFC-0016), so the multisets and therefore the proof inputs are
bit-identical across runs of the same manifest and witness. Trace building may
parallelize per-component witness generation, but `Writes`/`Reads` entries carry
their own `(tensor_id, index, time)` and are merged into the multisets by a
deterministic stable sort on `(tensor_id, index)` for `Writes` and on
`(tensor_id, index, time)` for ordering, so parallelism does not affect the
committed columns.

### Failure modes and system response

| ID                       | Condition                                                                    | Detected by                | Response                                                                                                |
| ------------------------ | ---------------------------------------------------------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------- |
| `E-TRACE-TENSOR-RANK`    | Tensor declared with rank > 4.                                               | Trace builder              | Abort proving; `docs/spec/04-error-model.md#failure-modes`. No proof produced.                          |
| `E-TRACE-DUP-WRITE`      | Two writes to the same `(tensor_id, index)` (INV-TM-01).                      | Trace builder + AIR        | Abort proving; if forged into a proof, verifier rejects via the sortedness/uniqueness constraint.        |
| `E-TRACE-CONCAT-SCALE`   | `concat` inputs disagree on `scale_id`.                                      | Trace builder              | Abort proving; concat across scales is undefined.                                                       |
| `E-TRACE-BCAST-UNDECL`   | Broadcast axis not declared in the manifest op (INV-TM-07).                   | Trace builder              | Abort proving; the op graph is invalid against the committed manifest.                                  |
| `V-TM-READ-UNMATCHED`    | A read key has no matching write (INV-TM-02): forged inter-component value.   | Verifier (LogUp imbalance) | `VerifyError`; `docs/spec/04-error-model.md#verifier-rejections`.                                       |
| `V-TM-SCALE-MISMATCH`    | Read at wrong `scale_id` (INV-TM-03).                                         | Verifier (LogUp imbalance) | `VerifyError`; identical mechanism to `V-TM-READ-UNMATCHED` (it is just a different key).               |
| `V-TM-ORDER`             | Read precedes its write in `time` (INV-TM-04).                               | Verifier (range argument)  | `VerifyError`.                                                                                          |
| `V-TM-STRUCT-INDEX`      | Structural-op index map violated (wrong transpose perm, slice start, etc.).   | Verifier (affine constraint)| `VerifyError`.                                                                                          |
| `V-TM-CONST`             | Static-constant write differs from preprocessed/committed value (INV-TM-05).  | Verifier (fixed-column)     | `VerifyError`.                                                                                          |
| `V-TM-BOUNDARY`          | Graph-input/output cell not bound to public input (INV-TM-06).               | Verifier (public-input bind)| `VerifyError`.                                                                                          |

All `V-TM-*` rejections surface as the `VerifyError` variants enumerated in
`docs/spec/04-error-model.md#verifier-rejections`; this RFC does not introduce a
new error channel.

## Alternatives Considered

**A. Direct equality columns at each component boundary (no global memory).**
Wire two components by adding constraints `out_k[i] - in_{k+1}[i] = 0` for every
shared element. Considered because it is the most obvious approach and needs no
permutation argument. Rejected: it is `O(boundaries x elements)` bespoke
constraints, it forces each component to know its neighbors' layouts (breaking the
component-agnostic goal), and it does not naturally express fan-out (one tensor
read by several ops) or structural reindexing without yet more special cases.
Worse for audit: soundness of wiring would be smeared across every component
rather than concentrated in one `tensor_memory` argument. The chosen multiset
approach is `O(reads + writes)` LogUp terms with one audit target.

**B. A general read/write RAM with prover-chosen, range-checked addresses
(offline-memory-checking / sorted-address argument).** This is the standard
zkVM memory argument and would support data-dependent gather/scatter. Considered
because it is well understood and would future-proof for dynamic indexing.
Rejected for V0: V0's access pattern is entirely static — every read and write
address is known at trace-build time from the manifest op graph and declared
shapes — so paying for prover-chosen addresses and the address-sorting argument
buys generality the V0 statements (P0–P2) never use. The static `TensorCell`
multiset is strictly cheaper (no address range checks, no sorted-address
contiguity constraints) and equally sound for the static case. The general RAM is
recorded as the migration path if data-dependent indexing is ever needed (see
Open Questions); it can be added as a second relation version without disturbing
the static path.

**C. Flatten the entire graph into one monolithic component with internal
SSA-style values (no tensor identity, no `time`).** Considered because a single
component avoids cross-component wiring entirely. Rejected: it destroys the
component model of `docs/spec/01-architecture.md#component-model`, prevents reusing
`linear`/`matmul`/`attention` as independent, independently testable AIR
components, and makes the trace one giant table that cannot be batched or, later,
decomposed for recursion (RFC-0012). It also makes structural ops (reshape/
transpose) implicit in column addressing, which is exactly the kind of untyped
wiring this RFC exists to make explicit and checkable.

## Drawbacks

- The multiset argument adds one interaction-trace relation and a multiplicity
  column for every distinct written cell; for a six-block predictor the cell
  count is large (every activation element is a cell), so the wiring LogUp is a
  non-trivial fraction of trace area. The mitigation — batching cells per op into
  contiguous trace regions — is RFC-0005's layout concern, not solved here.
- Structural ops materialize destination tensors as fresh writes (notably
  broadcast fan-out), so a broadcast does cost `Writes` proportional to the
  destination size rather than being free. This is the price of the explicit,
  single-writer model; it is accepted because implicit aliasing was the unsound
  alternative.
- `time` is a global counter; trace building must assign it deterministically,
  which serializes the *ordering* decision even though witness computation can be
  parallel. The stable-sort merge keeps this from being a real bottleneck but it
  is a coordination point.
- Restricting to rank `<= 4` (`index: [u32; 4]`) is a hard limit baked into the
  type. V0 architectures fit, but a future op needing rank 5 would require a new
  `TensorCell` schema version and a relation-id bump.

## Migration / Rollout

- This relation ships in the `v0.1 — Foundations` milestone as part of
  `pwm-air` (`tensor_memory` component) and is a hard dependency of every
  arithmetic component, so it lands before RFC-0005's `linear`/`matmul`.
- Versioning: the `TensorCell` layout and the consistency-argument shape are part
  of the AIR bound by `relation_id` (`pwm.lewm.<statement>.v<N>`, contract §4).
  Any change to the key composition, the structural-op index maps, the invariant
  set, or the `TensorCell` fields is a semantic change that mints a new
  `relation_id` per `docs/spec/09-release-and-versioning.md#relation-versioning`;
  old proofs remain valid only against their declared id.
- Schema evolution for `TensorCell` (e.g., adding rank, or a `region_id` lane for
  batching) follows `docs/spec/03-data-model.md#schema-versioning`: additive lanes
  default to zero and are introduced behind a new manifest/relation version, never
  silently.
- Feature gating: the general-RAM path (Alternative B) and data-dependent indexing
  are not wired into V0. When introduced they ship as a distinct `relation_id`
  (e.g. `...rollout.v2`) and a Cargo feature in `pwm-air`, so the static path is
  never destabilized; the static and dynamic memory arguments coexist behind the
  relation id a proof declares.

## Testing Strategy

Cross-references `docs/spec/07-testing-strategy.md` and the rejection taxonomy in
`docs/spec/04-error-model.md#verifier-rejections`.

Accepting tests (`#golden-vectors`, `#test-pyramid`):

- `tm_accept_single_write_single_read`: one tensor written, read once by a
  downstream op; LogUp balances; proof verifies.
- `tm_accept_fanout`: one tensor read by three ops; multiplicity column = 3;
  balance holds; proof verifies.
- `tm_accept_reshape_3x192_to_576`: reshape preserves the value multiset via the
  flat-offset bijection; proof verifies and the destination read-back equals the
  source element-by-element (golden vector).
- `tm_accept_transpose_qk`: transpose for the attention `Q K^T` index map (the
  predictor head layout, source §7.7); `dst.index == apply_perm(perm, src.index)`;
  proof verifies.
- `tm_accept_concat_rollout_window`: concat of predicted latent with prior window
  along the sequence axis (the rollout append, RFC-0008); segment offsets correct;
  proof verifies.
- `tm_accept_slice_history_window`: slice the last `history_size=3` latents from a
  longer trajectory; `start`/`len` correct; proof verifies.
- `tm_accept_broadcast_bias`: declared broadcast of a per-channel bias across the
  sequence axis; fan-out writes correct; proof verifies.
- `tm_accept_constant_posembed`: positional-embedding constants entered as epoch-0
  committed writes (INV-TM-05) and read by the predictor; proof verifies.
- `tm_diff_against_reference` (`#differential-tests`): the Rust trace builder's
  `Writes`/`Reads` multisets equal those derived from the Python fixed-point
  reference (RFC-0001) for a full P0 graph, bit-for-bit on `(tensor_id, index,
  value, scale_id)`.

Rejecting / negative tests (`#negative-tests`, mapped to the failure table):

- `tm_reject_forged_read` -> `V-TM-READ-UNMATCHED`: mutate one read `value` so it
  matches no write; LogUp imbalance; verifier rejects. (source §7.5 acceptance:
  "Mutating one read value rejects".)
- `tm_reject_wrong_scale` -> `V-TM-SCALE-MISMATCH`: read the correct value under a
  different `scale_id`; verifier rejects (INV-TM-03; source §7.5: "Using a value
  from the wrong scale_id rejects").
- `tm_reject_swapped_indices_no_transpose` -> `V-TM-STRUCT-INDEX`: swap two index
  lanes without a declared transpose op; verifier rejects (source §7.5: "Swapping
  two tensor indices rejects unless operation is transpose").
- `tm_reject_undeclared_broadcast` -> `E-TRACE-BCAST-UNDECL` / INV-TM-07:
  broadcast along an axis the manifest did not authorize; trace build aborts, and
  a hand-forged trace fails the broadcast index constraint at verify.
- `tm_reject_double_write` -> `E-TRACE-DUP-WRITE` / INV-TM-01: two writes to the
  same `(tensor_id, index)`; sortedness/uniqueness constraint fails.
- `tm_reject_read_before_write` -> `V-TM-ORDER`: order a read at `time` earlier
  than its producing write; range argument `read.time - write.time - 1 >= 0`
  fails; verifier rejects (INV-TM-04).
- `tm_reject_mutated_constant` -> `V-TM-CONST`: alter a positional-embedding
  constant cell away from the committed preprocessed value; fixed-column check
  fails (INV-TM-05).
- `tm_reject_unbound_output` -> `V-TM-BOUNDARY`: emit a claimed-output cell not
  bound to `claimed_output_commitment`; public-input binding fails (INV-TM-06).
- `tm_mutation_drop_ordering_constraint` (`#mutation-tests`): remove the INV-TM-04
  range constraint from the AIR; the `tm_reject_read_before_write` vector must now
  pass, proving the constraint is load-bearing (constraint-mutation gate,
  `#ci-gates`).
- `tm_mutation_drop_value_from_key` (`#mutation-tests`): drop the `value` lane from
  the LogUp key; `tm_reject_forged_read` must now pass, proving the key composition
  is load-bearing.

CI gate (`#ci-gates`): no component that reads or writes tensors merges without
both an accepting and a rejecting test exercising its use of `TensorMemory`
(RFC-0013's accept-and-reject rule applied to wiring).

## Open Questions

- OPEN QUESTION (owner: area:air maintainer; resolution: RFC-0008 at v0.2):
  whether the rollout's autoregressive append is best expressed as repeated
  `concat`/`slice` structural ops over a growing trajectory tensor, or as a
  dedicated windowing relation in the `rollout` component. This RFC provides both
  primitives; RFC-0008 chooses and locks the rollout-side usage.
- OPEN QUESTION (owner: area:air maintainer; resolution: Future, gated on P3/CEM
  in RFC-0010): data-dependent gather/scatter (e.g., top-k elite selection
  indexing) needs prover-chosen addresses, which the static `TensorCell` multiset
  does not provide. The migration path is the general-RAM relation (Alternatives
  Considered, item B) under a new `relation_id`; no V0 statement requires it.
- OPEN QUESTION (owner: area:air + area:performance maintainers; resolution:
  RFC-0005 at v0.1): the per-op trace batching layout that keeps each tensor's
  cells contiguous to bound the wiring LogUp's trace area. This RFC fixes the
  relation; RFC-0005 fixes the column layout that makes it cheap.

## References

- `docs/feasibility-study.md` §7.5 (tensor memory model), §7.4 (interaction
  trace / read-write consistency), §9.1 (binding requirements), §7.7 (attention
  transpose/reshape usage) — the founding analysis this RFC expands.
- `docs/spec/03-data-model.md#tensor-memory-cells` — canonical `TensorCell`
  schema (contract §6.4).
- `docs/spec/03-data-model.md#public-input`, `#witness`, `#model-manifest`,
  `#schema-versioning` — graph-boundary and constant bindings, scale table.
- `docs/spec/01-architecture.md#component-model`, `#trace-model`, `#data-flow` —
  component model and preprocessed/main/interaction traces.
- `docs/spec/04-error-model.md#failure-modes`, `#verifier-rejections` — the
  `E-TRACE-*` and `V-TM-*` responses.
- `docs/spec/06-security.md#soundness-requirements`, `#binding-requirements` —
  composition soundness obligations.
- `docs/spec/07-testing-strategy.md#test-pyramid`, `#golden-vectors`,
  `#negative-tests`, `#differential-tests`, `#mutation-tests`, `#ci-gates`.
- `docs/spec/09-release-and-versioning.md#relation-versioning` — relation_id
  immutability for AIR-shape changes.
- `docs/rfcs/RFC-0002-fixed-point-arithmetic-over-m31.md` — `BoundedInt`, M31
  encoding, comparison-as-range-check used by INV-TM-04.
- `docs/rfcs/RFC-0003-range-check-and-lookup-infrastructure.md` — the LogUp
  permutation/multiplicity primitive this RFC consumes.
- `docs/rfcs/RFC-0001-model-manifest-and-export-pipeline.md` — deterministic
  `tensor_id` allocation and constant commitment.
- `docs/rfcs/RFC-0005-linear-matmul-and-requantization-components.md` — first
  consumer of `TensorMemory`; owns batching layout.
- `docs/rfcs/RFC-0007-leworldmodel-predictor-air.md`,
  `docs/rfcs/RFC-0008-rollout-air.md` — downstream consumers (reshape/transpose/
  concat/slice over latents).
- `docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md` — Fiat-Shamir
  channel that provides the `QM31` key challenge `alpha`.
- Stwo `stwo-constraint-framework` LogUp (`constraint-framework/src/logup.rs`),
  M31/QM31 field module (`crates/stwo/src/core/fields/`), verified against
  upstream at 2026-06-03.
