# RFC-0008: Rollout AIR

- Status: Accepted
- Authors: ProvableWorldModel maintainers
- Created: 2026-06-03
- Target milestone: v0.2

## Summary

This RFC locks the autoregressive rollout relation `pwm.lewm.rollout.v1` (statement
tier P1) and its AIR component, `RolloutComponent`. The rollout takes an initial
latent history of `history_size = 3` latents plus a fixed action sequence and
produces a latent trajectory by repeatedly invoking the predictor step proven by
[RFC-0007](RFC-0007-leworldmodel-predictor-air.md) (relation
`pwm.lewm.predictor_step.v1`). The decision this RFC makes permanent: the rollout
recurrence is a *sliding window* over the latent/action streams (the most recent
`history_size` latents and their aligned actions), each predicted latent is
*appended in place* into the same canonical tensor stream so that it becomes an
input to subsequent windows, the produced trajectory is committed via the tensor
memory relation of [RFC-0004](RFC-0004-tensor-memory-and-wiring-air.md),
and the AIR carries explicit *window-selection-correctness* constraints proving
that every per-step predictor invocation reads exactly the right slice of the
stream with no off-circuit substitution. This is the recurrence that
[RFC-0009](RFC-0009-fixed-candidate-planner-proof.md) batches over
candidates and scores against the goal latent for the headline V0 (P2) statement.

## Motivation

Planning requires predicting multiple steps ahead, not a single transition. The
predictor step proven in [RFC-0007](RFC-0007-leworldmodel-predictor-air.md)
maps one window of latents and actions to one next latent; rollout chains those
steps so that the model's own predictions feed back as inputs. The founding
analysis defines this as statement tier P1 (docs/feasibility-study.md §1.2,
Statement P1) and the rollout component design in
docs/feasibility-study.md §7.8, and verifies against upstream that LeWorldModel's
`jepa.py rollout()` "concatenates predicted embeddings and next actions
autoregressively" and computes cost as MSE on the final predicted latent versus
the goal latent (verified against upstream at 2026-06-03).

The soundness hazard rollout introduces beyond a single step is *wiring*: a step
`t+1` must consume the latent that step `t` actually produced, sliced over the
correct `history_size` window, with actions aligned to the same window. If the AIR
left window selection unconstrained, a prover could feed each predictor invocation
a favorable, fabricated history and still satisfy every per-step predictor
constraint in isolation, breaking the rollout relation while every component
verifies locally. The founding analysis names exactly these constraints
(docs/feasibility-study.md §7.8): "window selection is correct", "predictor input
equals selected latent/action window", "predictor output equals next latent",
"next latent is appended to future windows". This RFC turns that list into a typed
component contract.

Concrete system scenario: the V0 planner
([RFC-0009](RFC-0009-fixed-candidate-planner-proof.md)) rolls out `S`
candidate action sequences over `horizon = 5` steps with `action_block = 5`
(verified against upstream at 2026-06-03: `config/eval/pusht.yaml plan_config`
`horizon=5`, `receding_horizon=5`, `action_block=5`). Each candidate is one
invocation of this rollout relation. The trajectory commitment defined here is the
object the cost component reads its final latent from, and the per-candidate
rollout instances are what
[RFC-0012](RFC-0012-recursive-aggregated-verification.md) would later
aggregate.

## Goals

- Lock the rollout recurrence: a sliding window of `history_size = 3` latents and
  their aligned actions, advanced one step at a time over a fixed `horizon`, with
  each predicted latent appended into the canonical latent stream and reused by
  later windows.
- Define `RolloutComponent` and its column layout, including the per-step indexing
  (`step`, `window_base`) and the boundary selectors that distinguish prefilled
  history rows from predicted rows.
- Define the trajectory commitment: the rollout output is a single canonical
  `Tensor` written through the tensor memory relation, and its commitment is bound
  into `PublicInput.claimed_output_commitment`.
- State and name every window-selection-correctness invariant so a malformed
  window read is rejected.
- Specify exactly how `RolloutComponent` composes with `PredictorComponent`
  (RFC-0007) via the tensor memory relation (RFC-0004), with no re-statement of
  predictor internals here.
- Enumerate every rollout-specific failure mode with the verifier's response,
  cross-referencing
  [docs/spec/04-error-model.md#verifier-rejections](../spec/04-error-model.md#verifier-rejections).
- Specify accepting and rejecting tests, cross-referencing
  [docs/spec/07-testing-strategy.md](../spec/07-testing-strategy.md).

## Non-Goals

- Predictor internals (ActionEncoder_Q, ARPredictor_Q, PredProj_Q, Predict_Q).
  Owned by [RFC-0007](RFC-0007-leworldmodel-predictor-air.md). Rollout
  treats one predictor step as an opaque relation invocation linked through tensor
  memory.
- Cost (MSE-to-goal) and argmin/selection over candidates. Owned by
  [RFC-0009](RFC-0009-fixed-candidate-planner-proof.md). This RFC ends at
  "a committed trajectory exists and is consistent with the recurrence".
- Candidate batching strategy and trace tiling across `S` candidates. The
  recurrence here is per-candidate; the batching axis is owned by RFC-0009 and the
  linear/matmul batching of
  [RFC-0005](RFC-0005-linear-matmul-and-requantization-components.md).
- Fixed-point semantics (encoding, requantize, overflow=reject). Owned by
  [RFC-0002](RFC-0002-fixed-point-arithmetic-over-m31.md).
- Tensor memory relation construction (TensorCell, read/write multiset argument).
  Owned by [RFC-0004](RFC-0004-tensor-memory-and-wiring-air.md); this RFC
  consumes it.
- CEM sampling/iteration recurrence (statement P3); explicitly deferred to
  [RFC-0010](RFC-0010-cem-planner-proof.md) and out of V0 scope.

## Proposed Design

### Statement and relation

Rollout is statement tier P1 with immutable `relation_id`
`pwm.lewm.rollout.v1` (encoded as the 32-byte digest carried in
`PublicInput.relation_id`; see
[docs/spec/03-data-model.md#relation-id](../spec/03-data-model.md#relation-id)).
A proof is valid only for this exact id; any semantic change to the recurrence
mints a new id (`v2`) per the corpus relation-versioning rule
([docs/spec/09-release-and-versioning.md#relation-versioning](../spec/09-release-and-versioning.md#relation-versioning)).

```text
Public (PublicInput, see docs/spec/03-data-model.md#public-input):
  relation_id                  = pwm.lewm.rollout.v1
  model_commitment             (binds architecture, weights, scales, rounding,
                                lookup tables, horizon — see RFC-0001)
  quantization_commitment
  planner_config_commitment    (binds history_size, horizon, action_block)
  statement_type               = StatementType::P1Rollout
  latent_history_{commitment|public}     z[0 .. H-1]
  candidate_actions_{commitment|public}  a[0 .. T-1]
  claimed_output_commitment              commitment to the rollout trajectory
  goal_latent_*                = None   (cost is RFC-0009, not P1)
  selected_index               = None
  selected_cost                = None

Witness (Witness, see docs/spec/03-data-model.md#witness):
  latent_history               z[0 .. H-1]
  candidate_actions            a[0 .. T-1]
  action_embeddings            ActionEncoder_Q(a)            (from RFC-0007)
  predictor_activations        per-step predictor intermediates (RFC-0007)
  rollout_trajectory           z[0 .. H-1+horizon]           (the appended stream)
  range_witnesses, lookup_witnesses
```

Let `H = history_size = 3` (the V0 reference configuration targets
`history_size = 3`; verified against upstream at 2026-06-03) and `horizon` be the
rollout length declared in the planner config and bound by
`planner_config_commitment`. The V0 reference configuration targets `horizon = 5`.
The latent stream has `H + horizon` entries indexed `0 .. H+horizon-1`; indices
`0 .. H-1` are the prefilled history, indices `H .. H+horizon-1` are predicted.

### The recurrence (locked)

Each step advances a sliding window by one position. Step `s` for
`s = 0 .. horizon-1` produces stream entry `H + s` from the window ending at the
previous entry:

```text
window_base(s) = s                         # first stream index in the window
z_window(s)    = z[ s .. s + H - 1 ]        # H latents, the most recent H
a_window(s)    = a[ s .. s + H - 1 ]        # H actions aligned to those latents
z[H + s]       = Predict_Q( z_window(s), ActionEncoder_Q(a_window(s)) )
```

`Predict_Q` is the relation
`pwm.lewm.predictor_step.v1` from
[RFC-0007](RFC-0007-leworldmodel-predictor-air.md). After step `s`
writes `z[H+s]`, that entry is in scope for `z_window(s+1)` because the window base
advances by one. This is the "append the predicted latent and reuse it in later
windows" decision, made permanent. The action stream `a[0 .. T-1]` must have
`T >= H + horizon - 1` so that every window has its actions; the binding between
`T`, `H`, and `horizon` is checked at trace build and is an enumerated failure
mode (`E-ROLL-ACTLEN`, below).

This windowing matches LeWorldModel's `jepa.py rollout()`, which concatenates
predicted embeddings and next actions autoregressively (verified against upstream
at 2026-06-03). The AIR proves the *exported quantized* recurrence, not live
PyTorch; semantic alignment of the exported graph to the checkpoint is the
responsibility of the export pipeline
([RFC-0001](RFC-0001-model-manifest-and-export-pipeline.md)) and the
differential parity gate, not of this AIR.

### `RolloutComponent` column layout

`RolloutComponent` is a thin orchestration component: it does not re-derive
predictor arithmetic. It owns the per-step bookkeeping columns and the lookups
that pin each step's window into the canonical latent and action streams. One trace
row corresponds to one rollout step `s` (the per-step predictor sub-trace lives in
`PredictorComponent`'s own rows, linked by the `step` selector through the
interaction trace).

Main trace columns (one row per step `s = 0 .. horizon-1`):

```text
step           u32    rollout step index, 0..horizon-1
window_base    u32    = step (first stream index of this step's window)
out_index      u32    = H + step (stream index this step writes)
is_first       bit    1 iff step == 0
is_last        bit    1 iff step == horizon-1
z_in[0..H-1]   M31     the H latent values read for this window
a_in[0..H-1]   M31     the H action values read for this window
z_out          M31     the predicted latent written at out_index (scalar-per-row
                       in the layout: latent_dim is carried as a sub-axis; see note)
```

Note on `latent_dim`: latents are `latent_dim = 192`-vectors (the V0 reference
configuration targets `latent_dim = 192`). The layout above is per-element; the
physical trace tiles the `latent_dim` axis exactly as the predictor I/O is tiled in
[RFC-0007](RFC-0007-leworldmodel-predictor-air.md), so `z_in[h]`, `z_out`
denote one element of the latent vector at a fixed `(step, dim)` and the row count
is `horizon * latent_dim` (window-selection constraints are independent of `dim`
and replicated across it). The action axis is tiled over `action_block = 5`
identically. This RFC fixes the *indexing* relation; the physical tiling is shared
machinery with RFC-0005/RFC-0007 and not re-specified.

Preprocessed (fixed) columns, committed before proving and recomputable by the
verifier from `planner_config_commitment`
([docs/spec/01-architecture.md#trace-model](../spec/01-architecture.md#trace-model)):

```text
sel_step       per-row constant 0..horizon-1   (selector for step identity)
sel_window     per-(step,h) constant = step+h  (the expected stream index of z_in[h])
sel_action     per-(step,h) constant = step+h  (the expected stream index of a_in[h])
sel_out        per-step constant = H+step      (the expected write index)
```

### Window-selection-correctness constraints (the heart of the lock)

Every value the predictor consumes at step `s` must be the exact stream element the
recurrence prescribes, and every value it produces must be appended at the exact
index. These are enforced as TensorCell reads/writes against the canonical
`rollout_trajectory` and `candidate_actions` tensors via the multiset/permutation
argument of
[RFC-0004](RFC-0004-tensor-memory-and-wiring-air.md#read-write-consistency).
The latent stream is `tensor_id = ROLLOUT_LATENT_STREAM`; actions are
`tensor_id = CANDIDATE_ACTIONS`.

```text
# (1) window read pins to the prescribed stream index, for h = 0..H-1:
read TensorCell{ tensor_id = ROLLOUT_LATENT_STREAM,
                 index = [ sel_window[s,h], dim, 0, 0 ],
                 value = z_in[h], scale_id = LATENT_SCALE, time = t_read }

read TensorCell{ tensor_id = CANDIDATE_ACTIONS,
                 index = [ sel_action[s,h], act_dim, 0, 0 ],
                 value = a_in[h], scale_id = ACTION_SCALE, time = t_read }

# (2) predictor linkage: the predictor step keyed by sel_step[s] consumes
#     exactly z_in[*], a_in[*] (after ActionEncoder_Q) and produces z_out.
#     This is a lookup into RFC-0007's predictor_step relation, keyed by step.

# (3) append: the produced latent is written at the prescribed out index:
write TensorCell{ tensor_id = ROLLOUT_LATENT_STREAM,
                  index = [ sel_out[s], dim, 0, 0 ],
                  value = z_out, scale_id = LATENT_SCALE, time = t_write }
with t_write > t_read   (write ordering, RFC-0004 #read-write-consistency)

# (4) prefill: the history rows 0..H-1 of the latent stream are the witness
#     latent_history, bound to PublicInput.latent_history_{public|commitment}:
is_first-gated seeding: for j = 0..H-1
  write TensorCell{ ROLLOUT_LATENT_STREAM, [j,dim,0,0], latent_history[j], ... }
```

Constraints over the bookkeeping columns (low-degree, per row):

```text
INV-ROLL-01 window_base = step                  (window_base - step = 0)
INV-ROLL-02 out_index   = H + step              (out_index - step - H = 0)
INV-ROLL-03 boundary: is_first*(step) = 0;
            is_last*(step - (horizon-1)) = 0;
            is_first, is_last are bits
INV-ROLL-04 step advances by 1: on adjacent rows step_{r+1} - step_r = 1
            (gated off across step boundaries by sel_step)
```

The window read indices `sel_window[s,h] = s + h` and `sel_action[s,h] = s + h`
are *preprocessed constants*, not witness values. The prover cannot choose them;
they are fixed by the planner config and recomputed by the verifier. The only
freedom the prover has is the *values* at those cells, and those are pinned by the
TensorCell multiset argument to be exactly what was written (history seed for
indices `< H`, or a prior step's `z_out` for indices `>= H`). This is what makes a
fabricated favorable window impossible: there is no unwritten cell to read, and a
read of a wrong index fails the preprocessed-constant equality, while a read of a
wrong value fails the multiset/permutation balance.

### Named invariants

| Invariant | Statement | Enforcement |
|-----------|-----------|-------------|
| INV-ROLL-01 | `window_base(s) = s` | per-row constraint vs `sel_step` |
| INV-ROLL-02 | `out_index(s) = H + s`; the write target is the prescribed append slot | per-row constraint; TensorCell write index = `sel_out[s]` |
| INV-ROLL-03 | step selectors are bits; `is_first` only at `s=0`, `is_last` only at `s=horizon-1` | boolean + boundary constraints |
| INV-ROLL-04 | step index increments by exactly 1 across the rollout | adjacent-row constraint gated by `sel_step` |
| INV-ROLL-05 | every window latent read equals the stream cell at index `s+h` (window-selection correctness, latents) | TensorCell read vs preprocessed `sel_window`, multiset-balanced (RFC-0004) |
| INV-ROLL-06 | every window action read equals the action cell at index `s+h` (window-selection correctness, actions) | TensorCell read vs preprocessed `sel_action`, multiset-balanced (RFC-0004) |
| INV-ROLL-07 | each step's predictor invocation consumes exactly `(z_in, a_in)` and produces `z_out` (no off-circuit predictor) | interaction-trace lookup into `pwm.lewm.predictor_step.v1` keyed by `sel_step` |
| INV-ROLL-08 | predicted latent appended at index `H+s` is the same value later windows read (recurrence closure) | single-producer write + later read both hit cell `[H+s, dim]` under one TensorCell relation, `t_write < t_read` |
| INV-ROLL-09 | history seed cells `0..H-1` equal `latent_history` bound in `PublicInput` | `is_first`-gated seed writes; public-input binding (RFC-0014) |
| INV-ROLL-10 | trajectory commitment = `claimed_output_commitment`; trajectory shape is `[H+horizon, latent_dim]` | canonical serialization digest (RFC-0014) checked against `PublicInput.claimed_output_commitment` |
| INV-ROLL-11 | `horizon` and `H` used by the trace equal the values bound by `planner_config_commitment` | preprocessed columns recomputed by verifier from committed config |

INV-ROLL-08 is the recurrence-closure invariant and is the precise property that
distinguishes a real rollout from `horizon` independent predictor steps with
attacker-chosen inputs: because the write at `[H+s]` and the read at `[s'+h]=[H+s]`
for the later step `s'` resolve to one cell in one TensorCell relation, the prover
cannot substitute a different latent between producing and consuming it.

### Trajectory commitment (locked)

The rollout output is the single canonical tensor `rollout_trajectory` of shape
`[H + horizon, latent_dim]`, row-major, `scale_id = LATENT_SCALE`, serialized by the
canonical scheme of
[RFC-0014](RFC-0014-canonical-serialization-and-transcript.md). Its
32-byte digest is `PublicInput.claimed_output_commitment` (INV-ROLL-10). For P1, the
full trajectory is committed (not only the final latent) so that P1 proofs are
independently meaningful and so that
[RFC-0009](RFC-0009-fixed-candidate-planner-proof.md) and
[RFC-0012](RFC-0012-recursive-aggregated-verification.md) can read the
final latent (`index = H + horizon - 1`) from a committed object without re-running
the rollout. The commitment binds shape, scale, and values; a prover changing any
trajectory element changes the digest and fails INV-ROLL-10.

### Data flow and lifecycle

```text
manifest + planner_config  ->  RolloutComponent preprocessed columns
                               (sel_step, sel_window, sel_action, sel_out)
latent_history (public/committed) ─┐
candidate_actions (public/committed)─┤
                                     v
  trace_builder seeds stream[0..H-1] = latent_history (INV-ROLL-09)
  for s = 0..horizon-1:
     read window  z[s..s+H-1], a[s..s+H-1]            (INV-ROLL-05/06)
     invoke PredictorComponent (RFC-0007) keyed by s  (INV-ROLL-07)
     write z[H+s]                                      (INV-ROLL-02/08)
  commit rollout_trajectory -> claimed_output_commitment (INV-ROLL-10)
```

Determinism: the rollout trace is a pure function of `(manifest, planner_config,
latent_history, candidate_actions)`. There is no randomness in P1 (randomness is a
P3/CEM concern, RFC-0010). Same inputs yield byte-identical trace columns and
identical `claimed_output_commitment`, satisfying the reproducibility contract of
[RFC-0016](RFC-0016-cli-artifact-bundle-and-reproducibility.md). The
canonical rounding mode and overflow=reject policy are inherited unchanged from
[RFC-0002](RFC-0002-fixed-point-arithmetic-over-m31.md); rollout adds no
new arithmetic, only wiring.

### Failure modes and the system's response

All rejections map to the verifier taxonomy at
[docs/spec/04-error-model.md#verifier-rejections](../spec/04-error-model.md#verifier-rejections);
build-time errors map to
[docs/spec/04-error-model.md#failure-modes](../spec/04-error-model.md#failure-modes).

| Code | Condition | Phase | System response |
|------|-----------|-------|-----------------|
| `E-ROLL-ACTLEN` | `len(candidate_actions) < H + horizon - 1` | trace build | refuse to build trace; structured error (TraceError); no proof emitted |
| `E-ROLL-CONFIG` | trace `horizon`/`H` differ from `planner_config_commitment` | verify | reject (INV-ROLL-11): preprocessed column mismatch |
| `E-ROLL-WINDOW-Z` | a window latent read does not match stream cell at `s+h` | verify | reject (INV-ROLL-05): TensorCell multiset imbalance / index mismatch |
| `E-ROLL-WINDOW-A` | a window action read does not match action cell at `s+h` | verify | reject (INV-ROLL-06): same mechanism, action stream |
| `E-ROLL-LINK` | predictor invocation at step `s` consumes inputs other than `(z_in,a_in)` or yields a value other than `z_out` | verify | reject (INV-ROLL-07): interaction-trace lookup into `predictor_step.v1` fails |
| `E-ROLL-APPEND` | predicted latent written at index `!= H+s`, or `t_write <= t_read` | verify | reject (INV-ROLL-02/INV-ROLL-08): write index / time-ordering constraint |
| `E-ROLL-CLOSURE` | a later window reads a different value than the one produced for that index | verify | reject (INV-ROLL-08): single-cell multiset balance fails |
| `E-ROLL-SEED` | stream history cells `0..H-1` differ from `latent_history` | verify | reject (INV-ROLL-09): public-input binding mismatch |
| `E-ROLL-TRAJ` | `rollout_trajectory` digest != `claimed_output_commitment` | verify | reject (INV-ROLL-10): commitment mismatch |
| `E-ROLL-STEPIDX` | `step` non-monotone or `window_base != step` or boundary bits malformed | verify | reject (INV-ROLL-01/03/04) |
| `E-ROLL-RELID` | proof submitted under a `relation_id` other than `pwm.lewm.rollout.v1` (e.g. P0 submitted as P1) | verify | reject: relation_id / statement_type mismatch (see RFC-0000) |

Range-safety for the latent and action values themselves is delegated to
[RFC-0002](RFC-0002-fixed-point-arithmetic-over-m31.md) and
[RFC-0003](RFC-0003-range-check-and-lookup-infrastructure.md); rollout
adds no new bounded quantities beyond the bookkeeping indices, which are
preprocessed constants and need no range witness.

## Alternatives Considered

### Alternative A: monolithic predictor-times-horizon AIR (no separate rollout component)

Inline the predictor `horizon` times into one giant component, with hard-wired
columns for each step's window instead of a TensorCell-mediated stream.

Rejected. It couples the rollout length into the predictor component, forcing a
new component per `horizon` and a new audit each time. It also discards the
TensorCell multiset argument that makes INV-ROLL-08 (recurrence closure) provable
by construction; with hard-wired columns the equality "step `s` output equals step
`s+1` input" becomes a bespoke per-step constraint that is easy to get subtly wrong
and hard to reuse across candidates. The separate-component design lets RFC-0009
batch candidates over one rollout relation and lets RFC-0012 aggregate per-step
proofs later. The cost is one extra interaction-trace lookup per step (predictor
linkage, INV-ROLL-07), which is negligible against predictor MAC counts.

### Alternative B: commit only the final latent, not the full trajectory

Bind `claimed_output_commitment` to `z[H + horizon - 1]` alone and drop the
intermediate latents from the committed object.

Rejected. A final-latent-only commitment makes the P1 statement non-self-checking:
a verifier could not confirm the recurrence produced a coherent trajectory, only
that *some* final latent was claimed, and INV-ROLL-08's closure would have nothing
external to anchor to for the intermediate cells. Worse, RFC-0012's per-step
aggregation needs intermediate commitments as the seams between sub-proofs.
Committing the full trajectory costs `(H+horizon)*latent_dim` field elements in the
canonical digest (a few thousand elements at the V0 reference dimensions), a fixed,
small serialization cost paid once. We accept that cost for composability and
auditability. The cost component (RFC-0009) still reads only the final latent, so
runtime cost is unaffected.

### Alternative C: prover-supplied window indices as witness, range-checked

Let the prover write `window_base` and the per-`h` read indices as witness columns
and range-check them against `[0, H+horizon)`, rather than fixing them as
preprocessed constants.

Rejected as unsound-by-omission. Range-checking an index only proves it is *in
bounds*, not that it is *the prescribed window*. A prover could read an in-range but
wrong cell (e.g. always the history, never the freshly predicted latent) and pass
every range check, defeating the recurrence. Making the window indices preprocessed
constants recomputed by the verifier from `planner_config_commitment` removes that
freedom entirely (INV-ROLL-05/06/11). The only legitimate prover freedom is the
cell *values*, which the TensorCell multiset argument already pins.

## Drawbacks

- The full-trajectory commitment (INV-ROLL-10) adds `(H+horizon)*latent_dim`
  elements to the public output digest, larger than a final-latent-only commitment.
  Accepted for composability (Alternative B rationale).
- Per-step predictor linkage via interaction-trace lookup (INV-ROLL-07) adds one
  LogUp term per step beyond a fully inlined design, and binds rollout correctness
  to the correctness of RFC-0004's read/write argument; a bug there is a bug here.
  Mitigated by the shared, separately audited tensor memory component and its own
  negative tests.
- Fixing window indices as preprocessed constants means `horizon` (and the implied
  trace shape) is part of the committed planner config, so changing `horizon`
  requires a new `planner_config_commitment` and a re-export. This is intended
  (Alternative C rationale) but reduces flexibility for ad hoc horizons.
- The per-element latent tiling (`horizon * latent_dim` rows for bookkeeping)
  replicates window-selection constraints across `dim`. This is simple and correct
  but not the densest possible layout; a vectorized layout is a future optimization
  tracked under [RFC-0012](RFC-0012-recursive-aggregated-verification.md)
  and the performance budget, not a soundness concern.

## Migration / Rollout

- New component, no prior version to migrate. Ships in milestone `v0.2 — Predictor
  & Rollout` after P0 (RFC-0007).
- Relation versioning: the recurrence is locked as `pwm.lewm.rollout.v1`. Any
  change to the windowing (different `H` semantics, non-unit window advance,
  different append rule) mints `pwm.lewm.rollout.v2` and does not reuse the v1 id,
  per
  [docs/spec/09-release-and-versioning.md#relation-versioning](../spec/09-release-and-versioning.md#relation-versioning).
- Feature gating: `RolloutComponent` is built behind the `pwm-air` feature
  `rollout`, which depends on the `predictor` feature (RFC-0007). The prover CLI
  exposes `pwm-prover prove-rollout`
  ([docs/spec/02-public-api.md#cli](../spec/02-public-api.md#cli),
  [RFC-0016](RFC-0016-cli-artifact-bundle-and-reproducibility.md)); the
  verifier dispatches on `StatementType::P1Rollout`.
- Config binding: `history_size` and `horizon` enter `planner_config_commitment`.
  Manifests that omit `horizon` are rejected at export (RFC-0001) rather than
  defaulted, so the bound config is always explicit.
- Schema: rollout reuses `PublicInput`/`Witness` from
  [docs/spec/03-data-model.md#public-input](../spec/03-data-model.md#public-input)
  and
  [docs/spec/03-data-model.md#witness](../spec/03-data-model.md#witness)
  unchanged; `rollout_trajectory` is the already-present `Witness.rollout_trajectory`
  field. No schema bump is required for V0.

## Testing Strategy

Cross-references the test stack in
[docs/spec/07-testing-strategy.md](../spec/07-testing-strategy.md) and the
rejection taxonomy in
[docs/spec/04-error-model.md#verifier-rejections](../spec/04-error-model.md#verifier-rejections).
No rollout component ships without both accepting and rejecting tests
(RFC-0013 rule).

Accepting tests:

- `test_rollout_single_horizon1_accepts` — `horizon = 1`, `H = 3`: one append,
  trajectory commitment matches; exercises INV-ROLL-01/02/09/10
  ([#golden-vectors](../spec/07-testing-strategy.md#golden-vectors)).
- `test_rollout_horizon5_accepts` — V0 reference `horizon = 5`, `H = 3`, full
  recurrence; exercises INV-ROLL-04/05/06/07/08 across all five steps.
- `test_rollout_matches_rust_reference` — differential test: AIR-produced
  trajectory equals the Rust fixed-point reference rollout bit-for-bit
  ([#differential-tests](../spec/07-testing-strategy.md#differential-tests)).
- `test_rollout_matches_python_reference` — differential test: Rust reference and
  Python fixed-point reference agree on the trajectory for shared golden vectors,
  closing the export-parity loop with RFC-0001.
- `test_rollout_trajectory_commitment_stable` — same inputs, two builds, identical
  `claimed_output_commitment` (reproducibility, RFC-0016).
- `test_rollout_closure_reuses_predicted` — golden vector where a later step's
  window provably includes an earlier predicted latent (not a history latent);
  asserts INV-ROLL-08 is actually exercised, not vacuous.

Rejecting (negative) tests
([#negative-tests](../spec/07-testing-strategy.md#negative-tests)):

- `test_rollout_mutated_intermediate_latent_rejects` — flip one element of an
  intermediate predicted latent; verifier rejects via `E-ROLL-CLOSURE`
  (INV-ROLL-08).
- `test_rollout_wrong_window_index_rejects` — point step `s` at window base `s-1`;
  rejects via `E-ROLL-WINDOW-Z` (INV-ROLL-05).
- `test_rollout_misaligned_action_rejects` — read action at index `s+h+1`; rejects
  via `E-ROLL-WINDOW-A` (INV-ROLL-06).
- `test_rollout_off_circuit_predictor_rejects` — supply a `z_out` not produced by
  the linked predictor step; rejects via `E-ROLL-LINK` (INV-ROLL-07).
- `test_rollout_wrong_append_index_rejects` — write step `s` output at `H+s+1`;
  rejects via `E-ROLL-APPEND` (INV-ROLL-02).
- `test_rollout_append_time_order_rejects` — set `t_write <= t_read`; rejects via
  `E-ROLL-APPEND` (read-before-write, INV-ROLL-08).
- `test_rollout_history_seed_tamper_rejects` — change a seeded history cell;
  rejects via `E-ROLL-SEED` (INV-ROLL-09).
- `test_rollout_trajectory_commitment_tamper_rejects` — alter the committed
  trajectory bytes; rejects via `E-ROLL-TRAJ` (INV-ROLL-10).
- `test_rollout_config_horizon_mismatch_rejects` — trace built for `horizon = 6`,
  config commits `horizon = 5`; rejects via `E-ROLL-CONFIG` (INV-ROLL-11).
- `test_rollout_short_action_sequence_errors` — `len(actions) < H+horizon-1`;
  trace build fails with `E-ROLL-ACTLEN` (no proof emitted).
- `test_rollout_p0_submitted_as_p1_rejects` — a `predictor_step.v1` proof submitted
  under `StatementType::P1Rollout`; rejects via `E-ROLL-RELID` (RFC-0000).

Constraint mutation tests
([#mutation-tests](../spec/07-testing-strategy.md#mutation-tests)): mutate each of
INV-ROLL-01..04 and INV-ROLL-08 in turn (drop the constraint, weaken `=` to a
tautology) and assert that at least one accepting golden vector flips to accept an
invalid witness, confirming the constraint is load-bearing.

CI gates ([#ci-gates](../spec/07-testing-strategy.md#ci-gates)): all accepting,
rejecting, differential, and mutation tests above are required-green before the
`v0.2` milestone tag; the differential parity tests are also a release gate per
[docs/spec/09-release-and-versioning.md#changelog](../spec/09-release-and-versioning.md#changelog).

## Open Questions

- OWNER: `area:air` maintainer. RESOLUTION: RFC-0009. Whether the per-candidate
  rollout trajectories in P2 are each individually committed (one
  `claimed_output_commitment` per candidate, aggregated) or rolled into a single
  per-candidate-batched commitment is a P2 packaging decision; RFC-0008 commits one
  trajectory per rollout invocation and leaves the cross-candidate packaging to
  [RFC-0009](RFC-0009-fixed-candidate-planner-proof.md).
- OWNER: `area:air` maintainer. RESOLUTION: milestone `Future` /
  [RFC-0012](RFC-0012-recursive-aggregated-verification.md). Whether the
  per-element latent tiling (`horizon * latent_dim` bookkeeping rows) is replaced by
  a vectorized window-selection layout is a performance optimization gated on the
  performance budget ([docs/spec/08-performance-budget.md#scaling](../spec/08-performance-budget.md#scaling))
  and does not affect the locked recurrence or any invariant.

## References

- Founding analysis: docs/feasibility-study.md §1.2 (Statement P1), §7.8 (rollout
  component AIR), §9.5 (planner soundness), §12 Milestone 4 (rollout proof).
- [docs/spec/00-overview.md#scope-and-statement-tiers](../spec/00-overview.md#scope-and-statement-tiers),
  [docs/spec/00-overview.md#v0-statement](../spec/00-overview.md#v0-statement)
- [docs/spec/01-architecture.md#component-model](../spec/01-architecture.md#component-model),
  [docs/spec/01-architecture.md#trace-model](../spec/01-architecture.md#trace-model)
- [docs/spec/02-public-api.md#cli](../spec/02-public-api.md#cli)
- [docs/spec/03-data-model.md#public-input](../spec/03-data-model.md#public-input),
  [docs/spec/03-data-model.md#witness](../spec/03-data-model.md#witness),
  [docs/spec/03-data-model.md#tensor-memory-cells](../spec/03-data-model.md#tensor-memory-cells),
  [docs/spec/03-data-model.md#relation-id](../spec/03-data-model.md#relation-id)
- [docs/spec/04-error-model.md#failure-modes](../spec/04-error-model.md#failure-modes),
  [docs/spec/04-error-model.md#verifier-rejections](../spec/04-error-model.md#verifier-rejections)
- [docs/spec/06-security.md#soundness-requirements](../spec/06-security.md#soundness-requirements),
  [docs/spec/06-security.md#binding-requirements](../spec/06-security.md#binding-requirements)
- [docs/spec/07-testing-strategy.md](../spec/07-testing-strategy.md)
- [docs/spec/08-performance-budget.md#scaling](../spec/08-performance-budget.md#scaling)
- [docs/spec/09-release-and-versioning.md#relation-versioning](../spec/09-release-and-versioning.md#relation-versioning)
- Related RFCs:
  [RFC-0000](RFC-0000-security-model-and-statement-taxonomy.md),
  [RFC-0001](RFC-0001-model-manifest-and-export-pipeline.md),
  [RFC-0002](RFC-0002-fixed-point-arithmetic-over-m31.md),
  [RFC-0003](RFC-0003-range-check-and-lookup-infrastructure.md),
  [RFC-0004](RFC-0004-tensor-memory-and-wiring-air.md),
  [RFC-0005](RFC-0005-linear-matmul-and-requantization-components.md),
  [RFC-0007](RFC-0007-leworldmodel-predictor-air.md),
  [RFC-0009](RFC-0009-fixed-candidate-planner-proof.md),
  [RFC-0012](RFC-0012-recursive-aggregated-verification.md),
  [RFC-0013](RFC-0013-testing-fuzzing-and-audit-strategy.md),
  [RFC-0014](RFC-0014-canonical-serialization-and-transcript.md),
  [RFC-0016](RFC-0016-cli-artifact-bundle-and-reproducibility.md)
- LeWorldModel `jepa.py rollout()` (autoregressive concatenation of predicted
  embeddings and next actions), verified against upstream at 2026-06-03.
