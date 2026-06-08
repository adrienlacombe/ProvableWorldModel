# RFC-0007: LeWorldModel predictor AIR

- Status: Accepted
- Authors: ProvableWorldModel maintainers
- Created: 2026-06-03
- Target milestone: v0.2

## Summary

This RFC fixes the AIR-level decomposition of the LeWorldModel predictor and locks
the single-step prediction relation that the P0 proof (`pwm.lewm.predictor_step.v1`)
attests. The predictor is the autoregressive transformer that maps a latent history
plus encoded actions to a next latent. We decide its exact composition,
`z_next = PredProj_Q(ARPredictor_Q(z_history, ActionEncoder_Q(actions)))`, evaluated
over fixed-point integers embedded into M31; we decide the per-block structure
(AdaLN-zero `ConditionalBlock` modulation, scaled-dot-product attention, gated FFN)
at the V0 reference dimensions (depth 6, heads 16, dim_head 64, mlp_dim 2048, latent
192, history 3); and we decide that the AIR proves the EXPORTED quantized graph
emitted by `pwm-export`, never live PyTorch. This is the first proof that exercises
the full nonlinear primitive stack from
[docs/rfcs/RFC-0006-nonlinear-primitive-components.md](RFC-0006-nonlinear-primitive-components.md)
and the linear stack from
[docs/rfcs/RFC-0005-linear-matmul-and-requantization-components.md](RFC-0005-linear-matmul-and-requantization-components.md),
and it is the building block that
[docs/rfcs/RFC-0008-rollout-air.md](RFC-0008-rollout-air.md) iterates.

## Motivation

[docs/feasibility-study.md](../feasibility-study.md) §4.1 establishes that the V0
proof must support `action_encoder + predictor + pred_proj + rollout recurrence`
in latent space, and §7.7 sketches the attention AIR that this RFC makes
normative. The feasibility study (§2 risk register, §6) flags the predictor as the
core soundness risk precisely because it is "not a trivial MLP": it is an
autoregressive transformer with attention, AdaLN-style conditional blocks, gated
MLPs, learned positional embeddings, and action conditioning.

The problem this RFC solves: the predictor is the relation that P0 attests, and it
is also the inner kernel that P1 rollout
([docs/rfcs/RFC-0008-rollout-air.md](RFC-0008-rollout-air.md)) and P2 planning
([docs/rfcs/RFC-0009-fixed-candidate-planner-proof.md](RFC-0009-fixed-candidate-planner-proof.md))
invoke once per step per candidate. If its decomposition is not frozen, every
downstream proof's trace layout, public-input digest, and `relation_id` drift with
it. A new contributor must be able to wire the predictor component graph, allocate
trace columns, and emit the correct LogUp interaction relations from this document
alone.

Concrete scenario: a prover holds a committed `QuantizedLeWM` manifest
([docs/spec/03-data-model.md#model-manifest](../spec/03-data-model.md#model-manifest)),
a latent history of `history_size = 3` latents of dimension 192, and an action
window. It claims a `claimed_z_next` of dimension 192. The verifier
([docs/spec/02-public-api.md#rust-public-api](../spec/02-public-api.md#rust-public-api))
must reject unless every quantized linear, every attention probability, every AdaLN
modulation, and the final projection were computed bit-for-bit as the manifest
declares, with every integer range-checked so no field wrap can launder a wrong
value into a valid trace.

## Goals

- Lock the single-step predictor relation
  `z_next = PredProj_Q(ARPredictor_Q(z_history, ActionEncoder_Q(actions)))` as the
  body of `pwm.lewm.predictor_step.v1` (P0).
- Lock the `ConditionalBlock` composition for one predictor block: AdaLN-zero
  modulation (SiLU-gated), scaled-dot-product attention, gated FFN (GELU), and
  residual adds, at depth 6 / heads 16 / dim_head 64 / mlp_dim 2048 / latent 192 /
  history 3.
- Decide and name the per-module activation assignment (predictor FFN uses GELU;
  AdaLN modulation and the action `Embedder` use SiLU) so the AIR binds the correct
  lookup tables per op.
- Decide that the AIR proves the exported quantized op graph emitted by
  `pwm-export`, identified by op id and bound by `model_commitment` +
  `quantization_commitment`, not the live PyTorch module.
- Specify the `PredictorBlockComponent` / `ActionEncoderComponent` /
  `PredProjComponent` trace column layouts, their composition into
  `PredictorComponent`, and the tensor-memory and LogUp wiring between them.
- Enumerate every predictor-specific failure mode with the verifier response, and
  define the accepting and rejecting test set.

## Non-Goals

- The autoregressive windowing recurrence across steps (the multi-step `Rollout_Q`)
  is owned by [docs/rfcs/RFC-0008-rollout-air.md](RFC-0008-rollout-air.md). This RFC
  proves exactly one predictor step; the rollout RFC wires P0 steps into a
  trajectory and commits it.
- The internal arithmetization of nonlinear primitives (GELU_Q, Softmax_Q,
  LayerNorm_Q / AdaLN_Q numerics, lookup tables, Newton iterations) is owned by
  [docs/rfcs/RFC-0006-nonlinear-primitive-components.md](RFC-0006-nonlinear-primitive-components.md).
  This RFC consumes those primitives as components and only specifies how the
  predictor composes them.
- The linear / matmul / requant component internals are owned by
  [docs/rfcs/RFC-0005-linear-matmul-and-requantization-components.md](RFC-0005-linear-matmul-and-requantization-components.md);
  fixed-point semantics by
  [docs/rfcs/RFC-0002-fixed-point-arithmetic-over-m31.md](RFC-0002-fixed-point-arithmetic-over-m31.md);
  range/lookup infrastructure by
  [docs/rfcs/RFC-0003-range-check-and-lookup-infrastructure.md](RFC-0003-range-check-and-lookup-infrastructure.md);
  tensor memory by
  [docs/rfcs/RFC-0004-tensor-memory-and-wiring-air.md](RFC-0004-tensor-memory-and-wiring-air.md).
- Cost (goal-latent MSE) and argmin are owned by
  [docs/rfcs/RFC-0009-fixed-candidate-planner-proof.md](RFC-0009-fixed-candidate-planner-proof.md).
- The pixel encoder (`Encode_Q`, `PatchEmbed_Q`, `ViTEncoder_Q`, `Projector_Q`) is
  deferred to [docs/rfcs/RFC-0011-pixel-encoder-proof.md](RFC-0011-pixel-encoder-proof.md)
  (V3). The predictor proof takes `latent_history` and `goal_latent` as inputs and
  does not derive them from pixels.
- Training, the SIGReg regularizer, and CEM sampling are out of proving scope per
  [docs/spec/00-overview.md#non-goals](../spec/00-overview.md#non-goals); mentioned
  only as architectural context.

## Proposed Design

### D.1 The exported graph is the ground truth (locked)

The predictor AIR proves the **exported quantized op graph**, not live PyTorch. The
graph is the ordered `ops[]` list in the manifest
([docs/spec/03-data-model.md#model-manifest](../spec/03-data-model.md#model-manifest)),
produced by `pwm-export` per
[docs/rfcs/RFC-0001-model-manifest-and-export-pipeline.md](RFC-0001-model-manifest-and-export-pipeline.md).
Each op has a stable `id` (e.g. `predictor.block0.attn.qkv`), an `op` kind, scale
ids, and a `commitment`. The `model_commitment` binds architecture, weights, biases,
scales, rounding, lookup tables, and op order; `quantization_commitment` binds the
arithmetic/rounding/overflow policy. The AIR consumes op ids; the live PyTorch module
(`module.py` `ARPredictor` / `ConditionalBlock`) is the design reference for the
exporter, never an input to the prover or verifier.

Rationale: PyTorch bf16/GPU kernels are not a clean proof relation
([docs/feasibility-study.md](../feasibility-study.md) §1.1, §13 risk
"Floating-point ambiguity"). Binding to the exported integer graph makes the
relation exact and reproducible
([docs/spec/06-security.md#binding-requirements](../spec/06-security.md#binding-requirements)).

`INV-RFC0007-01` (graph fidelity): the set, order, and op kinds of the predictor
ops driven by the AIR equal the manifest `ops[]` entries whose ids are prefixed
`action_encoder.`, `predictor.`, or `pred_proj.`. The trace builder asserts this
before column allocation; mismatch is `TraceError::PredictorGraphMismatch`
([docs/spec/04-error-model.md#failure-modes](../spec/04-error-model.md#failure-modes)).

### D.2 The locked single-step relation

`pwm.lewm.predictor_step.v1` (P0) attests, over the M31-embedded fixed-point
semantics of
[docs/rfcs/RFC-0002-fixed-point-arithmetic-over-m31.md](RFC-0002-fixed-point-arithmetic-over-m31.md):

```text
z_next = PredProj_Q( ARPredictor_Q( z_history, ActionEncoder_Q(actions) ) )
```

where, with `D = latent_dim = 192`, `H = history_size = 3`:

- `z_history : Tensor[shape=[H, D], scale_id=s_latent]`  (BoundedInt cells)
- `actions   : Tensor[shape=[H, action_dim]]`            (raw action ids/values)
- `a_emb = ActionEncoder_Q(actions) : Tensor[shape=[H, D]]`
- `ARPredictor_Q : (Tensor[H,D], Tensor[H,D]) -> Tensor[D]` (last-position latent;
  `num_preds = 1`)
- `z_next = PredProj_Q(.) : Tensor[shape=[D]]`

The verifier checks `commit(z_next) == claimed_output_commitment` in
[`PublicInput`](../spec/03-data-model.md#public-input) (§6.3 of the contract). For
P0, `statement_type = StatementType::P0Step`,
`latent_history_public = Some(z_history as Vec<M31>)` (or its commitment),
`candidate_actions_public = Some(actions as Vec<M31>)`, `selected_index = None`,
`selected_cost = None`.

`INV-RFC0007-02` (relation binding): a P0 proof is valid only for
`relation_id == hash("pwm.lewm.predictor_step.v1")`. A semantic change to any op,
activation table, or composition mints a new id per
[docs/spec/09-release-and-versioning.md#relation-versioning](../spec/09-release-and-versioning.md#relation-versioning).

### D.3 ActionEncoder_Q

The action encoder is the `Embedder` from the reference model: a small MLP that
maps an action window to per-position latent-dimension embeddings, using **SiLU**
activation (verified against upstream at 2026-06-03). Decomposition:

```text
ActionEncoder_Q(actions):
  for each history position h in 0..H-1:
    u_h     = Linear_Q(action_in[h], W=action_encoder.l0.w, b=...)   # op: linear
    g_h     = SiLU_Q(u_h)                                            # op: silu_lookup_v1
    a_emb_h = Linear_Q(g_h,        W=action_encoder.l1.w, b=...)     # op: linear
  return a_emb : Tensor[H, D]
```

Each `Linear_Q` is an instance of `LinearComponent`
([docs/rfcs/RFC-0005-linear-matmul-and-requantization-components.md](RFC-0005-linear-matmul-and-requantization-components.md))
with its own op id, scales, and requant. `SiLU_Q` is `ActivationLookupComponent`
bound to the manifest table `silu_table_v1`
([docs/rfcs/RFC-0006-nonlinear-primitive-components.md](RFC-0006-nonlinear-primitive-components.md)).

`INV-RFC0007-03` (action activation): the action-encoder nonlinearity binds the
SiLU table id declared on the `action_encoder.*.silu` op, never the predictor's GELU
table. Cross-binding is `TraceError::ActivationTableMismatch`.

### D.4 ARPredictor_Q: positional/conditioning preamble + depth-6 blocks

The autoregressive predictor concatenates the latent history with the action
embeddings along the channel/sequence axis, adds learned positional embeddings,
forms an AdaLN conditioning signal from the action embedding, applies `depth = 6`
`ConditionalBlock`s, and reads off the last-position latent (`num_preds = 1`).

```text
ARPredictor_Q(z_history, a_emb):
  x   = AddPos_Q(combine(z_history, a_emb), pos_emb)   # broadcast add; pos_emb preprocessed
  c   = a_emb_last                                     # conditioning vector, dim D
  for blk in 0..depth-1:   # depth = 6
    x = ConditionalBlock_Q[blk](x, c)
  return last_position(x)   # Tensor[D]
```

`combine` and `AddPos_Q` are tensor-memory wiring ops
([docs/rfcs/RFC-0004-tensor-memory-and-wiring-air.md](RFC-0004-tensor-memory-and-wiring-air.md));
`pos_emb` is a fixed, manifest-committed tensor placed in the preprocessed trace
([docs/spec/01-architecture.md#trace-model](../spec/01-architecture.md#trace-model)).
The conditioning vector `c` is the action embedding at the conditioning position as
emitted by the exported graph.

`INV-RFC0007-04` (positional binding): `pos_emb` values used in the AIR equal the
manifest-committed positional-embedding tensor; a mutation changes
`model_commitment` and the LogUp read from the preprocessed table fails.

### D.5 ConditionalBlock_Q (the locked per-block composition)

One block, AdaLN-zero modulation (verified against upstream at 2026-06-03). The
block produces six modulation parameters from the conditioning vector via a
SiLU-gated linear, applies them around attention and FFN, and uses **GELU** inside
the FFN. AdaLN-zero means the gate parameters are initialized to zero so the
residual path dominates at init; at inference the exported scales are fixed and
committed.

```text
ConditionalBlock_Q(x, c):                      # x : Tensor[S, D], S = H + 1 (history + action token)
  # AdaLN-zero modulation parameters (SiLU then linear -> 6*D), op: predictor.blockK.adaln.mod
  m = Linear_Q(SiLU_Q(c), W=adaln.mod.w)        # op: silu_lookup_v1 then linear
  (shift_attn, scale_attn, gate_attn,
   shift_mlp,  scale_mlp,  gate_mlp) = split6(m)

  # --- Attention sub-block ---
  h1   = LayerNorm_Q(x, affine=false)           # op: layernorm (AdaLN: no learned affine)
  h1   = Modulate_Q(h1, shift_attn, scale_attn) # h1*(1+scale) + shift, fixed-point
  attn = Attention_Q(h1)                        # see D.6
  x    = x + Gate_Q(gate_attn, attn)            # residual; gate is per-channel scale

  # --- FFN sub-block ---
  h2   = LayerNorm_Q(x, affine=false)
  h2   = Modulate_Q(h2, shift_mlp, scale_mlp)
  ff   = FFN_Q(h2)                              # see D.7, uses GELU
  x    = x + Gate_Q(gate_mlp, ff)
  return x
```

`Modulate_Q(t, shift, scale)` computes `t * (1 + scale) + shift` per channel in
fixed point: one elementwise multiply (`PointwiseMul`-class), one add, one requant,
all range-checked. `Gate_Q(gate, t)` computes `t * gate` per channel with requant.
`LayerNorm_Q` here is the affine-free variant (AdaLN supplies affine via
`Modulate_Q`); its numerics (mean, variance, inverse-sqrt) are owned by
[docs/rfcs/RFC-0006-nonlinear-primitive-components.md](RFC-0006-nonlinear-primitive-components.md).

`INV-RFC0007-05` (modulation activation): the AdaLN modulation generator binds the
SiLU table, not GELU. `INV-RFC0007-06` (block residual integrity): each block emits
exactly two residual adds (`x + gate*attn`, `x + gate*ff`) wired through tensor
memory with `time` ordering
([docs/spec/03-data-model.md#tensor-memory-cells](../spec/03-data-model.md#tensor-memory-cells));
a missing or reordered residual is rejected by the tensor-memory multiset argument
of [docs/rfcs/RFC-0004-tensor-memory-and-wiring-air.md](RFC-0004-tensor-memory-and-wiring-air.md).

### D.6 Attention_Q (heads 16, dim_head 64)

Per head `k in 0..15`, with sequence length `S = H + 1 = 4` at the V0 reference and
`dim_head = 64` (note `heads * dim_head = 1024 != 192`; Q/K/V projections map
`D=192 -> 1024` and the output projection maps `1024 -> 192`, exactly as the
exported graph declares — the AIR reads the projection shapes from the op metadata,
it does not assume `heads*dim_head == D`). Faithful to
[docs/feasibility-study.md](../feasibility-study.md) §7.7:

```text
Attention_Q(x):                                # x : Tensor[S, D]
  Q = Linear_Q(x, W=attn.q)  -> [S, heads, dim_head]   # op: predictor.blockK.attn.q
  K = Linear_Q(x, W=attn.k)  -> [S, heads, dim_head]
  V = Linear_Q(x, W=attn.v)  -> [S, heads, dim_head]
  for each head k, each (i,j) in S x S:
    score[k,i,j]        = Sum_d Q[i,k,d] * K[j,k,d]              # MatMulComponent
    score_scaled[k,i,j] = Requantize_Q(score[k,i,j], scale=1/sqrt(dim_head)) # op: requant
    score_masked[k,i,j] = causal_mask(i,j) ? score_scaled : NEG_INF_Q        # preprocessed mask
  prob[k,i,:] = Softmax_Q(score_masked[k,i,:])                  # op: softmax_approx_v1, per row
  out[k,i,:]  = Sum_j prob[k,i,j] * V[j,k,:]                    # MatMulComponent
  return Linear_Q(reshape(out), W=attn.proj)                    # op: predictor.blockK.attn.proj
```

`Softmax_Q` (lookup-exp + reciprocal-denominator) is owned by
[docs/rfcs/RFC-0006-nonlinear-primitive-components.md](RFC-0006-nonlinear-primitive-components.md).
The causal mask and the `NEG_INF_Q` sentinel are preprocessed columns
([docs/spec/01-architecture.md#trace-model](../spec/01-architecture.md#trace-model));
`NEG_INF_Q` is the manifest-declared masking constant whose post-softmax probability
is provably the table's zero entry.

`INV-RFC0007-07` (attention range safety): every `score`, `score_scaled`, `prob`,
and `out` cell is range-checked
([docs/spec/06-security.md#soundness-requirements](../spec/06-security.md#soundness-requirements));
unconstrained "off-circuit softmax probabilities" are rejected by construction (no
probability column is admitted unless it is the constrained output of `Softmax_Q`).
`INV-RFC0007-08` (mask binding): `causal_mask` equals the manifest-committed mask;
a flipped mask bit fails the preprocessed-column LogUp read.

### D.7 FFN_Q (mlp_dim 2048, GELU)

The predictor FFN is a two-layer MLP with **GELU** (verified against upstream at
2026-06-03):

```text
FFN_Q(x):                                  # x : Tensor[S, D=192]
  u  = Linear_Q(x, W=mlp.fc1)  -> [S, 2048]     # op: predictor.blockK.mlp.fc1
  g  = GELU_Q(u)                                # op: gelu_lookup_v1
  y  = Linear_Q(g, W=mlp.fc2)  -> [S, D]        # op: predictor.blockK.mlp.fc2
  return y
```

The `mlp_dim = 2048` dot products are the dominant MAC cost
([docs/spec/08-performance-budget.md#cost-model](../spec/08-performance-budget.md#cost-model)).
The int8-MAC bound `2048 * 127 * 127 = 33,032,192 < 2^31 - 1`
([docs/feasibility-study.md](../feasibility-study.md) §5.2) means an int8 fc1/fc2
dot product fits in one signed M31 accumulator; if the manifest declares int16
activations on either op, the `LinearComponent` switches to its limb accumulator per
[docs/rfcs/RFC-0005-linear-matmul-and-requantization-components.md](RFC-0005-linear-matmul-and-requantization-components.md).

`INV-RFC0007-09` (FFN activation): the FFN nonlinearity binds the GELU table
(`gelu_table_v1`) declared on `predictor.blockK.mlp.gelu`, never SiLU.

### D.8 PredProj_Q

The prediction projector maps the last-position latent to the output latent. Per the
manifest `pred_proj.type` it is `mlp_or_affine`; for V0 it is exported as one or two
`Linear_Q` ops (no nonlinearity unless the manifest declares one):

```text
PredProj_Q(z):                              # z : Tensor[D]
  return Linear_Q(z, W=pred_proj.w, b=pred_proj.b)   # op: pred_proj.l0 (+ optional pred_proj.l1)
```

`INV-RFC0007-10` (projection shape): `PredProj_Q` output shape equals
`[latent_dim] == [192]`; a shape mismatch is rejected by the tensor-memory shape
metadata in the preprocessed trace.

### D.9 Component model and trace columns

The predictor is a composite AIR component built from the primitives of RFC-0004/05/06.
Per [docs/spec/01-architecture.md#component-model](../spec/01-architecture.md#component-model),
each is a Stwo constraint-framework component over the preprocessed/main/interaction
trace split
([docs/spec/01-architecture.md#trace-model](../spec/01-architecture.md#trace-model)).

```rust
// pwm-air::components::predictor
//
// ActionEncoderComponent : ops action_encoder.*
// PredictorBlockComponent: ops predictor.blockK.*  (one instance per block, K = 0..5)
// PredProjComponent      : ops pred_proj.*
// PredictorComponent     : composes the above; owns the P0 single-step relation.

pub struct PredictorComponent {
    pub relation_id: [u8; 32],          // pwm.lewm.predictor_step.v1
    pub latent_dim: u32,                // 192
    pub history_size: u32,              // 3
    pub depth: u32,                     // 6
    pub heads: u32,                     // 16
    pub dim_head: u32,                  // 64
    pub mlp_dim: u32,                   // 2048
    pub action_encoder: ActionEncoderComponent,
    pub blocks: Vec<PredictorBlockComponent>,   // len == depth
    pub pred_proj: PredProjComponent,
}
```

Per-block main-trace columns (one logical row group per op instance; concrete column
sets are owned by the primitive RFCs and referenced here):

```text
PredictorBlockComponent main-trace column groups
  block_id, op_selector                       # which sub-op a row belongs to
  adaln.mod   : LinearComponent columns over (c) -> 6*D, with SiLU input column
  shift/scale/gate (attn, mlp) : 6 modulation parameter vectors, range-checked
  ln1, ln2    : LayerNormComponent intermediates (mean, var, inv_std witness)
  attn.q/k/v  : LinearComponent columns -> [S, heads, dim_head]
  attn.score  : MatMulComponent columns (Q.K^T per head)
  attn.prob   : Softmax_Q columns (exp-lookup, denom, reciprocal witness)
  attn.out    : MatMulComponent columns (prob.V per head)
  attn.proj   : LinearComponent columns -> [S, D]
  mlp.fc1     : LinearComponent columns -> [S, mlp_dim]
  mlp.gelu    : ActivationLookupComponent columns
  mlp.fc2     : LinearComponent columns -> [S, D]
  residual    : two add columns wired to tensor memory
```

`INV-RFC0007-11` (depth fidelity): `blocks.len() == depth` and the K-th block binds
exactly the `predictor.blockK.*` op ids; a count or index mismatch is
`TraceError::PredictorGraphMismatch`.

### D.10 Tensor-memory and LogUp wiring (data flow)

Every inter-op value flows through the `TensorCell` relation (§6.4 of the contract;
[docs/rfcs/RFC-0004-tensor-memory-and-wiring-air.md](RFC-0004-tensor-memory-and-wiring-air.md)):

```text
TensorCell { tensor_id, index:[u32;4], value:M31, scale_id, time }
```

- Each op reads its inputs as LogUp lookups against prior `TensorCell` writes and
  writes its outputs as new cells with strictly increasing `time`.
- Weights are read from the weight table (`WeightTableComponent`) bound to
  `model_commitment`; for V0 `weights.visibility` may be `public` or
  `private_committed` ([docs/spec/03-data-model.md#model-manifest](../spec/03-data-model.md#model-manifest)).
- Range checks (u8/i8/i16/bounded-limb) and activation/softmax/inv-sqrt tables are
  LogUp interaction relations per
  [docs/rfcs/RFC-0003-range-check-and-lookup-infrastructure.md](RFC-0003-range-check-and-lookup-infrastructure.md);
  multiplicities are checked and the claimed sum must be consistent.

`INV-RFC0007-12` (wiring soundness): every predictor read corresponds to a unique
prior write or a manifest-committed constant; the tensor-memory multiset/permutation
argument fails otherwise.

### D.11 Determinism and error propagation

The trace builder (`pwm-prover::trace_builder`) executes the Rust fixed-point
reference over the exported graph in op order, emitting witness columns; the same
reference runs in `pwm-export` (Python) for golden vectors. Both must agree
bit-for-bit ([docs/spec/06-security.md#soundness-requirements](../spec/06-security.md#soundness-requirements)).
Challenges are derived in the shared Fiat-Shamir order of
[docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md](RFC-0014-canonical-serialization-and-transcript.md).

Predictor-specific failure modes and the system response
([docs/spec/04-error-model.md#failure-modes](../spec/04-error-model.md#failure-modes),
[docs/spec/04-error-model.md#verifier-rejections](../spec/04-error-model.md#verifier-rejections)):

| Failure mode | Where | System response |
| --- | --- | --- |
| Op graph differs from manifest `ops[]` (count/order/kind) | trace build | `TraceError::PredictorGraphMismatch`; abort before proving |
| Block count != `depth`, or block K not bound to `predictor.blockK.*` | trace build | `TraceError::PredictorGraphMismatch` |
| Wrong activation table bound (GELU where SiLU expected or vice versa) | trace build | `TraceError::ActivationTableMismatch` |
| Accumulator/product/score/prob/out out of declared range | constraint eval | LogUp range relation unsatisfied; `VerifyError::RangeCheckFailed` |
| Off-circuit / unconstrained attention probability supplied | constraint eval | no probability admitted outside `Softmax_Q`; `VerifyError::ConstraintUnsatisfied` |
| Mutated intermediate activation (any sub-op output) | constraint eval | constraint or tensor-memory mismatch; `VerifyError::ConstraintUnsatisfied` |
| Mutated `z_history` / `actions` vs public input | verify | `VerifyError::PublicInputMismatch` |
| `claimed_z_next` != computed `z_next` | verify | `VerifyError::OutputCommitmentMismatch` |
| `relation_id` != `pwm.lewm.predictor_step.v1` | verify | `VerifyError::UnsupportedRelation` |
| `model_commitment` or `quantization_commitment` mismatch | verify | `VerifyError::CommitmentMismatch` |
| Missing/reordered residual add | constraint eval | tensor-memory multiset mismatch; `VerifyError::ConstraintUnsatisfied` |
| Positional-embedding or causal-mask mutation | constraint eval | preprocessed-column LogUp read fails; `VerifyError::ConstraintUnsatisfied` |

## Alternatives Considered

**A1. Replace the AdaLN-zero conditional transformer with a proof-native distilled
predictor (RMSNorm + linear attention + committed-table activations), prove the
distilled model.** Considered because softmax, LayerNorm inverse-sqrt, and AdaLN are
the costliest primitives ([docs/feasibility-study.md](../feasibility-study.md) §6.2,
§6.3, §13). Rejected for V0: distillation produces a *different* model, so the proof
would attest `QuantizedLeWM-distilled`, not the committed LeWorldModel checkpoint,
breaking the V0 thesis of proving the exported reference architecture
([docs/spec/00-overview.md#thesis](../spec/00-overview.md#thesis)). The cost is paid
honestly through lookup-based primitives in RFC-0006 instead. Distillation remains a
legitimate option for a separate, clearly-labeled relation_id, not a substitute for
this one.

**A2. Lower the entire predictor to scalar circuit gates (the `stwo-circuits`
`Add/Sub/Mul/PointwiseMul/Eq/...` set) and prove it as one flat circuit.** Considered
because `stwo-circuits` already provides audited low-level gates and prover/verifier
plumbing ([docs/feasibility-study.md](../feasibility-study.md) §3.2). Rejected:
compiling the depth-6, mlp_dim-2048 dense matmuls to generic scalar gates is far more
expensive than dedicated tensor-operator AIR components, and `stwo-circuits` exposes
NO range/bit-extraction gate (verified against upstream at 2026-06-03 —
`extract_bits()` is a helper built from `sub`/`mul`/`assert_bits`, and range checking
is enforced by those constraints), so range safety would still require the LogUp
infrastructure of RFC-0003. Direct AIR is reserved for dense hot paths; circuit gates
are used only for glue (hashing, transcript) per
[docs/spec/01-architecture.md#air-strategy](../spec/01-architecture.md#air-strategy).

**A3. Prove the predictor with a uniform activation function (treat all
nonlinearities as one table).** Considered for trace simplicity. Rejected as
unsound: the reference model's activations are NOT uniform — the FFN uses GELU while
AdaLN modulation and the action `Embedder` use SiLU (verified against upstream at
2026-06-03). Binding the wrong table to an op would prove a different function than
the model computes, violating `INV-RFC0007-03/05/09`.

**A4. Make the AIR track live PyTorch op semantics (e.g. read `module.py` shapes at
prove time).** Rejected: PyTorch execution is nondeterministic across devices/dtypes
and is not a clean relation ([docs/feasibility-study.md](../feasibility-study.md)
§1.1); the verifier must never run PyTorch (§10.4). The exported graph (D.1) is the
only ground truth.

## Drawbacks

- The depth-6 / heads-16 / mlp_dim-2048 predictor is a large trace; even one P0 step
  is dominated by the two 2048-wide FFN matmuls per block and the per-head attention
  matmuls. Proving cost and column counts are non-trivial and tracked in
  [docs/spec/08-performance-budget.md#cost-model](../spec/08-performance-budget.md#cost-model).
- Faithfulness to the exported graph means the AIR is coupled to the manifest op
  schema; an export-schema change can force a predictor-component revision even when
  the math is unchanged. Mitigated by manifest schema versioning
  ([docs/spec/03-data-model.md#schema-versioning](../spec/03-data-model.md#schema-versioning)).
- The relation proves the quantized exported model, not float equivalence; downstream
  consumers must understand that `z_next` is the fixed-point result, not the PyTorch
  bf16 result ([docs/spec/00-overview.md#non-goals](../spec/00-overview.md#non-goals)).
- The attention layout assumes a small sequence length (`S = H + 1 = 4` at the V0
  reference). It is correct for larger `S` but the score matrix is `O(S^2)` per head;
  this is acceptable for the predictor but is a known scaling concern shared with the
  pixel encoder ([docs/rfcs/RFC-0011-pixel-encoder-proof.md](RFC-0011-pixel-encoder-proof.md)).

## Migration / Rollout

- **Relation versioning.** This RFC mints `pwm.lewm.predictor_step.v1`. Any change to
  the composition (D.2–D.8), activation assignment, or op set mints `.v2`; the two ids
  never share semantics ([docs/spec/09-release-and-versioning.md#relation-versioning](../spec/09-release-and-versioning.md#relation-versioning)).
- **Feature flag.** The predictor component lands behind a `pwm-air` cargo feature
  `predictor` (default-off until v0.2 stabilizes), so v0.1 (foundations) builds are
  unaffected. The flag is removed when v0.2 ships.
- **Manifest gating.** The exporter emits the predictor op graph only when
  `manifest_version >= pwm-model-manifest-v1` and `architecture.predictor.type ==
  ar_transformer`; older manifests are rejected at load with
  `ManifestError::UnsupportedPredictorType`
  ([docs/spec/04-error-model.md#error-taxonomy](../spec/04-error-model.md#error-taxonomy)).
- **Sequencing.** This RFC depends on RFC-0005 (linear/matmul/requant) and RFC-0006
  (nonlinear primitives) landing first; it is consumed by RFC-0008 (rollout). It is
  scoped to milestone `v0.2 — Predictor & Rollout` and gates the P0 proof.
- **Backward compatibility.** Adding the `predictor` component does not change any
  existing v0.1 type signature in contract §6; `PublicInput`/`Witness` already carry
  the fields the predictor proof needs (`predictor_activations`, `action_embeddings`).

## Testing Strategy

Cross-references [docs/spec/07-testing-strategy.md](../spec/07-testing-strategy.md)
and [docs/spec/04-error-model.md](../spec/04-error-model.md). No predictor sub-op
ships without both an accepting and a rejecting test
([docs/rfcs/RFC-0013-testing-fuzzing-and-audit-strategy.md](RFC-0013-testing-fuzzing-and-audit-strategy.md)).

Accepting tests ([docs/spec/07-testing-strategy.md#golden-vectors](../spec/07-testing-strategy.md#golden-vectors)):

- `accept_action_encoder_q_golden`: `ActionEncoder_Q` over a golden action window
  equals the exported reference `a_emb` bit-for-bit (SiLU table bound).
- `accept_conditional_block_q_golden`: one `ConditionalBlock_Q` (AdaLN-zero + attn +
  GELU FFN + residuals) matches the Rust fixed-point reference for golden inputs.
- `accept_predictor_step_p0_golden`: full
  `z_next = PredProj_Q(ARPredictor_Q(z_history, ActionEncoder_Q(actions)))` at depth
  6, proven and verified, `commit(z_next) == claimed_output_commitment`.
- `accept_predictor_batch_steps`: a batch of independent P0 steps each verify (the
  building block for rollout/planning batching).
- `accept_predictor_private_committed_weights`: same as above with
  `weights.visibility = private_committed`; verifier accepts using only commitments.
- Differential ([docs/spec/07-testing-strategy.md#differential-tests](../spec/07-testing-strategy.md#differential-tests)):
  `diff_python_rust_predictor_step` asserts the Python and Rust fixed-point references
  agree bit-for-bit across random seeded inputs.

Rejecting tests ([docs/spec/07-testing-strategy.md#negative-tests](../spec/07-testing-strategy.md#negative-tests),
mapped to [docs/spec/04-error-model.md#verifier-rejections](../spec/04-error-model.md#verifier-rejections)):

- `reject_changed_action`: mutate one action -> `VerifyError::PublicInputMismatch`.
- `reject_changed_latent_history`: mutate one history latent ->
  `VerifyError::PublicInputMismatch`.
- `reject_changed_claimed_output`: mutate `claimed_z_next` ->
  `VerifyError::OutputCommitmentMismatch`.
- `reject_mutated_block_activation`: perturb one block's attn/FFN intermediate ->
  `VerifyError::ConstraintUnsatisfied`.
- `reject_offcircuit_softmax_prob`: supply an attention probability not produced by
  `Softmax_Q` -> `VerifyError::ConstraintUnsatisfied`.
- `reject_wrong_gelu_silu_binding`: bind SiLU table to the FFN op (or GELU to AdaLN)
  -> `TraceError::ActivationTableMismatch` at build (covers
  `INV-RFC0007-03/05/09`).
- `reject_dropped_residual`: omit a block residual add -> tensor-memory multiset
  mismatch -> `VerifyError::ConstraintUnsatisfied` (covers `INV-RFC0007-06`).
- `reject_mutated_pos_emb` / `reject_mutated_causal_mask`: preprocessed-column LogUp
  read fails -> `VerifyError::ConstraintUnsatisfied` (covers `INV-RFC0007-04/08`).
- `reject_accumulator_field_wrap`: craft an accumulator that wraps mod p but lands on
  a valid-looking field value -> range relation unsatisfied ->
  `VerifyError::RangeCheckFailed` (covers `INV-RFC0007-07`).
- `reject_wrong_relation_id`: submit with `pwm.lewm.rollout.v1` ->
  `VerifyError::UnsupportedRelation` (covers `INV-RFC0007-02`).
- `reject_graph_count_mismatch`: manifest declares depth 6 but trace builds 5 blocks
  -> `TraceError::PredictorGraphMismatch` (covers `INV-RFC0007-01/11`).
- Constraint mutation ([docs/spec/07-testing-strategy.md#mutation-tests](../spec/07-testing-strategy.md#mutation-tests)):
  `mutate_modulate_constraint`, `mutate_attention_dot_constraint`, and
  `mutate_residual_constraint` each must cause at least one accepting test to fail
  (no dead constraints).

## Open Questions

- OPEN QUESTION (owner: area:air maintainer; resolution: RFC-0006 at v0.2): whether
  the affine-free `LayerNorm_Q` used inside AdaLN shares the same inverse-sqrt table
  as a future standalone LayerNorm, or binds a distinct table id. This RFC requires
  only that the bound table is committed in the manifest; the table-sharing policy is
  RFC-0006's call.
- OPEN QUESTION (owner: area:export maintainer; resolution: RFC-0001 at v0.2):
  whether `pred_proj` exports as one or two `Linear_Q` ops for the V0 checkpoint. The
  AIR (D.8) supports both via op ids; the exporter pins the concrete count, which
  then becomes part of `model_commitment`.
- OPEN QUESTION (owner: area:air maintainer; resolution: milestone v1.0): whether the
  per-head attention score matrices are laid out as one batched `MatMulComponent` over
  the head axis or as per-head instances. Both satisfy this RFC's relation; the choice
  is a performance decision deferred to
  [docs/spec/08-performance-budget.md#scaling](../spec/08-performance-budget.md#scaling).

## References

- [docs/feasibility-study.md](../feasibility-study.md) §4.1 (LeWorldModel components,
  V0 scope), §6 (nonlinear operations), §7.7 (attention AIR sketch), §1.1 (model
  vs float distinction), §5.2 (int8 MAC bound).
- [docs/spec/00-overview.md#thesis](../spec/00-overview.md#thesis),
  [#non-goals](../spec/00-overview.md#non-goals),
  [#v0-statement](../spec/00-overview.md#v0-statement).
- [docs/spec/01-architecture.md#air-strategy](../spec/01-architecture.md#air-strategy),
  [#component-model](../spec/01-architecture.md#component-model),
  [#trace-model](../spec/01-architecture.md#trace-model),
  [#data-flow](../spec/01-architecture.md#data-flow).
- [docs/spec/02-public-api.md#rust-public-api](../spec/02-public-api.md#rust-public-api).
- [docs/spec/03-data-model.md#model-manifest](../spec/03-data-model.md#model-manifest),
  [#public-input](../spec/03-data-model.md#public-input),
  [#witness](../spec/03-data-model.md#witness),
  [#tensor-memory-cells](../spec/03-data-model.md#tensor-memory-cells),
  [#schema-versioning](../spec/03-data-model.md#schema-versioning).
- [docs/spec/04-error-model.md#error-taxonomy](../spec/04-error-model.md#error-taxonomy),
  [#failure-modes](../spec/04-error-model.md#failure-modes),
  [#verifier-rejections](../spec/04-error-model.md#verifier-rejections).
- [docs/spec/06-security.md#soundness-requirements](../spec/06-security.md#soundness-requirements),
  [#binding-requirements](../spec/06-security.md#binding-requirements).
- [docs/spec/07-testing-strategy.md#golden-vectors](../spec/07-testing-strategy.md#golden-vectors),
  [#negative-tests](../spec/07-testing-strategy.md#negative-tests),
  [#differential-tests](../spec/07-testing-strategy.md#differential-tests),
  [#mutation-tests](../spec/07-testing-strategy.md#mutation-tests).
- [docs/spec/08-performance-budget.md#cost-model](../spec/08-performance-budget.md#cost-model),
  [#scaling](../spec/08-performance-budget.md#scaling).
- [docs/spec/09-release-and-versioning.md#relation-versioning](../spec/09-release-and-versioning.md#relation-versioning).
- Related RFCs:
  [RFC-0001](RFC-0001-model-manifest-and-export-pipeline.md),
  [RFC-0002](RFC-0002-fixed-point-arithmetic-over-m31.md),
  [RFC-0003](RFC-0003-range-check-and-lookup-infrastructure.md),
  [RFC-0004](RFC-0004-tensor-memory-and-wiring-air.md),
  [RFC-0005](RFC-0005-linear-matmul-and-requantization-components.md),
  [RFC-0006](RFC-0006-nonlinear-primitive-components.md),
  [RFC-0008](RFC-0008-rollout-air.md),
  [RFC-0009](RFC-0009-fixed-candidate-planner-proof.md),
  [RFC-0013](RFC-0013-testing-fuzzing-and-audit-strategy.md),
  [RFC-0014](RFC-0014-canonical-serialization-and-transcript.md).
- LeWorldModel reference architecture (github.com/lucas-maes/le-wm, MIT;
  `ARPredictor`/`ConditionalBlock` in `module.py`, rollout in `jepa.py`; verified
  against upstream at 2026-06-03).
