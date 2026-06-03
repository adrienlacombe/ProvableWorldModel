# RFC-0011: Pixel encoder proof

- Status: Draft
- Authors: ProvableWorldModel maintainers
- Created: 2026-06-03
- Target milestone: Future

## Summary

This RFC locks the scope and arithmetization of the pixel encoder proof,
proof statement P4 (`pwm.lewm.pixel_to_plan.v1`), which closes the gap between
proven latent computation (P0/P1/P2/P3) and the raw pixel observations a real
deployment consumes. It defines four proof-native components,
`PatchEmbed_Q`, `ViTEncoder_Q`, `Projector_Q`, and the composite `Encode_Q`,
that prove `z = Encode_Q(image)` for the LeWorldModel ViT-tiny encoder
(224x224 input, patch size 14, hidden 192). The decision this RFC makes
permanent: P4 is **deferred to release V3**, the encoder is arithmetized as a
distinct set of AIR components rather than folded into the predictor AIR, and
the encoder proof **composes with the rollout/planner proof by latent
commitment equality** rather than by sharing a single monolithic trace. The
encoder dominates proving cost because it produces an order of magnitude more
ViT tokens than the predictor sees (256 patch tokens plus CLS, versus a
history window of 3), and each ViT block carries the same attention, softmax,
and LayerNorm/AdaLN cost the predictor block carries but over a token count
that is roughly 64x larger. P4 is written to full quality here so that V3 work
starts from a frozen design, but it is gated behind the dependencies named in
Goals and Open Questions and is out of scope for v0.1, v0.2, and v1.0.

## Motivation

The V0 statement
([docs/spec/00-overview.md#v0-statement](../spec/00-overview.md#v0-statement))
proves quantized latent planning given an initial latent history and a goal
latent **as inputs**. It assumes those latents are valid encoder outputs but
does not prove it. A verifier accepting a P2 proof learns that the committed
predictor was rolled out correctly over committed latents; it learns nothing
about whether those latents came from the observation and goal images a caller
believes they did. The founding analysis
([docs/feasibility-study.md](../feasibility-study.md), source RFC-011 and the
P4 definition in source §1.2) identifies this as the final, most expensive
arithmetization tier and recommends deferring it until the latent path is
stable. The risk register entry "Encoder cost" (source §13) rates it High and
prescribes deferral to V3.

Concrete scenario this RFC enables: a caller submits a history of camera
frames and a goal image, and receives a single proof that (a) each frame and
the goal were encoded by the committed quantized encoder into the exact latents
fed to planning, and (b) the planner selected the minimum-cost candidate over
those latents. Without P4, the binding between pixels and the planning proof is
asserted by the caller, not proven, which is the gap an end-to-end pixel-to-plan
claim must close.

The encoder structure to arithmetize is fixed by the upstream model. The
LeWorldModel `encode` path flattens the time dimension, applies the ViT-tiny
encoder, extracts the CLS token, projects it through `projector`, and reshapes
back into latent-sequence form (verified against upstream `jepa.py` at
2026-06-03). The encoder is a standard ViT-tiny: a patch-embedding convolution,
a learned CLS token and positional embeddings, a stack of pre-norm transformer
blocks (LayerNorm, multi-head attention with `F.scaled_dot_product_attention`,
GELU MLP), a final LayerNorm, and CLS extraction. This RFC arithmetizes that
exact graph as exported, not the live PyTorch module, consistent with the
predictor rule in
[docs/rfcs/RFC-0007-leworldmodel-predictor-air.md](RFC-0007-leworldmodel-predictor-air.md).

## Goals

- Lock P4 scope to four components: `PatchEmbed_Q`, `ViTEncoder_Q`,
  `Projector_Q`, and the composite `Encode_Q`, over the V3 reference
  configuration (224x224, patch 14, ViT-tiny hidden 192).
- Lock the relation identifier `pwm.lewm.pixel_to_plan.v1` and the
  encoder-only sub-relation `pwm.lewm.encode.v1`, both immutable per
  [docs/spec/09-release-and-versioning.md#relation-versioning](../spec/09-release-and-versioning.md#relation-versioning).
- Lock that the encoder proof composes with the rollout/planner proof by
  **latent commitment equality**, not by a shared monolithic trace, so the
  encoder is provable, benchmarkable, and recursable independently.
- Make the pixel-domain binding explicit and testable: the encoder input is
  bound to a committed `image` tensor with a declared pixel quantization, so a
  prover cannot substitute pixels off-circuit.
- Reuse the existing AIR component vocabulary (linear/matmul/requant from
  [docs/rfcs/RFC-0005-linear-matmul-and-requantization-components.md](RFC-0005-linear-matmul-and-requantization-components.md),
  nonlinear primitives from
  [docs/rfcs/RFC-0006-nonlinear-primitive-components.md](RFC-0006-nonlinear-primitive-components.md))
  rather than minting encoder-specific arithmetic, adding only a
  patch-embedding adapter and a token-batching axis.
- State the cost drivers precisely (token count, attention/softmax/LayerNorm
  over image tokens, activation memory) and the deferral gating, so V3
  planning has a frozen surface and a measured budget target
  ([docs/spec/08-performance-budget.md#scaling](../spec/08-performance-budget.md#scaling)).

## Non-Goals

- This RFC does not ship P4 in V0/V1/V2. P4 is V3; V0 is the P2 statement
  ([docs/spec/00-overview.md#scope-and-statement-tiers](../spec/00-overview.md#scope-and-statement-tiers)).
- It does not prove the encoder is faithful to the original floating-point
  ViT. As with all PWM statements, the proof binds the exported quantized graph
  ("QuantizedLeWM" encoder), not bf16/GPU PyTorch
  ([docs/spec/06-security.md#approximation-requirements](../spec/06-security.md#soundness-requirements)).
- It does not introduce new nonlinear arithmetization. GELU/Softmax/LayerNorm
  semantics are owned by
  [docs/rfcs/RFC-0006-nonlinear-primitive-components.md](RFC-0006-nonlinear-primitive-components.md);
  this RFC consumes them at a larger token count and adds no new primitive.
- It does not specify recursion. Whether P4 is proven monolithically or split
  per-image and aggregated is owned by
  [docs/rfcs/RFC-0012-recursive-aggregated-verification.md](RFC-0012-recursive-aggregated-verification.md);
  this RFC only fixes the composition interface (latent commitment equality)
  that recursion would consume.
- It does not address image preprocessing outside the proof boundary (camera
  capture, resize, color conversion). The proof begins at the committed
  quantized `image` tensor; upstream preprocessing is a caller responsibility,
  flagged as an Open Question for the security owner.
- It does not address training, SIGReg, or the encoder's role in the JEPA loss.
  Training is out of proving scope corpus-wide.

## Proposed Design

### Scope and statement

P4 proves, for a committed quantized encoder and a committed pixel input, that
the claimed latents are exactly the encoder output, and that planning over
those latents is correct:

```text
Public:
  relation_id = pwm.lewm.pixel_to_plan.v1
  model_commitment, quantization_commitment, planner_config_commitment
  image_history_commitment        // committed pixel observation history
  image_goal_commitment            // committed pixel goal
  selected_index, selected_cost

Claim:
  z_history == Encode_Q(image_history)         // per frame, sequence-batched
  z_goal    == Encode_Q(image_goal)
  selected_index, selected_cost are the correct P2 result over
      (z_history, z_goal, candidate_action_sequences)
  under deterministic argmin tie-break.
```

The encoder-only sub-statement `pwm.lewm.encode.v1` proves just
`z == Encode_Q(image)` for one image (or a sequence-batched stack), with
`claimed_output_commitment` binding the produced latent. P4 is the composition
of N+1 `encode.v1` instances (N history frames + 1 goal) with one
`fixed_candidate_planning.v1` instance, joined by latent commitment equality
(see Composition below).

`StatementType::P4PixelToPlan` is the canonical variant
([docs/spec/03-data-model.md#public-input](../spec/03-data-model.md#public-input),
contract §6.3). A proof is valid only for its declared `relation_id`
([docs/rfcs/RFC-0000-security-model-and-statement-taxonomy.md](RFC-0000-security-model-and-statement-taxonomy.md)).

### Reference configuration (V3 target)

Verified against upstream le-wm at 2026-06-03 (MIT-licensed; the export
pipeline consumes a checkpoint and config, it does not vendor le-wm source):

| Parameter | Value | Source |
| --- | --- | --- |
| Image size | 224 x 224 | le-wm config |
| Channels | 3 | RGB |
| Patch size | 14 | le-wm config |
| Patches per axis | 16 (224/14) | derived |
| Patch tokens | 256 (16 x 16) | derived |
| Tokens incl. CLS | 257 | derived |
| Encoder hidden / embed_dim | 192 | ViT-tiny |
| Encoder depth | OPEN QUESTION (see below) | ViT-tiny variant |
| Output latent_dim | 192 | le-wm config |
| Activations | GELU (MLP), SiLU (none in encoder MLP) | upstream |

The encoder MLP uses GELU. SiLU in le-wm appears in the predictor's AdaLN
modulation and the action `Embedder`, not the encoder
([docs/rfcs/RFC-0006-nonlinear-primitive-components.md](RFC-0006-nonlinear-primitive-components.md));
encoder blocks are standard pre-norm ViT (LayerNorm + attention + GELU MLP).
Per-module activation selection is bound by `quantization_commitment`.

OPEN QUESTION: the exact ViT-tiny encoder depth and head count for the V3
checkpoint are not fixed in this RFC because the latent path (V0/V2) does not
exercise them. Owner: export pipeline maintainer
([area:export](../rfcs/RFC-0001-model-manifest-and-export-pipeline.md)).
Resolution path: recorded in the V3 manifest at the milestone `Future` when the
encoder checkpoint is frozen; the manifest binds depth/heads via
`model_commitment` regardless, so soundness does not depend on naming the value
here.

### Component vocabulary

P4 adds three encoder components plus a composite, all reusing existing
arithmetic. No new fixed-point or nonlinear primitive is introduced.

```text
PatchEmbed_Q   — pixel patches -> patch embeddings (conv-as-matmul) + CLS + pos
ViTEncoder_Q   — depth pre-norm ViT blocks over 257 tokens
Projector_Q    — CLS-token projection to latent_dim
Encode_Q       — PatchEmbed_Q -> ViTEncoder_Q -> final LN -> CLS -> Projector_Q
```

#### PatchEmbed_Q

The patch embedding is a `Conv2d(3, 192, kernel=14, stride=14)` with no padding
and no overlap. Because stride equals kernel size, every output patch is an
independent dot product over a disjoint `3 x 14 x 14 = 588`-element input window.
This is arithmetized as a `matmul` (RFC-0005), not a generic convolution
component: the 588-element patch is the contraction axis, 192 is the output
channel axis, and 256 is the patch (token) batch axis. Pixel values are
quantized per the manifest pixel-quantization entry (see Pixel binding below).

```text
For patch p in 0..255, output channel j in 0..191:
  acc_{p,j,0} = bias_j
  for k in 0..587:
      prod_{p,j,k} = pixel_{p,k} * w_{k,j}
      acc_{p,j,k+1} = acc_{p,j,k} + prod_{p,j,k}
  patch_embed_{p,j} = Requantize_Q(acc_{p,j,588}, shift, zero_point, clamp)
After patch embedding:
  token_0 = CLS_embedding              (preprocessed/committed constant)
  token_{p+1} = patch_embed_p + pos_{p+1}
  token_0    = CLS_embedding + pos_0
```

Accumulator bound: int8 pixels and int8 weights over 588 terms give a
worst-case magnitude `588 * 127 * 127 = 9,483,852 < 2^31 - 1`, so a single
signed M31 accumulator is safe (no limb decomposition needed for patch
embedding). This is checked from tensor metadata by the AIR, not assumed; if
the manifest declares wider pixel or weight ranges, the component switches to
the limb accumulator strategy of RFC-0005. CLS and positional embeddings are
fixed columns in the preprocessed trace
([docs/spec/01-architecture.md#trace-model](../spec/01-architecture.md#trace-model)),
bound by `model_commitment`.

#### ViTEncoder_Q

A stack of `depth` pre-norm ViT blocks over 257 tokens:

```text
For each block:
  h1 = LayerNorm_Q(x)                          // RFC-0006
  a  = Attention_Q(h1)                          // RFC-0006, RFC-0005 for Q/K/V/out
  x  = x + a                                    // residual add (RFC-0002 add)
  h2 = LayerNorm_Q(x)
  m  = Linear_Q(GELU_Q(Linear_Q(h2)))           // RFC-0005 + RFC-0006
  x  = x + m
After all blocks:
  x  = LayerNorm_Q(x)                            // final norm
```

Attention here is identical in form to the predictor's attention (source §7.7)
but the token (sequence) axis is 257, not the predictor's history window of 3.
The attention score matrix is `257 x 257` per head; softmax is over 257
elements per row. This is the cost driver (see Cost drivers). The encoder is
non-causal (full bidirectional attention over image tokens), so there is no
causal mask; the mask selector columns of the attention component
([docs/rfcs/RFC-0006-nonlinear-primitive-components.md](RFC-0006-nonlinear-primitive-components.md))
are all-pass for the encoder and that fact is bound by the encoder op entry in
the manifest.

#### Projector_Q

CLS extraction selects token 0 from the final-LN output (a tensor-memory read
of `index == 0` on the token axis, RFC-0004), then applies `projector` (an MLP
or affine, exported shape). Output is the latent vector for that image.

#### Encode_Q and sequence batching

`encode` flattens time and encodes each frame independently. The encoder
component batches over the frame (time) axis as an additional batch dimension,
identical in structure to the candidate/rollout batching axes of RFC-0005. For
a history of H frames plus 1 goal, `Encode_Q` runs H+1 times sharing the same
weight tables and preprocessed constants; outputs are written to tensor memory
as the latent sequence consumed by rollout.

### Pixel binding (INV-RFC0011-01)

The proof boundary begins at a committed quantized `image` tensor. Pixels are
represented as `BoundedInt` (contract §6.1) with a declared inclusive range in
the manifest pixel-quantization entry (canonical default `[0, 255]` for u8
pixels, centered if the manifest declares a zero-point). The `image` tensor is
bound by `image_history_commitment` / `image_goal_commitment` in `PublicInput`.

```rust
// Manifest pixel-quantization entry (bound by quantization_commitment).
ops:
  - id: encoder.patch_embed.input
    op: pixel_quant
    pixel_range: { lo: 0, hi: 255 }      // BoundedInt range, inclusive
    zero_point: 0
    scale_id: <u32>                       // index into manifest scale table
```

INV-RFC0011-01 (pixel binding): every value entering `PatchEmbed_Q` is a
`BoundedInt` whose range is declared in the manifest pixel-quantization entry
and range-checked via the LogUp infrastructure of
[docs/rfcs/RFC-0003-range-check-and-lookup-infrastructure.md](RFC-0003-range-check-and-lookup-infrastructure.md).
A prover cannot feed an unbounded or out-of-range pixel; the field wraps,
integer inference must not (source §5.1, §9.3).

### Composition with the rollout/planner proof (INV-RFC0011-02)

The decision: encoder and planner are **separate AIR instances joined by latent
commitment equality**, not one trace.

INV-RFC0011-02 (latent commitment equality): for P4 to verify, the
`claimed_output_commitment` of each `encode.v1` instance must equal the
corresponding latent commitment consumed by the `fixed_candidate_planning.v1`
instance. Specifically:

```text
encode.v1[frame_i].claimed_output_commitment  == planner.latent_history_commitment[i]
encode.v1[goal].claimed_output_commitment      == planner.goal_latent_commitment
```

The commitment is over the canonical serialization of the latent `Tensor`
([docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md](RFC-0014-canonical-serialization-and-transcript.md)).
This is the same `Tensor` / commitment machinery used everywhere
(contract §6.2, §6.3). In V0/V2 these latents are public/committed inputs; in
P4 they become outputs of the encoder proof and inputs to the planner proof,
checked equal.

Why commitment equality and not a shared trace:

- The encoder trace is roughly 64x the per-block token count of the predictor
  (257 vs 3 tokens). A monolithic P4 trace would be dominated by encoder rows
  and would force encoder and planner to share one proving run, defeating
  independent benchmarking and recursion. Commitment equality keeps each
  statement provable, testable, and aggregatable on its own.
- It matches the recursion decomposition surface that
  [docs/rfcs/RFC-0012-recursive-aggregated-verification.md](RFC-0012-recursive-aggregated-verification.md)
  consumes: per-image encode proofs plus a planner proof, joined by output
  commitments, aggregate cleanly.

The composition is realized in one of two rollout modes, decided at V3:
(a) **monolithic** — a single P4 trace containing all encoder and planner
components, with the equality checked as in-circuit tensor-memory reads; or
(b) **aggregated** — separate proofs verified together by RFC-0012, with
equality checked over public `claimed_output_commitment` fields. This RFC fixes
the interface (commitment equality) so both modes are sound; the choice between
them is an Open Question for the verifier owner gated on RFC-0012 benchmarks.

### Public input and witness types

P4 reuses the canonical `PublicInput` and `Witness` (contract §6.3) verbatim;
no new top-level type is introduced. The pixel inputs ride in the existing
optional commitment/public fields, reinterpreted for the encoder statement:

```rust
pub enum StatementType { P0Step, P1Rollout, P2FixedCandidatePlanning,
                         P3Cem, P4PixelToPlan }

pub struct PublicInput {
    pub relation_id: [u8; 32],
    pub model_commitment: [u8; 32],
    pub quantization_commitment: [u8; 32],
    pub planner_config_commitment: [u8; 32],
    pub statement_type: StatementType,
    pub latent_history_commitment: Option<[u8; 32]>,   // P4: encoder OUTPUT commitment
    pub latent_history_public: Option<Vec<M31>>,
    pub goal_latent_commitment: Option<[u8; 32]>,       // P4: encoder OUTPUT commitment
    pub goal_latent_public: Option<Vec<M31>>,
    pub candidate_actions_commitment: Option<[u8; 32]>,
    pub candidate_actions_public: Option<Vec<M31>>,
    pub claimed_output_commitment: [u8; 32],
    pub selected_index: Option<u32>,
    pub selected_cost: Option<BoundedInt>,
}
```

For P4, `latent_history_commitment` and `goal_latent_commitment` are the
encoder-produced latent commitments (the join points of INV-RFC0011-02). The
committed pixel inputs are carried as committed witness tensors; the
`image_*_commitment` fields named in the statement above are realized as
`model`/`quantization`-independent input commitments recorded in the
schema-versioned artifact ([docs/spec/02-public-api.md#artifact-formats](../spec/02-public-api.md#artifact-formats)).

OPEN QUESTION: whether to add explicit `image_history_commitment` /
`image_goal_commitment` fields to `PublicInput` or carry them as the first
entries of the input-commitment list. Owner: data-model maintainer
([area:core](../spec/03-data-model.md#schema-versioning)). Resolution path: a
`PublicInput` schema bump tracked at milestone `Future`, decided with RFC-0016's
artifact bundle layout; the canonical signature in contract §6.3 is unchanged
for V0/V1/V2 and a V3 schema version adds the fields if needed.

The witness adds encoder activations to the existing `predictor_activations`
discipline:

```rust
pub struct Witness {
    pub model_weights: Option<QuantizedWeights>,
    pub latent_history: Tensor,            // P4: equals Encode_Q(image_history) output
    pub goal_latent: Option<Tensor>,        // P4: equals Encode_Q(image_goal) output
    pub candidate_actions: Option<Tensor>,
    pub action_embeddings: Tensor,
    pub predictor_activations: Vec<Tensor>, // P4: also holds encoder block activations
    pub rollout_trajectory: Tensor,
    pub costs: Option<Vec<BoundedInt>>,
    pub argmin_witness: Option<ArgminWitness>,
    pub range_witnesses: Vec<RangeWitness>,
    pub lookup_witnesses: Vec<LookupWitness>,
}
```

The image tensors and per-token encoder activations are carried in
`predictor_activations` (the catch-all activation list) for V3, or split into a
dedicated `encoder_activations` field at the same V3 schema bump as the pixel
commitments. Either way they are redacted private witnesses
([docs/spec/05-observability.md#redaction](../spec/05-observability.md#redaction)).

### Cost drivers (the deferral rationale)

P4 is deferred to V3 because the encoder is the most expensive arithmetization
tier in the corpus. The drivers, made concrete:

| Driver | Predictor (P0 block) | Encoder (ViT block) | Ratio |
| --- | --- | --- | --- |
| Tokens per block | 3 (history window) | 257 (256 patches + CLS) | ~86x |
| Attention score matrix | 3 x 3 per head | 257 x 257 per head | ~7,300x |
| Softmax rows per block | 3 | 257 | ~86x |
| LayerNorm invocations | per token | per token | ~86x by tokens |
| Patch-embed MACs | n/a | 256 x 192 x 588 ~= 28.9M | new cost |

The attention score matrix grows quadratically in token count; at 257 tokens
the `O(tokens^2)` softmax and score-matmul rows dominate the encoder trace, and
the encoder runs H+1 times (once per frame plus goal). Activation memory for
257-token intermediates across `depth` blocks is the second constraint. The
performance budget owner records the encoder cost model and the P4 trace-size
estimate at
[docs/spec/08-performance-budget.md#scaling](../spec/08-performance-budget.md#scaling)
and gates V3 on it; the deferral is not "encoder is hard" hand-waving but a
budgeted decision tracked against
[docs/spec/08-performance-budget.md#targets](../spec/08-performance-budget.md#targets).

### Determinism and error propagation

The encoder export, quantization, GELU/softmax/LayerNorm approximations, pixel
quantization, and patch-embedding accumulation are all deterministic and bound
by `model_commitment` / `quantization_commitment`
([docs/spec/06-security.md#binding-requirements](../spec/06-security.md#binding-requirements)).
Any divergence between the Python fixed-point reference and the Rust reference
is an export/parity failure surfaced before proving
([docs/spec/04-error-model.md#failure-modes](../spec/04-error-model.md#failure-modes)).
Failure modes specific to P4 and the system response:

| Failure mode | System response |
| --- | --- |
| Pixel value outside declared `pixel_range` | Trace build rejects; `ManifestError`/`TraceError` before proving ([docs/spec/04-error-model.md#error-taxonomy](../spec/04-error-model.md#error-taxonomy)) |
| Patch-embed accumulator exceeds declared bound | Range-check fails; `VerifyError::RangeCheckFailed` ([docs/spec/04-error-model.md#verifier-rejections](../spec/04-error-model.md#verifier-rejections)) |
| Encoder output commitment != planner latent commitment | INV-RFC0011-02 violated; `VerifyError::CommitmentMismatch` |
| `relation_id` is `encode.v1` submitted as `pixel_to_plan.v1` | `VerifyError::RelationMismatch` ([docs/rfcs/RFC-0000-security-model-and-statement-taxonomy.md](RFC-0000-security-model-and-statement-taxonomy.md)) |
| Encoder activation tampered (wrong intermediate token) | Component constraint fails; `VerifyError::ConstraintUnsatisfied` |
| Softmax/GELU/LayerNorm witness out of committed domain | Lookup fails ([docs/rfcs/RFC-0006-nonlinear-primitive-components.md](RFC-0006-nonlinear-primitive-components.md)); `VerifyError::LookupFailed` |
| Patch overlap/stride mismatch vs manifest patch entry | Trace build rejects; `ManifestError` |

## Alternatives Considered

### Alternative A: fold the encoder into the predictor/planner AIR as one monolithic P4 trace, mandatory

What it is: one trace and one proving run containing patch embedding, all ViT
blocks, the predictor rollout, cost, and argmin, with no commitment-equality
join. Why considered: a single proof is the simplest object for a caller and
needs no aggregation layer. Why rejected: the encoder trace is roughly 64-86x
the predictor's per-block token count, so a monolithic trace is encoder-bound;
it prevents independent benchmarking of encoder vs planner, blocks the
per-image aggregation that
[docs/rfcs/RFC-0012-recursive-aggregated-verification.md](RFC-0012-recursive-aggregated-verification.md)
needs, and forces re-proving the entire encoder if only the planner changes.
We keep monolithic as an *allowed* V3 rollout mode (Composition, mode a) but
reject mandating it; the commitment-equality interface (INV-RFC0011-02) is the
locked contract regardless of mode.

### Alternative B: generic convolution component for patch embedding

What it is: a dedicated convolution AIR component with kernel/stride/padding
columns and a sliding-window read pattern. Why considered: ViT patch embedding
is nominally a `Conv2d`, and a general conv component would also serve future
CNN models. Why rejected: with kernel size equal to stride (14 == 14) and no
overlap, every patch is a disjoint dot product, which is exactly the
`matmul`/`linear` component of
[docs/rfcs/RFC-0005-linear-matmul-and-requantization-components.md](RFC-0005-linear-matmul-and-requantization-components.md)
with a patch-batch axis. A general conv component adds overlap-handling,
padding, and stride wiring we do not need for this encoder, enlarging the
audited surface for no benefit. If a future model needs overlapping
convolutions, that is a new component RFC, not a reason to over-build P4.

### Alternative C: distill the encoder to a proof-cheaper architecture (smaller patch count or linear attention) before V3

What it is: train or distill a proof-native encoder (larger patch size, fewer
tokens, or linear attention) and prove that instead of the upstream ViT-tiny.
Why considered: it would cut the dominant `O(tokens^2)` attention cost
substantially and is the analogue of the proof-native-predictor path the
nonlinear RFC keeps open. Why rejected as the *locked* P4 design: P4's purpose
is to close the pixel-to-latent gap for the committed LeWorldModel encoder; a
distilled encoder would prove a *different* model and would not bind the
existing checkpoint's latents. Distillation remains a legitimate separate model
(a new `model_commitment`, a new `relation_id`) and a valid V3+ optimization,
but it is not what "pixel encoder proof of LeWorldModel" means, so it cannot be
the locked scope. This RFC proves the exported ViT-tiny encoder as-is.

## Drawbacks

- P4 is the largest and slowest proof in the corpus; even with commitment-based
  composition, the encoder trace is dominated by 257-token attention/softmax
  and is gated on a performance budget that V3 must meet
  ([docs/spec/08-performance-budget.md#targets](../spec/08-performance-budget.md#targets)).
- Commitment-equality composition adds a verification step and a schema concern
  (the pixel/latent commitment fields) that monolithic proving would avoid.
- Deferral to V3 means the V0/V1/V2 public claim explicitly does not cover
  pixel-to-latent fidelity; callers must understand that latents are inputs,
  not proven encoder outputs, until V3 ships. This is stated in
  [docs/spec/00-overview.md#non-goals](../spec/00-overview.md#non-goals).
- The encoder inherits the full nonlinear-approximation risk of RFC-0006 at a
  much larger token count, so any softmax/LayerNorm approximation error is
  exercised over far more rows; the manifest-bound error bounds must hold over
  the encoder's domain, not just the predictor's.

## Migration / Rollout

- **Milestone gating.** P4 is milestone `Future`, after V0 (P2, milestone
  `v1.0`) and V2 (P3). It does not enter v0.1/v0.2/v1.0 scope. The gating
  dependencies are: a stable P2 statement
  ([docs/rfcs/RFC-0009-fixed-candidate-planner-proof.md](RFC-0009-fixed-candidate-planner-proof.md)),
  the nonlinear primitives at encoder scale (RFC-0006), and the recursion
  interface decision (RFC-0012) if aggregated mode is chosen.
- **Relation versioning.** `pwm.lewm.pixel_to_plan.v1` and `pwm.lewm.encode.v1`
  are minted fresh and are immutable; any semantic change to the encoder
  arithmetization mints `.v2`
  ([docs/spec/09-release-and-versioning.md#relation-versioning](../spec/09-release-and-versioning.md#relation-versioning)).
  Existing P0-P3 relations are unaffected.
- **Schema versioning.** Adding pixel input commitments (and an optional
  `encoder_activations` witness field) is a `PublicInput`/`Witness` schema bump
  performed at V3, guarded by the `artifact_version` field of `ProofArtifact`
  (contract §6.5) and the schema-versioning rules of
  [docs/spec/03-data-model.md#schema-versioning](../spec/03-data-model.md#schema-versioning).
  V0-V2 artifacts keep the contract §6.3 signature unchanged; the canonical
  signature is not modified by this RFC.
- **Feature flag.** The P4 prover/verifier path ships behind a `pixel-encoder`
  Cargo feature in `pwm-air`/`pwm-prover`/`pwm-verifier`, off by default until
  V3, so the encoder components and their dependencies do not bloat the V0-V2
  build or verifier `no_std` surface. The verifier rejects
  `StatementType::P4PixelToPlan` with `VerifyError::UnsupportedRelation` when
  the feature is absent.
- **Export pipeline.** RFC-0001's exporter gains an encoder export path
  (`PatchEmbed`/`ViTEncoder`/`Projector`) and a pixel-quantization manifest
  entry at V3; until then the exporter rejects encoder ops with a clear
  "encoder export is V3" error
  ([docs/spec/04-error-model.md#failure-modes](../spec/04-error-model.md#failure-modes)).

## Testing Strategy

Per the rule that no component ships without both accepting and rejecting tests
([docs/rfcs/RFC-0013-testing-fuzzing-and-audit-strategy.md](RFC-0013-testing-fuzzing-and-audit-strategy.md),
[docs/spec/07-testing-strategy.md#test-pyramid](../spec/07-testing-strategy.md#test-pyramid)).
All tests are V3-gated.

Accepting tests:

- `golden_patch_embed_q`: known image patch through `PatchEmbed_Q` matches the
  Rust and Python fixed-point references bit-for-bit
  ([docs/spec/07-testing-strategy.md#golden-vectors](../spec/07-testing-strategy.md#golden-vectors)).
- `golden_vit_block_q`: one ViT block (LN, attention, GELU MLP, residuals) over
  257 tokens matches the reference; reuses RFC-0006 nonlinear golden vectors.
- `golden_encode_q_single_image`: full `Encode_Q` of one 224x224 image produces
  the reference latent and the expected `claimed_output_commitment`.
- `golden_encode_q_history`: sequence-batched encoding of an H-frame history
  plus goal produces the reference latent sequence.
- `accept_p4_pixel_to_plan`: a valid P4 proof over committed images, candidate
  actions, and the correct argmin result verifies.
- `accept_p4_composition_equality`: encoder output commitments equal the
  planner latent commitments (INV-RFC0011-02), in both monolithic and
  aggregated rollout modes.

Differential tests
([docs/spec/07-testing-strategy.md#differential-tests](../spec/07-testing-strategy.md#differential-tests)):

- `diff_encoder_python_vs_rust`: random seeded images encoded by the Python and
  Rust fixed-point references agree bit-for-bit.
- `diff_p4_vs_composed_p2`: P4 over `Encode_Q(images)` equals P2 over the
  precomputed latents, confirming the encoder/planner join is semantics-
  preserving.

Rejecting (negative) tests
([docs/spec/07-testing-strategy.md#negative-tests](../spec/07-testing-strategy.md#negative-tests),
[docs/spec/04-error-model.md#verifier-rejections](../spec/04-error-model.md#verifier-rejections)):

- `reject_pixel_out_of_range`: a pixel outside the declared `pixel_range` ->
  rejected (INV-RFC0011-01).
- `reject_patch_embed_accumulator_tamper`: mutated patch-embed accumulator with
  an equal field value -> range-check rejects (the field-wrap soundness case,
  source §9.3).
- `reject_encoder_latent_mismatch`: encoder output commitment differs from the
  planner's consumed latent commitment -> `CommitmentMismatch`
  (INV-RFC0011-02).
- `reject_encode_relation_as_p4`: a proof minted for `pwm.lewm.encode.v1`
  submitted as `pwm.lewm.pixel_to_plan.v1` -> `RelationMismatch`.
- `reject_wrong_cls_token`: tampered CLS token or positional embedding (a
  preprocessed constant) -> constraint unsatisfied / commitment mismatch.
- `reject_softmax_out_of_domain`: an attention softmax witness outside the
  committed activation table domain -> `LookupFailed` (RFC-0006).
- `reject_changed_image_same_latent`: a different committed image claiming the
  same latent commitment -> rejected.
- `mutation_patch_embed_constraint`: constraint mutation testing on
  `PatchEmbed_Q` and one ViT block; every surviving mutant is a gap
  ([docs/spec/07-testing-strategy.md#mutation-tests](../spec/07-testing-strategy.md#mutation-tests)).

CI gates: P4 tests run behind the `pixel-encoder` feature and are excluded from
the V0-V2 default CI gate set
([docs/spec/07-testing-strategy.md#ci-gates](../spec/07-testing-strategy.md#ci-gates));
they become a required gate at the V3 milestone.

## Open Questions

- ViT-tiny encoder depth and head count for the V3 checkpoint. Owner: export
  pipeline maintainer (area:export). Resolution path: frozen in the V3 manifest
  at milestone `Future`; bound by `model_commitment` regardless, so soundness is
  independent of the value.
- Whether pixel input commitments become explicit `PublicInput` fields or ride
  the input-commitment list. Owner: data-model maintainer (area:core).
  Resolution path: `PublicInput` schema bump at milestone `Future`, decided with
  [docs/rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md](RFC-0016-cli-artifact-bundle-and-reproducibility.md).
- Monolithic vs aggregated P4 rollout mode. Owner: verifier maintainer
  (area:verifier). Resolution path: decided by the benchmarks in
  [docs/rfcs/RFC-0012-recursive-aggregated-verification.md](RFC-0012-recursive-aggregated-verification.md)
  at milestone `Future`; both modes satisfy INV-RFC0011-02.
- Off-proof image preprocessing (resize/color/normalization) trust boundary.
  Owner: security maintainer (area:security). Resolution path: documented as a
  caller responsibility in
  [docs/spec/06-security.md#trust-boundaries](../spec/06-security.md#trust-boundaries)
  at milestone `Future`; the proof boundary begins at the committed quantized
  `image` tensor.

## References

- Founding analysis: [docs/feasibility-study.md](../feasibility-study.md),
  source RFC-011 (Pixel encoder proof), P4 definition (§1.2), feasibility table
  (§2), risk register "Encoder cost" (§13).
- Overview and scope tiers:
  [docs/spec/00-overview.md#scope-and-statement-tiers](../spec/00-overview.md#scope-and-statement-tiers),
  [docs/spec/00-overview.md#v0-statement](../spec/00-overview.md#v0-statement),
  [docs/spec/00-overview.md#non-goals](../spec/00-overview.md#non-goals).
- Architecture (components, traces):
  [docs/spec/01-architecture.md#component-model](../spec/01-architecture.md#component-model),
  [docs/spec/01-architecture.md#trace-model](../spec/01-architecture.md#trace-model).
- Public API and artifacts:
  [docs/spec/02-public-api.md#artifact-formats](../spec/02-public-api.md#artifact-formats).
- Data model and schema versioning:
  [docs/spec/03-data-model.md#public-input](../spec/03-data-model.md#public-input),
  [docs/spec/03-data-model.md#tensor-types](../spec/03-data-model.md#tensor-types),
  [docs/spec/03-data-model.md#schema-versioning](../spec/03-data-model.md#schema-versioning).
- Errors:
  [docs/spec/04-error-model.md#error-taxonomy](../spec/04-error-model.md#error-taxonomy),
  [docs/spec/04-error-model.md#verifier-rejections](../spec/04-error-model.md#verifier-rejections).
- Security and soundness:
  [docs/spec/06-security.md#soundness-requirements](../spec/06-security.md#soundness-requirements),
  [docs/spec/06-security.md#binding-requirements](../spec/06-security.md#binding-requirements),
  [docs/spec/06-security.md#trust-boundaries](../spec/06-security.md#trust-boundaries).
- Performance:
  [docs/spec/08-performance-budget.md#scaling](../spec/08-performance-budget.md#scaling),
  [docs/spec/08-performance-budget.md#targets](../spec/08-performance-budget.md#targets).
- Release/versioning:
  [docs/spec/09-release-and-versioning.md#relation-versioning](../spec/09-release-and-versioning.md#relation-versioning).
- Testing:
  [docs/spec/07-testing-strategy.md#golden-vectors](../spec/07-testing-strategy.md#golden-vectors),
  [docs/spec/07-testing-strategy.md#negative-tests](../spec/07-testing-strategy.md#negative-tests).
- Related RFCs:
  [RFC-0000](RFC-0000-security-model-and-statement-taxonomy.md) (statement
  taxonomy),
  [RFC-0001](RFC-0001-model-manifest-and-export-pipeline.md) (export),
  [RFC-0003](RFC-0003-range-check-and-lookup-infrastructure.md) (range/lookup),
  [RFC-0004](RFC-0004-tensor-memory-and-wiring-air.md) (tensor memory),
  [RFC-0005](RFC-0005-linear-matmul-and-requantization-components.md) (linear/
  matmul/requant),
  [RFC-0006](RFC-0006-nonlinear-primitive-components.md) (nonlinear primitives),
  [RFC-0007](RFC-0007-leworldmodel-predictor-air.md) (predictor AIR),
  [RFC-0009](RFC-0009-fixed-candidate-planner-proof.md) (fixed-candidate
  planner),
  [RFC-0012](RFC-0012-recursive-aggregated-verification.md) (recursion),
  [RFC-0013](RFC-0013-testing-fuzzing-and-audit-strategy.md) (testing),
  [RFC-0014](RFC-0014-canonical-serialization-and-transcript.md) (serialization/
  transcript),
  [RFC-0016](RFC-0016-cli-artifact-bundle-and-reproducibility.md) (artifact
  bundle).
- Upstream LeWorldModel encoder (`jepa.py` `encode`, ViT-tiny encoder, 224x224,
  patch 14, hidden 192; MIT-licensed), verified against upstream at 2026-06-03.
