# Provable LeWorldModel on Stwo: feasibility, specification, and RFC set

## 0. Executive verdict

This project is **technically feasible only if the proof statement is made exact, deterministic, and quantized**. The first sound version should **not** attempt to prove “PyTorch inference,” “GPU bf16 inference,” “real-valued JEPA inference,” or “the world model is correct about the world.” It should prove a precise arithmetic relation:

> Given a committed quantized LeWorldModel predictor, a latent history, an action sequence or action-candidate set, and a planning rule, the claimed predicted latent trajectory, costs, and selected action are exactly the result of the specified fixed-point inference algorithm.

That is a valid STARK/zk-STARK-style project. The unsafe version is claiming that the proof verifies the original floating-point model without defining every rounding, approximation, normalization, and preprocessing rule.

Stwo is an appropriate base because it exposes a custom-AIR-oriented Circle STARK stack over M31, including field crates, FRI/PCS, Fiat-Shamir channels, constraint-framework components, air utilities, LogUp-style lookups, and a verifier path intended to be small and `no_std`-compatible. ([GitHub][1]) Stwo’s own documentation explicitly frames custom AIR work as suitable for specialized domains including ML inference. ([Starknet Documentation][2]) The `stwo-circuits` repository provides a useful reference for low-level circuit-style gates and proof plumbing, but it should be treated as an implementation substrate to audit and pin, not as a mature high-level NN proving framework. Its workspace currently contains circuit prover/verifier/common/cairo verifier crates, and its circuit layer exposes low-level gates such as add, sub, mul, pointwise mul, equality, bit extraction, Blake, permutation, and output gates. ([GitHub][3])

LeWorldModel is a small JEPA-style world model, but “small” is relative. The repository describes a roughly 15M-parameter model trained from pixels with a next-embedding prediction loss and Gaussian latent regularizer. ([GitHub][4]) The implementation contains an encoder, predictor, action encoder, projector, and prediction projector; planning/evaluation uses latent rollouts and a final latent MSE-to-goal cost. ([GitHub][5]) The predictor is not a trivial MLP: it is an autoregressive transformer-like predictor with attention, conditional AdaLN-style blocks, MLPs, positional embeddings, and action conditioning. ([GitHub][6])

The recommended first deliverable is therefore:

> **V0:** prove the quantized latent predictor rollout and candidate-cost/argmin planning over a fixed candidate set.

Then, only after that is stable:

> **V1:** prove batched candidate planning efficiently.
> **V2:** prove full CEM search, if required.
> **V3:** prove pixel encoder inference end-to-end.

The default LeWorldModel eval config uses horizon/action-block planning and CEM-style sampling parameters; proving the entire sampling/search loop is materially harder than proving fixed candidate scoring. ([GitHub][7])

stwo and stwo circuits should be used directly with a copy of latest workable version, in a third_party folder in our repo.
We don't want to depend on the upstream canonical repos. we copy them and modify as we need for this POC.

---

## 1. What exactly is being proven

### 1.1 Non-negotiable distinction

The proof proves **model computation**, not physical truth.

It can prove:

```text
z_pred = QuantizedPredictor(z_history, actions; committed_weights)
cost_i = MSE_Q(z_pred_i_final, z_goal)
i_star = deterministic_argmin(cost_0, ..., cost_{S-1})
```

It does **not** prove:

```text
the predicted future is true
the model is calibrated
the chosen action is globally optimal in the real environment
the floating-point PyTorch model would produce the same bits
the training process was valid
```

Those can be separate claims, but they are not part of the cryptographic soundness of the inference proof.

### 1.2 Proof statement tiers

Define four proof statements.

#### Statement P0: latent transition proof

Prove one predictor step:

```text
Public:
  model_commitment
  quantization_commitment
  relation_id
  z_history
  action_history
  claimed_z_next

Private:
  weights, if not public
  intermediate activations
  auxiliary range/lookup witnesses

Claim:
  claimed_z_next == Predictor_Q(z_history, action_history; weights)
```

This is the smallest meaningful proof.

#### Statement P1: latent rollout proof

Prove autoregressive rollout:

```text
Public:
  model_commitment
  relation_id
  initial_latent_history
  action_sequence
  claimed_latent_trajectory

Claim:
  for t = 0..T-1:
      z_{t+1} == Predictor_Q(window(z), window(actions); weights)
```

This proves “prediction of the predictor.”

#### Statement P2: fixed-candidate planning proof

Prove a batch of candidate rollouts and deterministic selection:

```text
Public:
  model_commitment
  initial_latent_history
  goal_latent
  candidate_action_sequences[0..S-1]
  selected_index
  selected_cost
  optional claimed trajectory root

Claim:
  for every candidate s:
      trajectory_s == Rollout_Q(initial_latents, candidate_actions_s)
      cost_s == MSE_Q(final_latent_s, goal_latent)

  selected_cost == cost_selected_index

  for all s:
      selected_cost <= cost_s

  tie break:
      selected_index is the smallest index attaining the minimum cost
```

This is the correct V0 planning proof.

#### Statement P3: full CEM planning proof

Prove the entire planner:

```text
Public:
  model_commitment
  initial_latent_history
  goal_latent
  planner_config
  random_seed
  selected_action_sequence

Claim:
  candidates were generated from seed according to CEM_Q
  every candidate was scored by Rollout_Q
  top-k selection was correct at every CEM iteration
  mean/variance/action distribution updates were correct
  final selected sequence matches the deterministic planner output
```

This is much larger. It should not be the first milestone.

#### Statement P4: end-to-end pixel-to-plan proof

Prove encoder + predictor + planner:

```text
Public:
  model_commitment
  pixel_observation_history
  pixel_goal
  selected_action_sequence

Claim:
  z_history == Encoder_Q(pixel_observation_history)
  z_goal == Encoder_Q(pixel_goal)
  selected_action_sequence == Planner_Q(z_history, z_goal)
```

This is the final form, not the starting point.

---

## 2. Feasibility assessment

| Scope                                  |              Feasibility | Soundness status                                                                     | Recommendation                     |
| -------------------------------------- | -----------------------: | ------------------------------------------------------------------------------------ | ---------------------------------- |
| One quantized linear/MLP layer         |                     High | Straightforward with range checks and exact fixed-point semantics                    | Build first                        |
| One quantized predictor step           |                   Medium | Feasible, but attention, LayerNorm, GELU, and action conditioning must be formalized | V0 core target                     |
| Latent rollout for one action sequence |                   Medium | Feasible if recurrence and rounding are fixed                                        | V0                                 |
| Fixed-candidate planning with argmin   |                   Medium | Feasible; scales linearly with candidates and horizon                                | V0/V1                              |
| Full CEM proof                         |            Low-to-medium | Possible but heavy: must prove sampling, top-k, updates, all candidate scoring       | V2 only                            |
| Pixel encoder proof                    |            Low-to-medium | Possible but expensive: ViT patch/token attention from 224×224 pixels dominates      | V3                                 |
| Floating-point PyTorch equivalence     |                      Low | Not sound unless every kernel and rounding rule is formally specified                | Do not claim                       |
| Zero-knowledge weight privacy          | Unresolved until audited | Validity proof is easier than private proof                                          | Treat ZK as optional security mode |

The biggest practical problem is not STARK soundness; it is **arithmetizing a transformer-like neural network economically**. The LeWorldModel predictor config uses `embed_dim = 192`, predictor depth `6`, `heads = 16`, `dim_head = 64`, and MLP hidden dimension `2048`. ([GitHub][8]) A single predictor call therefore contains millions of multiply-accumulate operations before considering rollout horizon, candidate count, or CEM iterations. Proving all CEM candidates across all CEM iterations naively would be far larger than proving one rollout.

The clean path is:

1. prove one fixed-point predictor step;
2. prove one rollout;
3. batch many rollouts;
4. prove candidate selection;
5. only then consider proving the CEM optimizer itself.

---

## 3. System architecture

### 3.1 Crate layout

Recommended repository layout:

```text
provable-world-model/
  crates/
    pwm-core/
      fixed_point.rs
      tensor.rs
      manifest.rs
      transcript.rs
      relation_id.rs

    pwm-export/
      torch_export/
      onnx_import/
      quantize.rs
      manifest_writer.rs
      parity_tests.rs

    pwm-air/
      components/
        range_check.rs
        tensor_memory.rs
        linear.rs
        matmul.rs
        requant.rs
        activation_lookup.rs
        layernorm.rs
        attention.rs
        mlp.rs
        predictor.rs
        rollout.rs
        cost.rs
        argmin.rs
        cem.rs
      proof.rs
      verifier.rs

    pwm-circuits/
      low_level_gates.rs
      adapters_stwo_circuits.rs
      witness_builder.rs

    pwm-prover/
      cli.rs
      trace_builder.rs
      prove_rollout.rs
      prove_planning.rs

    pwm-verifier/
      lib.rs
      recursive.rs
      public_input.rs

    specs/
      main-spec.md
      rfc-000-security-model.md
      rfc-001-model-manifest.md
      ...
```

### 3.2 Direct AIR versus circuit-style lowering

There are two plausible implementation strategies.

#### Strategy A: direct custom AIR components

Use Stwo’s constraint framework directly and implement NN-specific AIR components:

```text
LinearComponent
AttentionComponent
LayerNormComponent
ActivationLookupComponent
RolloutComponent
ArgminComponent
```

This is the preferred production route. Dense NN inference is dominated by regular tensor operations, so a high-level operator AIR will be more efficient and easier to audit than expressing every operation as a generic scalar gate.

#### Strategy B: low-level circuit lowering via stwo-circuits-style gates

Use a circuit IR inspired by `stwo-circuits`:

```text
add
sub
mul
pointwise_mul
eq
range/extract_bits
permutation
output
hash
```

This is useful for small glue logic, hashing, transcript logic, serialization, and prototyping. But compiling every dense matmul to generic scalar gates is likely too expensive. The `stwo-circuits` codebase is valuable as a reference because it already exposes circuit-level gates and prover/verifier crates, but it is not a complete NN proving system out of the box. ([GitHub][3])

Recommended split:

```text
Use direct AIR for:
  dense linear layers
  batched matmul
  attention
  rollout recurrence
  cost and argmin

Use low-level circuits for:
  hashing
  public input binding
  small arithmetic gadgets
  compatibility experiments
  recursive verifier plumbing
```

---

## 4. Model formalization

### 4.1 LeWorldModel components to support

The LeWorldModel class contains:

```text
encoder
predictor
action_encoder
projector
pred_proj
```

Its `encode` path converts pixel observations into latent embeddings and action embeddings; its `predict` path applies the predictor and prediction projection; its rollout path autoregressively applies the predictor over candidate actions; its cost path compares final predicted latent to a goal latent with MSE. ([GitHub][5])

The V0 proof should support:

```text
action_encoder
predictor
pred_proj
rollout recurrence
goal-latent MSE
argmin over fixed candidates
```

The V0 proof should not require:

```text
pixel encoder
training loss
SIGReg regularization
PyTorch dataloading
Hydra config behavior
CEM sampling/update loop
```

### 4.2 Model manifest

Every proof must be bound to a canonical model manifest.

Example:

```yaml
manifest_version: pwm-model-manifest-v1
model_family: lewm
relation_id: pwm.lewm.predictor_rollout.v1

field:
  base: M31
  extension: QM31
  signed_encoding: centered_mod_p
  max_abs_value: specified_per_tensor

quantization:
  arithmetic: fixed_point
  default_rounding: nearest_ties_to_even
  overflow_policy: reject
  clamp_policy: explicit
  activation_tables_commitment: 0x...

architecture:
  latent_dim: 192
  history_size: 3
  predictor:
    type: ar_transformer
    depth: 6
    heads: 16
    dim_head: 64
    mlp_dim: 2048
  action_encoder:
    type: embedder
  pred_proj:
    type: mlp_or_affine

weights:
  visibility: public | private_committed
  commitment_scheme: blake3 | poseidon | merkle
  root: 0x...

ops:
  - id: action_encoder.layer0
    op: linear
    input_scale: ...
    weight_scale: ...
    output_scale: ...
    weight_commitment: ...

  - id: predictor.block0.attn.qkv
    op: linear
    ...

  - id: predictor.block0.attn.softmax
    op: softmax_approx_v1
    lookup_table: softmax_table_v1
    error_bound: ...

  - id: predictor.block0.mlp.gelu
    op: gelu_lookup_v1
    lookup_table: gelu_table_v1

serialization:
  canonical_json_hash: 0x...
```

The model commitment must bind:

```text
architecture
weights
biases
quantization scales
rounding rules
lookup tables
activation approximations
normalization approximations
tensor shapes
planner config
relation version
serialization version
```

A proof without this binding is not sound, because the prover could silently change the model, quantization, activation approximation, or planner rule.

---

## 5. Arithmetic semantics

### 5.1 Base field

Stwo uses M31-family arithmetic, with extension fields such as CM31/QM31 used in the proof system. ([GitHub][1]) The NN arithmetic should be specified primarily over bounded signed integers embedded into M31.

Let:

```text
p = 2^31 - 1
```

Represent a signed integer `x` as:

```text
field(x) = x mod p
```

but enforce a range proof:

```text
-B <= x <= B
```

where `B` is small enough that all additions, products, differences, and comparisons are unambiguous.

### 5.2 Recommended quantization

Use integer fixed-point inference.

Recommended V0:

```text
weights:      int8
activations:  int8 or int16, depending on accuracy
accumulators: bounded int32 represented in M31 when safe
biases:       int32 or split-limb
scales:       powers of two where possible
rounding:     deterministic, specified globally
overflow:     reject, not wrap
```

For many LeWorldModel dimensions, int8 multiply-accumulate fits comfortably in M31. For example, a worst-case signed int8 dot product of length `2048` has magnitude:

```text
2048 * 127 * 127 = 33,032,192 < 2^31 - 1
```

So MLP dot products of length 2048 can fit in one signed M31 value if all inputs and weights are truly int8 and biases are bounded. If activations, weights, or biases use wider ranges, the AIR must switch to limb accumulators.

### 5.3 Linear layer semantics

For a linear layer:

```text
y_j = Requantize( bias_j + Σ_i x_i * w_{i,j} )
```

AIR constraints:

```text
acc_{j,0} = bias_j

for i = 0..N-1:
    prod_{j,i} = x_i * w_{i,j}
    acc_{j,i+1} = acc_{j,i} + prod_{j,i}

y_j = requantize(acc_{j,N}, shift, zero_point, clamp_range)
```

Required range checks:

```text
x_i       in activation range
w_i,j     in weight range
prod_i,j  in product range, if represented explicitly
acc_j,i   in accumulator range
y_j       in output activation range
```

If `acc` may exceed the safe signed M31 interval, use limb decomposition:

```text
acc = acc_lo + 2^k * acc_hi
```

with range checks on each limb.

### 5.4 Requantization

Do not use informal division.

For right shift by `r`:

```text
n = q * 2^r + rem
0 <= rem < 2^r
```

Then define rounding exactly.

For nearest-ties-to-even:

```text
half = 2^(r-1)

if rem < half:
    rounded = q
if rem > half:
    rounded = q + sign(n)
if rem == half:
    rounded = q adjusted to even
```

This is annoying but necessary. A simpler V0 can use truncation toward zero, but then the exported model must use exactly the same truncation.

### 5.5 Comparisons

Comparisons are not native finite-field operations. For:

```text
a <= b
```

prove:

```text
d = b - a
0 <= d <= D_max
```

using range checks. For argmin:

```text
cost_s - selected_cost >= 0
```

must be range-checked for every candidate.

### 5.6 MSE cost

For goal latent `g` and predicted final latent `z`:

```text
cost = Σ_j (z_j - g_j)^2
```

Constraints:

```text
diff_j = z_j - g_j
sq_j = diff_j * diff_j
cost_{j+1} = cost_j + sq_j
```

Range-check:

```text
diff_j
sq_j
cost_j
```

If latent values are int8 and latent dimension is 192, the worst-case cost is manageable. If latent values are int16, the cost may require limbs.

---

## 6. Nonlinear operations

This is the highest-risk part of the project.

The LeWorldModel predictor contains attention and conditional transformer blocks. The code uses scaled dot-product attention and conditional AdaLN-style modulation. ([GitHub][6]) These are expensive or awkward in finite-field arithmetic if implemented faithfully.

### 6.1 GELU / activation functions

Use lookup-table or piecewise-polynomial approximation.

The manifest must specify:

```text
input domain
output domain
scale
rounding
table commitment
maximum approximation error, if any
```

AIR:

```text
(x, y) ∈ ActivationTable
```

using LogUp-style lookup arguments. Stwo’s docs describe lookup usage for range checks and dynamic relationships between tables, and LogUp interaction traces are part of the supported design pattern. ([ZK Security][9])

### 6.2 LayerNorm / AdaLN

LayerNorm requires:

```text
mean = Σ x_i / d
variance = Σ (x_i - mean)^2 / d
inv_std ≈ 1 / sqrt(variance + eps)
y_i = gamma_i * (x_i - mean) * inv_std + beta_i
```

Options:

| Option                                                            | Soundness            |        Cost | Recommendation         |
| ----------------------------------------------------------------- | -------------------- | ----------: | ---------------------- |
| Exact rational fixed-point LayerNorm with reciprocal-sqrt lookup  | Strong               |        High | V1/V2                  |
| Newton iteration for inverse sqrt with bounded domain             | Strong if bounded    | Medium-high | Possible               |
| Replace LayerNorm with RMSNorm or affine norm and retrain/distill | Strong for new model |       Lower | Best proof-native path |
| Ignore LayerNorm or approximate off-circuit                       | Unsound              |         Low | Reject                 |

For V0, either implement a bounded lookup-based LayerNorm or create a proof-native distilled predictor that replaces LayerNorm with a cheaper normalization. If the latter is chosen, the proof is for the distilled model, not the original LeWorldModel checkpoint.

### 6.3 Attention softmax

Softmax is the most problematic primitive.

Scaled dot-product attention computes:

```text
scores = QK^T / sqrt(d)
probs = softmax(scores + mask)
out = probs V
```

Options:

| Option                                    | Description                                                            | Soundness status                   |
| ----------------------------------------- | ---------------------------------------------------------------------- | ---------------------------------- |
| Softmax lookup approximation              | Quantize scores, look up approximate exp, prove denominator reciprocal | Sound for quantized approximation  |
| Polynomial/rational softmax approximation | Fixed polynomial over bounded domain                                   | Sound if manifest binds polynomial |
| Linear attention replacement              | Retrain/distill predictor with proof-native attention                  | Sound for new model                |
| Off-circuit softmax                       | Prover supplies probabilities                                          | Unsound unless fully constrained   |

Recommended path:

```text
V0a: replace or distill softmax into proof-native attention
V0b: implement lookup softmax if exact LeWorldModel architecture must be preserved
```

If the project must prove inference of the existing checkpoint, use lookup/rational approximations and call the result:

```text
QuantizedLeWM-v1
```

not “the exact PyTorch model.”

---

## 7. AIR design

### 7.1 Component model

Use multiple Stwo components. Stwo’s custom AIR examples and docs describe proving and verifying multiple components, fixed preprocessed columns, trace columns, Fiat-Shamir channel ordering, and lookup arguments. ([ZK Security][10])

Recommended components:

```text
RangeCheckComponent
TensorMemoryComponent
WeightTableComponent
LinearComponent
MatMulComponent
RequantizeComponent
ActivationLookupComponent
LayerNormComponent
AttentionComponent
MLPComponent
ActionEncoderComponent
PredictorBlockComponent
RolloutComponent
CostComponent
ArgminComponent
CEMComponent, optional
```

### 7.2 Preprocessed trace

Preprocessed columns should include fixed data known before proving:

```text
row selectors
operation selectors
tensor shape metadata
static masks
causal attention masks
positional embeddings, if public/fixed
quantization scales
rounding constants
lookup-table columns
range-check table columns
```

Stwo’s book describes preprocessed traces as fixed columns agreed by prover and verifier, suitable for selectors and constants. ([ZK Security][11])

### 7.3 Main trace

Main trace columns contain witness data:

```text
tensor values
weights, if private
intermediate activations
accumulators
products
quotients/remainders
normalization intermediates
attention scores/probabilities
costs
argmin comparison witnesses
```

### 7.4 Interaction trace

Use interaction trace / LogUp relations for:

```text
range checks
activation lookup tables
weight table lookups
tensor read/write consistency
permutation checks
memory consistency
candidate-cost membership
top-k membership, if CEM is proven
```

Dynamic lookup/permutation arguments are appropriate when the values are not known before proving. Stwo’s docs distinguish static lookups such as range checks from dynamic lookups/permutation-style arguments. ([ZK Security][9])

### 7.5 Tensor memory model

Define a canonical tensor-address relation:

```text
TensorCell:
  tensor_id
  index_0
  index_1
  index_2
  value
  scale_id
  time
```

For every operator:

```text
reads  -> lookup from previous TensorCell writes
writes -> inserted into TensorCell table
```

Use a permutation/multiset equality argument to ensure that every read corresponds to a unique valid write, unless broadcasting is explicitly allowed.

For broadcast constants:

```text
constant_id
index pattern
value
```

must be committed in the manifest or preprocessed trace.

### 7.6 Linear component AIR sketch

Columns:

```text
op_id
batch_id
out_idx
in_idx
x
w
prod
acc
acc_next
is_first
is_last
bias
shift
q
rem
y
```

Constraints:

```text
is_first * (acc - bias) = 0

prod = x * w

acc_next = acc + prod

is_last * (acc_next - (q * 2^shift + rem)) = 0

0 <= rem < 2^shift

y = RoundAndClamp(q, rem, shift, policy)
```

Lookups:

```text
x read from tensor memory
w read from weight table
y written to tensor memory
x, w, acc, y range-checked
```

### 7.7 Attention component AIR sketch

For each layer and head:

```text
Q = Linear_Q(x)
K = Linear_Q(x)
V = Linear_Q(x)

score_{i,j} = Σ_k Q_{i,k} K_{j,k}
score_scaled = Requantize(score_{i,j}, scale)

if causal_mask(i,j) == 0:
    score_masked = NEG_INF_Q
else:
    score_masked = score_scaled

prob_{i,j} = SoftmaxApprox_Q(score_masked over j)

out_{i,k} = Σ_j prob_{i,j} V_{j,k}
```

Constraints required:

```text
Q/K/V linear constraints
score dot-product constraints
mask selector constraints
softmax approximation constraints
probability normalization constraints, if applicable
out dot-product constraints
requantization constraints
```

For V0, if the predictor input sequence length is only the history window, attention sequence length is small, but hidden dimension and projection size are still significant.

### 7.8 Rollout component AIR

The LeWorldModel rollout autoregressively appends predicted embeddings and actions and then uses the final predicted embedding for cost. ([GitHub][5])

Formal recurrence:

```text
Given:
  H = history_size
  z[0..H-1]
  a[0..T-1]

For t = H-1 .. T-1:
  ctx_z = z[t-H+1 .. t]
  ctx_a = a[t-H+1 .. t]
  z[t+1] = Predictor_Q(ctx_z, ActionEncoder_Q(ctx_a))
```

Constraints:

```text
window selection is correct
action embeddings match action_encoder output
predictor input equals selected latent/action window
predictor output equals next latent
next latent is appended to future windows
```

### 7.9 Cost and argmin AIR

For each candidate `s`:

```text
cost_s = MSE_Q(z_final_s, z_goal)
```

For selected index `i*`:

```text
selected_cost = cost_{i*}
for all s:
    cost_s - selected_cost >= 0
```

Tie-breaking:

```text
for all s < i*:
    cost_s - selected_cost > 0
```

Because strict greater-than is awkward, use:

```text
cost_s - selected_cost - 1 >= 0
```

with range checks for all `s < i*`.

---

## 8. Zero-knowledge and privacy

Do not call the V0 proof “zero-knowledge” unless hiding has been explicitly audited.

A STARK proof of validity is not automatically a privacy-preserving proof for model weights or private observations. To support private weights or private inputs, the system must guarantee that:

```text
weights are not public inputs
only a binding commitment to weights is public
trace commitments are hiding or appropriately masked
lookup arguments do not leak private table contents
claimed sums and public outputs do not leak unintended values
serialization does not expose witness data
```

Stwo Cairo documentation discusses a production proof stack and a conjectured soundness target for Cairo execution proofs, but privacy for this specific custom NN relation must be established separately. ([GitHub][12])

Recommended terminology:

```text
V0: succinct validity proof
V1: optionally zero-knowledge validity proof after hiding audit
```

---

## 9. Soundness requirements

### 9.1 Binding requirements

The proof must bind to:

```text
relation_id
model architecture
model weights
quantization scales
rounding modes
lookup tables
planner configuration
input tensors or input commitments
output tensors
Stwo/security parameters
serialization version
```

Anything not bound is mutable by the prover.

### 9.2 Determinism requirements

The following must be deterministic:

```text
model export
quantization
rounding
clamping
attention approximation
LayerNorm approximation
activation approximation
candidate generation, if CEM is proven
argmin tie-breaking
public input serialization
```

### 9.3 Range-safety requirements

Every value interpreted as an integer must be range-checked.

Required range checks:

```text
input latents
actions
weights
biases
products
accumulators
requantization quotients
requantization remainders
activation outputs
attention scores
softmax probabilities
LayerNorm intermediates
cost differences
argmin comparison witnesses
```

Finite fields wrap. Integer ML inference must not.

### 9.4 Approximation requirements

For each approximate primitive:

```text
GELU
softmax
inverse sqrt
LayerNorm
division
reciprocal
```

the manifest must include:

```text
approximation identifier
domain
scale
table or polynomial commitment
rounding mode
maximum error
test vectors
```

The proof enforces the approximation. It does not prove the approximation is close to the original unless an explicit error theorem or exhaustive table check is included.

### 9.5 Planner soundness

For fixed-candidate planning, proving only the selected candidate is insufficient. The prover must prove costs for all candidates or prove a committed candidate-cost table plus a correct minimum argument.

For CEM planning, proving only the final candidate is insufficient. The proof must include:

```text
seeded sample generation
candidate clipping
all costs
top-k selection
mean update
variance/std update
iteration recurrence
final selection
```

Otherwise, the prover can choose favorable candidates off-circuit.

---

## 10. Main technical specification

### 10.1 Public input schema

```rust
pub struct PublicInput {
    pub relation_id: [u8; 32],
    pub model_commitment: [u8; 32],
    pub quantization_commitment: [u8; 32],
    pub planner_config_commitment: [u8; 32],

    pub statement_type: StatementType,

    pub latent_history_commitment: Option<[u8; 32]>,
    pub latent_history_public: Option<Vec<FieldElement>>,

    pub goal_latent_commitment: Option<[u8; 32]>,
    pub goal_latent_public: Option<Vec<FieldElement>>,

    pub candidate_actions_commitment: Option<[u8; 32]>,
    pub candidate_actions_public: Option<Vec<FieldElement>>,

    pub claimed_output_commitment: [u8; 32],
    pub selected_index: Option<u32>,
    pub selected_cost: Option<FieldElement>,
}
```

### 10.2 Witness schema

```rust
pub struct Witness {
    pub model_weights: Option<QuantizedWeights>,
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
```

### 10.3 Prover flow

```text
1. Load model manifest.
2. Verify manifest hash equals model_commitment.
3. Load quantized weights.
4. Validate weight hash.
5. Canonicalize public inputs.
6. Execute fixed-point reference inference.
7. Build operator traces.
8. Build range-check and lookup traces.
9. Commit traces to Stwo channel.
10. Derive challenges in canonical order.
11. Build interaction traces.
12. Generate proof.
13. Emit proof + public input + optional output tensors.
```

Stwo’s proving flow requires committing data and deriving/verifying challenges in the same Fiat-Shamir channel order on both prover and verifier sides. ([ZK Security][10])

### 10.4 Verifier flow

```text
1. Parse public input.
2. Check relation_id is supported.
3. Recompute public input digest.
4. Recompute preprocessed trace commitments from manifest/config, or verify their commitments.
5. Verify Stwo proof.
6. Decode claimed outputs.
7. Enforce application-level checks:
     selected index in range
     tensor shapes match
     commitments match canonical serialization
```

Verifier must not run PyTorch. It verifies the arithmetic relation only.

---

## 11. RFC set

## RFC-000: Security model and statement taxonomy

**Status:** Required before coding.

### Motivation

The project needs a precise statement hierarchy so “proved world model” does not become ambiguous.

### Specification

Define supported statements:

```text
P0: one latent predictor step
P1: latent rollout
P2: fixed-candidate planner
P3: full CEM planner
P4: pixel-to-plan end-to-end proof
```

Each statement must define:

```text
public inputs
private witnesses
model commitment
output commitment
accepted tensor shapes
security level
privacy mode
```

### Soundness requirements

A proof is valid only for the declared `relation_id`. Different relation versions must not share the same ID.

### Acceptance criteria

```text
A verifier rejects a proof generated for P0 when submitted as P1.
A verifier rejects if model_commitment changes.
A verifier rejects if quantization_commitment changes.
A verifier rejects if planner_config changes.
```

---

## RFC-001: Model manifest and export pipeline

**Status:** Required for V0.

### Motivation

The prover and verifier need a canonical artifact that defines the exact model being proven.

### Specification

Build exporter:

```text
PyTorch checkpoint
  -> canonical graph
  -> quantized graph
  -> manifest
  -> weight tensors
  -> golden test vectors
```

Manifest must bind:

```text
architecture
weights
biases
tensor shapes
operator order
quantization scales
rounding modes
lookup tables
planner config
```

### LeWorldModel-specific requirements

The exporter must support:

```text
action_encoder
ARPredictor
ConditionalBlock
Attention
FeedForward/MLP
pred_proj
```

The codebase’s JEPA class and rollout logic should be used to derive the operator graph, but the proof system must not depend on dynamic PyTorch execution. ([GitHub][5])

### Acceptance criteria

```text
Export same checkpoint twice -> byte-identical manifest.
Changing one weight changes model_commitment.
Changing one quantization scale changes quantization_commitment.
Rust fixed-point inference matches Python fixed-point reference bit-for-bit.
```

---

## RFC-002: Fixed-point arithmetic over M31

**Status:** Required for V0.

### Motivation

Soundness depends on preventing silent field wraparound and ambiguous integer interpretation.

### Specification

Define:

```text
Signed integer encoding
Per-tensor bounds
Add/sub semantics
Mul semantics
Accumulator semantics
Requantization semantics
Clamp semantics
Comparison semantics
```

### Required constraints

```text
Every integer column has a declared range.
Every division/requantization has quotient/remainder constraints.
Every comparison has a nonnegative difference witness.
Every accumulator is proven not to overflow its declared range.
```

### Acceptance criteria

```text
Negative test: mutate accumulator -> verifier rejects.
Negative test: overflow accumulator but same field value -> verifier rejects.
Negative test: invalid rounding remainder -> verifier rejects.
Negative test: comparison wraparound -> verifier rejects.
```

---

## RFC-003: Range-check and lookup infrastructure

**Status:** Required for V0.

### Motivation

Range checks, activations, normalization approximations, and memory consistency all need lookup-style arguments.

### Specification

Implement reusable lookup relations:

```text
u8 range
i8 range
u16 range
bounded signed limb
activation table
requantization table, optional
softmax table, optional
inverse-sqrt table, optional
```

Stwo’s docs describe static lookups for range checks and LogUp-style interaction traces for lookup relations. ([ZK Security][9])

### Acceptance criteria

```text
All bounded integer columns are wired to a range relation.
Lookup multiplicities are checked.
Verifier rejects if a witness uses a value outside the table.
Verifier rejects if interaction claimed sum is inconsistent.
```

---

## RFC-004: Tensor memory and wiring AIR

**Status:** Required for nontrivial graphs.

### Motivation

NN graphs have many tensors and reuse values across operators. The proof needs a sound wiring model.

### Specification

Define canonical tensor cells:

```text
tensor_id
indices
value
scale_id
producer_op
time
```

Define read/write rules:

```text
each read must match a prior write or static constant
each write has one producer
broadcasting must be explicit
reshapes must preserve multiset of values
transposes must preserve index mapping
concats/slices must prove index mapping
```

### Acceptance criteria

```text
Mutating one read value rejects.
Swapping two tensor indices rejects unless operation is transpose.
Using a value from the wrong scale_id rejects.
Broadcasting without manifest permission rejects.
```

---

## RFC-005: Linear, matmul, and convolution components

**Status:** Required for V0.

### Motivation

Dense linear algebra dominates proving cost.

### Specification

Implement specialized AIR components for:

```text
linear
batched linear
matmul
batched matmul
patch embedding / convolution, for V3
bias add
requantization
```

The component must support batching over:

```text
candidate index
rollout step
sequence position
attention head
output channel
```

### Optimization requirements

```text
Use tensorized trace layout.
Avoid generic scalar gates for dense matmul hot paths.
Exploit static weights when public.
Support private committed weights if needed.
Support chunked accumulators for large dot products.
```

### Acceptance criteria

```text
Linear component proves random test vectors.
Batched linear equals repeated scalar linear.
Accumulator bounds are machine-checked from tensor metadata.
Verifier rejects changed weight, input, bias, scale, or output.
```

---

## RFC-006: Nonlinear primitive components

**Status:** Required for faithful predictor proof.

### Motivation

Attention, LayerNorm, GELU, and AdaLN are the main semantic risk.

### Specification

Define proof-native versions of:

```text
GELU_Q
Softmax_Q
LayerNorm_Q
AdaLN_Q
Dropout_Q = identity in eval mode
BatchNorm_Q = folded affine in eval mode
```

LeWorldModel’s config uses dropout in the model definition, but inference should be in eval mode, where dropout is disabled. The repository’s checkpoint-loading instructions also set the module to evaluation mode. ([GitHub][4])

### Allowed implementations

```text
lookup-table activation
piecewise-polynomial activation
fixed rational approximation
Newton reciprocal/sqrt with bounded iterations
folded affine normalization where mathematically valid
```

### Rejected implementations

```text
off-circuit softmax probabilities
unconstrained reciprocal
unconstrained inverse sqrt
floating-point helper values
implicit PyTorch semantics
```

### Acceptance criteria

```text
Every nonlinear primitive has golden vectors.
Every primitive has explicit domain bounds.
Verifier rejects out-of-domain witness values.
Approximation tables are committed in the manifest.
```

---

## RFC-007: LeWorldModel predictor AIR

**Status:** V0 core.

### Motivation

This is the first real project milestone: prove predictor inference in latent space.

### Specification

Implement:

```text
ActionEncoder_Q
ARPredictor_Q
PredProj_Q
Predict_Q
```

The predictor implementation must match the exported quantized graph, not the live Python module.

### Required relation

```text
z_next = PredProj_Q(ARPredictor_Q(z_history, ActionEncoder_Q(actions)))
```

### Required public inputs

```text
model_commitment
z_history or z_history_commitment
actions or actions_commitment
claimed_z_next
```

### Acceptance criteria

```text
Prove one predictor step.
Prove batch of predictor steps.
Prove predictor output matches Rust reference.
Verifier rejects changed action.
Verifier rejects changed latent history.
Verifier rejects changed claimed output.
```

---

## RFC-008: Rollout AIR

**Status:** V0 core.

### Motivation

Planning needs autoregressive prediction, not just one transition.

### Specification

Implement:

```text
Rollout_Q(z_initial_history, action_sequence):
    for each step:
        z_next = Predict_Q(current_window, current_action_window)
        append z_next
    return trajectory
```

The LeWorldModel code rolls out candidate action sequences autoregressively over latent embeddings and action embeddings before computing final cost. ([GitHub][5])

### Acceptance criteria

```text
Prove one rollout.
Prove batched rollouts.
Verifier rejects if any intermediate predicted latent is changed.
Verifier rejects if final trajectory is inconsistent with intermediate recurrence.
```

---

## RFC-009: Fixed-candidate planner proof

**Status:** V0/V1.

### Motivation

This is the cleanest defensible form of “provable planning inference.”

### Specification

Given `S` candidate action sequences:

```text
for s in 0..S-1:
    traj_s = Rollout_Q(z_history, actions_s)
    cost_s = MSE_Q(final(traj_s), z_goal)

selected = argmin(cost_s), with deterministic tie-break
```

### Required constraints

```text
all candidate rollouts proven
all candidate costs proven
selected cost linked to selected index
selected cost <= every candidate cost
tie-break enforced
```

### Acceptance criteria

```text
Verifier rejects if a lower-cost candidate exists.
Verifier rejects if selected_index is out of range.
Verifier rejects if selected_cost does not match cost[selected_index].
Verifier rejects if tie-break is violated.
```

---

## RFC-010: CEM planner proof

**Status:** V2, optional.

### Motivation

If the project claims to prove the actual CEM planner, candidate generation and distribution updates must be proven.

### Specification

Define deterministic CEM:

```text
seed -> PRNG stream
sample candidates
clip/project candidates to action domain
score all candidates
select top-k
update mean
update variance/std
repeat n_steps
return best action sequence
```

The default eval solver config in the LeWorldModel repository uses CEM-style parameters such as sample count, number of iterations, and top-k. ([GitHub][13])

### Required constraints

```text
PRNG output correctness
sample transform correctness
candidate bounds
all costs
top-k correctness
mean update
variance update
iteration recurrence
final selection
```

### Major risk

Normal/Gaussian sampling and square roots are expensive. A proof-native planner may need discrete sampling, fixed uniform noise, or a committed candidate set instead.

### Acceptance criteria

```text
Given same seed/config, Python fixed-point planner and Rust planner match bit-for-bit.
Verifier rejects if any CEM iteration is skipped.
Verifier rejects if top-k set is wrong.
Verifier rejects if final action was not produced by the deterministic CEM recurrence.
```

---

## RFC-011: Pixel encoder proof

**Status:** V3.

### Motivation

A latent-only proof assumes the initial and goal latents are valid inputs. To prove pixel-to-plan, the encoder must be included.

### Specification

Implement:

```text
PatchEmbed_Q
ViTEncoder_Q
Projector_Q
Encode_Q
```

The LeWorldModel `encode` method flattens time, applies the encoder, extracts the CLS token, projects it, and reshapes back into latent sequence form. ([GitHub][5])

### Risks

```text
224x224 image size
patch size 14
many ViT tokens
attention over image tokens
LayerNorm/softmax cost
large activation memory
```

### Acceptance criteria

```text
Prove one image encoding.
Prove history encoding.
Prove goal encoding.
Pixel-to-latent proof composes with rollout proof.
```

---

## RFC-012: Recursive / aggregated verification

**Status:** V1/V2.

### Motivation

A complete planning proof may be too large as one monolithic trace. Split proofs can be aggregated.

### Specification

Possible proof decomposition:

```text
proof A: model commitment and weight validity
proof B_s: candidate rollout s
proof C: cost and argmin over candidate cost commitments
proof D: recursive aggregation
```

Stwo’s ecosystem includes Cairo/STARK verifier paths and recursion-oriented work, but the recursive design for this custom relation must be engineered and benchmarked separately. ([GitHub][12])

### Acceptance criteria

```text
Aggregated verifier accepts iff all child proofs are valid.
Changing one child output commitment invalidates aggregate proof.
Recursive verifier public input schema is canonical.
```

---

## RFC-013: Testing, fuzzing, and audit strategy

**Status:** Required before any public claim.

### Test layers

```text
1. Python floating-point reference tests
2. Python fixed-point reference tests
3. Rust fixed-point reference tests
4. AIR witness generation tests
5. Prover/verifier accept tests
6. Negative reject tests
7. Differential tests across random seeds
8. Constraint mutation tests
9. Manifest serialization tests
10. Security review
```

### Required negative tests

```text
wrong model commitment
wrong quantization commitment
wrong action
wrong latent
wrong intermediate activation
wrong accumulator
wrong rounding remainder
wrong activation lookup value
wrong LayerNorm reciprocal
wrong attention probability
wrong candidate cost
wrong argmin
wrong tie-break
```

### Acceptance criteria

No component is complete unless it has both:

```text
accepting tests for valid witnesses
rejecting tests for invalid witnesses
```

---

## 12. Implementation milestones

### Milestone 0: design freeze

Deliver:

```text
main technical spec
RFC-000 through RFC-005
canonical manifest schema
fixed-point arithmetic spec
reference quantization policy
```

Exit criteria:

```text
one frozen relation_id
one frozen serialization format
one frozen rounding policy
```

### Milestone 1: arithmetic and linear layer prover

Deliver:

```text
range-check component
linear component
requantization component
tensor memory component
```

Exit criteria:

```text
prove one quantized linear layer
prove batched linear layer
reject malformed witnesses
```

### Milestone 2: proof-native LeWM predictor primitive

Deliver:

```text
action_encoder
MLP
activation
normalization
attention approximation
one predictor block
```

Exit criteria:

```text
one block proof matches Rust fixed-point reference
all nonlinear approximations committed
```

### Milestone 3: full predictor step

Deliver:

```text
six-block ARPredictor_Q
pred_proj
one-step prediction proof
```

Exit criteria:

```text
P0 proof works end-to-end
```

### Milestone 4: rollout proof

Deliver:

```text
autoregressive windowing
multi-step rollout
trajectory commitment
```

Exit criteria:

```text
P1 proof works for configurable horizon
```

### Milestone 5: fixed-candidate planning proof

Deliver:

```text
batched rollouts
MSE costs
argmin/tie-break proof
```

Exit criteria:

```text
P2 proof works for fixed candidate set
```

### Milestone 6: planner extension

Deliver one of:

```text
CEM proof
or
proof-native planner replacement
or
external committed-candidate protocol
```

Exit criteria:

```text
claim about planning exactly matches what is proven
```

### Milestone 7: pixel encoder

Deliver:

```text
patch embedding
ViT encoder
projector
pixel-to-latent proof
```

Exit criteria:

```text
P4 proof works for observation history and goal image
```

---

## 13. Critical risk register

| Risk                     | Severity | Description                                                                   | Mitigation                                              |
| ------------------------ | -------: | ----------------------------------------------------------------------------- | ------------------------------------------------------- |
| Floating-point ambiguity | Critical | PyTorch/bf16/GPU kernels are not a clean proof relation                       | Use exported fixed-point model                          |
| Field wraparound         | Critical | Finite-field values can satisfy constraints while violating integer semantics | Range-check every integer interpretation                |
| Softmax cost             |     High | Attention softmax is expensive and approximation-sensitive                    | Lookup/rational approximation or proof-native attention |
| LayerNorm cost           |     High | Inverse sqrt/division are awkward                                             | Bounded lookup/Newton or retrained normalization        |
| CEM explosion            |     High | Candidate count × rollout horizon × iterations is huge                        | Start with fixed candidates; prove CEM later            |
| Encoder cost             |     High | Pixel ViT proof is much larger than latent predictor proof                    | Defer to V3                                             |
| Unsound planner claim    | Critical | Proving only selected rollout does not prove planning                         | Prove all candidates or full CEM                        |
| ZK overclaim             | Critical | Validity proof may leak witnesses                                             | Do not claim ZK until hiding audited                    |
| Model commitment gaps    | Critical | Unbound scales/tables let prover change semantics                             | Commit full manifest                                    |
| `stwo-circuits` maturity |   Medium | Useful but not full NN framework                                              | Pin revisions, audit, use direct AIR for hot paths      |

---

## 14. Recommended V0 proof statement

The most defensible initial public statement is:

```text
This proof verifies quantized LeWorldModel latent planning for a fixed candidate set.

Given:
  - a committed QuantizedLeWM predictor manifest,
  - an initial latent history,
  - a goal latent,
  - S candidate action sequences,

the proof verifies that:
  - every candidate was rolled out through the committed quantized predictor,
  - every final latent cost was computed as specified,
  - the selected candidate has minimum cost under deterministic tie-breaking.
```

That is accurate, useful, and implementable.

Do **not** initially claim:

```text
proves the original PyTorch model
proves the true future
proves full CEM
proves zero-knowledge privacy
proves end-to-end pixel planning
```

unless those relations are actually included.

---

## 15. Bottom line

The project should be built as a **custom Stwo AIR for deterministic fixed-point LeWorldModel inference**, with the first milestone focused on **latent predictor rollout and fixed-candidate planning**.

The core technical work is not “use ZK for AI” in the abstract. It is:

```text
1. Freeze an exact quantized LeWorldModel relation.
2. Bind the full model and quantization manifest.
3. Implement range-safe fixed-point tensor AIR.
4. Implement linear/matmul efficiently.
5. Implement or replace softmax/LayerNorm/GELU soundly.
6. Prove rollout recurrence.
7. Prove costs and argmin.
8. Only then extend to CEM and pixel encoder.
```

That gives a rigorous, auditable path to a provable JEPA-style world model without making claims the proof does not support.

[1]: https://github.com/starkware-libs/stwo "GitHub - starkware-libs/stwo: StarkWare's next gen prover · GitHub"
[2]: https://docs.starknet.io/learn/S-two-book/introduction "Introduction - Starknet Documentation"
[3]: https://github.com/starkware-libs/stwo-circuits/tree/main/crates "stwo-circuits/crates at main · starkware-libs/stwo-circuits · GitHub"
[4]: https://github.com/lucas-maes/le-wm "GitHub - lucas-maes/le-wm: Official code base for LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels · GitHub"
[5]: https://github.com/lucas-maes/le-wm/blob/main/jepa.py "le-wm/jepa.py at main · lucas-maes/le-wm · GitHub"
[6]: https://github.com/lucas-maes/le-wm/blob/main/module.py "le-wm/module.py at main · lucas-maes/le-wm · GitHub"
[7]: https://github.com/lucas-maes/le-wm/blob/main/config/eval/pusht.yaml "le-wm/config/eval/pusht.yaml at main · lucas-maes/le-wm · GitHub"
[8]: https://github.com/lucas-maes/le-wm/blob/main/config/train/model/lewm.yaml "le-wm/config/train/model/lewm.yaml at main · lucas-maes/le-wm · GitHub"
[9]: https://zksecurity.github.io/stwo-book/air-development/static-lookups/index.html "Static Lookups - Stwo Book"
[10]: https://zksecurity.github.io/stwo-book/air-development/writing-a-simple-air/proving-an-air.html "Proving and Verifying an AIR - Stwo Book"
[11]: https://zksecurity.github.io/stwo-book/air-development/preprocessed-trace/index.html "Preprocessed Trace - Stwo Book"
[12]: https://github.com/starkware-libs/stwo-cairo "GitHub - starkware-libs/stwo-cairo: Prove Cairo programs with the blazing-fast S-two prover, powered by the cryptographic breakthrough of Circle STARKs. · GitHub"
[13]: https://raw.githubusercontent.com/lucas-maes/le-wm/main/config/eval/solver/cem.yaml "raw.githubusercontent.com"
