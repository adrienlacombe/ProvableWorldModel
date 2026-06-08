# RFC-0010: CEM planner proof

- Status: Draft
- Authors: ProvableWorldModel maintainers
- Created: 2026-06-03
- Target milestone: Future

## Summary

This RFC defines the complete requirement set for a sound proof of the
Cross-Entropy Method (CEM) planner used by LeWorldModel evaluation, and locks
the project-level decision that V0 (statement P2, fixed-candidate planning) does
NOT attempt CEM. A CEM proof corresponds to proof statement tier P3
(`relation_id` `pwm.lewm.cem_planning.v1`), shipped no earlier than release V2.
The decision locked here is twofold. First: a sound CEM proof MUST constrain the
entire optimizer recurrence end-to-end over the verified eval configuration
(`num_samples = 300`, `n_steps = 30`, `topk = 30`, `horizon = 5`,
`action_block = 5`) — seeded PRNG sample generation, candidate clipping to the
action domain, cost of every candidate in every iteration, top-k elite
selection, the elite mean update, the elite variance/standard-deviation update,
the iteration recurrence binding one iteration's distribution to the next, and
the final selection — because proving only the returned action sequence lets a
malicious prover pick favorable candidates off-circuit. Second: V0 ships P2 only;
CEM is explicitly deferred behind a gating dependency on a proof-native sampling
primitive. We additionally name the Gaussian-sampling and inverse-square-root
cost risk as the dominant feasibility obstacle and lock the two acceptable
proof-native escape hatches (discrete or fixed-uniform sampling, or a committed
candidate set) as the resolution path, not as silent fallbacks.

## Motivation

`docs/feasibility-study.md` §9.5 establishes the planner-soundness rule: "For CEM
planning, proving only the final candidate is insufficient. The proof must
include seeded sample generation, candidate clipping, all costs, top-k selection,
mean update, variance/std update, iteration recurrence, final selection.
Otherwise, the prover can choose favorable candidates off-circuit." §13 lists
"CEM explosion" (candidate count x rollout horizon x iterations) as a High risk
and "Unsound planner claim" as Critical. The scope ladder in
`docs/spec/00-overview.md#scope-and-statement-tiers` places CEM at P3/V2, after
the P2 fixed-candidate planner is stable.

The concrete system scenario: a deployment wants to prove "this is the action
sequence the CEM planner actually selected for this goal under this seed", not
merely "this action sequence has low cost". The weaker claim is exactly the
P2 statement (`docs/rfcs/RFC-0009-fixed-candidate-planner-proof.md`), which proves
a committed candidate set was scored and an argmin chosen, but says nothing about
where the candidates came from. CEM closes that gap by binding the candidate
generation process itself. The cost of closing it is large and the dominant
single risk is the Gaussian sampling and `1/sqrt` operations inside CEM, which
are awkward in M31 fixed-point arithmetic; this RFC exists to lock the
requirements and the escape hatches before any P3 implementation begins so that
the V0/V1 work (`docs/rfcs/RFC-0009-fixed-candidate-planner-proof.md`) is not
contaminated by premature CEM machinery.

The CEM configuration is fixed and verified against the LeWorldModel repository
upstream at 2026-06-03 (`config/eval/solver/cem.yaml` and
`config/eval/pusht.yaml plan_config`):

| Parameter         | Value | Source config                       | Meaning                                        |
| ----------------- | ----: | ----------------------------------- | ---------------------------------------------- |
| `num_samples`     |   300 | `solver/cem.yaml`                   | candidates sampled per CEM iteration           |
| `n_steps`         |    30 | `solver/cem.yaml`                   | CEM iterations (distribution refinements)      |
| `topk`            |    30 | `solver/cem.yaml`                   | elite count used to update the distribution    |
| `var_scale`       |   1.0 | `solver/cem.yaml`                   | initial variance scale                         |
| `batch_size`      |     1 | `solver/cem.yaml`                   | planning batch                                 |
| `horizon`         |     5 | `pusht.yaml plan_config`            | planned steps per action sequence              |
| `receding_horizon`|     5 | `pusht.yaml plan_config`            | steps executed before replan                   |
| `action_block`    |     5 | `pusht.yaml plan_config`            | actions grouped per planning block             |

These names and values are normative for P3 and for
`docs/spec/08-performance-budget.md#scaling`. A naive P3 trace scores
`num_samples * n_steps = 300 * 30 = 9000` rollouts of `horizon = 5` predictor
steps each, versus P2's single pass over a committed candidate set; this 9000x
multiplier over the P2 hot path is the reason for deferral.

## Goals

- Lock the full set of constraints a P3 proof MUST enforce to be sound, such that
  no prover can substitute, omit, or reorder any candidate, elite, distribution
  update, or iteration.
- Lock the seeded-PRNG determinism contract so the Python fixed-point reference
  and the Rust fixed-point reference and the AIR produce bit-identical candidate
  streams from the same `planner_seed`.
- Lock the project decision that V0 = P2 and that CEM is out of scope until V2,
  with an explicit gating dependency.
- Name the Gaussian-sampling / inverse-sqrt cost risk and lock the two
  proof-native alternatives (discrete/fixed-uniform sampling, or a committed
  candidate set) as the only acceptable resolution paths, each minting a distinct
  `relation_id`.
- Define the P3 `PublicInput` population and the `planner_config_commitment`
  binding so that the CEM hyperparameters above are cryptographically fixed.

## Non-Goals

- Implementing P3 in V0 or V1. This RFC specifies; it does not schedule build
  work inside the v1.0 milestone.
- Proving that CEM converges to a globally optimal action, or that the selected
  action is good in the real environment. P3 proves the planner executed its
  own deterministic recurrence correctly, nothing about world truth
  (`docs/spec/00-overview.md#non-goals`).
- Proving floating-point parity with the PyTorch CEM implementation. P3 proves a
  quantized fixed-point CEM relation; the PyTorch planner is a reference for
  porting, not the proven object (`docs/feasibility-study.md` §1.1).
- Zero-knowledge. P3 inherits the validity-proof-not-ZK posture from
  `docs/spec/06-security.md#privacy-and-zk`. The seed and config are public.
- Receding-horizon multi-replan control loops. P3 proves one CEM solve
  (`n_steps` iterations producing one `action_block`); composing successive
  solves with executed observations is a P4-adjacent concern deferred with the
  pixel encoder (`docs/rfcs/RFC-0011-pixel-encoder-proof.md`).
- The fixed-candidate scoring/argmin machinery itself, which P3 reuses verbatim
  from `docs/rfcs/RFC-0009-fixed-candidate-planner-proof.md`.

## Proposed Design

P3 is layered strictly on top of P2: every CEM iteration scores its `num_samples`
candidates with the exact rollout + MSE-cost + comparison machinery already
locked for P2 (`docs/rfcs/RFC-0008-rollout-air.md`,
`docs/rfcs/RFC-0009-fixed-candidate-planner-proof.md`). The new constraints are
the sampling front-end and the distribution-update / iteration recurrence
back-end that wrap that scoring.

### Statement P3 (locked relation)

`relation_id` = `pwm.lewm.cem_planning.v1` (Gaussian-sampling variant) — see the
relation-minting rule below for the proof-native variants. `relation_id`s are
immutable per `docs/spec/03-data-model.md#relation-id`; any semantic change mints
a new id.

```text
Public:
  relation_id
  model_commitment
  quantization_commitment
  planner_config_commitment   # binds num_samples, n_steps, topk, var_scale,
                              # horizon, action_block, sampler variant, clip
                              # bounds, init mean/var, PRNG algorithm id
  latent_history (public or committed)
  goal_latent   (public or committed)
  planner_seed                # public; part of planner_config domain
  selected_action_sequence    # the returned action_block, committed in
                              # claimed_output_commitment

Private witness:
  weights (committed-only or public, per manifest)
  per-iteration: sampled candidate set, clipped candidate set, all costs,
                 elite index set (top-k), updated mean, updated variance,
                 PRNG state stream, all rollout/cost/range/lookup sub-witnesses

Claim (for iteration t = 0 .. n_steps-1, with (mean_0, var_0) from config):
  1. SAMPLE:   cand_{t,j} = Sampler_Q(mean_t, var_t, PRNG_stream(seed, t, j))   for j in 0..num_samples-1
  2. CLIP:     clipped_{t,j} = Clip_Q(cand_{t,j}, action_lo, action_hi)
  3. SCORE:    cost_{t,j} = MSE_Q(final(Rollout_Q(latent_history, clipped_{t,j})), goal_latent)
  4. TOPK:     elites_t = the topk indices of smallest cost_{t,*}, deterministic tie-break
  5. MEAN:     mean_{t+1} = (1/topk) * sum_{j in elites_t} clipped_{t,j}
  6. VAR:      var_{t+1}  = (1/topk) * sum_{j in elites_t} (clipped_{t,j} - mean_{t+1})^2
  7. RECUR:    (mean_{t+1}, var_{t+1}) feed iteration t+1
  8. FINAL:    selected_action_sequence == final_select(mean_{n_steps}, elites_{n_steps-1})
```

`final_select` is fixed by `planner_config` to one of {return `mean_{n_steps}`,
return the lowest-cost elite of the last iteration}; the choice is bound in
`planner_config_commitment` and is part of the relation semantics. The reference
implementation MUST match the LeWorldModel solver's choice bit-for-bit; the
porting task records which it is in the manifest (see Open Questions).

### `PublicInput` population (canonical type, no divergence)

P3 reuses the canonical `PublicInput` from
`docs/spec/03-data-model.md#public-input` and §6.3 of the authoring contract.
The P3-specific population:

```rust
pub struct PublicInput {
    pub relation_id: [u8; 32],                              // pwm.lewm.cem_planning.v1
    pub model_commitment: [u8; 32],
    pub quantization_commitment: [u8; 32],
    pub planner_config_commitment: [u8; 32],                // REQUIRED, binds CEM config + seed
    pub statement_type: StatementType,                      // StatementType::P3Cem
    pub latent_history_commitment: Option<[u8; 32]>,
    pub latent_history_public: Option<Vec<M31>>,
    pub goal_latent_commitment: Option<[u8; 32]>,
    pub goal_latent_public: Option<Vec<M31>>,
    pub candidate_actions_commitment: Option<[u8; 32]>,     // None: CEM generates candidates
    pub candidate_actions_public: Option<Vec<M31>>,         // None: see candidate-set variant
    pub claimed_output_commitment: [u8; 32],                // commits selected_action_sequence
    pub selected_index: Option<u32>,                        // None for P3 (no single committed set)
    pub selected_cost: Option<BoundedInt>,                  // None for P3 (final select is by mean)
}
```

`StatementType::P3Cem` is the discriminant from §6.3. For the Gaussian variant,
`candidate_actions_commitment` and `candidate_actions_public` are `None` because
candidates are derived inside the proof from `planner_seed`; for the
committed-candidate-set variant they are populated (see Alternatives). The seed
is carried inside the bytes committed by `planner_config_commitment`, so it is
public but tamper-evident: changing the seed changes the commitment and the
proof is then valid only for the new (seed, config) pair.

### `planner_config_commitment` binding

The CEM configuration is unsound if any field is left unbound, because the prover
could pick the favorable hyperparameters after seeing candidate costs. The
committed planner config schema (extends the manifest `planner` block of
`docs/spec/03-data-model.md#model-manifest`):

```yaml
planner:
  type: cem
  relation_id: pwm.lewm.cem_planning.v1
  sampler: gaussian_q | discrete_q | uniform_fixed_q | committed_set
  prng_algorithm_id: <committed algorithm identifier>   # e.g. counter-mode keyed permutation
  planner_seed: <u64, public>
  num_samples: 300
  n_steps: 30
  topk: 30
  var_scale: 1.0
  action_block: 5
  horizon: 5
  init_mean_commitment: 0x...        # initial CEM mean tensor
  init_var_commitment: 0x...         # initial CEM variance tensor
  clip_bounds: { action_lo: <BoundedInt>, action_hi: <BoundedInt> }
  final_select: mean | best_elite
  tie_break: smallest_index          # shared with argmin, RFC-0009
  topk_tie_break: smallest_index
```

INV-RFC0010-01 (config binding): every field above is covered by
`planner_config_commitment`. A verifier rejects with
`VerifyError::PlannerConfigMismatch`
(`docs/spec/04-error-model.md#verifier-rejections`) if the recomputed digest of
the supplied planner config does not equal `planner_config_commitment`.

### AIR component model (new components on the `cem` axis)

P3 adds two components to `pwm-air` (the `cem` component named in the crate map),
both wrapping the existing P2 components. They follow the
preprocessed/main/interaction trace split of
`docs/spec/01-architecture.md#trace-model`.

```text
SamplerComponent     # PRNG stream + sample transform + clip, per (iteration, sample)
DistributionUpdateComponent  # top-k membership + mean update + var update + recurrence
```

P3 reuses, unchanged:

```text
RolloutComponent   (RFC-0008)   # per candidate, per iteration
CostComponent      (RFC-0009)   # MSE_Q final-latent-vs-goal
ArgminComponent    (RFC-0009)   # reused for top-k boundary comparisons
RangeCheckComponent / ActivationLookupComponent (RFC-0003)
TensorMemoryComponent (RFC-0004)
```

#### SamplerComponent constraints

For each (iteration `t`, sample `j`), the PRNG is a committed counter-mode keyed
permutation seeded by `planner_seed`; the counter is the deterministic tuple
`(t, j, k)` over action coordinate `k`. This makes the stream addressable and
independent of trace row order, which is required for the multiset/permutation
argument over candidates.

```text
prng_{t,j,k}      = PRNG_Q(seed, t, j, k)                    # committed algorithm
raw_{t,j,k}       = SampleTransform_Q(prng_{t,j,k}, mean_{t,k}, var_{t,k})
clipped_{t,j,k}   = Clip_Q(raw_{t,j,k}, action_lo, action_hi)

range checks:     prng_{t,j,k}, raw_{t,j,k}, clipped_{t,j,k}, mean_{t,k}, var_{t,k}
lookup:           any sqrt / Gaussian-inverse-CDF table is committed (RFC-0006) and
                  enforced by ActivationLookupComponent; out-of-domain witness rejects
clip semantics:   clipped = action_lo if raw < action_lo
                  clipped = action_hi if raw > action_hi
                  clipped = raw        otherwise        (each branch range-proven)
```

`SampleTransform_Q` for the Gaussian variant requires `sqrt(var)` and a
Gaussian-shaping step; both are committed approximations under RFC-0006 (proof-
native GELU/Softmax/LayerNorm components also live there). This is the cost risk
named below.

INV-RFC0010-02 (sample completeness): the multiset of `(t, j)` sample rows
present in the SamplerComponent trace equals exactly
`{0..n_steps-1} x {0..num_samples-1}`, enforced by a permutation argument against
a preprocessed index table. A prover cannot omit, duplicate, or inject a sample.

INV-RFC0010-03 (PRNG fidelity): every `prng_{t,j,k}` equals the committed PRNG
applied to `(seed, t, j, k)`; the PRNG state recurrence is fully constrained, not
prover-supplied. A prover cannot substitute a hand-picked candidate.

#### DistributionUpdateComponent constraints

```text
# TOP-K membership (per iteration t)
elites_t is a size-topk subset of {0..num_samples-1}
for each elite e in elites_t and each non-elite n not in elites_t:
    cost_{t,n} - cost_{t,e} >= 0                # every elite no worse than every non-elite
topk_tie_break: among equal costs at the boundary, smallest index is elite
                enforced as cost_{t,n} - cost_{t,e} - 1 >= 0 for n < e at equal cost

# MEAN update (per coordinate k); topk fixed => exact integer fixed-point
sum_mean_{t,k}  = sum_{e in elites_t} clipped_{t,e,k}
mean_{t+1,k}    = Requantize_div_by_topk(sum_mean_{t,k})    # quotient/remainder, RFC-0002

# VAR update (per coordinate k)
sum_var_{t,k}   = sum_{e in elites_t} (clipped_{t,e,k} - mean_{t+1,k})^2
var_{t+1,k}     = Requantize_div_by_topk(sum_var_{t,k})

# RECURRENCE
(mean_{t+1}, var_{t+1}) are the inputs consumed by SamplerComponent at iteration t+1
  enforced by tensor-memory read/write linkage (RFC-0004): the write at time(t+1)
  is the read at sampler iteration t+1.
```

Division by `topk` uses the exact quotient/remainder requantization semantics of
`docs/rfcs/RFC-0002-fixed-point-arithmetic-over-m31.md` with the manifest rounding
mode (`nearest_ties_to_even` default); there is no informal division. Because
`topk = 30` is a fixed power-free constant, the divisor is a manifest constant,
not a witness.

INV-RFC0010-04 (top-k correctness): the elite set is exactly the `topk` smallest
costs under deterministic tie-break; enforced by the pairwise elite/non-elite
comparison above (`num_samples - topk` x `topk` nonnegativity checks per
iteration), reusing the `cost_s - selected_cost - 1 >= 0` range-check idiom from
`docs/rfcs/RFC-0009-fixed-candidate-planner-proof.md`. A prover cannot mark a
higher-cost candidate as elite.

INV-RFC0010-05 (mean/variance correctness): `mean_{t+1}` and `var_{t+1}` are the
exact fixed-point elite mean and variance, with quotient/remainder range-checked.
A prover cannot fabricate a distribution that biases the next iteration.

INV-RFC0010-06 (iteration recurrence): the distribution produced by iteration `t`
is the distribution consumed by iteration `t+1`, linked through tensor memory; no
iteration may be skipped, reordered, or run with a substituted distribution. The
initial distribution equals `init_mean_commitment` / `init_var_commitment`.

INV-RFC0010-07 (final selection): `selected_action_sequence` committed in
`claimed_output_commitment` equals `final_select` applied to the final-iteration
distribution/elites per the committed `final_select` mode. A prover cannot return
an action sequence that the recurrence did not produce.

These seven invariants are the locked sound-CEM requirement set. Their
correspondence to `docs/feasibility-study.md` §9.5: seeded sample generation
(INV-02, INV-03), candidate clipping (Clip_Q in SamplerComponent), all costs
(SCORE step, reusing CostComponent over all `num_samples` x `n_steps`), top-k
(INV-04), mean update (INV-05), variance/std update (INV-05), iteration
recurrence (INV-06), final selection (INV-07).

### Data flow and lifecycle

```text
manifest + planner_config -> verify model_commitment, quantization_commitment,
                             planner_config_commitment
                          -> (mean_0, var_0) from init commitments
for t in 0..n_steps:
    SamplerComponent:  PRNG(seed,t,*) -> raw -> Clip_Q -> clipped candidates  (num_samples)
    RolloutComponent:  per candidate, horizon=5 autoregressive predictor steps (RFC-0008)
    CostComponent:     MSE_Q(final_latent, goal_latent) per candidate
    DistributionUpdateComponent: top-k -> mean_{t+1}, var_{t+1}
final_select -> selected_action_sequence -> claimed_output_commitment
```

### Determinism, concurrency, error propagation

INV-RFC0010-08 (cross-language determinism): for a fixed `(planner_seed,
planner_config, manifest, latent_history, goal_latent)`, the Python fixed-point
reference, the Rust fixed-point reference, and the AIR witness MUST produce the
same candidate stream, the same per-candidate costs, the same elite sets, the
same distribution updates, and the same `selected_action_sequence`, bit-for-bit.
This is the P3 instance of the export-parity contract
(`docs/spec/06-security.md#binding-requirements`). The PRNG is addressed by
`(t, j, k)` rather than by sequential draw order specifically so that parallel
candidate evaluation across cores produces an order-independent identical stream.

Failure modes and the system response (verifier rejections per
`docs/spec/04-error-model.md#verifier-rejections`, export/prove failures per
`docs/spec/04-error-model.md#failure-modes`):

| Failure mode                                              | Detected at | System response                                         |
| --------------------------------------------------------- | ----------- | ------------------------------------------------------- |
| `planner_config_commitment` mismatch (INV-01)             | verify      | reject `VerifyError::PlannerConfigMismatch`             |
| seed altered after commitment                             | verify      | reject `VerifyError::PlannerConfigMismatch` (seed bound)|
| missing/duplicated/injected sample row (INV-02)           | verify      | reject `VerifyError::PermutationCheckFailed`            |
| PRNG output not equal to committed PRNG (INV-03)          | verify      | reject `VerifyError::ConstraintUnsatisfied`             |
| clip branch witness out of declared range                 | verify      | reject `VerifyError::RangeCheckFailed`                  |
| sqrt / Gaussian-shape table value out of domain           | verify      | reject `VerifyError::LookupMembershipFailed`            |
| elite set not the topk smallest costs (INV-04)            | verify      | reject `VerifyError::ConstraintUnsatisfied`             |
| mean/variance quotient or remainder invalid (INV-05)      | verify      | reject `VerifyError::RangeCheckFailed`                  |
| iteration distribution not linked t->t+1 (INV-06)         | verify      | reject `VerifyError::TensorMemoryInconsistent`          |
| final selection not from final distribution (INV-07)      | verify      | reject `VerifyError::ConstraintUnsatisfied`             |
| accumulator overflow during cost/var sum                  | verify      | reject `VerifyError::RangeCheckFailed` (overflow=reject)|
| Python and Rust reference disagree (INV-08)               | export      | abort export `ExportError::ParityMismatch`; no proof    |
| `relation_id` is a CEM variant the verifier lacks         | verify      | reject `VerifyError::UnsupportedRelation`               |
| proof generated for P3 submitted as P2 (or vice versa)    | verify      | reject `VerifyError::RelationMismatch`                  |

`VerifyError` is the canonical enum from
`docs/spec/02-public-api.md#rust-public-api`; the specific variants resolve in
`docs/spec/04-error-model.md#verifier-rejections`.

### The Gaussian-sampling / sqrt cost risk (locked)

CEM samples `cand ~ Normal(mean, var)` and the standard transform needs
`std = sqrt(var)` plus a Gaussian shaping of uniform PRNG output (Box-Muller or
inverse-CDF). In M31 fixed-point: `sqrt` and the Gaussian inverse-CDF are not
field-native, so they become committed lookup tables or bounded Newton iterations
under `docs/rfcs/RFC-0006-nonlinear-primitive-components.md`, each multiplied by
`num_samples * n_steps * action_block * horizon = 300 * 30 * 5 * 5 = 225000`
coordinate-samples plus 9000 full rollouts. This is the High-severity "CEM
explosion" risk of `docs/feasibility-study.md` §13.

LOCKED resolution: the Gaussian variant (`pwm.lewm.cem_planning.v1`) is permitted
but is NOT the recommended first P3. The recommended first P3 uses a proof-native
sampler that avoids `sqrt`/Gaussian-CDF entirely. The acceptable samplers, each
minting a distinct immutable `relation_id`, are:

| Sampler variant   | `relation_id`                       | Avoids sqrt? | Notes                                              |
| ----------------- | ----------------------------------- | ------------ | -------------------------------------------------- |
| `gaussian_q`      | `pwm.lewm.cem_planning.v1`          | no           | faithful CEM; committed sqrt + inverse-CDF tables  |
| `discrete_q`      | `pwm.lewm.cem_discrete.v1`          | yes          | candidates from a committed discrete action grid   |
| `uniform_fixed_q` | `pwm.lewm.cem_uniform.v1`          | yes          | uniform noise scaled by `var` (no sqrt of var)     |
| `committed_set`   | `pwm.lewm.cem_committed_set.v1`     | n/a          | external committed candidates; degenerates to P2   |

The proof of a `discrete_q` or `uniform_fixed_q` planner is a proof of a
proof-native CEM, not of the PyTorch Gaussian CEM; the manifest `model_family`
and `relation_id` make this explicit, and the public statement MUST say "quantized
proof-native CEM", never "the PyTorch CEM planner". The `committed_set` variant
collapses the sampling front-end entirely and is identical to P2 with externally
attested candidates; it exists as the safe floor.

## Alternatives Considered

### Alternative A: prove only the returned action sequence's rollout and cost

What it is: skip sampling, top-k, and distribution updates; prove only that the
single returned `selected_action_sequence` rolls out to some cost (a P1/P2-shaped
claim). Why considered: it is `n_steps * num_samples` = 9000x cheaper and reuses
P1/P2 directly. Why rejected: it is unsound as a CEM claim. It proves nothing
about candidate generation, so a malicious prover runs CEM honestly off-circuit,
or not at all, and submits any low-cost sequence it likes. This is precisely the
"prover can choose favorable candidates off-circuit" failure of
`docs/feasibility-study.md` §9.5. It is a valid P1/P2 claim but MUST NOT be
labeled CEM. Rejected as the CEM relation; available as the unrelated P2 relation.

### Alternative B: prove a committed candidate set + argmin, no sampling proof

What it is: the planner runs CEM off-circuit, commits the union of all candidates
it considered, and proves only scoring + argmin over that committed set (the
`committed_set` variant above). Why considered: it removes the entire
sqrt/Gaussian cost and is implementable on the V0/V1 P2 machinery today. Why
rejected as the primary CEM relation: it does not prove the candidates were
produced by CEM from the seed; it proves the chosen action is the argmin of a
prover-chosen set. It is honest only if the public statement says exactly that.
We keep it as the explicit `pwm.lewm.cem_committed_set.v1` floor relation, not as
the headline CEM proof, and we forbid presenting it as "proves the CEM planner".

### Alternative C: faithful Gaussian CEM with committed sqrt/inverse-CDF tables

What it is: the `gaussian_q` variant — implement Box-Muller or inverse-CDF
Gaussian sampling and `sqrt(var)` as committed lookup tables / bounded Newton
iterations, matching the PyTorch planner's distribution as closely as fixed-point
allows. Why considered: it is the only variant that can claim parity (within
quantization error) with the original CEM planner. Why not chosen as the first
P3: the `sqrt` + Gaussian-shaping cost across 225000 coordinate-samples plus the
table-domain range proofs make it the most expensive and audit-heavy variant, and
it still cannot claim float parity (only quantized parity). It remains a permitted
relation for when faithful-distribution CEM is genuinely required, gated behind a
working `discrete_q`/`uniform_fixed_q` P3 and an RFC-0006 sqrt-table audit.

## Drawbacks

- A sound P3 is large: 9000 rollouts and 225000 coordinate-samples per solve for
  the verified config, dominating any single P0/P1/P2 proof. Recursion/aggregation
  (`docs/rfcs/RFC-0012-recursive-aggregated-verification.md`) is effectively a
  prerequisite for practical P3 proving times; see
  `docs/spec/08-performance-budget.md#scaling`.
- The proof-native variants (`discrete_q`, `uniform_fixed_q`) prove a different
  planner than the PyTorch Gaussian CEM. We accept a behavior gap (the proven
  planner is not bit-identical to the reference checkpoint's planner) in exchange
  for tractable, auditable proofs. This is disclosed in the public statement and
  in the manifest `relation_id`.
- The faithful `gaussian_q` variant carries an approximation-quality dependency on
  RFC-0006 sqrt/inverse-CDF tables that has its own audit surface; an error in a
  committed table is enforced but not proven close to the true Gaussian.
- Multiple CEM `relation_id`s increase the verifier's supported-relation surface
  and the test matrix.

## Migration / Rollout

- Feature-gated: P3 ships behind a `cem` Cargo feature in `pwm-air` and
  `pwm-prover`, off by default through V0 and V1. The V0/V1 P2 path
  (`docs/rfcs/RFC-0009-fixed-candidate-planner-proof.md`) never compiles CEM code.
- Relation versioning: each sampler variant is a distinct immutable `relation_id`
  (`pwm.lewm.cem_planning.v1`, `pwm.lewm.cem_discrete.v1`,
  `pwm.lewm.cem_uniform.v1`, `pwm.lewm.cem_committed_set.v1`), per
  `docs/spec/09-release-and-versioning.md#relation-versioning`. A verifier that
  does not implement a variant rejects with `VerifyError::UnsupportedRelation`; it
  never silently treats one variant as another.
- Schema versioning: the `planner` manifest block gains the CEM fields under a
  bumped `manifest_version`; older manifests without a `planner.type: cem` block
  are unaffected and remain valid for P0/P1/P2
  (`docs/spec/03-data-model.md#schema-versioning`).
- `StatementType::P3Cem` already exists in the canonical enum
  (`docs/spec/03-data-model.md#witness`, contract §6.3), so no public-API break is
  required to introduce P3; the verifier dispatch on `statement_type` gains a P3
  arm.
- Rollout order within V2: land `committed_set` first (reuses P2 wholesale), then
  `uniform_fixed_q`/`discrete_q` (adds Sampler + DistributionUpdate without sqrt),
  then `gaussian_q` (adds sqrt/inverse-CDF tables) gated on the RFC-0006 audit.
- Deprecation: no deprecation in this RFC; P3 is additive. If a CEM relation is
  ever found unsound, it is retired by minting a new `relation_id` and marking the
  old one rejected at the verifier, never by mutating semantics in place
  (`docs/spec/09-release-and-versioning.md#deprecation`).

## Testing Strategy

All tests follow `docs/spec/07-testing-strategy.md`; every component ships with
both accepting and rejecting tests per
`docs/rfcs/RFC-0013-testing-fuzzing-and-audit-strategy.md`. Rejections assert the
specific `VerifyError` from `docs/spec/04-error-model.md#verifier-rejections`.

Accepting tests (`docs/spec/07-testing-strategy.md#golden-vectors`,
`#differential-tests`):

- `cem_golden_full_solve_uniform`: a full `n_steps=30`, `num_samples=300`,
  `topk=30`, `horizon=5` solve under `uniform_fixed_q` proves and verifies; the
  golden `selected_action_sequence` is fixed in the vector set.
- `cem_diff_python_rust_uniform`: Python fixed-point reference and Rust
  fixed-point reference produce bit-identical candidate stream, costs, elite sets,
  mean/var per iteration, and final selection for a fixed `(seed, config)`
  (INV-RFC0010-08).
- `cem_diff_seed_determinism`: two runs with the same `planner_seed` produce
  identical proofs-inputs; two runs with different seeds produce different
  candidate streams and different commitments.
- `cem_golden_committed_set`: `committed_set` variant degenerates to and matches a
  P2 golden vector for the same candidate set
  (`docs/rfcs/RFC-0009-fixed-candidate-planner-proof.md`).
- `cem_golden_topk_boundary_ties`: an iteration with exact cost ties at the
  topk/non-topk boundary selects the smallest indices as elites (INV-RFC0010-04
  tie-break).
- `cem_golden_gaussian_small`: a reduced-config `gaussian_q` solve
  (`num_samples=8`, `n_steps=2`) proves and verifies, exercising committed sqrt /
  inverse-CDF tables.

Rejecting / negative tests (`docs/spec/07-testing-strategy.md#negative-tests`,
`#mutation-tests`):

- `cem_reject_dropped_iteration`: remove iteration `t`'s sample rows ->
  `VerifyError::PermutationCheckFailed` (INV-RFC0010-02).
- `cem_reject_injected_candidate`: replace one `clipped_{t,j}` with a
  hand-picked low-cost candidate not produced by the PRNG ->
  `VerifyError::ConstraintUnsatisfied` (INV-RFC0010-03).
- `cem_reject_wrong_topk`: mark a higher-cost candidate as elite ->
  `VerifyError::ConstraintUnsatisfied` (INV-RFC0010-04).
- `cem_reject_topk_tiebreak`: at an exact tie, select the larger index as elite ->
  `VerifyError::ConstraintUnsatisfied` (INV-RFC0010-04 tie-break).
- `cem_reject_bad_mean`: perturb `mean_{t+1}` by one ULP ->
  `VerifyError::RangeCheckFailed` on the quotient/remainder (INV-RFC0010-05).
- `cem_reject_bad_variance`: perturb `var_{t+1}` -> `VerifyError::RangeCheckFailed`
  (INV-RFC0010-05).
- `cem_reject_broken_recurrence`: feed iteration `t+1` a distribution other than
  iteration `t`'s output -> `VerifyError::TensorMemoryInconsistent`
  (INV-RFC0010-06).
- `cem_reject_final_not_from_distribution`: return an action sequence not produced
  by `final_select` over the final distribution ->
  `VerifyError::ConstraintUnsatisfied` (INV-RFC0010-07).
- `cem_reject_config_mutation`: change `num_samples`, `n_steps`, `topk`, or
  `planner_seed` after commitment -> `VerifyError::PlannerConfigMismatch`
  (INV-RFC0010-01).
- `cem_reject_sqrt_out_of_domain`: supply a sqrt-table witness outside the
  committed domain (gaussian variant) -> `VerifyError::LookupMembershipFailed`.
- `cem_reject_overflow_var_accumulator`: drive the variance accumulator past its
  declared bound (same field value, different integer) ->
  `VerifyError::RangeCheckFailed`, never wrap (overflow=reject,
  `docs/rfcs/RFC-0002-fixed-point-arithmetic-over-m31.md`).
- `cem_reject_p3_as_p2`: submit a `pwm.lewm.cem_planning.v1` proof as a P2
  statement -> `VerifyError::RelationMismatch`.

Differential / ML-specific (`docs/spec/07-testing-strategy.md#differential-tests`):

- `cem_float_vs_fixed_drift`: report (not gate) the action-sequence and cost drift
  between the PyTorch float CEM and the quantized `gaussian_q` reference, to bound
  the documented behavior gap; this is a quality metric, not a soundness check.

## Open Questions

- OPEN QUESTION (owner: area:planner maintainer; resolution: P3 porting task in
  V2): does the LeWorldModel solver return the final-iteration mean or the
  lowest-cost elite as the planned action (`final_select`)? Resolve by reading the
  CEM solver against the checkpoint and recording the bit-exact choice in the
  manifest `planner.final_select` before minting `pwm.lewm.cem_planning.v1`.
- OPEN QUESTION (owner: area:planner maintainer; resolution: RFC amendment before
  V2): does the LeWorldModel CEM apply per-coordinate variance with a variance
  floor, and does it re-clip the updated mean? Resolve by porting the exact update
  arithmetic into the Python fixed-point reference and confirming INV-RFC0010-05
  matches; record floor and re-clip rules in `planner_config`.
- OPEN QUESTION (owner: area:security maintainer; resolution:
  `docs/rfcs/RFC-0006-nonlinear-primitive-components.md` amendment + audit, gating
  `gaussian_q` only): which committed approximation (lookup table vs bounded
  Newton) for `sqrt(var)` and the Gaussian inverse-CDF meets the per-coordinate
  error bound, and what is that bound? The proof-native variants
  (`uniform_fixed_q`, `discrete_q`) do not depend on this resolution.
- OPEN QUESTION (owner: area:verifier maintainer; resolution:
  `docs/rfcs/RFC-0012-recursive-aggregated-verification.md`, V2): is a monolithic
  P3 trace feasible, or is per-iteration proof aggregation required to meet the
  `docs/spec/08-performance-budget.md#targets` proving-time budget? P3 build start
  is gated on this answer.

## References

- `docs/feasibility-study.md` §1.2 (statement P3), §9.5 (planner soundness), §13
  (CEM explosion, unsound planner claim), and source RFC-010.
- `docs/spec/00-overview.md#scope-and-statement-tiers`,
  `docs/spec/00-overview.md#non-goals`, `docs/spec/00-overview.md#v0-statement`.
- `docs/spec/01-architecture.md#component-model`,
  `docs/spec/01-architecture.md#trace-model`.
- `docs/spec/02-public-api.md#rust-public-api`.
- `docs/spec/03-data-model.md#public-input`, `#witness`, `#model-manifest`,
  `#relation-id`, `#schema-versioning`.
- `docs/spec/04-error-model.md#verifier-rejections`,
  `docs/spec/04-error-model.md#failure-modes`.
- `docs/spec/06-security.md#binding-requirements`,
  `docs/spec/06-security.md#privacy-and-zk`.
- `docs/spec/07-testing-strategy.md#golden-vectors`, `#negative-tests`,
  `#differential-tests`, `#mutation-tests`.
- `docs/spec/08-performance-budget.md#scaling`,
  `docs/spec/08-performance-budget.md#targets`.
- `docs/spec/09-release-and-versioning.md#relation-versioning`, `#deprecation`.
- `docs/rfcs/RFC-0002-fixed-point-arithmetic-over-m31.md` (requantization,
  comparison, overflow=reject).
- `docs/rfcs/RFC-0003-range-check-and-lookup-infrastructure.md` (LogUp range /
  lookup relations).
- `docs/rfcs/RFC-0004-tensor-memory-and-wiring-air.md` (distribution recurrence
  linkage).
- `docs/rfcs/RFC-0006-nonlinear-primitive-components.md` (sqrt / Gaussian
  approximation, committed tables).
- `docs/rfcs/RFC-0008-rollout-air.md` (per-candidate rollout, horizon=5).
- `docs/rfcs/RFC-0009-fixed-candidate-planner-proof.md` (cost, argmin, tie-break
  reused for top-k; the P2 baseline this RFC defers CEM relative to).
- `docs/rfcs/RFC-0012-recursive-aggregated-verification.md` (aggregation for P3
  trace size).
- `docs/rfcs/RFC-0013-testing-fuzzing-and-audit-strategy.md` (accepting +
  rejecting test rule).
- LeWorldModel `config/eval/solver/cem.yaml` and `config/eval/pusht.yaml`
  (`plan_config`), verified against upstream at 2026-06-03: `num_samples=300`,
  `n_steps=30`, `topk=30`, `var_scale=1.0`, `batch_size=1`, `horizon=5`,
  `receding_horizon=5`, `action_block=5`.
