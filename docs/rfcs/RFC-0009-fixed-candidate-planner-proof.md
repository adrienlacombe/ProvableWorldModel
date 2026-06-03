# RFC-0009: Fixed-candidate planner proof

- Status: Accepted
- Authors: ProvableWorldModel maintainers
- Created: 2026-06-03
- Target milestone: v1.0

## Summary

This RFC locks the V0 headline proof statement P2 (`pwm.lewm.fixed_candidate_planning.v1`): given a committed quantized LeWorldModel predictor manifest, an initial latent history, a goal latent, and `S` fixed candidate action sequences, the proof verifies that **every** candidate was rolled out through the committed predictor, **every** final-latent MSE cost was computed as specified, and the selected candidate has minimum cost under a deterministic smallest-index tie-break. The decision this RFC makes permanent: the planner AIR proves all `S` rollouts, all `S` costs, and the full argmin relation; proving only the selected candidate (or a prover-supplied cost table without recomputation) is **unsound** and is rejected as a valid construction. The argmin is enforced with two range-checked families of constraints — `cost_s - selected_cost >= 0` for all `s` (minimality) and `cost_s - selected_cost - 1 >= 0` for all `s < selected_index` (strict tie-break) — both over centered M31 with explicit bounds. This RFC composes the components defined in [docs/rfcs/RFC-0008-rollout-air.md](RFC-0008-rollout-air.md) (Rollout_Q) and [docs/rfcs/RFC-0005-linear-matmul-and-requantization-components.md](RFC-0005-linear-matmul-and-requantization-components.md) (linear/requant), and adds the `cost` and `argmin` components plus the planner orchestration.

## Motivation

Planning is the user-facing value of LeWorldModel: select an action sequence whose predicted latent rollout lands closest to a goal latent. A proof that only attests to the chosen candidate's rollout leaves the prover free to score that candidate favorably and silently discard any cheaper candidate it does not want to reveal — the verifier has no way to know a better option existed. The founding analysis names this exact attack as a Critical risk ("Unsound planner claim: proving only selected rollout does not prove planning"; see [docs/feasibility-study.md](../feasibility-study.md) §13) and states the planner-soundness requirement directly: "For fixed-candidate planning, proving only the selected candidate is insufficient. The prover must prove costs for all candidates or prove a committed candidate-cost table plus a correct minimum argument" (§9.5). The P2 statement and its claim are given in §1.2 (Statement P2) and §14 (recommended V0 statement); the cost/argmin AIR sketch is §7.9; the MSE and comparison semantics are §5.6 and §5.5.

P2 is the first shippable deliverable (V0; see [docs/spec/00-overview.md#v0-statement](../spec/00-overview.md#v0-statement) and the statement tiers at [docs/spec/00-overview.md#scope-and-statement-tiers](../spec/00-overview.md#scope-and-statement-tiers)). It composes P0 (one predictor step), P1 (rollout), MSE cost, and argmin into one relation. This RFC is the load-bearing decision for the planner subsystem (`area:planner`) and the `cost`/`argmin` AIR components in `pwm-air`.

Concrete scenario the design must defeat: a prover holds candidates `c_0..c_{S-1}`, where `c_3` has the true minimum cost. The prover wants the verifier to accept `selected_index = 7`. Under a partial-proof scheme the prover proves only `c_7`'s rollout and asserts its cost is the minimum. Under this RFC the prover must commit and prove all `S` costs and the full minimality relation, so the constraint `cost_3 - selected_cost >= 0` fails (it would be negative) and the proof is unsatisfiable. The proof simply cannot be produced for a wrong selection.

## Goals

- Lock relation `pwm.lewm.fixed_candidate_planning.v1` (StatementType `P2FixedCandidatePlanning`) as the V0 deliverable, bound to its exact `relation_id` per [docs/spec/03-data-model.md#relation-id](../spec/03-data-model.md#relation-id).
- Require that the planner proves all `S` rollouts, all `S` MSE costs, and the complete argmin/tie-break relation in a single proof over a single public input. No partial-candidate construction is a valid prover.
- Define the `cost` AIR component (per-candidate MSE_Q over the final predicted latent vs the goal latent) with named columns, constraints, and accumulator/limb policy.
- Define the `argmin` AIR component with the two range-checked constraint families (minimality and strict smallest-index tie-break) and its binding to `selected_index`, `selected_cost`, and the public `ArgminWitness`.
- Define the planner orchestration: how `S` rollout instances, `S` cost instances, and one argmin instance share tensor memory and lookups, and how the per-candidate cost table is bound so no candidate can be omitted.
- Enumerate every planner-specific failure mode with the verifier's response, cross-referenced to [docs/spec/04-error-model.md#verifier-rejections](../spec/04-error-model.md#verifier-rejections).
- Specify accepting and rejecting tests sufficient for a contributor to implement and gate the component.

## Non-Goals

- Proving candidate **generation**. P2 takes the `S` candidate action sequences as committed inputs. Seeded sampling, clipping, top-k, and distribution updates are the CEM planner (P3), deferred to [docs/rfcs/RFC-0010-cem-planner-proof.md](RFC-0010-cem-planner-proof.md). This RFC does not attempt CEM and must not be read as doing so.
- Re-specifying the rollout recurrence, predictor block, nonlinear primitives, fixed-point arithmetic, or range/lookup infrastructure. Those are owned by [docs/rfcs/RFC-0008-rollout-air.md](RFC-0008-rollout-air.md), [docs/rfcs/RFC-0007-leworldmodel-predictor-air.md](RFC-0007-leworldmodel-predictor-air.md), [docs/rfcs/RFC-0006-nonlinear-primitive-components.md](RFC-0006-nonlinear-primitive-components.md), [docs/rfcs/RFC-0002-fixed-point-arithmetic-over-m31.md](RFC-0002-fixed-point-arithmetic-over-m31.md), and [docs/rfcs/RFC-0003-range-check-and-lookup-infrastructure.md](RFC-0003-range-check-and-lookup-infrastructure.md). This RFC consumes them.
- Batched/efficient `S`-candidate proving (shared-trace layout optimizations, candidate parallelism). V0 proves all candidates correctly; the efficiency rework is V1 (see [docs/spec/08-performance-budget.md#scaling](../spec/08-performance-budget.md#scaling)).
- Recursion/aggregation across per-candidate sub-proofs ([docs/rfcs/RFC-0012-recursive-aggregated-verification.md](RFC-0012-recursive-aggregated-verification.md), Future).
- Zero-knowledge. P2 is a succinct validity proof, not ZK; see [docs/spec/06-security.md#privacy-and-zk](../spec/06-security.md#privacy-and-zk).
- Proving the cost is meaningful in the real environment, or that the selected action is globally optimal in the world. The proof binds the arithmetic relation only; see [docs/spec/00-overview.md#non-goals](../spec/00-overview.md#non-goals).

## Proposed Design

### Relation and public surface

The proven relation is, for goal latent `g`, initial history `z_hist`, and candidate action sequences `a_0..a_{S-1}`:

```text
relation pwm.lewm.fixed_candidate_planning.v1:
  for s in 0..S-1:
      traj_s    == Rollout_Q(z_hist, a_s ; committed_weights)          # RFC-0008
      final_s   == last predicted latent of traj_s                     # vector in M31^latent_dim
      cost_s    == MSE_Q(final_s, g)                                   # this RFC, cost component
  selected_cost == cost_{selected_index}                              # selection binding
  for all s:        cost_s - selected_cost >= 0                       # minimality
  for all s < selected_index:  cost_s - selected_cost - 1 >= 0        # strict smallest-index tie-break
  0 <= selected_index < S                                             # index range
```

`final_s` is the final predicted latent of candidate `s`'s rollout. The number of rollout steps is the planning horizon `horizon = 5` with `action_block = 5` (V0 reference; verified against upstream le-wm at 2026-06-03 — `config/eval/pusht.yaml plan_config`). `latent_dim = 192`, `history_size = 3` (V0 reference configuration; verified against upstream le-wm at 2026-06-03). MSE_Q is the quantized sum-of-squared-differences over the `latent_dim` components, matching le-wm's `F.mse_loss` on the final predicted latent vs the goal latent. The proof binds the arithmetic, not the floating-point reduction; "as specified" means the manifest-declared fixed-point MSE_Q reduction, enforced bit-for-bit by the Python and Rust references and the AIR.

The public input uses the canonical `PublicInput` ([docs/spec/03-data-model.md#public-input](../spec/03-data-model.md#public-input), canonical signature in [docs/spec/01-architecture.md](../spec/01-architecture.md)). For P2:

```rust
// StatementType::P2FixedCandidatePlanning
PublicInput {
    relation_id,                       // == digest("pwm.lewm.fixed_candidate_planning.v1")
    model_commitment,                  // binds architecture + weights + shapes (RFC-0001)
    quantization_commitment,           // binds scales, rounding, clamp, activation tables (RFC-0002/0006)
    planner_config_commitment,         // binds S, horizon, action_block, MSE reduction, tie-break rule, cost bounds
    statement_type: P2FixedCandidatePlanning,
    latent_history_commitment: Some(..) | latent_history_public: Some(..),  // z_hist
    goal_latent_commitment:    Some(..) | goal_latent_public:    Some(..),  // g
    candidate_actions_commitment: Some(..) | candidate_actions_public: Some(..), // a_0..a_{S-1}
    claimed_output_commitment,         // commitment to per-candidate trajectories/costs table (see below)
    selected_index: Some(u32),         // i*
    selected_cost:  Some(BoundedInt),  // cost_{i*}, with declared [0, cost_max]
}
```

The witness uses the canonical `Witness` and `ArgminWitness` ([docs/spec/03-data-model.md#witness](../spec/03-data-model.md#witness)):

```rust
Witness {
    model_weights: Option<QuantizedWeights>, // None when committed-only
    latent_history: Tensor,                  // z_hist
    goal_latent:    Some(Tensor),            // g
    candidate_actions: Some(Tensor),         // a_0..a_{S-1}, shape [S, action_block, ...]
    action_embeddings: Tensor,               // ActionEncoder_Q outputs, per candidate
    predictor_activations: Vec<Tensor>,      // per-candidate, per-step predictor intermediates
    rollout_trajectory: Tensor,              // [S, horizon+history_size, latent_dim]
    costs: Some(Vec<BoundedInt>),            // cost_0..cost_{S-1}, len == S, each in [0, cost_max]
    argmin_witness: Some(ArgminWitness),
    range_witnesses: Vec<RangeWitness>,
    lookup_witnesses: Vec<LookupWitness>,
}

ArgminWitness {
    selected_index: u32,                     // i*, == PublicInput.selected_index
    diffs: Vec<BoundedInt>,                  // diffs[s] = cost_s - selected_cost (>= 0), len == S
}
```

INV-RFC0009-01 (completeness of the candidate set): `costs.len() == S` and `rollout_trajectory` carries exactly `S` trajectory slices, where `S == planner_config.num_candidates`, itself bound by `planner_config_commitment`. The cost component instantiates exactly one cost instance per trajectory slice and the argmin component consumes exactly `S` costs. There is no code path that proves a subset; `S` is read from the committed planner config, not from the witness.

### Component decomposition

The planner is three AIR components composed over shared tensor memory ([docs/rfcs/RFC-0004-tensor-memory-and-wiring-air.md](RFC-0004-tensor-memory-and-wiring-air.md)) plus the rollout component:

| Component | Crate / module | Responsibility |
| --- | --- | --- |
| `rollout` (×S instances) | `pwm-air/components/rollout.rs` | Per-candidate autoregressive rollout, producing `final_s` (RFC-0008). |
| `cost` (×S instances) | `pwm-air/components/cost.rs` | Per-candidate MSE_Q(`final_s`, `g`) -> `cost_s`. This RFC. |
| `argmin` (×1 instance) | `pwm-air/components/argmin.rs` | Selection binding + minimality + strict tie-break over `cost_0..cost_{S-1}`. This RFC. |
| planner orchestration | `pwm-prover/prove_planning.rs` | Builds the `S` rollout/cost instances and the single argmin instance, wires the cost table, derives challenges in canonical order (RFC-0014). This RFC. |

Trace model: rollout and cost contribute main-trace columns and interaction (LogUp) columns; the candidate-cost table is realized as a preprocessed/interaction binding so the argmin component reads exactly the same `cost_s` values the cost component wrote. See [docs/spec/01-architecture.md#trace-model](../spec/01-architecture.md#trace-model) and [docs/spec/01-architecture.md#component-model](../spec/01-architecture.md#component-model). The direct-AIR strategy ([docs/spec/01-architecture.md#air-strategy](../spec/01-architecture.md#air-strategy)) applies: cost is a dense reduction (use a tensorized accumulator layout); argmin is a small comparison gadget over `S` rows.

### Cost component (`cost.rs`)

For candidate `s`, with final predicted latent `z = final_s` and goal latent `g`, both length `d = latent_dim`:

```text
MSE_Q(z, g):
  cost_acc_0 = 0
  for j = 0..d-1:
      diff_j        = z_j - g_j
      sq_j          = diff_j * diff_j
      cost_acc_{j+1}= cost_acc_j + sq_j
  cost_s = Requantize(cost_acc_d, cost_shift, NoZeroPoint, [0, cost_max])   # manifest-declared reduction
```

Main-trace columns (one row per `(s, j)` reduction step; `candidate_id = s`):

```text
candidate_id            # s, 0..S-1
elem_idx                # j, 0..d-1
z                       # final_s[j]
g                       # goal[j]
diff                    # z - g
sq                      # diff * diff
cost_acc                # running sum entering this row
cost_acc_next           # cost_acc + sq
is_first                # j == 0 selector (preprocessed)
is_last                 # j == d-1 selector (preprocessed)
cost_q                  # requantization quotient at is_last
cost_rem                # requantization remainder at is_last
cost_s                  # final per-candidate cost, broadcast/exposed to the cost table
```

Constraints (all field-level, with explicit ranges per RFC-0002):

```text
C-COST-1  is_first * (cost_acc - 0) = 0
C-COST-2  diff = z - g
C-COST-3  sq = diff * diff
C-COST-4  cost_acc_next = cost_acc + sq
C-COST-5  is_last * (cost_acc_next - (cost_q * 2^cost_shift + cost_rem)) = 0
C-COST-6  0 <= cost_rem < 2^cost_shift                          # range-checked (RFC-0003)
C-COST-7  cost_s = RoundAndClamp(cost_q, cost_rem, cost_shift, manifest_rounding, [0, cost_max])
```

Lookups / reads (via tensor memory, RFC-0004):

```text
z   read from rollout_trajectory[s, final_step, j]   (TensorCell, RFC-0008 output)
g   read from goal_latent[j]                          (TensorCell, committed input)
cost_s written to the candidate-cost table cell (candidate_id = s)
diff, sq, cost_acc, cost_q, cost_rem, cost_s range-checked
```

Range/limb policy (INV-RFC0009-02, accumulator safety): `diff_j = z_j - g_j` lies in `[-(B_z+B_g), B_z+B_g]` where `B_z`, `B_g` are the declared latent bounds. `sq_j` is nonnegative and bounded by `(B_z+B_g)^2`. The running `cost_acc` over `d = 192` terms is bounded by `192 * (B_z+B_g)^2`. For int8 latents (`B = 127`) `sq_j <= 254^2 = 64516` and the sum is `<= 12,387,072 < 2^31 - 1`, so int8-latent MSE fits in one signed M31 accumulator. For int16 latents the accumulator exceeds the safe signed M31 interval; the manifest then declares limb accumulation `cost_acc = cost_acc_lo + 2^k * cost_acc_hi` with each limb range-checked (RFC-0002 §limb decomposition). The active path (single accumulator vs limb) is selected by the manifest's declared latent dtype and is bound by `quantization_commitment`; the AIR has no float fallback. `cost_max` is declared in the planner config and bound by `planner_config_commitment`.

### Argmin component (`argmin.rs`)

The argmin component consumes the candidate-cost table (`cost_0..cost_{S-1}`), `selected_index = i*`, and `selected_cost`. One row per candidate `s`:

```text
candidate_id            # s, 0..S-1
cost                    # cost_s, read from the candidate-cost table
selected_cost           # broadcast public value (same on every row)
selected_index          # broadcast public value i* (same on every row)
is_selected             # boolean selector: 1 iff s == i*
is_before_selected      # boolean selector: 1 iff s < i*
diff_ge                 # cost_s - selected_cost   (>= 0)   == ArgminWitness.diffs[s]
diff_gt                 # cost_s - selected_cost - 1 (>= 0 only where is_before_selected)
```

Constraints:

```text
C-ARG-1  is_selected, is_before_selected are boolean: x*(x-1) = 0
C-ARG-2  Σ_s is_selected = 1                                   # exactly one selected row
C-ARG-3  is_selected * (cost - selected_cost) = 0              # selected_cost == cost_{i*}  (selection binding)
C-ARG-4  diff_ge = cost - selected_cost
C-ARG-5  0 <= diff_ge                                          # minimality, range-checked for ALL s (RFC-0003)
C-ARG-6  is_before_selected * (diff_gt - (diff_ge - 1)) = 0    # diff_gt = cost - selected_cost - 1
C-ARG-7  is_before_selected * (1 - boundedness(diff_gt)) = 0   # 0 <= diff_gt, range-checked for s < i*  (strict tie-break)
C-ARG-8  consistency of is_before_selected with selected_index # see index encoding below
C-ARG-9  0 <= selected_index < S                               # index range, range-checked
```

INV-RFC0009-03 (minimality): for all `s`, `cost_s - selected_cost >= 0` (C-ARG-5). This is the core soundness property: no candidate has strictly lower cost than the selected one. The witness `ArgminWitness.diffs[s]` carries these differences and they are range-checked into `[0, cost_max]` (a negative integer would wrap to a large M31 value and fail the range check — see INV-RFC0009-05).

INV-RFC0009-04 (deterministic smallest-index tie-break): for all `s < selected_index`, `cost_s - selected_cost - 1 >= 0`, i.e. `cost_s > selected_cost` strictly. Combined with INV-RFC0009-03 for `s > selected_index` (`cost_s >= selected_cost`, ties allowed), the unique satisfying `selected_index` is the smallest index attaining the minimum. The strict-greater is encoded as `cost_s - selected_cost - 1 >= 0` because strict comparison is not native in a finite field (feasibility study §5.5, §7.9). This is the decided tie-break rule (binding contract §3) and is bound by `planner_config_commitment`.

Index encoding for C-ARG-8: `selected_index`, `is_selected`, and `is_before_selected` are tied through a preprocessed monotone position column `pos = s`. The constraint set enforces `is_before_selected = 1 iff pos < selected_index` and `is_selected = 1 iff pos == selected_index` via `selected_index = Σ_s is_before_selected` together with `is_selected` placed exactly at the first non-before row. Concretely: `is_before_selected` is non-increasing across `s` (a prefix of ones), `is_selected` marks the first zero of `is_before_selected`, and `selected_index = Σ_s is_before_selected`. These three relations make the selector pattern unique for a given `selected_index`, so the prover cannot mislabel rows to dodge C-ARG-7.

INV-RFC0009-05 (no field-wrap escape): every value interpreted as a signed integer in the cost and argmin components — `diff`, `sq`, `cost_acc`, `cost_s`, `diff_ge`, `diff_gt`, `selected_index`, `selected_cost` — is range-checked into its declared bound (RFC-0003). Because M31 wraps and integer inference must not (binding contract §3), a "negative" difference encoded as a large field element fails its range check, which is exactly what blocks the partial-proof / favorable-candidate attack.

### Candidate-cost table binding (the anti-partial-proof mechanism)

The single mechanism that makes "prove all candidates" enforceable rather than merely requested: the cost component **writes** each `cost_s` into a candidate-cost table cell keyed by `candidate_id = s`, and the argmin component **reads** `cost_s` from that same table. The table is a TensorCell relation (RFC-0004) reconciled by the LogUp/permutation multiset argument: every read in argmin must match a unique write from cost, and every cost write must be produced by a rollout whose `final_s` was itself produced by the rollout recurrence (RFC-0008) over `a_s` from the committed `candidate_actions`. There is no path for the argmin component to read a cost that was not produced by a full rollout + cost of that candidate. INV-RFC0009-06 (table closure): the multiset of cost-table writes equals the multiset of argmin cost-table reads, with cardinality exactly `S`; a missing candidate makes the multiset argument's claimed sum inconsistent and the proof unverifiable. `claimed_output_commitment` binds the full per-candidate trajectory/cost table so the verifier can re-derive it from declared outputs.

### Data flow and lifecycle

```text
manifest + planner_config  -> validate model_commitment, quantization_commitment, planner_config_commitment
candidate_actions (committed) -> ActionEncoder_Q -> action_embeddings (per candidate)         [RFC-0007]
for s in 0..S-1:
   z_hist, a_s -> Rollout_Q -> trajectory_s -> final_s                                        [RFC-0008]
   final_s, g  -> MSE_Q     -> cost_s   (written to candidate-cost table)                     [this RFC, cost]
cost_0..cost_{S-1}, i*, selected_cost -> argmin (minimality + tie-break)                       [this RFC, argmin]
commit traces -> derive Fiat-Shamir challenges in canonical order -> interaction traces -> proof  [RFC-0014]
```

The reference fixed-point inference (Python and Rust) computes `cost_s` for all `s` and the argmin/tie-break before any trace is built, so the witness is fully determined by `(manifest, z_hist, g, a_0..a_{S-1})`. Determinism: `selected_index` and `selected_cost` are a deterministic function of the inputs and the committed tie-break rule (INV-RFC0009-04); the prover does not choose them freely. See [docs/spec/06-security.md#soundness-requirements](../spec/06-security.md#soundness-requirements).

### CLI and artifact

The prover exposes `prove_planning` (CLI shape and `ProofArtifact` bundle owned by [docs/rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md](RFC-0016-cli-artifact-bundle-and-reproducibility.md); see [docs/spec/02-public-api.md#cli](../spec/02-public-api.md#cli) and [docs/spec/02-public-api.md#artifact-formats](../spec/02-public-api.md#artifact-formats)):

```text
pwm-prover prove-planning \
    --manifest <model_manifest.yaml> \
    --weights <quantized_weights> \
    --latent-history <z_hist> --goal-latent <g> \
    --candidates <candidate_actions> \
    --out <artifact.pwm>
# emits ProofArtifact { artifact_version, public_input (P2FixedCandidatePlanning),
#                       proof, claimed_outputs: per-candidate trajectories + costs }
```

Verification is the canonical entry (signature from binding contract §6.5):

```rust
pub fn verify(artifact: &ProofArtifact) -> Result<(), VerifyError>;
```

The verifier never runs PyTorch or the reference inference; it checks the relation, the public-input digest (RFC-0014), `0 <= selected_index < S`, and tensor shapes, then accepts iff the Stwo proof verifies. See [docs/spec/02-public-api.md#rust-public-api](../spec/02-public-api.md#rust-public-api).

### Failure modes and system response

| ID | Failure mode | Where detected | System response |
| --- | --- | --- | --- |
| FM-PLAN-01 | A candidate with strictly lower cost than `selected_cost` exists | C-ARG-5 (INV-RFC0009-03) | Proof unsatisfiable at prove time; if a forged witness is submitted, verifier returns `VerifyError::ConstraintUnsatisfied` ([docs/spec/04-error-model.md#verifier-rejections](../spec/04-error-model.md#verifier-rejections)). |
| FM-PLAN-02 | Tie-break violated: some `s < i*` has `cost_s == selected_cost` | C-ARG-7 (INV-RFC0009-04) | Range check on `diff_gt` fails; `VerifyError::ConstraintUnsatisfied`. |
| FM-PLAN-03 | `selected_cost != cost_{i*}` | C-ARG-3 | `VerifyError::ConstraintUnsatisfied`. |
| FM-PLAN-04 | `selected_index` out of `[0, S)` | C-ARG-9 / verifier shape check | `VerifyError::PublicInputOutOfRange`. |
| FM-PLAN-05 | A candidate cost omitted (fewer than `S` costs proven) | INV-RFC0009-06 multiset closure | Interaction claimed-sum inconsistent; `VerifyError::InteractionMismatch`. |
| FM-PLAN-06 | argmin reads a cost not produced by a full rollout+cost | candidate-cost table multiset (RFC-0004) | `VerifyError::InteractionMismatch`. |
| FM-PLAN-07 | A `cost_s` mutated between cost and argmin | table multiset (RFC-0004) | `VerifyError::InteractionMismatch`. |
| FM-PLAN-08 | MSE accumulator overflows declared bound (limb path not taken) | C-COST-6 / range check (INV-RFC0009-02) | Prove-time `TraceError::AccumulatorOverflow` ([docs/spec/04-error-model.md#failure-modes](../spec/04-error-model.md#failure-modes)); never silently wraps (overflow policy = reject). |
| FM-PLAN-09 | Goal latent or candidate set changed vs commitment | public-input digest (RFC-0014) | `VerifyError::PublicInputDigestMismatch`. |
| FM-PLAN-10 | Planner config (S, horizon, tie-break, cost_max) changed | `planner_config_commitment` mismatch | `VerifyError::CommitmentMismatch`. |
| FM-PLAN-11 | Proof submitted under wrong `relation_id` (e.g. P1 proof claimed as P2) | verifier relation_id check | `VerifyError::UnknownOrWrongRelation` ([docs/spec/06-security.md#binding-requirements](../spec/06-security.md#binding-requirements)). |

## Alternatives Considered

**A1. Prove only the selected candidate's rollout and cost; assert it is the minimum.** Smallest trace (one rollout instead of `S`), and the cost would scale `O(1)` in candidates. Rejected: it is unsound. The verifier has no constraint witnessing that no cheaper candidate exists; the prover can compute all costs off-circuit, pick a non-minimal candidate, and prove only that one. The feasibility study flags this precisely (§9.5; §13 "Unsound planner claim", Critical). The decision this RFC locks is exactly the rejection of A1.

**A2. Prover supplies a committed candidate-cost table; argmin runs over the table without recomputing costs in-circuit.** Smaller than recomputing `S` MSE reductions in the AIR — the argmin component would be `O(S)` comparisons and the rollouts/costs would be attested only by a commitment. Rejected: a bare commitment to costs is prover-chosen unless every `cost_s` is constrained to equal `MSE_Q(final_s, g)` for the `final_s` actually produced by the committed predictor over the committed candidate `a_s`. Binding a table the prover fabricated does not make its entries correct. Our candidate-cost table (the §"Candidate-cost table binding" mechanism) is the sound form of A2: the table cells are *outputs of in-circuit cost components fed by in-circuit rollouts*, reconciled by the multiset argument (INV-RFC0009-06), not free witness values. The feasibility study admits A2 only in this constrained form: "prove a committed candidate-cost table plus a correct minimum argument" (§9.5).

**A3. Decompose into `S` independent per-candidate proofs plus one aggregation proof over cost commitments.** Each candidate proof is small and they parallelize; aggregation handles the argmin. Rejected for V0: it requires recursive/aggregated verification, which is deferred to [docs/rfcs/RFC-0012-recursive-aggregated-verification.md](RFC-0012-recursive-aggregated-verification.md) (Future) and adds a recursive verifier surface and a canonical cross-proof public-input schema we do not want in the first sound release. V0 proves P2 as one monolithic relation; A3 is the V1+ efficiency/scale path, not a soundness alternative.

**A4. Encode the strict tie-break as `cost_s - selected_cost > 0` directly.** Rejected: strict greater-than is not a native finite-field relation and has no single low-degree constraint; in centered M31 a strict comparison must be reduced to a nonnegativity range check on a shifted difference. `cost_s - selected_cost - 1 >= 0` is that reduction and is the form the binding contract §3 and feasibility study §7.9 fix. A4 would either be unsound (no real strict check) or silently reintroduce A4's own range check, so we adopt the explicit `- 1` form.

## Drawbacks

- Proving cost scales `O(S * horizon)` predictor evaluations plus `O(S * latent_dim)` MSE work plus `O(S)` argmin rows. The dominant term is the `S` rollouts; the cost/argmin overhead this RFC adds is small relative to rollout, but the all-candidates requirement means there is no shortcut for large `S`. Quantified targets and the candidates×horizon scaling curve live in [docs/spec/08-performance-budget.md#scaling](../spec/08-performance-budget.md#scaling).
- The single-proof monolith trace can be large for big `S`; aggregation that would let candidates be proven and verified separately is deferred (A3 / RFC-0012). V0 accepts the larger trace in exchange for not introducing a recursive verifier.
- MSE_Q is the manifest-declared fixed-point reduction, not bit-identical to PyTorch `F.mse_loss` in float. This is intentional (the proof binds the quantized relation, not float equivalence; [docs/spec/00-overview.md#non-goals](../spec/00-overview.md#non-goals)) but means "cost" is the quantized cost, and downstream consumers must read it as such.
- The tie-break is a policy choice (smallest index). If a future statement needs a different tie-break it must mint a new `relation_id` (the tie-break is bound by `planner_config_commitment`), per [docs/spec/09-release-and-versioning.md#relation-versioning](../spec/09-release-and-versioning.md#relation-versioning).

## Migration / Rollout

- This is the V0 headline statement; there is no prior P2 relation to migrate from. The relation ships as `pwm.lewm.fixed_candidate_planning.v1` and its id is immutable ([docs/spec/09-release-and-versioning.md#relation-versioning](../spec/09-release-and-versioning.md#relation-versioning)).
- Component gating: the `cost` and `argmin` components land behind no runtime feature flag in the verifier (the verifier surface is fixed per relation), but during development they sit behind the workspace dev-feature `planner-p2` in `pwm-air`/`pwm-prover` so they can be merged before the rollout component is fully landed without affecting P0/P1 proving paths.
- Schema versioning: `PublicInput`, `Witness`, and `ArgminWitness` are the canonical V0 schemas ([docs/spec/03-data-model.md#witness](../spec/03-data-model.md#witness)); the `ProofArtifact.artifact_version` is bumped by [docs/rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md](RFC-0016-cli-artifact-bundle-and-reproducibility.md) and the manifest `planner` block (S, horizon, action_block, MSE reduction, tie-break rule, cost_max) is added to the manifest schema in [docs/rfcs/RFC-0001-model-manifest-and-export-pipeline.md](RFC-0001-model-manifest-and-export-pipeline.md), bound by `planner_config_commitment`.
- Any change to the cost reduction, the argmin minimality/tie-break encoding, the candidate-cost table binding, or the bound `cost_max` is a semantic change and mints `pwm.lewm.fixed_candidate_planning.v2`; v1 proofs continue to verify only under the v1 verifier path. No in-place mutation of the v1 relation is permitted.
- Deprecation: when V1 introduces batched/aggregated P2, the v1 monolithic relation remains supported per the support window in [docs/spec/09-release-and-versioning.md#deprecation](../spec/09-release-and-versioning.md#deprecation); deprecation is announced in the changelog ([docs/spec/09-release-and-versioning.md#changelog](../spec/09-release-and-versioning.md#changelog)) at least one minor version before removal.

## Testing Strategy

Cross-references: test layers and gates in [docs/spec/07-testing-strategy.md](../spec/07-testing-strategy.md); rejection taxonomy in [docs/spec/04-error-model.md#verifier-rejections](../spec/04-error-model.md#verifier-rejections); the "no component ships without both accepting and rejecting tests" rule from [docs/rfcs/RFC-0013-testing-fuzzing-and-audit-strategy.md](RFC-0013-testing-fuzzing-and-audit-strategy.md).

Accepting tests:

- `cost_mse_golden_int8`: golden vector — for known `final_s`, `g` with int8 latents, the Rust cost component output equals the Python fixed-point reference `cost_s` bit-for-bit ([docs/spec/07-testing-strategy.md#golden-vectors](../spec/07-testing-strategy.md#golden-vectors), [docs/spec/07-testing-strategy.md#differential-tests](../spec/07-testing-strategy.md#differential-tests)).
- `cost_mse_golden_int16_limb`: same, with int16 latents exercising the limb-accumulator path (INV-RFC0009-02).
- `argmin_unique_min_accepts`: `S` distinct costs, true minimum at index `k`; proof verifies with `selected_index = k`.
- `argmin_tie_smallest_index_accepts`: minimum cost attained at indices `{2, 5}`; proof verifies only with `selected_index = 2` (INV-RFC0009-04).
- `planner_p2_end_to_end_accepts`: full P2 over `S` candidates with `horizon = 5`, `action_block = 5`; `prove_planning` then `verify` returns `Ok(())`, and `claimed_outputs` per-candidate costs match the reference.
- `planner_p2_reference_parity`: Rust trace-builder costs/argmin equal the Python fixed-point planner for randomized `(z_hist, g, candidates)` across seeds ([docs/spec/07-testing-strategy.md#differential-tests](../spec/07-testing-strategy.md#differential-tests)).

Rejecting (negative) tests — each must produce the listed `VerifyError`:

- `reject_lower_cost_candidate_exists` (FM-PLAN-01): forge `selected_index` to a non-minimal candidate -> `ConstraintUnsatisfied` (C-ARG-5). This is the headline soundness test.
- `reject_partial_only_selected_candidate` (FM-PLAN-05): construct a witness proving only the selected candidate's rollout+cost, omitting the others -> `InteractionMismatch` (INV-RFC0009-06). Directly tests the anti-partial-proof mechanism.
- `reject_tie_break_violation` (FM-PLAN-02): minimum attained at indices `{2, 5}`, claim `selected_index = 5` -> `ConstraintUnsatisfied` (C-ARG-7).
- `reject_selected_cost_mismatch` (FM-PLAN-03): `selected_cost != cost_{i*}` -> `ConstraintUnsatisfied` (C-ARG-3).
- `reject_selected_index_out_of_range` (FM-PLAN-04): `selected_index = S` -> `PublicInputOutOfRange` (C-ARG-9).
- `reject_mutated_candidate_cost` (FM-PLAN-07): mutate one `cost_s` after the cost component -> `InteractionMismatch`.
- `reject_argmin_cost_not_from_rollout` (FM-PLAN-06): feed argmin a fabricated cost cell -> `InteractionMismatch`.
- `reject_mse_accumulator_overflow` (FM-PLAN-08): int16 latents with the single-accumulator (non-limb) path declared -> prove-time `AccumulatorOverflow` ([docs/spec/04-error-model.md#failure-modes](../spec/04-error-model.md#failure-modes)).
- `reject_changed_goal_latent` (FM-PLAN-09): alter `g` vs its commitment -> `PublicInputDigestMismatch`.
- `reject_changed_planner_config` (FM-PLAN-10): change `S` or the tie-break rule vs `planner_config_commitment` -> `CommitmentMismatch`.
- `reject_p1_proof_as_p2` (FM-PLAN-11): submit a `pwm.lewm.rollout.v1` proof under the P2 relation_id -> `UnknownOrWrongRelation`.

Constraint mutation tests ([docs/spec/07-testing-strategy.md#mutation-tests](../spec/07-testing-strategy.md#mutation-tests)): mutating each of C-COST-2..C-COST-7 and C-ARG-1..C-ARG-9 (e.g. flip C-ARG-5 from `>=` to `<=`, drop the `-1` in C-ARG-6) must cause at least one existing test to fail; a surviving mutant is a test-suite gap and blocks CI ([docs/spec/07-testing-strategy.md#ci-gates](../spec/07-testing-strategy.md#ci-gates)).

## Open Questions

- OPEN QUESTION (owner: `area:planner` maintainer; resolution path: [docs/spec/08-performance-budget.md#scaling](../spec/08-performance-budget.md#scaling) measurement before v1.0 freeze): the maximum `S` provable in one monolithic trace within the V0 memory target. If measured `S_max` is below the eval-config candidate count, the resolution is to gate large-`S` planning on the V1 batched-trace rework, not to weaken the all-candidates requirement.
- OPEN QUESTION (owner: `area:core` maintainer; resolution path: [docs/rfcs/RFC-0002-fixed-point-arithmetic-over-m31.md](RFC-0002-fixed-point-arithmetic-over-m31.md)): the exact `cost_shift` and `cost_max` defaults for the V0 reference latent dtype, to be fixed in the manifest `planner` block and golden vectors. Until fixed, `cost_int8_default` (single accumulator) is used in tests with `cost_max = 192 * 254^2`.

## References

- Founding analysis: [docs/feasibility-study.md](../feasibility-study.md) §1.2 (Statement P2), §5.5 (comparisons), §5.6 (MSE cost), §7.9 (cost and argmin AIR), §9.5 (planner soundness), §13 (risk register), §14 (recommended V0 statement).
- Binding contract §3 (argmin tie-break decision), §4 (statement tiers, relation_id scheme, milestones), §6 (canonical `PublicInput`, `Witness`, `ArgminWitness`, `BoundedInt`, `verify`), §9b (verified le-wm dims, `horizon=5`/`action_block=5`, GELU/SiLU split, MSE on final latent).
- Overview/scope: [docs/spec/00-overview.md#v0-statement](../spec/00-overview.md#v0-statement), [docs/spec/00-overview.md#scope-and-statement-tiers](../spec/00-overview.md#scope-and-statement-tiers), [docs/spec/00-overview.md#non-goals](../spec/00-overview.md#non-goals).
- Architecture: [docs/spec/01-architecture.md#air-strategy](../spec/01-architecture.md#air-strategy), [docs/spec/01-architecture.md#component-model](../spec/01-architecture.md#component-model), [docs/spec/01-architecture.md#trace-model](../spec/01-architecture.md#trace-model).
- Public API and artifacts: [docs/spec/02-public-api.md#rust-public-api](../spec/02-public-api.md#rust-public-api), [docs/spec/02-public-api.md#cli](../spec/02-public-api.md#cli), [docs/spec/02-public-api.md#artifact-formats](../spec/02-public-api.md#artifact-formats).
- Data model: [docs/spec/03-data-model.md#public-input](../spec/03-data-model.md#public-input), [docs/spec/03-data-model.md#witness](../spec/03-data-model.md#witness), [docs/spec/03-data-model.md#relation-id](../spec/03-data-model.md#relation-id).
- Errors: [docs/spec/04-error-model.md#verifier-rejections](../spec/04-error-model.md#verifier-rejections), [docs/spec/04-error-model.md#failure-modes](../spec/04-error-model.md#failure-modes).
- Security: [docs/spec/06-security.md#soundness-requirements](../spec/06-security.md#soundness-requirements), [docs/spec/06-security.md#binding-requirements](../spec/06-security.md#binding-requirements), [docs/spec/06-security.md#privacy-and-zk](../spec/06-security.md#privacy-and-zk).
- Testing: [docs/spec/07-testing-strategy.md#golden-vectors](../spec/07-testing-strategy.md#golden-vectors), [docs/spec/07-testing-strategy.md#negative-tests](../spec/07-testing-strategy.md#negative-tests), [docs/spec/07-testing-strategy.md#differential-tests](../spec/07-testing-strategy.md#differential-tests), [docs/spec/07-testing-strategy.md#mutation-tests](../spec/07-testing-strategy.md#mutation-tests), [docs/spec/07-testing-strategy.md#ci-gates](../spec/07-testing-strategy.md#ci-gates).
- Performance: [docs/spec/08-performance-budget.md#scaling](../spec/08-performance-budget.md#scaling).
- Release: [docs/spec/09-release-and-versioning.md#relation-versioning](../spec/09-release-and-versioning.md#relation-versioning), [docs/spec/09-release-and-versioning.md#deprecation](../spec/09-release-and-versioning.md#deprecation), [docs/spec/09-release-and-versioning.md#changelog](../spec/09-release-and-versioning.md#changelog).
- Composed/related RFCs: [docs/rfcs/RFC-0002-fixed-point-arithmetic-over-m31.md](RFC-0002-fixed-point-arithmetic-over-m31.md), [docs/rfcs/RFC-0003-range-check-and-lookup-infrastructure.md](RFC-0003-range-check-and-lookup-infrastructure.md), [docs/rfcs/RFC-0004-tensor-memory-and-wiring-air.md](RFC-0004-tensor-memory-and-wiring-air.md), [docs/rfcs/RFC-0005-linear-matmul-and-requantization-components.md](RFC-0005-linear-matmul-and-requantization-components.md), [docs/rfcs/RFC-0006-nonlinear-primitive-components.md](RFC-0006-nonlinear-primitive-components.md), [docs/rfcs/RFC-0007-leworldmodel-predictor-air.md](RFC-0007-leworldmodel-predictor-air.md), [docs/rfcs/RFC-0008-rollout-air.md](RFC-0008-rollout-air.md), [docs/rfcs/RFC-0010-cem-planner-proof.md](RFC-0010-cem-planner-proof.md), [docs/rfcs/RFC-0012-recursive-aggregated-verification.md](RFC-0012-recursive-aggregated-verification.md), [docs/rfcs/RFC-0013-testing-fuzzing-and-audit-strategy.md](RFC-0013-testing-fuzzing-and-audit-strategy.md), [docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md](RFC-0014-canonical-serialization-and-transcript.md), [docs/rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md](RFC-0016-cli-artifact-bundle-and-reproducibility.md).
- Prior art: feasibility-study citations to le-wm `jepa.py` rollout/MSE and `config/eval/pusht.yaml` plan_config (verified against upstream at 2026-06-03).
