# ProvableWorldModel — Specification (CommitLLM × LeWorldModel)

Status: living specification for the post‑pivot architecture. Companion to
[roadmap.md](roadmap.md) (plan) and [backlog.md](backlog.md) (work). Where this
file and an older `docs/spec/` document disagree, **this file governs** for the
commit‑and‑audit design; the legacy STARK corpus is archived under
`docs/legacy-stark/`.

---

## 1. Thesis and scope

ProvableWorldModel produces a **commit‑and‑audit proof** that a committed,
quantized LeWorldModel was executed exactly as specified on given inputs. The
prover runs deterministic integer fixed‑point inference and commits to the
execution trace; a CPU verifier audits the trace with **Freivalds** checks for
the large linear layers and **exact integer re‑execution** for everything else,
all bound by Merkle commitments and a Fiat‑Shamir transcript.

The proof attests an **exact arithmetic relation**, not the full PyTorch program,
not physical truth of predictions, not zero‑knowledge. Predictor bundles also
carry an offline float reference output plus a measured tolerance; the demo CLI
checks the verified integer output against that bound, but the no_std verifier
remains float-free and verifies the integer statement only.

### 1.1 Statement tiers (unchanged from pre‑pivot)

| Tier | `relation_id` | Claim |
|---|---|---|
| **P0** | `pwm.lewm.predictor_step.v1` | `z_next = PredProj_Q(ARPredictor_Q(z_hist, ActEnc_Q(actions)))` over the committed quantized graph. |
| **P1** | `pwm.lewm.rollout.v1` | Autoregressive rollout: each step `z_{t+1} = Predict_Q(window(z), window(a))`; trajectory consistent with the recurrence. |
| **P2 (V0)** | `pwm.lewm.fixed_candidate_planning.v1` | For all `S` candidates: `traj_s = Rollout_Q(...)`, `cost_s = MSE_Q(final_s, goal)`, `selected = argmin_s cost_s` with smallest‑index tie‑break. |
| P3 | `pwm.lewm.cem.v1` | Full CEM planner (deferred, V2). |
| P4 | `pwm.lewm.pixel_to_plan.v1` | Encoder (ViT) + predictor + planner end‑to‑end (deferred, V3). |

### 1.2 Non‑goals (V0)

Full floating‑point/bf16/GPU equivalence beyond the predictor bundle's offline
float-reference tolerance; physical‑truth claims; zero‑knowledge;
the CEM sampling loop (P3); the pixel encoder (P4); training, SIGReg, BatchNorm
*updates*, dataloading, the config framework. Proving only the winning
candidate (unsound — all candidates must be scored). Off‑trace nonlinearities
(every value the verifier checks must be derived from committed data).

---

## 2. The model: `QuantizedLeWM-v1`

The V0 reference is the le‑wm predictor at: `latent_dim = 192`, `history_size = 3`,
predictor `depth = 6`, `heads = 16`, `dim_head = 64`, `mlp_dim = 2048`,
`num_preds = 1` (verified against upstream `module.py`/`jepa.py`, 2026‑06‑05).
The encoder (ViT‑Tiny/14) and projector are **P4‑deferred**; V0 takes latents as
inputs. The V0 proven subgraph is `action_encoder → predictor → pred_proj`,
which is bit‑deterministic in `eval()` (empirically verified, max‑abs‑diff 0.0).

### 2.1 Operator graph (the exported `ops[]`)

```text
ActEnc_Q(actions):              # le-wm Embedder, per history position
  u   = Linear_Q(action_in)                    # conv1d(k=1) folded → linear
  g   = SiLU_Q(u)                              # silu_table
  a_emb = Linear_Q(g)                          # → [H, D]

ARPredictor_Q(z_hist, a_emb):
  x = z_hist + pos_emb                          # pos_emb committed (preprocessed)
  c = a_emb                                     # AdaLN conditioning
  for blk in 0..5: x = ConditionalBlock_Q[blk](x, c)
  x = LayerNorm_Q(x)                            # final norm
  return last_position(x)                       # [D]

ConditionalBlock_Q(x, c):                       # AdaLN-zero (DiT-style)
  (sh_a, sc_a, g_a, sh_m, sc_m, g_m) = chunk6(Linear_Q(SiLU_Q(c)))   # adaln.mod
  h = Modulate_Q(LayerNorm_Q(x, affine=false), sh_a, sc_a)
  x = x + Gate_Q(g_a, Attention_Q(h))           # causal SDPA, residual
  h = Modulate_Q(LayerNorm_Q(x, affine=false), sh_m, sc_m)
  x = x + Gate_Q(g_m, FFN_Q(h))                 # GELU FFN, residual

Attention_Q(x):                                 # heads=16, dim_head=64, causal
  Q,K,V = Linear_Q(x, Wqkv) -> [S, 16, 64]      # single fused qkv, bias=false
  score = Requantize_Q(QKᵀ, 1/sqrt(64)); causal-mask
  prob  = Softmax_Q(score)                       # softmax_table per row
  out   = prob · V
  return Linear_Q(reshape(out), Wproj)           # 1024 -> 192

FFN_Q(x):
  u = Linear_Q(x, fc1) -> [S, 2048]; g = GELU_Q(u); return Linear_Q(g, fc2) -> [S, 192]

PredProj_Q(z):                                   # le-wm MLP: Lin→BN→GELU→Lin
  return Linear_Q( GELU_Q( Linear_Q(z, fc1_bn_folded) ), fc2 )
```

### 2.2 Activations (non‑uniform — bound per op)

- **Predictor FFN** uses **GELU**; **`pred_proj`** uses **GELU**.
- **AdaLN modulation** and the **action `Embedder`** use **SiLU**.
- Each op binds its own committed table (`gelu_table_v1`, `silu_table_v1`,
  `softmax_table_v1`, `invsqrt_table_v1`); cross‑binding is rejected at trace
  build (`ActivationTableMismatch`).

### 2.3 BatchNorm folding

`pred_proj` (and the deferred `projector`) are le‑wm `MLP`s with a
`BatchNorm1d` whose running stats are **frozen at inference**. The exporter
**folds BatchNorm into the preceding `Linear`** (standard affine fusion); the
folded weights/biases are what the manifest commits. The running stats are
inputs to the fold and are recorded in the export provenance, not re‑evaluated
at prove/verify time. (BatchNorm *updates* are training‑only and out of scope.)

---

## 3. Number representation (carried over from RFC‑0002)

Exactly the pre‑pivot fixed‑point reference — **no floating point anywhere in
prove/verify**:

- Every value is a `BoundedInt` with a declared inclusive range `[lo, hi]`;
  arithmetic **never wraps** (`OverflowPolicy::Reject`).
- Weights int8; activations int8 or int16 (declared per tensor); accumulators
  int32 (bounded; limb‑decomposed if a bound escapes the audit field's signed
  range); biases int32; scales powers‑of‑two where possible.
- `requantize(n, r, zero_point, c_lo, c_hi, mode)` = arithmetic right shift by
  `r` with exact quotient/remainder split, rounding (`NearestTiesToEven`
  default / `TruncateTowardZero`), then clamp. Reference: `pwm-core::fixed_point`.
- Nonlinearities are **committed integer lookup tables** (GELU/SiLU/softmax‑exp/
  inverse‑sqrt), range‑checked — the verifier recomputes the table read exactly.

The int8 MLP MAC bound `2048·127·127 = 33,032,192 < 2³¹−1` holds, so fc1/fc2
accumulate in one int32; int16 activations switch to a limb accumulator.

---

## 4. Committed objects (reuse `pwm-core::commit`)

All commitments are Blake2s‑256, domain‑separated and length‑prefixed
(`commit(tag, payload)`); these structs already exist:

| Commitment | Binds |
|---|---|
| `model_commitment` (`ModelBinding`) | architecture sub‑commitment + `weights_root` + relation/serialization versions. |
| `quantization_commitment` (`QuantBinding`) | rounding mode, overflow policy, scale table, activation/lookup‑tables commitment. |
| `planner_config_commitment` (`PlannerBinding`) | horizon, action block, candidate count `S`, tie‑break rule id. |
| `weights_root` | Merkle root over quantized weight tensors (sorted by `tensor_id`). |
| `claimed_output_commitment` | the claimed `z_next` / trajectory / selection. |
| **`trace_root`** *(new)* | Merkle root over the execution‑trace leaves (§5). |
| `public_input` digest | canonical digest of all public inputs. |

`relation_id = hash(relation string)` is immutable per statement; a semantic
change mints a new id.

---

## 5. Execution‑trace model (new `pwm-core::trace`)

The trace is an **ordered list of op records**, one per exported op, each a small
pure‑data struct. The prover fills it from the reference inference; the verifier
audits it. Leaves of the `trace_root` Merkle tree are the canonical bytes of
each record (or each tensor cell group, to allow selective opening).

```rust
// pwm-core::trace  (data only, no_std)
pub enum OpRecord {
    Linear   { op_id: OpId, x: TensorRef, z_acc: AccTensor, /* i32 accumulators */ },
    Requant  { op_id: OpId, n: AccTensor, out: TensorRef, quot_rem: Vec<QuotRem> },
    Activation { op_id: OpId, table_id: TableId, x: TensorRef, out: TensorRef },
    LayerNorm { op_id: OpId, x: TensorRef, mean: AccCell, var: AccCell,
                inv_std: TensorRef, out: TensorRef },
    AttnScore { op_id: OpId, q: TensorRef, k: TensorRef, score_acc: AccTensor },
    Softmax   { op_id: OpId, table_id: TableId, score: TensorRef, prob: TensorRef,
                denom: AccCell, recip: Cell },
    AttnApply { op_id: OpId, prob: TensorRef, v: TensorRef, out_acc: AccTensor },
    Elementwise { op_id: OpId, kind: EwKind /* add|modulate|gate */, ins: Vec<TensorRef>,
                  out: TensorRef },
    RolloutStep { step: u32, window_in: TensorRef, pred_out: TensorRef },
    Cost     { candidate: u32, pred_final: TensorRef, goal: TensorRef, sse: AccCell },
    Argmin   { costs: Vec<AccCell>, selected_index: u32, selected_cost: AccCell,
               diffs: Vec<BoundedInt> /* selected ≤ each */ },
}
```

`Linear` records keep **i32 accumulators `z` (not requantized)** so Freivalds is
exact (requant is verified separately by the `Requant` record). `AttnScore`/
`AttnApply` keep accumulators for exact recompute.

INV‑TRACE‑01: the trace op set, order, and op kinds equal the manifest `ops[]`;
mismatch is `TraceError::GraphMismatch` at build, before commitment.

---

## 6. The audit protocol

Four phases. Phases 0–1 are offline/prove; 2–3 are the proof check.

### Phase 0 — Setup / export (offline, trusted)
`pwm-export` reads the le‑wm checkpoint + config, quantizes to the integer graph,
folds BatchNorm, emits `manifest`, weight tensors, and golden vectors. The
predictor bundle additionally carries a predictor‑scoped `weights_root` over the
proven block tensors (prover `tensor_id` order) that the prover binds into the
model commitment (bundle ⇄ export bound; checkpoint ⇄ export trusted). The same
bundle carries calibrated SiLU, GELU, inverse-sqrt, and softmax-exp tables, the
scale table inputs, the float reference predictor output on the encoded input,
and the measured int-vs-float tolerance. The parity gate requires Python ref ≡
Rust ref ≡ golden, bit‑for‑bit. Re‑export is byte‑identical.

### Phase 1 — Prove (commit)
`pwm-prover`:
1. Verify `hash(manifest) == model_commitment` and `weights_root` before any work.
2. Canonicalize public inputs (latent history, goal latent, candidate actions,
   claimed outputs); derive the public‑input digest.
3. Run the **exact integer reference inference** over the exported graph → all
   intermediate activations and accumulators → the `OpRecord` trace.
4. Merkle‑commit the trace → `trace_root`.
5. **Fiat‑Shamir:** absorb `relation_id`, public‑input digest, `model_commitment`,
   `quantization_commitment`, `planner_config_commitment`, `trace_root`; squeeze
   (a) the Freivalds vectors `r` (one per distinct weight matrix), and (b) the
   audit selection (for V0, the full set; for V1, a sampled subset).
6. Emit `AuditArtifact { manifest_ref, commitments, public_input, claimed_outputs,
   openings }` where `openings` are the trace records (or the selected subset +
   Merkle proofs) plus the weight openings needed for Freivalds.

### Phase 2 — Challenge
Non‑interactive: the verifier re‑derives `r` and the audit selection from the
same transcript (deterministic). Optional **interactive mode** (V1): the verifier
samples `r` secret and never reveals it (unconditional Freivalds soundness); the
prover precomputes nothing.

### Phase 3 — Verify (CPU, `no_std`, float‑free)
`pwm-verifier::verify(&AuditArtifact)`:
1. Recompute public‑input digest; check `relation_id` supported & matches.
2. Check `model_commitment`, `quantization_commitment`, `planner_config_commitment`
   against the manifest; check `trace_root` against opened leaves (Merkle proofs).
3. Replay the transcript → `r`, audit selection.
4. For every `Linear` op: **Freivalds** `v·x == r·z` where `v = rᵀW` from committed
   weights (§7).
5. For every `Requant`/`Activation`/`LayerNorm`/`Softmax`/`Elementwise` op:
   **exact integer recompute** from committed inputs + committed tables (§8).
6. For every `AttnScore`/`AttnApply`: **exact integer recompute** of the
   data‑dependent dot products (§8.2).
7. Check **rollout recurrence** wiring (§9), **MSE cost** and **argmin** (§10).
8. Return `Ok(())` or a typed `VerifyError`. Unsupported semantics fail closed.

---

## 7. Freivalds verification of linear layers {#freivalds}

For a linear op `z = W·x` (W an exported weight matrix, `rows×cols`):

```text
v = rᵀ W           (length cols)         # precomputed once per matrix
check:  v · x  ==  r · z   (mod p)       # per instance
```

because `rᵀ(Wx) = (rᵀW)x = v·x`. `x` is the int8/int16 input; `z` is the **i32
accumulator** (not requantized). All arithmetic in `F_p`.

- **Field:** `p = 2⁶¹ − 1` (Mersenne, fast reduction). One field for the audit;
  `BoundedInt` values embed by centered encoding. (M31 stays only for the
  fixed‑point *value* encoding where used; the Freivalds field is independent and
  larger for soundness headroom.)
- **Amortization (the win):** the same `W` is applied across every candidate ×
  rollout step × block, so `v = rᵀW` is computed **once per weight matrix** and
  reused for an `O(cols)` check per instance instead of an `O(rows·cols)`
  recompute. Reuse factor ≈ `S × horizon` per matrix.
- **Soundness:** a wrong `z ≠ Wx` passes with probability ≤ `1/p` per check
  (Freivalds/Schwartz‑Zippel). Union bound over `N` checked instances ≤ `N/p`
  (≈ `2⁻⁴⁴` at `N ≈ 10⁵`). Amplify with a second independent vector or a second
  prime if a deployment needs a tighter bound.
- **Challenge derivation:** `r` is squeezed from the Fiat‑Shamir transcript
  *after* `trace_root` is fixed (non‑interactive, computational soundness), or
  kept verifier‑secret (interactive, unconditional). Either way the prover
  commits `z` before `r` is known.
- Module: `pwm-core::freivalds` — `precompute_v(r, weight, rows, cols)` and
  `check(v, x, r, z)`, pure integer, `no_std`. (Direct re‑implementation of
  CommitLLM `verilm-core::freivalds` in our `BoundedInt` types.)

INV‑FREI‑01: a `Linear` record's `z` is the raw i32 accumulator; requant happens
only in the matching `Requant` record. INV‑FREI‑02: `v` is derived from
committed weights (`weights_root`), never from prover‑supplied data.

---

## 8. Exact re‑execution of everything else

These ops are cheap and deterministic; the verifier recomputes them exactly from
committed inputs — no probabilistic gap, no tolerance.

### 8.1 Requant / residual / modulate / gate
`Requant`: recompute `requantize(n, r, zp, c_lo, c_hi, mode)` from the committed
accumulator and check the opened `quot_rem` and output. `Elementwise` (add /
`modulate(x,sh,sc)=x*(1+sc)+sh` / `gate(g,t)=t*g`): recompute the integer ops and
check the output; every product/sum range‑checked.

### 8.2 Attention inner products (data‑dependent → exact recompute)
`Q,K,V` come from Freivalds‑checked `Linear` records. Then, at `S = H+1 = 4`,
`dim_head = 64`, `heads = 16`:
- `score[k,i,j] = Σ_d Q[i,k,d]·K[j,k,d]` (≈ 16·4·4·64 = 16,384 MACs) —
  **recompute exactly**, requantize by `1/√64`, apply the committed causal mask.
- `out[k,i,:] = Σ_j prob[k,i,j]·V[j,k,:]` — recompute exactly.

There is **no fixed weight** to amortize here, but the cost is trivial, so
Freivalds is unnecessary. This is what closes CommitLLM's attention hole: our
attention is integer‑exact and small, so it is *verified*, not merely audited.

### 8.3 Nonlinear tables (softmax / GELU / SiLU / inverse‑sqrt)
Each is a committed integer lookup table bound by `quantization_commitment`. The
verifier recomputes the table read and range‑checks inputs/outputs.
`Softmax_Q` = table‑exp + summed denominator + reciprocal witness (all integer).
`LayerNorm_Q` (affine‑free) = integer mean/variance + `invsqrt_table` read.

INV‑EXACT‑01: no probability or activation value is admitted unless it is the
constrained output of its op record (no off‑trace nonlinearities).

---

## 9. Rollout recurrence (P1)

The `RolloutStep` records must form the le‑wm autoregressive recurrence
(`jepa.py::rollout`, `history_size = 3`): each step's `pred_out` is the
last‑position predictor output over the trailing `HS` window of latents and
action embeddings, and is appended to the window consumed by the next step. The
verifier checks (a) each step's predictor sub‑trace verifies as a P0 step, and
(b) the windowing wiring (step `t`'s input window = previous latents + appended
`pred_out`). Trajectory committed under `claimed_output_commitment`.

INV‑ROLL‑01: a step that reads a latent not produced by a prior step (or the
committed initial history) fails the wiring check.

## 10. Cost and argmin (P2 = V0)

- **Cost** `cost_s = Σ_j (z_final,s[j] − goal[j])²`: the squared‑difference sum is
  recomputed exactly (or Freivalds‑checked as `d·d` with `d = z−goal`), accumulator
  range‑checked.
- **Argmin**: for the claimed `selected_index`, check `selected_cost ≤ cost_s` for
  all `s` via nonnegative difference witnesses, and the smallest‑index tie‑break
  (`cost_s − selected_cost − 1 ≥ 0` for `s < selected_index`). All `S` candidates
  must be present (proving only the winner is unsound).

---

## 11. Crate layout (post‑pivot: 5 crates)

```text
ProvableWorldModel/
  Cargo.toml                 # stable toolchain; no Stwo; members = 5 crates
  crates/
    pwm-core/                # field (native) + audit field; fixed_point; tensor;
                             #   manifest; commit (Merkle); transcript (FS);
                             #   serialize; public_input; relation; freivalds; trace; obs
    pwm-export/              # Python: le-wm checkpoint → quantized graph → manifest
                             #   + golden vectors + sw-data adapter.  Rust: integer reference + parity
    pwm-prover/              # reference inference → trace → commit → FS → AuditArtifact
    pwm-verifier/            # CPU, no_std, float-free: Freivalds + exact replay + checks
    pwm-testkit/             # golden / accept-reject / mutation harness
  docs/
    legacy-stark/            # archived pre-pivot corpus (RFCs, AIR specs)
```

### 11.1 Dependency DAG (acyclic; CI‑enforced)

```text
pwm-prover ──► pwm-export ──► pwm-core ◄── pwm-verifier
     └──────────────────────► pwm-core
pwm-testkit ──► (pwm-prover, pwm-verifier, pwm-export, pwm-core)
```

- `pwm-core` depends on **no proving substrate** and compiles `no_std` (INV‑ARCH‑01).
- `pwm-verifier` depends on **neither `pwm-export` nor any Python/float runtime**;
  it verifies arithmetic only (INV‑ARCH‑02). `freivalds`, `trace`, `fixed_point`,
  `commit`, `transcript`, `serialize` all live in `pwm-core` so prover and verifier
  share one implementation.
- No `pwm-air`, `pwm-circuits`, or `third_party/stwo` (removed).

### 11.2 Toolchain

Stable Rust (MSRV pinned in `rust-toolchain.toml`, no nightly). `pwm-verifier`
core builds under `no_std` with no floating point. CI gates: `no_std` build,
float‑free lint on the verifier, dependency‑DAG check, `cargo deny`, SPDX
headers, fmt/clippy.

---

## 12. Public API and artifact

```rust
// pwm-prover
pub fn prove_step(inputs, manifest, weights)     -> Result<AuditArtifact, ProveError>;   // P0
pub fn prove_rollout(inputs, manifest, weights)  -> Result<AuditArtifact, ProveError>;   // P1
pub fn prove_planning(inputs, manifest, weights) -> Result<AuditArtifact, ProveError>;   // P2

// pwm-verifier  (no_std)
pub fn verify(artifact: &AuditArtifact) -> Result<(), VerifyError>;            // P0 feed-forward
pub fn verify_predictor(artifact: &PredictorArtifact) -> Result<(), VerifyError>; // full predictor (RELATION_PREDICTOR)
pub fn verify_block(block, weights, tables, inputs) -> Result<Vec<i64>, VerifyError>; // standalone block audit (no commitments)

pub struct AuditArtifact {
    pub artifact_version: u32,
    pub relation_id: [u8; 32],
    pub public_input: PublicInput,
    pub commitments: Commitments,     // model/quant/planner/trace roots + output commitment
    pub openings: Openings,           // trace records (or sampled subset) + Merkle proofs + weight openings
    pub claimed_outputs: Option<Vec<Tensor>>,
}
```

`VerifyError` taxonomy: `UnsupportedRelation`, `RelationMismatch`,
`CommitmentMismatch`, `PublicInputMismatch`, `MerkleProofInvalid`,
`FreivaldsCheckFailed`, `ExactReplayMismatch`, `RangeCheckFailed`,
`RolloutWiringInvalid`, `CostMismatch`, `ArgminViolation`,
`OutputCommitmentMismatch`, `TranscriptMismatch`.

---

## 13. Soundness and threat model

**What is verified.** Every fixed‑weight matmul (Freivalds, error ≤ `N/p`); every
requant, residual, modulate, gate, attention inner product, nonlinear‑table read,
LayerNorm, MSE, and argmin (exact); the rollout recurrence wiring; all bindings.

**What is *not* a hole (vs CommitLLM).** CommitLLM cannot verify arbitrary‑position
attention because GPU bf16 FlashAttention is non‑reproducible. We run the
*quantized integer* model: attention is small and deterministic, so we recompute
it exactly. There is **no fake‑`a` residual hole**.

**Binding.** Mutating any load‑bearing value (weight, scale, rounding, table, op
order, planner config, public input, claimed output, any trace cell) changes a
commitment or fails an exact check → reject. INV‑BIND‑01: everything the relation
depends on is reachable from a commitment.

**Trust boundary (export chain).** Bundle ⇄ export is cryptographically bound:
the export emits, in the predictor bundle, a `weights_root` computed over exactly
the proven block tensors in the prover's canonical `tensor_id` order, and the
prover binds that carried value (not a recomputed one) into the model commitment,
so `verify_predictor` rejects (`CommitmentMismatch(Model)`) unless the proven
weights reproduce it bit‑for‑bit. Checkpoint ⇄ export is trusted preprocessing:
that the quantized tensors derive faithfully from the original checkpoint bytes
is asserted by the export step, not by the proof.

**Honesty boundary.** The proof attests the exact quantized relation named by
`relation_id`. For predictor bundles, the export also supplies an offline float
reference and measured tolerance that the CLI checks after integer verification.
It does **not** prove the full PyTorch program, physical truth, checkpoint-export
honesty, or ZK. Public claims must be a subset of the V0 statement.

**Determinism prerequisites (from le‑wm analysis).** Attention is recomputed as
explicit `QKᵀ/√d → softmax → ·V` integer matmuls (not `F.sdpa`, whose kernel
dispatch is platform‑dependent); inputs fixed at the trained grid so positional
interpolation never runs; BatchNorm folded; `eval()` semantics (no dropout). The
exporter is responsible for producing a graph that the integer reference
reproduces bit‑for‑bit.

---

## 14. Data pipeline (stable‑worldmodel adapter)

`pwm-export` includes a thin Python adapter over stable‑worldmodel:

- **Tuples for proving:** `swm.data.load_dataset(path, num_steps=2, frameskip=1,
  keys_to_load=['pixels','action','proprio'])` yields `(obs, action, next_obs)`;
  `load_episode`/`load_chunk` give whole trajectories. V0 consumes **latents**
  (the encoder is deferred), so the adapter encodes obs→latent once with the
  le‑wm encoder offline and stores `(z_history, action, z_next)` and goal latents.
- **Candidate sets:** the planning seam is `model.get_cost(info, action_candidates)
  → (B, S)` driven by the CEM/MPPI solvers (`solver/cem.py`). For V0 the candidate
  set is **fixed** (P2), exported alongside the goal latent and initial history.
- The adapter is offline and trusted; it produces committed inputs, never enters
  the verifier.

---

## 15. Testing strategy

- **Golden vectors:** per‑op and end‑to‑end integer reference outputs; Rust ≡
  Python ≡ golden, bit‑for‑bit (parity gate).
- **Accepting tests:** P0 step, one ConditionalBlock, rollout (horizon 5), full
  P2 planning — prove then verify `Ok`.
- **Rejecting tests:** mutate a weight / activation / accumulator / requant
  remainder / table value / cost / argmin / tie‑break / relation_id / any
  commitment → specific `VerifyError`.
- **Freivalds soundness tests:** a wrong `z` is rejected; transcript replay
  matches; `r` derivation is deterministic.
- **Mutation tests:** each verifier check must be load‑bearing (mutating it makes
  an accepting test fail) — no dead checks.
- **Differential:** Python ↔ Rust integer reference across seeded inputs.

---

## 16. Relationship to CommitLLM

| CommitLLM (verilm) | ProvableWorldModel (post‑pivot) |
|---|---|
| Freivalds over `F_p` (`p=2³²−5`/`2⁶¹−1`/`2¹²⁷−1`), `v=rᵀW`, check `v·x==r·z` | **Reused**, `p=2⁶¹−1`, in `pwm-core::freivalds` over `BoundedInt`. |
| Verifier‑secret `r` (interactive) | Non‑interactive Fiat‑Shamir `r` (V0); optional verifier‑secret mode (V1). |
| Merkle commit of irreducible state; canonical f64 replay of bridge ops | Merkle commit of trace; **exact integer** replay (no f64, no tolerance). |
| Attention = unverifiable hole (bf16) | Attention = **exactly recomputed** (quantized integer, small `S`). |
| Decode/sampling, RoPE/GQA/KV/causal‑mask machinery, LM‑head logits | **Removed** (LLM‑specific). Replaced by latent prediction + rollout + MSE + argmin. |
| std + rayon verifier | `no_std`, float‑free verifier. |
| Lean 4 formalization of Freivalds/Merkle | Future: optional formalization of the Freivalds + argmin checks. |
