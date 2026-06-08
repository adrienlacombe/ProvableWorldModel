# ProvableWorldModel — Roadmap

> **Pivot (2026-06-05):** ProvableWorldModel moves from a custom Circle‑STARK
> arithmetization (vendored Stwo, M31) to the **CommitLLM commit‑and‑audit
> scheme** as its cryptographic foundation, adapted to prove end‑to‑end inference
> of a **LeWorldModel** (action‑conditioned JEPA). The *proof target* is
> unchanged — deterministic, quantized inference of the LeWorldModel predictor,
> rollout, and fixed‑candidate planner. The *proving backend* is replaced.

This file is the high‑level plan. The normative specification is [specs.md](specs.md);
the actionable work is [backlog.md](backlog.md).

---

## 1. Why pivot

The STARK approach was sound but expensive on two axes:

1. **Engineering cost.** Every operation (linear, matmul, requant, softmax,
   LayerNorm, attention, rollout, cost, argmin) had to be arithmetized as custom
   AIR with range‑checks and LogUp lookups — the dominant risk in the
   feasibility study. ~17 RFCs and a six‑crate AIR/circuit stack were needed
   before a single proof.
2. **Toolchain.** The vendored Stwo + stwo‑circuits require **nightly Rust**,
   which conflicts with the workspace's stable MSRV and blocked the vendoring
   chain (issues #24/#25).

CommitLLM replaces full arithmetization with **commit‑and‑audit**: the prover
runs the model normally and commits to its execution trace; a CPU verifier
checks the trace with (a) **Freivalds** randomized algebraic checks for the large
linear layers, (b) **exact canonical re‑execution** of the cheap deterministic
ops, and (c) **Merkle commitments** binding everything. No proving circuit, no
arithmetization, **stable toolchain, no nightly**.

### What we gain over CommitLLM‑for‑LLMs

CommitLLM has one honest, documented hole: GPU `bf16` FlashAttention is not
bit‑reproducible, so **arbitrary‑position attention outputs are not verified**
(a consistent fake‑`a` forgery clears the audit‑only verifier).

**Our world model runs in exact integer fixed‑point, not bf16 on a GPU.** That
single fact closes the hole:

- The big **fixed‑weight** matmuls (Q/K/V/proj, FFN fc1/fc2, AdaLN modulation,
  action encoder, pred‑proj) are Freivalds‑checked — `v = rᵀW` precomputed once
  per weight matrix, then reused across every candidate × horizon step × block.
- The small **data‑dependent** attention products (QKᵀ, prob·V at `S = H+1 = 4`)
  and the nonlinearities (softmax, GELU, SiLU, LayerNorm inverse‑sqrt) are
  **exactly recomputed** as integer ops against committed lookup tables.

Result: **full verification, no residual attention hole, no floating point in the
verifier.** Stronger than the LLM setting, because we control the (quantized)
reference.

---

## 2. What carries over, what is removed

| Asset | Disposition |
|---|---|
| `pwm-core::fixed_point` (bit‑exact integer reference) | **Keep** — this *is* the canonical replay reference the audit needs. |
| `pwm-core::commit` (Blake2s commitments + weight Merkle root) | **Keep** — commit‑and‑audit is built on exactly these. |
| `pwm-core::transcript` (Fiat‑Shamir) | **Keep** — derives Freivalds vectors + audit selection. |
| `pwm-core::{tensor, manifest, serialize, public_input, relation, limb, obs}` | **Keep** — shared vocabulary; manifest/binding structs reused verbatim. |
| `pwm-core::field` (M31 re‑export from Stwo) | **Adapt** — replace Stwo re‑export with a native prime field; add the 61‑bit audit field for Freivalds. Sheds the last proving dependency. |
| `pwm-export` (PyTorch → quantized graph → manifest + golden vectors) | **Keep & expand** — now even more central; add the stable‑worldmodel data adapter. |
| `pwm-prover`, `pwm-verifier` (Stwo prover/verifier) | **Rewrite** around commit‑and‑audit (small, no Stwo; verifier stays `no_std`, float‑free). |
| `pwm-testkit` (golden / accept‑reject / mutation) | **Keep**. |
| `pwm-air`, `pwm-circuits` (AIR + stwo‑circuits) | **Remove** — STARK‑specific. |
| `third_party/stwo`, `third_party/stwo-circuits` | **Remove** — vendored prover no longer needed; toolchain returns to stable. |
| STARK‑specific RFCs (range‑check, tensor‑memory AIR, LogUp, recursion, vendoring) | **Archive** under `docs/legacy-stark/` and supersede with the new corpus. |

Net crate set after the pivot: **`pwm-core`, `pwm-export`, `pwm-prover`,
`pwm-verifier`, `pwm-testkit`** (five crates; `freivalds` and the `trace` data
model become modules of `pwm-core`). Smaller surface, stable toolchain.

---

## 3. Architecture at a glance

```text
  ┌──────────── OFFLINE (trusted, Python+Rust) ────────────┐
  │  le-wm checkpoint ──► pwm-export ──► quantized integer  │
  │  (MIT, consumed)      (quantize)     graph + manifest   │
  │  stable-worldmodel ──► (obs, action, next_obs) tuples,  │
  │  (data + planning)     goal latent, candidate actions   │
  │                              │                          │
  │        parity gate: Python ref ≡ Rust ref ≡ golden      │
  └──────────────────────────────┼─────────────────────────┘
                                  ▼
  ┌──────────── PROVE (pwm-prover, normal run) ─────────────┐
  │  run exact integer reference inference over the graph;   │
  │  record the execution trace (matmul accumulators,        │
  │  requant witnesses, activations, attention, rollout,     │
  │  costs); Merkle-commit trace + weights + public inputs;  │
  │  Fiat-Shamir → Freivalds vectors r + audit selection;    │
  │  emit AuditArtifact { manifest, commitments, openings }. │
  └──────────────────────────────┼─────────────────────────┘
                                  ▼
  ┌──────────── VERIFY (pwm-verifier, CPU, no_std) ─────────┐
  │  recompute public-input digest; check model/quant/       │
  │  planner commitments; replay transcript → r;             │
  │  Freivalds-check every fixed-weight matmul (v·x == r·z);  │
  │  exactly recompute requant + nonlinear-table ops +        │
  │  attention; check rollout recurrence, MSE cost, argmin.   │
  │                       ──► Ok(())  or  VerifyError         │
  └─────────────────────────────────────────────────────────┘
```

---

## 4. Statement tiers (unchanged) and versions

The proof statements are the same as before the pivot; only the backend changes.

| Tier | Statement | Backend mapping |
|---|---|---|
| **P0** | One predictor step: `z_next = PredProj(ARPredictor(z_hist, ActEnc(a)))` | Freivalds on projections/FFN/AdaLN; exact replay of attention + nonlinearities. |
| **P1** | Autoregressive rollout over a horizon | P0 per step + recurrence/windowing check (the predicted latent feeds the next step). |
| **P2 (V0)** | Fixed‑candidate planning: roll out all S candidates, MSE cost, argmin | P1 per candidate + MSE (exact/Freivalds) + argmin comparison check. |
| P3 | Full CEM planner | Deferred (sampling/top‑k/distribution updates). |
| P4 | Pixel‑to‑plan incl. ViT encoder | Deferred (encoder ViT; Freivalds on patch/linear, exact attention). |

| Version | Tier | Delivers |
|---|---|---|
| **V0** | P2 | First public deliverable: prove fixed‑candidate planning end‑to‑end on the quantized LeWorldModel; full negative suite. |
| V1 | P2 (efficient) | Audit‑subsampling for sub‑linear verification at scale; optional interactive (verifier‑secret `r`) mode for unconditional Freivalds soundness. |
| V2 | P3 | CEM planner proof. |
| V3 | P4 | Pixel encoder + full pixel‑to‑plan. |

---

## 5. Soundness model (new)

The proof is **commitment‑bound end‑to‑end**. Within the binding:

- **Linear / matmul (fixed weights):** information‑theoretically sound Freivalds.
  A wrong accumulator `z ≠ Wx` is accepted with probability ≤ `1/p` per check
  (field `p = 2⁶¹ − 1`); union bound over `N` instances ≤ `N/p` (≈ `2⁻⁴⁴` for
  `N ≈ 10⁵`), amplifiable with a second independent vector/prime.
- **Requant, residual, modulate, gate, attention inner products, nonlinear
  tables, MSE, argmin:** **exact** integer recomputation from committed inputs —
  no probabilistic gap, no tolerance.
- **Binding:** every load‑bearing value (weights, scales, rounding, lookup
  tables, op order, planner config, public inputs, claimed outputs) is reachable
  from a commitment; mutating any of them changes the digest and the verifier
  rejects.
- **Non‑interactive:** Freivalds `r` and audit selection are derived by
  Fiat‑Shamir from the trace commitment *after* the prover commits, so the proof
  is a self‑contained artifact any third party checks (Freivalds soundness
  becomes computational under the hash). An optional **interactive mode** keeps
  `r` verifier‑secret for unconditional soundness (V1).

**Honesty boundary (carried over):** the proof attests the *exact quantized
arithmetic relation*, not float/PyTorch equivalence, not physical truth, not
zero‑knowledge. These remain explicit non‑goals.

---

## 6. Workstreams and sequencing

```text
M0  Reset & scaffold ........ remove AIR/circuits/stwo; stable toolchain; 5-crate skeleton
        │
M1  Core crypto ............. native field + audit field; freivalds module; reuse
        │                     commit/transcript/serialize; trace data model
        ├───────────────┐
M2  Export & data           M3  Trace + reference
    le-wm → quantized           exact integer reference inference over the
    manifest + golden;          exported graph (reuse fixed_point); emits the
    sw data adapter             execution trace
        └───────┬───────┘
M4  Predictor (P0) .......... prove/verify one step (Freivalds + exact replay)
        │
M5  Rollout (P1) ............ autoregressive recurrence + trajectory commitment
        │
M6  Planning (P2 = V0) ...... all candidates, MSE cost, argmin selection
        │
M7  Hardening .............. negative/mutation/differential suite; CI gates; docs
        ⋮
M8  Deferred ............... CEM (P3), pixel encoder (P4), audit subsampling, ZK audit
```

Dependency order matches the statement composition: a step is the inner kernel
of a rollout; a rollout is the inner kernel of a candidate; planning composes
all candidates with cost + argmin. Build the kernel first.

---

## 7. Success criteria (V0)

- **SC1** P2 proves end‑to‑end: `prove_planning` → `AuditArtifact`; `verify` → `Ok`.
- **SC2** Every negative test rejects (wrong weight, wrong activation, wrong
  accumulator, wrong requant remainder, wrong table value, wrong cost, wrong
  argmin, wrong tie‑break, wrong relation_id, tampered commitment).
- **SC3** Export is byte‑identical on re‑export; commitments stable.
- **SC4** Rust integer reference ≡ Python integer reference, bit‑for‑bit, on every
  golden vector.
- **SC5** Verifier builds and runs `no_std`, float‑free, on stable Rust.

A soundness failure (any negative test that verifies) outranks all functional
progress.

---

## 8. Risks and open questions

| Risk / question | Mitigation / resolution path |
|---|---|
| Freivalds amortization needs the *same* `r` reused across instances of a weight; soundness is `N/p`. | Use `p = 2⁶¹−1`; amplify with a second prime/vector if `N/p` is marginal. Specced in [specs.md](specs.md#freivalds). |
| Nonlinear ops as committed integer tables must match the float model "closely enough" to be useful. | This is the same approximation decision as before the pivot (RFC‑0006 lineage): the proof attests the *quantized* model `QuantizedLeWM-v1`, not float. Tables are committed; parity gate enforces Rust≡Python. |
| Attention is data‑dependent (no fixed weight to amortize). | `S` is tiny (4); recompute exactly. No Freivalds needed; no hole. |
| `pwm-core` must stay `no_std` + float‑free with the new field/freivalds. | Pure integer modular arithmetic only; CI `no_std` gate retained. |
| Pixel encoder (ViT) attention is `O(S²)` with `S = 256` patches. | Deferred to V3; Freivalds on patch/linear projections, exact (or subsampled) attention. |
| Interactive vs non‑interactive soundness. | V0 ships non‑interactive (Fiat‑Shamir, computational). Optional verifier‑secret interactive mode for unconditional soundness in V1. |

---

## 9. External dependencies (consumed, not vendored)

| Project | Role | License | How consumed |
|---|---|---|---|
| [CommitLLM](https://github.com/lambdaclass/CommitLLM) | Cryptographic scheme (Freivalds, commit‑and‑audit, canonical replay patterns) | — | Design reference; we re‑implement the small, relevant kernels in pure‑integer Rust. |
| [le-wm](https://github.com/lucas-maes/le-wm) | The world model being proven (JEPA predictor) | MIT | Checkpoint + config read by `pwm-export`. Not vendored. |
| [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) | Data pipeline + environments + MPC planning harness | MIT | Produces `(obs, action, next_obs)` tuples, goal latents, and candidate action sets for proving; the `get_cost` planning seam is the verified target. |
