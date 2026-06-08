# ProvableWorldModel — Pivot Backlog

Actionable GitHub‑issue backlog for the CommitLLM × LeWorldModel pivot. Companion
to [roadmap.md](roadmap.md) (plan) and [specs.md](specs.md) (spec).

**Conventions.** Each issue: `ID — Title` · area label · size (S ≤ 0.5d, M ≈ 1–2d,
L ≈ 3–5d) · `deps:` blocking issues. Area labels reuse the existing scheme:
`area:core`, `area:export`, `area:prover`, `area:verifier`, `area:testing`,
`area:ci`, `area:docs`, `area:security`, `area:planner`. Acceptance criteria are
the merge gate. Keep files small and focused (no thousand‑line modules).

Legend for status of pre‑pivot work: **[reuse]** keep as‑is · **[adapt]** modify ·
**[remove]** delete · **[new]** create.

---

## Milestone M0 — Repository reset & scaffold

> Goal: a clean, building, stable‑toolchain 5‑crate skeleton with the STARK
> backend removed and the new corpus in place. No proofs yet.

- **P-001 — Branch & archive legacy STARK corpus** · area:docs · S
  - `git mv` STARK‑specific RFCs (range‑check, tensor‑memory AIR, linear/matmul
    AIR, nonlinear AIR, predictor AIR, rollout AIR, planner AIR, CEM AIR, encoder
    AIR, recursion, vendoring) and the AIR sections of the spec docs to
    `docs/legacy-stark/`. Add a `docs/legacy-stark/README.md` explaining they are
    superseded by [specs.md](specs.md).
  - **AC:** nothing deleted (moved only); `docs/legacy-stark/` builds in link‑check;
    top‑level docs reference the new corpus.
  - deps: —

- **P-002 — Remove AIR & circuits crates** · area:core · S · `[remove]`
  - Delete `crates/pwm-air`, `crates/pwm-circuits`; drop from `Cargo.toml` members
    and `[workspace.dependencies]`.
  - **AC:** workspace resolves without them; no first‑party crate references them.
  - deps: P-004

- **P-003 — Remove vendored Stwo & stwo‑circuits** · area:core · S · `[remove]`
  - Delete `third_party/stwo`, `third_party/stwo-circuits`; remove `stwo` from
    `[workspace.dependencies]`; update `third_party/INVENTORY.md`/`README.md`.
  - **AC:** no `stwo`/`stwo-circuits` path or dep anywhere; `cargo metadata` clean.
  - deps: P-004, P-005

- **P-004 — Switch to stable toolchain** · area:ci · S · `[adapt]`
  - Replace `rust-toolchain.toml` nightly pin with a stable channel pin; bump
    `workspace.package.rust-version` MSRV; remove nightly‑only flags.
  - **AC:** `cargo +stable build` succeeds for all remaining crates.
  - deps: P-002, P-003

- **P-005 — De‑Stwo `pwm-core::field`** · area:core · M · `[adapt]`
  - Replace the M31/QM31 re‑export from Stwo with a **native** M31 implementation
    (centered encoding helpers unchanged) so `pwm-core` has zero proving deps. Add
    the **audit field** `Fp61` (`p = 2⁶¹−1`, Mersenne) used by Freivalds.
  - **AC:** `pwm-core` `cargo tree` shows no `stwo`; `no_std` build passes; existing
    `field`/`fixed_point` tests pass against native M31.
  - deps: —

- **P-006 — Update workspace manifest & lints** · area:ci · S · `[adapt]`
  - `members = 5 crates`; remove vendored exclude; keep lints/profiles; blake2 stays.
  - **AC:** `cargo build`/`clippy`/`fmt` green on the 5‑crate workspace.
  - deps: P-002, P-003

- **P-007 — Rewrite README & SPEC pointers** · area:docs · S · `[adapt]`
  - Point README/SPEC to roadmap/specs/backlog; update the crate table and the
    "what it is" section to commit‑and‑audit.
  - **AC:** links resolve; no STARK/Stwo claims remain in top‑level docs.
  - deps: P-001

- **P-008 — Prune CI of AIR/Stwo gates** · area:ci · S · `[adapt]`
  - Remove AIR/LogUp/Stwo‑specific CI steps; keep SPDX, changelog, links, fmt,
    clippy, deny. Add the new gates as stubs (filled in M7).
  - **AC:** CI passes on the scaffold.
  - deps: P-004, P-006

---

## Milestone M1 — Core crypto & data model

> Goal: the shared, `no_std`, float‑free primitives the prover and verifier both use.

- **C-101 — `pwm-core::freivalds`** · area:core · M · `[new]`
  - `precompute_v(r: &[Fp61], weight: &[i8], rows, cols) -> Vec<Fp61>` and
    `check(v, x, r, z) -> bool` over `Fp61`; int8/int16 `x`, i32 `z`. Port from
    CommitLLM `verilm-core::freivalds` into our types. `no_std`.
  - **AC:** unit tests (identity, negative weights, wrong‑z reject) pass; property
    test: random `W,x` → `check` true iff `z == Wx`.
  - deps: P-005

- **C-102 — Freivalds challenge derivation** · area:core · S · `[new]`
  - Derive `r` vectors (one per weight `tensor_id`) from the Fiat‑Shamir transcript
    after `trace_root`; deterministic, documented channel order.
  - **AC:** prover and verifier derive identical `r`; reorder → different `r`.
  - deps: C-101, [reuse] transcript

- **C-103 — `pwm-core::trace` data model** · area:core · M · `[new]`
  - The `OpRecord` enum and tensor/accumulator reference types (§5), canonical
    encodings via `CanonicalEncode`. Pure data, `no_std`. Keep per‑variant structs
    in small submodules.
  - **AC:** round‑trip canonical bytes; golden encoding vectors stable.
  - deps: [reuse] serialize, tensor

- **C-104 — Trace Merkle commitment** · area:core · S · `[adapt]`
  - `trace_root` over trace leaves reusing `commit.rs` Merkle (selective‑open
    friendly leaf layout). Merkle inclusion proof + verify.
  - **AC:** root stable; valid proof verifies; tampered leaf/proof rejected.
  - deps: C-103, [reuse] commit

- **C-105 — Extend commitments for the new artifact** · area:core · S · `[adapt]`
  - Add `trace_root` to the binding/commitment set; keep `ModelBinding`/
    `QuantBinding`/`PlannerBinding` as‑is.
  - **AC:** mutating any bound field changes the digest; commitment tests pass.
  - deps: C-104

- **C-106 — Limb accumulator review for Freivalds field** · area:core · S · `[adapt]`
  - Ensure accumulator bounds and limb decomposition are consistent with `Fp61`
    embedding (int32 accumulators fit; limb path documented).
  - **AC:** bound‑escape cases produce limb decomposition, not wraparound.
  - deps: P-005

---

## Milestone M2 — Export pipeline & data adapter

> Goal: le‑wm checkpoint → committed quantized manifest + golden vectors; data tuples.

- **E-201 — le‑wm checkpoint ingest** · area:export · M · `[new]`
  - Python: load `JEPA` object / `weights.pt`+`config.json`; extract the V0 subgraph
    (`action_encoder`, `predictor` ×6 blocks, `pred_proj`); record shapes/dims.
  - **AC:** ingest the reference checkpoint; assert dims (192/3/6/16/64/2048).
  - deps: P-004

- **E-202 — Quantization pass** · area:export · L · `[adapt]`
  - int8 weights, int8/int16 activations (per‑tensor), int32 accumulators/biases,
    pow‑2 scales; per spec §3. Emit per‑op scale ids.
  - **AC:** quantized graph; MAC bounds respected; scales recorded.
  - deps: E-201

- **E-203 — BatchNorm folding** · area:export · M · `[new]`
  - Fold `pred_proj` (and projector, deferred) `BatchNorm1d` frozen stats into the
    preceding `Linear`; record provenance.
  - **AC:** folded affine ≡ Linear∘BN on golden inputs (within quant); fold is
    byte‑deterministic.
  - deps: E-202

- **E-204 — Nonlinear lookup‑table generation** · area:export · M · `[new]`
  - Emit committed integer tables: `gelu_table_v1`, `silu_table_v1`,
    `softmax_table_v1` (exp), `invsqrt_table_v1`; bind under `quantization_commitment`.
  - **AC:** tables committed; table reads reproduce the quantized nonlinearity.
  - deps: E-202

- **E-205 — Manifest writer & commitments** · area:export · M · `[adapt]`
  - Write canonical manifest (`ops[]`, scales, tables, BatchNorm provenance);
    compute `model_commitment`/`quantization_commitment`; byte‑identical re‑export.
  - **AC:** re‑export byte‑identical; commitments stable (SC3).
  - deps: E-203, E-204

- **E-206 — Rust integer reference inference** · area:export · L · `[new]`
  - Rust path over the exported graph reproducing the quantized ops exactly
    (`pwm-core::fixed_point`), emitting all activations/accumulators.
  - **AC:** Rust ref ≡ golden bit‑for‑bit on the V0 subgraph.
  - deps: E-205, [reuse] fixed_point

- **E-207 — Python↔Rust parity gate** · area:export/testing · M · `[new]`
  - Differential: Python quantized ref ≡ Rust ref ≡ golden across seeded inputs.
  - **AC:** parity gate green; any divergence fails CI (SC4).
  - deps: E-206

- **E-208 — stable‑worldmodel data adapter** · area:export · M · `[new]`
  - Thin Python: `load_dataset(num_steps=2)` → `(obs, action, next_obs)`; encode
    obs→latent offline; export `(z_history, action, z_next)`, goal latent, and a
    **fixed candidate action set** for P2.
  - **AC:** produces committed input bundles consumable by the prover.
  - deps: E-201

---

## Milestone M3 — Trace builder & prover skeleton

- **PR-301 — Reference→trace builder** · area:prover · L · `[new]`
  - Consume the Rust reference activations/accumulators → ordered `OpRecord` trace;
    assert graph fidelity vs manifest `ops[]` (INV‑TRACE‑01).
  - **AC:** trace matches manifest op set/order; `GraphMismatch` on mismatch.
  - deps: E-206, C-103

- **PR-302 — Prover commit & transcript** · area:prover · M · `[new]`
  - Merkle‑commit the trace; absorb the canonical transcript; squeeze `r` + audit
    selection (full set for V0).
  - **AC:** `trace_root` + `r` reproducible; transcript order documented.
  - deps: PR-301, C-102, C-104

- **PR-303 — AuditArtifact assembly & (de)serialize** · area:prover/core · M · `[new]`
  - Build `AuditArtifact { commitments, public_input, openings, claimed_outputs }`;
    canonical binary (de)serialization.
  - **AC:** round‑trip stable; artifact self‑contained.
  - deps: PR-302, C-105

- **PR-304 — Prover CLI** · area:prover · S · `[adapt]`
  - `prove_step|prove_rollout|prove_planning` entry points + CLI; load manifest +
    weights + inputs, emit artifact.
  - **AC:** CLI produces a P0 artifact end‑to‑end.
  - deps: PR-303

---

## Milestone M4 — Verifier & P0 (predictor step)

- **V-401 — Verifier skeleton (no_std)** · area:verifier · M · `[new]`
  - `verify(&AuditArtifact)`: parse, check `relation_id`, recompute public‑input
    digest, check commitments + `trace_root` Merkle proofs, replay transcript → `r`.
    `no_std`, float‑free.
  - **AC:** structural checks pass on a valid artifact; tamper → typed error.
  - deps: PR-303, C-102

- **V-402 — Freivalds matmul checks** · area:verifier · M · `[new]`
  - For each `Linear` record: `v = rᵀW` from committed weights, `check(v,x,r,z)`.
  - **AC:** valid matmuls pass; a single wrong accumulator → `FreivaldsCheckFailed`.
  - deps: V-401, C-101

- **V-403 — Exact replay: requant/elementwise/layernorm** · area:verifier · M · `[new]`
  - Recompute `Requant`, `Elementwise` (add/modulate/gate), `LayerNorm` exactly;
    range‑check every value.
  - **AC:** valid ops pass; mutated remainder/modulation/inv‑std → typed error.
  - deps: V-401, [reuse] fixed_point

- **V-404 — Exact replay: attention + nonlinear tables** · area:verifier · L · `[new]`
  - Recompute `AttnScore`/`AttnApply` integer dot products; verify `Softmax`,
    `Activation` (GELU/SiLU) table reads against committed tables.
  - **AC:** attention + activations verified exactly; off‑trace prob/activation
    rejected (no fake‑`a` hole — security test).
  - deps: V-402, V-403, E-204

- **V-405 — P0 end‑to‑end accept/reject** · area:verifier/testing · M · `[new]`
  - Wire V-402..404 into the P0 relation; accepting + the predictor reject set
    from the spec.
  - **AC:** P0 proves+verifies; every P0 negative test rejects with the right code.
  - deps: V-404, PR-304

---

## Milestone M5 — Rollout (P1)

- **R-501 — Rollout trace + recurrence check** · area:prover/verifier · M · `[new]`
  - Emit `RolloutStep` records (le‑wm windowing, `history_size=3`); verifier checks
    each step as a P0 step + windowing wiring (INV‑ROLL‑01).
  - **AC:** P1 horizon‑5 proves+verifies; mutated intermediate latent rejected.
  - deps: V-405

- **R-502 — Trajectory commitment** · area:core/prover · S · `[adapt]`
  - Commit the rollout trajectory under `claimed_output_commitment`.
  - **AC:** trajectory bound; tamper rejected.
  - deps: R-501

---

## Milestone M6 — Fixed‑candidate planning (P2 = V0)

- **PL-601 — Multi‑candidate rollout** · area:planner/prover · M · `[new]`
  - Roll out all `S` committed candidates (Freivalds `v` reused across candidates).
  - **AC:** all candidates traced; reuse factor measured.
  - deps: R-502

- **PL-602 — MSE cost check** · area:planner/verifier · M · `[new]`
  - `cost_s = Σ(z_final,s − goal)²` exact/Freivalds; accumulator range‑checked.
  - **AC:** costs verified; mutated cost rejected.
  - deps: PL-601

- **PL-603 — Argmin + tie‑break check** · area:planner/verifier · M · `[new]`
  - `selected_cost ≤ cost_s` ∀s via difference witnesses; smallest‑index tie‑break;
    all `S` present (no winner‑only proof).
  - **AC:** selection verified; lower unselected candidate / wrong index / bad
    tie‑break each rejected.
  - deps: PL-602

- **PL-604 — P2 end‑to‑end (V0 headline)** · area:planner · M · `[new]`
  - `prove_planning` → `verify` `Ok`; full negative suite (SC1, SC2).
  - **AC:** V0 statement holds end‑to‑end; soundness suite green.
  - deps: PL-603

---

## Milestone M7 — Hardening, CI, docs

- **T-701 — Golden vector corpus** · area:testing · M · per‑op + e2e goldens (SC4). deps: E-207
- **T-702 — Mutation tests (no dead checks)** · area:testing · M · each verifier check load‑bearing. deps: PL-604
- **T-703 — Freivalds soundness/fuzz** · area:security/testing · M · wrong‑z fuzz; `N/p` bound documented; transcript determinism. deps: V-402
- **T-704 — `no_std` + float‑free verifier gate** · area:ci · S · CI builds verifier `no_std`, lints out float. deps: V-401
- **T-705 — Dependency‑DAG + deny + SPDX gates** · area:ci · S · enforce layering (INV‑ARCH‑01/02). deps: P-006
- **T-706 — Reproducibility gate** · area:ci · S · same inputs → same artifact bytes. deps: PR-303
- **T-707 — Spec/threat‑model docs finalize** · area:docs/security · M · binding checklist, honesty boundary, the "no attention hole" argument. deps: PL-604

---

## Milestone M8 — Deferred (post‑V0)

- **D-801 — Audit subsampling (sub‑linear verify)** · area:verifier · L · sample a subset of instances/steps; statistical coverage flagged (V1). deps: PL-604
- **D-802 — Interactive verifier‑secret `r` mode** · area:security · M · unconditional Freivalds soundness option (V1). deps: V-402
- **D-803 — CEM planner proof (P3)** · area:planner · L · seeded sampling, clipping, top‑k, distribution updates (V2). deps: PL-604
- **D-804 — Pixel encoder proof (P4)** · area:export/verifier · XL · ViT‑Tiny/14 encode; Freivalds on patch/linear, exact (or subsampled) attention at `S=256` (V3). deps: PL-604
- **D-805 — Lean formalization (optional)** · area:security · L · formalize Freivalds + argmin checks (mirrors CommitLLM `lean/`). deps: V-402, PL-603
- **D-806 — Batched/efficient P2** · area:prover · L · batch candidate rollouts at scale (V1). deps: PL-604

---

## Critical path

```text
P-005 ─► C-101 ─► V-402 ─┐
P-004 ─► E-201 ─► E-206 ─► PR-301 ─► PR-302 ─► V-401 ─► V-405 ─► R-501 ─► PL-601 ─► PL-604 (V0)
                 E-205 ──┘                          C-104 ┘
```

M0 unblocks everything (stable toolchain + native field). The export reference
(E-206) and the Freivalds core (C-101) are the two long poles into the first P0
proof (V-405); the rest composes upward to the V0 headline (PL-604).
