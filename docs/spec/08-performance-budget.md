# 08 — Performance Budget

Status: Normative. Role: defines the proving cost model, trace-size accounting, engineering targets (budgets, not guarantees), scaling behavior, the profiling plan, and the perf-regression CI gates for ProvableWorldModel V0 (proof statement P2). Owner of numeric targets: ProvableWorldModel maintainers.

This document quantifies *what it costs to prove the V0 statement and how that cost scales*. It is the single source of truth for performance contracts. It does not redefine the arithmetic (see [`docs/spec/03-data-model.md#bounded-integers`](03-data-model.md#bounded-integers)), the AIR components (see [`docs/spec/01-architecture.md#component-model`](01-architecture.md#component-model)), the metrics surface (see [`docs/spec/05-observability.md#metrics`](05-observability.md#metrics)), or the CI test gates (see [`docs/spec/07-testing-strategy.md#ci-gates`](07-testing-strategy.md#ci-gates)). It consumes those and attaches numbers.

The reference configuration throughout is the V0 LeWorldModel predictor (verified against upstream `lucas-maes/le-wm` at 2026-06-03): `latent_dim = 192`, predictor `depth = 6`, `heads = 16`, `dim_head = 64`, `mlp_dim = 2048`, `history_size = 3`, rollout `horizon = 5`. See [`docs/feasibility-study.md`](../feasibility-study.md) §2 and §5.2 for provenance. The CEM solver parameters (`num_samples = 300`, `n_steps = 30`, `topk = 30`) are out of V0 scope and appear only in [#scaling](#scaling).

---

## cost-model

The dominant cost is the trace. STARK prover work (FRI commitment, low-degree extension, Merkle hashing, quotient evaluation) is roughly linear-to-quasilinear in the total number of trace rows, and the row count is itself dominated by multiply-accumulate (MAC) operations in the quantized predictor. Activation lookups, range checks, and tensor-memory consistency rows are real but second-order against the matmul row count for this architecture. Therefore the MAC count is the primary cost driver and the right unit for budgeting.

### Unit and conventions

- One **MAC** = one signed integer multiply followed by one accumulate into a running sum. In the AIR this is one `prod = x * w` plus one `acc_next = acc + prod` constraint pair (see [`docs/spec/01-architecture.md#component-model`](01-architecture.md#component-model) `LinearComponent`/`MatMulComponent`).
- A length-`N` dot product is `N` MACs and (in the natural layout) `N` trace rows in the linear component, plus one requantization tail.
- `inner = heads * dim_head = 16 * 64 = 1024` is the attention projection width.
- `L` is the number of latent tokens attention sees per predictor step. For V0 the AR predictor attends over the context window, so `L = history_size = 3`. This is small; attention-score and attention-value work is therefore negligible against the projections (confirmed numerically below).

### Per-linear-layer MACs (worked)

| Linear op (V0 dims) | Shape | MACs |
|---|---|---:|
| FFN up | `192 -> 2048` (per token) | `192 * 2048 = 393,216` |
| FFN down | `2048 -> 192` (per token) | `2048 * 192 = 393,216` |
| Attention out-proj | `1024 -> 192` (per token) | `1024 * 192 = 196,608` |
| One QKV linear | `192 -> 1024` (per token) | `192 * 1024 = 196,608` |
| AdaLN modulation | `192 -> 6*192 = 1152` (per block) | `192 * 1152 = 221,184` |
| pred_proj | `192 -> 192` (per step) | `192 * 192 = 36,864` |
| action embed | `~192 -> 192` (per step) | `~36,864` |

The two FFN linears match the contract's stated `192->2048 = 393,216` and `2048->192 = 393,216` exactly.

### Per predictor block (depth-1) MACs

Per `ConditionalBlock`, applied to `L = 3` tokens:

| Sub-component | Formula | MACs |
|---|---|---:|
| QKV projections (3 linears) | `3 * L * latent_dim * inner = 3 * 3 * 192 * 1024` | `1,769,472` |
| Attention scores `Q Kᵀ` | `heads * L * L * dim_head = 16 * 3 * 3 * 64` | `9,216` |
| Attention-value `P V` | `heads * L * L * dim_head` | `9,216` |
| Output projection | `L * inner * latent_dim = 3 * 1024 * 192` | `589,824` |
| FFN (up + down) | `L * (192*2048 + 2048*192) = 3 * 786,432` | `2,359,296` |
| AdaLN-zero modulation | `latent_dim * 6 * latent_dim = 192 * 1152` | `221,184` |
| **Block total** | | **≈ 4,958,208** |

The score/value matmuls are `9,216` MACs each: 0.2% of the block. With `L = 3`, attention cost is overwhelmingly its linear projections (QKV + out-proj = `2,359,296`), and the FFN matches it almost exactly (`2,359,296`). The block is essentially "two dense `192<->2048`-scale matmuls."

### Per predictor step, per rollout, per proof

| Level | Formula | MACs |
|---|---|---:|
| One predictor step | `block * depth + pred_proj + action_embed = 4,958,208 * 6 + 73,728` | `≈ 29,822,976` |
| One candidate rollout (`horizon = 5`) | `step * 5` | `≈ 149,114,880` |
| P2 proof, `S` candidates | `rollout * S` | `≈ 1.49e8 * S` |

Order of magnitude: **~3e7 MACs per predictor step, ~1.5e8 MACs per candidate rollout, ~1.5e8 · S MACs per P2 proof.** A small fixed-candidate set (`S = 8`) is ~1.2e9 MACs; `S = 32` is ~4.8e9 MACs. These figures are dominated by the depth-6 stack of `192<->2048` linears; everything else (action encoding, projection, MSE, argmin) is rounding error against it. This is *why MAC count sets trace size* and therefore prover time, memory, and proof size.

```text
P2 MAC envelope (V0 reference config)
  S=1   : 1.49e8
  S=4   : 5.96e8
  S=8   : 1.19e9
  S=16  : 2.39e9
  S=32  : 4.77e9
  S=64  : 9.54e9   <- one CEM iteration's worth at num_samples~300/(crude) is ~4.5e10
```

### INV-PERF-01 (MAC dominance)

`INV-PERF-01`: For the V0 reference configuration, predictor linear/matmul MACs account for ≥ 95% of total main-trace rows. The profiling harness ([#profiling](#profiling)) MUST report the matmul-row fraction; if it drops below 95% without a documented architecture change, that is a measurement bug or an unexpected cost regression and the build investigates before trusting any other budget number in this document.

### Accumulator-safety bound (normative, load-bearing)

The cost model assumes single-element M31 accumulators for the int8 hot path. That assumption is only valid inside a magnitude bound, and the bound is a hard correctness precondition, not a performance nicety.

A length-`N` dot product of signed int8 values has worst-case magnitude `N * 127 * 127`. For the widest V0 reduction (FFN down, `N = 2048`):

```text
2048 * 127 * 127 = 33,032,192  <  2^31 - 1 = 2,147,483,647
```

So a length-2048 signed-int8 dot product fits in a single signed M31 value with ~65x headroom. The latent-width reductions are smaller still (`192 * 127 * 127 = 3,096,768`). Therefore, *under strict int8 weights and int8 activations*, every V0 linear accumulator fits one M31 field element and needs no limb decomposition — this is what keeps each MAC to a single accumulate row.

### INV-PERF-02 (accumulator fits one M31)

`INV-PERF-02`: For any reduction in the proven graph, `(reduction_length) * max_abs_input * max_abs_weight + |bias_max|` MUST be `< 2^31 - 1` for the single-M31-accumulator layout to be used. The manifest's per-tensor bounds (see [`docs/spec/03-data-model.md#bounded-integers`](03-data-model.md#bounded-integers)) make this computable at trace-build time. If a reduction exceeds the bound — e.g. int16 activations (`2048 * 32767 * 127 ≈ 8.5e12`), wider biases, or longer reductions in P4's encoder — the trace builder MUST switch that op to a limb accumulator (`acc = acc_lo + 2^k * acc_hi`, each limb range-checked; see [`docs/spec/03-data-model.md#bounded-integers`](03-data-model.md#bounded-integers) and RFC-0002 / RFC-0005). Limb accumulators multiply that op's accumulate-row count by the limb count and are the primary mechanism by which int16/P4 configs cost more than the int8 V0 figures here. Violating this invariant without limbs is a soundness bug (silent field wraparound), not a perf bug; the failure mode and system response are enumerated in [#perf-gates](#perf-gates) and [`docs/spec/04-error-model.md#failure-modes`](04-error-model.md#failure-modes).

---

## trace-model

This section maps MACs and lookups onto trace rows, and rows onto proof size and prover memory. Row counts are *natural-layout estimates*: the actual layout is fixed by RFC-0005 (linear/matmul component) and may pack multiple MACs per row via tensorized columns; packing reduces row count at the cost of column width, but does not change the order of magnitude or the scaling laws. Treat these as upper bounds on rows and lower bounds on per-row work.

### Rows per component (per predictor step)

| Component | Row driver | Rows/step (natural layout) |
|---|---|---:|
| Linear + matmul (all projections, FFN) | ≈ 1 row per MAC | `≈ 2.97e7` |
| Activation lookups — GELU (FFN hidden) | `L * mlp_dim * depth = 3 * 2048 * 6` | `36,864` |
| Activation lookups — SiLU (AdaLN + Embedder) | `6 * latent_dim * depth = 6 * 192 * 6` | `6,912` |
| Activation lookups — softmax | `heads * L * L * depth = 16 * 9 * 6` | `864` |
| Range checks | ~1 per bounded column instance | bounded by row count above |
| Tensor-memory consistency | 1 read + 1 write per tensor cell touched | second-order |

Per step the activation lookups total ≈ `44,640` rows against ≈ `2.97e7` matmul rows — about 0.15%. This is the numerical justification for `INV-PERF-01`. Activation functions are NOT uniform: GELU in the FFN, SiLU in AdaLN modulation and the action `Embedder` (verified against upstream at 2026-06-03). Each distinct activation needs its own committed lookup table (see RFC-0006 and [`docs/spec/03-data-model.md#model-manifest`](03-data-model.md#model-manifest)); the table *sizes* (preprocessed columns) are fixed and do not scale with `S` or `horizon`, only the lookup *multiplicities* (interaction-trace rows) do.

### Cost MSE and argmin rows

| Component | Row driver | Rows |
|---|---|---:|
| MSE cost (per candidate) | `latent_dim` diffs + squares + accumulate ≈ `192` | `192` per candidate |
| Argmin | one range-checked `diff = cost_s - selected_cost` per candidate | `S` total |

MSE and argmin are negligible: `192 * S + S` rows against `1.49e8 * S` matmul-equivalent rows.

### Rows -> proof size -> prover memory

Let `R` be total main-trace rows (rounded up to the next power of two `N = 2^k` for the FRI domain). Then, qualitatively:

| Quantity | Relationship | Notes |
|---|---|---|
| Proof size | `O(log² N)` to `O(log N)` query openings + fixed transcript | FRI proofs are polylogarithmic in `N`; size grows slowly with rows. Dominated by query authentication paths and the per-component commitment count. |
| Prover peak memory | `O(N * (columns) * extension_width)` | The committed traces (preprocessed + main + interaction) and their low-degree extensions are the memory floor. Memory is roughly linear in rows × columns. |
| Prover time | `O(N log N)` (FFT/FRI) + `O(N * constraint_degree)` (quotient) | Quasilinear in rows; the LDE FFTs and Merkle hashing dominate wall time. |

Two consequences the budget relies on:

1. **Proof size scales far more slowly than trace size.** Doubling `S` roughly doubles `N` and prover time/memory but adds only a small additive term to proof size (more query openings if security parameters force a larger domain). This is what makes "prove all candidates" tractable on the verifier side even when prover cost is large.
2. **Memory, not size, is the binding constraint for large `S`.** Because peak memory is ~linear in committed rows, large `S` (or P4) hits an out-of-memory wall before it produces an unusably large proof. Mitigations are component decomposition / recursive aggregation (RFC-0012, deferred) and streaming trace generation; both are out of V0 scope and tracked as future work.

### INV-PERF-03 (trace-size accounting is reported, not estimated)

`INV-PERF-03`: The prover MUST emit, per proof, a per-component row count, the padded domain size `N`, and the matmul-row fraction, as structured fields (see [`docs/spec/05-observability.md#metrics`](05-observability.md#metrics): `trace_rows_total`, `trace_rows_by_component`, `padded_domain_log2`). Budgets in [#targets](#targets) are reconciled against these measured numbers, never against re-derived estimates, after every benchmark run.

---

## targets

These are **engineering budgets, not guarantees.** Owner: ProvableWorldModel maintainers. They define the envelope V0 aims to fit and the threshold above which a result is treated as a regression or an architecture problem worth escalating. They are explicitly subject to revision after the first end-to-end P2 benchmark on reference hardware.

### Reference benchmark point

The canonical budget point is a P2 proof of the V0 reference predictor with:

- `S = 8` fixed candidate action sequences,
- `horizon = 5`,
- int8 weights, int8 activations (single-M31 accumulators, `INV-PERF-02` satisfied),
- weights public/committed (no hiding),
- single-machine prover, no recursion.

This point is ≈ `1.19e9` MACs (`≈ 1.2e9` main-trace rows in natural layout, padded to `N = 2^31` is far too large — in practice tensorized packing and per-component domains keep each component's `N` well below this; see the open question below).

### INV-PERF-04 (V0 budget envelope)

`INV-PERF-04`: The reference benchmark point SHOULD fit the following envelope on reference hardware. Each cell is an initial budget pending first measurement.

| Budget | Initial target | Basis / status |
|---|---|---|
| Prove time (`prove_time_seconds`) | OPEN QUESTION (see below) | Cannot be grounded pre-benchmark; ~1.2e9 trace rows is large for a single trace and likely requires per-component proving and packing to land in minutes rather than hours. |
| Peak prover memory (`prover_peak_rss_bytes`) | OPEN QUESTION (see below) | Linear in committed rows × columns; the binding constraint. |
| Proof size (`proof_size_bytes`) | ≤ low single-digit MB | FRI proof size is polylogarithmic in `N`; even at ~1e9 rows the proof is expected in the MB range, not GB. Treated as the most groundable of the three but still to be confirmed. |
| Verify time (`verify_time_seconds`) | ≤ ~1 s, single-threaded | Verifier work is polylogarithmic; should be sub-second to low-seconds and is the easiest budget to meet. |

`OPEN QUESTION:` What are the concrete numeric prove-time and peak-memory budgets for the `S = 8`, `horizon = 5` reference point, and on what reference hardware (CPU model, core count, RAM)? Owner: maintainers (`area:prover` + `area:performance`). Resolution path: milestone **v1.0** — record the first measured value as the baseline (see [#perf-gates](#perf-gates)), then set the budget at a documented multiple of that baseline. Until then, no numeric prove-time or memory guarantee is published, and any prose claiming one is incorrect.

### Hardware and configuration assumptions (must be recorded with every benchmark)

A target number is meaningless without its environment. Every recorded baseline MUST capture:

```text
cpu_model, physical_cores, logical_cores, ram_bytes,
rustc_version, opt_level, target_cpu (native vs portable),
stwo_vendored_revision (third_party/stwo/REVISION),
S, horizon, weight_visibility, activation_dtype,
parallelism (rayon thread count), allocator
```

This block is part of the baseline artifact consumed by [#perf-gates](#perf-gates); see [`docs/spec/05-observability.md#metrics`](05-observability.md#metrics) for the emitted metric names and [`docs/spec/02-public-api.md#cli`](02-public-api.md#cli) for the prover CLI that produces it.

### Non-targets (explicitly out of the V0 envelope)

| Out of scope for V0 budget | Why | Tracked in |
|---|---|---|
| P3 / CEM proving cost | `n_steps * num_samples = 9000` candidate-rollouts; orders of magnitude larger | [#scaling](#scaling), RFC-0010 |
| P4 / pixel encoder cost | ViT over 224×224 / patch-14 tokens; attention over many image tokens dominates and likely forces int16 + limb accumulators | RFC-0011, [`docs/spec/00-overview.md#scope-and-statement-tiers`](00-overview.md#scope-and-statement-tiers) |
| Recursive aggregation cost | Deferred proof-decomposition shape | RFC-0012 |
| Zero-knowledge / hiding overhead | V0 is a succinct validity proof, not ZK | [`docs/spec/06-security.md#privacy-and-zk`](06-security.md#privacy-and-zk) |

---

## scaling

Proof cost in trace rows (and therefore prover time and memory) is, to first order:

```text
rows(P2)  ≈  K_step * depth * horizon * S          (+ lower-order MSE/argmin terms)
          ≈  (per-step rows) * horizon * S
```

with `per-step rows ≈ 3.0e7` for the V0 reference config. The two free scaling axes for V0 are `S` and `horizon`.

| Axis | Effect on trace rows / prover cost | Effect on proof size |
|---|---|---|
| `S` (candidate count) | **Linear.** Each candidate is an independent rollout; doubling `S` doubles rollout rows. | Sub-linear (polylog in `N`); a few more FRI queries at most. |
| `horizon` (rollout length) | **Linear.** Each step is one full predictor pass (`≈ 3.0e7` rows). | Sub-linear. |
| `depth` (predictor blocks) | Linear (fixed at 6 for V0). | Sub-linear. |
| `latent_dim`, `mlp_dim` | Each linear scales as product of its two dims; FFN is `O(latent_dim * mlp_dim)`. | Sub-linear. |
| `history_size` / `L` (attention seq len) | Projections linear in `L`; score/value matmuls quadratic in `L` but tiny at `L=3`. Only matters if `L` grows (e.g. P4). | Sub-linear. |

### Why CEM (P3) is deferred — the 9000× multiplier

The CEM planner does not score a fixed candidate set once. Verified against upstream at 2026-06-03, the le-wm solver config (`config/eval/solver/cem.yaml`) uses `num_samples = 300`, `n_steps = 30`, `topk = 30`, `var_scale = 1.0`, `batch_size = 1`; the task `plan_config` (`config/eval/pusht.yaml`) sets `horizon = 5`, `receding_horizon = 5`, `action_block = 5`. A sound CEM proof (RFC-0010) must prove *every* candidate rollout across *every* iteration, because proving only the surviving elites or the final action lets the prover select favorable candidates off-circuit (planner-soundness requirement, [`docs/spec/06-security.md#soundness-requirements`](06-security.md#soundness-requirements)).

```text
CEM candidate-rollouts = n_steps * num_samples = 30 * 300 = 9000
```

So a faithful CEM proof costs **≈ 9000 candidate-rollouts** versus P2's `S` (single-digit to low-tens). At `S = 8`, P2 is `8` rollouts; CEM is `9000` — roughly a 1000×+ increase in trace rows, before accounting for the additional sampling, clipping, top-k, and mean/variance-update arithmetic CEM also must prove. That multiplier, plus the cost of proving seeded Gaussian sampling and square roots soundly, is the concrete reason P3/CEM is out of V0 and tracked in RFC-0010 and [`docs/spec/00-overview.md#scope-and-statement-tiers`](00-overview.md#scope-and-statement-tiers). V0 proves fixed-candidate planning (P2); the relationship "the proven candidate set was itself produced by CEM" is explicitly not claimed by V0.

### INV-PERF-05 (cost is linear in S and horizon)

`INV-PERF-05`: Measured trace rows for P2 MUST be linear (within ±10% after padding/packing effects) in both `S` and `horizon`. The profiling harness sweeps `S ∈ {1,2,4,8}` and `horizon ∈ {1,2,5}` and fits the line; a super-linear fit indicates either an accidental quadratic (e.g. an all-pairs candidate comparison instead of the `O(S)` argmin) or a layout bug, and blocks the release until explained. This invariant is what justifies budgeting P2 by a single per-rollout constant.

---

## profiling

The profiling plan exists to (1) ground the OPEN QUESTION targets in [#targets](#targets) with real numbers, (2) verify the scaling invariants, and (3) feed the regression gates in [#perf-gates](#perf-gates) a trustworthy baseline.

### Tooling

| Tool | Use | Crate / location |
|---|---|---|
| `criterion` micro-benchmarks | Per-component wall-time and throughput with statistical confidence intervals; tracks regressions across runs. | `crates/pwm-air/benches/`, `crates/pwm-prover/benches/` |
| Trace-size accounting | Per-component row counts, padded domain size, matmul-row fraction (emits `INV-PERF-03` fields). | `pwm-prover` trace builder, structured output |
| Memory profiling | Peak RSS via a custom global allocator counter or `dhat`/heaptrack under CI; reports `prover_peak_rss_bytes`. | `crates/pwm-prover` (feature-gated `dhat-heap`) |
| Flamegraphs | `cargo flamegraph` / `perf` for hot-path attribution (FFT vs Merkle vs quotient vs witness gen). | manual, on-demand, documented in run logs |

### Benchmark suite (the named benchmarks)

```text
bench_linear            # one 192->2048 linear, varying batch
bench_matmul            # attention score/value matmul
bench_requant           # requantization tail (quotient/remainder + round)
bench_activation_lookup # GELU, SiLU, softmax table lookups (separately)
bench_predictor_block   # one ConditionalBlock (attn + FFN + AdaLN)
bench_predictor_step    # full depth-6 step (P0-scale)
bench_rollout           # horizon sweep {1,2,5} (P1-scale)
bench_cost_argmin       # MSE + argmin over S candidates
bench_p2_end_to_end     # S sweep {1,2,4,8}, horizon=5 (the budget point)
```

Each component benchmark reports: wall time, peak memory, trace rows, and (for the end-to-end bench) proof size. The end-to-end benchmark's `S`- and `horizon`-sweeps directly test `INV-PERF-05`.

### Profiling procedure

1. Build with a recorded, reproducible toolchain (pin `rustc`, `opt-level`, `target-cpu`; record per [#targets](#targets) assumptions block).
2. Run `bench_p2_end_to_end` at the reference point; capture all metrics from [`docs/spec/05-observability.md#metrics`](05-observability.md#metrics).
3. Run the `S`/`horizon` sweeps; fit linearity, assert `INV-PERF-05`.
4. Attribute the top 3 hot functions via flamegraph; confirm they are FFT/Merkle/quotient (expected) and not witness generation or serialization (would indicate a non-prover bottleneck to fix first).
5. Record the result as the named baseline artifact (a committed JSON under the perf-baseline path), which `[#perf-gates]` compares against.

### INV-PERF-06 (baseline is recorded and reproducible)

`INV-PERF-06`: There MUST exist exactly one committed performance baseline artifact per release line, containing `proof_size_bytes`, `prove_time_seconds`, `prover_peak_rss_bytes`, `verify_time_seconds`, the full hardware/config assumptions block, and the `stwo` vendored revision. Re-running the benchmark on the same hardware and revision MUST reproduce these within the gate tolerance (see [#perf-gates](#perf-gates)); non-reproducibility at the recorded revision is itself a failure that blocks updating the baseline.

---

## perf-gates

Performance regression gates run in CI and are part of the release gate set. They are the enforcement arm of this document. They live alongside the correctness gates in [`docs/spec/07-testing-strategy.md#ci-gates`](07-testing-strategy.md#ci-gates) and consume the metrics defined in [`docs/spec/05-observability.md#metrics`](05-observability.md#metrics).

### Tracked quantities and thresholds

| Metric | Gate | Default threshold `X` | Rationale |
|---|---|---|---|
| `proof_size_bytes` | Fail build if `> baseline * (1 + X)` | `X = 0.05` (5%) | Proof size is deterministic for fixed inputs/revision; any growth is a structural change that needs review. Low tolerance. |
| `prove_time_seconds` | Fail build if `> baseline * (1 + X)` | `X = 0.20` (20%) | Wall time is noisier (scheduler, thermal); higher tolerance, but a sustained climb is a regression. |
| `prover_peak_rss_bytes` | Fail build if `> baseline * (1 + X)` | `X = 0.10` (10%) | Memory is the binding constraint at scale; modest tolerance. |
| `trace_rows_total` / matmul fraction | Fail build if matmul fraction `< 95%` or rows deviate `> 10%` non-linearly with `S`/`horizon` | enforces `INV-PERF-01`, `INV-PERF-05` | Catches accidental quadratics and layout regressions deterministically (row counts are exact, not timing-dependent). |

`proof_size_bytes` and `trace_rows_total` are *deterministic* given fixed inputs and a fixed `stwo` revision, so their gates are exact and not subject to timing noise — they are the primary, hard gates. `prove_time_seconds` and `prover_peak_rss_bytes` are environment-sensitive; their gates run only on the designated baseline runner (recorded per [#targets](#targets)) and are advisory-but-blocking with the wider tolerances above.

The threshold `X` per metric is configurable in the CI gate config; the values above are the initial defaults. `OPEN QUESTION:` Should `prove_time_seconds`/`prover_peak_rss_bytes` gates block PR merges or only release tags, given runner noise? Owner: maintainers (`area:ci`). Resolution path: milestone **v1.0** — decide after observing baseline-runner variance across the first 20 CI runs; default until then is *block release tags only* for the timing/memory gates, *block every PR* for the deterministic size/row gates.

### Failure modes and system response

| Failure mode | Detection | System response |
|---|---|---|
| Proof size grew beyond threshold | `proof_size_bytes > baseline*(1+X)` in CI | **Fail the build.** Require either a justified baseline bump (PR updating the baseline artifact with rationale, reviewed) or a fix. Never silently re-baseline. |
| Prove time / memory regressed | timing/memory metric over threshold on baseline runner | **Fail the gated build** (per the open-question policy: release tag, or PR if so configured). Attach the flamegraph diff to the failure. |
| Matmul fraction dropped below 95% (`INV-PERF-01`) | row accounting | **Fail the build.** Indicates either a measurement bug or that non-matmul work unexpectedly dominates; investigate before trusting other budgets. |
| Super-linear scaling in `S`/`horizon` (`INV-PERF-05`) | sweep linearity fit | **Fail the build.** Almost always an accidental quadratic; block release until explained. |
| Accumulator would overflow single M31 but trace used single accumulator (`INV-PERF-02`) | trace-build precondition check from manifest bounds | **Reject at trace-build time** (not a perf gate — a correctness gate; surfaced as a build/prove error per [`docs/spec/04-error-model.md#failure-modes`](04-error-model.md#failure-modes)). The trace builder MUST switch to limb accumulators or abort; it MUST NOT emit a wrapping accumulator. |
| Baseline artifact missing or for a different `stwo` revision | gate prelude check | **Fail the build** with an explicit "no comparable baseline" error; require recording a baseline at the current revision before perf gates can pass. |
| Out-of-memory during prove at large `S` | OOM / allocator failure | **Hard prove failure** (see [`docs/spec/04-error-model.md#failure-modes`](04-error-model.md#failure-modes)). Documented mitigation is reducing `S`, or deferring to component decomposition / recursion (RFC-0012); V0 publishes a maximum supported `S` once benchmarked. |

### INV-PERF-07 (no silent re-baselining)

`INV-PERF-07`: The performance baseline artifact MUST only change via an explicit, reviewed commit that states the cause of the change (architecture change, dependency bump, hardware change) and includes the new measured numbers and assumptions block. CI MUST NOT auto-update the baseline on a regression. This prevents a slow, unnoticed drift from defeating every other gate in this section.

---

## Cross-references

- Architecture and AIR components (`LinearComponent`, `MatMulComponent`, trace model): [`docs/spec/01-architecture.md#component-model`](01-architecture.md#component-model), [`docs/spec/01-architecture.md#trace-model`](01-architecture.md#trace-model)
- Public API / CLI emitting metrics and baselines: [`docs/spec/02-public-api.md#cli`](02-public-api.md#cli)
- Bounded integers, accumulators, limb decomposition: [`docs/spec/03-data-model.md#bounded-integers`](03-data-model.md#bounded-integers)
- Manifest binding of activation tables, scales, bounds: [`docs/spec/03-data-model.md#model-manifest`](03-data-model.md#model-manifest)
- Error taxonomy / failure responses referenced above: [`docs/spec/04-error-model.md#failure-modes`](04-error-model.md#failure-modes)
- Metric names tracked by gates and invariants: [`docs/spec/05-observability.md#metrics`](05-observability.md#metrics)
- Planner-soundness requirement (why all candidates are proven): [`docs/spec/06-security.md#soundness-requirements`](06-security.md#soundness-requirements)
- ZK/privacy boundary (no hiding overhead in V0): [`docs/spec/06-security.md#privacy-and-zk`](06-security.md#privacy-and-zk)
- CI gate framework: [`docs/spec/07-testing-strategy.md#ci-gates`](07-testing-strategy.md#ci-gates)
- Scope tiers / why P3/P4 deferred: [`docs/spec/00-overview.md#scope-and-statement-tiers`](00-overview.md#scope-and-statement-tiers)
- CEM proof requirements and the 9000× cost: RFC-0010 (`docs/rfcs/RFC-0010-cem-planner-proof.md`)
- Pixel encoder cost / int16 + limb implications: RFC-0011 (`docs/rfcs/RFC-0011-pixel-encoder-proof.md`)
- Recursive aggregation as the memory-wall mitigation: RFC-0012 (`docs/rfcs/RFC-0012-recursive-aggregated-verification.md`)
- Linear/matmul accumulator strategy: RFC-0005 (`docs/rfcs/RFC-0005-linear-matmul-and-requantization-components.md`)
- Founding analysis: [`docs/feasibility-study.md`](../feasibility-study.md) §2, §5.2, §6, §13
