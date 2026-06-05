# 07 — Testing Strategy

Status: Normative. Defines the mandatory test architecture, golden-vector
discipline, negative/differential/mutation testing, and the CI merge gates for
ProvableWorldModel. Governs `area:testing` and is the operational counterpart to
the soundness requirements in
[`docs/spec/06-security.md#soundness-requirements`](06-security.md#soundness-requirements)
and the error taxonomy in
[`docs/spec/04-error-model.md#verifier-rejections`](04-error-model.md#verifier-rejections).
The governing decision record is
[`docs/rfcs/RFC-0013-testing-fuzzing-and-audit-strategy.md`](../rfcs/RFC-0013-testing-fuzzing-and-audit-strategy.md).

---

## 0. Why testing is load-bearing here

ProvableWorldModel proves an exact arithmetic relation: that a claimed quantized
LeWorldModel rollout, cost, and argmin are the bit-for-bit result of the
algorithm bound by a committed manifest (see
[`docs/spec/00-overview.md#v0-statement`](00-overview.md#v0-statement)). Two
failure classes dominate, and ordinary ML testing covers neither:

1. **Reference drift.** The Python floating-point model, the Python fixed-point
   reference, the Rust fixed-point reference, and the AIR must all agree on the
   same integers. If any pair diverges by one ULP, the proof either proves the
   wrong relation or cannot be produced. Standard accuracy metrics (loss, MSE)
   hide single-bit divergence; only bit-equality assertions catch it.
2. **Silent unsoundness.** A constraint that is missing, too weak, or wired to
   the wrong column can still *accept every honest witness*. Accepting tests
   pass, the demo works, and the system is unsound. The only defense is
   adversarial: every constraint must be shown to *reject* a witness that
   violates it.

The governing rule of this document, inherited verbatim from RFC-0013:

> **INV-TEST-01 — Dual-test rule.** No component, AIR constraint, or primitive
> ships without BOTH an accepting test (valid witness verifies) AND at least one
> rejecting test (a witness violating that component's contract is rejected with
> a specific `VerifyError`). A component with only accepting tests is treated as
> untested and MUST NOT merge.

This document is exhaustive by design. A contributor implementing any
component must be able to read this file and know exactly which tests are
mandatory, what each proves, where it lives, and which CI gate enforces it.

---

<a id="test-pyramid"></a>

## Test pyramid
### The 10-layer test stack (from RFC-0013)

The stack is ordered by cost and scope, from cheap deterministic unit checks at
the base to expensive cross-language and adversarial checks at the top. Each
layer has a single thing it proves, a home crate/directory, and a CI gate that
runs it. Layers are not optional substitutes for each other; a component is
exercised by every layer that applies to it.

```text
 (10) Security review / audit        manual + fuzz harness     pre-release gate
 (9)  Manifest serialization         pwm-core + pwm-export      every PR
 (8)  Constraint mutation            mutation runner (CI job)   nightly + gate
 (7)  Differential (Py <-> Rust)     pwm-export + pwm-air       every PR
 (6)  Negative / reject              pwm-air + pwm-verifier     every PR
 (5)  Prover/verifier accept         pwm-prover + pwm-verifier  every PR
 (4)  AIR witness-generation         pwm-air + pwm-circuits     every PR
 (3)  Rust fixed-point reference     pwm-core + pwm-export      every PR
 (2)  Python fixed-point reference   pwm-export (pytest)        every PR
 (1)  Python floating-point ref      pwm-export (pytest)        every PR
```

The same stack, with the precise contract of each layer:

| # | Layer | What it proves | Lives in | Primary harness |
|---|-------|----------------|----------|-----------------|
| 1 | Python FP reference | The exported graph matches PyTorch *semantics* (operator order, shapes, eval-mode behavior) within a declared FP tolerance. NOT a soundness layer — it anchors intent, not bits. | `pwm-export/` (Python) | `pytest`, `torch` eager |
| 2 | Python fixed-point reference | The quantized integer algorithm (signed encoding, requantize, rounding, clamp, lookup tables) is fully specified and self-consistent in Python. Produces the canonical golden integers. | `pwm-export/` (Python) | `pytest` |
| 3 | Rust fixed-point reference | The Rust reference (`pwm-core` fixed-point + `pwm-export` graph executor) reproduces the Python fixed-point integers bit-for-bit. This is the parity boundary the AIR is built against. | `pwm-core/`, `pwm-export/` | `cargo test` + golden fixtures |
| 4 | AIR witness-generation | For a valid input, the trace builder produces a witness that *satisfies every constraint of every component* (range, lookup, memory, linear, requant, nonlinear, rollout, cost, argmin) — checked by the constraint-framework evaluator without running FRI. | `pwm-air/`, `pwm-circuits/` | `cargo test`, constraint eval |
| 5 | Prover/verifier accept | A full proof produced by `pwm-prover` is accepted by `verify()` (see [`docs/spec/02-public-api.md#rust-public-api`](02-public-api.md#rust-public-api)), including Fiat-Shamir transcript replay and FRI. | `pwm-prover/`, `pwm-verifier/` | `cargo test` (integration) |
| 6 | Negative / reject | Every enumerated tampering (see [`#negative-tests`](#negative-tests)) yields a *specific* `VerifyError`, not a generic failure and never an accept. | `pwm-air/`, `pwm-verifier/` | `cargo test` (integration) |
| 7 | Differential | Across randomized seeds and input distributions, Python fixed-point and Rust fixed-point agree bit-for-bit; property tests hold for range-safety, requantize rounding, argmin/tie-break. | `pwm-export/`, `pwm-air/` | `proptest` (Rust), `hypothesis` (Python) |
| 8 | Constraint mutation | The constraint *suite* is adequate: mutating any single constraint causes at least one rejecting test to fail. Measured as a CI metric. | mutation runner | custom `cargo` mutation job |
| 9 | Manifest serialization | Canonical serialization is deterministic, byte-identical on re-export, and round-trips; commitments change iff bound content changes. | `pwm-core/`, `pwm-export/` | `cargo test`, `pytest` |
| 10 | Security review / audit | Threat-model coverage, fuzzing of the verifier and deserializers, soundness review of new constraints, dependency/audit-boundary review (RFC-0015). | manual + fuzz harness | `cargo fuzz`, manual review |

```text
            ┌──────────────────────────────┐
            │ (10) Security review / audit  │   pre-release, manual + fuzz
            ├──────────────────────────────┤
            │ (8) Constraint mutation       │   nightly + merge gate (metric)
            ├──────────────────────────────┤
            │ (7) Differential  (6) Negative│   adversarial, every PR
            ├──────────────────────────────┤
            │ (5) Prover/verifier accept    │   full proof, every PR
            ├──────────────────────────────┤
            │ (4) AIR witness-generation    │   constraint eval, every PR
            ├──────────────────────────────┤
            │ (1)(2)(3) FP / fixed-point refs│  bit-parity base, every PR
            └──────────────────────────────┘
   wide base = many fast deterministic tests; narrow top = few, costly, manual
```

ASCII note: the pyramid is wide at the base (layers 1–4 are many, fast, fully
deterministic) and narrow at the top (layers 8 and 10 are few and expensive).
The adversarial layers 6, 7, and 8 are the soundness core; they are not optional
polish.

### Named invariants for the pyramid

| Invariant | Statement |
|-----------|-----------|
| **INV-TEST-01** | Dual-test rule (stated in §0): no component ships without both accepting and rejecting tests. |
| **INV-TEST-02** | Bit-equality chain. Layers 2 → 3 (Python fixed-point → Rust fixed-point) MUST agree on every output integer for every golden vector; tolerance is exactly zero. See [`#golden-vectors`](#golden-vectors). |
| **INV-TEST-03** | Determinism. Every layer except layer 1 is a pure function of its committed inputs: same inputs ⇒ identical bytes/integers, on any platform, in any thread, with no environment dependence. Cross-references [`docs/rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md`](../rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md). |
| **INV-TEST-04** | Constraint-evaluator before FRI. Layer 4 (constraint evaluation) MUST pass before a component's layer-5 proof is attempted; a constraint-eval failure is a build bug, not a proving bug, and is debugged at layer 4. |
| **INV-TEST-05** | Mutation adequacy. The constraint-mutation score (layer 8) MUST meet the target in [`#mutation-tests`](#mutation-tests); a regression below target blocks merge. |

### Test taxonomy versus the test pyramid

The 10 layers map onto the conventional unit/integration/property/ML axes so
contributors know which tool to reach for:

| Conventional class | Layers | Tooling | Notes |
|--------------------|--------|---------|-------|
| Unit | 2, 3, 4, 9 | `pytest`, `cargo test` | One primitive or one constraint in isolation. |
| Integration | 5, 6 | `cargo test` end-to-end | Full prove → verify for P0/P1/P2. |
| Property-based | 7 | `proptest`, `hypothesis` | Range-safety, rounding, argmin/tie-break invariants over generated inputs. |
| Differential | 1, 3, 7 | golden fixtures + generators | FP↔fixed semantic check (1↔2), Python↔Rust bit-parity (2↔3, 7). |
| ML-specific | 1, 2 | `pytest` + `torch` | Eval-mode handling (dropout off, BatchNorm folded), per-module activation correctness (GELU in FFN, SiLU in AdaLN/Embedder). |
| Adversarial / soundness | 6, 8, 10 | reject suite, mutation, fuzz | The dual-test and mutation guarantees. |

---

<a id="golden-vectors"></a>

## Golden vectors
Golden vectors are committed, versioned, deterministic fixtures that pin the
exact integers each reference must produce. They are the mechanism that enforces
INV-TEST-02. The feasibility study calls for a bit-for-bit chain across three
references; this section makes that operational.

### The bit-for-bit reference chain

```text
   PyTorch checkpoint (le-wm, MIT)
            │  export (pwm-export, Python)
            ▼
   (1) Python FP reference  ── tolerance: declared FP epsilon (semantic anchor)
            │  quantize (per manifest: scales, rounding, clamp, tables)
            ▼
   (2) Python fixed-point reference  ── emits golden integers
            │  COMMIT golden vectors to repo
            ▼
   (3) Rust fixed-point reference  ── MUST equal (2) bit-for-bit (tolerance 0)
            │  trace builder
            ▼
   (4) AIR witness  ── constraints satisfied iff integers match (3)
```

Prose restatement: PyTorch defines the *intent*. The Python fixed-point pass
turns intent into the exact integers the manifest's quantization rules dictate,
and those integers are committed as golden vectors. The Rust fixed-point
reference must reproduce them with zero tolerance, and the AIR is built so its
constraints are satisfiable only by exactly those integers. Layer 1 → 2 allows a
declared floating-point tolerance because quantization is lossy by construction;
every step after layer 2 has tolerance zero.

The link from layer 1 to layer 2 is governed by
[`docs/rfcs/RFC-0001-model-manifest-and-export-pipeline.md`](../rfcs/RFC-0001-model-manifest-and-export-pipeline.md)
(export pipeline) and
[`docs/rfcs/RFC-0002-fixed-point-arithmetic-over-m31.md`](../rfcs/RFC-0002-fixed-point-arithmetic-over-m31.md)
(arithmetic semantics). The FP tolerance is a quality metric, not a soundness
metric: a large FP↔fixed gap means the quantization is inaccurate, not unsound.

### Per-primitive golden vectors

Every primitive ships golden vectors with explicit, declared domain bounds.
"Explicit domain bounds" means each input field carries its inclusive
`[lo, hi]` range (the `BoundedInt` bounds from
[`docs/spec/03-data-model.md#bounded-integers`](03-data-model.md#bounded-integers)),
and vectors cover the interior plus the boundary and one-past-the-boundary cases.

| Primitive | Golden-vector inputs | Domain bounds to pin | Governing RFC |
|-----------|----------------------|----------------------|---------------|
| `add` / `sub` | operand pairs | per-tensor `[lo, hi]`; near-`±B` to probe wrap | RFC-0002 |
| `mul` | operand pairs | product range; max `|x*w|` | RFC-0002 |
| accumulate | accumulator sequences | int32 / limb boundary (e.g. dot-product len 2048, int8: `2048·127·127 = 33,032,192 < 2^31−1`) | RFC-0002, RFC-0005 |
| requantize | `(n, shift)` pairs | `0 ≤ rem < 2^shift`; ties for `NearestTiesToEven`; sign cases for `TruncateTowardZero` | RFC-0002 |
| clamp | values vs clamp range | exactly `lo`, `hi`, `lo−1`, `hi+1` | RFC-0002 |
| compare / argmin diff | `(a,b)` pairs | `d = b−a ∈ [0, D_max]`; `d = 0` (tie); `d = −1` (must reject) | RFC-0002, RFC-0009 |
| GELU_Q (FFN) | `(x, y)` table rows | declared input domain + scale; saturating tails | RFC-0006 |
| SiLU_Q (AdaLN, Embedder) | `(x, y)` table rows | declared input domain + scale; SiLU is distinct from GELU — see note below | RFC-0006 |
| Softmax_Q | score vectors over the attention window | quantized score domain; masked `NEG_INF_Q`; reciprocal of denominator | RFC-0006 |
| LayerNorm_Q / AdaLN_Q | feature vectors | mean/variance/`inv_std` domain; `eps`; reciprocal-sqrt table or Newton bounds | RFC-0006 |
| linear / matmul | `(x, w, bias)` blocks | per-axis batching (candidate, step, head, channel) | RFC-0005 |
| MSE cost | `(z_final, z_goal)` pairs | `diff`, `sq`, running `cost` ranges (latent_dim = 192) | RFC-0009 |
| rollout window | latent/action histories (history_size = 3) | window-selection indices; horizon = 5, action_block = 5 | RFC-0008 |

> **Activation note (verified against upstream at 2026-06-03).** LeWorldModel
> activations are NOT uniform. The predictor FFN/MLP uses **GELU**; the AdaLN
> modulation and the action `Embedder` use **SiLU**. Golden vectors and lookup
> tables MUST be generated per-module with the correct function. A test that
> applies GELU to an AdaLN/Embedder path (or vice versa) is a known bug and is
> itself covered by a negative differential check at layer 7.

### Generation, storage, and versioning

```text
pwm-export/
  golden/
    REVISION                         # le-wm checkpoint rev + export tool version
    manifest_v1/
      relation=pwm.lewm.predictor_step.v1/
        primitives/
          requantize.ntte.json       # per-primitive vectors
          gelu_q.ffn.json
          silu_q.adaln.json
          ...
        layers/
          predictor.block0.attn.json # per-op composed vectors
          ...
        end_to_end/
          p0_step.json               # full P0 input -> claimed output
          p2_planning.json           # full P2 input -> selected_index/cost
```

Rules:

- **Generation.** Golden vectors are produced *only* by the Python fixed-point
  reference (layer 2), driven by the committed manifest, by a single command:
  `python -m pwm_export.golden --manifest <path> --out pwm-export/golden/...`.
  Generation is deterministic (INV-TEST-03): fixed RNG seed recorded in the
  fixture header, no wall-clock, no GPU, CPU integer math only.
- **Storage.** Vectors are committed to the repository as canonical JSON
  (sorted keys, no floats in the fixed-point section — integers only). Each
  fixture file header records: `relation_id`, `manifest_version`,
  `quantization_commitment`, the export tool version, and the generator seed.
- **Versioning.** Golden vectors are versioned by `relation_id` and
  `manifest_version`. Because `relation_id` is immutable (a semantic change mints
  a new id, per [`docs/spec/00-overview.md#v0-statement`](00-overview.md#v0-statement)
  and [`docs/spec/09-release-and-versioning.md#relation-versioning`](09-release-and-versioning.md#relation-versioning)),
  golden vectors for a shipped relation are append-only: they are never edited in
  place. A change that alters any golden integer for an existing `relation_id`
  is a soundness regression and MUST instead introduce a new `relation_id` with
  a fresh vector set. Regenerating vectors that change committed integers without
  bumping `relation_id` is a CI-blocking error.

### Golden-vector invariants

| Invariant | Statement | Enforced by |
|-----------|-----------|-------------|
| **INV-TEST-06** | Layer 3 reproduces every committed golden integer with tolerance 0 for every shipped `relation_id`. | layer-3 `cargo test` (golden parity gate) |
| **INV-TEST-07** | Re-running golden generation on the same checkpoint + manifest yields byte-identical fixtures (else generation is non-deterministic). | layer-9 + RFC-0016 reproducibility check |
| **INV-TEST-08** | Golden integers for a shipped `relation_id` are immutable; changing them requires a new `relation_id`. | golden-immutability CI check (diff against committed set) |

---

<a id="negative-tests"></a>

## Negative / reject tests
Negative tests are the soundness backbone (INV-TEST-01). Each test takes a known-
good witness/proof, applies exactly one tampering, and asserts the verifier
rejects with a *specific* `VerifyError` from
[`docs/spec/04-error-model.md#verifier-rejections`](04-error-model.md#verifier-rejections).
Two anti-patterns are forbidden: (a) a tamper that still verifies (unsound —
INV-TEST-01 violated); (b) a tamper that rejects with the *wrong* or a *generic*
error (masks the true failure, defeats the error taxonomy).

### The enumerated reject suite

This is the full mandatory list. Every entry maps to a `VerifyError` variant.
The named variants below are the canonical contract that
[`docs/spec/04-error-model.md#verifier-rejections`](04-error-model.md#verifier-rejections)
defines; this document is the authoritative *test* list for them.

| # | Tampering | Layer / where injected | Expected `VerifyError` | Governing RFC |
|---|-----------|------------------------|------------------------|---------------|
| N1 | Wrong model commitment | flip a byte in `PublicInput.model_commitment` | `ModelCommitmentMismatch` | RFC-0000, RFC-0001 |
| N2 | Wrong quantization commitment | flip `PublicInput.quantization_commitment` | `QuantizationCommitmentMismatch` | RFC-0001, RFC-0002 |
| N3 | Wrong action | mutate one `candidate_actions` cell, keep proof for old | `ConstraintUnsatisfied{component:"rollout"}` or `PublicInputDigestMismatch` | RFC-0007, RFC-0008 |
| N4 | Wrong latent | mutate one `latent_history` cell | `ConstraintUnsatisfied{component:"rollout"}` / digest mismatch | RFC-0008 |
| N5 | Wrong intermediate activation | mutate one `predictor_activations` value | `ConstraintUnsatisfied{component:"<block>"}` | RFC-0006, RFC-0007 |
| N6 | Wrong accumulator | mutate `acc` mid-dot-product (keep `acc_next` honest) | `ConstraintUnsatisfied{component:"linear"}` | RFC-0005 |
| N7 | Wrong rounding remainder | set `rem ≥ 2^shift`, or wrong tie resolution | `ConstraintUnsatisfied{component:"requant"}` / `RangeCheckFailed` | RFC-0002 |
| N8 | Wrong activation lookup value | claim `(x, y)` not in committed table | `LookupArgumentFailed{table:"gelu"/"silu"}` | RFC-0003, RFC-0006 |
| N9 | Wrong LayerNorm reciprocal | supply `inv_std` not satisfying the reciprocal-sqrt constraint | `ConstraintUnsatisfied{component:"layernorm"}` | RFC-0006 |
| N10 | Wrong attention probability | supply `prob` row that does not match Softmax_Q constraint / denominator | `ConstraintUnsatisfied{component:"attention"}` / `LookupArgumentFailed` | RFC-0006 |
| N11 | Wrong candidate cost | mutate one `costs[s]` away from MSE_Q output | `ConstraintUnsatisfied{component:"cost"}` | RFC-0009 |
| N12 | Wrong argmin | declare `selected_index` whose cost is not minimal | `ConstraintUnsatisfied{component:"argmin"}` (a `diff < 0` range check fails) | RFC-0009 |
| N13 | Wrong tie-break | among equal minima, pick a non-smallest index | `ConstraintUnsatisfied{component:"argmin",rule:"tie_break"}` | RFC-0009 |
| N14 | Out-of-range selected_index | `selected_index ≥ S` | `SelectedIndexOutOfRange` | RFC-0009 |
| N15 | Statement-type confusion (P0-as-P1) | submit a P0 proof with `statement_type = P1Rollout` (or any `relation_id` mismatch) | `RelationIdMismatch` / `StatementTypeMismatch` | RFC-0000 |

Additional mandatory rejects derived from the soundness requirements in
[`docs/spec/06-security.md#binding-requirements`](06-security.md#binding-requirements)
and [`docs/spec/06-security.md#range-safety`](06-security.md#soundness-requirements):

| # | Tampering | Expected `VerifyError` | Governing RFC |
|---|-----------|------------------------|---------------|
| N16 | Field-wrap masking: choose an integer outside `[lo,hi]` whose `mod p` value collides with a valid one | `RangeCheckFailed` | RFC-0002, RFC-0003 |
| N17 | Wrong scale_id on a tensor read (right value, wrong scale) | `ConstraintUnsatisfied{component:"tensor_memory"}` | RFC-0004 |
| N18 | Tensor-memory read with no matching write (broadcast without manifest permission) | `MemoryConsistencyFailed` (permutation/multiset) | RFC-0004 |
| N19 | Partial planner proof: prove only the selected rollout, omit other candidates | `ConstraintUnsatisfied{component:"planner"}` / missing-candidate | RFC-0009 |
| N20 | Lookup multiplicity tamper: claimed LogUp sum inconsistent | `LookupArgumentFailed{kind:"multiplicity"}` | RFC-0003 |
| N21 | Fiat-Shamir transcript divergence: reorder/inject channel data | `TranscriptMismatch` / FRI failure | RFC-0014 |
| N22 | Wrong `relation_id` for a structurally valid proof | `RelationIdMismatch` | RFC-0000, RFC-0014 |
| N23 | Artifact schema-version mismatch (`ProofArtifact.artifact_version` unknown) | `ArtifactVersionUnsupported` | RFC-0016 |

> **OPEN QUESTION (owner: area:verifier maintainer; resolution: RFC-0014).** The
> exact boundary between `ConstraintUnsatisfied` and `PublicInputDigestMismatch`
> for input tampering (N3/N4) depends on whether the tampered input is bound by a
> public digest or carried as a public vector. The split is settled when RFC-0014
> fixes the public-input binding for each `PublicInput` field; until then,
> negative tests assert "rejects with one of {`ConstraintUnsatisfied`,
> `PublicInputDigestMismatch`}" and the test is tightened when RFC-0014 lands.

### Negative-test invariants

| Invariant | Statement |
|-----------|-----------|
| **INV-TEST-09** | Every reject in N1–N23 produces a *specific* `VerifyError`; a generic/`Other` error or an accept fails the test. |
| **INV-TEST-10** | Every AIR component listed in `pwm-air` (range_check, tensor_memory, linear, matmul, requant, activation_lookup, layernorm, attention, mlp, predictor, rollout, cost, argmin) has at least one reject test exercising at least one of its constraints. Components with zero reject tests fail the dual-test gate. |
| **INV-TEST-11** | The verifier NEVER runs PyTorch or any reference inference to reject; rejection is decided purely from the proof, public input, and committed preprocessed trace (mirrors [`docs/spec/06-security.md#trust-boundaries`](06-security.md#trust-boundaries)). |

---

<a id="differential-tests"></a>

## Differential tests
Differential testing is randomized cross-checking. It catches divergences that
fixed golden vectors miss because it explores the input space the author did not
think to pin.

### Python ↔ Rust random-seed differential

```text
for seed in CI_SEED_SET:                 # seed set committed, expanded nightly
    inputs = generate_inputs(seed, manifest)        # within declared domains
    py  = python_fixed_point_reference(inputs)      # layer 2
    rs  = rust_fixed_point_reference(inputs)        # layer 3
    assert py == rs        # every integer, tolerance 0  (INV-TEST-02)
```

- Input generators sample within each tensor's declared `[lo, hi]` and
  deliberately oversample boundaries (`lo`, `hi`, `0`, `±B`) and requantize ties.
- The CI seed set is small and fixed for per-PR runs (fast, deterministic);
  nightly runs expand the seed set and increase input dimensions toward the V0
  reference configuration (latent_dim = 192, depth = 6, horizon = 5).
- A divergence prints the first differing `(tensor_id, index)`, both integer
  values, and the seed, so it reproduces immediately.

### Property tests

Property tests assert invariants over generated inputs using `proptest` (Rust,
layers 3/7) and `hypothesis` (Python, layers 2/7). The three mandatory property
families:

| Property | Statement (must hold for all generated inputs in-domain) | Layer |
|----------|----------------------------------------------------------|-------|
| **Range-safety** | Every integer the reference produces stays within its declared `[lo, hi]`; if a generated input would force any value out of range, the reference signals overflow (`OverflowPolicy::Reject`) and does NOT wrap. No silent `mod p`. | 2, 3, 7 |
| **Requantize rounding** | For `n = q·2^shift + rem` with `0 ≤ rem < 2^shift`, `requantize(n)` equals the manifest-declared rounding (`NearestTiesToEven` default; `TruncateTowardZero` if declared). Ties round to even under NTTE; sign handling is exact under truncation. Round-trip and monotonicity hold where defined. | 2, 3, 7 |
| **Argmin / tie-break** | `argmin` returns the smallest index attaining the minimum cost. Encoded via `cost_s − selected_cost ≥ 0` for all `s`, and `cost_s − selected_cost − 1 ≥ 0` for all `s < selected_index`. Property generators include forced ties, all-equal costs, and a unique strict minimum. | 2, 3, 7 |

### Differential-test invariants

| Invariant | Statement |
|-----------|-----------|
| **INV-TEST-12** | For every committed seed, Python fixed-point and Rust fixed-point outputs are bit-identical (strengthens INV-TEST-02 to randomized inputs). |
| **INV-TEST-13** | The three property families (range-safety, requantize rounding, argmin/tie-break) have passing `proptest`/`hypothesis` suites; a shrunk counterexample is logged and committed as a regression golden vector when found. |
| **INV-TEST-14** | Counterexamples are deterministic: the failing seed plus generator version reproduce the failure exactly (INV-TEST-03). |

---

<a id="mutation-tests"></a>

## Constraint mutation tests
Mutation testing measures whether the *constraint suite* is strong enough to be
sound. It is the formal answer to "do my negative tests actually cover my
constraints?" The mutation runner programmatically perturbs one constraint at a
time and confirms the test suite notices.

### Mechanism

```text
for each constraint C in every AIR component:
    apply ONE mutation to C            # see mutation operator table
    rebuild constraints, run layers 4 + 6 (witness-gen + reject suite)
    if NO test fails:  C is a SURVIVING mutant   (suite gap — bad)
    else:              C is KILLED                (suite covers it — good)

mutation_score = killed / (killed + surviving)
```

Mutation operators (one applied per run):

| Operator | Example perturbation |
|----------|----------------------|
| Drop term | remove an addend from a constraint polynomial |
| Off-by-one bound | change a range bound `hi` to `hi+1` |
| Relax comparison | change `cost_s − selected_cost − 1 ≥ 0` to `cost_s − selected_cost ≥ 0` (breaks tie-break) |
| Swap column | wire a constraint to the wrong trace column |
| Drop constraint | delete an entire constraint row |
| Flip selector | invert `is_first` / `is_last` / mask selector |
| Weaken lookup | drop a lookup relation or its multiplicity check |

A surviving mutant means: that mutated (wrong) constraint set still passes every
test — i.e., some real defect of that shape would ship undetected. Each survivor
is triaged: either a missing reject test is added (preferred) or the survivor is
recorded with justification in the mutation baseline file.

### Target and gate

> **INV-TEST-05 / mutation target.** The constraint-mutation score MUST be
> **≥ 0.90** (at least 90% of single-constraint mutations killed) across the
> `pwm-air` and `pwm-circuits` constraint sets, with **zero surviving mutants in
> the soundness-critical components** `requant`, `range_check`, `argmin`,
> `tensor_memory`, and `activation_lookup`. The full per-component score is
> reported as a CI artifact; a drop below target, or any new survivor in a
> soundness-critical component, blocks merge.

The 0.90 overall target is chosen because some mutants are provably equivalent
(e.g., reordering commutative addends) and cannot be killed; the soundness-
critical components are held to 100% because a survivor there is a direct
unsoundness. Mutation runs are expensive, so the full sweep runs nightly and a
fast scoped sweep (only constraints touched by the PR diff) runs per-PR.

> **OPEN QUESTION (owner: area:testing maintainer; resolution: v0.2 milestone).**
> Equivalent-mutant detection is initially manual (survivors triaged by hand into
> an "equivalent" baseline). Automating equivalent-mutant pruning is deferred to
> the v0.2 milestone; until then the baseline file is reviewed on every change.

---

<a id="ci-gates"></a>

## CI gates
The merge gate is the enforcement surface for every invariant above. A PR merges
to the default branch only if every gate below passes. Gates run in `area:ci`
workflows.

### The merge gate set

| Gate | Command (canonical) | Blocks merge on | Enforces |
|------|---------------------|-----------------|----------|
| Format | `cargo fmt --all -- --check` | any unformatted Rust | style consistency |
| Lint | `cargo clippy --all-targets --all-features -- -D warnings` | any clippy warning | warning-free build |
| Unit + integration tests | `cargo test --workspace --all-features` + `pytest pwm-export/` | any failing test (layers 2–6, 9) | INV-TEST-01, 04, 09–11 |
| Golden parity | `cargo test -p pwm-export --features golden` (layer 3 vs committed golden) | any non-zero diff vs golden | INV-TEST-02, 06; bit-for-bit chain |
| Golden immutability | golden-diff check vs committed set for shipped `relation_id`s | changed golden integers without a new `relation_id` | INV-TEST-08 |
| `no_std` verifier build | `cargo build -p pwm-verifier --no-default-features` (+ `ensure-verifier-no_std` crate, mirrors Stwo's pattern) | verifier pulling in `std` | verifier stays `no_std`-capable |
| Doc build | `cargo doc --workspace --no-deps` + spec-link checker | broken rustdoc or dead cross-reference anchor | spec/RFC cross-refs resolve |
| Perf-budget regression | perf harness vs committed budget (see [`docs/spec/08-performance-budget.md#perf-gates`](08-performance-budget.md#perf-gates)) | trace-size / proving-time regression beyond tolerance | performance budget |
| License / SPDX | SPDX header check + `cargo deny check licenses` | missing Apache-2.0 header or disallowed license | RFC-0015 audit boundary; Apache-2.0 discipline |
| Mutation (scoped) | scoped mutation sweep over PR-diff constraints | survivor in a soundness-critical component | INV-TEST-05 |

Nightly (non-blocking-for-merge but gating for release): full mutation sweep,
expanded differential seed set, `cargo fuzz` corpora for the verifier and
deserializers (layer 10), and the perf harness at full V0 dimensions.

### The governing CI rule

> **INV-TEST-15 — Component completeness gate.** A PR that adds or modifies a
> component, AIR constraint, primitive, or `VerifyError` path MUST add BOTH an
> accepting test AND at least one rejecting test for it in the same PR (INV-TEST-01).
> The unit+integration and mutation gates together enforce this: a new component
> with no reject test either fails INV-TEST-10 (zero reject tests) or surfaces as
> a surviving mutant. No component ships with accepting tests alone.

### Pre-release gates (layer 10)

Release (a tagged V0/V1) additionally requires the security-review gate, which is
manual and tracked against
[`docs/spec/06-security.md#threat-model`](06-security.md#threat-model):

| Pre-release check | Owner role | Evidence required |
|-------------------|------------|-------------------|
| Soundness review of all new/changed constraints | area:security | sign-off referencing the constraint and its reject tests |
| Verifier + deserializer fuzzing | area:security | `cargo fuzz` corpus run with no new crashes/timeouts |
| Dependency / vendoring audit | area:core | RFC-0015 inventory current; `third_party/stwo/REVISION` and `third_party/stwo-circuits/REVISION` recorded and matched |
| Mutation full-sweep at target | area:testing | mutation report ≥ target, zero soundness-critical survivors |
| Golden-vector provenance | area:export | `pwm-export/golden/REVISION` matches the checkpoint and tool versions in the manifest |

### CI invariants

| Invariant | Statement |
|-----------|-----------|
| **INV-TEST-15** | Component completeness gate (stated above). |
| **INV-TEST-16** | The default branch is always green: every listed merge gate passes on every merge commit; a red merge gate is a release blocker, not a known-flaky exception. |
| **INV-TEST-17** | CI is hermetic and deterministic: no network access during gated tests, pinned toolchain (MSRV per [`docs/spec/09-release-and-versioning.md#msrv`](09-release-and-versioning.md#msrv)), pinned vendored revisions (RFC-0015); a gate that depends on wall-clock, network, or unpinned input is itself a bug. |

---

## Cross-reference index

| Topic | Location |
|-------|----------|
| 10-layer stack, dual-test rule (decision record) | [`docs/rfcs/RFC-0013-testing-fuzzing-and-audit-strategy.md`](../rfcs/RFC-0013-testing-fuzzing-and-audit-strategy.md) |
| `VerifyError` taxonomy targeted by reject tests | [`docs/spec/04-error-model.md#verifier-rejections`](04-error-model.md#verifier-rejections) |
| Soundness / binding / range-safety requirements tested here | [`docs/spec/06-security.md#soundness-requirements`](06-security.md#soundness-requirements) |
| `BoundedInt`, `Tensor`, `PublicInput`, `Witness`, `ArgminWitness` schemas | [`docs/spec/03-data-model.md#bounded-integers`](03-data-model.md#bounded-integers) |
| Reproducibility / determinism contract | [`docs/rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md`](../rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md) |
| Export pipeline + manifest (golden-vector generation) | [`docs/rfcs/RFC-0001-model-manifest-and-export-pipeline.md`](../rfcs/RFC-0001-model-manifest-and-export-pipeline.md) |
| Fixed-point arithmetic (rounding, range, requant tested here) | [`docs/rfcs/RFC-0002-fixed-point-arithmetic-over-m31.md`](../rfcs/RFC-0002-fixed-point-arithmetic-over-m31.md) |
| Range/lookup infrastructure (LogUp multiplicity tests) | [`docs/rfcs/RFC-0003-range-check-and-lookup-infrastructure.md`](../rfcs/RFC-0003-range-check-and-lookup-infrastructure.md) |
| Tensor memory wiring (scale_id / read-write reject tests) | [`docs/rfcs/RFC-0004-tensor-memory-and-wiring-air.md`](../rfcs/RFC-0004-tensor-memory-and-wiring-air.md) |
| Linear/matmul/requant components | [`docs/rfcs/RFC-0005-linear-matmul-and-requantization-components.md`](../rfcs/RFC-0005-linear-matmul-and-requantization-components.md) |
| Nonlinear primitives (GELU/SiLU/Softmax/LayerNorm golden vectors) | [`docs/rfcs/RFC-0006-nonlinear-primitive-components.md`](../rfcs/RFC-0006-nonlinear-primitive-components.md) |
| Predictor AIR | [`docs/rfcs/RFC-0007-leworldmodel-predictor-air.md`](../rfcs/RFC-0007-leworldmodel-predictor-air.md) |
| Rollout AIR | [`docs/rfcs/RFC-0008-rollout-air.md`](../rfcs/RFC-0008-rollout-air.md) |
| Fixed-candidate planner (cost/argmin/tie-break tests) | [`docs/rfcs/RFC-0009-fixed-candidate-planner-proof.md`](../rfcs/RFC-0009-fixed-candidate-planner-proof.md) |
| Transcript / public-input binding (N21/N22) | [`docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md`](../rfcs/RFC-0014-canonical-serialization-and-transcript.md) |
| Vendoring / SPDX / audit boundary (license + audit gates) | [`docs/rfcs/RFC-0015-third-party-vendoring-and-pinning.md`](../rfcs/RFC-0015-third-party-vendoring-and-pinning.md) |
| Performance budget (perf-gate) | [`docs/spec/08-performance-budget.md#perf-gates`](08-performance-budget.md#perf-gates) |
| Founding analysis (RFC-013 §11.13, risk register §13) | [`docs/feasibility-study.md`](../feasibility-study.md) |
