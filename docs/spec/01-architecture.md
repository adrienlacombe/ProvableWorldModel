# Architecture

Status: Normative. Role: defines the workspace crate layout, the module dependency DAG, the end-to-end data flow, the direct-AIR-vs-circuit-lowering strategy, the AIR component model, and the three-trace model (preprocessed / main / interaction). This document is binding on all crates. Type signatures shown here are reproduced verbatim from the canonical definitions in [docs/spec/03-data-model.md](03-data-model.md#tensor-types); this document explains *where* they live and *how they flow*, not their field-by-field semantics.

ProvableWorldModel proves deterministic, quantized inference of a JEPA-style world model (LeWorldModel) by arithmetizing it as a custom Circle-STARK AIR over the Mersenne-31 field (M31), built on a vendored Stwo prover. The proof verifies an exact integer arithmetic relation. It does not prove floating-point equivalence, physical truth, or zero-knowledge unless separately audited. The headline first deliverable is V0 = statement P2 (fixed-candidate planning). See [docs/spec/00-overview.md#v0-statement](00-overview.md#v0-statement) for the precise V0 claim and [docs/spec/00-overview.md#scope-and-statement-tiers](00-overview.md#scope-and-statement-tiers) for the P0–P4 tiers.

---

## crate-layout

The system is a single Cargo workspace under `crates/`, plus vendored dependencies under `third_party/`, plus a Python package colocated with `pwm-export` for the checkpoint-to-manifest pipeline. Six first-party crates partition the system along a strict layering boundary: pure data and arithmetic at the bottom, proving substrate in the middle, prover and verifier entry points at the top.

```text
ProvableWorldModel/
  Cargo.toml                      # workspace manifest; pins MSRV, lints, profiles
  crates/
    pwm-core/                     # field types, tensors, manifest types, transcript
    pwm-export/                   # Python + Rust export/quantize pipeline
    pwm-air/                      # NN-specific AIR components
    pwm-circuits/                 # low-level gates + stwo-circuits adapters
    pwm-prover/                   # prover CLI, trace builder, prove_* entry points
    pwm-verifier/                 # verifier library, public-input handling, recursion
  third_party/
    stwo/                         # vendored StarkWare Stwo, pinned (RFC-0015)
    stwo-circuits/                # vendored StarkWare stwo-circuits, pinned (RFC-0015)
  docs/                           # this corpus
```

### Crate responsibilities and key modules

Each crate has exactly one reason to change. The table is the contract; the prose under it elaborates the non-obvious responsibilities.

| Crate | Area label | Single responsibility | Key modules | Proving deps? |
| --- | --- | --- | --- | --- |
| `pwm-core` | `area:core` | Field/fixed-point types, tensors, manifest types, canonical serialization, transcript, `relation_id`. The shared vocabulary every other crate speaks. | `field` (M31/QM31 re-exports + `BoundedInt`), `fixed_point` (add/sub/mul/accumulate/requantize/clamp/compare reference semantics), `tensor` (`Tensor`, `QuantizedWeights`), `manifest` (schema types + canonical-JSON hashing), `transcript` (Fiat-Shamir channel ordering), `relation_id`, `serialize` (canonical bytes + public-input digest) | No |
| `pwm-export` | `area:export` | Consume a PyTorch checkpoint + config, quantize to an integer graph, write the manifest + weight tensors + golden vectors. Python for graph extraction; Rust for the canonical fixed-point reference and parity. | Python: `torch_export/` (graph + weight extraction), `onnx_import/` (alternate ingest), `quantize` (scale selection, int8/int16 assignment), `manifest_writer`. Rust: `reference` (fixed-point reference inference, must match the AIR bit-for-bit), `parity_tests` (Python↔Rust↔golden) | Reference inference yes; proving no |
| `pwm-air` | `area:air` | Define NN-specific AIR components: their preprocessed/main/interaction columns and constraint sets. The arithmetization of the model. | `components/{range_check, tensor_memory, weight_table, linear, matmul, requant, activation_lookup, layernorm, attention, mlp, action_encoder, predictor, rollout, cost, argmin, cem}`, `proof` (component composition), `verifier` (per-component verification glue) | Yes (constraint framework) |
| `pwm-circuits` | `area:circuits` | Low-level scalar gates and adapters to the vendored `stwo-circuits` gate set, plus the witness builder used for hashing, public-input binding, glue, and recursion plumbing. Not the dense-matmul hot path. | `low_level_gates` (thin wrappers over the 10-gate set), `adapters_stwo_circuits` (vendored API surface), `witness_builder` | Yes (circuit IR) |
| `pwm-prover` | `area:prover` | Prover CLI and orchestration: load manifest, run reference inference, build traces, drive Stwo commit/challenge/proof, emit the artifact bundle. | `cli`, `trace_builder` (assemble preprocessed/main/interaction traces from reference activations), `prove_rollout` (P0/P1), `prove_planning` (P2; calls into `area:planner` orchestration for cost/argmin/CEM) | Yes (full prover) |
| `pwm-verifier` | `area:verifier` | Verify a `ProofArtifact`: parse and digest public input, check `relation_id` support, verify the Stwo proof, enforce application-level checks. Library-first; no PyTorch, no export deps. | `lib` (`verify`), `public_input` (digest recomputation, shape/range checks), `recursive` (aggregated verification, deferred per RFC-0012) | Verify only |

`pwm-core` is the dependency root. It owns the canonical types so that the prover and verifier agree on bytes without sharing higher-level code. Critically, `pwm-core` carries **no proving dependencies** (no Stwo, no constraint framework): it must be linkable into the `no_std` verifier path. The `fixed_point` module is the *normative reference*: its add/sub/mul/accumulate/requantize/clamp/compare functions define the exact integer semantics that both the Python reference (via parity tests) and every `pwm-air` component must reproduce bit-for-bit. RFC-0002 (`docs/rfcs/RFC-0002-fixed-point-arithmetic-over-m31.md`) locks these semantics.

`pwm-export` is the only crate that touches PyTorch or ONNX, and it does so only in its Python subpackage. Its Rust side never imports a Python runtime; it holds the canonical Rust fixed-point reference and the parity harness. This split is deliberate: export and quantization are an offline, trusted preprocessing step that produces a *committed* manifest; once committed, neither the prover nor the verifier re-runs PyTorch. RFC-0001 (`docs/rfcs/RFC-0001-model-manifest-and-export-pipeline.md`) locks the schema and the byte-identical re-export requirement.

`pwm-air` and `pwm-circuits` are siblings, not a stack: `pwm-air` expresses dense operators directly in Stwo's constraint framework (the efficient path), while `pwm-circuits` provides scalar gates for the glue that does not arithmetize well as a dense tensor op. See [#air-strategy](#air-strategy) for the split rule.

### Vendored third-party crates

Stwo and stwo-circuits are vendored under `third_party/` at pinned revisions and modified as needed. The project does not depend on the upstream canonical crates directly; this protects the audit boundary and the soundness-critical Fiat-Shamir/PCS internals from upstream drift. RFC-0015 (`docs/rfcs/RFC-0015-third-party-vendoring-and-pinning.md`) is the authority on vendoring, the modification/audit policy, and the dependency inventory.

| Vendored tree | Upstream | License | Pinned shape (verified against upstream at 2026-06-03) |
| --- | --- | --- | --- |
| `third_party/stwo/` | github.com/starkware-libs/stwo | Apache-2.0 | Workspace v2.2.0. Crates: `stwo`, `stwo-air-utils`, `stwo-air-utils-derive`, `stwo-constraint-framework`, plus `examples` and `std-shims`. The Circle-STARK prover/verifier, M31 field module, FRI + PCS, Fiat-Shamir channels, LogUp. |
| `third_party/stwo-circuits/` | github.com/starkware-libs/stwo-circuits | Apache-2.0 | v0.1.0, edition 2024. Ten crates: `cairo_verifier`, `circuits`, `circuit_verifier`, `circuit_cairo_serialize`, `circuit_common`, `circuit_multiverifier`, `circuit_serialize`, `circuit_prover`, `stark_verifier`, `stark_verifier_examples`. Provides the low-level scalar gate set. |

The exact pinned revisions are recorded in `third_party/stwo/REVISION` and `third_party/stwo-circuits/REVISION` at vendoring time.

> RESOLVED (2026-06-03, RFC-0015 toolchain decision): the pinned revisions are `stwo` `v2.2.0` (`289c20de80b7c7f508de9c46151fb81dae404154`) and `stwo-circuits` `v0.1.0` (`b0db13e46d977e0bb10a28321a7f09bf5ea516aa`). Both require nightly Rust, so the project pins `nightly-2025-07-14` in `rust-toolchain.toml`. The exact hashes are written to each `REVISION` file when the tree is vendored (issues #24, #25); until then both files contain the placeholder string `pending-rfc-0015`.

Apache-2.0 is the project license, chosen for compatibility with the vendored Apache-2.0 Stwo and for the patent grant that matters in a cryptographic system. See [docs/spec/09-release-and-versioning.md#license](09-release-and-versioning.md#license).

---

## module-boundaries

The crates form a directed acyclic graph. The layering rule is enforced at the workspace level (a CI gate fails the build on a back-edge): a crate may depend only on crates strictly below it in the diagram.

```text
                         depends-on (edge points to the dependency)
                         =========================================

   pwm-prover  ─────────────┐                       pwm-verifier
   (area:prover)            │                       (area:verifier)
      │   │   │             │                          │      │
      │   │   │             ▼                          │      │
      │   │   └────────►  pwm-air ◄────────────────────┘      │
      │   │              (area:air)                           │
      │   │                 │  │                               │
      │   │                 │  └──────────►  pwm-circuits      │
      │   │                 │               (area:circuits)    │
      │   │                 │                    │             │
      │   └─────────────────┼────────────────────┘             │
      │                     ▼                                  │
      │                third_party/stwo ◄──────────────────────┤
      │              third_party/stwo-circuits                 │
      ▼                     │                                  │
   pwm-export               ▼                                  ▼
   (area:export) ───────► pwm-core  ◄──────────────────────────┘
                         (area:core)
                              ▲
                              │  (no proving deps; no_std-clean)
```

Read every arrow as "depends on". Stated as prose, top to bottom:

- `pwm-core` depends on nothing first-party. It depends on `third_party/stwo` only for the re-exported `M31`/`QM31`/`CM31` field types so the whole corpus shares one field definition; it pulls in *no* prover, FRI, PCS, or channel code, and it must compile on the `no_std` verifier path. INV-ARCH-01 below makes this binding.
- `pwm-export` depends on `pwm-core` (for `Tensor`, `BoundedInt`, manifest types, the fixed-point reference, canonical serialization). Nothing depends on `pwm-export`.
- `pwm-circuits` depends on `pwm-core` and on `third_party/stwo` + `third_party/stwo-circuits`.
- `pwm-air` depends on `pwm-core`, `pwm-circuits`, and `third_party/stwo` (the constraint framework, air utils). `pwm-air` may call into `pwm-circuits` for hashing and public-input-binding gadgets composed alongside its dense components.
- `pwm-prover` depends on `pwm-core`, `pwm-air`, `pwm-circuits`, `pwm-export` (for the Rust reference-inference path), and `third_party/stwo`.
- `pwm-verifier` depends on `pwm-core`, `pwm-air` (to learn each component's verification routine), `pwm-circuits`, and `third_party/stwo` (the verifier path only). It depends on **neither `pwm-export` nor any PyTorch/Python runtime**. INV-ARCH-02 makes this binding.

### Boundary invariants

| Invariant | Statement | Enforcement | Failure mode and system response |
| --- | --- | --- | --- |
| INV-ARCH-01 | `pwm-core` has no dependency, direct or transitive, on a prover, FRI, PCS, channel, or constraint-framework crate, and compiles under `no_std`. | CI gate: `cargo tree -p pwm-core` is asserted against an allowlist; an `ensure-core-no_std` test crate builds `pwm-core` with `--no-default-features`. | If a proving dep leaks into `pwm-core`: CI fails the build with `INV-ARCH-01 violated: forbidden dependency <crate> in pwm-core`. Merge is blocked. |
| INV-ARCH-02 | `pwm-verifier` has no dependency, direct or transitive, on `pwm-export`, PyTorch, ONNX, or any Python runtime. The verifier verifies arithmetic only; it never re-runs the model. | CI gate over `cargo tree -p pwm-verifier`; an `ensure-verifier-no_std` test crate (mirroring Stwo's own) builds the verifier core under `no_std`. | If `pwm-export` or a Python binding leaks in: CI fails with `INV-ARCH-02 violated: verifier depends on export/runtime`. Merge is blocked. |
| INV-ARCH-03 | The crate dependency graph is acyclic and respects the layering order `core < {export, circuits} < air < {prover, verifier}`. | CI gate: a workspace dependency-DAG check rejects any back-edge or cycle. | Back-edge introduced: CI fails with `INV-ARCH-03 violated: illegal edge <A> -> <B>`. Merge is blocked. |
| INV-ARCH-04 | Every artifact a proof binds to (architecture, weights, biases, scales, rounding, lookup/activation tables, approximations, shapes, planner config, relation version, serialization version) is reachable from `model_commitment` or `quantization_commitment`. No semantically load-bearing value is unbound. | The manifest writer in `pwm-export` computes commitments over the full canonical serialization; `pwm-verifier` recomputes the public-input digest. See [docs/spec/06-security.md#binding-requirements](06-security.md#binding-requirements). | An unbound value is detected during binding review or differential testing: the prover could silently mutate semantics, so the manifest schema is amended and a new `relation_id` is minted. The change is tracked in RFC-0001 and RFC-0014. |

Why these boundaries matter for soundness: the verifier is the trust anchor. If it could depend on the export pipeline or a floating-point runtime, an auditor would have to trust the entire ML stack to trust a verification. By forbidding those edges (INV-ARCH-02), the audit surface for *accepting a proof* shrinks to `pwm-core` + `pwm-air` verification routines + `third_party/stwo` verifier + `pwm-circuits`. The transcript, serialization, and `relation_id` logic that both sides must agree on lives entirely in `pwm-core` (INV-ARCH-01), so there is exactly one implementation of the Fiat-Shamir channel ordering and the public-input digest, shared by prover and verifier. RFC-0014 (`docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md`) locks that shared logic.

---

## data-flow

The system has two trust phases. Phase one (offline, trusted) turns a checkpoint into a committed manifest and golden vectors. Phase two (the proving protocol) turns inputs + manifest into a proof, which a third party verifies with no trust in phase one beyond the published commitments.

```text
  ┌───────────────────────── PHASE 1: EXPORT (offline, trusted) ──────────────────────────┐
  │                                                                                         │
  │  PyTorch checkpoint ──► canonical graph ──► quantized integer graph                     │
  │  + le-wm config           (pwm-export          (int8 weights;                           │
  │  (MIT; consumed,           Python)              int8/int16 activations;                 │
  │   not vendored)                                 int32 accumulators/biases;              │
  │                                                 pow-2 scales)                           │
  │                                       │                                                 │
  │                                       ▼                                                 │
  │                              ┌──────────────────┐                                       │
  │                              │ manifest writer  │                                       │
  │                              └──────────────────┘                                       │
  │                                  │      │      │                                        │
  │                                  ▼      ▼      ▼                                         │
  │                         manifest.yaml  weight  golden vectors                           │
  │                         (model_commit  tensors (per-op + end-to-end                     │
  │                          + quant_commit)        reference outputs)                      │
  │                                  │                     │                                │
  │             parity gate: Python fixed-point ref ≡ Rust fixed-point ref ≡ golden         │
  │                                  │                                                       │
  └──────────────────────────────────┼──────────────────────────────────────────────────────┘
                                      ▼
  ┌──────────────────── PHASE 2a: PROVE (pwm-prover) ─────────────────────────────────────┐
  │                                                                                         │
  │  manifest + weights + inputs (latent history, goal latent, candidate actions)           │
  │        │                                                                                │
  │        ▼  (1) verify manifest hash == model_commitment; load + verify weight root       │
  │        ▼  (2) canonicalize public inputs; derive public-input digest                    │
  │        ▼  (3) run Rust fixed-point reference inference  ──► all intermediate activations │
  │        ▼  (4) trace_builder: preprocessed trace + main (witness) trace                  │
  │        ▼  (5) commit traces to the Stwo channel (Blake2s / Keccak256 / Poseidon252)     │
  │        ▼  (6) derive Fiat-Shamir challenges in canonical order                          │
  │        ▼  (7) build interaction (LogUp) traces from challenges                          │
  │        ▼  (8) Stwo generates the proof (FRI + PCS)                                       │
  │        ▼  (9) emit ProofArtifact { artifact_version, public_input, proof,               │
  │                                    claimed_outputs }                                    │
  └──────────────────────────────────┼──────────────────────────────────────────────────────┘
                                      ▼
  ┌──────────────────── PHASE 2b: VERIFY (pwm-verifier; no PyTorch) ──────────────────────┐
  │  (1) parse public input;            (4) verify the Stwo proof (channel order must match) │
  │  (2) check relation_id supported;   (5) decode claimed outputs;                          │
  │  (3) recompute public-input digest  (6) enforce app checks: selected_index in range,     │
  │      + preprocessed-trace commits;      shapes match, commitments match serialization.   │
  │                              ──► Ok(())  or  VerifyError                                  │
  └─────────────────────────────────────────────────────────────────────────────────────────┘
```

In prose, the pipeline is: **checkpoint → export/quantize → (manifest + weights + golden vectors) → prover (reference inference → trace build → Stwo commit → Fiat-Shamir challenges → interaction trace → proof) → verifier.**

### Phase 1 — export (area:export)

`pwm-export` (Python) extracts the operator graph and weights from a PyTorch checkpoint, quantizes per the V0 policy (weights int8; activations int8 or int16 declared per-tensor; accumulators bounded int32 in M31 where safe, else limb-decomposed; biases int32; scales powers-of-two where possible), and the manifest writer emits three coupled outputs:

1. `manifest.yaml` — the canonical model description whose canonical-JSON hash is `serialization.canonical_json_hash`, and whose subtrees produce `model_commitment` and `quantization_commitment` (schema in [docs/spec/03-data-model.md#model-manifest](03-data-model.md#model-manifest)).
2. Weight tensors — committed under `weights.root` (matching `QuantizedWeights.commitment`).
3. Golden vectors — per-op and end-to-end reference outputs of the fixed-point inference, used as the differential oracle.

The parity gate is mandatory before any manifest is published: the Python fixed-point reference, the Rust fixed-point reference in `pwm-export`/`pwm-core`, and the golden vectors must agree bit-for-bit. Re-exporting the same checkpoint must be byte-identical (RFC-0001). le-wm is MIT-licensed and is *consumed*, not vendored: the export pipeline reads a checkpoint and config and attributes appropriately (see [docs/spec/09-release-and-versioning.md#license](09-release-and-versioning.md#license)). Training, SIGReg, dataloading, and the Hydra config framework are out of proving scope and never enter this pipeline.

### Phase 2a — prove (area:prover, area:planner)

`pwm-prover` is the orchestrator. The numbered steps in the diagram are the canonical prover flow. The two soundness-critical orderings are: (a) the manifest hash and weight root are checked *before* any trace is built, so a tampered manifest never reaches proving; and (b) the trace commit → challenge derivation → interaction-trace order matches the Fiat-Shamir channel order the verifier will replay (RFC-0014). The trace builder consumes the Rust reference inference's intermediate activations to fill the witness (main) trace; it never calls PyTorch. For P2, the planner orchestration (`area:planner`) drives the cost and argmin components over all candidates (RFC-0009 forbids partial proofs).

### Phase 2b — verify (area:verifier)

`pwm-verifier::verify` parses the public input, rejects unsupported or mismatched `relation_id`s, recomputes the public-input digest and the preprocessed-trace commitments from the (committed) manifest/config, verifies the Stwo proof, decodes claimed outputs, and enforces application-level checks. It runs no model. The full rejection taxonomy lives in [docs/spec/04-error-model.md#verifier-rejections](04-error-model.md#verifier-rejections); the canonical signature is reproduced under [#component-model](#component-model).

### Data-flow invariants

| Invariant | Statement | Failure mode and system response |
| --- | --- | --- |
| INV-ARCH-05 | The prover verifies `hash(manifest) == model_commitment` and the weight root `== weights.root` before building any trace. | Mismatch: the prover aborts with a `ManifestError`/`WeightError` (see [docs/spec/04-error-model.md#error-taxonomy](04-error-model.md#error-taxonomy)); no proof is produced. |
| INV-ARCH-06 | The Fiat-Shamir channel construction (channel backend, absorb/squeeze order, public-input digest) is byte-for-byte identical on prover and verifier, implemented once in `pwm-core::transcript`. | Divergence: the proof fails verification with `VerifyError::TranscriptMismatch`. A divergence in committed code is a soundness bug, caught by the transcript differential test in [docs/spec/07-testing-strategy.md#differential-tests](07-testing-strategy.md#differential-tests). |
| INV-ARCH-07 | A `ProofArtifact` is valid only for its declared `relation_id`; the verifier rejects a proof submitted under any other `relation_id`. | Wrong `relation_id` (e.g. a P0 proof submitted as P1): `VerifyError::RelationMismatch`. |
| INV-ARCH-08 | Reference inference is deterministic: identical (manifest, weights, inputs) yield identical activations, traces, and committed proof inputs/outputs. | Nondeterminism detected by the reproducibility gate (RFC-0016): the build fails; bit-for-bit reproducibility is a release requirement. See [docs/spec/09-release-and-versioning.md#changelog](09-release-and-versioning.md#changelog). |

---

## air-strategy

There are two ways to arithmetize the model: write custom AIR components directly against Stwo's constraint framework, or lower everything to a generic scalar circuit IR over the vendored stwo-circuits gate set. The project uses **both, by role**, because each is efficient for a different class of work.

### Strategy A — direct custom AIR (the dense hot path)

Dense neural-network inference is dominated by regular, structured tensor operations: linear layers, batched matmuls, attention, the rollout recurrence, MSE cost, and argmin. For these, a high-level operator AIR with a tensorized trace layout is both more efficient (one component amortizes constraints across a whole tensor op, exploits static weights, and batches over candidate / rollout-step / sequence / head / channel axes) and easier to audit (the constraint set reads like the operator's definition) than expressing each scalar multiply as a generic gate. This is the production route for everything compute-heavy. Stwo's constraint framework supports exactly this: multiple components, fixed preprocessed columns, witness trace columns, a defined Fiat-Shamir channel order, and LogUp lookup arguments.

### Strategy B — low-level circuit lowering (glue, hashing, recursion)

The vendored stwo-circuits crate exposes a generic scalar gate set well suited to small, irregular logic that does not arithmetize as a dense tensor op: commitment hashing, public-input binding, transcript glue, small arithmetic gadgets, and the plumbing for a future recursive verifier. Compiling dense matmul to these scalar gates would be far too expensive, so they are confined to glue. The first-class gate set is the ten fields of stwo-circuits' `struct Circuit` (verified against upstream at 2026-06-03):

```text
Add  Sub  Mul  PointwiseMul  Eq  TripleXor  M31ToU32  BlakeGGate  Permutation  Output
```

There is **no dedicated range or bit-extraction gate.** `extract_bits()` is a *helper* built from `Sub`/`Mul`/`assert_bits` constraints, and range checking is enforced by those constraints (and, in the AIR layer, by LogUp range relations), not by a special gate. `BlakeGGate` is a Blake2s G-function gate, not a full-hash gate; a full Blake2s is composed from `BlakeGGate` plus the surrounding constraints. Treat stwo-circuits as an implementation substrate to audit and pin (RFC-0015), not a turnkey NN proving framework.

### Recommended split

| Concern | Route | Rationale |
| --- | --- | --- |
| Dense linear / batched linear | Direct AIR (`pwm-air::linear`) | Dominates proving cost; tensorized layout + static weights. |
| Batched matmul | Direct AIR (`pwm-air::matmul`) | Same. |
| Attention (QKᵀ, masked scores, softmax-apply, prob·V) | Direct AIR (`pwm-air::attention`) | Structured dot-products; softmax via lookup (RFC-0006). |
| Rollout recurrence (autoregressive windowing) | Direct AIR (`pwm-air::rollout`) | Recurrence over latent/action windows; tensor-memory wiring. |
| MSE cost | Direct AIR (`pwm-air::cost`) | Σ of squared diffs; range-checked accumulator. |
| Argmin + tie-break | Direct AIR (`pwm-air::argmin`) | Difference range-checks across all candidates. |
| Range checking | LogUp relations in `pwm-air::range_check` (constraint-based) | No range gate exists; enforced by LogUp + `assert_bits`-style constraints. |
| Activation / normalization lookups | LogUp relations in `pwm-air::activation_lookup` | Dynamic table lookups; tables committed in the manifest (RFC-0006). |
| Commitment hashing (Blake2s) | Low-level circuits (`pwm-circuits`, via `BlakeGGate`) | Irregular bit logic; no dense structure to exploit. |
| Public-input binding / digest | Low-level circuits (`pwm-circuits`) | Small, fixed glue. |
| Small arithmetic gadgets, compatibility experiments | Low-level circuits (`pwm-circuits`) | Prototyping and odd corners. |
| Recursive-verifier plumbing | Low-level circuits (`pwm-circuits`) | Deferred; RFC-0012, Future. |

INV-ARCH-09: dense tensor hot paths (linear, matmul, attention, rollout, cost, argmin) are never lowered to generic scalar gates; they are implemented as direct AIR components. Enforcement: code review against this table, plus the performance-regression gate in [docs/spec/08-performance-budget.md#perf-gates](08-performance-budget.md#perf-gates) which would flag the blow-up a scalar-gate matmul causes. RFC-0005 (`docs/rfcs/RFC-0005-linear-matmul-and-requantization-components.md`) locks the dense-component layout; RFC-0003 (`docs/rfcs/RFC-0003-range-check-and-lookup-infrastructure.md`) locks the LogUp range/lookup relations and the fact that range-checking is constraint-based.

---

## component-model

The AIR is a composition of Stwo components, each owning a slice of the preprocessed/main/interaction trace and a constraint set. Components communicate only through the tensor-memory relation and shared LogUp tables, never by reaching into each other's columns. The full list, with the RFC that specifies each and the milestone that delivers it:

| Component | Crate module | Specified by | Milestone | Role |
| --- | --- | --- | --- | --- |
| RangeCheck | `pwm-air::range_check` | RFC-0003 | v0.1 | LogUp range relations (u8/i8/u16/bounded-limb) + multiplicity checks. Range-checking is constraint-based; there is no range gate. |
| TensorMemory | `pwm-air::tensor_memory` | RFC-0004 | v0.1 | `TensorCell` relation; read/write consistency via permutation/multiset argument; broadcast/reshape/transpose/concat/slice rules. |
| WeightTable | `pwm-air::weight_table` | RFC-0005 | v0.1 | Committed weight/bias lookup; supports public or private-committed weights. |
| Linear | `pwm-air::linear` | RFC-0005 | v0.1 | `y = Requantize(bias + Σ xᵢ·wᵢⱼ)` with accumulator/limb strategy and batching axes. |
| MatMul | `pwm-air::matmul` | RFC-0005 | v0.1 | Batched matmul; shared accumulator strategy with Linear. |
| Requantize | `pwm-air::requant` | RFC-0005 | v0.1 | Quotient/remainder constraints; `nearest_ties_to_even` (default) or manifest-declared `truncate_toward_zero`; clamp per tensor. |
| ActivationLookup | `pwm-air::activation_lookup` | RFC-0006 | v0.2 | LogUp lookups for GELU/SiLU and other committed tables (predictor FFN/MLP uses GELU; AdaLN modulation and the action Embedder use SiLU — stated per-module). |
| LayerNorm | `pwm-air::layernorm` | RFC-0006 | v0.2 | Mean / variance / inv-sqrt via bounded lookup or Newton; AdaLN-zero modulation. Approximations committed in the manifest. |
| Attention | `pwm-air::attention` | RFC-0006 | v0.2 | QKᵀ scaling, causal mask selectors, softmax approximation, prob·V. |
| MLP | `pwm-air::mlp` | RFC-0006 | v0.2 | FeedForward block: linear → activation → linear. |
| ActionEncoder | `pwm-air::action_encoder` | RFC-0007 | v0.2 | `ActionEncoder_Q` (the action `Embedder`; uses SiLU). |
| PredictorBlock | `pwm-air::predictor` | RFC-0007 | v0.2 | One `ConditionalBlock`: AdaLN-zero + attention + MLP with action conditioning; composes Attention + MLP + LayerNorm + ActionEncoder outputs. |
| Predictor (full) | `pwm-air::predictor` | RFC-0007 | v0.2 | `ARPredictor_Q` over depth-6 blocks + learned positional embeddings + `PredProj_Q`; the P0 relation `z_next = PredProj_Q(ARPredictor_Q(z_history, ActionEncoder_Q(actions)))`. |
| Rollout | `pwm-air::rollout` | RFC-0008 | v0.2 | Autoregressive windowing recurrence + trajectory commitment (the P1 relation). |
| Cost | `pwm-air::cost` | RFC-0009 | v1.0 | Goal-latent MSE: `Σⱼ (zⱼ − gⱼ)²` with range-checked accumulator. |
| Argmin | `pwm-air::argmin` | RFC-0009 | v1.0 | Selected-cost ≤ every candidate cost; tie-break via `cost_s − selected_cost − 1 ≥ 0` for `s < selected_index` (the P2 selection). |
| CEM (optional) | `pwm-air::cem` | RFC-0010 | Future | Seeded sampling, clipping, top-k, mean/var updates, iteration recurrence (the P3 relation). Out of v1.0 scope. |

Two RFC-0011 (Future) components — PatchEmbed/ViTEncoder/Projector for the pixel encoder (P4/V3) — are out of the V0 component model and are listed here only for completeness; they live in `pwm-air` under RFC-0011 when V3 begins.

The verifier learns, per component, how to verify its slice. The public entry point is the canonical signature (from contract §6.5, reproduced verbatim):

```rust
pub fn verify(artifact: &ProofArtifact) -> Result<(), VerifyError>;
```

where `ProofArtifact { artifact_version: u32, public_input: PublicInput, proof: Proof, claimed_outputs: Option<Vec<Tensor>> }`. The `StatementType` discriminant in `PublicInput` (`P0Step | P1Rollout | P2FixedCandidatePlanning | P3Cem | P4PixelToPlan`) selects which components must be present and verified. Field-level definitions of `PublicInput`, `Proof`, `ProofArtifact`, and `VerifyError` are in [docs/spec/02-public-api.md#rust-public-api](02-public-api.md#rust-public-api) and [docs/spec/03-data-model.md#public-input](03-data-model.md#public-input).

INV-ARCH-10: components interact only through the TensorMemory relation and shared LogUp tables; no component reads another component's witness columns directly. Enforcement: the trace builder assigns disjoint column ranges per component; cross-component wiring goes through `TensorCell`. Failure mode: a wiring bug that lets one component fabricate another's output is caught by the constraint-mutation tests in [docs/spec/07-testing-strategy.md#mutation-tests](07-testing-strategy.md#mutation-tests).

---

## trace-model

A Stwo proof commits to three traces. The split is fixed and shared by prover and verifier. Note on naming: in Stwo the field arithmetic lives in a **module** (`crates/stwo/src/core/fields/`) inside the single `stwo` crate — it is a field module, not a set of separate "field crates" (verified against upstream at 2026-06-03). The Fiat-Shamir channel backends available are **Blake2s, Keccak256, and Poseidon252**.

### Preprocessed trace (fixed columns)

Fixed data known to both prover and verifier before proving begins; agreed by commitment, never witness-dependent. The verifier recomputes or checks these commitments from the manifest/config (data-flow step 3 of verify).

```text
row selectors                      operation selectors
tensor-shape metadata              static masks
causal attention masks             positional embeddings (learned, fixed per manifest)
quantization scales                rounding constants
lookup-table columns               range-check table columns
```

Because positional embeddings and the activation/normalization tables are *committed in the manifest* (INV-ARCH-04), they are preprocessed/fixed data: the prover cannot vary them. The rounding constants encode the one active rounding mode per manifest (`nearest_ties_to_even` default, or declared `truncate_toward_zero`).

### Main trace (witness columns)

The per-proof witness produced by the trace builder from reference-inference activations:

```text
tensor values                      private weights (when visibility = private_committed)
intermediate activations           accumulators
products                           requantization quotients / remainders
normalization intermediates        attention scores / probabilities
costs                              argmin comparison witnesses (the diffs in ArgminWitness)
```

These columns mirror the canonical witness type (contract §6.3, reproduced verbatim):

```rust
pub struct Witness {
    pub model_weights: Option<QuantizedWeights>,  // None when public/committed-only
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

The tensor-memory cell carried in the main trace and its TensorMemory wiring is (contract §6.4, verbatim):

```rust
pub struct TensorCell {
    pub tensor_id: u32,
    pub index: [u32; 4],   // up to 4 dims; unused dims = 0
    pub value: M31,
    pub scale_id: u32,
    pub time: u32,         // write ordering for read/write consistency
}
```

### Interaction trace (LogUp relations)

Built *after* Fiat-Shamir challenges are derived (data-flow step 7), the interaction trace carries the LogUp/permutation arguments that bind the witness to the tables and enforce memory consistency:

```text
range checks                       activation / normalization lookups
weight-table lookups               tensor read/write consistency (permutation/multiset)
candidate-cost membership          top-k membership (only when CEM/P3 is proven)
```

Dynamic lookup/permutation arguments are used where values are not known before proving (e.g. which activation-table entries a given run touches); static range checks bind every integer-interpreted column to its declared range. Stwo's LogUp lives in `constraint-framework/src/logup.rs` in the vendored tree.

### Trace-model invariants

| Invariant | Statement | Failure mode and system response |
| --- | --- | --- |
| INV-ARCH-11 | Every column interpreted as an integer (latents, actions, weights, biases, products, accumulators, quotients, remainders, activation outputs, attention scores/probs, layernorm intermediates, cost diffs, argmin diffs) is wired to a range relation in the interaction trace. M31 wraps; integer ML inference must not. | An unranged integer column is a soundness hole: the prover could exploit field wraparound. Caught by the wiring/coverage check in RFC-0003 and the overflow negative tests in [docs/spec/07-testing-strategy.md#negative-tests](07-testing-strategy.md#negative-tests). Verifier response at runtime: an out-of-range witness makes the LogUp claimed sum inconsistent → `VerifyError`. |
| INV-ARCH-12 | The preprocessed trace is fully determined by the committed manifest/config; the prover cannot choose any preprocessed value. | A prover-chosen "fixed" column: the verifier's recomputed preprocessed-trace commitment differs → `VerifyError::PreprocessedMismatch`. |
| INV-ARCH-13 | The interaction trace is built only from challenges derived in the canonical Fiat-Shamir order *after* the main trace is committed. | Challenge derived out of order (e.g. interaction columns committed before the main trace): the verifier's channel replay diverges → `VerifyError::TranscriptMismatch` (cf. INV-ARCH-06). |

The QM31 secure field is used only for Fiat-Shamir challenges and soundness amplification; it never represents a quantized tensor value (contract §6.1). Quantized values are `BoundedInt`s embedded into base M31 via centered encoding and range-checked to `[lo, hi]` (RFC-0002). The signed encoding, per-tensor bounds, and the reason wraparound is unsound are specified in [docs/spec/06-security.md#soundness-requirements](06-security.md#soundness-requirements).

---

## subsystem-to-area-label map

Every subsystem maps to exactly one GitHub area label, so issues, RFCs, and CI ownership line up with crate boundaries.

| Subsystem | Area label | Owning crate / location |
| --- | --- | --- |
| Field, fixed-point, tensors, manifest types, transcript, serialization, `relation_id` | `area:core` | `pwm-core`; RFC-0002, RFC-0014, RFC-0015 |
| Export / quantization pipeline, manifest writer, golden vectors, parity | `area:export` | `pwm-export`; RFC-0001 |
| NN AIR components and the trace model | `area:air` | `pwm-air`; RFC-0003 – RFC-0008, RFC-0011 |
| Low-level gates, stwo-circuits adapters, witness builder | `area:circuits` | `pwm-circuits` |
| Prover CLI, trace builder, prove_rollout / prove_planning, artifact bundle | `area:prover` | `pwm-prover`; RFC-0016 |
| Verifier library, public-input handling, recursive verifier | `area:verifier` | `pwm-verifier`; RFC-0012 |
| Cost / argmin / CEM proving orchestration | `area:planner` | orchestration across `pwm-air` + `pwm-prover`; RFC-0009, RFC-0010 |
| Specification corpus and RFCs | `area:docs` | `docs/` |
| Continuous integration, build, layering gates | `area:ci` | workspace + CI config |
| Test pyramid, golden vectors, negative/differential/mutation tests | `area:testing` | all crates; RFC-0013, [docs/spec/07-testing-strategy.md#test-pyramid](07-testing-strategy.md#test-pyramid) |
| Threat model, soundness/binding/determinism, privacy/ZK boundary, disclosure | `area:security` | cross-cutting; RFC-0000, [docs/spec/06-security.md#threat-model](06-security.md#threat-model) |

---

## References

- Founding analysis: [docs/feasibility-study.md](../feasibility-study.md) §3 (system architecture), §7.1–7.5 (component model, traces, tensor memory).
- Overview and scope tiers: [docs/spec/00-overview.md#scope-and-statement-tiers](00-overview.md#scope-and-statement-tiers), [docs/spec/00-overview.md#v0-statement](00-overview.md#v0-statement).
- Data model and canonical types: [docs/spec/03-data-model.md#model-manifest](03-data-model.md#model-manifest), [docs/spec/03-data-model.md#public-input](03-data-model.md#public-input), [docs/spec/03-data-model.md#witness](03-data-model.md#witness), [docs/spec/03-data-model.md#tensor-memory-cells](03-data-model.md#tensor-memory-cells).
- Public API: [docs/spec/02-public-api.md#rust-public-api](02-public-api.md#rust-public-api).
- Error and rejection taxonomy: [docs/spec/04-error-model.md#error-taxonomy](04-error-model.md#error-taxonomy), [docs/spec/04-error-model.md#verifier-rejections](04-error-model.md#verifier-rejections).
- Security and soundness: [docs/spec/06-security.md#binding-requirements](06-security.md#binding-requirements), [docs/spec/06-security.md#soundness-requirements](06-security.md#soundness-requirements).
- Testing: [docs/spec/07-testing-strategy.md#differential-tests](07-testing-strategy.md#differential-tests), [docs/spec/07-testing-strategy.md#mutation-tests](07-testing-strategy.md#mutation-tests), [docs/spec/07-testing-strategy.md#negative-tests](07-testing-strategy.md#negative-tests).
- Performance: [docs/spec/08-performance-budget.md#perf-gates](08-performance-budget.md#perf-gates).
- Release and license: [docs/spec/09-release-and-versioning.md#license](09-release-and-versioning.md#license).
- RFCs: RFC-0001 (`docs/rfcs/RFC-0001-model-manifest-and-export-pipeline.md`), RFC-0002 (`docs/rfcs/RFC-0002-fixed-point-arithmetic-over-m31.md`), RFC-0003 (`docs/rfcs/RFC-0003-range-check-and-lookup-infrastructure.md`), RFC-0004 (`docs/rfcs/RFC-0004-tensor-memory-and-wiring-air.md`), RFC-0005 (`docs/rfcs/RFC-0005-linear-matmul-and-requantization-components.md`), RFC-0006 (`docs/rfcs/RFC-0006-nonlinear-primitive-components.md`), RFC-0007 (`docs/rfcs/RFC-0007-leworldmodel-predictor-air.md`), RFC-0008 (`docs/rfcs/RFC-0008-rollout-air.md`), RFC-0009 (`docs/rfcs/RFC-0009-fixed-candidate-planner-proof.md`), RFC-0010 (`docs/rfcs/RFC-0010-cem-planner-proof.md`), RFC-0011 (`docs/rfcs/RFC-0011-pixel-encoder-proof.md`), RFC-0012 (`docs/rfcs/RFC-0012-recursive-aggregated-verification.md`), RFC-0014 (`docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md`), RFC-0015 (`docs/rfcs/RFC-0015-third-party-vendoring-and-pinning.md`), RFC-0016 (`docs/rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md`).
- Upstream (verified against upstream at 2026-06-03): Stwo (github.com/starkware-libs/stwo, Apache-2.0, workspace v2.2.0); stwo-circuits (github.com/starkware-libs/stwo-circuits, Apache-2.0, v0.1.0, edition 2024); LeWorldModel (github.com/lucas-maes/le-wm, MIT).
