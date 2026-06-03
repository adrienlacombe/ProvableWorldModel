# Glossary: Canonical Terms

Status: Normative. This document defines the canonical vocabulary of the ProvableWorldModel corpus. A term used elsewhere in `docs/` carries exactly the meaning fixed here unless that document overrides it explicitly. Each entry gives one tight definition, the field/file that owns the detailed treatment, and (where load-bearing) the invariant or commitment the term participates in. Cross-references use repo-relative paths plus GitHub heading anchors.

Reading guide for cross-references in this file:
- Type signatures are owned by `docs/spec/03-data-model.md` (see [`#tensor-types`](03-data-model.md#tensor-types), [`#public-input`](03-data-model.md#public-input), [`#witness`](03-data-model.md#witness), [`#bounded-integers`](03-data-model.md#bounded-integers), [`#tensor-memory-cells`](03-data-model.md#tensor-memory-cells)). The canonical signatures themselves are frozen in the authoring contract section 6.
- The founding analysis is `docs/feasibility-study.md`.
- External facts about Stwo, stwo-circuits, and LeWorldModel were verified against upstream at 2026-06-03.

Conventions used in entries:
- "Owns:" names the spec doc/RFC that is authoritative for the term.
- "Binds:" indicates the term is committed (see [commitment](#commitment)).
- Verification markers: external facts asserted here were checked against upstream repositories at 2026-06-03.

---

## Group A: Proof system

### circle-stark
A STARK (Scalable Transparent ARgument of Knowledge) instantiated over the **circle group** of a field whose order makes a classical multiplicative-subgroup FFT awkward, using the unit circle `x^2 + y^2 = 1` as the evaluation domain instead. Circle STARK is the proving paradigm of [Stwo](#stwo) and is what ProvableWorldModel arithmetizes against. It is transparent (no trusted setup), uses [FRI](#fri) for low-degree testing, and operates natively over [M31](#m31). Verified against upstream at 2026-06-03. Owns: `docs/spec/01-architecture.md#air-strategy`.

### stwo
StarkWare's Circle STARK prover/verifier stack written in Rust, vendored into this project under `third_party/stwo/` at a pinned revision (workspace v2.2.0 at vendoring reference time). It provides the [field module](#field-module), [FRI](#fri)/[PCS](#pcs), [Fiat-Shamir channels](#channel), the constraint framework, air utilities, and [LogUp](#logup). The project does NOT depend on the upstream canonical crates; it vendors and modifies them. Stwo's workspace package names are `stwo`, `stwo-air-utils`, `stwo-air-utils-derive`, `stwo-constraint-framework`, plus `examples` and `std-shims`. Verified against upstream (github.com/starkware-libs/stwo, Apache-2.0) at 2026-06-03. Owns: `docs/rfcs/RFC-0015-third-party-vendoring-and-pinning.md`.

### air
Algebraic Intermediate Representation: the encoding of a computation as a set of polynomial constraints over [trace](#main-trace) columns. ProvableWorldModel implements custom AIR [components](#component) for neural-network operators (linear, matmul, requant, attention, rollout, cost, argmin) rather than lowering everything to scalar gates. The AIR is what the [validity proof](#validity-proof) certifies. Owns: `docs/spec/01-architecture.md#air-strategy`; per-component RFCs RFC-0004 through RFC-0009.

### fri
Fast Reed-Solomon Interactive Oracle Proof of Proximity: the low-degree test that gives a STARK its [succinctness](#succinctness) and a large part of its [soundness](#soundness). FRI proves that committed column evaluations are close to a low-degree polynomial. Supplied by [Stwo](#stwo); ProvableWorldModel does not reimplement it. Verified against upstream at 2026-06-03. Owns (usage context): `docs/spec/06-security.md#soundness-requirements`.

### pcs
Polynomial Commitment Scheme: the mechanism that commits to trace polynomials and later opens them at verifier-chosen points. In [Stwo](#stwo) the PCS is FRI-based over [M31](#m31). The [PublicInput](03-data-model.md#public-input) digest and trace commitments are bound through the PCS and the [channel](#channel). Verified against upstream at 2026-06-03. Owns (usage context): `docs/spec/06-security.md#binding-requirements`.

### fiat-shamir
The transform that makes the interactive STARK non-interactive by deriving verifier challenges from a transcript [channel](#channel) instead of from a live verifier. Soundness depends on prover and verifier absorbing identical bytes in identical order. ProvableWorldModel fixes that order canonically so the two sides cannot diverge. Owns: `docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md`; usage in `docs/spec/01-architecture.md#trace-model`.

### channel
The Fiat-Shamir transcript object that absorbs committed data and squeezes challenges. [Stwo](#stwo) ships channel backends for **Blake2s**, **Keccak256**, and **Poseidon252** (`crates/stwo/src/core/channel/`). ProvableWorldModel selects one channel per [relation_id](#relation-id) and binds the choice in the manifest; prover and verifier MUST use the identical backend and absorb order. Verified against upstream at 2026-06-03. Owns: `docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md`.

### logup
A logarithmic-derivative lookup argument that proves a multiset of witnessed tuples is contained in (or equal to) a reference multiset, by checking an identity over sums of rational terms `1/(X - value)`. ProvableWorldModel uses LogUp for [range checks](#range-check), [activation lookups](#activation-lookup), weight-table lookups, and [tensor-memory](#tensor-memory) read/write consistency. Implemented in [Stwo](#stwo)'s `constraint-framework/src/logup.rs`. Verified against upstream at 2026-06-03. Owns: `docs/rfcs/RFC-0003-range-check-and-lookup-infrastructure.md`.

### lookup-argument
The general class of arguments (of which [LogUp](#logup) is one) proving that witnessed values appear in a committed table. Used for both **static lookups** (table known before proving, e.g. [range checks](#range-check), [activation tables](#activation-lookup)) and **dynamic lookups** (values not known before proving, e.g. [tensor-memory](#tensor-memory) consistency, [permutation arguments](#permutation-argument)). Owns: `docs/rfcs/RFC-0003-range-check-and-lookup-infrastructure.md`.

### multiplicity
The number of times a given table row is referenced by witness rows in a [lookup argument](#lookup-argument). LogUp soundness requires that the sum of inverse-denominator terms weighted by multiplicities matches between witness side and table side; an inconsistent multiplicity is a rejecting condition (INV-glossary-01: every bounded-integer column referenced in a lookup carries a checked multiplicity). Owns: `docs/rfcs/RFC-0003-range-check-and-lookup-infrastructure.md`; rejection in `docs/spec/04-error-model.md#verifier-rejections`.

### preprocessed-trace
Trace columns fixed before proving and agreed by both prover and verifier: row/operation selectors, static masks (including causal attention masks), public positional embeddings, [quantization](#quantization) scales, rounding constants, and lookup/range table columns. The verifier either recomputes their commitments from the manifest/config or checks committed values. Distinct from the [main trace](#main-trace) and [interaction trace](#interaction-trace). Owns: `docs/spec/01-architecture.md#trace-model`.

### main-trace
The witness trace: columns containing values not fixed in advance, including tensor values, private [weights](#leworldmodel), intermediate activations, accumulators, products, [requantization](#requantization) quotients/remainders, normalization intermediates, attention scores/probabilities, [costs](#mse-q-cost), and [argmin](#argmin) comparison witnesses. Owns: `docs/spec/01-architecture.md#trace-model`.

### interaction-trace
The trace holding the [LogUp](#logup) running sums and [permutation-argument](#permutation-argument) columns derived after the [main trace](#main-trace) is committed and challenges are squeezed from the [channel](#channel). Carries [range checks](#range-check), [activation lookups](#activation-lookup), weight-table lookups, [tensor-memory](#tensor-memory) read/write consistency, and candidate-cost membership. Owns: `docs/spec/01-architecture.md#trace-model`.

### component
A self-contained [AIR](#air) unit with its own columns and constraints, composed with other components into one proof. ProvableWorldModel's components are listed under crate `pwm-air` (range_check, tensor_memory, linear, matmul, requant, activation_lookup, layernorm, attention, mlp, predictor, rollout, cost, argmin, cem). Multiple components share one [Fiat-Shamir channel](#channel). Owns: `docs/spec/01-architecture.md#component-model`.

### soundness
The property that a [verifier](#validity-proof) accepts a false statement only with negligible probability. For ProvableWorldModel, soundness means: a proof that the [verifier](#validity-proof) accepts implies the claimed [latent](#latent) trajectory, costs, and [argmin](#argmin) selection are exactly the result of the committed [quantized](#quantization) [fixed-point](#fixed-point) inference. Soundness is scoped to the declared [relation_id](#relation-id) and presupposes all integer values are [range-checked](#range-check) (finite fields wrap; integer ML inference must not). Owns: `docs/spec/06-security.md#soundness-requirements`.

### completeness
The property that a [verifier](#validity-proof) accepts every honestly generated proof of a true statement. In practice this means the Rust [reference inference](#parity-test), the [witness](03-data-model.md#witness) builder, and the [AIR](#air) agree bit-for-bit so a correctly built proof never spuriously rejects. Owns: `docs/spec/07-testing-strategy.md#golden-vectors`.

### validity-proof
A succinct proof that a statement is true, with no hiding guarantee for the [witness](03-data-model.md#witness). The V0 release produces validity proofs, NOT [zero-knowledge proofs](#zero-knowledge-proof). The corpus never calls V0 zero-knowledge. Owns: `docs/spec/06-security.md#privacy-and-zk`.

### zero-knowledge-proof
A proof that reveals nothing about the [witness](03-data-model.md#witness) beyond the truth of the statement. ProvableWorldModel treats zero-knowledge as an OPTIONAL future security mode gated on a hiding audit (trace commitments hiding/masked, lookups not leaking private tables, public outputs not leaking unintended values). V0 is a [validity proof](#validity-proof), not zero-knowledge. Owns: `docs/spec/06-security.md#privacy-and-zk`; OPEN QUESTION: hiding-audit scope is owned by `area:security`, resolution path RFC (future, post-v1.0).

### succinctness
The property that proof size and verification time are sublinear in (ideally polylogarithmic in) the size of the proven computation. Succinctness is why a STARK proof of a multi-million-MAC predictor rollout can be checked far faster than re-executing it. Delivered by [FRI](#fri)/[PCS](#pcs). Owns (cost framing): `docs/spec/08-performance-budget.md#cost-model`.

---

## Group B: Field and arithmetic

### m31
The Mersenne-31 base field: integers modulo `p = 2^31 - 1`. The canonical Rust type is `M31(u32)` with the value held in `[0, p)`. All [quantized](#quantization) tensor values are embedded into M31 via [centered signed encoding](#centered-signed-encoding). Verified against upstream (`crates/stwo/src/core/fields/m31.rs`) at 2026-06-03. Owns: `docs/spec/03-data-model.md#tensor-types`; RFC-0002.

### cm31
The degree-2 extension of [M31](#m31) (`CM31`). Used internally by the [Stwo](#stwo) proof system as an intermediate toward the [secure field](#qm31). Not used to represent quantized tensor values. Verified against upstream at 2026-06-03. Owns (usage context): `docs/spec/01-architecture.md#air-strategy`.

### qm31
The degree-4 extension of [M31](#m31) (`QM31([M31; 4])`), called the **secure field**. Used for [Fiat-Shamir](#fiat-shamir) challenges and [soundness](#soundness) amplification, NOT to represent quantized tensor values. Verified against upstream at 2026-06-03. Owns: `docs/spec/03-data-model.md#tensor-types`.

### secure-field
Synonym for [QM31](#qm31): the degree-4 [M31](#m31) extension where challenges live and where the soundness error is driven down to a cryptographically negligible level. See [QM31](#qm31).

### field-module
The Rust module `crates/stwo/src/core/fields/` inside the single `stwo` crate that defines [M31](#m31), [CM31](#cm31), and [QM31](#qm31). It is a MODULE, not a set of separate "field crates"; the corpus uses "field module" precisely. Verified against upstream at 2026-06-03. Owns: `docs/rfcs/RFC-0015-third-party-vendoring-and-pinning.md`.

### centered-signed-encoding
The rule mapping a mathematical signed integer `x` into [M31](#m31) as `x mod p`, paired with a [range check](#range-check) proving `lo <= x <= hi`. "Centered" because the representable signed window is centered around zero rather than treating the field as unsigned. The encoding is declared per tensor and is required: an integer embedded without an accompanying range proof is unsound because the field wraps. (INV-glossary-02: every value interpreted as a signed integer carries a centered encoding plus a range witness.) Owns: `docs/spec/03-data-model.md#bounded-integers`; RFC-0002.

### range-check
A [LogUp](#logup)-based [lookup argument](#lookup-argument) proving a witnessed value lies in a declared inclusive interval (e.g. i8 `[-128, 127]`, u16 `[0, 65535]`, or a bounded-[limb](#limb) range). Range checks are the mechanism enforcing [centered signed encoding](#centered-signed-encoding) and preventing silent [field](#m31) wraparound. Every integer column has a declared range. Owns: `docs/rfcs/RFC-0003-range-check-and-lookup-infrastructure.md`. Note: in vendored [stwo-circuits](#third_party) there is NO dedicated range gate; range checking there is built from `sub`/`mul`/`assert_bits` constraints (see [crate](#crate) note under stwo-circuits gates).

### accumulator
The running sum column that aggregates products in a [linear layer](#leworldmodel) or [matmul](#tensor-memory): `acc_{j,i+1} = acc_{j,i} + x_i * w_{i,j}`, initialized at the bias. The accumulator MUST be proven not to exceed its declared range; when it can exceed the safe signed [M31](#m31) interval it is [limb](#limb)-decomposed. (INV-glossary-03: every accumulator is range-checked against bounds derived from tensor metadata; the [overflow policy](#overflow-reject) is reject.) Owns: RFC-0005; RFC-0002.

### limb
A fixed-width digit in a multi-digit decomposition of a value too large for a single safe [M31](#m31) element, e.g. `acc = acc_lo + 2^k * acc_hi`, with each limb [range-checked](#range-check). Used for wide [accumulators](#accumulator), int32 biases, and large [MSE costs](#mse-q-cost). Owns: RFC-0002; RFC-0005.

### overflow-reject
The single overflow policy in V0: arithmetic that would exceed a declared range is a rejecting condition, never a silent wrap. Encoded as `OverflowPolicy::Reject`. Wrapping is unsound for integer ML inference. (INV-glossary-04: there is exactly one overflow policy, `Reject`.) Owns: `docs/spec/03-data-model.md#bounded-integers`; RFC-0002; failure handling `docs/spec/04-error-model.md#failure-modes`.

### clamp
An explicit, per-tensor saturation of a value to a declared output range as part of [requantization](#requantization). The [clamp](#clamp) policy is `explicit` (declared per tensor in the manifest, never implicit). Distinct from [overflow=reject](#overflow-reject): clamping is an intended modeling operation; overflow is an error. Owns: RFC-0002.

---

## Group C: Quantization

### fixed-point
Integer arithmetic that represents a real value as an integer times a fixed [scale](#scale) (a power of two where possible). ProvableWorldModel proves fixed-point inference, NOT floating-point equivalence. The exported model, the Python reference, the Rust reference, and the [AIR](#air) all execute identical fixed-point arithmetic bit-for-bit. Owns: RFC-0002; `docs/spec/00-overview.md#thesis`.

### quantization
The deterministic process of converting a trained floating-point model into the bit-exact [fixed-point](#fixed-point) integer graph that is proven: weights to int8, activations to int8 or int16, biases to int32, accumulators to bounded int32 (or [limb](#limb)-decomposed). The quantization choices are bound by [quantization_commitment](#quantization-commitment). Owns: `docs/rfcs/RFC-0001-model-manifest-and-export-pipeline.md`; RFC-0002.

### scale
The per-tensor power-of-two (where possible) multiplier relating an integer fixed-point value to the real value it encodes. Each [Tensor](03-data-model.md#tensor-types) carries a `scale_id` indexing the manifest scale table. The scale set is bound by [quantization_commitment](#quantization-commitment). Owns: `docs/spec/03-data-model.md#tensor-types`.

### scale-id
The `u32` index (a field of [`Tensor`](03-data-model.md#tensor-types) and [`TensorCell`](#tensorcell)) into the manifest scale table identifying which [scale](#scale) a tensor uses. Reading a value under the wrong `scale_id` is a rejecting condition. Owns: `docs/spec/03-data-model.md#tensor-types`; tensor-memory rejection RFC-0004.

### requantization
The exact integer operation converting an [accumulator](#accumulator) at one [scale](#scale) to an output activation at another, via division-by-power-of-two with explicit quotient/remainder and a declared [rounding](#rounding) mode, followed by an explicit [clamp](#clamp). Never informal division. Defined as `n = q * 2^r + rem`, `0 <= rem < 2^r`, then round per policy. Owns: RFC-0002; RFC-0005; component `pwm-air/requant`.

### rounding
The declared rule resolving the fractional part during [requantization](#requantization). The canonical default is [nearest-ties-to-even](#nearest-ties-to-even); a manifest MAY override to [truncate-toward-zero](#truncate-toward-zero). Exactly one active rounding mode exists per manifest, bound by [quantization_commitment](#quantization-commitment) and enforced bit-for-bit by the Python and Rust references and the [AIR](#air). (INV-glossary-05: a manifest declares exactly one rounding mode.) Encoded as `enum Rounding`. Owns: RFC-0002.

### nearest-ties-to-even
The canonical default [rounding](#rounding) mode (`Rounding::NearestTiesToEven`): round to nearest, ties go to the even quotient. With half `= 2^(r-1)`: `rem < half` keeps `q`; `rem > half` moves toward `q + sign(n)`; `rem == half` adjusts `q` to even. Owns: RFC-0002.

### truncate-toward-zero
The optional [rounding](#rounding) mode (`Rounding::TruncateTowardZero`): drop the fractional part toward zero. A manifest MAY select it instead of [nearest-ties-to-even](#nearest-ties-to-even); the exported model must then use exactly this truncation. Owns: RFC-0002.

### boundedint
The canonical signed-integer type with declared inclusive bounds, embedded into [M31](#m31) via [centered signed encoding](#centered-signed-encoding). Fields: `value: i64` (mathematical signed value), `lo: i64`, `hi: i64`. The [AIR](#air) carries `value mod p` plus a [range-check](#range-check) witness proving `lo <= value <= hi`. Owns: `docs/spec/03-data-model.md#bounded-integers`.

### int8
8-bit signed integer range `[-128, 127]`. V0 weight dtype and one of two activation dtypes (per-tensor, declared in manifest). A length-2048 int8 dot product has worst-case magnitude `2048 * 127 * 127 = 33,032,192 < 2^31 - 1`, so it fits in one signed [M31](#m31) [accumulator](#accumulator) when inputs, weights, and biases are truly int8-bounded. Owns: RFC-0002; accumulator-fit analysis `docs/spec/08-performance-budget.md#cost-model`.

### int16
16-bit signed integer. The wider of the two declared activation dtype options; choosing int16 activations may force [limb](#limb) [accumulators](#accumulator) and [limb](#limb)-decomposed [MSE costs](#mse-q-cost). Owns: RFC-0002.

### int32
32-bit signed integer. V0 bias dtype, and the representation for bounded [accumulators](#accumulator) (held in [M31](#m31) when safe, else [limb](#limb)-decomposed). Owns: RFC-0002.

---

## Group D: AIR and traces

### tensor-memory
The canonical addressed-value model for routing tensor values between [AIR](#air) [components](#component). Each addressed value is a [`TensorCell`](#tensorcell); every operator read must match a prior write (or a declared static constant) and every write has one producer, enforced by a [permutation argument](#permutation-argument)/[multiset](#multiplicity) equality over the [interaction trace](#interaction-trace). Implements the wiring of the NN graph. Owns: `docs/rfcs/RFC-0004-tensor-memory-and-wiring-air.md`; `docs/spec/03-data-model.md#tensor-memory-cells`.

### tensorcell
The canonical record in [tensor memory](#tensor-memory). Fields: `tensor_id: u32`, `index: [u32; 4]` (up to 4 dims; unused dims = 0), `value: M31`, `scale_id: u32`, `time: u32` (write ordering for read/write consistency). Owns: `docs/spec/03-data-model.md#tensor-memory-cells`.

### read-write-consistency
The guarantee that every [tensor-memory](#tensor-memory) read corresponds to a unique valid prior write under the `time` ordering, enforced by a [permutation argument](#permutation-argument). Mutating one read value, swapping indices without a transpose op, or reading at the wrong [scale_id](#scale-id) are rejecting conditions. (INV-glossary-06: each read matches exactly one prior write or a declared constant.) Owns: `docs/rfcs/RFC-0004-tensor-memory-and-wiring-air.md`; rejections `docs/spec/04-error-model.md#verifier-rejections`.

### permutation-argument
A [lookup](#lookup-argument)/[multiset](#multiplicity)-equality argument proving two sequences are permutations of each other. Used for [tensor-memory](#tensor-memory) [read/write consistency](#read-write-consistency), reshape (preserve multiset of values), transpose (preserve index mapping), and concat/slice index mappings. Owns: `docs/rfcs/RFC-0004-tensor-memory-and-wiring-air.md`.

### broadcast
The explicit replication of a constant or smaller tensor across indices of a larger one. Broadcast must be declared (constant_id, index pattern, value) in the manifest or [preprocessed trace](#preprocessed-trace); broadcasting without manifest permission is a rejecting condition. (INV-glossary-07: every broadcast is explicitly declared.) Owns: `docs/rfcs/RFC-0004-tensor-memory-and-wiring-air.md`.

### requant-component
The `pwm-air` [component](#component) implementing [requantization](#requantization): it constrains `acc_next = q * 2^shift + rem`, `0 <= rem < 2^shift`, the declared [rounding](#rounding), and the explicit [clamp](#clamp), wiring quotient/remainder through [range checks](#range-check). Owns: RFC-0005; RFC-0002.

### activation-lookup
The [LogUp](#logup) [lookup argument](#lookup-argument) proving `(x, y)` is a row of a committed [activation](#gelu) table (e.g. GELU_Q, Softmax_Q tables). The table's input domain, output domain, [scale](#scale), [rounding](#rounding), commitment, and maximum approximation error are declared in the manifest. The proof enforces the table; it does NOT prove the table approximates the original function unless an explicit error theorem or exhaustive check is included. Owns: RFC-0003; RFC-0006; approximation binding `docs/spec/06-security.md#binding-requirements`.

---

## Group E: Model

### leworldmodel
The JEPA-style world model proven by this project (upstream: github.com/lucas-maes/le-wm, MIT). Components: [encoder](#encoder), [predictor](#predictor), [action_encoder](#action-encoder), [projector](#projector), [pred_proj](#pred-proj). ~15M trainable params (use "~15M", not an exact count); `num_preds = 1`. The export pipeline consumes a checkpoint and config and does NOT vendor le-wm source; attribute appropriately. Verified against upstream at 2026-06-03. V0 reference configuration: `latent_dim/embed_dim = 192`, `history_size = 3`, predictor `depth = 6`, `heads = 16`, `dim_head = 64`, `mlp_dim = 2048`; image 224x224, [ViT](#encoder) patch 14 (encoder, P4 only). Owns: `docs/spec/00-overview.md#thesis`; RFC-0007.

### jepa
Joint-Embedding Predictive Architecture: a self-supervised paradigm that predicts in a learned latent ([embedding](#latent)) space rather than reconstructing pixels. [LeWorldModel](#leworldmodel) is JEPA-style; its training loss is next-embedding prediction plus [SIGReg](#sigreg). Training is OUT of proving scope. Owns: `docs/spec/00-overview.md#thesis`.

### encoder
The [LeWorldModel](#leworldmodel) component mapping pixel observations to [latents](#latent). It is a **ViT-tiny** (hidden dim 192) with a patch-embedding front end (image 224x224, patch size 14). The `encode` path flattens time, applies the encoder, extracts the CLS token, projects it via the [projector](#projector), and reshapes to latent-sequence form. Excluded from V0; proven only in P4/V3. Verified against upstream at 2026-06-03. Owns: `docs/rfcs/RFC-0011-pixel-encoder-proof.md`.

### patch-embed
The encoder front end that splits a 224x224 image into 14x14-pixel patches and linearly embeds each into a token. Proven as `PatchEmbed_Q` in P4/V3 only. Owns: `docs/rfcs/RFC-0011-pixel-encoder-proof.md`.

### predictor
The [LeWorldModel](#leworldmodel) component that predicts the next [latent](#latent) from a [latent history](#latent-history) and action conditioning. Concretely an [ARPredictor](#arpredictor). Proven as the V0 core (`ARPredictor_Q`). Owns: `docs/rfcs/RFC-0007-leworldmodel-predictor-air.md`.

### arpredictor
The autoregressive transformer-style [predictor](#predictor) class. It uses learned positional embeddings, [ConditionalBlock](#conditionalblock) layers with [AdaLN](#adaln)-zero modulation, an [MLP](#gelu)/FeedForward, `F.scaled_dot_product_attention`, and action conditioning, repeated for `depth = 6`. Verified against upstream at 2026-06-03. Owns: `docs/rfcs/RFC-0007-leworldmodel-predictor-air.md`.

### conditionalblock
One [ARPredictor](#arpredictor) transformer block: self-attention plus FeedForward, both gated by [AdaLN](#adaln)-zero modulation conditioned on the action embedding. Owns: RFC-0007; RFC-0006.

### adaln
Adaptive Layer Normalization (specifically **AdaLN-zero** in [LeWorldModel](#leworldmodel)): normalization whose scale/shift/gate are produced from a conditioning signal (the action embedding), with the gate initialized to zero. The AdaLN modulation uses [SiLU](#silu). Proven as `AdaLN_Q`. Owns: `docs/rfcs/RFC-0006-nonlinear-primitive-components.md`.

### gelu
Gaussian Error Linear Unit: the activation used in the [ARPredictor](#arpredictor) FFN/MLP. Proven as `GELU_Q` via [activation lookup](#activation-lookup) / piecewise-polynomial approximation, with the table/polynomial committed in the manifest. NOTE: activation functions are NOT uniform in [LeWorldModel](#leworldmodel) — GELU is used in the FFN/MLP, while [SiLU](#silu) is used in [AdaLN](#adaln) modulation and the action [Embedder](#action-encoder). Verified against upstream at 2026-06-03. Owns: RFC-0006.

### silu
Sigmoid Linear Unit (a.k.a. swish): the activation used in [LeWorldModel](#leworldmodel)'s [AdaLN](#adaln) modulation and action [Embedder](#action-encoder) (NOT in the FFN/MLP, which uses [GELU](#gelu)). Proven via committed lookup/polynomial approximation. State per-module which activation applies; do not assume uniformity. Verified against upstream at 2026-06-03. Owns: RFC-0006.

### action-encoder
The [LeWorldModel](#leworldmodel) component (an `Embedder`) mapping raw actions to action embeddings consumed by the [predictor](#predictor) conditioning path. The Embedder uses [SiLU](#silu). Proven as `ActionEncoder_Q`; part of the V0 scope. Verified against upstream at 2026-06-03. Owns: RFC-0007.

### embedder
Synonym for the [action_encoder](#action-encoder) module class in [LeWorldModel](#leworldmodel). Uses [SiLU](#silu). See [action-encoder](#action-encoder).

### projector
The [LeWorldModel](#leworldmodel) component projecting the encoder's CLS token into the [latent](#latent) space. Part of the [encoder](#encoder) path; proven as `Projector_Q` in P4/V3 only. Owns: RFC-0011. Not to be confused with [pred_proj](#pred-proj).

### pred-proj
The prediction projector: the [LeWorldModel](#leworldmodel) component applied to the [predictor](#predictor) output to produce the next [latent](#latent). Proven as `PredProj_Q`; part of V0 scope. The V0 predictor relation is `z_next = PredProj_Q(ARPredictor_Q(z_history, ActionEncoder_Q(actions)))`. Owns: RFC-0007.

### sigreg
Sketch Isotropic Gaussian Regularizer: the anti-collapse term in [LeWorldModel](#leworldmodel)'s training loss (weight 0.09), alongside next-embedding prediction. SIGReg is a TRAINING-time regularizer and is OUT of proving scope; it appears in the corpus only as context. Verified against upstream at 2026-06-03. Owns (context only): `docs/spec/00-overview.md#non-goals`.

### latent
A point in [LeWorldModel](#leworldmodel)'s learned [embedding](#latent) space (dim 192). Synonym: embedding. The proof operates over latents represented as [quantized](#quantization) [BoundedInt](#boundedint) tensors. In [`PublicInput`](03-data-model.md#public-input)/[`Witness`](03-data-model.md#witness), latents appear as `Vec<M31>`/[`Tensor`](03-data-model.md#tensor-types). Owns: `docs/spec/03-data-model.md#tensor-types`.

### embedding
Synonym for [latent](#latent). LeWorldModel predicts in embedding space (JEPA); "latent" and "embedding" are used interchangeably in this corpus.

### latent-history
The window of the most recent `history_size = 3` [latents](#latent) forming the [predictor](#predictor) context. Carried in [`PublicInput`](03-data-model.md#public-input) as `latent_history_commitment`/`latent_history_public` and in [`Witness`](03-data-model.md#witness) as `latent_history: Tensor`. Owns: `docs/spec/03-data-model.md#public-input`.

### goal-latent
The target [latent](#latent) against which a rollout's final predicted latent is scored by [MSE_Q cost](#mse-q-cost). Carried in [`PublicInput`](03-data-model.md#public-input) as `goal_latent_commitment`/`goal_latent_public` and in [`Witness`](03-data-model.md#witness) as `goal_latent: Option<Tensor>`. Owns: `docs/spec/03-data-model.md#public-input`.

---

## Group F: Planning

### rollout
The autoregressive application of [Predict_Q](#pred-proj) over an [action sequence](#action-sequence): each step predicts the next [latent](#latent) from the current latent/action window, appends it, and slides the window, for [horizon](#horizon) steps. Mirrors LeWorldModel `jepa.py rollout()`, which concatenates predicted embeddings and next actions. Proven by P1 (`Rollout_Q`). (INV-glossary-08: a rollout proof binds every intermediate predicted latent, not only the final one.) Verified against upstream at 2026-06-03. Owns: `docs/rfcs/RFC-0008-rollout-air.md`.

### horizon
The number of [rollout](#rollout) steps. V0 reference configuration: `horizon = 5` and `action_block = 5` (from the task `plan_config`, `config/eval/pusht.yaml`). Verified against upstream at 2026-06-03. Owns: `docs/rfcs/RFC-0009-fixed-candidate-planner-proof.md`; scaling `docs/spec/08-performance-budget.md#scaling`.

### action-block
The number of actions applied per planning decision in [LeWorldModel](#leworldmodel)'s task `plan_config`; `action_block = 5` in the V0 reference. Distinct from CEM iteration count. Verified against upstream at 2026-06-03. Owns: `docs/spec/08-performance-budget.md#scaling`.

### candidate
One [action sequence](#action-sequence) `s` evaluated by the planner. In fixed-candidate planning (P2), the proof rolls out ALL `S` candidates and proves all costs. Carried in [`PublicInput`](03-data-model.md#public-input) as `candidate_actions_*` and [`Witness`](03-data-model.md#witness) as `candidate_actions: Option<Tensor>`. Owns: `docs/rfcs/RFC-0009-fixed-candidate-planner-proof.md`.

### action-sequence
An ordered sequence of actions over the [horizon](#horizon) driving one [rollout](#rollout). A [candidate](#candidate) is an action sequence under evaluation. Owns: RFC-0008; RFC-0009.

### mse-q-cost
The quantized fixed-point cost of a [rollout](#rollout): the sum of squared differences between the final predicted [latent](#latent) and the [goal latent](#goal-latent), `cost = Σ_j (z_j - g_j)^2`, with `diff_j`, `sq_j`, and the running `cost_j` each [range-checked](#range-check) (and [limb](#limb)-decomposed if int16 latents make the sum exceed safe [M31](#m31)). Mirrors LeWorldModel `F.mse_loss` on the final latent. Proven by the `pwm-air/cost` [component](#component). Verified against upstream at 2026-06-03. Owns: `docs/rfcs/RFC-0009-fixed-candidate-planner-proof.md`; RFC-0002.

### argmin
The deterministic selection of the [candidate](#candidate) with minimum [MSE_Q cost](#mse-q-cost). Proven by binding `selected_cost == cost[selected_index]` and `selected_cost <= cost_s` for all `s` (via nonnegative-difference [range checks](#range-check)), plus the [tie-break](#tie-break). Proving only the selected rollout is INSUFFICIENT; all candidate costs must be proven or the prover can pick favorable candidates off-circuit. (INV-glossary-09: argmin soundness requires all candidate costs proven.) Owns: `docs/rfcs/RFC-0009-fixed-candidate-planner-proof.md`; [`ArgminWitness`](03-data-model.md#witness) in `docs/spec/03-data-model.md#witness`.

### tie-break
The deterministic rule resolving equal-minimum costs: the smallest index attaining the minimum wins. Enforced with `cost_s - selected_cost - 1 >= 0` [range checks](#range-check) for all `s < selected_index` (strict greater-than is awkward in-field, so this is encoded as a nonnegative difference of `cost_s - selected_cost - 1`). (INV-glossary-10: the smallest minimizing index is selected.) Owns: RFC-0009; `docs/spec/06-security.md#soundness-requirements`.

### cem
Cross-Entropy Method: the stochastic planner [LeWorldModel](#leworldmodel) uses for evaluation. It seeds a PRNG, samples [candidates](#candidate), clips them to the action domain, scores all, selects [top-k](#top-k) elites, updates the sampling mean/variance, and iterates. V0 does NOT prove CEM (it proves fixed candidates, P2); a sound CEM proof (P3/V2) must bind seeded sampling, clipping, all costs, top-k, mean/var updates, and the iteration recurrence. V0 reference solver values (`config/eval/solver/cem.yaml`): `num_samples = 300`, `n_steps = 30` (CEM iterations), `topk = 30` (elite count), `var_scale = 1.0`, `batch_size = 1`. Verified against upstream at 2026-06-03. Owns: `docs/rfcs/RFC-0010-cem-planner-proof.md`.

### num-samples
The CEM candidate count per iteration: `num_samples = 300` in the V0 reference solver config. Lives in the solver config, not the task `plan_config`. Verified against upstream at 2026-06-03. Owns: RFC-0010; `docs/spec/08-performance-budget.md#scaling`.

### n-steps
The number of [CEM](#cem) iterations: `n_steps = 30` in the V0 reference solver config. Distinct from [horizon](#horizon) (rollout length) and [action_block](#action-block). Verified against upstream at 2026-06-03. Owns: RFC-0010.

### top-k
The elite-selection step of [CEM](#cem): keep the `topk = 30` lowest-cost [candidates](#candidate) to update the sampling distribution. A sound CEM proof must prove top-k membership at every iteration. Verified against upstream at 2026-06-03. Owns: RFC-0010.

### receding-horizon
The planning regime where the planner re-plans after applying part of the chosen [action sequence](#action-sequence), advancing the window. In the V0 reference task `plan_config`, `receding_horizon = 5`. A planning concept for the controller loop; the proof certifies one planning invocation. Verified against upstream at 2026-06-03. Owns: `docs/spec/08-performance-budget.md#scaling`.

---

## Group G: Commitments and versioning

### commitment
A binding cryptographic digest (`[u8; 32]`) that fixes a value so it cannot later be changed without detection. Anything NOT committed is mutable by the prover and therefore unsound. The corpus's commitments are [model_commitment](#model-commitment), [quantization_commitment](#quantization-commitment), [planner_config_commitment](#planner-config-commitment), and the per-input/output commitments in [`PublicInput`](03-data-model.md#public-input). Owns: `docs/spec/06-security.md#binding-requirements`.

### model-commitment
The `[u8; 32]` digest binding the model: architecture, weights, biases, quantization scales, rounding rules, lookup tables, activation/normalization approximations, tensor shapes, planner config, relation version, and serialization version. A field of [`PublicInput`](03-data-model.md#public-input). (INV-glossary-11: every semantically relevant model parameter is under model_commitment.) Owns: `docs/spec/03-data-model.md#model-manifest`; RFC-0001.

### quantization-commitment
The `[u8; 32]` digest binding the [quantization](#quantization) regime: dtypes, [scales](#scale), [rounding](#rounding) mode, [overflow policy](#overflow-reject), [clamp](#clamp) policy, and the activation-tables commitment. A field of [`PublicInput`](03-data-model.md#public-input). Owns: `docs/spec/03-data-model.md#model-manifest`; RFC-0001.

### planner-config-commitment
The `[u8; 32]` digest binding the planner configuration ([horizon](#horizon), [action_block](#action-block), tie-break rule, and, for P3, [CEM](#cem) sampling parameters). A field of [`PublicInput`](03-data-model.md#public-input); changing the planner config without changing this digest is a rejecting condition. Owns: `docs/spec/03-data-model.md#public-input`; RFC-0009.

### relation-id
The immutable identifier of the exact arithmetic relation a proof certifies, scheme `pwm.lewm.<statement>.v<N>` (e.g. `pwm.lewm.predictor_step.v1`, `pwm.lewm.rollout.v1`, `pwm.lewm.fixed_candidate_planning.v1`). Carried in [`PublicInput`](03-data-model.md#public-input) as `relation_id: [u8; 32]`. A proof is valid ONLY for its declared relation_id; any semantic change mints a new id, and ids are never reused. (INV-glossary-12: relation_ids are immutable and statement-scoped.) Owns: `docs/spec/09-release-and-versioning.md#relation-versioning`; RFC-0000.

### manifest
The canonical YAML artifact defining the exact model proven, bound by [model_commitment](#model-commitment) and [quantization_commitment](#quantization-commitment). Top-level keys: `manifest_version, model_family, relation_id, field{...}, quantization{...}, architecture{...}, weights{...}, ops[]{...}, serialization{canonical_json_hash}`. Re-exporting the same checkpoint yields a byte-identical manifest. Owns: `docs/spec/03-data-model.md#model-manifest`; RFC-0001.

### manifest-version
The schema version string of the [manifest](#manifest) (e.g. `pwm-model-manifest-v1`). Governs how the manifest is parsed and is part of the committed serialization version. Owns: `docs/spec/09-release-and-versioning.md#relation-versioning`; `docs/spec/03-data-model.md#schema-versioning`.

### artifact-version
The `u32` schema version of the on-disk [`ProofArtifact`](#validity-proof) bundle (`artifact_version`, with `public_input`, `proof`, `claimed_outputs`). Distinct from [manifest_version](#manifest-version) and [relation_id](#relation-id): it versions the bundle envelope, not the relation. Owns: `docs/spec/02-public-api.md#artifact-formats`; RFC-0016.

### relation-id-binding
Shorthand for the rule that the [verifier](#validity-proof) rejects any proof whose declared [relation_id](#relation-id) it does not support, and that a proof generated for one statement (e.g. P0) submitted as another (e.g. P1) is rejected. See [relation-id](#relation-id). Owns: `docs/spec/04-error-model.md#verifier-rejections`.

---

## Group H: Testing and project structure

### golden-vector
A fixed input/output test case capturing the exact [fixed-point](#fixed-point) reference behavior, used to lock down [completeness](#completeness): the Python reference, the Rust reference, and the [AIR](#air) witness all reproduce it bit-for-bit. Every nonlinear primitive ([GELU](#gelu), [SiLU](#silu), [AdaLN](#adaln), softmax) ships golden vectors. Owns: `docs/spec/07-testing-strategy.md#golden-vectors`; RFC-0013.

### parity-test
A test asserting two implementations agree bit-for-bit, principally Rust [fixed-point](#fixed-point) reference inference versus the Python fixed-point reference. Re-exporting a checkpoint twice yielding a byte-identical [manifest](#manifest) is a parity test of the export pipeline. Owns: `docs/spec/07-testing-strategy.md#differential-tests`; RFC-0001.

### negative-test
A REJECT test: an invalid [witness](03-data-model.md#witness) or tampered [public input](03-data-model.md#public-input) that the [verifier](#validity-proof) must reject (e.g. mutated accumulator, out-of-domain activation, wrong [argmin](#argmin), violated [tie-break](#tie-break), changed [commitment](#commitment)). No [component](#component) ships without both accepting AND rejecting tests. (INV-glossary-13: every component has at least one accepting and one rejecting test.) Owns: `docs/spec/07-testing-strategy.md#negative-tests`; RFC-0013.

### reject-test
Synonym for [negative-test](#negative-test). See [negative-test](#negative-test).

### differential-test
A test comparing outputs across random seeds or across two reference implementations to surface divergence (e.g. Python vs Rust fixed-point planners producing identical bit-for-bit selections given the same seed/config). Owns: `docs/spec/07-testing-strategy.md#differential-tests`; RFC-0013.

### constraint-mutation-test
A test that deliberately mutates a single [AIR](#air) constraint (or a witness value) and asserts the [verifier](#validity-proof) now rejects, confirming the constraint is load-bearing rather than vacuous. Owns: `docs/spec/07-testing-strategy.md#mutation-tests`; RFC-0013.

### vendoring
The practice of copying an external dependency's source into `third_party/` at a pinned revision and modifying it as needed, instead of depending on the upstream published crate. [Stwo](#stwo) and [stwo-circuits](#third_party) are vendored. The pinned revision is recorded in `third_party/stwo/REVISION` (and the equivalent for stwo-circuits). Owns: `docs/rfcs/RFC-0015-third-party-vendoring-and-pinning.md`.

### third_party
The repository directory holding [vendored](#vendoring) external code at pinned revisions: `third_party/stwo/` and `third_party/stwo-circuits/`. stwo-circuits' workspace has 10 crates (`cairo_verifier`, `circuits`, `circuit_verifier`, `circuit_cairo_serialize`, `circuit_common`, `circuit_multiverifier`, `circuit_serialize`, `circuit_prover`, `stark_verifier`, `stark_verifier_examples`); its first-class gate set (the 10 fields of `struct Circuit`) is `Add, Sub, Mul, PointwiseMul, Eq, TripleXor, M31ToU32, BlakeGGate, Permutation, Output` — there is NO range/bit-extraction gate (`extract_bits()` is a helper built from `sub`/`mul`/`assert_bits` constraints, and `BlakeGGate` is a Blake2s G-function gate, not a full-hash gate). Verified against upstream (Apache-2.0, v0.1.0, edition 2024) at 2026-06-03. Owns: `docs/rfcs/RFC-0015-third-party-vendoring-and-pinning.md`; `docs/spec/01-architecture.md#air-strategy`.

### crate
A Rust workspace member under `crates/`. The ProvableWorldModel workspace members are:

| Crate | Responsibility | Area label |
| --- | --- | --- |
| `pwm-core` | Field/fixed-point types, tensors, manifest types, transcript, relation_id, canonical serialization | `area:core` |
| `pwm-export` | Python+Rust export/quantization pipeline, manifest writer, golden-vector + parity tests | `area:export` |
| `pwm-air` | AIR [components](#component) (range_check, tensor_memory, linear, matmul, requant, activation_lookup, layernorm, attention, mlp, predictor, rollout, cost, argmin, cem) | `area:air` |
| `pwm-circuits` | Low-level circuit gates + adapters to vendored stwo-circuits, witness builder | `area:circuits` |
| `pwm-prover` | Prover CLI, trace builder, prove_rollout, prove_planning | `area:prover` |
| `pwm-verifier` | Verifier library, public-input handling, recursive verifier | `area:verifier` |

Distinct from a vendored [stwo](#stwo)/[stwo-circuits](#third_party) crate, which lives under [third_party](#third_party). Owns: `docs/spec/01-architecture.md#crate-layout`.

---

## Appendix: invariant index

Invariants named in this glossary, for cross-reference by other documents. Each is stated at its defining term above.

| Invariant | Statement | Defining term |
| --- | --- | --- |
| INV-glossary-01 | Every bounded-integer column in a lookup carries a checked multiplicity | [multiplicity](#multiplicity) |
| INV-glossary-02 | Every signed-integer value carries centered encoding plus a range witness | [centered-signed-encoding](#centered-signed-encoding) |
| INV-glossary-03 | Every accumulator is range-checked against metadata-derived bounds; overflow=reject | [accumulator](#accumulator) |
| INV-glossary-04 | There is exactly one overflow policy: Reject | [overflow-reject](#overflow-reject) |
| INV-glossary-05 | A manifest declares exactly one rounding mode | [rounding](#rounding) |
| INV-glossary-06 | Each tensor-memory read matches exactly one prior write or a declared constant | [read-write-consistency](#read-write-consistency) |
| INV-glossary-07 | Every broadcast is explicitly declared | [broadcast](#broadcast) |
| INV-glossary-08 | A rollout proof binds every intermediate predicted latent | [rollout](#rollout) |
| INV-glossary-09 | Argmin soundness requires all candidate costs proven | [argmin](#argmin) |
| INV-glossary-10 | The smallest minimizing index is selected | [tie-break](#tie-break) |
| INV-glossary-11 | Every semantically relevant model parameter is under model_commitment | [model-commitment](#model-commitment) |
| INV-glossary-12 | relation_ids are immutable and statement-scoped | [relation-id](#relation-id) |
| INV-glossary-13 | Every component has at least one accepting and one rejecting test | [negative-test](#negative-test) |

These invariants restate, at glossary granularity, requirements owned in detail by `docs/spec/06-security.md#soundness-requirements`, `docs/spec/06-security.md#binding-requirements`, and `docs/spec/07-testing-strategy.md#negative-tests`. Where a more specific invariant in those documents conflicts with a glossary restatement, the owning document governs.
