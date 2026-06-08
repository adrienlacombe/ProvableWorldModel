# Security: Threat Model, Trust Boundaries, Soundness, Privacy, Disclosure

Status: Normative. This document defines what ProvableWorldModel guarantees against a
malicious prover, what the verifier trusts, the binding/determinism/range-safety/
approximation/planner obligations that make the V0 relation sound, the limits of the
V0 privacy story, secrets handling, and the vulnerability-disclosure process. It is
binding on `pwm-core`, `pwm-air`, `pwm-circuits`, `pwm-prover`, and `pwm-verifier`.
Soundness requirements here are enforced by the RFCs cited per item; a requirement
that is stated but not traced to an enforcing RFC is a specification defect.

This document does not re-derive the arithmetic semantics (see
[docs/spec/03-data-model.md#bounded-integers](03-data-model.md#bounded-integers)),
the error taxonomy (see
[docs/spec/04-error-model.md#verifier-rejections](04-error-model.md#verifier-rejections)),
or the test plan that exercises every claim here (see
[docs/spec/07-testing-strategy.md#negative-tests](07-testing-strategy.md#negative-tests)).
It defines the adversary, the boundary, and the obligations.

---

## Threat model
<a id="threat-model"></a>

### Adversary

The adversary is a **malicious prover** with full control of the proving host. Its
single goal is to make the verifier `accept` a proof for a statement that is false
under the declared `relation_id`. There is no honest-but-curious variant in V0: the
prover is assumed Byzantine. The verifier is the only trusted party in the protocol;
the prover, the export pipeline, and the proving host are all untrusted.

We adopt the standard STARK security model. The relation is sound under the
conjectured soundness of the underlying Circle-STARK + FRI argument as implemented in
the vendored Stwo prover/verifier (Stwo, Apache-2.0, workspace v2.2.0; verified
against upstream at 2026-06-03). "Soundness" in this document means: no
polynomial-time prover can produce an accepting proof for a public input that does not
admit a valid witness for the declared relation, except with probability negligible in
the soundness security parameter `lambda`. The probability that a cheating prover
succeeds is bounded by the FRI/PCS soundness error over the secure field QM31; the
target `lambda` and the concrete FRI parameters (blowup, number of queries,
proof-of-work bits) are public inputs bound into the transcript (RFC-0014) and recorded
in the manifest `security` block (RFC-0001). Lowering `lambda` below the configured
floor is itself a rejected statement (INV-SEC-12).

### Adversary capabilities

The threat model assumes the prover can do all of the following, and the design must
remain sound against each. The right column is the design feature that closes the
capability; "unbound" means the capability succeeds if and only if the corresponding
value is not committed into the public input.

| # | Capability | What the adversary tries | Closed by |
|---|------------|--------------------------|-----------|
| C1 | Choose the witness freely | Fabricate intermediate activations, accumulators, quotients/remainders, attention probabilities | AIR constraints make every intermediate a function of bound inputs (RFC-0002, RFC-0005, RFC-0006); read/write consistency (RFC-0004) |
| C2 | Swap weights/biases if unbound | Prove a cheaper or different model than claimed | `model_commitment` binds the full weight/bias Merkle root (RFC-0001); weight reads are lookups against the committed table (RFC-0003) |
| C3 | Swap approximations if unbound | Replace the committed GELU/softmax/inv-sqrt table or polynomial with a favorable one | `quantization_commitment` binds every activation/normalization table and polynomial commitment (RFC-0001, RFC-0006) |
| C4 | Change rounding/clamp/overflow rules if unbound | Round to make a comparison flip, or wrap instead of reject | Exactly one rounding mode + clamp policy + `overflow_policy=reject` bound by `quantization_commitment` (RFC-0002) |
| C5 | Exploit field wraparound | Pick a witness whose mathematical integer overflows but whose value mod `p` satisfies the constraint | Every integer-interpreted column is range-checked to its declared `[lo, hi]` (RFC-0002, RFC-0003); INV-SEC-08 |
| C6 | Misroute tensors | Read a value from the wrong cell, scale, or time; reuse a stale write; broadcast without permission | TensorCell read/write multiset/permutation argument (RFC-0004); INV-SEC-09 |
| C7 | Prove a subset of the planner | Roll out only the winning candidate, or score against an off-circuit cost table | Prove ALL candidate rollouts, ALL costs, and the argmin with tie-break in-circuit (RFC-0009); INV-SEC-10 |
| C8 | Mislabel the statement | Submit a P0 proof as a P1/P2 proof, or reuse a proof under a different `relation_id` | `relation_id` and `statement_type` are bound into the public input and the transcript (RFC-0000, RFC-0014); INV-SEC-01 |
| C9 | Forge the public-input digest | Reorder, re-encode, or pad public inputs so a stale proof verifies against new inputs | Canonical serialization + single public-input digest absorbed first into the Fiat-Shamir channel (RFC-0014); INV-SEC-02 |
| C10 | Grind challenges | Re-seed Fiat-Shamir off the canonical channel order to find a lucky transcript | Prover and verifier share the exact channel ordering; verifier re-derives all challenges (RFC-0014); INV-SEC-03 |
| C11 | Tamper the vendored prover/verifier | Modify Stwo to skip a check, then ship the patched verifier | The verifier the relying party runs is pinned and reproducible; vendoring + audit boundary (RFC-0015); the soundness claim is gated on the security review (see [#disclosure](#disclosure)) |
| C12 | Downgrade security params | Set `lambda`, FRI queries, or PoW bits below the floor | Security params are bound and checked against a configured minimum (RFC-0014); INV-SEC-12 |

The unifying rule (INV-SEC-04): **anything not bound by a public commitment is mutable
by the prover and therefore must be treated as adversary-controlled.** The full binding
list is in [#binding-requirements](#binding-requirements).

### Assets

| Asset | Protected in V0? | Mechanism / note |
|-------|------------------|------------------|
| Soundness of the declared relation | Yes (primary asset) | STARK soundness over QM31 + the full binding list. This is the asset V0 exists to protect. |
| Integrity of the public statement | Yes | Public-input digest + transcript (RFC-0014). |
| Privacy of weights | No in V0 (optional later) | V0 is a validity proof, not ZK. Weights MAY be committed-only (not public), but the trace commitments are not hiding. See [#privacy-and-zk](#privacy-and-zk). |
| Privacy of inputs/latents | No in V0 (optional later) | Same as weights: commitment-only does not imply hiding in V0. |
| Availability of the prover | Out of scope | Denial of service against the proving host is not a soundness concern; a failed prove run yields no proof, never a false one. |

V0 protects exactly one asset cryptographically: the soundness of the arithmetic
relation. Privacy is explicitly out of scope for V0 (RFC-0000) and is only achievable
after the hiding audit described in [#privacy-and-zk](#privacy-and-zk).

### Out of scope (and why this is honest, not a gap)

The proof binds an arithmetic relation, not the world. The following are NOT claimed
and a malicious prover gains nothing by attacking them, because the verifier never
asserts them:

- That the predicted future is physically true, or the model is calibrated.
- That the selected action is globally optimal in the real environment.
- That the original floating-point PyTorch model produces the same bits. V0 proves the
  exported quantized graph (`QuantizedLeWM`), not bf16/GPU PyTorch. See
  [docs/spec/00-overview.md#non-goals](00-overview.md#non-goals).
- That training was valid (next-embedding loss + SIGReg, weight 0.09, are training-time
  concerns; training is outside the proving scope entirely).

Conflating any of these with the proof statement is a disclosure/marketing failure, not
a cryptographic one; the rule in [#disclosure](#disclosure) forbids it.

---

## Trust boundaries
<a id="trust-boundaries"></a>

The boundary is the line the verifier will not cross. Everything the verifier executes
or asserts is inside; everything it refuses to execute is outside.

```text
                          TRUST BOUNDARY
   UNTRUSTED (outside)    |          TRUSTED (inside the verifier)
   ----------------------- | -----------------------------------------
   PyTorch / bf16 / GPU    |   the relation_id registry (allow-list)
   the export pipeline     |   the bound public commitments
   the quantizer           |   canonical (de)serialization (RFC-0014)
   the manifest author     |   the Fiat-Shamir transcript order (RFC-0014)
   the proving host        |   the vendored Stwo verifier (pinned, RFC-0015)
   the Proof bytes         |   the AIR constraint set / math
   the Witness (private)   |   the field arithmetic over M31/QM31
   ----------------------- | -----------------------------------------
                Proof + PublicInput cross the boundary;
                Witness never does (it stays prover-side).
```

Prose statement of the diagram: the verifier receives a `ProofArtifact` (the proof
bytes plus the `PublicInput`; see
[docs/spec/02-public-api.md#artifact-formats](02-public-api.md#artifact-formats)).
It trusts only the items in the right column. It treats the proof bytes and any
prover-asserted output as untrusted until the STARK verification and the
application-level checks pass. The private `Witness` never crosses the boundary.

### What the verifier trusts

| Trusted item | Why it is in scope | Source of truth |
|--------------|--------------------|-----------------|
| The `relation_id` registry | The verifier only accepts statements it recognizes; an unknown id is rejected, not interpreted | RFC-0000; [docs/spec/02-public-api.md#rust-public-api](02-public-api.md#rust-public-api) |
| The bound public commitments | `model_commitment`, `quantization_commitment`, `planner_config_commitment`, input/output commitments — the verifier trusts that these pin the semantics | RFC-0001, RFC-0014; [#binding-requirements](#binding-requirements) |
| Canonical serialization | The bytes hashed into the public-input digest are unambiguous | RFC-0014; [docs/spec/03-data-model.md#schema-versioning](03-data-model.md#schema-versioning) |
| The Fiat-Shamir channel order | Soundness of the non-interactive argument depends on identical absorb/squeeze order | RFC-0014 |
| The vendored Stwo verifier | The Circle-STARK + FRI soundness assumption, as implemented in the pinned, audited verifier path (`no_std`-capable) | RFC-0015 |
| The AIR / the math | The constraint set is the relation; the math (field, FRI, LogUp) is assumed sound | RFC-0002 through RFC-0009 |

### What the verifier explicitly does NOT trust or run

INV-SEC-05: **The verifier never executes the model in floating point, and never runs
Python, PyTorch, ONNX, the exporter, or the quantizer.** It verifies the arithmetic
relation only. This is not an optimization; running the float model would reintroduce
the nondeterminism the whole project exists to eliminate.

| Untrusted item | Consequence if it lies | Containment |
|----------------|------------------------|-------------|
| PyTorch / float model | Could disagree with the exported graph bit-for-bit | Verifier never runs it; the proof is about the exported graph, bound by `model_commitment` |
| The export pipeline / quantizer | Could mis-quantize or mislabel scales | The manifest is bound; a wrong manifest changes the commitment and so changes the statement, not the verdict. Re-export must be byte-identical (RFC-0001). |
| The proving host | Could be fully compromised | Byzantine prover is the assumed adversary; soundness holds regardless |
| The `Proof` bytes | Could be malformed or forged | STARK verification rejects; a malformed bundle is a structured `VerifyError` (see [docs/spec/04-error-model.md#verifier-rejections](04-error-model.md#verifier-rejections)) |
| The private `Witness` | Adversary-chosen by definition | Constrained by the AIR; never crosses the boundary |

A consequence of INV-SEC-05: the verifier's correctness depends only on the vendored
Stwo verifier path being the one actually deployed. The vendoring/pinning policy
(RFC-0015) and the reproducibility contract (RFC-0016) exist so a relying party can
confirm the verifier they run matches the audited revision. A relying party that runs
an unaudited or modified verifier is outside the security model.

---

## Soundness requirements
<a id="soundness-requirements"></a>

Soundness is the conjunction of five obligation sets: binding, determinism,
range-safety, approximation, and planner soundness. Each is normative. Each item is
named and traced to its enforcing RFC. The negative-test obligation for every item is
in [docs/spec/07-testing-strategy.md#negative-tests](07-testing-strategy.md#negative-tests):
no requirement here ships without a rejecting test proving the verifier catches its
violation.

### Determinism requirements

INV-SEC-06: Every stage that contributes to the bound commitments or the proof inputs
MUST be deterministic — same inputs produce the same bytes on every host, OS, and
toolchain version within the supported matrix. Floating point appears nowhere in the
proving or verification path. The reproducibility contract is RFC-0016; the canonical
serialization is RFC-0014.

| Stage | Determinism obligation | Enforcing RFC |
|-------|------------------------|---------------|
| Model export | Same checkpoint+config -> byte-identical manifest and weight tensors | RFC-0001 |
| Quantization | Fixed scales, fixed rounding; no float intermediates that affect output bits | RFC-0001, RFC-0002 |
| Rounding | Exactly one active mode per manifest (`NearestTiesToEven` default, `TruncateTowardZero` optional) | RFC-0002 |
| Clamping | Explicit per-tensor clamp range; deterministic saturation | RFC-0002 |
| Activation / normalization approximation | Committed table or polynomial; same input -> same output | RFC-0006 |
| Rollout windowing | Fixed window selection (horizon=5, action_block=5 for the V0 reference task) | RFC-0008 |
| Argmin tie-break | Smallest index attaining the minimum wins | RFC-0009 |
| Public-input serialization | Canonical, versioned, byte-stable | RFC-0014 |
| Fiat-Shamir transcript | Identical absorb/squeeze order, prover and verifier | RFC-0014 |

A nondeterministic stage is a soundness hole: if the reference and the AIR can diverge,
the AIR is not proving the reference. The parity gate (Python fixed-point reference ==
Rust fixed-point reference == AIR, bit-for-bit) in RFC-0001 and RFC-0013 is the
machine-checked enforcement of INV-SEC-06.

### Range-safety requirements

INV-SEC-08: **Every value interpreted as an integer MUST be range-checked to its
declared `[lo, hi]`.** Finite fields wrap; integer ML inference must not. The
`overflow_policy` is `reject` (never wrap). A column whose mathematical value exceeds
its declared bound is rejected even when its value mod `p` happens to satisfy a
downstream constraint — this is the exact attack in capability C5. Range checks are
LogUp-based lookups against `u8`/`i8`/`u16`/bounded-limb tables (RFC-0003); accumulators
that can exceed the safe signed M31 interval are limb-decomposed and each limb is
range-checked (RFC-0005).

The minimum required range-checked columns (a value-bearing list, not a category):

```text
input latents                 attention scores
actions                       softmax/probability numerators and denominators
action embeddings             layernorm/adaln mean, centered values, inv-std operands
weights                       requantization quotients
biases                        requantization remainders (0 <= rem < 2^shift)
products (x_i * w_{i,j})       activation lookup outputs
accumulators (and limbs)      cost differences (z_j - g_j) and squares
per-step predicted latents    argmin diffs (cost_s - selected_cost >= 0)
```

For comparisons, there is no native field `<=`. The verifier accepts `a <= b` only via
a witnessed nonnegative difference `d = b - a` with `0 <= d <= D_max` range-checked. For
the strict inequality in tie-break (`cost_s > selected_cost` for `s < selected_index`),
the constraint is `cost_s - selected_cost - 1 >= 0` range-checked. A comparison that
wraps the field is a rejected witness (INV-SEC-08 + RFC-0002).

### Approximation requirements

INV-SEC-07: Every approximate primitive MUST be fully committed in the manifest and
fully enforced by the AIR. The proof enforces the approximation; it does NOT prove the
approximation is close to the original float operator unless an explicit error theorem
or exhaustive table check is included (RFC-0006). For each approximate primitive the
manifest binds the following schema (bound by `quantization_commitment`):

```yaml
approximation:
  id: gelu_lookup_v1            # immutable identifier; semantic change mints a new id
  primitive: gelu               # one of: gelu | softmax | inv_sqrt | layernorm | reciprocal | division
  module: predictor.ffn         # WHERE it applies; activations are NOT uniform (see below)
  domain: { lo: -2048, hi: 2047 }   # bound integer input domain
  scale_id: 7                   # fixed-point scale of the operand
  representation: lookup_table  # lookup_table | piecewise_poly | newton_iteration | folded_affine
  table_commitment: 0x...       # commitment to the (x, y) pairs, if lookup_table
  poly_commitment: 0x...        # commitment to coefficients, if piecewise_poly (else omitted)
  rounding: nearest_ties_to_even
  max_error: 1                  # max |approx - reference| in output ULPs, for the audit, not the soundness
  test_vectors_commitment: 0x...  # golden (x, y) vectors, committed
```

The activation choice is **per-module, not uniform** (verified against upstream at
2026-06-03): the LeWorldModel predictor FFN/MLP uses **GELU**, while the AdaLN
modulation and the action `Embedder` use **SiLU**. The `module` field above is therefore
load-bearing: a manifest that declares one global activation is wrong and will not match
the exported graph parity test (RFC-0001, RFC-0007). The rejected approaches —
off-circuit softmax probabilities, unconstrained reciprocal/inverse-sqrt, floating-point
helper values, implicit PyTorch semantics — are enumerated in RFC-0006 and each has a
rejecting test.

### Planner soundness requirements

INV-SEC-10: For fixed-candidate planning (P2 / V0), proving only the selected candidate
is unsound. The proof MUST establish, in-circuit, for the full candidate set `s in
0..S-1`:

```text
for every s:   trajectory_s == Rollout_Q(latent_history, candidate_actions_s)   (RFC-0008)
for every s:   cost_s        == MSE_Q(final_latent_s, goal_latent)              (RFC-0009)
               selected_cost == cost_{selected_index}                            (RFC-0009)
for every s:   selected_cost <= cost_s                                           (range-checked diff)
tie-break:     for every s < selected_index:  cost_s - selected_cost - 1 >= 0    (range-checked)
```

If any candidate rollout or cost is omitted, the prover can score favorable candidates
off-circuit and the planning claim is hollow (capability C7). The argmin witness carries
one nonnegative diff per candidate (`ArgminWitness.diffs`, see
[docs/spec/03-data-model.md#witness](03-data-model.md#witness)); a missing or
negative diff is a rejected witness.

INV-SEC-11: For CEM planning (P3 / V2, out of V0 scope), proving only the final
candidate is unsound. A sound CEM proof additionally requires seeded sample generation,
candidate clipping, ALL costs, top-k selection, mean update, variance/std update, and
the iteration recurrence (RFC-0010). The V0 reference CEM configuration (verified
against upstream at 2026-06-03) is `num_samples=300`, `n_steps=30`, `topk=30`,
`var_scale=1.0` (solver config) and `horizon=5`, `receding_horizon=5`, `action_block=5`
(task `plan_config`). V0 does NOT attempt P3; stating these values is for scoping the
future proof, not a V0 claim.

---

## Binding requirements
<a id="binding-requirements"></a>

INV-SEC-04 restated as the operational rule: a value affects the statement if and only
if it is bound by a public commitment. The complete binding list below is the contract
that closes capabilities C2, C3, C4, C8, C9, and C12. It is enforced primarily through
the model and quantization commitments (RFC-0001), the public-input digest (RFC-0014),
and the `relation_id` discipline (RFC-0000).

| # | Bound item | Commitment that binds it | Enforcing RFC | Capability closed |
|---|------------|--------------------------|---------------|-------------------|
| B1 | `relation_id` (statement identity) | public input + transcript | RFC-0000, RFC-0014 | C8 |
| B2 | Architecture (dims, depth, heads, op order, shapes) | `model_commitment` | RFC-0001 | C2 |
| B3 | Weights | `model_commitment` (Merkle root) | RFC-0001, RFC-0003 | C2 |
| B4 | Biases | `model_commitment` | RFC-0001 | C2 |
| B5 | Quantization scales | `quantization_commitment` | RFC-0001, RFC-0002 | C3 |
| B6 | Rounding mode (exactly one active) | `quantization_commitment` | RFC-0002 | C4 |
| B7 | Clamp policy (per-tensor) | `quantization_commitment` | RFC-0002 | C4 |
| B8 | Overflow policy (`reject`) | `quantization_commitment` | RFC-0002 | C4 |
| B9 | Lookup tables (activation/range) | `quantization_commitment` (`activation_tables_commitment`) | RFC-0003, RFC-0006 | C3 |
| B10 | Activation/normalization approximations (per-module) | `quantization_commitment` | RFC-0006 | C3 |
| B11 | Planner config (horizon, action_block, candidate count, tie-break) | `planner_config_commitment` | RFC-0009 | C7 |
| B12 | Inputs: latent history, goal latent, candidate actions (public OR commitment) | public input fields / their commitments | RFC-0014 | C1, C9 |
| B13 | Outputs: claimed trajectory / cost / selected index | `claimed_output_commitment`, `selected_index`, `selected_cost` | RFC-0009, RFC-0014 | C1, C9 |
| B14 | Security parameters (`lambda`, FRI queries, PoW bits) | public input + transcript | RFC-0014 | C12 |
| B15 | Serialization/schema version | public-input digest; `manifest_version`, `artifact_version` | RFC-0014, RFC-0016 | C9 |

The `PublicInput` type carries B1, B11, B12, B13, B14 directly; B2–B10 and B15 are
folded into `model_commitment` / `quantization_commitment` / `planner_config_commitment`.
See the canonical `PublicInput` in
[docs/spec/03-data-model.md#public-input](03-data-model.md#public-input). Any
field not listed here is, by INV-SEC-04, prover-controlled and must not influence the
relation; if it does, that is a soundness bug to be filed under `area:security`.

INV-SEC-01: A proof is valid only for its declared `relation_id`. `relation_id`s are
immutable (`pwm.lewm.<statement>.v<N>`); any semantic change mints a new id. A P0 proof
submitted as P1, or a proof reused under a different id, is rejected (RFC-0000).

INV-SEC-02: The verifier recomputes the public-input digest from the canonical bytes and
absorbs it into the Fiat-Shamir channel before any challenge is squeezed; a stale proof
cannot be rebound to new inputs (RFC-0014).

INV-SEC-03: Prover and verifier share the exact Fiat-Shamir channel ordering; the
verifier re-derives every challenge in canonical order. Off-channel re-seeding (challenge
grinding) does not produce an accepting proof (RFC-0014).

### Binding failure modes and the verifier response

| Failure mode | Trigger | Verifier response |
|--------------|---------|-------------------|
| Unknown statement | `relation_id` not in registry | Reject: `UnsupportedRelation` |
| Statement mismatch | `statement_type` inconsistent with `relation_id` | Reject: `StatementMismatch` |
| Commitment mismatch | recomputed digest != bound commitment | Reject: `PublicInputDigestMismatch` |
| Model/quant/planner commitment changed | any of B2–B11 altered | Reject: `CommitmentMismatch` |
| Security param below floor | `lambda`/queries/PoW under configured minimum | Reject: `InsufficientSecurityParameters` |
| Out-of-range integer | range-check witness fails | Reject: `RangeCheckFailure` |
| Tensor misroute | TensorCell multiset/permutation argument fails | Reject: `MemoryConsistencyFailure` |
| Planner subset | argmin diff missing/negative, or candidate count != S | Reject: `PlannerSoundnessFailure` |
| STARK invalid | FRI/PCS/constraint check fails | Reject: `ProofVerificationFailure` |

The concrete `VerifyError` enum and its variants are owned by
[docs/spec/04-error-model.md#verifier-rejections](04-error-model.md#verifier-rejections);
the names above are the security-relevant subset and MUST map onto that enum.

---

<a id="privacy-and-zk"></a>

## Privacy and the ZK boundary
INV-SEC-13: **V0 is a succinct VALIDITY proof, not a zero-knowledge proof.** No
document, log line, README, commit message, release note, or marketing surface may
label V0 (or any artifact it produces) "zero-knowledge", "ZK", "private", or
"confidential". A validity STARK proves the relation holds; it does not by itself hide
the witness, and the V0 trace commitments are not hiding. This is a hard rule
(RFC-0000), not a default.

Committing a value (e.g. weights as `private_committed` rather than `public`) reduces
the bytes on the wire but does NOT make the system zero-knowledge. A commitment binds;
hiding requires the additional properties below.

### What ZK would additionally demand (the hiding requirements)

A future ZK mode (V1+, gated on a hiding audit) MUST establish all of the following.
Each is a property V0 does not provide:

| # | Hiding requirement | What it prevents leaking | V0 status |
|---|--------------------|--------------------------|-----------|
| H1 | Weights are not public inputs; only a binding commitment is public | weight values | Allowed in V0 (`Witness.model_weights = None`, committed-only) but insufficient alone |
| H2 | Trace commitments are hiding or appropriately masked (e.g. randomized/blinded openings) | witness columns via the commitment scheme and openings | NOT provided in V0 |
| H3 | Lookup arguments do not leak private table contents | private activation/weight tables via multiplicities or accessed entries | NOT provided in V0 |
| H4 | Claimed sums and public outputs do not leak unintended witness values | private latents/costs via the output encoding | NOT provided in V0 |
| H5 | Serialization does not expose witness data | anything in the on-disk `ProofArtifact` | Partially: `Witness` is never serialized into the artifact, but proof bytes are not analyzed for residual leakage in V0 |

The hiding audit (a named milestone gate, not a checkbox) must demonstrate H1–H5 against
this specific custom NN relation. Stwo-cairo discusses a production proof stack and a
conjectured soundness target for Cairo execution proofs, but that is soundness, not
privacy, and privacy for this relation must be established separately. Until that audit
exists and is referenced, the canonical terminology is:

```text
V0:  succinct validity proof
V1+: optionally zero-knowledge validity proof, AFTER the hiding audit
```

OPEN QUESTION (owner: `area:security` maintainers; resolution path: RFC-0000 amendment +
a future ZK RFC at the V1 milestone): the exact hiding mechanism (masked trace
commitments vs. a ZK-friendly PCS variant in the vendored Stwo) is not decided for V0
and need not be; V0 must simply never claim the property it does not have.

---

## Secrets handling
<a id="secrets-handling"></a>

The proving host handles checkpoints and quantized weights that may be confidential even
though V0 is not ZK. "Confidential" here is an operational property of the host, distinct
from the cryptographic hiding of [#privacy-and-zk](#privacy-and-zk): keeping a secret off
disk-logs does not make the proof zero-knowledge, and vice versa.

INV-SEC-14: Secrets (checkpoints, raw weights, private inputs) are handled under these
rules:

| Rule | Statement | Cross-reference |
|------|-----------|-----------------|
| At rest | Checkpoints/weights live only where the operator places them; the project writes no copy outside the declared export output and the `ProofArtifact` bundle | RFC-0001, RFC-0016 |
| Logs | No secret (weight value, private latent, raw checkpoint path content, witness column) appears in any log, span, or metric. Private witnesses and weights are redacted | [docs/spec/05-observability.md#redaction](05-observability.md#redaction) |
| Artifacts | No secret appears in the `ProofArtifact` unless it is an intended public output. `Witness` is never serialized into the artifact; `claimed_outputs` carries only declared public outputs | [docs/spec/02-public-api.md#artifact-formats](02-public-api.md#artifact-formats) |
| Commitments-only | When weights are `private_committed`, only the Merkle root is public; the weight tensors stay prover-side (subject to the H1 caveat above) | RFC-0001 |

Failure modes for secrets handling:

| Failure mode | System response |
|--------------|-----------------|
| A weight/witness value would be logged | The redaction layer ([docs/spec/05-observability.md#redaction](05-observability.md#redaction)) drops or masks it before emission; emitting an unredacted secret is a `area:security` defect |
| A secret would be written to the artifact | The artifact serializer excludes `Witness` by construction; only `PublicInput` and declared outputs are written |
| A private input is mistakenly marked public in the manifest | The statement changes (the value becomes a public input), the commitment changes, and the operator must re-export; the verifier is unaffected but the secret is exposed — this is an operator error caught by the export review, not by the verifier |

---

<a id="disclosure"></a>

## Vulnerability disclosure
A cryptographic soundness system requires a private reporting channel and an explicit
embargo. The repository MUST ship a top-level `SECURITY.md` (referenced from
[docs/spec/09-release-and-versioning.md#license](09-release-and-versioning.md#license))
with the following content. This section is the normative source for that file.

INV-SEC-15: No public claim of soundness for any `relation_id` may be made before the
security review gate for that statement has passed. The security review is the final
layer of the test stack (RFC-0013, layer 10); a `relation_id` may exist and be exercised
in CI before review, but the README/release notes must not assert "sound" or "proven"
for it until the review is recorded.

### SECURITY.md required contents

```text
1. Scope
   - Soundness of the declared relations (the relation_id registry).
   - The vendored Stwo verifier path as pinned (RFC-0015).
   - Explicitly OUT of scope: float/PyTorch equivalence, physical truth,
     ZK/privacy in V0 (see docs/spec/06-security.md#privacy-and-zk),
     and prover-host availability/DoS.

2. Private reporting channel
   - A dedicated security contact (security advisory via the repository's
     private vulnerability reporting, plus a published security email).
   - Reporters MUST NOT open public issues for soundness bugs.

3. Acknowledgement window
   - Initial acknowledgement within 3 business days of a valid report.
   - A triage assessment (severity, affected relation_ids, affected
     revisions) within 10 business days.

4. Embargo policy
   - Coordinated disclosure. Default embargo: 90 days from acknowledgement,
     or until a fix ships and downstream verifiers can update, whichever is
     sooner. Extension only by mutual agreement with the reporter.
   - Soundness breaks (a false statement made acceptable) are treated as
     the highest severity and may trigger an immediate relation_id
     revocation advisory.

5. Severity guidance
   - Critical: any path that lets a prover get a false statement accepted
     (a soundness break) for a shipped relation_id.
   - High: a binding gap (an item in docs/spec/06-security.md#binding-requirements
     that is not actually enforced) not yet exploitable end-to-end.
   - Medium: a determinism/reproducibility divergence (RFC-0016) that does
     not yet break soundness.
   - Privacy reports against V0 are acknowledged but classified per the V0
     scope: V0 is not ZK, so witness exposure is not a V0 vulnerability
     (see docs/spec/06-security.md#privacy-and-zk).

6. Security review gate
   - No public soundness claim for a relation_id before its security review
     passes (INV-SEC-15, RFC-0013 layer 10).
   - On a confirmed soundness break: mint a new relation_id for the fixed
     relation, publish an advisory, and document the revocation in
     docs/spec/09-release-and-versioning.md#relation-versioning.
```

### Disclosure failure modes

| Failure mode | System response |
|--------------|-----------------|
| Soundness bug reported publicly before fix | Maintainers privately confirm, prepare a fix and a new `relation_id`, then coordinate disclosure; the affected `relation_id` is flagged in an advisory |
| Confirmed soundness break in a shipped `relation_id` | Mint a new id for the corrected relation (ids are immutable, RFC-0000); publish a revocation advisory; old proofs remain valid only for the now-distrusted relation, which relying parties must stop accepting |
| Public "proven"/"sound" claim made before the review gate | Treated as a disclosure-policy violation (INV-SEC-15); the claim is retracted and the README/release notes corrected |

---

## Invariant index

| Invariant | Statement (abbreviated) | Enforcing RFC(s) |
|-----------|-------------------------|------------------|
| INV-SEC-01 | A proof is valid only for its declared, immutable `relation_id` | RFC-0000, RFC-0014 |
| INV-SEC-02 | Verifier recomputes and binds the public-input digest before squeezing challenges | RFC-0014 |
| INV-SEC-03 | Prover/verifier share exact Fiat-Shamir channel order; challenges re-derived | RFC-0014 |
| INV-SEC-04 | Anything not bound by a public commitment is prover-mutable and adversary-controlled | RFC-0000, RFC-0001 |
| INV-SEC-05 | Verifier never runs float/PyTorch/Python/the exporter; it verifies the relation only | RFC-0015, RFC-0016 |
| INV-SEC-06 | Every commitment-affecting stage is deterministic; no float in the proof path | RFC-0001, RFC-0002, RFC-0016 |
| INV-SEC-07 | Every approximation is committed in the manifest and enforced by the AIR (per-module) | RFC-0006 |
| INV-SEC-08 | Every integer-interpreted value is range-checked; overflow rejected, never wrapped | RFC-0002, RFC-0003 |
| INV-SEC-09 | Tensor reads match committed writes via the TensorCell multiset/permutation argument | RFC-0004 |
| INV-SEC-10 | Fixed-candidate planning proves ALL rollouts, ALL costs, argmin, and tie-break | RFC-0009 |
| INV-SEC-11 | A sound CEM proof requires the full sampling/top-k/update/recurrence set (P3, future) | RFC-0010 |
| INV-SEC-12 | Security parameters are bound and checked against a configured floor | RFC-0014 |
| INV-SEC-13 | V0 is a validity proof, never labeled zero-knowledge/ZK/private | RFC-0000 |
| INV-SEC-14 | Secrets are not logged, not written to artifacts unless intended public outputs | RFC-0001, RFC-0016 |
| INV-SEC-15 | No public soundness claim for a `relation_id` before its security review passes | RFC-0013 |
