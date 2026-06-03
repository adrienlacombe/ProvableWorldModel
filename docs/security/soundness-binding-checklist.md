<!-- SPDX-License-Identifier: Apache-2.0 -->
# Soundness and Binding Checklist

Status: Normative-derived (traceability artifact). Role: maps every soundness
obligation and binding requirement in
[docs/spec/06-security.md](../spec/06-security.md) to the concrete artifact that
enforces it — an RFC, an implementation issue, a verifier rejection, and a test
class. It does not introduce new requirements; it is the auditable index that the
security-review gate consumes.

## How this checklist is used

- **Input to the review gate.** The P2 relation security review gate (issue #79,
  [docs/spec/07-testing-strategy.md#ci-gates](../spec/07-testing-strategy.md#ci-gates)
  layer 10) signs off against this checklist: every row must resolve to a passing
  enforcing test before a public soundness claim is made for a `relation_id`
  (INV-SEC-15).
- **Completeness rule.** Every binding item (B1–B15) and every range-safety
  column maps to *either* an enforcing test that exists *or* the issue that will
  carry that test. A row with neither is a gap to be filed under `area:security`.
- **Dual-test obligation.** Per
  [docs/spec/07-testing-strategy.md#negative-tests](../spec/07-testing-strategy.md#negative-tests)
  (INV-TEST-15), no requirement ships without both an accepting test and a
  rejecting test proving the verifier catches its violation. The "Reject test"
  column names the `VerifyError` the rejecting test must observe.

Legend for the **Enforced** column:

- `CI` — enforced now by a merge gate on the current tree (#73).
- `pending #N` — enforcement lands with implementation issue #N; the row is open
  until that issue's accept+reject tests pass.

`VerifyError` variants are the canonical ones from
[docs/spec/04-error-model.md#verifier-rejections](../spec/04-error-model.md). The
security-doc rejection names (e.g. `CommitmentMismatch`, `RangeCheckFailure`) are
the security-relevant subset and map onto that enum as noted.

---

## 1. Binding requirements (INV-SEC-04, B1–B15)

Operational rule (INV-SEC-04): a value affects the statement **iff** a public
commitment binds it. Each binding item from
[docs/spec/06-security.md#binding-requirements](../spec/06-security.md#binding-requirements)
maps to the commitment, the enforcing issue(s), and the rejection observed when
the bound value is altered.

| # | Bound item | Commitment | Enforcing RFC | Enforcing issue(s) | Reject test (`VerifyError`) | Enforced |
|---|------------|------------|---------------|--------------------|------------------------------|----------|
| B1 | `relation_id` (statement identity) | public input + transcript | RFC-0000, RFC-0014 | #33, #34, #63 | `RelationIdUnsupported` | pending #63 |
| B2 | Architecture (dims, depth, heads, op order, shapes) | `model_commitment` | RFC-0001 | #35, #36, #38, #63 | `ModelCommitmentMismatch` / `PublicInputDigestMismatch` | pending #63 |
| B3 | Weights | `model_commitment` (Merkle root) | RFC-0001, RFC-0003 | #35, #38, #47 | `ModelCommitmentMismatch` | pending #63 |
| B4 | Biases | `model_commitment` | RFC-0001 | #35, #38, #47 | `ModelCommitmentMismatch` | pending #63 |
| B5 | Quantization scales | `quantization_commitment` | RFC-0001, RFC-0002 | #29, #30, #35, #36 | `QuantizationCommitmentMismatch` | pending #63 |
| B6 | Rounding mode (exactly one active) | `quantization_commitment` | RFC-0002 | #29, #35 | `QuantizationCommitmentMismatch`; on a forged remainder `RangeViolation` | pending #63 |
| B7 | Clamp policy (per-tensor) | `quantization_commitment` | RFC-0002 | #29, #35 | `QuantizationCommitmentMismatch` | pending #63 |
| B8 | Overflow policy (`reject`) | `quantization_commitment` | RFC-0002 | #28, #31, #35 | `AccumulatorOverflow` / `RangeViolation` | pending #63 |
| B9 | Lookup tables (activation/range) | `quantization_commitment` (`activation_tables_commitment`) | RFC-0003, RFC-0006 | #35, #36, #44 | `QuantizationCommitmentMismatch`; out-of-table `OutOfDomainLookup` | pending #63 |
| B10 | Activation/normalization approximations (per-module) | `quantization_commitment` | RFC-0006 | #35, #36, #44, #50, #51, #52, #53 | `QuantizationCommitmentMismatch` / `OutOfDomainLookup` | pending #63 |
| B11 | Planner config (horizon, action_block, candidate count, tie-break) | `planner_config_commitment` | RFC-0009 | #35, #69, #71, #72 | `PlannerConfigMismatch` | pending #71 |
| B12 | Inputs (latent history, goal latent, candidate actions) | public-input fields / their commitments | RFC-0014 | #32, #33, #63 | `PublicInputDigestMismatch` / `ShapeMismatch` | pending #63 |
| B13 | Outputs (claimed trajectory / cost / selected index) | `claimed_output_commitment`, `selected_index`, `selected_cost` | RFC-0009, RFC-0014 | #63, #68, #69, #71 | `SelectedIndexOutOfRange` / `SelectedCostMismatch` | pending #71 |
| B14 | Security parameters (`lambda`, FRI queries, PoW bits) | public input + transcript | RFC-0014 | #34, #63 | param-floor rejection (owned by #63 / 04-error-model; see note) | pending #63 |
| B15 | Serialization / schema version | public-input digest; `manifest_version`, `artifact_version` | RFC-0014, RFC-0016 | #32, #33, #63 | `ArtifactVersionUnsupported` / `PublicInputDigestMismatch` | pending #63 |

Binding invariants that are not per-item:

| Invariant | Statement | Enforcing issue(s) | Reject test |
|-----------|-----------|--------------------|-------------|
| INV-SEC-01 | A proof is valid only for its declared, immutable `relation_id` | #34, #63 | `RelationIdUnsupported` on a reused/mismatched id |
| INV-SEC-02 | Verifier recomputes and binds the public-input digest before squeezing challenges | #33, #34, #63 | `PublicInputDigestMismatch` |
| INV-SEC-03 | Prover/verifier share exact Fiat-Shamir channel order; challenges re-derived | #34, #62, #63 | `ProofInvalid { stage: "transcript" }` |

> Note on B14: the security doc names the rejection `InsufficientSecurityParameters`.
> The canonical `VerifyError` enum
> ([docs/spec/04-error-model.md](../spec/04-error-model.md#verifier-rejections))
> does not yet carry that variant; the parameter-floor check and its exact variant
> are owned by the verifier-core issue (#63) and the error-model document. This
> naming reconciliation is tracked as a checklist follow-up under `area:security`
> and must be closed before #79 signs off B14.

---

## 2. Range-safety requirements (INV-SEC-08)

Every value interpreted as an integer is range-checked to its declared `[lo, hi]`;
`overflow_policy = reject` (never wrap). Range checks are LogUp lookups against
`u8`/`i8`/`u16`/bounded-limb tables (#42, #43); accumulators beyond the safe signed
M31 interval are limb-decomposed and each limb range-checked (#31). Each
value-bearing column below maps to the component that produces it and the
rejection observed when the column escapes its bound.

| Column (integer-interpreted) | Produced by issue | Range relation | Reject test (`VerifyError`) | Enforced |
|------------------------------|-------------------|----------------|------------------------------|----------|
| input latents | #62 (trace builder) | #43 | `RangeViolation` | pending #43 |
| actions / action embeddings | #54, #62 | #43 | `RangeViolation` | pending #43 |
| weights, biases | #38, #47 | #43 | `RangeViolation` | pending #43 |
| products (xᵢ·wᵢⱼ) | #47, #48 | #43 | `RangeViolation` | pending #47 |
| accumulators and limbs | #31, #47, #48 | #43, #31 | `AccumulatorOverflow` | pending #47 |
| requantization quotients | #49 | #43 | `RangeViolation` | pending #49 |
| requantization remainders (0 ≤ rem < 2^shift) | #49 | #43 | `RangeViolation` (F3) | pending #49 |
| activation lookup outputs | #44, #50 | #44 | `OutOfDomainLookup` | pending #44 |
| attention scores | #56 | #43 | `RangeViolation` | pending #56 |
| softmax/probability numerators and denominators | #51 | #43, #44 | `RangeViolation` / `OutOfDomainLookup` | pending #51 |
| layernorm/adaln mean, centered values, inv-std operands | #52, #53 | #43, #44 | `RangeViolation` / `OutOfDomainLookup` | pending #52 |
| per-step predicted latents | #58, #60 | #43 | `RangeViolation` | pending #58 |
| cost differences (zⱼ − gⱼ) and squares | #68 | #43 | `RangeViolation` | pending #68 |
| argmin diffs (cost_s − selected_cost ≥ 0) | #69 | #43 | `RangeViolation` / `TieBreakViolation` | pending #69 |

Comparison rule (INV-SEC-08): there is no native field `≤`. `a ≤ b` is accepted
only via a witnessed nonnegative difference `d = b − a` with `0 ≤ d ≤ D_max`
range-checked; the strict tie-break uses `cost_s − selected_cost − 1 ≥ 0`. A
comparison that wraps the field is a rejected witness (F4 →
`RangeViolation` / `TieBreakViolation`). Enforced by #69 with reject tests in #78.

Coverage invariant (INV-ARCH-11 / INV-SEC-08): the wiring/coverage check in #42
asserts that *every* integer-interpreted column is wired to a range relation; an
unranged column is itself a soundness defect, independent of any single component.

---

## 3. Determinism requirements (INV-SEC-06)

Every stage that contributes to a bound commitment or a proof input is
deterministic; floating point appears nowhere in the proving or verification path.
Enforcement is the parity gate (Python fixed-point ≡ Rust fixed-point ≡ AIR,
bit-for-bit) and the reproducibility digest.

| Stage | Obligation | Enforcing RFC | Enforcing issue(s) | Enforcing test |
|-------|------------|---------------|--------------------|----------------|
| Model export | byte-identical manifest + weights | RFC-0001 | #37, #38 | re-export determinism test (#38) |
| Quantization | fixed scales, fixed rounding, no float bits | RFC-0001, RFC-0002 | #29, #37 | parity gate (#40) |
| Rounding | exactly one active mode per manifest | RFC-0002 | #29 | golden vectors (#39), parity (#40) |
| Clamping | explicit per-tensor clamp, deterministic saturation | RFC-0002 | #29, #30 | parity (#40) |
| Activation/normalization approximation | committed table/poly, same input → same output | RFC-0006 | #44, #50–#53 | golden vectors (#39), accept/reject (#78) |
| Rollout windowing | fixed window selection | RFC-0008 | #60 | accept test (#61) |
| Argmin tie-break | smallest index attaining the minimum wins | RFC-0009 | #69 | reject test `TieBreakViolation` (#78) |
| Public-input serialization | canonical, versioned, byte-stable | RFC-0014 | #32, #33 | serialization round-trip + digest test (#32) |
| Fiat-Shamir transcript | identical absorb/squeeze order both sides | RFC-0014 | #34 | transcript differential test (#34) |
| End-to-end reproducibility | identical inputs → identical proof inputs/outputs | RFC-0016 | #65 | determinism digest gate (#65) |

A nondeterministic stage is a soundness hole (if reference and AIR can diverge, the
AIR is not proving the reference); the parity gate is the machine-checked
enforcement of INV-SEC-06.

---

## 4. Approximation requirements (INV-SEC-07)

Every approximate primitive is fully committed in the manifest (under
`quantization_commitment`) and fully enforced by the AIR. The activation choice is
**per-module, not uniform**: the predictor FFN/MLP uses GELU; AdaLN modulation and
the action `Embedder` use SiLU. A manifest declaring one global activation is wrong
and fails the export parity test.

| Approximation field | Bound by | Enforcing issue(s) | Reject test |
|---------------------|----------|--------------------|-------------|
| `id` / `primitive` / `module` (per-module activation) | `quantization_commitment` | #36, #50, #53 | manifest-parity mismatch (#40); wrong-module reject (#78) |
| `domain` (bound integer input domain) | `quantization_commitment` | #44 | `OutOfDomainLookup` for out-of-domain input (#78) |
| `representation` + `table_commitment` / `poly_commitment` | `quantization_commitment` | #44, #50, #51, #52 | `QuantizationCommitmentMismatch` on a swapped table (#78) |
| `rounding` | `quantization_commitment` | #29 | parity mismatch (#40) |
| `test_vectors_commitment` (golden (x,y)) | `quantization_commitment` | #39 | golden-immutability gate (#39, #78) |

Rejected approaches (off-circuit softmax probabilities, unconstrained
reciprocal/inverse-sqrt, floating-point helper values, implicit PyTorch semantics)
are enumerated in RFC-0006; each carries a rejecting test under #51/#52/#78.

---

## 5. Planner soundness requirements (INV-SEC-10)

For fixed-candidate planning (P2 / V0), proving only the selected candidate is
unsound. The proof establishes, in-circuit, for the full candidate set
`s ∈ 0..S-1`:

| Clause | Statement | Enforcing issue(s) | Reject test (`VerifyError`) |
|--------|-----------|--------------------|------------------------------|
| all rollouts | `trajectory_s == Rollout_Q(latent_history, candidate_actions_s)` for every `s` | #61, #70 | `ProofInvalid` (rollout relation) / `MemoryConsistencyFailure` |
| all costs | `cost_s == MSE_Q(final_latent_s, goal_latent)` for every `s` | #68, #70 | `RangeViolation` / `ProofInvalid` |
| selection | `selected_cost == cost_{selected_index}` | #69, #71 | `SelectedCostMismatch` |
| minimality | `selected_cost ≤ cost_s` (range-checked diff) for every `s` | #69 | `RangeViolation` |
| tie-break | `cost_s − selected_cost − 1 ≥ 0` for every `s < selected_index` | #69 | `TieBreakViolation` |
| candidate count | proven candidate count `== S` (no omitted candidate) | #70, #71 | `PlannerSoundnessFailure` (count ≠ S) |

If any candidate rollout or cost is omitted, the prover can score favorable
candidates off-circuit and the planning claim is hollow (capability C7). The
argmin witness carries one nonnegative diff per candidate
(`ArgminWitness.diffs`); a missing or negative diff is a rejected witness.
Enforced by #69/#70/#71 with the negative suite in #78; reviewed at #79.

INV-SEC-11 (CEM / P3) is out of V0 scope and is not part of this checklist's
enforced set; it is recorded for the future proof only.

---

## 6. Trust-boundary and disclosure invariants

These are enforced by architecture gates and policy rather than per-witness
rejections.

| Invariant | Statement | Enforcement | Enforced |
|-----------|-----------|-------------|----------|
| INV-SEC-05 / INV-ARCH-02 | Verifier never runs float/PyTorch/Python/the exporter | `cargo tree` allowlist + bare-metal `no_std` verifier build | CI (#73) |
| INV-ARCH-01 | `pwm-core` carries no proving dependency; `no_std`-clean | `no_std` core build + dep allowlist | CI (#73) |
| INV-SEC-13 | V0 is a validity proof, never labeled ZK/private | docs/README/release-note review; terminology check | policy; review gate #79 |
| INV-SEC-14 | Secrets are not logged / not written to artifacts unless intended outputs | redaction guard + artifact serializer excludes `Witness` | pending #66 (redaction), #64 (artifact) |
| INV-SEC-15 | No public soundness claim before the relation's review gate passes | this checklist + review sign-off | policy; review gate #79 |

---

## 7. Coverage assertion

- **Every binding item B1–B15** maps to at least one enforcing issue and a named
  rejection (Section 1). Open items: all rejections land with the verifier core
  (#63) and the planner proof (#71); B14's variant name is the one tracked
  follow-up.
- **Every range-safety column** in INV-SEC-08 maps to a producing issue and the
  range relation (#42/#43/#31) (Section 2). The #42 coverage check forbids any
  unranged integer column.
- **Every determinism stage** maps to a parity/reproducibility test (Section 3).
- **Every approximate primitive** is committed and AIR-enforced (Section 4).
- **The full planner candidate set** is proven (Section 5); partial proofs are
  rejected with `PlannerSoundnessFailure`.

No binding or range-safety requirement is left without an enforcing test or issue.
This satisfies the acceptance criteria of issue #80. The security-review gate (#79)
consumes this checklist; INV-SEC-15 forbids a public soundness claim for any
`relation_id` until its rows resolve to passing tests.
