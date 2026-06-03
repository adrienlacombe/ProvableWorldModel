# Release and Versioning

Status: Normative. Role: defines the versioning, compatibility, deprecation, changelog, toolchain, and license rules that govern every public artifact of ProvableWorldModel. This document is binding on `pwm-core`, `pwm-export`, `pwm-air`, `pwm-circuits`, `pwm-prover`, `pwm-verifier`, the on-disk manifest and proof-artifact formats, and the `relation_id` namespace.

ProvableWorldModel proves an exact arithmetic relation: deterministic, quantized inference of the LeWorldModel predictor and fixed-candidate planner, arithmetized as a custom Circle-STARK AIR over M31 and discharged by the vendored Stwo prover. Because soundness depends on the prover and verifier agreeing bit-for-bit on every rule (quantization, rounding, approximation, planner tie-break, serialization, transcript ordering), versioning here is not a release-management convenience. It is a soundness mechanism. A version number that does not change when semantics change is a soundness defect, equivalent to a broken constraint. This document treats it that way.

The single overriding principle:

> Semantics are identified by content. The same `relation_id` and the same set of commitments MUST mean the same computation forever. If any rule that affects which witnesses verify changes, a new identifier is minted; the old identifier is never silently repointed.

This document specifies five interacting version axes and the rules binding them:

| Axis | Identifier | Owner | Governs | Anchor |
|---|---|---|---|---|
| Crate / API version | semver `MAJOR.MINOR.PATCH` per crate | each crate | Rust source/API compatibility | [#semver](#semver) |
| Release version | `V0`, `V1`, `V2`, `V3` (proof-statement tiers) | project | which proof statements ship | [#semver](#semver) |
| Relation version | `pwm.lewm.<statement>.v<N>` | `pwm-core` | exact proven arithmetic semantics | [#relation-versioning](#relation-versioning) |
| Manifest schema version | `manifest_version: u32` | `pwm-export` | manifest field layout/meaning | [#relation-versioning](#relation-versioning) |
| Artifact schema version | `artifact_version: u32` | `pwm-prover`/`pwm-verifier` | on-disk proof bundle layout | [#relation-versioning](#relation-versioning) |

These axes are deliberately separable: an API patch (e.g. a faster prover) MUST NOT change any `relation_id`; a relation bump MAY occur without an incompatible API change. The rules below make those independences explicit.

---

## semver

### Scope of the public API

Semver applies to the **public Rust API surface** enumerated in [docs/spec/02-public-api.md#rust-public-api](02-public-api.md#rust-public-api), the **CLI** in [docs/spec/02-public-api.md#cli](02-public-api.md#cli), and the **on-disk artifact formats** in [docs/spec/02-public-api.md#artifact-formats](02-public-api.md#artifact-formats). The stability tiers (public / unstable / internal) are defined by [docs/spec/02-public-api.md#stability-policy](02-public-api.md#stability-policy); this section defines what a version bump *means* for each tier. Anything marked `#[doc(hidden)]`, gated behind an `unstable-*` Cargo feature, or located in a module documented as internal is NOT covered by semver and may change in any release.

### Versioning model

The workspace is versioned with **independent per-crate semver**, not a single lockstep version, because the crates have different stability profiles: `pwm-core` and `pwm-verifier` are consumed by external integrators and must be conservative, whereas `pwm-air` and `pwm-circuits` are implementation-internal and churn faster. The release *tiers* `V0`–`V1`–`V2`–`V3` (see contract §4; [docs/spec/00-overview.md#scope-and-statement-tiers](00-overview.md#scope-and-statement-tiers)) are marketing-free milestone names that map onto crate versions; they do not replace semver.

Initial public crate versions:

| Crate | Initial version | Semver discipline | Notes |
|---|---|---|---|
| `pwm-core` | `0.1.0` | strict | field/fixed-point types, manifest types, `relation_id`, transcript, serialization |
| `pwm-export` | `0.1.0` | strict on manifest output; tooling-internal otherwise | Python+Rust pipeline; manifest bytes are the contract, not the Rust API |
| `pwm-air` | `0.1.0` | internal (`unstable`) until V1 | AIR components; constraint layout is not a stable API |
| `pwm-circuits` | `0.1.0` | internal (`unstable`) until V1 | low-level gate adapters to vendored stwo-circuits |
| `pwm-prover` | `0.1.0` | strict on CLI + `ProofArtifact`; library API `unstable` | |
| `pwm-verifier` | `0.1.0` | strict | `verify()` is the most stability-critical surface in the project |

Pre-`1.0.0` (the V0 line) follows Cargo's `0.x` convention: a bump of the `MINOR` digit (`0.1 -> 0.2`) MAY contain breaking changes; the `PATCH` digit (`0.1.0 -> 0.1.1`) MUST NOT. The first `1.0.0` of `pwm-verifier` and `pwm-core` is cut at the v1.0 milestone ([docs/spec/00-overview.md#scope-and-statement-tiers](00-overview.md#scope-and-statement-tiers)) when the P2 statement ships; after that, full semver applies.

### What each bump means

The bump rules below are *normative*. The deciding question for every change is: **could a caller that compiled/verified successfully against the old version now fail to compile, or behave observably differently, without changing their own code?** If yes, it is breaking.

| Change | Bump | Rationale |
|---|---|---|
| Remove/rename a public item; change a public signature; remove an enum variant or struct field; tighten a precondition | MAJOR (`1.x`) / MINOR (`0.x`) | Source-breaking for downstream code. |
| Make a previously accepted `ProofArtifact` fail verification (other than via a documented soundness fix) | MAJOR / MINOR | Verification-breaking; treat with the same severity as API removal. See the carve-out below. |
| Add a public item; add a `#[non_exhaustive]` enum variant; add an optional CLI flag; widen an accepted input set | MINOR (`x.Y`) / MINOR (`0.x`) | Backward compatible additions. |
| Bug fix with no API or accepted-input change; docs; performance; internal refactor; dependency patch with no surface change | PATCH (`x.y.Z`) | No observable contract change. |
| New `relation_id` value added (new statement or new relation version) | MINOR at minimum | New capability; never reuses an identifier. |
| Bump of a vendored Stwo / stwo-circuits revision that changes proof bytes or accepted artifacts | see [#relation-versioning](#relation-versioning) and RFC-0015 | A change in the proof system that alters which artifacts verify is a relation-level event, not a mere dependency patch. |

**Soundness-fix carve-out.** A change that causes a *previously accepted but actually unsound* proof to be rejected is, formally, a breaking change for any party relying on that (broken) acceptance. We still ship it, but never silently:

- It MUST be released as a `MAJOR` bump (or `MINOR` in the `0.x` line) of the affected crate, never a `PATCH`.
- It MUST appear under `Security` in the changelog ([#changelog](#changelog)) and trigger the disclosure process in [docs/spec/06-security.md#disclosure](06-security.md#disclosure).
- If the unsoundness stems from under-binding (the relation accepted witnesses it should have rejected), it mints a **new `relation_id`** per [#relation-versioning](#relation-versioning); the old relation is marked withdrawn (see [#deprecation](#deprecation)). We never "fix in place" the meaning of an existing `relation_id`.

INV-REL-01 (semver-honest API): No `PATCH` release changes any public signature, removes any public item, narrows any accepted input, or changes which `ProofArtifact`s `verify()` accepts. A change with any of those effects is `MINOR` (in `0.x`) or `MAJOR` (in `>=1.0`) by definition.

INV-REL-02 (verification monotonicity within a patch line): For a fixed `relation_id`, `artifact_version`, and verifier `MAJOR.MINOR`, every `PATCH` within that line accepts exactly the same set of artifacts. Verification acceptance is a function of `(relation_id, artifact_version, verifier MAJOR.MINOR)` and nothing finer.

#### Failure modes (semver)

| Failure mode | System response |
|---|---|
| A `PATCH` release would change an accepted-artifact set | CI release gate ([#changelog](#changelog), RFC-0016) blocks the release; the change must be re-tagged as MINOR/MAJOR. The gate runs the differential acceptance suite ([docs/spec/07-testing-strategy.md#differential-tests](07-testing-strategy.md#differential-tests)) against the prior tag. |
| Caller pins `pwm-verifier = "0.1"` (caret) and a `0.2` with a soundness fix exists | Caller continues on `0.1`; the soundness advisory ([docs/spec/06-security.md#disclosure](06-security.md#disclosure)) is published via `cargo audit`-readable RUSTSEC metadata so the stale pin surfaces a warning. |
| Public item accidentally exposed (leaked internal type) | Treated as if it were never public; removable in a `MINOR`/`PATCH` only if `cargo-semver-checks` confirms no semver break, otherwise MAJOR. Documented under `Fixed`. |

Tooling: `cargo-semver-checks` runs in CI on every PR touching a public crate and on every release tag; a detected breaking change without a corresponding MAJOR/MINOR bump fails the build. This makes INV-REL-01 machine-enforced, not aspirational.

---

## relation-versioning

This is the core soundness section. It governs the three content-identified version axes: `relation_id` (the proven computation), `manifest_version` (the description of the model being proven), and `artifact_version` (the on-disk proof bundle).

### `relation_id` immutability

A `relation_id` is a 32-byte field in `PublicInput` ([docs/spec/03-data-model.md#public-input](03-data-model.md#public-input), canonical signature in contract §6.3). Its human-readable form is `pwm.lewm.<statement>.v<N>` (contract §4), e.g. `pwm.lewm.predictor_step.v1`, `pwm.lewm.rollout.v1`, `pwm.lewm.fixed_candidate_planning.v1`. The on-wire 32-byte value is the canonical hash of the relation descriptor (the full enumerated semantics below), computed by `pwm-core` with the canonical serialization and transcript hash function fixed in [docs/spec/03-data-model.md#schema-versioning](03-data-model.md#schema-versioning) and RFC-0014. The string is a label; the bytes are the binding identity.

INV-REL-03 (relation immutability): A `relation_id` value (both string and the 32-byte hash) denotes exactly one computation for all time. Its semantics are frozen at the moment the relation reaches `Accepted` status. The mapping `relation_id -> semantics` is append-only: identifiers are added, never mutated, never reused, never removed from the recognized set while any deployed verifier might encounter them.

INV-REL-04 (relation-bound validity): A proof is valid only for the `relation_id` declared in its `PublicInput`. A verifier MUST reject an artifact whose `relation_id` it does not recognize, and MUST NOT attempt to "upgrade" or reinterpret a proof under a different relation. (This is the binding rule locked by RFC-0000; see [docs/spec/06-security.md#binding-requirements](06-security.md#binding-requirements).)

#### What a `relation_id` encodes (the change-triggers)

The relation descriptor that the `relation_id` hashes over MUST cover every rule that can change which `(PublicInput, Witness)` pairs satisfy the relation. Any change to any item below mints a **new** `relation_id` with an incremented `v<N>` (and, where the change is structural rather than versioned semantics, possibly a new `<statement>` segment). This list is exhaustive for V0; additions to it are themselves a relation-versioning event.

| Category | Examples of a change that mints a new `relation_id` |
|---|---|
| Architecture | predictor depth/heads/`dim_head`/`mlp_dim`/`latent_dim`/`history_size`; op set; which components are in scope (e.g. adding the pixel encoder = P4, a new statement entirely); operator ordering in the exported graph |
| Quantization | int8 vs int16 activations; per-tensor vs per-channel scales; accumulator width or limb decomposition strategy; scale table semantics; signed encoding (`centered_mod_p`) |
| Rounding | switching `nearest_ties_to_even` <-> `truncate_toward_zero`; any change to tie-handling or shift semantics (contract §3) |
| Approximation | GELU table contents/domain; softmax approximation polynomial/table; LayerNorm/AdaLN reciprocal-sqrt method or domain; activation-table commitment scheme |
| Planner rule | cost function (MSE definition, accumulation order); argmin tie-break (smallest-index rule and its `cost_s - selected_cost - 1 >= 0` enforcement, contract §3); for P3, CEM sampling/clipping/top-k/update recurrence |
| Serialization | canonical byte layout of `PublicInput`/manifest; field ordering; endianness; hash function used for commitments or the public-input digest (RFC-0014) |
| Transcript | Fiat-Shamir channel ordering, domain separators, challenge derivation (RFC-0014) |
| Field/security | field parameters; extension-field use for challenges; FRI/PCS parameters that change soundness or which proofs verify |

Note the distinction from per-instance commitments. The *values* of `model_commitment`, `quantization_commitment`, and `planner_config_commitment` (in `PublicInput`) bind a *specific* model/quantization/planner instance and change per model. The `relation_id` binds the *rules of the game* — the algorithm those commitments are fed into. A different trained checkpoint reuses the same `relation_id` with a different `model_commitment`; a different rounding mode requires a different `relation_id` even for the same checkpoint. This separation is why the activation/normalization *approximation method* lives in the relation (it changes the algorithm) while the *weights* live behind `model_commitment` (they parameterize a fixed algorithm). See [docs/spec/06-security.md#binding-requirements](06-security.md#binding-requirements).

#### Why this is a soundness rule, not a convenience

If a prover could change rounding from ties-to-even to truncation while keeping `pwm.lewm.rollout.v1`, a verifier accepting `v1` proofs would accept two mutually inconsistent computations under one identity. A relying party reading "this proof is `pwm.lewm.rollout.v1`" would have no way to know which arithmetic actually ran. Mutable relation semantics is therefore indistinguishable, from the verifier's standpoint, from a prover free to choose favorable arithmetic — the exact attack the binding requirements ([docs/spec/06-security.md#soundness-requirements](06-security.md#soundness-requirements)) exist to prevent. Immutability of `relation_id` is what makes "this is a `pwm.lewm.fixed_candidate_planning.v1` proof" a statement with a fixed, checkable meaning.

#### Minting and registry

- The authoritative list of recognized `relation_id`s, their string form, their 32-byte hash, their frozen descriptor, and their status (`Accepted` / `Deprecated` / `Withdrawn`) lives in a registry table in `pwm-core` (compiled-in) and is mirrored in this corpus's RFC for the relevant statement (RFC-0007 for predictor, RFC-0008 for rollout, RFC-0009 for fixed-candidate planning).
- A verifier recognizes exactly the relations compiled into its `pwm-core` dependency. Adding a relation is a `MINOR` source change ([#semver](#semver)) plus a registry append.
- The `v<N>` counter is per-`<statement>` and monotonic. `pwm.lewm.rollout.v1` and `pwm.lewm.rollout.v2` are distinct, unrelated relations as far as soundness is concerned; v2 does not "supersede" v1 cryptographically, it merely is the recommended one (see [#deprecation](#deprecation)).

INV-REL-05 (no hash collision across descriptors): The relation-descriptor serialization is injective up to semantic equivalence: two relations with different change-triggers (table above) MUST serialize to different bytes and therefore different 32-byte ids. The export pipeline ([docs/spec/03-data-model.md#schema-versioning](03-data-model.md#schema-versioning)) includes a test that perturbing any descriptor field changes the id (a differential test, [docs/spec/07-testing-strategy.md#differential-tests](07-testing-strategy.md#differential-tests)).

#### Failure modes (relation versioning)

| Failure mode | System response |
|---|---|
| Prover emits a proof whose actual arithmetic does not match the declared `relation_id` | The proof fails to verify: the AIR encodes the declared relation's rules; a witness from different arithmetic does not satisfy the constraints. The verifier returns a `VerifyError` ([docs/spec/04-error-model.md#verifier-rejections](04-error-model.md#verifier-rejections)). |
| Two relations accidentally serialize to the same id (collision) | Caught by the INV-REL-05 differential test in CI before release; release gate (RFC-0016) blocks. |
| Verifier receives an unknown `relation_id` | Reject with the unrecognized-relation `VerifyError` variant; do not attempt fallback ([docs/spec/04-error-model.md#verifier-rejections](04-error-model.md#verifier-rejections)). |
| A maintainer proposes editing an `Accepted` relation's descriptor | Disallowed by INV-REL-03. The registry is append-only; the change must be a new `v<N>`. CI rejects diffs that mutate an existing frozen descriptor row. |

### Manifest schema versioning (`manifest_version`)

The model manifest (contract §6.6, canonical schema in [docs/spec/03-data-model.md#model-manifest](03-data-model.md#model-manifest)) carries `manifest_version: u32`. It versions the *layout and field semantics of the manifest document*, independent of the `relation_id` the manifest declares. The two are linked but distinct: a manifest declares a `relation_id`; `manifest_version` governs how to *parse and interpret* that manifest's fields.

`manifest_version` bump rules:

| Change to the manifest schema | `manifest_version` |
|---|---|
| Add an optional key with a defined default that does not change the meaning of existing manifests | no bump (additive within a version is allowed only when fully defaulting; otherwise bump) |
| Add a required key; rename/remove a key; change a key's type or units; change the canonical-JSON-hash computation | bump (`v1 -> v2`) |
| Change how an existing key feeds the relation (this also changes `relation_id`) | bump `manifest_version` AND mint a new `relation_id` |

The export pipeline (`pwm-export`, RFC-0001) always writes the current `manifest_version`. There is no manifest *upgrader*: a manifest is consumed at the version it was written. A consumer (`pwm-prover`, `pwm-verifier`) that does not implement a given `manifest_version` rejects the manifest rather than guessing ([docs/spec/04-error-model.md#failure-modes](04-error-model.md#failure-modes)).

INV-REL-06 (manifest self-describing): Every manifest declares its `manifest_version` as the first field, and every consumer dispatches parsing on that value before reading any other field. A manifest with an unknown or missing `manifest_version` is rejected unparsed.

### Artifact schema versioning (`artifact_version`) and the compatibility matrix

The on-disk proof bundle is `ProofArtifact` (contract §6.5) with an `artifact_version: u32` as its first field. It versions the *envelope*: how the `PublicInput`, the opaque Stwo `Proof` bytes, and the optional `claimed_outputs` are framed on disk. It is distinct from the proof's internal soundness, which is governed by `relation_id`. An `artifact_version` bump means "the bytes are laid out differently"; a `relation_id` bump means "the computation is different."

`artifact_version` bump rules mirror the manifest rules: additive-with-default is allowed within a version; any change to framing, field set, ordering, the `Proof` serialization, or the public-input digest computation bumps it.

INV-REL-07 (artifact self-describing): `artifact_version` is the first field of every serialized `ProofArtifact`. A verifier reads it before any other field and dispatches accordingly; an unknown `artifact_version` is rejected before the proof is touched ([docs/spec/04-error-model.md#verifier-rejections](04-error-model.md#verifier-rejections)).

#### Verifier / artifact compatibility matrix

A verifier of a given `MAJOR.MINOR` accepts a fixed, declared set of `artifact_version`s. The matrix is the contract; it is published per release and is part of the changelog ([#changelog](#changelog)).

| Verifier crate version | Accepts `artifact_version` | Recognizes `manifest_version` | Recognizes `relation_id`s | Notes |
|---|---|---|---|---|
| `pwm-verifier 0.1.x` (v0.2 milestone) | `1` | `1` | `pwm.lewm.predictor_step.v1`, `pwm.lewm.rollout.v1` | P0/P1 line |
| `pwm-verifier 0.2.x` (v0.2 milestone) | `1` | `1` | adds nothing beyond `0.1.x` set unless a relation is bumped | bug/perf only if no acceptance change |
| `pwm-verifier 1.0.x` (v1.0 milestone) | `1` (and `2` if the bundle is reframed for P2) | `1` (and later) | adds `pwm.lewm.fixed_candidate_planning.v1` | P2 headline statement |

Rules governing the matrix:

- INV-REL-08 (no silent format break): A verifier MUST reject any `artifact_version` not in its accepted set; it MUST NOT attempt best-effort parsing. Dropping support for a previously accepted `artifact_version` is a `MAJOR` change ([#semver](#semver)) and requires a deprecation window ([#deprecation](#deprecation)).
- A newer verifier MAY accept older `artifact_version`s (backward read compatibility) but is never *required* to; when it does, the matrix row lists every accepted version explicitly.
- A `Proof` produced for a given `(relation_id, artifact_version)` is verifiable by exactly the verifier versions whose matrix row lists both. There is no implicit forward compatibility: an old verifier never accepts a newer `artifact_version` it does not list.

#### Failure modes (manifest / artifact versioning)

| Failure mode | System response |
|---|---|
| Verifier reads unknown `artifact_version` | Reject pre-parse with a versioned-envelope `VerifyError`; emit the observed and supported versions in the error ([docs/spec/04-error-model.md#verifier-rejections](04-error-model.md#verifier-rejections)). |
| Prover writes an artifact with a `relation_id` the targeted verifier lacks | Verification fails at relation lookup (INV-REL-04). The prover CLI SHOULD warn at emit time if the configured target verifier version is known and lacks the relation (RFC-0016). |
| Manifest `manifest_version` newer than `pwm-export`/consumer supports | Reject with the unsupported-manifest error ([docs/spec/04-error-model.md#failure-modes](04-error-model.md#failure-modes)); no downgrade attempted. |
| Artifact carries `claimed_outputs` whose framing changed without an `artifact_version` bump | Caught by the round-trip serialization golden test ([docs/spec/07-testing-strategy.md#golden-vectors](07-testing-strategy.md#golden-vectors)); release gate blocks. |

---

## deprecation

Deprecation applies to four kinds of entity, each with its own retirement path because the soundness stakes differ. Relations and artifact formats are the high-stakes cases; API items and CLI flags are conventional software deprecations.

### Lifecycle states

| State | Meaning | Verifier behavior |
|---|---|---|
| `Accepted` | Current, recommended. | Accepts. |
| `Deprecated` | Still recognized; a successor exists; new proofs SHOULD NOT target it. | Still accepts; SHOULD emit an observability warning ([docs/spec/05-observability.md#logging](05-observability.md#logging)). |
| `Withdrawn` | Recognition removed because verifying it is unsound or actively harmful. | Rejects. |
| `Superseded by RFC-MMMM` | The *decision* (RFC) was replaced; carried in the RFC header, not the verifier. | n/a (documentation state) |

### Relation deprecation

A `relation_id` moves `Accepted -> Deprecated` when a successor `v<N+1>` is published (e.g. a better-bounded LayerNorm approximation lands as `pwm.lewm.rollout.v2`). Deprecation does NOT remove recognition: a `Deprecated` relation still verifies, because proofs already issued against it remain valid statements about the computation they ran — immutability (INV-REL-03) is the whole point. The deprecation is advisory: tooling steers new proofs to the successor.

A `relation_id` moves to `Withdrawn` only when continued recognition is itself a hazard — specifically when the relation is *unsound* (it accepts witnesses it should reject, i.e. under-binding). Withdrawal is a `MAJOR`/`MINOR` verifier bump, a `Security` changelog entry, and a disclosure ([docs/spec/06-security.md#disclosure](06-security.md#disclosure)). Withdrawal is the *only* mechanism that ever causes a verifier to stop accepting a relation it once accepted; routine deprecation never does.

INV-REL-09 (deprecation preserves recognition): Moving a relation to `Deprecated` MUST NOT change which proofs the verifier accepts. Only `Withdrawn` removes recognition, and only for unsoundness, and only via a non-`PATCH` release with a `Security` changelog entry.

### Deprecation window and warning mechanism

| Entity | Minimum window before recognition/support is removed | Warning mechanism |
|---|---|---|
| Public Rust API item | one `MINOR` release (`0.x`) / two `MINOR` releases (`>=1.0`) at `Deprecated` before removal in a `MAJOR` | `#[deprecated(since = "x.y.z", note = "use ... ; see CHANGELOG")]` |
| CLI flag/command | one `MINOR` release with a runtime warning to stderr | stderr `warning:` line + changelog `Deprecated` |
| `manifest_version` / `artifact_version` | one `MINOR` release recognized as `Deprecated`; removal is `MAJOR` | consumer logs a deprecation warning on read ([docs/spec/05-observability.md#logging](05-observability.md#logging)) |
| `relation_id` (deprecation) | indefinite — never removed by deprecation; only `Withdrawn` removes it | verifier logs `relation deprecated, prefer <successor>` |
| `relation_id` (withdrawal, unsound) | immediate on the fixing release; no soft window, because the window would be a window of accepted unsound proofs | `Security` advisory + disclosure |

The asymmetry is deliberate: ordinary deprecation grants a grace window because the only cost is staleness; unsoundness withdrawal grants none because every day of grace is a day of a verifier vouching for a broken statement.

### Retiring an RFC

RFCs are the decision records (contract §7, §8). When a decision is replaced, the old RFC's header `Status:` field changes to `Superseded by RFC-MMMM` (this is the exact form fixed by the RFC template, contract §8) and the new RFC's `Motivation` references it. The superseded RFC's file is retained (history is append-only); it is never deleted. A superseded RFC that locked a `relation_id` does not retroactively un-mint that relation — the relation's lifecycle is governed by the registry, not by the RFC's status. Example: if RFC-0008 (rollout AIR) is superseded by a hypothetical RFC-0017 that bumps to `pwm.lewm.rollout.v2`, RFC-0008 becomes `Superseded by RFC-0017`, `rollout.v1` becomes `Deprecated` (not `Withdrawn`, since it is still sound), and `rollout.v2` is the new `Accepted` relation.

#### Failure modes (deprecation)

| Failure mode | System response |
|---|---|
| Item removed without serving its deprecation window | Release gate (RFC-0016) checks that every removed public item was `#[deprecated]` for the required window in prior tags; missing window blocks the release. |
| A `Deprecated` relation is withdrawn without a `Security` entry | Changelog lint (RFC-0016) blocks: any verifier-acceptance-narrowing change requires a `Security` or `Removed` section entry. |
| Downstream still targets a `Deprecated` relation at withdrawal time | Their last-issued proofs were valid statements about a sound computation if the relation was sound; if it was unsound, the disclosure ([docs/spec/06-security.md#disclosure](06-security.md#disclosure)) instructs re-proving under the successor. The verifier now rejects, surfacing the staleness. |

---

## changelog

### Format

The repository maintains a single top-level `CHANGELOG.md` per public crate (or one workspace `CHANGELOG.md` with per-crate sections), in **Keep a Changelog** style with **semver-conformant** version headers. Every entry is human-written, links to the PR/issue, and is filed under exactly the standard section that matches its effect:

| Section | Use for |
|---|---|
| `Added` | new public API, new CLI flag, new recognized `relation_id`, new `manifest_version`/`artifact_version` |
| `Changed` | behavior change in existing functionality that is NOT a soundness narrowing (e.g. perf, default tweaks, additive widening) |
| `Deprecated` | items entering a deprecation window; relations moved to `Deprecated` |
| `Removed` | items whose deprecation window expired; dropped `artifact_version`/`manifest_version` support |
| `Fixed` | bug fixes with no acceptance-set change |
| `Security` | soundness fixes, relation withdrawals, any change that narrows verifier acceptance, dependency advisories; every such entry links to the disclosure ([docs/spec/06-security.md#disclosure](06-security.md#disclosure)) |

Discipline rules:

- INV-REL-10 (changelog completeness): No release tag is cut without a matching `CHANGELOG.md` section whose version equals the tag. The release gate (RFC-0016) blocks a tag with a missing or `Unreleased`-only changelog.
- Every change that alters which artifacts verify (a relation bump, a relation withdrawal, an `artifact_version` change, a soundness fix) MUST appear under `Security` or `Removed`/`Changed` as appropriate AND name the affected `relation_id`(s) and `artifact_version`(s) and the resulting compatibility-matrix change ([#relation-versioning](#relation-versioning)).
- The `Unreleased` section accumulates entries on `main`; cutting a release renames it to the version and date and opens a fresh `Unreleased`.
- Relation and version events are first-class changelog content. A new `relation_id` is an `Added` line that states the string id, the statement tier (P0–P4), and the milestone.

### Release process (reference)

The end-to-end release mechanics — version-bump verification, `cargo-semver-checks`, the differential acceptance suite, changelog lint, tagging, artifact publication, and the bit-for-bit reproducibility contract for emitted proof inputs/outputs — are specified in [docs/rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md](../rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md). This document defines *what the version numbers and changelog must say*; RFC-0016 defines *how a release is produced*. The CI release gates referenced throughout this file (semver check, differential acceptance, changelog lint, deprecation-window check, compatibility-matrix publication) are the concrete gates enumerated in that RFC and in [docs/spec/07-testing-strategy.md#ci-gates](07-testing-strategy.md#ci-gates).

#### Failure modes (changelog)

| Failure mode | System response |
|---|---|
| Release tag without changelog section | Release gate blocks (INV-REL-10). |
| Acceptance-narrowing change filed only under `Fixed` | Changelog lint blocks: such a change must be `Security` (or `Removed`/`Changed`) and must name the relation/artifact versions. |
| New `relation_id` shipped without an `Added` entry | Lint cross-checks the `pwm-core` relation registry against the changelog; an unlisted new relation blocks the release. |

---

## msrv

### Rust edition and MSRV

| Property | Value | Reason |
|---|---|---|
| Workspace Rust edition (first-party crates) | 2021 | Stable, broadly available; sufficient for the first-party crates. |
| Vendored `stwo-circuits` edition | 2024 | Verified against upstream at 2026-06-03: `stwo-circuits` (v0.1.0) uses edition 2024. Vendored under `third_party/` per RFC-0015; the build must use a toolchain new enough for edition 2024. |
| MSRV (minimum supported Rust version) | the lowest stable Rust that compiles the entire workspace including vendored `third_party/` at the pinned revisions | Edition 2024 requires Rust >= 1.85; the project's MSRV is therefore at least that, pinned to the exact minimum recorded in `rust-toolchain.toml`. |
| MSRV policy | MSRV may only rise on a `MINOR` (`0.x`) / `MAJOR` (`>=1.0`) release; never on a `PATCH`. Every MSRV bump is a `Changed` changelog entry. | A toolchain-floor rise can break downstream builds and is therefore a non-`PATCH` event. |
| Toolchain pinning | `rust-toolchain.toml` at the workspace root pins the exact channel used for reproducible builds; CI also tests against the declared MSRV and against current stable. | The reproducibility contract (RFC-0016) requires a known toolchain. |

OPEN QUESTION (owner: `area:core` maintainer; resolution: RFC-0015, before the v0.1 design-freeze tag): record the exact pinned Rust version in `rust-toolchain.toml` and the exact minimum edition-2024-capable Rust version as the project MSRV, derived from the pinned `third_party/stwo` and `third_party/stwo-circuits` revisions. Until that pin is recorded, the MSRV is stated as "the minimum stable Rust that builds the pinned vendored tree, which is >= 1.85 due to edition 2024." Do not assert a single exact MSRV number before the vendoring revision is frozen.

INV-REL-11 (MSRV monotonic within a patch line): The MSRV is constant across all `PATCH` releases of a given `MINOR` line. CI runs the build and verifier test suite on the declared MSRV toolchain on every PR; a feature requiring a newer toolchain fails until the MSRV is formally bumped in a non-`PATCH` release with a changelog entry.

### Python floor (export pipeline only)

`pwm-export` includes a Python reference-inference and quantization pipeline ([docs/spec/01-architecture.md#crate-layout](01-architecture.md#crate-layout); RFC-0001). Python is confined to export and golden-vector generation; it never runs in the prover hot path and **never** in the verifier (contract §3).

| Property | Value | Reason |
|---|---|---|
| Python floor | `>= 3.10` | Modern typing (`X | Y` unions, `match`) used by the export tooling; broadly available. |
| Dependency surface | PyTorch (consume LeWorldModel checkpoint + config) and ONNX (graph interchange); both pinned in the export environment lockfile. | The pipeline consumes the MIT-licensed `le-wm` checkpoint/config; it does not vendor `le-wm` source (verified against upstream at 2026-06-03). |
| Floor policy | the Python floor may only rise on a `MINOR`/`MAJOR` release; it is irrelevant to verifier compatibility (no Python in the verifier) | Verifier integrators never need Python; the floor affects only the export workflow. |

INV-REL-12 (Python isolation): No verifier or runtime-verification artifact depends on Python. The Python floor governs the export/test tooling exclusively. A change to the Python floor never changes any `relation_id`, `artifact_version`, or verifier acceptance set.

#### Failure modes (toolchain)

| Failure mode | System response |
|---|---|
| Feature needing newer Rust merged without MSRV bump | MSRV CI job fails on the declared-MSRV toolchain; PR blocked. |
| Vendored revision bump raises the edition-2024 floor | RFC-0015 records the new pinned revision and the new MSRV; the change ships as `Changed` in a non-`PATCH` release. |
| Export tooling uses a Python feature below the floor or above an untested ceiling | Export test matrix (floor + current) catches it; CI blocks ([docs/spec/07-testing-strategy.md#ci-gates](07-testing-strategy.md#ci-gates)). |

---

## license

### Project license

ProvableWorldModel is licensed under **Apache-2.0** (decided in contract §3; the repository `LICENSE` file is the canonical Apache-2.0 text). Apache-2.0 is chosen for two concrete reasons, not by default:

1. It is compatible with the vendored Stwo and stwo-circuits, which are themselves Apache-2.0 (verified against upstream at 2026-06-03), so vendoring under `third_party/` introduces no license conflict.
2. Its explicit patent grant matters for a cryptographic proving system, where the implemented arithmetization and protocol may touch patentable methods; the grant gives downstream verifiers and integrators defensive certainty that a permissive but patent-silent license (e.g. MIT alone) would not.

### SPDX headers

INV-REL-13 (SPDX on every first-party source file): Every first-party source file (`.rs`, `.py`, and build/CI scripts that carry meaningful logic) begins with the SPDX identifier line:

```text
// SPDX-License-Identifier: Apache-2.0
```

(`#` comment form for Python and shell). A CI lint (a header-check gate, [docs/spec/07-testing-strategy.md#ci-gates](07-testing-strategy.md#ci-gates)) fails the build if any first-party source file under `crates/` or the export pipeline lacks the SPDX line. Vendored files under `third_party/` retain their upstream headers unmodified; the SPDX lint excludes `third_party/`.

### Third-party license inventory and NOTICE

- The authoritative third-party license inventory — every vendored component (`stwo`, `stwo-circuits`), its upstream source URL, its pinned revision, and its license terms — is maintained in [docs/rfcs/RFC-0015-third-party-vendoring-and-pinning.md](../rfcs/RFC-0015-third-party-vendoring-and-pinning.md). This document does not duplicate that inventory; it requires that the inventory exist and be kept current as a release gate.
- Each vendored component retains its own `third_party/<component>/LICENSE` and copyright/NOTICE files unmodified. Apache-2.0 (Stwo, stwo-circuits) requires preserving such notices; this is mandatory, not optional.
- The repository root `NOTICE` file is maintained per the Apache-2.0 attribution requirement. It already references RFC-0015 as the authoritative vendored-component list (verified: the repo `NOTICE` points to `docs/rfcs/RFC-0015-third-party-vendoring-and-pinning.md`). The `NOTICE` MUST be updated whenever a vendored component is added, removed, or its license changes.
- The LeWorldModel reference (`le-wm`) is MIT-licensed (verified against upstream at 2026-06-03). The export pipeline consumes its checkpoint and config; it does not vendor `le-wm` source. MIT attribution for the consumed checkpoint/config is recorded in the export documentation (RFC-0001) and acknowledged in `NOTICE` if any `le-wm`-derived material is redistributed; if no `le-wm` material is redistributed, only the export docs carry the attribution.

INV-REL-14 (license inventory currency): No release is cut if the vendored tree under `third_party/` contains a component absent from the RFC-0015 inventory, or if a vendored revision differs from the inventory's recorded pin. The release gate (RFC-0016) cross-checks `third_party/<component>/REVISION` files against the inventory and the `NOTICE`.

### Security policy reference

The vulnerability disclosure process — how to report a soundness or implementation vulnerability, the embargo and coordinated-disclosure timeline, and how a fix is shipped (relation withdrawal, `Security` changelog, advisory) — is specified in [docs/spec/06-security.md#disclosure](06-security.md#disclosure). The release-side obligations triggered by a disclosed vulnerability (the soundness-fix carve-out in [#semver](#semver), relation withdrawal in [#deprecation](#deprecation), and the `Security` changelog section in [#changelog](#changelog)) are defined here; the disclosure mechanics live in the security document.

#### Failure modes (license)

| Failure mode | System response |
|---|---|
| First-party source file missing SPDX header | SPDX lint gate fails; PR blocked. |
| Vendored component added without inventory/NOTICE update | Inventory cross-check gate fails (INV-REL-14); release blocked. |
| Vendored upstream changes its license on a revision bump | RFC-0015 inventory update is required and reviewed by `area:security` and `area:core` before the bump merges; an incompatible license blocks the bump. |
| `LICENSE` text drifts from canonical Apache-2.0 | A checksum gate on `LICENSE` against the canonical Apache-2.0 text fails the build. |

---

## Invariant index

| Invariant | Statement (summary) | Anchor |
|---|---|---|
| INV-REL-01 | No `PATCH` changes the public API, narrows inputs, or changes accepted artifacts. | [#semver](#semver) |
| INV-REL-02 | Verification acceptance is a function of `(relation_id, artifact_version, verifier MAJOR.MINOR)`; constant within a patch line. | [#semver](#semver) |
| INV-REL-03 | A `relation_id` denotes one computation forever; the map is append-only. | [#relation-versioning](#relation-versioning) |
| INV-REL-04 | A proof is valid only for its declared `relation_id`; unknown relations are rejected. | [#relation-versioning](#relation-versioning) |
| INV-REL-05 | Distinct relation semantics serialize to distinct 32-byte ids (no collisions). | [#relation-versioning](#relation-versioning) |
| INV-REL-06 | Every manifest is self-describing via `manifest_version`, dispatched before any other field. | [#relation-versioning](#relation-versioning) |
| INV-REL-07 | Every `ProofArtifact` is self-describing via `artifact_version`, read before the proof. | [#relation-versioning](#relation-versioning) |
| INV-REL-08 | A verifier rejects any `artifact_version` not in its accepted set; no best-effort parsing. | [#relation-versioning](#relation-versioning) |
| INV-REL-09 | Deprecation never changes acceptance; only `Withdrawn` (for unsoundness) removes recognition. | [#deprecation](#deprecation) |
| INV-REL-10 | No release tag without a matching changelog section. | [#changelog](#changelog) |
| INV-REL-11 | MSRV is constant across a patch line; rises only on non-`PATCH` releases. | [#msrv](#msrv) |
| INV-REL-12 | No verifier/runtime artifact depends on Python; the Python floor is export-only. | [#msrv](#msrv) |
| INV-REL-13 | Every first-party source file carries the Apache-2.0 SPDX header. | [#license](#license) |
| INV-REL-14 | The `third_party/` tree matches the RFC-0015 inventory and `NOTICE` exactly. | [#license](#license) |

## References

- Founding analysis: [docs/feasibility-study.md](../feasibility-study.md) §9.1 (binding requirements), §10 (public-input/verifier flow).
- [docs/spec/00-overview.md#scope-and-statement-tiers](00-overview.md#scope-and-statement-tiers) — release tiers V0–V3, statements P0–P4.
- [docs/spec/02-public-api.md#stability-policy](02-public-api.md#stability-policy), [#rust-public-api](02-public-api.md#rust-public-api), [#cli](02-public-api.md#cli), [#artifact-formats](02-public-api.md#artifact-formats).
- [docs/spec/03-data-model.md#model-manifest](03-data-model.md#model-manifest), [#public-input](03-data-model.md#public-input), [#schema-versioning](03-data-model.md#schema-versioning).
- [docs/spec/04-error-model.md#verifier-rejections](04-error-model.md#verifier-rejections), [#failure-modes](04-error-model.md#failure-modes).
- [docs/spec/05-observability.md#logging](05-observability.md#logging).
- [docs/spec/06-security.md#binding-requirements](06-security.md#binding-requirements), [#soundness-requirements](06-security.md#soundness-requirements), [#disclosure](06-security.md#disclosure).
- [docs/spec/07-testing-strategy.md#differential-tests](07-testing-strategy.md#differential-tests), [#golden-vectors](07-testing-strategy.md#golden-vectors), [#ci-gates](07-testing-strategy.md#ci-gates).
- [docs/rfcs/RFC-0015-third-party-vendoring-and-pinning.md](../rfcs/RFC-0015-third-party-vendoring-and-pinning.md) — vendoring, pinning, license inventory.
- [docs/rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md](../rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md) — CLI, artifact bundle, reproducibility, release process.
- RFC-0000 (statement taxonomy / relation binding), RFC-0001 (manifest + export), RFC-0007/0008/0009 (predictor/rollout/planner relations and their `relation_id`s).
