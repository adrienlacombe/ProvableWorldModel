# RFC-0015: Third-party vendoring, dependency pinning, and audit boundary

- Status: Accepted
- Authors: ProvableWorldModel maintainers
- Created: 2026-06-03
- Target milestone: v0.1

## Summary

ProvableWorldModel's soundness rests entirely on the correctness of the Circle-STARK
proving substrate it builds on: the Stwo prover/verifier stack and the
`stwo-circuits` low-level gate library. This RFC makes the dependency posture
permanent: both projects are **vendored** under `third_party/` at **pinned
revisions** recorded in `third_party/<comp>/REVISION`, the workspace **does not
depend on the upstream canonical crates directly**, every local modification is
recorded as a patch and every revision bump triggers a re-audit, and a single
SPDX dependency/license inventory at `third_party/INVENTORY.md` is the
authoritative list of everything in our trusted computing base. This freezes the
proving substrate against silent upstream changes (which would silently change
the soundness argument), makes the audit boundary explicit, and keeps the
Apache-2.0 license obligations satisfiable. The decision is locked: no part of the
proving or verification path may pull a Stwo or `stwo-circuits` crate from
crates.io or a git URL; it must resolve to the vendored copy.

## Motivation

The founding analysis states the posture directly: "stwo and stwo circuits should
be used directly with a copy of latest workable version, in a third_party folder
in our repo. We don't want to depend on the upstream canonical repos. we copy them
and modify as we need for this POC."
(`docs/feasibility-study.md` §0, also §3.2 and the risk register §13 entry
"`stwo-circuits` maturity ... Pin revisions, audit, use direct AIR for hot
paths"). The decided-choices list in the authoring contract carries the same
ruling: "`stwo` and `stwo-circuits` are vendored under `third_party/` at pinned
revisions and modified as needed for this project. The project does NOT depend on
the upstream canonical crates directly. (See RFC-0015.)"

Concrete scenarios this RFC must handle:

- **Silent soundness drift.** A floating dependency on Stwo would let `cargo
  update` move the constraint framework, FRI parameters, channel hashing, or
  LogUp implementation under our feet. Because our AIR's soundness is argued
  against a *specific* Stwo commit, any unaudited move invalidates the argument
  while leaving the build green. Vendoring at a pinned revision is the only way to
  make the proving substrate part of the reproducible, reviewable artifact.
- **Local modifications are mandatory, not incidental.** The substrate is a
  research-grade prover plus a circuit library that is "not a complete NN proving
  system out of the box" (`docs/feasibility-study.md` §3.2). We will modify it:
  expose internal APIs, adjust visibility, add component hooks, and possibly patch
  bugs. A normal Cargo dependency cannot carry our patches; a vendored tree can.
- **Audit boundary clarity.** `docs/spec/06-security.md#trust-boundaries` must be
  able to state exactly what code is trusted. "Whatever crates.io served today" is
  not auditable. A pinned, vendored, patch-tracked tree is.
- **License compliance.** Both projects are Apache-2.0
  (`docs/feasibility-study.md` §9b, verified against upstream at 2026-06-03);
  the ProvableWorldModel license is Apache-2.0
  (`docs/spec/09-release-and-versioning.md#license`). Apache-2.0 §4 requires we
  retain copyright/`NOTICE` and state modifications. Vendoring forces us to do
  this concretely rather than hand-waving transitive obligations.

This RFC is the dependency-management complement to
`docs/spec/06-security.md#soundness-requirements` (item "Stwo/security
parameters" in the binding list, `docs/feasibility-study.md` §9.1) and to the
reproducibility contract in
`docs/rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md`.

## Goals

- Vendor `stwo` and `stwo-circuits` under `third_party/` at pinned revisions, each
  revision recorded in a machine-readable `third_party/<comp>/REVISION` file.
- Lock the **no-direct-upstream-dependency** policy: no workspace crate may
  declare a `crates.io` or git dependency on any Stwo or `stwo-circuits` package;
  all such dependencies are `path = "../../third_party/..."`.
- Define the **modification workflow**: every local change is an entry in
  `third_party/<comp>/PATCHES/` with a rationale, and the upstream tree is never
  edited in place without a recorded patch.
- Define the **revision-bump workflow**: bumping a vendored revision is a
  reviewed, gated change that re-runs the substrate-conformance test suite and
  requires a fresh `area:security` sign-off recorded in the audit log.
- Produce and maintain `third_party/INVENTORY.md`: an SPDX-style inventory of
  every vendored component and every transitive third-party crate in the trusted
  computing base, with package name, version/revision, SPDX license id, source
  URL, and trust tier.
- Make the vendored revisions and patch hashes part of the build provenance so a
  proof artifact can be tied to the exact substrate that produced it.
- Use the verified, corrected substrate facts (crate lists, gate set) from the
  authoring contract §9b, not the founding analysis's pre-correction versions.

## Non-Goals

- This RFC does not vendor LeWorldModel (`le-wm`). `le-wm` is MIT-licensed and is
  consumed only as a checkpoint + config by the export pipeline; it is not a build
  dependency and its source is not copied. Attribution for it lives in
  `docs/rfcs/RFC-0001-model-manifest-and-export-pipeline.md` and the inventory's
  "consumed inputs" appendix. (`docs/feasibility-study.md` §9b export note.)
- This RFC does not vendor `stwo-cairo`. It is reference prior art for recursion
  (`docs/rfcs/RFC-0012-recursive-aggregated-verification.md`), not a V0 build
  dependency.
- This RFC does not define the Fiat-Shamir channel ordering or the public-input
  digest; those are locked in
  `docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md`. This RFC only
  guarantees that the *implementation* of those primitives is pinned.
- This RFC does not specify the AIR component design that consumes the substrate;
  see `docs/spec/01-architecture.md#air-strategy` and RFC-0003 through RFC-0009.
- This RFC does not establish a general policy for non-substrate Rust crates
  (e.g. `serde`, `clap`); those follow ordinary semver pinning via
  `Cargo.lock` and `docs/spec/09-release-and-versioning.md#msrv`. It governs the
  *trusted* substrate and the inventory that lists everything trusted.

## Proposed Design

### Vendored components and what they contain (verified against upstream at 2026-06-03)

Two upstream projects are vendored. The crate lists below are the corrected,
verified lists from the authoring contract §9b, not the founding analysis's
earlier counts.

**Component `stwo`** — `github.com/starkware-libs/stwo`, Apache-2.0, workspace
version `2.2.0` at vendoring time. Vendored package set:

| Package | Role | Notes |
| --- | --- | --- |
| `stwo` | Circle-STARK prover/verifier core | Contains the M31 field **module** at `src/core/fields/` (`m31.rs` with `P = 2^31 - 1`), extensions CM31/QM31, FRI + PCS, and Fiat-Shamir channels (`src/core/channel/`, Blake2s/Keccak256/Poseidon252 backends). The verifier path is `no_std` (`#![cfg_attr(not(feature="std"), no_std)]`). |
| `stwo-air-utils` | AIR trace/column helpers | Used by `pwm-air` for trace layout. |
| `stwo-air-utils-derive` | Proc-macros for the above | Build-time only. |
| `stwo-constraint-framework` | Constraint framework + LogUp (`constraint-framework/src/logup.rs`) | The substrate for every `pwm-air` component and all range/lookup relations (RFC-0003). |
| `std-shims` | `no_std` shims for the verifier | Required to keep the verifier path `no_std`-compatible. |

The `examples` crate from the upstream workspace is **not** vendored into the
build graph; if copied for reference it is excluded from the workspace and marked
`tier: reference` in the inventory. The `ensure-verifier-no_std` test crate is
vendored only as part of the substrate-conformance suite (see Testing Strategy),
not as a build dependency.

Correction enforced: the fields are a **module** inside the single `stwo` crate,
not separate "field crates". Documentation in this corpus says "field module".

**Component `stwo-circuits`** — `github.com/starkware-libs/stwo-circuits`,
Apache-2.0, version `0.1.0`, Rust edition 2024. The workspace has **10 crates**:
`cairo_verifier`, `circuits`, `circuit_verifier`, `circuit_cairo_serialize`,
`circuit_common`, `circuit_multiverifier`, `circuit_serialize`, `circuit_prover`,
`stark_verifier`, `stark_verifier_examples`. ProvableWorldModel uses this layer
for hashing, public-input binding, small arithmetic gadgets, and recursive-verifier
plumbing experiments (`docs/feasibility-study.md` §3.2 recommended split), not for
dense matmul hot paths.

Correction enforced: the first-class gate set is the 10 fields of `struct
Circuit`: `Add, Sub, Mul, PointwiseMul, Eq, TripleXor, M31ToU32, BlakeGGate,
Permutation, Output`. There is **no** `range` / `bit-extraction` gate.
`extract_bits()` is a helper built from `sub`/`mul`/`assert_bits` constraints, and
range checking is enforced by those constraints, not a dedicated gate. `BlakeGGate`
is a Blake2s G-function gate, not a full-hash gate. Any place in this corpus that
lists the gate set (notably `docs/spec/01-architecture.md#air-strategy` and
RFC-0003) uses this corrected set. The founding analysis's gate list
(`docs/feasibility-study.md` §3.2 Strategy B, which lists a "range/extract_bits"
gate) is superseded by this RFC for the gate-set facts.

### On-disk layout

```text
third_party/
  INVENTORY.md                # SPDX dependency + license inventory (authoritative TCB list)
  stwo/
    REVISION                  # pinned upstream commit + metadata (see schema below)
    LICENSE                   # upstream Apache-2.0, verbatim
    NOTICE                    # upstream NOTICE if present, verbatim
    UPSTREAM.md               # source URL, vendoring procedure, exclusions
    PATCHES/
      0001-<slug>.patch       # one file per local modification, applied in order
      0001-<slug>.md          # rationale, audit note, upstream-PR/issue link or "none"
    src/...                   # vendored source tree (workspace member crates)
  stwo-circuits/
    REVISION
    LICENSE
    NOTICE
    UPSTREAM.md
    PATCHES/
    src/...
```

`third_party/<comp>/REVISION` is the single source of truth for the pin. Schema
(TOML, one file per component):

```toml
# third_party/stwo/REVISION
component       = "stwo"
upstream_url    = "https://github.com/starkware-libs/stwo"
upstream_ref    = "refs/heads/main"     # the branch/tag the commit was taken from
commit          = "<40-hex-sha>"        # exact pinned revision; immutable for this pin
vendored_at     = "2026-06-03"          # ISO-8601 date of the vendoring/bump
workspace_version = "2.2.0"             # upstream workspace version at that commit
license         = "Apache-2.0"          # SPDX id; MUST match LICENSE file
tree_sha256     = "<64-hex>"            # sha256 of the canonicalized vendored src/ tree
patches_sha256  = "<64-hex>"            # sha256 of the concatenated, ordered PATCHES/*.patch
audited_by      = "area:security"       # role that signed off this pin
audit_note      = "third_party/stwo/PATCHES/AUDIT-2026-06-03.md"
```

Until the first vendoring lands, the `commit`/`tree_sha256` fields are stated as
"pinned at vendoring time, recorded in `third_party/stwo/REVISION`"; the
`v0.1 — Foundations` milestone task "vendoring" fills them in. The
`workspace_version` for `stwo` is `2.2.0` and for `stwo-circuits` is `0.1.0`,
verified against upstream at 2026-06-03.

### No-direct-upstream-dependency policy

The locked rule: **no workspace crate declares a dependency on any Stwo or
`stwo-circuits` package by registry or git source.** Every such dependency is a
path dependency into the vendored tree.

```toml
# crates/pwm-air/Cargo.toml — REQUIRED form
[dependencies]
stwo                       = { path = "../../third_party/stwo/src/stwo" }
stwo-constraint-framework  = { path = "../../third_party/stwo/src/stwo-constraint-framework" }
stwo-air-utils             = { path = "../../third_party/stwo/src/stwo-air-utils" }

# FORBIDDEN forms (CI gate rejects either):
# stwo = "2.2.0"
# stwo = { git = "https://github.com/starkware-libs/stwo", rev = "..." }
```

Enforcement is mechanical, not advisory. The CI gate `vendor-policy`
(`docs/spec/07-testing-strategy.md#ci-gates`) runs `cargo metadata
--format-version 1` and asserts:

- **INV-RFC0015-01 (no-upstream-source):** every resolved package whose name is in
  the Stwo or `stwo-circuits` package set has a `source` of `null` (path
  dependency). Any `registry+`/`git+` source for those names fails the gate.
- **INV-RFC0015-02 (single-copy):** each vendored package name resolves to exactly
  one version/path in the dependency graph — no duplicate or shadow copy of a
  substrate crate.

### Modification workflow

The vendored tree is treated as a forked-but-quarantined source. Modifications are
recorded, never silent.

1. The pristine upstream tree at the pinned `commit` is the baseline. The vendored
   `src/` may carry applied modifications, but every applied change is reproducible
   from `PATCHES/`.
2. Each modification is one numbered patch `PATCHES/NNNN-<slug>.patch` plus a
   sibling `PATCHES/NNNN-<slug>.md` stating: what changed, why
   (API exposure / visibility / bugfix / build), whether it is offered upstream
   (PR/issue link or "not offered"), and the soundness impact assessment
   (`none` / `local-only` / `affects-constraints`).
3. `PATCHES/*.md` with soundness impact `affects-constraints` require an
   `area:security` review recorded in the audit note before merge. This is the
   audit boundary: changes that touch the constraint framework, FRI, channels, or
   LogUp are security-reviewed; trace-helper or visibility tweaks are not.
4. `patches_sha256` in `REVISION` is the sha256 of the ordered, concatenated patch
   set. The CI gate `vendor-integrity` recomputes it and `tree_sha256` and fails
   if either drifts from `REVISION` — i.e. you cannot edit the vendored tree
   without updating the recorded hashes, and you cannot update the hashes without
   the patch files explaining the change.

- **INV-RFC0015-03 (patch-completeness):** applying `PATCHES/*.patch` in order to a
  fresh checkout of `upstream_url@commit` reproduces the vendored `src/` tree
  byte-for-byte (`tree_sha256` matches). No undocumented in-place edits exist.

### Revision-bump workflow

Bumping a vendored revision (to pull upstream fixes or features) is a gated change,
not a routine update.

1. Re-vendor at the new upstream `commit`; rebase the `PATCHES/` series onto it,
   resolving conflicts; update `REVISION` (`commit`, `vendored_at`,
   `workspace_version`, `tree_sha256`, `patches_sha256`).
2. Run the **substrate-conformance suite** (Testing Strategy below). It must pass
   unchanged, or any change in proof bytes/verification behavior is documented and
   security-reviewed.
3. Obtain a fresh `area:security` sign-off; write `PATCHES/AUDIT-<date>.md`
   summarizing the upstream diff between old and new `commit` for the audited
   surface (constraint framework, FRI, channels, LogUp, M31 field module) and the
   conclusion. Point `audit_note` at it.
4. A revision bump that changes the substrate's soundness-relevant behavior is a
   **relation-affecting change**: per
   `docs/spec/09-release-and-versioning.md#relation-versioning`, if proof bytes or
   the verification relation change semantically, the affected `relation_id`s mint
   new versions (e.g. `pwm.lewm.fixed_candidate_planning.v1` ->
   `.v2`). A pure additive/no-behavior-change bump does not.

- **INV-RFC0015-04 (audited-pin):** every value of `commit` recorded in any merged
  `REVISION` has a corresponding `area:security` audit note reachable from
  `audit_note`. An un-audited pin cannot reach the default branch.

### Dependency / license inventory (SPDX)

`third_party/INVENTORY.md` is the authoritative enumeration of the trusted
computing base. It is generated and checked, not hand-maintained drift-prone prose.
It is produced from `cargo metadata` plus the `REVISION` files and lists, for every
package compiled into the prover or verifier:

| Field | Meaning |
| --- | --- |
| `name` | Crate/package name |
| `version` | Semver (registry crates) or `rev:<short-sha>` (vendored) |
| `spdx_license` | SPDX license id (e.g. `Apache-2.0`, `MIT`, `MIT OR Apache-2.0`) |
| `source` | `crates.io` / `vendored:third_party/<comp>` / `consumed-input` |
| `tier` | `substrate` (vendored, audited), `support` (registry, semver-pinned), `reference` (not in build graph), `consumed-input` (e.g. `le-wm` checkpoint) |
| `in_verifier` | bool — whether it is on the `no_std` verifier path (`docs/spec/06-security.md#trust-boundaries`) |
| `notice_required` | bool — whether the license requires `NOTICE`/attribution retention |

Two appendices:

- **License compatibility statement.** All `tier: substrate` and `tier: support`
  packages must carry a license compatible with ProvableWorldModel's Apache-2.0
  (`docs/spec/09-release-and-versioning.md#license`). Permitted: `Apache-2.0`,
  `MIT`, `MIT OR Apache-2.0`, `BSD-2-Clause`, `BSD-3-Clause`, `ISC`,
  `Unlicense OR MIT`, `Zlib`. Copyleft (`GPL-*`, `AGPL-*`, `LGPL-*`) and unknown
  licenses are **denied** and fail the gate.
- **Consumed inputs.** `le-wm` (MIT) listed as `consumed-input`: source not
  vendored, attribution retained, checkpoint/config consumed by the export
  pipeline only.

The CI gate `inventory-license`
(`docs/spec/07-testing-strategy.md#ci-gates`) regenerates the inventory and fails
if (a) it differs from the committed `INVENTORY.md`, or (b) any in-build package
carries a denied/unknown SPDX id.

- **INV-RFC0015-05 (license-allowlist):** no package with `tier` in
  `{substrate, support}` carries a license outside the permitted allowlist;
  `Apache-2.0` and `MIT` for the vendored substrate are verified present.
- **INV-RFC0015-06 (inventory-complete):** every package in the prover and verifier
  dependency graphs appears in `INVENTORY.md` with a non-empty `spdx_license` and
  `tier`. No silent additions to the trusted computing base.

### Build provenance binding

The vendored pins are part of build provenance so a proof can be tied to the exact
substrate. The prover embeds, in `ProofArtifact` metadata
(`docs/spec/02-public-api.md#artifact-formats`, see `Proof` /
`ProofArtifact` in §6.5 of the data model), a `substrate_provenance` record:

```rust
/// Build-time provenance of the vendored proving substrate. Emitted into
/// ProofArtifact metadata; not a soundness input by itself, but lets a verifier
/// operator confirm the substrate matches an audited pin out-of-band.
pub struct SubstrateProvenance {
    pub stwo_commit: [u8; 20],            // pinned stwo commit
    pub stwo_patches_sha256: [u8; 32],    // patches_sha256 from third_party/stwo/REVISION
    pub stwo_circuits_commit: [u8; 20],
    pub stwo_circuits_patches_sha256: [u8; 32],
    pub inventory_sha256: [u8; 32],       // sha256 of third_party/INVENTORY.md at build time
}
```

The verifier (`verify(artifact: &ProofArtifact) -> Result<(), VerifyError>`,
`docs/spec/02-public-api.md#rust-public-api`) does **not** trust this record for
soundness — the soundness argument is over the verifier binary actually running,
not over a self-reported string. It is an operator-facing provenance aid: an
operator can compare `SubstrateProvenance` against the audited `REVISION` values to
detect that a proof was produced by an un-audited substrate. The reproducibility
contract in `docs/rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md` treats
`SubstrateProvenance` as part of the reproducible build identity.

### Failure modes and system response

| ID | Failure mode | System response |
| --- | --- | --- |
| F-RFC0015-01 | A `Cargo.toml` declares a registry/git dependency on a substrate package | CI gate `vendor-policy` fails (INV-RFC0015-01); merge blocked. Error class: build-config error, `docs/spec/04-error-model.md#error-taxonomy`. |
| F-RFC0015-02 | Two versions/copies of a substrate crate resolve in the graph | CI gate `vendor-policy` fails (INV-RFC0015-02); merge blocked. |
| F-RFC0015-03 | Vendored `src/` edited without a recorded patch (hashes drift) | CI gate `vendor-integrity` fails (INV-RFC0015-03): `tree_sha256`/`patches_sha256` mismatch; merge blocked with the offending file diff. |
| F-RFC0015-04 | `REVISION.commit` changed without an audit note | CI gate `vendor-audit` fails (INV-RFC0015-04); merge blocked pending `area:security` sign-off. |
| F-RFC0015-05 | A new transitive crate enters the build with a denied/unknown license | CI gate `inventory-license` fails (INV-RFC0015-05); merge blocked. |
| F-RFC0015-06 | A build crate is missing from `INVENTORY.md` | CI gate `inventory-license` fails (INV-RFC0015-06): generated inventory differs from committed; merge blocked. |
| F-RFC0015-07 | A revision bump silently changes proof bytes / verification behavior | Substrate-conformance suite detects a golden-vector or proof-bytes diff; treated as relation-affecting, requires new `relation_id` version (`docs/spec/09-release-and-versioning.md#relation-versioning`) and security review before merge. |
| F-RFC0015-08 | `SubstrateProvenance` in an artifact does not match any audited `REVISION` | Verification does **not** fail on this alone (provenance is advisory). Operator tooling surfaces a warning; CI's release gate refuses to publish an artifact whose provenance does not match the current pin. |
| F-RFC0015-09 | Upstream relicenses or yanks the pinned commit | No build impact (we are vendored and pinned). Inventory's source URL is annotated; the next bump's audit note records the upstream license change and the compatibility re-check. |

## Alternatives Considered

**A. Depend on upstream crates via `Cargo.lock` pinning (no vendoring).** Declare
`stwo = { git = "...", rev = "<sha>" }` and rely on `Cargo.lock` to pin. *Why
considered:* it is the idiomatic Rust approach and avoids carrying a source tree.
*Why rejected:* we will modify the substrate (the founding analysis says so
explicitly, `docs/feasibility-study.md` §0/§3.2), and a git/registry dependency
cannot carry local patches without a fork anyway — at which point we are vendoring,
just less legibly. It also keeps the audited code outside the repo: an auditor
cannot review the trusted computing base from a single checkout, and a fetch
failure (yanked rev, repo move) breaks reproducible builds of a cryptographic
artifact. `Cargo.lock` pins versions but not the audit boundary or the patch
history.

**B. Cargo `[patch]` overrides on top of an upstream dependency.** Keep the
upstream dependency and apply local changes through `[patch.crates-io]` /
`[patch."git"]` pointing at small fork crates. *Why considered:* it is a
lighter-weight way to carry a few local fixes while tracking upstream. *Why
rejected:* `[patch]` is brittle for whole-workspace, multi-crate substrates with
proc-macro members (`stwo-air-utils-derive`) and a `no_std` verifier split; it
distributes the trusted code across an upstream source plus patch crates, defeating
the "one auditable tree" goal; and it still depends on upstream remaining
fetchable. It optimizes for staying close to upstream, which is the opposite of
this project's goal of freezing the substrate.

**C. Git submodules for `third_party/stwo` and `third_party/stwo-circuits`.** Pin
upstream as submodules at specific commits. *Why considered:* preserves upstream
git history and makes bumps a one-line submodule update. *Why rejected:* submodules
keep the trusted code in a separate repo (auditor friction, fetch dependency at
build time), make local modifications awkward (you must fork the submodule's repo),
and split provenance across two histories. Our `REVISION` + `PATCHES/` mechanism
gives the same "track an upstream commit" capability while keeping a single
checkout, single audit boundary, and explicit patch trail.

## Drawbacks

- **Maintenance cost of staying current.** Vendoring decouples us from upstream
  fixes; pulling them is a deliberate, gated bump (re-vendor, rebase patches,
  re-audit, conformance run). We accept slower uptake of upstream improvements in
  exchange for a frozen, auditable substrate. This is the intended trade.
- **Repository size and review surface.** The vendored trees add a large source
  volume to the repo and to code-review tooling. We mitigate by excluding upstream
  `examples`/test-only crates from the build graph and marking them `reference` in
  the inventory.
- **Patch rebase friction.** Local patches must be rebased on every bump; a large
  upstream refactor can make rebasing expensive. We mitigate by keeping patches
  minimal and preferring additive hooks over invasive edits, and by recording each
  patch's rationale so a costly rebase can be reconsidered.
- **Provenance is advisory, not enforced.** `SubstrateProvenance` cannot be a
  soundness input (a malicious prover can forge a string). It only helps honest
  operators detect substrate mismatch. The real guarantee is that the *verifier
  binary* is built from the audited pin, which is an operational/CI property, not
  a cryptographic one.

## Migration / Rollout

This RFC lands in `v0.1 — Foundations` as part of the "vendoring" milestone task;
there is no pre-existing dependency to migrate away from, so rollout is greenfield.

1. **Initial vendoring (v0.1).** Copy both upstream trees at chosen commits into
   `third_party/`, fill `REVISION` (`commit`, `tree_sha256`, `patches_sha256`
   with an empty patch set initially), copy `LICENSE`/`NOTICE` verbatim, write
   `UPSTREAM.md`, generate `INVENTORY.md`. Convert all `pwm-*` crates to path
   dependencies. Land the CI gates `vendor-policy`, `vendor-integrity`,
   `vendor-audit`, `inventory-license`
   (`docs/spec/07-testing-strategy.md#ci-gates`) as required checks from day one.
2. **Modification gating.** From first vendoring onward, any change to the
   vendored tree must go through `PATCHES/`; the integrity gate enforces it
   immediately. No grace period — a forked tree with silent edits is exactly the
   risk this RFC exists to prevent.
3. **Revision bumps.** Governed by the revision-bump workflow above. Bumps are
   normal PRs gated by `vendor-audit` and the substrate-conformance suite.
   Relation-affecting bumps mint new `relation_id` versions per
   `docs/spec/09-release-and-versioning.md#relation-versioning`; the change is
   recorded in the changelog (`docs/spec/09-release-and-versioning.md#changelog`).
4. **Feature flag for the verifier `no_std` path.** The vendored `stwo` verifier
   path is `no_std`-capable via `std-shims`; `pwm-verifier` exposes a `std`
   feature (default on) and a `no_std` build that the `ensure-verifier-no_std`
   conformance crate exercises. No deprecation is needed; both build modes are
   first-class.

There is no deprecation window because nothing pre-dates this policy. Future
changes to the *policy* (e.g. allowing a new license, adding a vendored
component) are themselves RFCs or amendments to this one and follow the
`docs/spec/09-release-and-versioning.md#deprecation` discipline for the inventory
schema.

## Testing Strategy

All gates and suites are registered in
`docs/spec/07-testing-strategy.md#ci-gates`; negative behaviors map to
`docs/spec/04-error-model.md#error-taxonomy`. The **substrate-conformance suite**
is the named collection of tests that must pass on every revision bump.

Accepting tests:

- `test_vendor_policy_all_substrate_paths`: parses `cargo metadata`; asserts every
  Stwo/`stwo-circuits` package resolves to a `null`-source path dependency
  (INV-RFC0015-01) and to exactly one copy (INV-RFC0015-02).
- `test_vendor_integrity_tree_reproduces`: clones `upstream_url@commit`, applies
  `PATCHES/*.patch` in order, recomputes `tree_sha256` and `patches_sha256`, and
  asserts they match `REVISION` (INV-RFC0015-03).
- `test_inventory_matches_metadata`: regenerates the inventory from `cargo
  metadata` + `REVISION` files and asserts byte-equality with `INVENTORY.md`
  (INV-RFC0015-06); asserts `stwo` and `stwo-circuits` are present as
  `tier: substrate`, `spdx_license = Apache-2.0`, and `le-wm` as
  `tier: consumed-input`, `spdx_license = MIT`.
- `test_license_allowlist_holds`: asserts every `tier in {substrate, support}`
  package's SPDX id is in the permitted allowlist (INV-RFC0015-05).
- `test_audited_pin_has_note`: asserts every `REVISION.commit` on the default
  branch has a reachable `area:security` audit note (INV-RFC0015-04).
- `test_verifier_builds_no_std`: builds `pwm-verifier` against the vendored
  `no_std` `stwo` path via `std-shims` and runs the `ensure-verifier-no_std`
  conformance crate.
- `test_substrate_conformance_golden`: runs the corpus golden vectors
  (`docs/spec/07-testing-strategy.md#golden-vectors`) for P0/P1/P2 against the
  pinned substrate and asserts proof bytes and `verify(...)` results are unchanged
  versus the recorded baseline (the bump tripwire for F-RFC0015-07).
- `test_provenance_round_trip`: builds a `ProofArtifact`, asserts its
  `SubstrateProvenance` equals the values in the two `REVISION` files and the
  `inventory_sha256` of the committed inventory.

Rejecting / negative tests (each asserts the named gate fails or the named error
is produced):

- `reject_registry_dependency_on_stwo`: a fixture `Cargo.toml` with
  `stwo = "2.2.0"` makes `vendor-policy` fail with the build-config error
  (F-RFC0015-01).
- `reject_git_dependency_on_stwo_circuits`: a `git`-sourced `stwo-circuits`
  dependency fails `vendor-policy` (F-RFC0015-01).
- `reject_duplicate_substrate_crate`: a second path/version copy of `stwo` in the
  graph fails `vendor-policy` (F-RFC0015-02).
- `reject_unrecorded_tree_edit`: mutate one byte of vendored `src/` without a
  patch; `vendor-integrity` fails with a `tree_sha256` mismatch (F-RFC0015-03).
- `reject_patch_hash_drift`: alter a `PATCHES/*.patch` without updating
  `patches_sha256`; `vendor-integrity` fails (F-RFC0015-03).
- `reject_unaudited_revision_bump`: change `REVISION.commit` with no audit note;
  `vendor-audit` fails (F-RFC0015-04).
- `reject_copyleft_dependency`: inject a fixture crate declaring `GPL-3.0` into the
  graph; `inventory-license` fails the allowlist check (F-RFC0015-05).
- `reject_missing_inventory_entry`: add a build crate but not its `INVENTORY.md`
  row; `inventory-license` fails on the regenerate-and-compare step
  (F-RFC0015-06).
- `reject_release_with_stale_provenance`: attempt to publish an artifact whose
  `SubstrateProvenance` does not match the current pin; the release gate refuses
  (F-RFC0015-08). Note this is a *release* gate, not a `verify(...)` rejection —
  `verify(...)` still accepts the proof if it is cryptographically valid, because
  provenance is advisory (see Drawbacks).

Differential test:

- `diff_upstream_audit_surface_on_bump`: on a proposed revision bump, computes the
  upstream diff between old and new `commit` restricted to the audited surface
  (M31 field module, FRI, channels, constraint framework, LogUp) and attaches it
  to the audit note; the test fails if the diff is non-empty and no `area:security`
  sign-off is present.

## Open Questions

- OPEN QUESTION (owner: `area:security`): exact granularity of "audited surface"
  for revision bumps — whether to security-review only constraint-framework/FRI/
  channel/M31 changes or the full upstream diff. Resolution path: settle in the
  first `third_party/stwo/PATCHES/AUDIT-*.md` produced under the `v0.1 —
  Foundations` milestone; codify the chosen surface list in this RFC by amendment.
- OPEN QUESTION (owner: `area:core`): whether `SubstrateProvenance` commits should
  be 20-byte git SHA-1 or migrate to SHA-256 object ids if upstream repos move to
  SHA-256 git. Resolution path:
  `docs/rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md` owns the artifact
  metadata schema; resolve there before the `v0.2` artifact bundle freeze, keeping
  the field width versioned by `ProofArtifact.artifact_version`.
- OPEN QUESTION (owner: `area:ci`): whether to additionally run `cargo deny`
  (advisory DB + license + bans) as the implementation of the
  `inventory-license`/`vendor-policy` gates rather than a bespoke `cargo metadata`
  checker. Resolution path: decide during the `v0.1 — Foundations` CI scaffolding
  task; either tool satisfies the invariants here, so this is an implementation
  choice, not a policy change.

## References

- Founding analysis: `docs/feasibility-study.md` §0 (vendoring posture), §3.2
  (direct-AIR vs circuit lowering, `stwo-circuits` substrate role), §13 (risk
  register, `stwo-circuits` maturity / pin-and-audit mitigation).
- Authoring contract §3 (decided choices: license Apache-2.0, vendoring), §4
  (crate/area naming, milestones), §9b (verified external facts and corrections:
  Stwo field module, crate lists, `stwo-circuits` 10 crates and 10-gate set).
- `docs/spec/01-architecture.md#crate-layout`, `docs/spec/01-architecture.md#air-strategy`
  (consumers of the vendored substrate; corrected gate set).
- `docs/spec/02-public-api.md#artifact-formats`,
  `docs/spec/02-public-api.md#rust-public-api` (`ProofArtifact`, `verify`).
- `docs/spec/04-error-model.md#error-taxonomy` (build-config and verifier error
  classes referenced by failure modes).
- `docs/spec/06-security.md#trust-boundaries`,
  `docs/spec/06-security.md#soundness-requirements` (the trusted computing base
  this RFC pins; the binding of Stwo/security parameters).
- `docs/spec/07-testing-strategy.md#ci-gates`,
  `docs/spec/07-testing-strategy.md#golden-vectors`,
  `docs/spec/07-testing-strategy.md#differential-tests` (gates and suites).
- `docs/spec/09-release-and-versioning.md#license`,
  `docs/spec/09-release-and-versioning.md#relation-versioning`,
  `docs/spec/09-release-and-versioning.md#changelog`,
  `docs/spec/09-release-and-versioning.md#deprecation`,
  `docs/spec/09-release-and-versioning.md#msrv`.
- Related RFCs:
  `docs/rfcs/RFC-0001-model-manifest-and-export-pipeline.md` (le-wm attribution,
  consumed input), `docs/rfcs/RFC-0003-range-check-and-lookup-infrastructure.md`
  (corrected gate set, LogUp), `docs/rfcs/RFC-0012-recursive-aggregated-verification.md`
  (stwo-cairo as reference prior art),
  `docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md` (pinned
  channel/transcript implementation),
  `docs/rfcs/RFC-0016-cli-artifact-bundle-and-reproducibility.md` (provenance in the
  artifact bundle, reproducibility contract).
- Upstream sources (verified against upstream at 2026-06-03):
  `github.com/starkware-libs/stwo` (Apache-2.0, workspace v2.2.0);
  `github.com/starkware-libs/stwo-circuits` (Apache-2.0, v0.1.0, edition 2024);
  `github.com/lucas-maes/le-wm` (MIT, consumed input, not vendored);
  `github.com/starkware-libs/stwo-cairo` (Apache-2.0, reference only).
