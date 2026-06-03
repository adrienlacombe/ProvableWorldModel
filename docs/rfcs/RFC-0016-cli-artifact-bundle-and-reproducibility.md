# RFC-0016: Prover/verifier CLI, proof artifact bundle, and reproducibility contract

- Status: Accepted
- Authors: ProvableWorldModel maintainers
- Created: 2026-06-03
- Target milestone: v0.2

## Summary

This RFC locks the operator-facing surface that turns a committed quantized
LeWorldModel manifest and its inputs into a verifiable proof, and back. It fixes
three things permanently: (1) the `pwm` CLI command shapes for `export`,
`manifest`, `prove`, and `verify`, including their flags, stdout/stderr
discipline, and process exit codes; (2) the on-disk `ProofArtifact` bundle
layout, its directory structure, the files it contains, and the `artifact_version`
that gates compatibility; and (3) the bit-for-bit reproducibility contract:
identical inputs plus a pinned toolchain MUST yield identical proof inputs (the
public-input digest and the reference inference trace inputs) and identical
claimed outputs, witnessed by a `determinism digest` that the CLI computes and
that anyone can recompute. The proof bytes themselves are NOT required to be
byte-identical (the prover may sample randomness), but everything they bind to
is. Wire-level serialization of the `Proof` and `PublicInput` structures is owned
by `docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md`; this RFC
consumes that format and does not redefine it.

## Motivation

The founding analysis specifies a prover flow (source §10.3) and a verifier flow
(source §10.4) as ordered step lists, and the crate layout (source §3.1) places a
`cli.rs` in `pwm-prover`, but neither pins the command surface, the artifact a
proof actually is on disk, nor what "same model, same inputs" reproducibility
guarantees in concrete bytes. Without that pinned, three failure scenarios are
live:

- An operator on machine A produces a proof; an auditor on machine B cannot
  reproduce the same public-input digest and claimed outputs, so the proof's
  binding to a specific model and inputs cannot be independently re-derived. This
  defeats the audit story that
  `docs/spec/06-security.md#binding-requirements` depends on.
- A CI job and a release job disagree on what `prove` emits, so the artifact a
  user downloads is not the artifact that was tested.
- A verifier consumes a bundle written by a newer prover with a layout it does
  not understand and either misreads it silently or rejects with an unhelpful
  error, instead of a typed version mismatch.

The `ProofArtifact` type is already canonical (contract §6.5, reproduced in
`docs/spec/03-data-model.md`). This RFC gives it an on-disk realization, a CLI
that produces and consumes it, and a reproducibility contract strong enough that
the determinism guarantees in `docs/spec/06-security.md#soundness-requirements`
(source §9.2) are checkable by a third party rather than asserted.

Concrete scenarios this serves:

- A maintainer cuts a release: `pwm export` then `pwm prove` then
  `pwm verify`, and CI re-runs the same three commands and compares determinism
  digests to a committed golden value
  (`docs/spec/07-testing-strategy.md#golden-vectors`).
- An external auditor receives a `.pwmproof` bundle and a manifest, runs
  `pwm verify --artifact x.pwmproof --manifest model.yaml`, and gets exit code 0
  with a printed public-input digest they can match against an independent
  transcript.
- A researcher re-runs `pwm prove` on the same inputs a week later on different
  hardware and confirms the determinism digest is unchanged, proving the inputs
  and claimed outputs did not drift even though the proof bytes differ.

## Goals

- Lock the `pwm` subcommand set (`export`, `manifest`, `prove`, `verify`), their
  flags, their argument types, and their exit codes, so scripts and CI can depend
  on them across v0.2 without breakage.
- Lock the on-disk `ProofArtifact` bundle: a single `.pwmproof` file (a
  deterministic, uncompressed tar) with a fixed member layout, and an
  `artifact_version: u32` that gates reader/writer compatibility.
- Lock the reproducibility contract: define exactly what MUST be bit-identical
  (proof inputs and claimed outputs) versus what MAY differ (proof bytes), and
  define the `determinism digest` that makes the guarantee checkable.
- Define the `pwm` process contract: stdout carries machine-readable results
  (one JSON object per invocation), stderr carries human logs, exit codes encode
  outcome classes that map to `docs/spec/04-error-model.md#error-taxonomy`.
- Define how the bundle and CLI version independently of the relation
  (`relation_id`) and the manifest schema, and the migration path when any of
  the three change.

## Non-Goals

- The byte layout of the `Proof` and `PublicInput` structures themselves, the
  public-input digest construction, and the Fiat-Shamir transcript ordering. Those
  are owned by `docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md`. This
  RFC references the digest; it does not define it.
- The manifest schema and the export quantization algorithm, owned by
  `docs/rfcs/RFC-0001-model-manifest-and-export-pipeline.md`. This RFC defines the
  `pwm export` and `pwm manifest` command shapes that invoke that pipeline, not the
  pipeline's internals.
- The Rust library API (`pwm_prover::prove_*`, `pwm_verifier::verify`). That is
  `docs/spec/02-public-api.md#rust-public-api`. The CLI is a thin shell over it.
- Proving the CEM planner (P3), the pixel encoder (P4), recursion, or ZK mode.
  The CLI is forward-compatible with new `--statement` values, but only P0/P1 ship
  in v0.2 and P2 in v1.0; P3/P4 are gated on
  `docs/rfcs/RFC-0010-cem-planner-proof.md` and
  `docs/rfcs/RFC-0011-pixel-encoder-proof.md`.
- Performance targets for `prove`/`verify`. See
  `docs/spec/08-performance-budget.md#targets`.

## Proposed Design

### 1. The `pwm` CLI

A single binary `pwm` (built by `pwm-prover`; the `verify` subcommand links only
`pwm-verifier` and `pwm-core` so a verifier-only build excludes the prover and
the Python reference). Global conventions, fixed for all subcommands:

- stdout: exactly one JSON object, the command result, printed once on
  completion. Nothing else goes to stdout. This makes `pwm ... | jq` reliable.
- stderr: human-readable structured logs per
  `docs/spec/05-observability.md#logging`, controlled by `--log-level`
  (`error|warn|info|debug|trace`, default `info`) and `--log-format`
  (`text|json`, default `text`). Private witness material and raw weights are
  never logged (`docs/spec/05-observability.md#redaction`).
- Exit codes (the locked mapping; categories align with
  `docs/spec/04-error-model.md#error-taxonomy`):

  | Code | Class                 | Meaning                                                              |
  | ---- | --------------------- | -------------------------------------------------------------------- |
  | 0    | Success               | Operation completed; for `verify`, the proof is valid.               |
  | 2    | Usage                 | Bad flags/args (clap-level); no work attempted.                      |
  | 3    | ExportError           | Export/quantization failure (`04-error-model.md#error-taxonomy`).    |
  | 4    | ManifestError         | Manifest parse/validation/commitment-mismatch failure.              |
  | 5    | TraceError            | Reference inference or trace build failure (e.g. overflow=reject).   |
  | 6    | ProveError            | Proving failure after a valid trace.                                 |
  | 7    | VerifyRejected        | `verify` ran and the proof is INVALID (`VerifyError`). Not an error in the operational sense; it is a verdict. |
  | 8    | ArtifactError         | Bundle read/write/layout/`artifact_version` mismatch.                |
  | 9    | ReproMismatch         | `--expect-digest` given and the computed determinism digest differs. |
  | 70   | InternalError         | Bug/invariant violation (panics are caught and mapped here).         |

  Exit code 7 (`VerifyRejected`) is deliberately distinct from 0 and from the
  error classes: a verifier that runs to completion and decides "invalid" did its
  job. Scripts gate on `== 0` for "valid", `== 7` for "definitively invalid",
  and anything else for "could not decide".

`pwm` flag and exit-code semantics are stable for the life of `artifact_version`
1 under the policy in `docs/spec/02-public-api.md#stability-policy`.

#### 1.1 `pwm export`

Quantizes a PyTorch checkpoint into a manifest + weight tensors + golden vectors.
Delegates to the `pwm-export` pipeline
(`docs/rfcs/RFC-0001-model-manifest-and-export-pipeline.md`).

```text
pwm export
  --checkpoint <PATH>        # PyTorch .pt/.pth (le-wm checkpoint; MIT-licensed, not vendored)
  --config <PATH>            # le-wm model/eval config used to derive the operator graph
  --statement <P0|P1|P2>     # which statement's op-subset to export; default P2
  --quant <PATH>             # quantization policy file (int8/int16 widths, scales, rounding)
  --out-dir <DIR>            # destination for manifest + weights + golden vectors
  [--rounding nearest_ties_to_even|truncate_toward_zero]   # default nearest_ties_to_even
  [--golden-count <N>]       # number of golden input/output vectors to emit; default 16
```

Emits `<out-dir>/manifest.yaml`, `<out-dir>/weights/`, `<out-dir>/golden/`, and a
result JSON on stdout:

```json
{ "ok": true, "model_commitment": "0x...", "quantization_commitment": "0x...",
  "manifest_path": "out/manifest.yaml", "golden_vectors": 16 }
```

Re-running `pwm export` on the same checkpoint + config + quant policy + rounding
MUST produce a byte-identical `manifest.yaml` and byte-identical weight tensors
(this is the export half of the reproducibility contract; see §3 and
`docs/rfcs/RFC-0001-model-manifest-and-export-pipeline.md`). Failure class:
exit 3 (export) or 4 (manifest write/validate).

#### 1.2 `pwm manifest`

Inspects and verifies a manifest without proving. Two modes:

```text
pwm manifest verify --manifest <PATH>
    # recompute model_commitment + quantization_commitment from the canonical
    # serialization; confirm they match the manifest's declared roots and the
    # serialization.canonical_json_hash. Exit 0 if consistent, 4 otherwise.

pwm manifest show --manifest <PATH> [--field <dotted.key>]
    # print the manifest (or one field) as JSON to stdout. Read-only. Exit 0/4.
```

Result JSON for `verify`:

```json
{ "ok": true, "model_commitment": "0x...", "quantization_commitment": "0x...",
  "relation_id": "pwm.lewm.fixed_candidate_planning.v1", "manifest_version": "pwm-model-manifest-v1" }
```

#### 1.3 `pwm prove`

Builds the reference trace and produces a `ProofArtifact` bundle. This is the CLI
realization of source §10.3.

```text
pwm prove
  --manifest <PATH>                  # committed model manifest
  --weights <DIR|FILE>               # quantized weights matching weights.root
  --statement <P0|P1|P2>             # must equal the manifest's exported statement subset
  --inputs <PATH>                    # JSON: latent_history, goal_latent, candidate_actions, etc.
  --out <PATH.pwmproof>              # output bundle path
  [--include-outputs]                # embed claimed_outputs tensors in the bundle (default on)
  [--weights-visibility public|private_committed]   # default: read from manifest
  [--expect-digest <HEX>]            # fail (exit 9) if determinism digest != this value
  [--threads <N>]                    # prover parallelism; MUST NOT affect any committed value
```

Ordered behavior (binding the source §10.3 flow to named invariants):

1. Load and canonicalize the manifest; recompute `model_commitment` and
   `quantization_commitment`; reject (exit 4) on mismatch [INV-ART-01].
2. Load weights; verify their commitment equals `weights.root`; reject (exit 4)
   on mismatch [INV-ART-01].
3. Canonicalize `--inputs` into the `PublicInput`/`Witness` split
   (contract §6.3) using the canonical serialization of
   `docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md`.
4. Execute the Rust fixed-point reference inference (source §10.3 step 6) under
   `OverflowPolicy::Reject`; any overflow/clamp violation aborts with exit 5
   [INV-ART-04].
5. Build operator, range, and lookup traces; commit to the Stwo channel and
   derive challenges in the canonical order owned by RFC-0014.
6. Generate the proof; assemble the `ProofArtifact` (contract §6.5) with
   `artifact_version = 1`.
7. Compute the `determinism digest` over the proof INPUTS and claimed outputs
   (§3); if `--expect-digest` was given and differs, exit 9 without writing
   [INV-ART-05].
8. Write the `.pwmproof` bundle atomically (§2.2) [INV-ART-03].

Result JSON:

```json
{ "ok": true, "artifact_version": 1, "relation_id": "pwm.lewm.rollout.v1",
  "statement_type": "P1Rollout", "public_input_digest": "0x...",
  "determinism_digest": "0x...", "artifact_path": "proof.pwmproof",
  "proof_bytes": 1184256, "claimed_outputs_included": true }
```

#### 1.4 `pwm verify`

The CLI realization of source §10.4. Verifier-only build; never executes the
Python reference; never executes PyTorch (source §10.4 closing rule;
`docs/spec/06-security.md#trust-boundaries`).

```text
pwm verify
  --artifact <PATH.pwmproof>
  [--manifest <PATH>]            # if given, also bind-check the bundle's commitments against it
  [--expect-relation <ID>]       # require this relation_id; mismatch => exit 7 (rejected)
  [--expect-digest <HEX>]        # require this determinism digest; mismatch => exit 9
  [--print-public-input]         # include the decoded PublicInput in the result JSON
```

Ordered behavior (binding source §10.4):

1. Open the bundle; check `artifact_version` is readable by this build; reject
   with exit 8 (`ArtifactError`) on an unknown version [INV-ART-02].
2. Parse `PublicInput`; if `--expect-relation` is set and differs, exit 7.
3. Recompute the public-input digest (RFC-0014) and confirm it matches the
   digest embedded in the bundle index; mismatch => exit 7.
4. If `--manifest` is given, recompute `model_commitment`,
   `quantization_commitment`, `planner_config_commitment` from the manifest and
   confirm equality with the `PublicInput`; mismatch => exit 7.
5. Verify the Stwo proof against the relation's constraint set.
6. Decode `claimed_outputs` (if present); enforce application-level checks
   (selected index in range, tensor shapes match the relation, output commitment
   matches the canonical serialization of the decoded outputs) (source §10.4
   step 7).
7. If `--expect-digest` is set, recompute the determinism digest from the
   bundle and compare; mismatch => exit 9.

A valid proof exits 0; an invalid proof (any soundness/binding rejection from
`docs/spec/04-error-model.md#verifier-rejections`) exits 7. Result JSON:

```json
{ "ok": true, "valid": true, "relation_id": "pwm.lewm.rollout.v1",
  "public_input_digest": "0x...", "determinism_digest": "0x...",
  "artifact_version": 1 }
```

On rejection, `valid` is `false`, `ok` is `true` (the command ran fine), the
exit code is 7, and a `reject_reason` field names the `VerifyError` variant.

### 2. The on-disk `ProofArtifact` bundle (`.pwmproof`)

#### 2.1 Layout

A `.pwmproof` file is a single uncompressed `tar` archive (USTAR, sorted member
order, fixed mtime=0, uid/gid=0, mode 0644) so that two bundles built from the
same artifact are byte-identical [INV-ART-03]. Compression is intentionally
omitted: it would introduce a nondeterministic encoder and defeat byte-identity;
callers may gzip the whole file out of band. The member layout for
`artifact_version = 1`:

```text
proof.pwmproof  (tar)
  ├── index.json         # bundle index: artifact_version, member manifest, digests
  ├── public_input.bin   # canonical serialization of PublicInput (RFC-0014)
  ├── proof.bin          # canonical serialization of Proof (RFC-0014)
  └── outputs/           # present iff claimed_outputs is Some
        ├── meta.json    # tensor_id -> {shape, scale_id} for each claimed output
        └── t<ID>.bin    # canonical serialization of each claimed-output Tensor
```

`index.json` is the only file a reader parses before deciding it understands the
bundle. Its schema (the field set is part of the locked decision):

```json
{
  "artifact_version": 1,
  "relation_id": "pwm.lewm.rollout.v1",
  "statement_type": "P1Rollout",
  "members": {
    "public_input.bin": { "len": 384, "sha256": "0x..." },
    "proof.bin":        { "len": 1184256, "sha256": "0x..." },
    "outputs/meta.json": { "len": 142, "sha256": "0x..." }
  },
  "public_input_digest": "0x...",
  "determinism_digest": "0x...",
  "claimed_outputs_included": true,
  "created_utc": "2026-06-03T00:00:00Z",
  "producer": "pwm/0.2.0 stwo@<rev>"
}
```

`created_utc` and `producer` are informational and are EXCLUDED from every
digest (§3), so they do not break byte-identity of the digests across machines or
build times; they DO make the archive itself non-byte-identical across runs,
which is acceptable because the digests, not the archive bytes, are the
reproducibility unit. The per-member `sha256` values bind each file to the index
so a verifier detects truncation/tampering before parsing (exit 8 on mismatch)
[INV-ART-02].

#### 2.2 Write atomicity and read validation

- `prove` writes to `<out>.tmp` then renames to `<out>` (atomic on POSIX). A
  reader never sees a partial bundle [INV-ART-03].
- `verify` validates, in order: tar well-formedness, `index.json` parses,
  `artifact_version` known, every declared member present with matching length
  and `sha256`. Any failure is exit 8 (`ArtifactError`) with the specific cause.
  This happens before any cryptographic verification, so a corrupt bundle never
  reaches the prover-trusting code path [INV-ART-02].

#### 2.3 `artifact_version`

`artifact_version: u32` (contract §6.5) gates layout compatibility and is
independent of both `relation_id` and `manifest_version`:

- A reader rejects (exit 8) any `artifact_version` it does not implement. It does
  NOT attempt best-effort parsing.
- `artifact_version` increments only on an incompatible layout change (member
  set, index schema, tar conventions). Adding an OPTIONAL member that older
  readers can ignore is still a version bump in v0.x because v0.x makes no
  forward-compat promise; from v1.0 on, additive-optional changes are allowed
  within a version per `docs/spec/09-release-and-versioning.md#semver`.
- The version is duplicated in both the typed `ProofArtifact.artifact_version`
  and `index.json.artifact_version`; they MUST agree or the bundle is rejected
  (exit 8) [INV-ART-02].

### 3. The reproducibility contract and determinism digest

The contract, stated precisely so it is testable:

> Given identical inputs (the same manifest, the same weights, the same
> `--inputs`, the same `--statement`, the same declared rounding) and a pinned
> toolchain, two independent runs of `pwm prove` MUST produce identical proof
> INPUTS and identical claimed OUTPUTS. The proof bytes (`proof.bin`) MAY differ.

What is in scope of "identical":

| Quantity                          | Bit-identical across runs? | Why                                            |
| --------------------------------- | -------------------------- | ---------------------------------------------- |
| `manifest.yaml` from `pwm export` | Yes                        | Canonical serialization; binding to `model_commitment`. |
| Exported weight tensors           | Yes                        | Binding to `weights.root`.                     |
| Canonical `PublicInput` bytes     | Yes                        | RFC-0014 canonical serialization.              |
| `public_input_digest`             | Yes                        | Determined by the canonical `PublicInput`.     |
| Reference inference result / claimed outputs | Yes             | Deterministic fixed-point arithmetic (source §9.2). |
| `determinism_digest`              | Yes                        | Function of the two preceding rows only.       |
| `proof.bin`                       | No (MAY differ)            | The Stwo prover MAY use sampled randomness; soundness does not require proof byte-identity, only that every valid proof verifies. |
| `index.json.created_utc/producer` | No                         | Informational; excluded from all digests.      |

The **determinism digest** is the single value that makes the in-scope guarantee
checkable without re-deriving everything by hand. It is defined as:

```text
determinism_digest = Blake2s( DOMAIN_TAG_PWM_DETERMINISM
                            || public_input_digest                     // 32 bytes (RFC-0014)
                            || u32_le(num_claimed_outputs)
                            || for each claimed output in tensor_id order:
                                 canonical_tensor_bytes(output) )       // RFC-0014 Tensor encoding
```

The digest deliberately covers the proof's inputs (via `public_input_digest`,
which itself binds `relation_id`, all commitments, and any public input tensors)
and the claimed outputs, and deliberately excludes `proof.bin`. Blake2s is chosen
to match the Stwo Fiat-Shamir Blake2s channel backend (verified against upstream
at 2026-06-03; `crates/stwo/src/core/channel/`), avoiding a second hash primitive
in the trust base. The domain tag is a fixed ASCII constant
(`"pwm.determinism.v1"`) preventing cross-protocol collisions. `Blake2s` and the
exact `DOMAIN_TAG_PWM_DETERMINISM` byte string are fixed by RFC-0014; this RFC
fixes the structure that consumes them.

Pinned toolchain (the "pinned toolchain" half of the precondition) is the union
of: the Rust toolchain version recorded in `rust-toolchain.toml`
(`docs/spec/09-release-and-versioning.md#msrv`), the vendored stwo/stwo-circuits
revisions in `third_party/*/REVISION`
(`docs/rfcs/RFC-0015-third-party-vendoring-and-pinning.md`), and the `--rounding`
mode bound by `quantization_commitment`. The contract makes no claim across
different toolchains; it claims determinism within a pinned one.

Named invariant: any change that would alter a determinism digest for unchanged
inputs under a pinned toolchain is a defect, not a configuration option
[INV-ART-05]. `--threads` is explicitly within scope: parallelism MUST NOT affect
the reference inference result, the claimed outputs, or the digest.

### 4. Invariants

| ID          | Invariant                                                                                          | Enforced by                                  |
| ----------- | -------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| INV-ART-01  | `prove` refuses to build a trace unless the manifest's recomputed commitments and the weights' recomputed commitment match their declared roots. | `pwm prove` steps 1-2; exit 4.               |
| INV-ART-02  | A reader rejects any bundle whose `artifact_version` it does not implement, whose two version fields disagree, or whose member length/sha256 do not match `index.json`, BEFORE cryptographic verification. | `pwm verify` step 1; `.pwmproof` open path; exit 8. |
| INV-ART-03  | A `.pwmproof` is written atomically and the tar is canonical (sorted members, fixed mtime/uid/gid/mode), so two bundles from the same `ProofArtifact` are byte-identical. | `pwm prove` step 8; tar writer.              |
| INV-ART-04  | Reference inference under `OverflowPolicy::Reject` aborts `prove` (exit 5) rather than emitting a proof for a wrapped value. | `pwm prove` step 4; source §9.3.             |
| INV-ART-05  | For unchanged inputs and a pinned toolchain, the determinism digest is invariant; `--threads` and any other operational flag MUST NOT change it. | §3; differential CI gate.                    |
| INV-ART-06  | stdout carries exactly one JSON result object per invocation; private witness/weight material never appears on stdout or in the bundle's index. | CLI result writer; `docs/spec/05-observability.md#redaction`. |

### 5. Failure modes

| Failure                                              | Detected by                | System response                                                       |
| ---------------------------------------------------- | -------------------------- | -------------------------------------------------------------------- |
| Manifest commitment mismatch                         | `prove`/`manifest verify`  | Abort, exit 4 (`ManifestError`); name the mismatching commitment.    |
| Weights commitment != `weights.root`                 | `prove` step 2             | Abort, exit 4; do not load further.                                  |
| `--inputs` malformed / shape mismatch with relation  | `prove` step 3             | Abort, exit 5 (`TraceError`); name the offending field/shape.        |
| Overflow under reject policy                          | `prove` step 4             | Abort, exit 5; report the op and tensor that overflowed.             |
| Proving failure after a valid trace                  | `prove` step 6             | Abort, exit 6 (`ProveError`).                                        |
| `--expect-digest` mismatch in `prove`                | `prove` step 7             | Abort WITHOUT writing the bundle, exit 9 (`ReproMismatch`); print expected vs computed. |
| Unknown `artifact_version`                            | `verify` step 1            | Reject, exit 8 (`ArtifactError`); print the unsupported version and the readable range. |
| Member length/sha256 mismatch (truncation/tamper)    | `verify` open              | Reject, exit 8; name the member.                                     |
| `relation_id` != `--expect-relation`                 | `verify` step 2            | Reject, exit 7 (`VerifyRejected`); `reject_reason: RelationMismatch`.|
| Public-input digest mismatch with bundle index       | `verify` step 3            | Reject, exit 7; `reject_reason: PublicInputDigestMismatch`.          |
| Manifest commitments != `PublicInput` (`--manifest`) | `verify` step 4            | Reject, exit 7; `reject_reason: CommitmentMismatch`.                 |
| Stwo proof invalid                                   | `verify` step 5            | Reject, exit 7; `reject_reason` = the `VerifyError` variant (`docs/spec/04-error-model.md#verifier-rejections`). |
| Claimed-output shape / output-commitment mismatch    | `verify` step 6            | Reject, exit 7; `reject_reason: OutputBindingMismatch`.              |
| `--expect-digest` mismatch in `verify`               | `verify` step 7            | Reject, exit 9 (`ReproMismatch`).                                    |
| Caught panic / broken invariant                      | global handler             | Exit 70 (`InternalError`); print a defect notice, no partial bundle. |

## Alternatives Considered

**A. Bundle as a directory of loose files instead of one `.pwmproof` tar.**
Considered because it is trivially inspectable (`cat proof.bin`) and needs no tar
reader. Rejected: a directory has no atomic-publish primitive (a reader can race a
half-written tree), no single byte-identity unit to digest, and it is awkward to
ship as one artifact through CI and release tooling. A single canonical tar gives
atomic rename, one byte-identity unit, and still trivially unpacks with standard
tools. The cost (a tar reader/writer in `pwm-core`) is small and the writer is a
few hundred lines with fixed USTAR conventions.

**B. Require `proof.bin` itself to be bit-for-bit reproducible.** Considered
because "fully reproducible proof" is a clean slogan and would let CI diff the
whole artifact. Rejected on two grounds. First, it constrains the prover: the
vendored Stwo prover MAY legitimately sample randomness (e.g. in FRI query
folding or blinding), and forcing a fixed PRNG seed into the proof path to get
byte-identity buys nothing soundness-wise (every valid proof already verifies)
while creating a fragile coupling to upstream prover internals we vendor and may
modify (`docs/rfcs/RFC-0015-third-party-vendoring-and-pinning.md`). Second, it is
the wrong reproducibility target: what an auditor needs to re-derive is that the
proof is bound to a specific model and specific inputs producing specific outputs,
which is exactly the determinism digest over inputs and outputs (§3), not the
proof bytes. We therefore lock reproducibility of inputs+outputs and explicitly
declare proof bytes free.

**C. Fold the bundle wire format into RFC-0014 and have this RFC be CLI-only.**
Considered because RFC-0014 already owns canonical serialization. Rejected: the
member-level byte serialization of `PublicInput`, `Proof`, and `Tensor` does
belong to RFC-0014 and this RFC consumes it, but the bundle's container shape
(tar conventions, `index.json`, `artifact_version` gating, atomic write) is an
artifact-packaging concern coupled tightly to the CLI's produce/consume contract
and to the reproducibility digest. Splitting it out would scatter one operator
contract across two RFCs. The chosen boundary is: RFC-0014 owns the bytes inside
each member; RFC-0016 owns the container, the index, and the CLI.

**D. Use exit code 1 for "proof invalid" (conflating it with general error).**
Considered because most CLIs use 1 for any failure. Rejected: an auditor's script
must distinguish "the proof is definitively invalid" (a verdict, exit 7) from "I
could not run the verification" (an error, exit 3-8/70). Collapsing them onto 1
would make a flaky environment indistinguishable from a forged proof, which is a
security-relevant confusion. We reserve a dedicated `VerifyRejected = 7`.

## Drawbacks

- A custom canonical tar writer/reader is new code in the trust base of `verify`.
  Mitigated by keeping it to fixed USTAR conventions with no compression and
  fuzzing it (see Testing Strategy), and by validating member digests before any
  cryptographic step (INV-ART-02).
- The determinism digest is a second commitment beside the public-input digest;
  consumers must understand it covers inputs+outputs, not the proof. Mitigated by
  documenting the scope table in §3 and in `docs/spec/03-data-model.md` and by
  `verify --expect-digest` making the check one command.
- Declaring proof bytes non-reproducible means CI cannot diff `proof.bin` to a
  golden file; it must instead verify the proof and compare the determinism
  digest. This is more work in CI but is the correct, prover-internals-agnostic
  check.
- `created_utc`/`producer` make the raw archive bytes differ run-to-run, so the
  archive itself is not a reproducibility unit; only the digests are. This is an
  intentional trade (provenance metadata over archive byte-identity) and is
  documented so no one writes a test that diffs whole `.pwmproof` files.

## Migration / Rollout

- **Initial landing (v0.2).** `artifact_version = 1`, subcommands `export`,
  `manifest`, `prove`, `verify`. v0.2 ships P0 and P1 statements; `--statement P2`
  is accepted by the CLI grammar but returns exit 5 with `NotYetImplemented`
  until `docs/rfcs/RFC-0009-fixed-candidate-planner-proof.md` lands in v1.0. This
  keeps the command grammar stable across the P2 milestone.
- **Feature flags.** A cargo feature `verifier-only` on `pwm` excludes the prover
  and the Python reference, producing a minimal `verify`-capable binary for
  auditors and for the `no_std`-adjacent verifier path
  (verified against upstream at 2026-06-03: Stwo exposes a `no_std` verifier).
  The prover is behind the default `prover` feature.
- **`artifact_version` evolution.** A new layout (changed member set or index
  schema) increments `artifact_version`. The reader's set of supported versions is
  a closed set per release; an unknown version is exit 8, never a guess. The
  deprecation window for an old `artifact_version` follows
  `docs/spec/09-release-and-versioning.md#deprecation`: at least one minor release
  where both old and new are readable, with a `warn` log on the old one, before
  read support is dropped.
- **Relation versioning is orthogonal.** A new `relation_id`
  (`pwm.lewm.<statement>.vN`) does NOT bump `artifact_version`; the bundle layout
  is relation-agnostic. `verify` simply rejects (exit 7) a relation its build does
  not support, per `docs/rfcs/RFC-0000-security-model-and-statement-taxonomy.md`.
- **Manifest schema versioning is orthogonal.** `manifest_version` changes are
  handled by `pwm export`/`pwm manifest` and
  `docs/rfcs/RFC-0001-model-manifest-and-export-pipeline.md`; the bundle stores
  whatever `PublicInput` commitments resulted, so the bundle layout is unaffected.
- **Determinism-digest versioning.** The digest construction is versioned by its
  domain tag (`pwm.determinism.v1`). A change to what the digest covers mints
  `pwm.determinism.v2` and is a coordinated change with RFC-0014; old and new
  digests never silently compare equal because the domain tag differs.
- **Backward compatibility of the CLI grammar.** Flags are additive within
  `artifact_version` 1 per `docs/spec/02-public-api.md#stability-policy`; removing
  or repurposing a flag requires a deprecation cycle.

## Testing Strategy

Accepting tests (named; cross-referencing
`docs/spec/07-testing-strategy.md#golden-vectors` and `#differential-tests`):

- `cli_export_then_prove_then_verify_p1_exit0` — full pipeline on a golden P1
  fixture; `prove` exits 0, `verify` exits 0, `valid == true`.
- `bundle_roundtrip_byte_identical` — two `prove` runs on identical inputs and a
  pinned toolchain produce `.pwmproof` files whose members (excluding
  `index.json`'s `created_utc`/`producer`) are byte-identical, and whose
  `public_input_digest` and `determinism_digest` are equal (INV-ART-03,
  INV-ART-05).
- `determinism_digest_threads_invariant` — `prove --threads 1` and
  `prove --threads 8` yield the same `determinism_digest` (INV-ART-05).
- `verify_with_manifest_bind_check_passes` — `verify --manifest` recomputes all
  three commitments and matches `PublicInput`; exit 0.
- `verify_expect_digest_match_exit0` — `verify --expect-digest <golden>` matches;
  exit 0.
- `manifest_verify_consistent_exit0` — `pwm manifest verify` on a golden manifest
  recomputes both commitments and matches; exit 0.
- `bundle_index_member_digests_match` — every `index.json` member length/sha256
  matches the actual member (INV-ART-02 positive path).

Rejecting / negative tests (named; cross-referencing
`docs/spec/04-error-model.md#verifier-rejections` and
`docs/spec/07-testing-strategy.md#negative-tests`):

- `verify_unknown_artifact_version_exit8` — bundle with `artifact_version = 99`;
  `verify` exits 8 before any crypto (INV-ART-02).
- `verify_version_fields_disagree_exit8` — typed `artifact_version` and
  `index.json.artifact_version` differ; exit 8 (INV-ART-02).
- `verify_truncated_member_exit8` — a member is truncated so its length/sha256
  mismatch `index.json`; exit 8 (INV-ART-02).
- `verify_tampered_proof_bytes_exit7` — flip a bit in `proof.bin` AND fix its
  index sha256 (so the index check passes); Stwo verification fails; exit 7 with
  `reject_reason` from `#verifier-rejections`.
- `verify_relation_mismatch_exit7` — submit a P0 bundle with
  `--expect-relation pwm.lewm.rollout.v1`; exit 7, `RelationMismatch`
  (mirrors RFC-0000's "reject P0 submitted as P1").
- `verify_commitment_mismatch_exit7` — `--manifest` with a changed weight (so
  `model_commitment` differs); exit 7, `CommitmentMismatch`.
- `verify_public_input_digest_mismatch_exit7` — `index.json.public_input_digest`
  altered; exit 7, `PublicInputDigestMismatch`.
- `verify_output_binding_mismatch_exit7` — claimed-output tensor shape altered;
  exit 7, `OutputBindingMismatch`.
- `prove_overflow_reject_exit5` — inputs forcing an accumulator past its declared
  bound under `OverflowPolicy::Reject`; `prove` exits 5, no bundle written
  (INV-ART-04).
- `prove_weights_commitment_mismatch_exit4` — weights not matching
  `weights.root`; exit 4 (INV-ART-01).
- `prove_expect_digest_mismatch_exit9_no_write` — `--expect-digest` set to a
  wrong value; exit 9 and assert the output path does not exist (INV-ART-05,
  atomic-no-partial).
- `cli_usage_error_exit2` — missing required `--manifest`; exit 2, no work.

Differential / ML-specific tests:

- `python_rust_claimed_outputs_match` — the `pwm export` golden vectors
  (Python fixed-point reference) and the Rust reference inference run inside
  `prove` produce bit-identical claimed outputs; their `determinism_digest`
  components match (`docs/spec/07-testing-strategy.md#differential-tests`).
- `cross_machine_determinism_digest` (CI gate) — the same fixture proved on two
  CI runners with the pinned toolchain yields identical `determinism_digest`;
  enforced as a CI gate per `docs/spec/07-testing-strategy.md#ci-gates`.

Fuzzing:

- `fuzz_pwmproof_reader` — fuzz the tar/`index.json` reader; all inputs either
  parse to a well-formed bundle or exit 8 cleanly; no panic reaches exit 70 in CI
  (`docs/rfcs/RFC-0013-testing-fuzzing-and-audit-strategy.md`).

## Open Questions

- OPEN QUESTION (owner: maintainers / `area:prover`): whether `prove` should
  support streaming very large bundles (claimed outputs for large candidate sets
  in P2) without holding the whole tar in memory. Resolution path: revisit when
  `docs/rfcs/RFC-0009-fixed-candidate-planner-proof.md` lands for v1.0; if the P2
  output set exceeds a memory budget in `docs/spec/08-performance-budget.md#targets`,
  add a chunked-write mode under the same `artifact_version` if it preserves
  byte-identity, else mint `artifact_version = 2`.
- OPEN QUESTION (owner: maintainers / `area:security`): whether the determinism
  digest should additionally bind the pinned-toolchain identity (rust toolchain
  hash + vendored revisions) so a digest match across mismatched toolchains is
  impossible by construction rather than by the contract precondition.
  Resolution path: decide jointly with
  `docs/rfcs/RFC-0015-third-party-vendoring-and-pinning.md` before v1.0; if
  adopted, it mints `pwm.determinism.v2`.

## References

- Founding analysis: `docs/feasibility-study.md` §3.1 (crate layout), §10.3
  (prover flow), §10.4 (verifier flow), §9.2 (determinism requirements), §9.3
  (range-safety / overflow=reject).
- `docs/spec/02-public-api.md#cli`, `#artifact-formats`, `#rust-public-api`,
  `#stability-policy`.
- `docs/spec/03-data-model.md#public-input`, `#witness`, `#tensor-types`,
  `#schema-versioning` (canonical `ProofArtifact`, `PublicInput`, `Tensor`).
- `docs/spec/04-error-model.md#error-taxonomy`, `#failure-modes`,
  `#verifier-rejections`, `#recovery`.
- `docs/spec/05-observability.md#logging`, `#redaction`.
- `docs/spec/06-security.md#trust-boundaries`, `#binding-requirements`,
  `#soundness-requirements`.
- `docs/spec/07-testing-strategy.md#golden-vectors`, `#negative-tests`,
  `#differential-tests`, `#ci-gates`.
- `docs/spec/08-performance-budget.md#targets`.
- `docs/spec/09-release-and-versioning.md#semver`, `#deprecation`, `#msrv`.
- `docs/rfcs/RFC-0000-security-model-and-statement-taxonomy.md` (relation_id
  rule, P0-as-P1 rejection).
- `docs/rfcs/RFC-0001-model-manifest-and-export-pipeline.md` (export pipeline,
  byte-identical re-export).
- `docs/rfcs/RFC-0009-fixed-candidate-planner-proof.md` (P2, v1.0 gating).
- `docs/rfcs/RFC-0013-testing-fuzzing-and-audit-strategy.md` (test stack, fuzzing).
- `docs/rfcs/RFC-0014-canonical-serialization-and-transcript.md` (canonical byte
  serialization of `PublicInput`/`Proof`/`Tensor`, public-input digest, Blake2s
  domain tags, Fiat-Shamir ordering).
- `docs/rfcs/RFC-0015-third-party-vendoring-and-pinning.md` (pinned stwo /
  stwo-circuits revisions; toolchain pinning).
- Stwo (github.com/starkware-libs/stwo, Apache-2.0, workspace v2.2.0): Blake2s /
  Keccak256 / Poseidon252 Fiat-Shamir channel backends and `no_std` verifier path,
  verified against upstream at 2026-06-03.
