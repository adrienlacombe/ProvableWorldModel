# Vendored: Stwo

This directory is a **pristine, pinned vendoring** of StarkWare's Stwo prover.
The pin and audit policy are governed by
[RFC-0015](../../docs/rfcs/RFC-0015-third-party-vendoring-and-pinning.md); the
machine-readable pin is in [`REVISION`](REVISION).

| Field | Value |
| --- | --- |
| Upstream | https://github.com/starkware-libs/stwo |
| Tag | `v2.2.0` |
| Commit | `289c20de80b7c7f508de9c46151fb81dae404154` |
| License | Apache-2.0 (see [`LICENSE`](LICENSE)) |
| Vendored | 2026-06-03 |

## Vendoring procedure

The tree was vendored verbatim from the pinned commit:

```sh
git clone --branch v2.2.0 https://github.com/starkware-libs/stwo.git
# copy the working tree minus .git and target/ into third_party/stwo/
```

It is **unmodified** from upstream at the pinned commit — there are no local
patches (`PATCHES/` is empty; `patches_sha256` is the hash of the empty set).
`tree_sha256` in `REVISION` is reproducible from a fresh checkout of the pinned
commit by hashing the sorted per-file SHA-256 of every tracked file except this
crate's RFC-management files (`REVISION`, `UPSTREAM.md`, `NOTICE`, `PATCHES/`).

## Wiring

First-party crates depend on the vendored packages **by path only** — never a
crates.io or git source (INV-RFC0015-01). The Stwo packages and their vendored
paths:

| Package | Path |
| --- | --- |
| `stwo` | `third_party/stwo/crates/stwo` |
| `stwo-constraint-framework` | `third_party/stwo/crates/constraint-framework` |
| `stwo-air-utils` | `third_party/stwo/crates/air-utils` |
| `stwo-air-utils-derive` | `third_party/stwo/crates/air-utils-derive` |
| `stwo-std-shims` | `third_party/stwo/crates/std-shims` |

The vendored tree is its own Cargo workspace and is `exclude`d from the
first-party workspace (root `Cargo.toml`), so it keeps its own manifest and
lockfile while remaining resolvable as a path dependency.

## Exclusions and notes

- Nothing is excluded from the vendored tree: the full upstream working tree at
  the pinned commit is preserved (including `crates/examples`, benchmarks, and
  scripts), so the tree matches upstream exactly and the audit surface is the
  full diff. `crates/examples` is not in any first-party dependency path, so it
  is never compiled by the first-party build.
- Upstream ships no `NOTICE` file; see [`NOTICE`](NOTICE) for the recorded
  attribution.
- Toolchain: Stwo requires nightly Rust; the project pins `nightly-2025-07-14`
  (RFC-0015 toolchain decision), which matches upstream's own
  `rust-toolchain.toml` at this tag.

## Revision bumps

A bump re-runs the procedure at a new commit, updates `REVISION` (including
`tree_sha256`), records any local modifications as numbered `PATCHES/`, and
follows the revision-bump + audit workflow in RFC-0015.
