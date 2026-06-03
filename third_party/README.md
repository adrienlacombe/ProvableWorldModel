<!-- SPDX-License-Identifier: Apache-2.0 -->
# Vendored third-party dependencies

ProvableWorldModel vendors its proving substrate at pinned revisions rather than
depending on the upstream canonical crates, to protect the audit boundary and
the soundness-critical Fiat-Shamir/PCS internals from upstream drift. The
vendoring, modification, and re-audit policy is governed by
[RFC-0015](../docs/rfcs/RFC-0015-third-party-vendoring-and-pinning.md).

| Tree | Upstream | License | Pinned shape |
| --- | --- | --- | --- |
| `stwo/` | github.com/starkware-libs/stwo | Apache-2.0 | Workspace v2.2.0 — Circle-STARK prover/verifier, M31 field module, FRI + PCS, Fiat-Shamir channels, LogUp. |
| `stwo-circuits/` | github.com/starkware-libs/stwo-circuits | Apache-2.0 | v0.1.0 (edition 2024) — the low-level scalar gate set. |

The authoritative TCB / license inventory is [`INVENTORY.md`](INVENTORY.md).

## Status

- **`stwo/`** — vendored (#24). Pristine at `v2.2.0`
  (`289c20de80b7c7f508de9c46151fb81dae404154`); see [`stwo/REVISION`](stwo/REVISION)
  and [`stwo/UPSTREAM.md`](stwo/UPSTREAM.md). First-party crates depend on it by
  path; the tree is its own workspace and is `exclude`d from the first-party
  workspace.
- **`stwo-circuits/`** — not vendored yet (#25); `REVISION` holds the placeholder
  `pending-rfc-0015`.

Vendored files retain their upstream license headers unmodified; the first-party
SPDX header lint excludes `third_party/`
(see [docs/spec/09-release-and-versioning.md](../docs/spec/09-release-and-versioning.md)).
