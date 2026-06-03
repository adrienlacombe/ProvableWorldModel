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

## Status

This is the workspace skeleton (issue #23). The actual upstream code is **not**
vendored yet; that is the work of issues #24 (`stwo`) and #25 (`stwo-circuits`).
Each subtree carries a `REVISION` file recording the exact pinned commit. Until
the revisions are frozen during the `v0.1 — Foundations` milestone, both files
contain the placeholder string `pending-rfc-0015`.

Vendored files retain their upstream license headers unmodified; the first-party
SPDX header lint excludes `third_party/`
(see [docs/spec/09-release-and-versioning.md](../docs/spec/09-release-and-versioning.md)).
