# Third-party dependency and license inventory

The authoritative inventory of vendored components and their license surface,
per [RFC-0015](../docs/rfcs/RFC-0015-third-party-vendoring-and-pinning.md). The
`cargo deny check licenses` gate ([`deny.toml`](../deny.toml)) enforces the
allowed-license set against the resolved dependency graph on every PR.

## Vendored components

| Component | Path | Pin | License | REVISION |
| --- | --- | --- | --- | --- |
| Stwo | `third_party/stwo` | `v2.2.0` (`289c20de`) | Apache-2.0 | [`stwo/REVISION`](stwo/REVISION) |
| stwo-circuits | `third_party/stwo-circuits` | `v0.1.0` (`b0db13e4`) | Apache-2.0 | pending (#25) |

Each vendored component is pinned by commit, carries its upstream `LICENSE`
verbatim and a recorded `NOTICE`, and is referenced by path only (INV-RFC0015-01:
no crates.io/git source for a substrate package).

## License surface of the resolved graph (Stwo vendored)

Licenses present in the transitive dependency tree once `stwo` is wired in. Every
crate resolves to at least one allowed license (dual-licensed crates list every
option; the allowed alternative satisfies the gate).

| SPDX license | Allowed | Notes |
| --- | --- | --- |
| Apache-2.0 | yes | project license; most of the tree |
| MIT | yes | most of the tree (dual with Apache-2.0) |
| BSD-2-Clause | yes | `arrayref` (single), `zerocopy` (dual) |
| BSD-3-Clause | yes | `subtle` |
| Unicode-3.0 | yes | `unicode-ident` (dual) |
| Zlib | yes | `foldhash` (single), `bytemuck` (dual) |
| Apache-2.0 WITH LLVM-exception | via Apache-2.0 | `blake3`, `wasi` (dual with Apache-2.0) |
| CC0-1.0 / MIT-0 / Unlicense | via alternative | `blake3`, `constant_time_eq`, `memchr` — each dual with an allowed license |

The allow-list lives in [`deny.toml`](../deny.toml). Duplicate dependency
versions (e.g. `itertools` 0.10/0.12, `hashbrown` 0.14/0.17, `syn` 1/2) are
permitted (`bans.multiple-versions = "allow"`) because they originate in the
pinned upstream tree; tightening is tracked by the vendoring-audit work (#26).
