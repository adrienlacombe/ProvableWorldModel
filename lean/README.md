# Lean formalization (backlog D-805)

Lean 4 formalization of the ProvableWorldModel soundness arguments. Mirrors the
intent of CommitLLM's `lean/` tree for the parts this project relies on.

| File | Deps | Status |
|---|---|---|
| `ProvableWorldModel/Freivalds.lean` | none (core Lean) | **Proved, no `sorry`.** The `verify_argmin` soundness contract (`IsArgmin`) and `argmin_unique` (the accepted selection is unique). |
| `ProvableWorldModel/FreivaldsBound.lean` | Mathlib | **Proved, no `sorry`.** The Freivalds probability bound: for `d ≠ 0` over a finite field `F`, exactly `|F|^(n-1)` challenges accept a wrong claim (`freivalds_accept_card`), so `|F| · #accept = |F|ⁿ` — accept probability exactly `1/|F|` (`freivalds_accept_mul_card`). |

Both soundness pillars behind `pwm-core` (`planning::verify_argmin` and
`freivalds`) are now formally verified.

## Build

Toolchain is pinned by `lean-toolchain` (`leanprover/lean4:v4.30.0`).

```bash
cd lean
lake exe cache get      # one-time: download prebuilt Mathlib oleans (~8 GB)
lake build              # builds Freivalds + FreivaldsBound + the library root
```

`Freivalds.lean` is Mathlib-free and also compiles standalone:

```bash
lean lean/ProvableWorldModel/Freivalds.lean   # no Mathlib, no warnings
```

`lake-manifest.json` pins the exact Mathlib revision; `.lake/` (build artifacts +
the fetched Mathlib clone) is gitignored.
