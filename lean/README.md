# Lean formalization (backlog D-805)

Lean 4 formalization of the ProvableWorldModel soundness arguments. Mirrors the
intent of CommitLLM's `lean/` tree for the parts this project relies on.

| File | Status |
|---|---|
| `ProvableWorldModel/Freivalds.lean` | **Compiles, no `sorry`** (self-contained, no Mathlib). States the `verify_argmin` soundness contract (`IsArgmin`) and **proves** `argmin_unique` (the accepted selection is unique); documents the Freivalds probability-bound theorem to be proved next. |

Build (Lean toolchain pinned by `lean-toolchain`):

```bash
lean lean/ProvableWorldModel/Freivalds.lean   # compiles cleanly, no warnings
```

Remaining formalization work:
- The **Freivalds soundness bound** (`#{r | r·z = r·(W·x)} ≤ |F|^(m-1)` when
  `z ≠ W·x`) needs a finite-field counting development — `ZMod p`, `Fintype.card`,
  nonzero-linear-functional kernel size — i.e. a Mathlib dependency (a `lakefile`
  pulling Mathlib). That is the substantive research step.
