-- SPDX-License-Identifier: Apache-2.0
import ProvableWorldModel.Freivalds
import ProvableWorldModel.FreivaldsBound

/-!
# ProvableWorldModel soundness formalization (backlog D-805)

Library root. `Freivalds` is self-contained (no Mathlib) and also compiles
standalone; `FreivaldsBound` uses Mathlib for the finite-field counting behind the
Freivalds probability bound.
-/
