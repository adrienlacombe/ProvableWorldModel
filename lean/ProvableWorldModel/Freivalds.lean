-- SPDX-License-Identifier: Apache-2.0
/-!
# Soundness statements (formalization skeleton — backlog D-805)

Self-contained (no Mathlib): the argmin-selection soundness behind
`pwm_core::planning::verify_argmin` and `topk`. The other soundness pillar — the
Freivalds probability bound (`pwm_core::freivalds`) — is stated in prose below; its
proof needs a finite-field counting development (Mathlib `ZMod p`, `Fintype.card`),
which is the remaining formalization work and is why it is not in this file.

Freivalds soundness (target): over a field `F` with `|F| = p`, for `W : Fin m → Fin
n → F` and vectors `x z`, if `z ≠ W·x` then `#{ r | r ⬝ z = r ⬝ (W·x) } ≤ |F|^(m-1)`
— the accepting challenges form the kernel of the nonzero linear form
`r ↦ r ⬝ (z − W·x)`, so the accept probability is `≤ 1/p`.
-/
namespace ProvableWorldModel

/-- The `verify_argmin` contract: `sel` selects cost `c`, no candidate is cheaper,
and no strictly-earlier index ties it (smallest-index tie-break). -/
def IsArgmin (costs : List Int) (sel : Nat) (c : Int) : Prop :=
  costs[sel]? = some c ∧
    (∀ (j : Nat) (cj : Int), costs[j]? = some cj → c ≤ cj) ∧
    (∀ (j : Nat) (cj : Int), j < sel → costs[j]? = some cj → c < cj)

/-- Soundness: the argmin is unique. If two indices both satisfy the contract they
are equal — so a verifier that accepts the contract pins down exactly one
selection (a wrong selection cannot also satisfy it). Proof pending (D-805). -/
theorem argmin_unique {costs : List Int} {i j : Nat} {ci cj : Int}
    (_hi : IsArgmin costs i ci) (_hj : IsArgmin costs j cj) : i = j := by
  sorry

end ProvableWorldModel
