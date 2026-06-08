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
selection (a wrong selection cannot also satisfy it). -/
theorem argmin_unique {costs : List Int} {i j : Nat} {ci cj : Int}
    (hi : IsArgmin costs i ci) (hj : IsArgmin costs j cj) : i = j := by
  obtain ⟨hi_get, hi_min, hi_tie⟩ := hi
  obtain ⟨hj_get, hj_min, hj_tie⟩ := hj
  -- Costs at i and j are equal (each ≤ the other).
  have hcc : ci = cj := Int.le_antisymm (hi_min j cj hj_get) (hj_min i ci hi_get)
  rcases Nat.lt_trichotomy i j with hlt | heq | hgt
  · -- i < j: tie-break at j forces cj < ci = cj, impossible.
    have h : cj < ci := hj_tie i ci hlt hi_get
    rw [hcc] at h
    exact absurd h (Int.lt_irrefl cj)
  · exact heq
  · -- j < i: tie-break at i forces ci < cj = ci, impossible.
    have h : ci < cj := hi_tie j cj hgt hj_get
    rw [hcc] at h
    exact absurd h (Int.lt_irrefl cj)

end ProvableWorldModel
