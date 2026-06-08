-- SPDX-License-Identifier: Apache-2.0
import Mathlib

/-!
# Freivalds soundness bound (backlog D-805)

The probability bound behind `pwm_core::freivalds`: over a finite field `F`, a
uniform challenge `r ∈ Fⁿ` accepts a **wrong** matmul claim with probability at
most `1/|F|`.

The Freivalds check `r · z = r · (W·x)` is `∑ i, d i * r i = 0` with
`d = z − W·x`. When the claim is wrong, `d ≠ 0`, and the accepting challenges are
exactly the kernel of the nonzero linear form `r ↦ ∑ i, d i * r i` — a hyperplane
of size `|F|^(n-1)`. Hence `|F| · #accept = |F|ⁿ`: the accept fraction is exactly
`1/|F|`, so soundness error `≤ 1/|F|` (and `≤ k/|F|` over `k` independent rows).
-/

open scoped BigOperators

namespace ProvableWorldModel

variable {F : Type*} [Field F]

/-- The dot-product-with-`d` linear form `r ↦ ∑ i, d i * r i`. -/
noncomputable def dotForm {n : ℕ} (d : Fin n → F) : (Fin n → F) →ₗ[F] F :=
  ∑ i, (d i) • LinearMap.proj i

@[simp] lemma dotForm_apply {n : ℕ} (d r : Fin n → F) :
    dotForm d r = ∑ i, d i * r i := by
  simp only [dotForm, LinearMap.coe_sum, Finset.sum_apply, LinearMap.smul_apply,
    LinearMap.proj_apply, smul_eq_mul]

/-- A nonzero coefficient vector gives a nonzero form. -/
lemma dotForm_ne_zero {n : ℕ} {d : Fin n → F} (hd : d ≠ 0) : dotForm d ≠ 0 := by
  intro h
  apply hd
  funext j
  have hcongr := LinearMap.congr_fun h (Pi.single j 1)
  simp only [dotForm_apply, LinearMap.zero_apply] at hcongr
  rw [Finset.sum_eq_single j (fun i _ hij => by simp [Pi.single_eq_of_ne hij]) (by simp)] at hcongr
  simpa using hcongr

/-- The range of a nonzero form into the field `F` is everything (so it is
surjective): pick `r` with `dotForm d r ≠ 0`, then scale it to hit any `y`. -/
lemma dotForm_range_eq_top {n : ℕ} {d : Fin n → F} (hd : d ≠ 0) :
    LinearMap.range (dotForm d) = ⊤ := by
  obtain ⟨r, hr⟩ : ∃ r, dotForm d r ≠ 0 := by
    by_contra hcon
    apply dotForm_ne_zero hd
    refine LinearMap.ext fun v => ?_
    have hv : ¬ (dotForm d v ≠ 0) := fun h => hcon ⟨v, h⟩
    simpa using not_not.mp hv
  rw [LinearMap.range_eq_top]
  intro y
  refine ⟨(y * (dotForm d r)⁻¹) • r, ?_⟩
  rw [map_smul, smul_eq_mul, mul_assoc, inv_mul_cancel₀ hr, mul_one]

variable [Fintype F] [DecidableEq F]

/-- **Freivalds accept count.** For `d ≠ 0`, exactly `|F|^(n-1)` challenges accept
the wrong claim — the kernel of the nonzero linear form `dotForm d`. -/
theorem freivalds_accept_card {n : ℕ} {d : Fin n → F} (hd : d ≠ 0) :
    Fintype.card {r : Fin n → F // ∑ i, d i * r i = 0}
      = Fintype.card F ^ (n - 1) := by
  classical
  haveI : Fintype (LinearMap.ker (dotForm d)) := Fintype.ofFinite _
  -- The accept subtype is exactly the kernel of `dotForm d`.
  have hset : Fintype.card {r : Fin n → F // ∑ i, d i * r i = 0}
      = Fintype.card (LinearMap.ker (dotForm d)) :=
    Fintype.card_congr
      (Equiv.subtypeEquivRight fun r => by simp [LinearMap.mem_ker, dotForm_apply])
  rw [hset, Module.card_eq_pow_finrank (K := F)]
  congr 1
  -- finrank of the kernel is `n - 1`, by rank–nullity.
  have htop : Module.finrank F (Fin n → F) = n := by
    rw [Module.finrank_fintype_fun_eq_card, Fintype.card_fin]
  have hrange : Module.finrank F (LinearMap.range (dotForm d)) = 1 := by
    rw [dotForm_range_eq_top hd, finrank_top]
    exact CommSemiring.finrank_self F
  have hrn := (dotForm d).finrank_range_add_finrank_ker
  rw [hrange, htop] at hrn
  omega

/-- The accept fraction is exactly `1/|F|`: `|F| · #accept = |F|ⁿ` (for `n ≥ 1`),
so the single-row Freivalds soundness error is `≤ 1/|F|`. -/
theorem freivalds_accept_mul_card {n : ℕ} (hn : 1 ≤ n) {d : Fin n → F} (hd : d ≠ 0) :
    Fintype.card F * Fintype.card {r : Fin n → F // ∑ i, d i * r i = 0}
      = Fintype.card F ^ n := by
  rw [freivalds_accept_card hd, ← pow_succ']
  congr 1
  omega

end ProvableWorldModel
