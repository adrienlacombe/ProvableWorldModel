// SPDX-License-Identifier: Apache-2.0
//! Precomputed Freivalds verification of linear layers (specs.md §7).
//!
//! Key identity: for `z = W·x`, instead of recomputing `W·x` the verifier checks
//! `v · x == r · z` in the audit field `F_p` (`p = 2^61 - 1`, [`crate::field::Fp61`]),
//! where `v = rᵀW` is **precomputed once per weight matrix** and reused across
//! every instance that applies the same `W` (candidate × rollout step × block).
//! Because `rᵀ(Wx) = (rᵀW)x = v·x`, an honest `z` always passes; a wrong
//! `z ≠ Wx` passes with probability ≤ `1/p` per check.
//!
//! The input `x` is the int8/int16 activation; `z` is the **i32 accumulator**
//! (not requantized — requant is verified separately by the trace's `Requant`
//! record, specs.md §8.1). The challenge `r` is squeezed from the Fiat-Shamir
//! transcript after the trace commitment (non-interactive), or kept
//! verifier-secret (interactive mode, V1).
//!
//! This is a pure-integer, `no_std` re-implementation of the CommitLLM
//! `verilm-core::freivalds` kernel in the project's audit field.

use alloc::vec::Vec;

use crate::field::{Fp61, FREIVALDS_P, P_HALF};

/// Soundness bound on every Freivalds operand (specs.md §7, RFC-0002 §7 / INV-FP-10).
///
/// Freivalds checks `v·x == r·z` in `F_p` (`p = 2^61 − 1`), which only proves
/// `z ≡ Wx (mod p)` — **not** `z = Wx` over the integers. Without a range check a
/// prover can submit `z' = Wx + k·p`: for *every* challenge `r`, `r·z' ≡ r·Wx
/// (mod p)`, so the check passes with probability 1 and the forged accumulator
/// propagates into a different accepted output (the mod-`p` aliasing forgery).
///
/// The fix is to bound every operand (input `x`, bias, claimed accumulator `z`) so
/// the true product `Wx + bias` and the claimed `z` lie within one length-`p`
/// window (`|z − (Wx+bias)| < p`), which makes congruence imply integer equality.
/// V0 accumulators fit a single signed M31 (`SAFE_HI = P_HALF`); the widest V0 dot
/// product (the `mlp_dim = 2048` int8 MLP) peaks at `2048·127·127 ≈ 2^25 ≪ P_HALF`,
/// so this bound never rejects an honest proof.
pub const FREIVALDS_OPERAND_BOUND: i64 = P_HALF;

/// True iff every element of `vals` lies in `[−FREIVALDS_OPERAND_BOUND, +FREIVALDS_OPERAND_BOUND]`.
pub fn within_operand_bound(vals: &[i64]) -> bool {
    vals.iter()
        .all(|&v| (-FREIVALDS_OPERAND_BOUND..=FREIVALDS_OPERAND_BOUND).contains(&v))
}

/// Static soundness margin for a linear op of input dimension `cols` with int8
/// weights (`|W| ≤ 127`): with every operand bounded by `B = FREIVALDS_OPERAND_BOUND`
/// the true accumulator satisfies `|Wx + bias| ≤ 127·cols·B + B`, and the claimed
/// `z` satisfies `|z| ≤ B`, so `|z − (Wx+bias)| ≤ 127·cols·B + 2·B`. This must stay
/// `< p` for congruence to imply equality; the check guards against a pathologically
/// wide layer where even bounded operands could alias across a multiple of `p`.
pub fn dims_within_soundness_margin(cols: usize) -> bool {
    let b = FREIVALDS_OPERAND_BOUND as i128;
    // |z − (Wx+bias)| ≤ 127·cols·B + 2·B  must be < p.
    127i128 * (cols as i128) * b + 2 * b < FREIVALDS_P as i128
}

/// `Σ_i coeffs[i] · vals[i]` over `F_p`, with `vals` an i64 slice. Trace records
/// store activation/accumulator vectors uniformly as `i64`, so this is the
/// general dot used by the verifier's Freivalds checks.
pub(crate) fn dot_fp_i64(coeffs: &[Fp61], vals: &[i64]) -> Fp61 {
    debug_assert_eq!(coeffs.len(), vals.len());
    let mut acc = Fp61::ZERO;
    for (&c, &z) in coeffs.iter().zip(vals.iter()) {
        acc = acc.add(c.mul(Fp61::from_i64(z)));
    }
    acc
}

/// Verify a linear op with bias, `out = W·x + bias`, in one Freivalds equation:
/// `r·out == v·x + r·bias`, where `v = rᵀW`. All vectors are `i64`. This is the
/// check the verifier runs per `Linear` trace record (specs.md §7).
///
/// Returns `false` (reject) unless the **soundness range guard** also holds: every
/// input, bias, and claimed accumulator must lie within [`FREIVALDS_OPERAND_BOUND`]
/// and the layer width must satisfy [`dims_within_soundness_margin`]. This guard is
/// what makes the mod-`p` Freivalds check sound over the integers — without it a
/// prover can add a multiple of `p` to an accumulator and pass with probability 1.
/// Folding it in here makes every caller (flat trace, block, batched) sound by
/// construction; a caller cannot forget it.
pub fn check_linear_biased(v: &[Fp61], x: &[i64], bias: &[i64], r: &[Fp61], out: &[i64]) -> bool {
    if !dims_within_soundness_margin(x.len())
        || !within_operand_bound(x)
        || !within_operand_bound(bias)
        || !within_operand_bound(out)
    {
        return false;
    }
    let lhs = dot_fp_i64(v, x).add(dot_fp_i64(r, bias));
    let rhs = dot_fp_i64(r, out);
    lhs == rhs
}

/// Precompute `v = rᵀ W` over `F_p`.
///
/// `weight` is row-major `W[row * cols + col]`, shape `(rows, cols)`. `r` has
/// length `rows` (the output dimension); the returned `v` has length `cols` (the
/// input dimension). Cost `O(rows·cols)`, paid **once** per weight matrix.
///
/// # Panics
///
/// Panics if `r.len() != rows` or `weight.len() != rows * cols` (a caller-side
/// shape error, not attacker input).
pub fn precompute_v(r: &[Fp61], weight: &[i8], rows: usize, cols: usize) -> Vec<Fp61> {
    assert_eq!(r.len(), rows, "r length must equal the output dimension");
    assert_eq!(weight.len(), rows * cols, "weight size must be rows * cols");

    let mut v = alloc::vec![Fp61::ZERO; cols];
    for (row, &ri) in r.iter().enumerate() {
        let base = row * cols;
        for (col, vc) in v.iter_mut().enumerate() {
            let w = Fp61::from_i64(weight[base + col] as i64);
            *vc = vc.add(ri.mul(w));
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r3() -> Vec<Fp61> {
        alloc::vec![Fp61::new(10), Fp61::new(20), Fp61::new(30)]
    }

    /// No-bias Freivalds check through the real (range-guarded) API.
    fn check0(v: &[Fp61], x: &[i64], r: &[Fp61], out: &[i64]) -> bool {
        let bias = alloc::vec![0i64; out.len()];
        check_linear_biased(v, x, &bias, r, out)
    }

    #[test]
    fn accepts_correct_output() {
        // W = [[1,2],[3,4],[5,6]], x = [7,8]; z = W·x = [23, 53, 83].
        let w: [i8; 6] = [1, 2, 3, 4, 5, 6];
        let r = r3();
        let v = precompute_v(&r, &w, 3, 2);
        assert!(check0(&v, &[7, 8], &r, &[23, 53, 83]));
    }

    #[test]
    fn rejects_wrong_output() {
        let w: [i8; 6] = [1, 2, 3, 4, 5, 6];
        let r = r3();
        let v = precompute_v(&r, &w, 3, 2);
        assert!(!check0(&v, &[7, 8], &r, &[23, 53, 84])); // 84 != 83
    }

    #[test]
    fn handles_negative_weights() {
        // W = [[-1,2],[3,-4]], x=[3,7]; z = [-1*3+2*7, 3*3-4*7] = [11, -19].
        let w: [i8; 4] = [-1, 2, 3, -4];
        let r = alloc::vec![Fp61::new(5), Fp61::new(10)];
        let v = precompute_v(&r, &w, 2, 2);
        assert!(check0(&v, &[3, 7], &r, &[11, -19]));
    }

    #[test]
    fn identity_matrix() {
        let w: [i8; 9] = [1, 0, 0, 0, 1, 0, 0, 0, 1];
        let r = alloc::vec![Fp61::new(42), Fp61::new(99), Fp61::new(7)];
        let v = precompute_v(&r, &w, 3, 3);
        assert!(check0(&v, &[10, 20, 30], &r, &[10, 20, 30]));
    }

    /// Deterministic LCG for seeded pseudo-random fuzzing (no_std, no rand dep).
    fn lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state
    }

    /// Soundness fuzz (T-703): for many random (W, x, r), the correct `z = W·x`
    /// always passes, and corrupting any single output element always fails.
    #[test]
    fn freivalds_soundness_fuzz() {
        let mut s: u64 = 0x1234_5678_9abc_def0;
        for _ in 0..200 {
            let rows = 1 + (lcg(&mut s) % 6) as usize;
            let cols = 1 + (lcg(&mut s) % 6) as usize;
            let w: Vec<i8> = (0..rows * cols)
                .map(|_| (lcg(&mut s) % 255) as i64 as i8)
                .collect();
            let x: Vec<i64> = (0..cols)
                .map(|_| (lcg(&mut s) % 255) as i64 as i8 as i64)
                .collect();
            let r: Vec<Fp61> = (0..rows).map(|_| Fp61::new(lcg(&mut s))).collect();
            // Correct z = W·x (well within the single-M31 accumulator envelope).
            let z: Vec<i64> = (0..rows)
                .map(|i| (0..cols).map(|j| w[i * cols + j] as i64 * x[j]).sum())
                .collect();
            let v = precompute_v(&r, &w, rows, cols);
            assert!(check0(&v, &x, &r, &z), "correct z must verify");
            // Corrupt one element: must be rejected (challenge is random, so the
            // 1/p escape probability is negligible across these cases).
            let k = (lcg(&mut s) as usize) % rows;
            let mut bad = z.clone();
            bad[k] = bad[k].wrapping_add(1);
            assert!(!check0(&v, &x, &r, &bad), "corrupted z must be rejected");
        }
    }
}
