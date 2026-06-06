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

use crate::field::Fp61;

/// `Σ_i coeffs[i] · vals[i]` over `F_p`, with `vals` an int8 slice.
pub fn dot_fp_i8(coeffs: &[Fp61], vals: &[i8]) -> Fp61 {
    debug_assert_eq!(coeffs.len(), vals.len());
    let mut acc = Fp61::ZERO;
    for (&c, &x) in coeffs.iter().zip(vals.iter()) {
        acc = acc.add(c.mul(Fp61::from_i64(x as i64)));
    }
    acc
}

/// `Σ_i coeffs[i] · vals[i]` over `F_p`, with `vals` an i32 slice (int16
/// activations or i32 accumulators).
pub fn dot_fp_i32(coeffs: &[Fp61], vals: &[i32]) -> Fp61 {
    debug_assert_eq!(coeffs.len(), vals.len());
    let mut acc = Fp61::ZERO;
    for (&c, &z) in coeffs.iter().zip(vals.iter()) {
        acc = acc.add(c.mul(Fp61::from_i64(z as i64)));
    }
    acc
}

/// `Σ_i coeffs[i] · vals[i]` over `F_p`, with `vals` an i64 slice. Trace records
/// store activation/accumulator vectors uniformly as `i64`, so this is the
/// general dot used by the verifier's Freivalds checks.
pub fn dot_fp_i64(coeffs: &[Fp61], vals: &[i64]) -> Fp61 {
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
pub fn check_linear_biased(v: &[Fp61], x: &[i64], bias: &[i64], r: &[Fp61], out: &[i64]) -> bool {
    let lhs = dot_fp_i64(v, x).add(dot_fp_i64(r, bias));
    let rhs = dot_fp_i64(r, out);
    lhs == rhs
}

/// Precompute `v = rᵀ W` over `F_p`.
///
/// `weight` is row-major `W[row * cols + col]`, shape `(rows, cols)`. `r` has
/// length `rows` (the output dimension); the returned `v` has length `cols` (the
/// input dimension). Cost `O(rows·cols)`, paid **once** per weight matrix.
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

/// Verify one matmul instance `z =? W·x`: returns true iff `v · x == r · z`.
///
/// `v = precompute_v(r, W, rows, cols)` (length `cols`), `x` is the int8 input
/// (length `cols`), `r` is the challenge (length `rows`), `z` is the claimed i32
/// accumulator output (length `rows`).
pub fn check(v: &[Fp61], x: &[i8], r: &[Fp61], z: &[i32]) -> bool {
    dot_fp_i8(v, x) == dot_fp_i32(r, z)
}

/// Like [`check`] but with an int16/i32 input `x` (int16 activations).
pub fn check_i32(v: &[Fp61], x: &[i32], r: &[Fp61], z: &[i32]) -> bool {
    dot_fp_i32(v, x) == dot_fp_i32(r, z)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r3() -> Vec<Fp61> {
        alloc::vec![Fp61::new(10), Fp61::new(20), Fp61::new(30)]
    }

    #[test]
    fn accepts_correct_output() {
        // W = [[1,2],[3,4],[5,6]], x = [7,8]; z = W·x = [23, 53, 83].
        let w: [i8; 6] = [1, 2, 3, 4, 5, 6];
        let x: [i8; 2] = [7, 8];
        let z: [i32; 3] = [23, 53, 83];
        let r = r3();
        let v = precompute_v(&r, &w, 3, 2);
        assert!(check(&v, &x, &r, &z));
    }

    #[test]
    fn rejects_wrong_output() {
        let w: [i8; 6] = [1, 2, 3, 4, 5, 6];
        let x: [i8; 2] = [7, 8];
        let z: [i32; 3] = [23, 53, 84]; // 84 != 83
        let r = r3();
        let v = precompute_v(&r, &w, 3, 2);
        assert!(!check(&v, &x, &r, &z));
    }

    #[test]
    fn handles_negative_weights() {
        // W = [[-1,2],[3,-4]], x=[3,7]; z = [-1*3+2*7, 3*3-4*7] = [11, -19].
        let w: [i8; 4] = [-1, 2, 3, -4];
        let x: [i8; 2] = [3, 7];
        let z: [i32; 2] = [11, -19];
        let r = alloc::vec![Fp61::new(5), Fp61::new(10)];
        let v = precompute_v(&r, &w, 2, 2);
        assert!(check(&v, &x, &r, &z));
    }

    #[test]
    fn identity_matrix() {
        let w: [i8; 9] = [1, 0, 0, 0, 1, 0, 0, 0, 1];
        let x: [i8; 3] = [10, 20, 30];
        let z: [i32; 3] = [10, 20, 30];
        let r = alloc::vec![Fp61::new(42), Fp61::new(99), Fp61::new(7)];
        let v = precompute_v(&r, &w, 3, 3);
        assert!(check(&v, &x, &r, &z));
    }

    /// Deterministic LCG for seeded pseudo-random fuzzing (no_std, no rand dep).
    fn lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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
            let w: alloc::vec::Vec<i8> = (0..rows * cols)
                .map(|_| (lcg(&mut s) % 255) as i64 as i8)
                .collect();
            let x: alloc::vec::Vec<i8> = (0..cols)
                .map(|_| (lcg(&mut s) % 255) as i64 as i8)
                .collect();
            let r: alloc::vec::Vec<Fp61> = (0..rows).map(|_| Fp61::new(lcg(&mut s))).collect();
            // Correct z = W·x.
            let z: alloc::vec::Vec<i32> = (0..rows)
                .map(|i| {
                    (0..cols)
                        .map(|j| w[i * cols + j] as i32 * x[j] as i32)
                        .sum()
                })
                .collect();
            let v = precompute_v(&r, &w, rows, cols);
            assert!(check(&v, &x, &r, &z), "correct z must verify");
            // Corrupt one element: must be rejected (challenge is random, so the
            // 1/p escape probability is negligible across these cases).
            let k = (lcg(&mut s) as usize) % rows;
            let mut bad = z.clone();
            bad[k] = bad[k].wrapping_add(1);
            assert!(!check(&v, &x, &r, &bad), "corrupted z must be rejected");
        }
    }
}
