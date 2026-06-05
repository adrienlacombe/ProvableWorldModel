// SPDX-License-Identifier: Apache-2.0
//! Exact integer op kernels for the LeWorldModel predictor (specs.md §2.1, §8).
//!
//! These are the verifiable integer semantics of the predictor's elementwise and
//! nonlinear ops — the kernels both the prover (to build the trace) and the
//! verifier (to exactly recompute) call. They are pure, deterministic, and
//! `no_std`: nonlinearities (`LayerNorm` inverse-sqrt, `Softmax` exp) are realized
//! as committed integer lookup tables ([`crate::tables::ActivationTable`]), so the
//! verifier replays them with no floating point and no approximation gap.
//!
//! The full predictor wires these into AdaLN-zero conditional blocks (attention +
//! gated FFN) over a named-buffer trace; this module locks the per-op math.

use alloc::vec::Vec;

use crate::fixed_point::{requantize, Rounding};
use crate::tables::ActivationTable;

/// Round-to-nearest integer division by a positive divisor `n` (ties away from
/// zero). Used for the LayerNorm mean/variance reductions.
pub fn round_div(a: i64, n: i64) -> i64 {
    debug_assert!(n > 0);
    if a >= 0 {
        (a + n / 2) / n
    } else {
        -((-a + n / 2) / n)
    }
}

/// AdaLN-zero modulation, per channel: `out = requant(x · (one + scale)) + shift`
/// (le-wm `modulate(x, shift, scale) = x*(1+scale)+shift`, in fixed point). `one`
/// is the fixed-point unit at `scale`'s scale; `shift_bits` rescales the product.
#[allow(clippy::too_many_arguments)]
pub fn modulate(
    x: i64,
    scale: i64,
    shift: i64,
    one: i64,
    shift_bits: u32,
    clamp_lo: i64,
    clamp_hi: i64,
    mode: Rounding,
) -> i64 {
    let prod = x * (one + scale);
    requantize(prod, shift_bits, 0, clamp_lo, clamp_hi, mode) + shift
}

/// Per-channel AdaLN modulation over a vector.
#[allow(clippy::too_many_arguments)]
pub fn modulate_vec(
    x: &[i64],
    scale: &[i64],
    shift: &[i64],
    one: i64,
    shift_bits: u32,
    clamp_lo: i64,
    clamp_hi: i64,
    mode: Rounding,
) -> Vec<i64> {
    x.iter()
        .zip(scale.iter())
        .zip(shift.iter())
        .map(|((&xi, &sc), &sh)| modulate(xi, sc, sh, one, shift_bits, clamp_lo, clamp_hi, mode))
        .collect()
}

/// AdaLN-zero gate, per channel: `out = requant(gate · x)`.
pub fn gate(g: i64, x: i64, shift_bits: u32, clamp_lo: i64, clamp_hi: i64, mode: Rounding) -> i64 {
    requantize(g * x, shift_bits, 0, clamp_lo, clamp_hi, mode)
}

/// Per-channel gate over a vector.
pub fn gate_vec(
    g: &[i64],
    x: &[i64],
    shift_bits: u32,
    clamp_lo: i64,
    clamp_hi: i64,
    mode: Rounding,
) -> Vec<i64> {
    g.iter()
        .zip(x.iter())
        .map(|(&gi, &xi)| gate(gi, xi, shift_bits, clamp_lo, clamp_hi, mode))
        .collect()
}

/// Residual add: `out = a + b` (the block's `x + gate·sublayer(x)` wiring).
pub fn residual_add(a: &[i64], b: &[i64]) -> Vec<i64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

/// Affine-free LayerNorm over `x` with a committed inverse-sqrt table.
///
/// Exact integer recipe: `mean = round(Σx / n)`, `var = round(Σ(x−mean)² / n)`,
/// `inv_std = inv_sqrt_table(var)`, `out_i = requant((x_i − mean) · inv_std)`.
/// Returns `None` if `var` is outside the committed table domain (a verifier
/// rejection). `eps` is folded into the table by the exporter.
pub fn layernorm(
    x: &[i64],
    inv_sqrt: &ActivationTable,
    shift_bits: u32,
    clamp_lo: i64,
    clamp_hi: i64,
    mode: Rounding,
) -> Option<Vec<i64>> {
    let n = x.len() as i64;
    if n == 0 {
        return Some(Vec::new());
    }
    let sum: i64 = x.iter().sum();
    let mean = round_div(sum, n);
    let sq_sum: i64 = x.iter().map(|&xi| (xi - mean) * (xi - mean)).sum();
    let var = round_div(sq_sum, n);
    let inv_std = inv_sqrt.eval(var)?;
    Some(
        x.iter()
            .map(|&xi| {
                requantize(
                    (xi - mean) * inv_std,
                    shift_bits,
                    0,
                    clamp_lo,
                    clamp_hi,
                    mode,
                )
            })
            .collect(),
    )
}

/// Numerically-stable row softmax with a committed exp table, returning integer
/// "probabilities" scaled by `one` (`Σ ≈ one`).
///
/// Exact integer recipe: `m = max(scores)`, `e_i = exp_table(s_i − m)`,
/// `prob_i = (e_i · one) / Σ e`. Returns `None` if any `s_i − m` is outside the
/// committed table domain, or all exps are zero.
pub fn softmax(scores: &[i64], exp: &ActivationTable, one: i64) -> Option<Vec<i64>> {
    if scores.is_empty() {
        return Some(Vec::new());
    }
    let m = *scores.iter().max().unwrap();
    let mut exps = Vec::with_capacity(scores.len());
    let mut sum: i64 = 0;
    for &s in scores {
        let e = exp.eval(s - m)?;
        sum += e;
        exps.push(e);
    }
    if sum == 0 {
        return None;
    }
    Some(exps.iter().map(|&e| (e * one) / sum).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn idtable(id: u32, lo: i64, hi: i64) -> ActivationTable {
        ActivationTable {
            table_id: id,
            lo,
            outputs: (lo..=hi).collect(),
        }
    }

    #[test]
    fn round_div_rounds_to_nearest() {
        assert_eq!(round_div(7, 2), 4); // 3.5 -> 4 (away from zero)
        assert_eq!(round_div(-7, 2), -4);
        assert_eq!(round_div(10, 5), 2);
        assert_eq!(round_div(2, 4), 1); // 0.5 -> 1
    }

    #[test]
    fn modulate_matches_le_wm_formula_at_unit_scale() {
        // shift_bits 0, one = 1: modulate = x*(1+scale)+shift exactly.
        assert_eq!(
            modulate(5, 2, 3, 1, 0, -1000, 1000, Rounding::NearestTiesToEven),
            5 * (1 + 2) + 3
        );
        let out = modulate_vec(
            &[1, 2, 3],
            &[0, 1, 2],
            &[10, 20, 30],
            1,
            0,
            -1000,
            1000,
            Rounding::NearestTiesToEven,
        );
        assert_eq!(out, vec![11, 24, 39]); // x*(1+scale)+shift
    }

    #[test]
    fn gate_and_residual() {
        assert_eq!(gate(3, 4, 0, -1000, 1000, Rounding::NearestTiesToEven), 12);
        assert_eq!(residual_add(&[1, 2, 3], &[10, 20, 30]), vec![11, 22, 33]);
    }

    #[test]
    fn layernorm_centers_and_scales() {
        // x = [2,4,6,8], mean = 5, centered = [-3,-1,1,3], var = round(20/4)=5.
        // inv_sqrt table over var domain: identity (so inv_std = 5), shift 0.
        let table = idtable(0, 0, 100);
        let out = layernorm(
            &[2, 4, 6, 8],
            &table,
            0,
            -10000,
            10000,
            Rounding::NearestTiesToEven,
        )
        .unwrap();
        // out_i = (x_i - 5) * 5
        assert_eq!(out, vec![-15, -5, 5, 15]);
    }

    #[test]
    fn softmax_sums_to_one_scale_and_peaks_at_max() {
        // exp table: identity over [-10,10] mapped to nonneg via offset is awkward;
        // use a small explicit monotone table: exp(s) ~ s+11 for s in [-10,0].
        let exp = ActivationTable {
            table_id: 1,
            lo: -3,
            outputs: vec![1, 2, 4, 8], // domain [-3,-2,-1,0] -> exp-ish
        };
        // scores [0, -1, -3]; m=0; shifted [0,-1,-3] -> exp [8,4,1]; sum 13.
        let one = 1300;
        let p = softmax(&[0, -1, -3], &exp, one).unwrap();
        assert_eq!(p, vec![800, 400, 100]); // e=[8,4,1], sum 13, *1300/13
        // The largest score gets the largest probability mass.
        assert!(p[0] > p[1] && p[1] > p[2]);
    }

    #[test]
    fn layernorm_rejects_out_of_domain_variance() {
        let table = idtable(0, 0, 2); // var domain only [0,2]
                                      // var here is 5 -> out of domain -> None.
        assert!(layernorm(
            &[2, 4, 6, 8],
            &table,
            0,
            -100,
            100,
            Rounding::NearestTiesToEven
        )
        .is_none());
    }
}
