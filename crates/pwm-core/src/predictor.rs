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

use crate::field::{in_signed_range, M31_SIGNED_HI, M31_SIGNED_LO};
use crate::fixed_point::{requantize, BoundedInt, Rounding};
use crate::tables::ActivationTable;

/// Round-to-nearest integer division by a positive divisor `n` (ties away from
/// zero). Used for the LayerNorm mean/variance reductions.
pub fn round_div(a: i64, n: i64) -> i64 {
    round_div_i128(a as i128, n as i128) as i64
}

/// [`round_div`] over `i128`, for the LayerNorm reductions whose intermediate
/// sums can exceed `i64` on adversarial (envelope-bounded) inputs.
fn round_div_i128(a: i128, n: i128) -> i128 {
    debug_assert!(n > 0);
    if a >= 0 {
        (a + n / 2) / n
    } else {
        -((-a + n / 2) / n)
    }
}

/// True iff every element lies in the single-M31 envelope `[-P_HALF, P_HALF]` —
/// the same bound the Freivalds operand guard enforces. The exact-recompute
/// kernels fail closed (return `None`) on out-of-envelope inputs rather than
/// risking an `i64` wrap (a panic under `overflow-checks`), so a malicious
/// buffer is *rejected*, never computed on (issue #180).
fn within_envelope(vals: &[i64]) -> bool {
    vals.iter().all(|&v| in_signed_range(v))
}

/// True iff an `i128` accumulator fits the single-M31 envelope (the bound
/// mirror of [`crate::fixed_point`]'s `finish`).
fn acc_in_envelope(acc: i128) -> bool {
    (M31_SIGNED_LO as i128..=M31_SIGNED_HI as i128).contains(&acc)
}

/// Exact integer dense linear `out[r] = bias[r] + Σ_c W[r·cols + c] · x[c]` over a
/// committed int8 weight matrix `W` (row-major, shape `(rows, cols)`).
///
/// This is the single prover/reference forward kernel for a fixed-weight linear;
/// the verifier audits the same relation with a Freivalds check
/// ([`crate::freivalds::check_linear_biased`]) instead of recomputing it. `bias` has
/// length `rows` (pass an all-zero slice for no bias). Sharing this one definition
/// across the reference model and both prover linear ops keeps them from drifting.
pub fn linear(
    weight: &[BoundedInt],
    x: &[i64],
    bias: &[i64],
    rows: usize,
    cols: usize,
) -> Vec<i64> {
    debug_assert_eq!(weight.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(bias.len(), rows);
    (0..rows)
        .map(|r| {
            let base = r * cols;
            bias[r]
                + (0..cols)
                    .map(|c| weight[base + c].value() * x[c])
                    .sum::<i64>()
        })
        .collect()
}

/// Exact integer batched dense linear over a flattened `[seq, cols]` input.
///
/// Returns a flattened `[seq, rows]` buffer, applying [`linear`] independently to
/// each sequence row with the same weight matrix and bias vector.
pub fn batched_linear(
    weight: &[BoundedInt],
    x: &[i64],
    bias: &[i64],
    seq: usize,
    rows: usize,
    cols: usize,
) -> Vec<i64> {
    debug_assert_eq!(x.len(), seq * cols);
    let mut out = Vec::with_capacity(seq * rows);
    for tt in 0..seq {
        let start = tt * cols;
        out.extend(linear(weight, &x[start..start + cols], bias, rows, cols));
    }
    out
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
/// Returns `None` if `var` is outside the committed table domain, if any input
/// escapes the single-M31 envelope, or if `(x_i − mean) · inv_std` does not fit
/// an `i64` (verifier rejections, fail-closed — issue #180). The variance sum
/// accumulates in `i128`: even envelope-bounded inputs can push `Σ(x−mean)²`
/// past `i64`. `eps` is folded into the table by the exporter.
pub fn layernorm(
    x: &[i64],
    inv_sqrt: &ActivationTable,
    shift_bits: u32,
    clamp_lo: i64,
    clamp_hi: i64,
    mode: Rounding,
) -> Option<Vec<i64>> {
    let n = x.len() as i128;
    if n == 0 {
        return Some(Vec::new());
    }
    if !within_envelope(x) {
        return None;
    }
    let sum: i128 = x.iter().map(|&xi| xi as i128).sum();
    // |mean| <= max|x| <= P_HALF, so the i64 narrowing is exact.
    let mean = round_div_i128(sum, n) as i64;
    let sq_sum: i128 = x
        .iter()
        .map(|&xi| {
            let d = (xi - mean) as i128;
            d * d
        })
        .sum();
    // var <= max (x_i − mean)² < 2^62, so the i64 narrowing is exact.
    let var = round_div_i128(sq_sum, n) as i64;
    let inv_std = inv_sqrt.eval(var)?;
    x.iter()
        .map(|&xi| {
            let prod = (xi - mean) as i128 * inv_std as i128;
            let prod = i64::try_from(prod).ok()?;
            Some(requantize(prod, shift_bits, 0, clamp_lo, clamp_hi, mode))
        })
        .collect()
}

/// Numerically-stable row softmax with a committed exp table, returning integer
/// "probabilities" scaled by `one` (`Σ ≈ one`).
///
/// Exact integer recipe: `m = max(scores)`, `e_i = exp_table(s_i − m)`,
/// `prob_i = (e_i · one) / Σ e`. Returns `None` if any `s_i − m` is outside the
/// committed table domain, all exps are zero, any score escapes the single-M31
/// envelope, or a probability escapes it (fail-closed — issue #180). The exp
/// sum and the `e_i · one` products accumulate in `i128`: committed table
/// outputs and `one` are prover-chosen `i64`s, so the `i64` math could wrap.
pub fn softmax(scores: &[i64], exp: &ActivationTable, one: i64) -> Option<Vec<i64>> {
    if scores.is_empty() {
        return Some(Vec::new());
    }
    if !within_envelope(scores) {
        return None;
    }
    let m = *scores.iter().max().unwrap();
    let mut exps = Vec::with_capacity(scores.len());
    let mut sum: i128 = 0;
    for &s in scores {
        let e = exp.eval(s - m)?;
        sum += e as i128;
        exps.push(e);
    }
    if sum == 0 {
        return None;
    }
    exps.iter()
        .map(|&e| {
            let p = (e as i128 * one as i128) / sum;
            acc_in_envelope(p).then_some(p as i64)
        })
        .collect()
}

/// Exact integer matmul for attention's data-dependent products. `a` is
/// `[rows, inner]` row-major; if `transpose_b`, `b` is `[cols, inner]` and the
/// result is `a·bᵀ` (the QKᵀ scores), else `b` is `[inner, cols]` and the result
/// is `a·b` (prob·V). Returns `[rows, cols]` row-major, or `None` on a length
/// mismatch, an operand outside the single-M31 envelope, or an accumulator that
/// escapes it (fail-closed — issue #180). Accumulation is in `i128`: products of
/// envelope-bounded operands reach `2^60`, so a wide contraction can wrap `i64`.
pub fn matmul(
    a: &[i64],
    b: &[i64],
    rows: usize,
    inner: usize,
    cols: usize,
    transpose_b: bool,
) -> Option<Vec<i64>> {
    if a.len() != rows * inner || b.len() != cols * inner {
        return None;
    }
    if !within_envelope(a) || !within_envelope(b) {
        return None;
    }
    let mut out = alloc::vec![0i64; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let mut acc = 0i128;
            for k in 0..inner {
                let bv = if transpose_b {
                    b[j * inner + k]
                } else {
                    b[k * cols + j]
                };
                acc += a[i * inner + k] as i128 * bv as i128;
            }
            if !acc_in_envelope(acc) {
                return None;
            }
            out[i * cols + j] = acc as i64;
        }
    }
    Some(out)
}

/// Row-wise softmax over a `[rows, row_len]` buffer (attention probabilities):
/// apply [`softmax`] to each contiguous row. Returns `None` on a shape mismatch
/// or an out-of-domain exp.
pub fn softmax_rows(
    scores: &[i64],
    row_len: usize,
    exp: &ActivationTable,
    one: i64,
) -> Option<Vec<i64>> {
    if row_len == 0 || scores.len() % row_len != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(scores.len());
    for chunk in scores.chunks(row_len) {
        out.extend(softmax(chunk, exp, one)?);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    use crate::field::Fp61;
    use crate::freivalds::{check_linear_biased, precompute_v};

    fn idtable(id: u32, lo: i64, hi: i64) -> ActivationTable {
        ActivationTable {
            table_id: id,
            lo,
            outputs: (lo..=hi).collect(),
        }
    }

    fn lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state
    }

    #[test]
    fn round_div_rounds_to_nearest() {
        assert_eq!(round_div(7, 2), 4); // 3.5 -> 4 (away from zero)
        assert_eq!(round_div(-7, 2), -4);
        assert_eq!(round_div(10, 5), 2);
        assert_eq!(round_div(2, 4), 1); // 0.5 -> 1
    }

    #[test]
    fn linear_kernels_match_freivalds_relation() {
        let mut state = 0x1820_5eed_cafe_f00d;
        for _ in 0..128 {
            let rows = 1 + (lcg(&mut state) % 6) as usize;
            let cols = 1 + (lcg(&mut state) % 6) as usize;
            let seq = 1 + (lcg(&mut state) % 5) as usize;

            let w_i8: Vec<i8> = (0..rows * cols)
                .map(|_| (lcg(&mut state) % 17) as i8 - 8)
                .collect();
            let weight: Vec<BoundedInt> = w_i8
                .iter()
                .map(|&v| BoundedInt::new(v as i64, -128, 127).unwrap())
                .collect();
            let bias: Vec<i64> = (0..rows)
                .map(|_| (lcg(&mut state) % 23) as i64 - 11)
                .collect();
            let x: Vec<i64> = (0..seq * cols)
                .map(|_| (lcg(&mut state) % 31) as i64 - 15)
                .collect();
            let r: Vec<Fp61> = (0..rows).map(|_| Fp61::new(lcg(&mut state))).collect();
            let v = precompute_v(&r, &w_i8, rows, cols);

            let batched = batched_linear(&weight, &x, &bias, seq, rows, cols);
            for tt in 0..seq {
                let xrow = &x[tt * cols..(tt + 1) * cols];
                let out = linear(&weight, xrow, &bias, rows, cols);
                let bout = &batched[tt * rows..(tt + 1) * rows];
                assert_eq!(bout, out.as_slice());
                assert!(check_linear_biased(&v, xrow, &bias, &r, bout));
            }
        }
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

    // --- Fail-closed overflow guards (issue #180): adversarial buffers must be
    // rejected with `None`, never wrap an i64 (a panic under overflow-checks). ---

    use crate::field::M31_SIGNED_HI;

    #[test]
    fn matmul_small_case_is_exact() {
        // a = [[1,2],[3,4]] · b = [[5,6],[7,8]] = [[19,22],[43,50]].
        let out = matmul(&[1, 2, 3, 4], &[5, 6, 7, 8], 2, 2, 2, false).unwrap();
        assert_eq!(out, vec![19, 22, 43, 50]);
    }

    #[test]
    fn matmul_rejects_out_of_envelope_operand() {
        assert!(matmul(&[M31_SIGNED_HI + 1], &[1], 1, 1, 1, false).is_none());
        assert!(matmul(&[1], &[-M31_SIGNED_HI - 1], 1, 1, 1, false).is_none());
    }

    #[test]
    fn matmul_fails_closed_on_wide_accumulator() {
        // 16 envelope-edge products: 16 · P_HALF² ≈ 1.8e19 wraps an i64 sum; the
        // i128 accumulator computes it exactly and the envelope check rejects it.
        let a = vec![M31_SIGNED_HI; 16];
        let b = vec![M31_SIGNED_HI; 16];
        assert!(matmul(&a, &b, 1, 16, 1, false).is_none());
    }

    #[test]
    fn layernorm_rejects_out_of_envelope_input() {
        let table = idtable(0, 0, 100);
        assert!(layernorm(
            &[M31_SIGNED_HI + 1, 0],
            &table,
            0,
            -100,
            100,
            Rounding::NearestTiesToEven
        )
        .is_none());
    }

    #[test]
    fn layernorm_fails_closed_on_variance_overflow() {
        // 16 envelope-edge centered squares: Σ(x−mean)² ≈ 1.8e19 wraps an i64; the
        // i128 sum computes the variance exactly, which then misses the committed
        // table domain — a clean `None`, not a panic.
        let x: Vec<i64> = (0..16)
            .map(|i| {
                if i % 2 == 0 {
                    M31_SIGNED_HI
                } else {
                    -M31_SIGNED_HI
                }
            })
            .collect();
        let table = idtable(0, 0, 1000);
        assert!(layernorm(&x, &table, 0, -100, 100, Rounding::NearestTiesToEven).is_none());
    }

    #[test]
    fn softmax_rejects_out_of_envelope_score() {
        let exp = idtable(1, -10, 0);
        assert!(softmax(&[M31_SIGNED_HI + 1, 0], &exp, 100).is_none());
    }

    #[test]
    fn softmax_fails_closed_on_out_of_envelope_probability() {
        // Adversarial committed table with huge mixed-sign outputs: Σe = 2, so a
        // probability (e·one)/Σe ≈ 2^49 escapes the envelope. The i128 product
        // computes it exactly (the old i64 `e · one` could wrap) and rejects.
        let exp = ActivationTable {
            table_id: 1,
            lo: -1,
            outputs: vec![-(1 << 40) + 2, 1 << 40],
        };
        assert!(softmax(&[0, -1], &exp, 1024).is_none());
    }
}
