// SPDX-License-Identifier: Apache-2.0
//! Committed nonlinear lookup-table generation (specs.md §2.2, §8.3; backlog
//! E-204).
//!
//! The exporter turns a floating-point nonlinearity (GELU, SiLU, exp for softmax,
//! inverse-sqrt for LayerNorm) into a **committed integer table** by evaluating it
//! at every quantized input over a domain and rounding to the output scale. The
//! float math runs here, offline, exactly once; the prover and verifier then use
//! only the integer table (no floating point at prove/verify time). This needs no
//! model checkpoint — a table is a pure function of (domain, scales, op).

use pwm_core::tables::ActivationTable;

/// Quantize a float nonlinearity `f` into a committed integer table over the
/// integer input domain `[lo, hi]`. Input integer `q` denotes the real value
/// `q · 2^(-in_frac)`; the output integer is `round(f(real) · 2^(out_frac))`.
pub fn quantize_nonlinearity<F: Fn(f64) -> f64>(
    table_id: u32,
    lo: i64,
    hi: i64,
    in_frac: u32,
    out_frac: u32,
    f: F,
) -> ActivationTable {
    let in_scale = 2f64.powi(-(in_frac as i32));
    let out_scale = 2f64.powi(out_frac as i32);
    let outputs = (lo..=hi)
        .map(|q| {
            let real_in = q as f64 * in_scale;
            (f(real_in) * out_scale).round() as i64
        })
        .collect();
    ActivationTable {
        table_id,
        lo,
        outputs,
    }
}

/// GELU (tanh approximation, matching the common PyTorch `approximate="tanh"`):
/// `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`.
pub fn gelu(x: f64) -> f64 {
    const C: f64 = 0.797_884_560_802_865_4; // sqrt(2/pi)
    0.5 * x * (1.0 + (C * (x + 0.044715 * x * x * x)).tanh())
}

/// SiLU / swish: `x · sigmoid(x) = x / (1 + e^{-x})`.
pub fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// Generate the GELU table over `[lo, hi]` at the given fractional bits.
pub fn gelu_table(table_id: u32, lo: i64, hi: i64, in_frac: u32, out_frac: u32) -> ActivationTable {
    quantize_nonlinearity(table_id, lo, hi, in_frac, out_frac, gelu)
}

/// Generate the SiLU table over `[lo, hi]`.
pub fn silu_table(table_id: u32, lo: i64, hi: i64, in_frac: u32, out_frac: u32) -> ActivationTable {
    quantize_nonlinearity(table_id, lo, hi, in_frac, out_frac, silu)
}

/// Generate the softmax exp table `e^x` over `[lo, 0]` (inputs are score − max ≤ 0).
pub fn exp_table(table_id: u32, lo: i64, in_frac: u32, out_frac: u32) -> ActivationTable {
    quantize_nonlinearity(table_id, lo, 0, in_frac, out_frac, |x| x.exp())
}

/// Generate the inverse-sqrt table `1/√x` over `[1, hi]` (variance ≥ 1; the
/// exporter folds `eps` into the domain offset).
pub fn inv_sqrt_table(table_id: u32, hi: i64, in_frac: u32, out_frac: u32) -> ActivationTable {
    quantize_nonlinearity(table_id, 1, hi, in_frac, out_frac, |x| 1.0 / x.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_table_has_expected_shape() {
        // Unit fractional scales: integer in == real in, out scaled by 1.
        let t = gelu_table(0, -8, 8, 0, 0);
        assert_eq!(t.eval(0), Some(0)); // gelu(0) = 0
                                        // Large positive -> ~x; large negative -> ~0.
        assert_eq!(t.eval(8), Some(8)); // gelu(8) ≈ 8
        assert_eq!(t.eval(-8), Some(0)); // gelu(-8) ≈ 0
                                         // Monotonic non-decreasing.
        let mut prev = i64::MIN;
        for q in -8..=8 {
            let v = t.eval(q).unwrap();
            assert!(v >= prev, "gelu table must be non-decreasing");
            prev = v;
        }
    }

    #[test]
    fn silu_and_exp_tables() {
        let s = silu_table(1, -8, 8, 0, 0);
        assert_eq!(s.eval(0), Some(0)); // silu(0) = 0
                                        // exp table: e^0 = 1 at out_frac 0; e^{-large} ~ 0.
        let e = exp_table(2, -8, 0, 4); // lo=-8, in_frac=0, out scaled by 16
        assert_eq!(e.eval(0), Some(16)); // e^0 * 16 = 16
        assert!(e.eval(-8).unwrap() >= 0);
    }

    #[test]
    fn inv_sqrt_table_decreasing_and_committed() {
        let t = inv_sqrt_table(3, 16, 0, 4); // 1/sqrt(x) * 16
        assert_eq!(t.eval(1), Some(16)); // 1/sqrt(1) * 16 = 16
        assert_eq!(t.eval(4), Some(8)); // 1/sqrt(4) * 16 = 8
                                        // Regenerating yields an identical (committed) table.
        let t2 = inv_sqrt_table(3, 16, 0, 4);
        assert_eq!(t.commitment(), t2.commitment());
    }
}
