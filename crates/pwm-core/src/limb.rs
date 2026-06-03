// SPDX-License-Identifier: Apache-2.0
//! Accumulator representation: single-M31 vs two-limb (RFC-0002 §7).
//!
//! A dot-product accumulator stays in a **single** signed M31 value while its
//! propagated magnitude fits `SAFE_HI = P_HALF`. The widest V0 dot product is the
//! predictor MLP (`mlp_dim = 2048`); with symmetric int8 operands the worst-case
//! magnitude is `2048 * 127 * 127 = 33_032_192 < 2^31 - 1`, ~65× inside the safe
//! interval, so the V0 int8 linear/MLP path needs no limbs. When a bound exceeds
//! `SAFE_HI` (int16 activations into long accumulators, or wider quantization), the
//! value is **two-limb decomposed**: `value = low + 2^K · high`, each limb
//! independently range-checked (INV-FP-10).
//!
//! This module fixes the *trigger* (`> SAFE_HI`), the *two-limb shape*, the
//! representation *selector* driven by declared metadata, and the decompose/
//! reconstruct reference. The full limb-arithmetic AIR layout lives with the
//! linear component (RFC-0005, #48).

use crate::field::P_HALF;
use crate::fixed_point::BoundedInt;

/// The largest magnitude that fits a single signed M31 value (`= P_HALF`).
pub const SAFE_HI: i64 = P_HALF;

/// Default limb split `K` (a per-op manifest field, bound by
/// `quantization_commitment`): `value = low + 2^K · high`.
pub const DEFAULT_LIMB_K: u32 = 16;

/// Whether a value with propagated bound `[lo, hi]` exceeds the single-M31 safe
/// interval and therefore requires limb decomposition (RFC-0002 §7 trigger).
pub const fn needs_limbs(lo: i64, hi: i64) -> bool {
    lo < -SAFE_HI || hi > SAFE_HI
}

/// The accumulator representation chosen for a value, from its propagated bound
/// and the declared limb split `K`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccumRepr {
    /// Fits a single signed M31 value.
    SingleM31,
    /// Two-limb: `value = low + 2^K · high`.
    TwoLimb {
        /// The limb split exponent.
        k: u32,
    },
}

/// A two-limb decomposition: `value = low + 2^K · high`, `low ∈ [0, 2^K − 1]`
/// (unsigned, range-checked) and `high` signed (range-checked). Reconstructs
/// exactly (INV-FP-10).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LimbValue {
    /// Low limb, range-checked to `[0, 2^K − 1]`.
    pub low: BoundedInt,
    /// High limb, signed, carrying the magnitude and sign.
    pub high: BoundedInt,
    /// The limb split exponent `K`.
    pub k: u32,
}

/// Error decomposing into or selecting a limb representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimbError {
    /// `K` is outside the supported range `1..=30`.
    InvalidK {
        /// The offending `K`.
        k: u32,
    },
    /// The value needs more than two limbs in V0 (the high limb would escape
    /// `M31_SIGNED`): `Error::AccumulatorRange`, a manifest outside the V0
    /// quantization envelope.
    ExceedsTwoLimbs {
        /// Propagated lower bound.
        lo: i64,
        /// Propagated upper bound.
        hi: i64,
        /// The limb split `K` attempted.
        k: u32,
    },
}

const MAX_K: u32 = 30;

/// Select the accumulator representation for a value with propagated bound
/// `[lo, hi]` and declared split `K`: [`AccumRepr::SingleM31`] when the magnitude
/// fits `SAFE_HI`, otherwise [`AccumRepr::TwoLimb`] when two limbs suffice (the
/// high limb fits `M31_SIGNED`). More than two limbs is [`LimbError::ExceedsTwoLimbs`].
pub fn select_accum_repr(lo: i64, hi: i64, k: u32) -> Result<AccumRepr, LimbError> {
    if !needs_limbs(lo, hi) {
        return Ok(AccumRepr::SingleM31);
    }
    if !(1..=MAX_K).contains(&k) {
        return Err(LimbError::InvalidK { k });
    }
    let pow = 1i64 << k;
    // High-limb bounds: floor(lo/2^K) .. floor(hi/2^K). Both must fit M31_SIGNED.
    let high_lo = lo.div_euclid(pow);
    let high_hi = hi.div_euclid(pow);
    if high_lo < -SAFE_HI || high_hi > SAFE_HI {
        return Err(LimbError::ExceedsTwoLimbs { lo, hi, k });
    }
    Ok(AccumRepr::TwoLimb { k })
}

/// Decompose `value` (with propagated bound `[lo, hi]`) into two range-checked
/// limbs at split `K` (RFC-0002 §7): `low = value mod 2^K ∈ [0, 2^K − 1]`,
/// `high = floor(value / 2^K) ∈ [floor(lo/2^K), floor(hi/2^K)]`.
pub fn decompose(value: i64, lo: i64, hi: i64, k: u32) -> Result<LimbValue, LimbError> {
    if !(1..=MAX_K).contains(&k) {
        return Err(LimbError::InvalidK { k });
    }
    let pow = 1i64 << k;
    let low_v = value.rem_euclid(pow);
    let high_v = value.div_euclid(pow);
    let high_lo = lo.div_euclid(pow);
    let high_hi = hi.div_euclid(pow);

    let low = BoundedInt::new(low_v, 0, pow - 1).map_err(|_| LimbError::InvalidK { k })?;
    let high = BoundedInt::new(high_v, high_lo, high_hi)
        .map_err(|_| LimbError::ExceedsTwoLimbs { lo, hi, k })?;
    Ok(LimbValue { low, high, k })
}

/// Reconstruct the integer value from its limbs: `low + 2^K · high` (INV-FP-10).
pub fn reconstruct(limb: &LimbValue) -> i64 {
    limb.low.value() + (1i64 << limb.k) * limb.high.value()
}
