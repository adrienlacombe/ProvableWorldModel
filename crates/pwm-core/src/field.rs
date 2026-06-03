// SPDX-License-Identifier: Apache-2.0
//! Field types and the centered signed-integer encoding (RFC-0002).
//!
//! Quantized LeWorldModel inference is integer arithmetic that must never wrap,
//! but the proof field M31 (`p = 2^31 - 1`) does wrap. This module re-exports the
//! field types from the vendored Stwo and locks the **centered signed encoding**
//! that embeds a mathematical signed integer into M31 so that, as long as the
//! value stays within the representable interval, encode/decode round-trips
//! exactly (RFC-0002 §2).
//!
//! `QM31` is the secure (degree-4 extension) field used only for Fiat-Shamir
//! challenges and soundness amplification; it **never** carries a quantized
//! tensor value. All quantized data lives in base `M31` via this encoding.

pub use stwo::core::fields::cm31::CM31;
pub use stwo::core::fields::m31::{M31, P};
pub use stwo::core::fields::qm31::QM31;

/// The Mersenne-31 modulus, `p = 2^31 - 1`, re-exported from the field module.
pub const FIELD_MODULUS: u32 = P;

/// Half-interval bound `P_HALF = (p - 1) / 2 = 1073741823`. The centered signed
/// representable interval is `[-P_HALF, +P_HALF]` (RFC-0002 §2).
pub const P_HALF: i64 = ((P as i64) - 1) / 2;

/// Inclusive lower bound of the centered representable interval (`-P_HALF`).
pub const M31_SIGNED_LO: i64 = -P_HALF;

/// Inclusive upper bound of the centered representable interval (`+P_HALF`).
pub const M31_SIGNED_HI: i64 = P_HALF;

/// Returned by [`try_encode`] when an integer lies outside `M31_SIGNED`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutOfRange {
    /// The offending value.
    pub value: i64,
    /// Inclusive lower bound of the representable interval.
    pub lo: i64,
    /// Inclusive upper bound of the representable interval.
    pub hi: i64,
}

/// True iff `x` lies in the centered representable interval `[-P_HALF, P_HALF]`.
pub const fn in_signed_range(x: i64) -> bool {
    M31_SIGNED_LO <= x && x <= M31_SIGNED_HI
}

/// Encode a signed integer into M31 via the centered encoding `enc(x) = x mod p`
/// (RFC-0002 §2): non-negative `x` map to `[0, P_HALF]`, negative `x` map to the
/// upper half `[P_HALF + 1, p)` (`enc(-1) = p - 1`).
///
/// # Preconditions
///
/// `x` must lie in `[M31_SIGNED_LO, M31_SIGNED_HI]`; this is checked with
/// `debug_assert!`. Use [`try_encode`] for a checked form that returns an error.
pub fn encode(x: i64) -> M31 {
    debug_assert!(in_signed_range(x), "encode: {x} outside M31_SIGNED");
    // rem_euclid yields a non-negative residue in [0, p), which is canonical M31.
    let residue = x.rem_euclid(P as i64);
    M31::from_u32_unchecked(residue as u32)
}

/// Checked [`encode`]: returns [`OutOfRange`] if `x` is outside `M31_SIGNED`
/// instead of panicking. Prefer this at trust boundaries (`prefer explicit
/// failures over silent coercion`).
pub fn try_encode(x: i64) -> Result<M31, OutOfRange> {
    if in_signed_range(x) {
        Ok(encode(x))
    } else {
        Err(OutOfRange {
            value: x,
            lo: M31_SIGNED_LO,
            hi: M31_SIGNED_HI,
        })
    }
}

/// Decode an M31 element to its unique signed preimage in `M31_SIGNED`
/// (RFC-0002 §2): `dec(f) = f` if `f <= P_HALF`, else `f - p`. This is the exact
/// inverse of [`encode`] on the representable interval (INV-FP-01).
pub fn decode(f: M31) -> i64 {
    let v = f.0 as i64; // canonical M31 is always in [0, p)
    if v <= P_HALF {
        v
    } else {
        v - (P as i64)
    }
}
