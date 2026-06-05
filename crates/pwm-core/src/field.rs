// SPDX-License-Identifier: Apache-2.0
//! Field types and the centered signed-integer encoding.
//!
//! Two fields, two roles:
//!
//! - [`M31`] (`p = 2^31 - 1`) is the **value encoding** field: quantized
//!   LeWorldModel tensor cells embed into `M31` via the centered signed encoding
//!   so encode/decode round-trips exactly within the representable interval.
//!   Implemented natively here (post-pivot: no Stwo re-export).
//! - [`Fp61`] (`p = 2^61 - 1`, a Mersenne prime) is the **audit** field used by
//!   the Freivalds checks (see [`crate::freivalds`]) and by the Fiat-Shamir
//!   transcript when squeezing challenge vectors. Its larger modulus gives a
//!   `1/p ≈ 2^-61` per-check soundness error.
//!
//! Both are pure integer arithmetic and `no_std`-clean; `pwm-core` carries no
//! proving substrate (INV-ARCH-01).

// ===========================================================================
// M31 — value-encoding field (p = 2^31 - 1)
// ===========================================================================

/// The Mersenne-31 modulus, `p = 2^31 - 1`.
pub const P: u32 = 0x7fff_ffff;

/// A canonical Mersenne-31 field element, always in `[0, P)`.
///
/// The single public field `0` is the canonical residue, matching the surface
/// the canonical serialization and the fixed-point encoding rely on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct M31(pub u32);

impl M31 {
    /// Construct from a residue **assumed canonical** (`value < P`). Callers that
    /// cannot guarantee canonicity must reduce first; the canonical-serialization
    /// decoder validates `value < P` before calling this.
    pub const fn from_u32_unchecked(value: u32) -> Self {
        M31(value)
    }

    /// The canonical residue in `[0, P)`.
    pub const fn value(self) -> u32 {
        self.0
    }
}

/// The Mersenne-31 modulus as a free constant (kept for call sites that referred
/// to a re-exported `FIELD_MODULUS`).
pub const FIELD_MODULUS: u32 = P;

/// Half-interval bound `P_HALF = (p - 1) / 2 = 1073741823`. The centered signed
/// representable interval is `[-P_HALF, +P_HALF]`.
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

/// Encode a signed integer into M31 via the centered encoding `enc(x) = x mod p`:
/// non-negative `x` map to `[0, P_HALF]`, negative `x` map to the upper half
/// `[P_HALF + 1, p)` (`enc(-1) = p - 1`).
///
/// # Preconditions
///
/// `x` must lie in `[M31_SIGNED_LO, M31_SIGNED_HI]`; checked with `debug_assert!`.
/// Use [`try_encode`] for a checked form.
pub fn encode(x: i64) -> M31 {
    debug_assert!(in_signed_range(x), "encode: {x} outside M31_SIGNED");
    let residue = x.rem_euclid(P as i64);
    M31::from_u32_unchecked(residue as u32)
}

/// Checked [`encode`]: returns [`OutOfRange`] if `x` is outside `M31_SIGNED`.
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

/// Decode an M31 element to its unique signed preimage in `M31_SIGNED`:
/// `dec(f) = f` if `f <= P_HALF`, else `f - p`. Exact inverse of [`encode`].
pub fn decode(f: M31) -> i64 {
    let v = f.0 as i64; // canonical M31 is always in [0, p)
    if v <= P_HALF {
        v
    } else {
        v - (P as i64)
    }
}

// ===========================================================================
// Fp61 — audit field (p = 2^61 - 1, Mersenne prime)
// ===========================================================================

/// The audit-field modulus `p = 2^61 - 1` (Mersenne prime).
pub const FREIVALDS_P: u64 = (1u64 << 61) - 1;

/// A canonical element of the audit field `F_p`, `p = 2^61 - 1`, in `[0, p)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Fp61(pub u64);

impl Fp61 {
    /// The additive identity.
    pub const ZERO: Fp61 = Fp61(0);
    /// The multiplicative identity.
    pub const ONE: Fp61 = Fp61(1);

    /// Reduce an arbitrary `u64` into `[0, p)`.
    pub const fn new(x: u64) -> Self {
        Fp61(reduce64(x))
    }

    /// Embed a signed integer into `F_p` (centered: negative maps to `p - |x|`).
    pub fn from_i64(x: i64) -> Self {
        if x >= 0 {
            Fp61(reduce64(x as u64))
        } else {
            // |x| fits u64; reduce then negate mod p.
            let m = reduce64((-(x as i128)) as u64);
            Fp61(if m == 0 { 0 } else { FREIVALDS_P - m })
        }
    }

    /// Modular addition.
    pub const fn add(self, other: Fp61) -> Fp61 {
        // Both operands < p < 2^61, so the sum < 2^62 and never overflows u64.
        let s = self.0 + other.0;
        Fp61(if s >= FREIVALDS_P { s - FREIVALDS_P } else { s })
    }

    /// Modular subtraction.
    pub const fn sub(self, other: Fp61) -> Fp61 {
        Fp61(if self.0 >= other.0 {
            self.0 - other.0
        } else {
            self.0 + FREIVALDS_P - other.0
        })
    }

    /// Modular multiplication via 128-bit product and Mersenne reduction.
    pub const fn mul(self, other: Fp61) -> Fp61 {
        Fp61(reduce128((self.0 as u128) * (other.0 as u128)))
    }
}

/// Reduce a `u64` modulo `2^61 - 1` to `[0, p)`.
const fn reduce64(x: u64) -> u64 {
    // x = hi * 2^61 + lo ; 2^61 ≡ 1 (mod p) ⇒ x ≡ hi + lo (mod p).
    let lo = x & FREIVALDS_P;
    let hi = x >> 61;
    let mut r = lo + hi; // < 2^61 + 2^3
    if r >= FREIVALDS_P {
        r -= FREIVALDS_P;
    }
    r
}

/// Reduce a `u128` modulo `2^61 - 1` to `[0, p)`.
const fn reduce128(x: u128) -> u64 {
    let mask = FREIVALDS_P as u128;
    // Fold 122-bit product down using 2^61 ≡ 1 (mod p).
    let lo = (x & mask) as u64;
    let mid = ((x >> 61) & mask) as u64;
    let hi = (x >> 122) as u64; // < 2^6
                                // lo, mid < 2^61; hi < 2^61. Sum < 3 * 2^61 < 2^63: safe in u64.
    let mut r = (lo as u128) + (mid as u128) + (hi as u128);
    // Final fold (r may be up to ~2^63 → at most one more reduction round).
    let folded = ((r & mask) + (r >> 61)) as u64;
    r = folded as u128;
    let mut out = r as u64;
    if out >= FREIVALDS_P {
        out -= FREIVALDS_P;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn m31_encode_decode_roundtrip() {
        for x in [-P_HALF, -1, 0, 1, 12345, P_HALF] {
            assert_eq!(decode(encode(x)), x);
        }
        assert_eq!(encode(-1), M31::from_u32_unchecked(P - 1));
        assert!(try_encode(P_HALF + 1).is_err());
    }

    #[test]
    fn fp61_basic_arithmetic() {
        let a = Fp61::new(FREIVALDS_P - 1);
        let b = Fp61::new(5);
        assert_eq!(a.add(b), Fp61::new(4)); // (p-1)+5 = p+4 ≡ 4
        assert_eq!(Fp61::ZERO.sub(b), Fp61::new(FREIVALDS_P - 5));
        assert_eq!(a.mul(Fp61::ONE), a);
        // (p-1)*(p-1) ≡ 1 (mod p)
        assert_eq!(a.mul(a), Fp61::ONE);
    }

    #[test]
    fn fp61_from_i64_signed() {
        assert_eq!(Fp61::from_i64(-1), Fp61::new(FREIVALDS_P - 1));
        assert_eq!(Fp61::from_i64(-3).add(Fp61::from_i64(3)), Fp61::ZERO);
        assert_eq!(Fp61::from_i64(7), Fp61::new(7));
    }

    #[test]
    fn fp61_reduce_is_canonical() {
        assert_eq!(reduce64(FREIVALDS_P), 0);
        assert_eq!(reduce64(FREIVALDS_P + 1), 1);
        // p ≡ 0, so p·p ≡ 0; and (p-1)·(p-1) ≡ 1 (mod p).
        assert_eq!(reduce128((FREIVALDS_P as u128) * (FREIVALDS_P as u128)), 0);
        let pm1 = (FREIVALDS_P - 1) as u128;
        assert_eq!(reduce128(pm1 * pm1), 1);
    }
}
