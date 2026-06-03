// SPDX-License-Identifier: Apache-2.0
//! Bounded-integer reference semantics (RFC-0002 §3–§6).
//!
//! Every value in quantized inference carries a declared inclusive range
//! `[lo, hi]`. The range is not advisory: it is the exact interval the AIR
//! range-checks, and arithmetic **must never wrap** — `OverflowPolicy::Reject`
//! is the only V0 policy. This module is the normative *reference*: its
//! add/sub/mul/compare define the exact integer semantics every `pwm-air`
//! component must reproduce bit-for-bit (INV-FP-06).
//!
//! [`BoundedInt`] keeps its fields private and is constructed only through the
//! checked constructors, so the invariant `lo <= value <= hi` and (for non-limb
//! values) `[lo, hi] ⊆ M31_SIGNED` cannot be violated by construction. Limb
//! decomposition for accumulators that legitimately exceed `M31_SIGNED`
//! (RFC-0002 §7) is a separate concern (#31); here, a propagated bound that
//! escapes `M31_SIGNED` is the static half of overflow=reject and is an error.

use crate::field::{encode, in_signed_range, M31, M31_SIGNED_HI, M31_SIGNED_LO};

/// A bounded signed integer: a mathematical value with a declared inclusive
/// range `[lo, hi]`. Invariants (enforced at construction): `lo <= value <= hi`
/// and `[lo, hi] ⊆ M31_SIGNED`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundedInt {
    value: i64,
    lo: i64,
    hi: i64,
}

/// Error constructing a [`BoundedInt`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundError {
    /// `lo > hi`: the declared interval is empty.
    InvertedBounds {
        /// Declared lower bound.
        lo: i64,
        /// Declared upper bound.
        hi: i64,
    },
    /// `value` is outside the declared `[lo, hi]`.
    ValueOutOfBounds {
        /// The value.
        value: i64,
        /// Declared lower bound.
        lo: i64,
        /// Declared upper bound.
        hi: i64,
    },
    /// `[lo, hi]` is not contained in `M31_SIGNED`; a non-limb value cannot
    /// represent it without wrapping (RFC-0002 §7 limb decomposition is #31).
    BoundsEscapeSignedRange {
        /// Declared lower bound.
        lo: i64,
        /// Declared upper bound.
        hi: i64,
    },
}

/// Error from a bounded arithmetic operation: the propagated result bound escapes
/// `M31_SIGNED` (overflow=reject, the static half — RFC-0002 §6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Overflow {
    /// The operation that overflowed (`"add"`, `"sub"`, `"mul"`).
    pub op: &'static str,
    /// Propagated lower bound that escaped the representable interval.
    pub lo: i128,
    /// Propagated upper bound that escaped the representable interval.
    pub hi: i128,
}

/// Error from a comparison-witness construction (RFC-0002 §4.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareError {
    /// The relation does not hold, so no nonnegative difference witness exists.
    NotOrdered {
        /// Left value.
        a: i64,
        /// Right value.
        b: i64,
    },
    /// The difference bound escapes `M31_SIGNED`.
    Overflow(Overflow),
}

impl BoundedInt {
    /// Construct a bounded integer, enforcing `lo <= value <= hi` and
    /// `[lo, hi] ⊆ M31_SIGNED`.
    pub fn new(value: i64, lo: i64, hi: i64) -> Result<Self, BoundError> {
        if lo > hi {
            return Err(BoundError::InvertedBounds { lo, hi });
        }
        if value < lo || value > hi {
            return Err(BoundError::ValueOutOfBounds { value, lo, hi });
        }
        if !in_signed_range(lo) || !in_signed_range(hi) {
            return Err(BoundError::BoundsEscapeSignedRange { lo, hi });
        }
        Ok(Self { value, lo, hi })
    }

    /// Construct a constant: a value whose bounds are exactly `[value, value]`.
    pub fn exact(value: i64) -> Result<Self, BoundError> {
        Self::new(value, value, value)
    }

    /// The mathematical value.
    pub fn value(&self) -> i64 {
        self.value
    }

    /// The declared inclusive lower bound.
    pub fn lo(&self) -> i64 {
        self.lo
    }

    /// The declared inclusive upper bound.
    pub fn hi(&self) -> i64 {
        self.hi
    }

    /// The centered M31 encoding of the value (RFC-0002 §2). Always valid because
    /// `value ∈ [lo, hi] ⊆ M31_SIGNED`. Named `to_field` to avoid colliding with
    /// the canonical-serialization `encode` (`crate::serialize::CanonicalEncode`).
    pub fn to_field(&self) -> M31 {
        encode(self.value)
    }

    /// `self + other` with bound propagation `[a.lo+b.lo, a.hi+b.hi]`; errors if
    /// the propagated bound escapes `M31_SIGNED` (RFC-0002 §3, §4.1).
    pub fn add(&self, other: &BoundedInt) -> Result<BoundedInt, Overflow> {
        let lo = self.lo as i128 + other.lo as i128;
        let hi = self.hi as i128 + other.hi as i128;
        let value = self.value as i128 + other.value as i128;
        finish("add", value, lo, hi)
    }

    /// `self - other` with bound propagation `[a.lo-b.hi, a.hi-b.lo]` (RFC-0002 §3).
    pub fn sub(&self, other: &BoundedInt) -> Result<BoundedInt, Overflow> {
        let lo = self.lo as i128 - other.hi as i128;
        let hi = self.hi as i128 - other.lo as i128;
        let value = self.value as i128 - other.value as i128;
        finish("sub", value, lo, hi)
    }

    /// `self * other` with bound propagation over the four corner products
    /// (RFC-0002 §3, §4.2).
    pub fn mul(&self, other: &BoundedInt) -> Result<BoundedInt, Overflow> {
        let corners = [
            self.lo as i128 * other.lo as i128,
            self.lo as i128 * other.hi as i128,
            self.hi as i128 * other.lo as i128,
            self.hi as i128 * other.hi as i128,
        ];
        let lo = *corners.iter().min().expect("non-empty");
        let hi = *corners.iter().max().expect("non-empty");
        let value = self.value as i128 * other.value as i128;
        finish("mul", value, lo, hi)
    }

    /// Nonnegative-difference witness that `self <= other`: returns `d = other -
    /// self` bounded in `[0, other.hi - self.lo]` (RFC-0002 §4.6). Errors with
    /// [`CompareError::NotOrdered`] if `self > other` (no nonnegative witness).
    pub fn le_witness(&self, other: &BoundedInt) -> Result<BoundedInt, CompareError> {
        diff_witness(self, other, 0)
    }

    /// Strict witness that `self < other`: returns `d = other - self - 1` bounded
    /// in `[0, other.hi - self.lo - 1]` (RFC-0002 §4.6). Errors if `self >= other`.
    pub fn lt_witness(&self, other: &BoundedInt) -> Result<BoundedInt, CompareError> {
        diff_witness(self, other, 1)
    }
}

/// Finish an arithmetic op: validate the propagated bound is in `M31_SIGNED`,
/// then build the result. `value ∈ [lo, hi]` holds by construction of the inputs.
fn finish(op: &'static str, value: i128, lo: i128, hi: i128) -> Result<BoundedInt, Overflow> {
    let smin = M31_SIGNED_LO as i128;
    let smax = M31_SIGNED_HI as i128;
    if lo < smin || hi > smax {
        return Err(Overflow { op, lo, hi });
    }
    // Bounds are within M31_SIGNED ⊂ i64, and value ∈ [lo, hi], so all fit i64.
    Ok(BoundedInt {
        value: value as i64,
        lo: lo as i64,
        hi: hi as i64,
    })
}

/// Shared body of `le_witness`/`lt_witness`: `d = other - self - strict`, bounded
/// in `[0, other.hi - self.lo - strict]`.
fn diff_witness(a: &BoundedInt, b: &BoundedInt, strict: i64) -> Result<BoundedInt, CompareError> {
    let d = b.value - a.value - strict;
    if d < 0 {
        return Err(CompareError::NotOrdered {
            a: a.value,
            b: b.value,
        });
    }
    let d_max = b.hi - a.lo - strict;
    BoundedInt::new(d, 0, d_max).map_err(|_| {
        CompareError::Overflow(Overflow {
            op: "compare",
            lo: 0,
            hi: d_max as i128,
        })
    })
}

/// The rounding mode applied by [`requantize`]. Exactly one is active per
/// manifest, bound by `quantization_commitment` (INV-FP-04).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rounding {
    /// Round half to even (the canonical default).
    NearestTiesToEven,
    /// Truncate toward zero on the original signed value (optional override).
    TruncateTowardZero,
}

impl Rounding {
    /// Immutable canonical-serialization discriminant (RFC-0014 §1).
    pub const fn discriminant(self) -> u8 {
        match self {
            Rounding::NearestTiesToEven => 0,
            Rounding::TruncateTowardZero => 1,
        }
    }

    /// Inverse of [`Rounding::discriminant`]; `None` for an unknown value.
    pub const fn from_discriminant(value: u8) -> Option<Self> {
        match value {
            0 => Some(Rounding::NearestTiesToEven),
            1 => Some(Rounding::TruncateTowardZero),
            _ => None,
        }
    }
}

/// The overflow policy. V0 has exactly one: `Reject` (wrapping is unsound). A
/// `Wrap`/`Saturate` variant would be a new relation version, not a flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverflowPolicy {
    /// Reject any value whose bound escapes the representable interval.
    Reject,
}

impl OverflowPolicy {
    /// Immutable canonical-serialization discriminant (RFC-0014 §1).
    pub const fn discriminant(self) -> u8 {
        match self {
            OverflowPolicy::Reject => 0,
        }
    }

    /// Inverse of [`OverflowPolicy::discriminant`]; `None` for an unknown value.
    pub const fn from_discriminant(value: u8) -> Option<Self> {
        match value {
            0 => Some(OverflowPolicy::Reject),
            _ => None,
        }
    }
}

/// The exact quotient/remainder split of `n` by `2^r` with a **nonnegative**
/// remainder (RFC-0002 §4.4, INV-FP-05): `q = floor(n / 2^r)` (toward −∞) and
/// `rem = n − q·2^r`, so `0 <= rem < 2^r` and `n == q·2^r + rem` exactly. These
/// are the witnesses the requantization AIR range-checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuotRem {
    /// Floor quotient (toward −∞).
    pub q: i64,
    /// Nonnegative remainder, `0 <= rem < 2^r`.
    pub rem: i64,
}

/// Compute the exact split [`QuotRem`] of `n` by `2^r`. `r` must be `< 63`.
pub fn split_pow2(n: i64, r: u32) -> QuotRem {
    debug_assert!(r < 63, "shift {r} too large for i64");
    let divisor = 1i64 << r;
    QuotRem {
        q: n.div_euclid(divisor),
        rem: n.rem_euclid(divisor),
    }
}

/// Apply the rounding mode to a split (RFC-0002 §4.4). `n` is the original signed
/// value (used by `TruncateTowardZero`).
pub fn round(split: QuotRem, r: u32, n: i64, mode: Rounding) -> i64 {
    let QuotRem { q, rem } = split;
    if rem == 0 {
        // Exact: no rounding for either mode (also the only case when r == 0).
        return q;
    }
    match mode {
        Rounding::NearestTiesToEven => {
            // r >= 1 here, since a nonzero remainder requires 2^r > 1.
            let half = 1i64 << (r - 1);
            if rem < half {
                q
            } else if rem > half {
                q + 1
            } else {
                q + (q & 1) // tie: bump to even
            }
        }
        Rounding::TruncateTowardZero => {
            // q is floor(n/2^r); toward-zero keeps q for n >= 0 and q+1 for n < 0
            // (rem != 0 here).
            if n >= 0 {
                q
            } else {
                q + 1
            }
        }
    }
}

/// Clamp `x` to `[c_lo, c_hi]` (RFC-0002 §4.5). Clamp is bound-narrowing, applied
/// only where the manifest declares it; it never repairs an out-of-range value.
pub fn clamp(x: i64, c_lo: i64, c_hi: i64) -> i64 {
    x.clamp(c_lo, c_hi)
}

/// Requantize an accumulator: rescale by an arithmetic right shift of `r` bits
/// with the exact split, apply the rounding `mode`, add `zero_point`, and clamp
/// to `[c_lo, c_hi]` (RFC-0002 §4.4). This is the locked reference the AIR
/// reproduces bit-for-bit.
pub fn requantize(n: i64, r: u32, zero_point: i64, c_lo: i64, c_hi: i64, mode: Rounding) -> i64 {
    let split = split_pow2(n, r);
    let rounded = round(split, r, n, mode);
    clamp(rounded + zero_point, c_lo, c_hi)
}

/// Accumulate a left fold `acc_0 = bias`, `acc_{i+1} = acc_i + term_i` over a
/// single signed M31 value (RFC-0002 §4.3). Each step propagates the bound and
/// rejects (never wraps) if the running bound escapes `M31_SIGNED` — the point at
/// which the value must instead be limb-decomposed (`crate::limb`). Returns the
/// final accumulator.
pub fn accumulate(bias: &BoundedInt, terms: &[BoundedInt]) -> Result<BoundedInt, Overflow> {
    let mut acc = *bias;
    for term in terms {
        acc = acc.add(term)?;
    }
    Ok(acc)
}
