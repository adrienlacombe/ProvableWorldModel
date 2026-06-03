// SPDX-License-Identifier: Apache-2.0
//! Tests for requantization (RFC-0002 §4.4–§4.5): the exact quotient/remainder
//! split, nearest-ties-to-even and truncate-toward-zero rounding, zero-point,
//! and clamp.

use pwm_core::fixed_point::{
    clamp, requantize, round, split_pow2, Rounding::NearestTiesToEven, Rounding::TruncateTowardZero,
};

#[test]
fn split_is_exact_with_nonnegative_remainder() {
    for &n in &[0i64, 1, 7, 8, 9, 255, 256, -1, -7, -8, -9, -256, -257] {
        for r in 0..12u32 {
            let s = split_pow2(n, r);
            let pow = 1i64 << r;
            // 0 <= rem < 2^r  (INV-FP-05)
            assert!(
                0 <= s.rem && s.rem < pow,
                "rem out of range for n={n} r={r}"
            );
            // n == q*2^r + rem  exactly
            assert_eq!(s.q * pow + s.rem, n, "split identity failed n={n} r={r}");
        }
    }
}

#[test]
fn ntte_matches_committed_golden() {
    // The committed golden requantize.ntte.sample.json (shift = 3):
    // exact, tie-up-to-even, tie-down-to-even, round-up, negative-tie-to-even.
    let cases = [(256i64, 32i64), (12, 2), (20, 2), (13, 2), (-12, -2)];
    for (n, want) in cases {
        let got = requantize(n, 3, 0, i64::MIN, i64::MAX, NearestTiesToEven);
        assert_eq!(got, want, "NTTE requantize({n}, 3) = {got}, want {want}");
    }
}

#[test]
fn ntte_tie_bumps_to_even_both_directions() {
    // rem == half ties: 1.5 -> 2 (up to even), 2.5 -> 2 (down to even),
    // 3.5 -> 4 (up to even), -2.5 -> -2 (toward even).
    let q = |n| requantize(n, 1, 0, i64::MIN, i64::MAX, NearestTiesToEven);
    assert_eq!(q(3), 2); // 1.5 -> 2
    assert_eq!(q(5), 2); // 2.5 -> 2
    assert_eq!(q(7), 4); // 3.5 -> 4
    assert_eq!(q(-5), -2); // -2.5 -> -2
                           // non-ties round to nearest
    assert_eq!(q(2), 1); // 1.0
    assert_eq!(q(1), 0); // 0.5 -> 0 (even)
}

#[test]
fn truncate_toward_zero_by_sign() {
    let q = |n| requantize(n, 3, 0, i64::MIN, i64::MAX, TruncateTowardZero);
    // toward zero: drop the fraction regardless of magnitude
    assert_eq!(q(12), 1); // 1.5 -> 1
    assert_eq!(q(15), 1); // 1.875 -> 1
    assert_eq!(q(-12), -1); // -1.5 -> -1 (toward zero, not floor)
    assert_eq!(q(-15), -1); // -1.875 -> -1
    assert_eq!(q(16), 2); // exact
    assert_eq!(q(-16), -2); // exact
    assert_eq!(q(0), 0);
}

#[test]
fn ntte_and_ttz_differ_on_negatives() {
    // floor-based NTTE vs toward-zero differ in sign handling of the fraction.
    let n = -12; // -1.5 at r=3
    assert_eq!(
        requantize(n, 3, 0, i64::MIN, i64::MAX, NearestTiesToEven),
        -2
    );
    assert_eq!(
        requantize(n, 3, 0, i64::MIN, i64::MAX, TruncateTowardZero),
        -1
    );
}

#[test]
fn zero_point_and_clamp_applied() {
    // rounded(8>>3)=1, + zero_point 10 = 11, clamp to [0, 5] -> 5
    assert_eq!(requantize(8, 3, 10, 0, 5, NearestTiesToEven), 5);
    // rounded + zp within range passes through
    assert_eq!(requantize(8, 3, -1, -10, 10, NearestTiesToEven), 0);
    // direct clamp
    assert_eq!(clamp(7, 0, 5), 5);
    assert_eq!(clamp(-3, 0, 5), 0);
    assert_eq!(clamp(3, 0, 5), 3);
}

#[test]
fn round_exact_needs_no_mode() {
    // rem == 0: both modes return q (also the r == 0 case).
    let s = split_pow2(64, 4); // 64 / 16 = 4 exactly
    assert_eq!(s.rem, 0);
    assert_eq!(round(s, 4, 64, NearestTiesToEven), 4);
    assert_eq!(round(s, 4, 64, TruncateTowardZero), 4);
    let s0 = split_pow2(7, 0); // shift 0: q = n, rem = 0
    assert_eq!((s0.q, s0.rem), (7, 0));
    assert_eq!(round(s0, 0, 7, NearestTiesToEven), 7);
}
