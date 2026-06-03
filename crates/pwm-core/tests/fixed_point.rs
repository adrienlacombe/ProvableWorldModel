// SPDX-License-Identifier: Apache-2.0
//! Tests for `BoundedInt` reference semantics (RFC-0002 §3–§6): construction
//! invariants, bound propagation, overflow=reject, and comparison witnesses.

use pwm_core::field::{decode, P_HALF};
use pwm_core::fixed_point::{BoundError, BoundedInt, CompareError};

#[test]
fn construction_enforces_invariants() {
    assert!(BoundedInt::new(5, 0, 10).is_ok());
    assert!(BoundedInt::new(0, 0, 0).is_ok());
    assert!(BoundedInt::new(-7, -10, -1).is_ok());

    // value outside [lo, hi]  (the acceptance negative test).
    assert_eq!(
        BoundedInt::new(11, 0, 10),
        Err(BoundError::ValueOutOfBounds {
            value: 11,
            lo: 0,
            hi: 10
        })
    );
    // inverted bounds.
    assert!(matches!(
        BoundedInt::new(0, 10, 0),
        Err(BoundError::InvertedBounds { .. })
    ));
    // bounds escape M31_SIGNED.
    assert!(matches!(
        BoundedInt::new(0, 0, P_HALF + 1),
        Err(BoundError::BoundsEscapeSignedRange { .. })
    ));
}

#[test]
fn add_sub_mul_bound_propagation() {
    let a = BoundedInt::new(3, -5, 5).unwrap();
    let b = BoundedInt::new(-2, -4, 4).unwrap();

    let s = a.add(&b).unwrap();
    assert_eq!((s.value(), s.lo(), s.hi()), (1, -9, 9));

    let d = a.sub(&b).unwrap();
    assert_eq!((d.value(), d.lo(), d.hi()), (5, -9, 9)); // lo=a.lo-b.hi=-9, hi=a.hi-b.lo=9

    let m = a.mul(&b).unwrap();
    // corners of [-5,5]*[-4,4]: min=-20, max=20; value=3*-2=-6
    assert_eq!((m.value(), m.lo(), m.hi()), (-6, -20, 20));
}

#[test]
fn int8_hot_path_product_stays_in_range() {
    // a,b in [-127,127] -> prod in [-16129,16129] (RFC-0002 §4.2).
    let a = BoundedInt::new(127, -127, 127).unwrap();
    let b = BoundedInt::new(-127, -127, 127).unwrap();
    let m = a.mul(&b).unwrap();
    assert_eq!(m.value(), -16129);
    assert_eq!((m.lo(), m.hi()), (-16129, 16129));
}

#[test]
fn arithmetic_rejects_overflow_never_wraps() {
    // Two values bounded near P_HALF: their product's bound escapes M31_SIGNED.
    let big = BoundedInt::new(P_HALF, 0, P_HALF).unwrap();
    let err = big.mul(&big).unwrap_err();
    assert_eq!(err.op, "mul");
    // The reject is by bound, not a wrapped (mod p) value.
    assert!(err.hi > P_HALF as i128);

    // Addition that escapes the upper bound also rejects.
    let hi = BoundedInt::new(P_HALF, 0, P_HALF).unwrap();
    assert!(hi.add(&hi).is_err());
}

#[test]
fn compare_witnesses() {
    let a = BoundedInt::new(3, 0, 10).unwrap();
    let b = BoundedInt::new(7, 0, 10).unwrap();

    // a <= b: d = 4, bounds [0, b.hi - a.lo] = [0, 10].
    let le = a.le_witness(&b).unwrap();
    assert_eq!((le.value(), le.lo(), le.hi()), (4, 0, 10));

    // a < b: d = 3, bounds [0, 9].
    let lt = a.lt_witness(&b).unwrap();
    assert_eq!((lt.value(), lt.lo(), lt.hi()), (3, 0, 9));

    // b <= a is false -> no nonnegative witness.
    assert!(matches!(
        b.le_witness(&a),
        Err(CompareError::NotOrdered { a: 7, b: 3 })
    ));

    // equal values: <= holds (d=0), < does not.
    let c = BoundedInt::new(5, 0, 10).unwrap();
    assert_eq!(c.le_witness(&c).unwrap().value(), 0);
    assert!(matches!(
        c.lt_witness(&c),
        Err(CompareError::NotOrdered { .. })
    ));
}

#[test]
fn property_random_bounded_ops_stay_in_bounds_and_encode() {
    // Deterministic LCG (no RNG dependency) over small ranges so products stay in
    // M31_SIGNED; assert each result's value lies in its propagated bounds and
    // that the centered encoding round-trips.
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as i64
    };
    let pick = |n: i64, lim: i64| (n % (2 * lim + 1)) - lim;

    for _ in 0..20_000 {
        let av = pick(next(), 1000);
        let (al, ah) = (av - (next().rem_euclid(500)), av + (next().rem_euclid(500)));
        let bv = pick(next(), 1000);
        let (bl, bh) = (bv - (next().rem_euclid(500)), bv + (next().rem_euclid(500)));

        let a = BoundedInt::new(av, al, ah).unwrap();
        let b = BoundedInt::new(bv, bl, bh).unwrap();

        for r in [a.add(&b), a.sub(&b), a.mul(&b)] {
            let r = r.expect("small ranges stay in M31_SIGNED");
            assert!(r.lo() <= r.value() && r.value() <= r.hi());
            assert_eq!(decode(r.encode()), r.value());
        }
    }
}
