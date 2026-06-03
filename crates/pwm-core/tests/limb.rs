// SPDX-License-Identifier: Apache-2.0
//! Tests for accumulator representation selection and limb decomposition
//! (RFC-0002 §7): the safe single-M31 path and the two-limb path.

use pwm_core::fixed_point::{accumulate, BoundedInt};
use pwm_core::limb::{
    decompose, needs_limbs, reconstruct, select_accum_repr, AccumRepr, LimbError, DEFAULT_LIMB_K,
    SAFE_HI,
};

#[test]
fn v0_mlp_dot_product_fits_single_m31() {
    // The widest V0 dot product: 2048 int8 products of 127*127 each.
    // 2048 * 127 * 127 = 33_032_192 < 2^31 - 1, so it stays single-M31.
    let worst = 2048i64 * 127 * 127;
    assert_eq!(worst, 33_032_192);
    assert!(worst < SAFE_HI);
    assert!(!needs_limbs(-worst, worst));
    assert_eq!(
        select_accum_repr(-worst, worst, DEFAULT_LIMB_K).unwrap(),
        AccumRepr::SingleM31
    );

    // And the actual accumulate fold stays in a single value.
    let prod = BoundedInt::new(127 * 127, -16129, 16129).unwrap();
    let terms = vec![prod; 2048];
    let bias = BoundedInt::new(0, 0, 0).unwrap();
    let acc = accumulate(&bias, &terms).unwrap();
    assert_eq!(acc.value(), worst);
}

#[test]
fn accumulate_rejects_when_running_bound_escapes() {
    // A term whose repeated addition pushes the bound past SAFE_HI must reject,
    // never wrap.
    let big = BoundedInt::new(SAFE_HI, 0, SAFE_HI).unwrap();
    let bias = BoundedInt::new(0, 0, SAFE_HI).unwrap();
    assert!(accumulate(&bias, &[big, big]).is_err());
}

#[test]
fn selects_limbs_when_bound_exceeds_safe_interval() {
    // A bound beyond SAFE_HI (e.g. an int32-ranged accumulator) selects two limbs.
    let hi = SAFE_HI + 1_000_000;
    assert!(needs_limbs(-hi, hi));
    assert_eq!(
        select_accum_repr(-hi, hi, DEFAULT_LIMB_K).unwrap(),
        AccumRepr::TwoLimb { k: DEFAULT_LIMB_K }
    );
}

#[test]
fn decompose_reconstructs_exactly() {
    let k = DEFAULT_LIMB_K;
    let pow = 1i64 << k;
    let lo = -(SAFE_HI + 500_000);
    let hi = SAFE_HI + 500_000;
    // Every probe value must lie within the declared [lo, hi].
    for value in [lo, -pow - 1, -1, 0, 1, pow, pow + 7, 1_000_000_000, hi] {
        let limbs = decompose(value, lo, hi, k).unwrap();
        // low limb is in [0, 2^K - 1]; high limb is range-checked.
        assert!(0 <= limbs.low.value() && limbs.low.value() < pow);
        assert_eq!(limbs.low.lo(), 0);
        assert_eq!(limbs.low.hi(), pow - 1);
        // exact reconstruction (INV-FP-10)
        assert_eq!(reconstruct(&limbs), value, "reconstruct failed at {value}");
    }
}

#[test]
fn rejects_invalid_k_and_more_than_two_limbs() {
    // K out of range.
    assert_eq!(
        select_accum_repr(-(SAFE_HI + 1), SAFE_HI + 1, 0),
        Err(LimbError::InvalidK { k: 0 })
    );
    assert!(matches!(
        decompose(10, 0, 100, 40),
        Err(LimbError::InvalidK { .. })
    ));

    // A value whose high limb would escape M31_SIGNED needs > 2 limbs. With a
    // small K the high limb grows fast: at K = 1, high = hi / 2, so a hi bound
    // just over 2 * SAFE_HI forces high > SAFE_HI -> ExceedsTwoLimbs.
    let hi = SAFE_HI * 2 + 4; // high = hi / 2 = SAFE_HI + 2 > SAFE_HI
    assert!(matches!(
        select_accum_repr(-hi, hi, 1),
        Err(LimbError::ExceedsTwoLimbs { .. })
    ));
}
