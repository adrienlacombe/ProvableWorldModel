// SPDX-License-Identifier: Apache-2.0
//! Tests for the centered signed-integer encoding (RFC-0002 §2): encode/decode
//! round-trips on the representable interval, the locked boundary values, and the
//! out-of-range rejection.

use pwm_core::field::{
    decode, encode, in_signed_range, try_encode, M31, M31_SIGNED_HI, M31_SIGNED_LO, P, P_HALF,
};

#[test]
fn modulus_and_bounds_match_rfc0002() {
    assert_eq!(P, 2_147_483_647); // 2^31 - 1
    assert_eq!(P_HALF, 1_073_741_823); // (p - 1) / 2
    assert_eq!(M31_SIGNED_LO, -1_073_741_823);
    assert_eq!(M31_SIGNED_HI, 1_073_741_823);
}

#[test]
fn locked_boundary_encodings() {
    // RFC-0002 §2: enc(0) = 0, enc(-1) = p - 1, enc(-P_HALF) = P_HALF + 1.
    assert_eq!(encode(0).0, 0);
    assert_eq!(encode(-1).0, P - 1);
    assert_eq!(encode(1).0, 1);
    assert_eq!(encode(M31_SIGNED_LO).0, (P_HALF as u32) + 1);
    assert_eq!(encode(M31_SIGNED_HI).0, P_HALF as u32);
}

#[test]
fn round_trip_exhaustive_near_zero_and_boundaries() {
    // Exhaustive over a dense band around zero plus the exact interval edges.
    let b: i64 = 300_000;
    for x in -b..=b {
        assert_eq!(decode(encode(x)), x, "round-trip failed at {x}");
    }
    for x in [
        M31_SIGNED_LO,
        M31_SIGNED_LO + 1,
        -1,
        0,
        1,
        M31_SIGNED_HI - 1,
        M31_SIGNED_HI,
    ] {
        assert_eq!(decode(encode(x)), x, "round-trip failed at boundary {x}");
    }
}

#[test]
fn round_trip_sampled_full_interval() {
    // Deterministic stride across the whole representable interval (no RNG dep):
    // step is coprime-ish to the interval so it visits both halves densely.
    let step: i64 = 2_718_281; // ~e * 1e6, an arbitrary large stride
    let mut x = M31_SIGNED_LO;
    let mut count = 0u64;
    while x <= M31_SIGNED_HI {
        assert_eq!(decode(encode(x)), x, "round-trip failed at sampled {x}");
        // Encoded value is always canonical M31 in [0, p).
        assert!(encode(x).0 < P);
        x = x.saturating_add(step);
        count += 1;
    }
    assert!(count > 700, "expected a dense sweep, got {count} samples");
}

#[test]
fn try_encode_rejects_out_of_range() {
    assert!(in_signed_range(M31_SIGNED_HI));
    assert!(!in_signed_range(M31_SIGNED_HI + 1));
    assert!(!in_signed_range(M31_SIGNED_LO - 1));

    let too_high = try_encode(M31_SIGNED_HI + 1).expect_err("must reject");
    assert_eq!(too_high.value, M31_SIGNED_HI + 1);
    assert_eq!(too_high.hi, M31_SIGNED_HI);

    let too_low = try_encode(M31_SIGNED_LO - 1).expect_err("must reject");
    assert_eq!(too_low.value, M31_SIGNED_LO - 1);

    // In range: matches the unchecked encode.
    assert_eq!(try_encode(42).unwrap().0, encode(42).0);
}

#[test]
fn encode_produces_base_field_not_secure_field() {
    // The encoding maps to base M31; QM31 (the secure field) never carries a
    // quantized value (RFC-0002). This is a type-level guarantee — `encode`
    // returns `M31` — exercised here so a refactor that widened the return type
    // to a secure-field value would fail to compile.
    let v: M31 = encode(123);
    assert_eq!(decode(v), 123);
}
