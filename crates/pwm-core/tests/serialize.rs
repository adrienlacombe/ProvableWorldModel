// SPDX-License-Identifier: Apache-2.0
//! Tests for canonical serialization (RFC-0014 §1): determinism, round-trip,
//! committed golden byte vectors, and rejection of non-canonical encodings.

use pwm_core::field::M31;
use pwm_core::fixed_point::BoundedInt;
use pwm_core::serialize::{
    canonical_bytes, from_canonical_bytes, CanonicalDecode, CanonicalEncode, DecodeError,
};
use pwm_core::tensor::Tensor;

fn roundtrip<T>(value: &T)
where
    T: CanonicalEncode + CanonicalDecode + PartialEq + core::fmt::Debug,
{
    let bytes = canonical_bytes(value);
    // Determinism: re-encoding yields identical bytes.
    assert_eq!(
        bytes,
        canonical_bytes(value),
        "encoding is not deterministic"
    );
    // Round-trip: decode reproduces the original exactly.
    let decoded: T = from_canonical_bytes(&bytes).expect("decodes");
    assert_eq!(&decoded, value, "round-trip mismatch");
}

#[test]
fn committed_golden_vectors() {
    // Hand-computed canonical encodings (RFC-0014 §1). These bytes are the
    // committed golden serialization vectors; a change here is a relation event.
    assert_eq!(canonical_bytes(&5u32), vec![0x05, 0, 0, 0]);
    assert_eq!(
        canonical_bytes(&(-5i64)),
        vec![0xFB, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]
    );
    assert_eq!(canonical_bytes(&true), vec![0x01]);
    assert_eq!(canonical_bytes(&false), vec![0x00]);
    assert_eq!(
        canonical_bytes(&M31::from_u32_unchecked(7)),
        vec![0x07, 0, 0, 0]
    );
    assert_eq!(canonical_bytes(&Option::<u32>::None), vec![0x00]);
    assert_eq!(canonical_bytes(&Some(5u32)), vec![0x01, 0x05, 0, 0, 0]);
    assert_eq!(
        canonical_bytes(&vec![1u32, 2u32]),
        vec![0x02, 0, 0, 0, 0x01, 0, 0, 0, 0x02, 0, 0, 0]
    );
    assert_eq!(
        canonical_bytes(&"ab".to_string()),
        vec![0x02, 0, 0, 0, 0x61, 0x62]
    );
    // BoundedInt(3, -5, 5) = value(3) || lo(-5) || hi(5), each i64 LE.
    let bi = BoundedInt::new(3, -5, 5).unwrap();
    let mut expected = Vec::new();
    expected.extend_from_slice(&3i64.to_le_bytes());
    expected.extend_from_slice(&(-5i64).to_le_bytes());
    expected.extend_from_slice(&5i64.to_le_bytes());
    assert_eq!(canonical_bytes(&bi), expected);
}

#[test]
fn round_trips() {
    roundtrip(&0u32);
    roundtrip(&u64::MAX);
    roundtrip(&(-123_456_789i64));
    roundtrip(&true);
    roundtrip(&[0xABu8; 32]);
    roundtrip(&M31::from_u32_unchecked(2_147_483_646)); // P - 1, canonical max
    roundtrip(&Some(42u32));
    roundtrip(&Option::<u32>::None);
    roundtrip(&vec![1u32, 2, 3, 4]);
    roundtrip(&"hello world".to_string());
    roundtrip(&BoundedInt::new(-7, -10, 0).unwrap());

    let data = (0..6).map(|v| BoundedInt::new(v, -8, 8).unwrap()).collect();
    roundtrip(&Tensor::new(11, vec![2, 3], 1, data).unwrap());
}

#[test]
fn rejects_invalid_bool() {
    assert_eq!(
        from_canonical_bytes::<bool>(&[0x02]),
        Err(DecodeError::InvalidBool(2))
    );
}

#[test]
fn rejects_non_canonical_m31() {
    // residue == P (2147483647 = 0x7FFFFFFF) is not canonical (must be < P).
    let p_bytes = 2_147_483_647u32.to_le_bytes();
    assert_eq!(
        from_canonical_bytes::<M31>(&p_bytes),
        Err(DecodeError::NonCanonicalM31(2_147_483_647))
    );
}

#[test]
fn rejects_invalid_option_tag() {
    assert_eq!(
        from_canonical_bytes::<Option<u32>>(&[0x02, 0, 0, 0, 0]),
        Err(DecodeError::InvalidOptionTag(2))
    );
}

#[test]
fn rejects_trailing_and_truncated() {
    // trailing byte after a complete u32
    assert_eq!(
        from_canonical_bytes::<u32>(&[0x01, 0, 0, 0, 0x99]),
        Err(DecodeError::TrailingBytes { remaining: 1 })
    );
    // truncated u32
    assert!(matches!(
        from_canonical_bytes::<u32>(&[0x01, 0x00]),
        Err(DecodeError::UnexpectedEof { .. })
    ));
}

#[test]
fn rejects_non_canonical_bounded_int() {
    // value || lo || hi with lo > hi is a non-canonical BoundedInt.
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0i64.to_le_bytes()); // value
    bytes.extend_from_slice(&5i64.to_le_bytes()); // lo
    bytes.extend_from_slice(&0i64.to_le_bytes()); // hi  (lo > hi)
    assert!(matches!(
        from_canonical_bytes::<BoundedInt>(&bytes),
        Err(DecodeError::Bound(_))
    ));
}

#[test]
fn rejects_tensor_len_shape_mismatch() {
    // Craft a Tensor encoding whose declared len != product(shape).
    let mut bytes = Vec::new();
    0u32.encode(&mut bytes); // tensor_id
    0u32.encode(&mut bytes); // scale_id
    1u32.encode(&mut bytes); // rank = 1
    5u32.encode(&mut bytes); // shape[0] = 5  -> product 5
    3u32.encode(&mut bytes); // len = 3  (mismatch)
    for v in 0..3i64 {
        BoundedInt::new(v, 0, 10).unwrap().encode(&mut bytes);
    }
    assert!(matches!(
        from_canonical_bytes::<Tensor>(&bytes),
        Err(DecodeError::Tensor(_))
    ));
}
