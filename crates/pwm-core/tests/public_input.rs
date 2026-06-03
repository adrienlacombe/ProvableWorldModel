// SPDX-License-Identifier: Apache-2.0
//! Tests for the public input and its digest (RFC-0014 §2): determinism,
//! field-sensitivity (changing any public field changes the digest), canonical
//! round-trip, and the mismatch-detection basis for
//! `VerifyError::PublicInputDigestMismatch` (#63).

use pwm_core::field::M31;
use pwm_core::fixed_point::BoundedInt;
use pwm_core::public_input::PublicInput;
use pwm_core::relation::StatementType;
use pwm_core::serialize::{canonical_bytes, from_canonical_bytes};
use pwm_core::transcript::{blake2s256, public_input_digest, PID_DOMAIN_TAG};

fn sample() -> PublicInput {
    PublicInput {
        relation_id: [1u8; 32],
        model_commitment: [2u8; 32],
        quantization_commitment: [3u8; 32],
        planner_config_commitment: [4u8; 32],
        statement_type: StatementType::P2FixedCandidatePlanning,
        latent_history_commitment: Some([5u8; 32]),
        latent_history_public: None,
        goal_latent_commitment: None,
        goal_latent_public: Some(vec![M31::from_u32_unchecked(7), M31::from_u32_unchecked(9)]),
        candidate_actions_commitment: Some([6u8; 32]),
        candidate_actions_public: None,
        claimed_output_commitment: [7u8; 32],
        selected_index: Some(3),
        selected_cost: Some(BoundedInt::new(42, 0, 1000).unwrap()),
    }
}

#[test]
fn pid_domain_tag_is_16_bytes() {
    assert_eq!(PID_DOMAIN_TAG.len(), 16);
    assert_eq!(&PID_DOMAIN_TAG[..10], b"pwm.pid.v1");
}

#[test]
fn digest_is_stable_across_runs() {
    let pi = sample();
    let a = public_input_digest(&pi);
    let b = public_input_digest(&sample());
    assert_eq!(a, b, "digest must be deterministic");
    // And it is the domain-tagged hash of the canonical bytes.
    let mut preimage = PID_DOMAIN_TAG.to_vec();
    preimage.extend_from_slice(&canonical_bytes(&pi));
    assert_eq!(a, blake2s256(&preimage));
}

#[test]
fn changing_any_public_field_changes_the_digest() {
    let base = public_input_digest(&sample());

    let mut variants: Vec<PublicInput> = Vec::new();
    let mut v;

    v = sample();
    v.relation_id[0] ^= 0xFF;
    variants.push(v);
    v = sample();
    v.model_commitment[5] ^= 1;
    variants.push(v);
    v = sample();
    v.quantization_commitment[0] ^= 1;
    variants.push(v);
    v = sample();
    v.planner_config_commitment[31] ^= 1;
    variants.push(v);
    v = sample();
    v.statement_type = StatementType::P3Cem;
    variants.push(v);
    v = sample();
    v.latent_history_commitment = Some([0x55; 32]);
    variants.push(v);
    v = sample();
    v.goal_latent_public = Some(vec![
        M31::from_u32_unchecked(7),
        M31::from_u32_unchecked(10),
    ]);
    variants.push(v);
    v = sample();
    v.candidate_actions_commitment = None; // role flip
    variants.push(v);
    v = sample();
    v.claimed_output_commitment[0] ^= 1;
    variants.push(v);
    v = sample();
    v.selected_index = Some(4);
    variants.push(v);
    v = sample();
    v.selected_cost = Some(BoundedInt::new(43, 0, 1000).unwrap());
    variants.push(v);
    v = sample();
    v.selected_cost = Some(BoundedInt::new(42, 0, 999).unwrap()); // bounds are load-bearing
    variants.push(v);

    for (i, variant) in variants.iter().enumerate() {
        assert_ne!(
            public_input_digest(variant),
            base,
            "variant {i} did not change the digest"
        );
    }
}

#[test]
fn public_input_round_trips() {
    let pi = sample();
    let bytes = canonical_bytes(&pi);
    let decoded: PublicInput = from_canonical_bytes(&bytes).expect("decodes");
    assert_eq!(decoded, pi);
}

#[test]
fn mismatched_digest_is_detected() {
    // The basis for VerifyError::PublicInputDigestMismatch (#63): a digest
    // recomputed over tampered public input differs from the one a proof carried.
    let claimed = public_input_digest(&sample());
    let mut tampered = sample();
    tampered.selected_index = Some(99);
    let recomputed = public_input_digest(&tampered);
    assert_ne!(recomputed, claimed);
}

#[test]
fn statement_type_discriminant_round_trips() {
    for st in [
        StatementType::P0Step,
        StatementType::P1Rollout,
        StatementType::P2FixedCandidatePlanning,
        StatementType::P3Cem,
        StatementType::P4PixelToPlan,
    ] {
        let bytes = canonical_bytes(&st);
        assert_eq!(bytes.len(), 1);
        assert_eq!(from_canonical_bytes::<StatementType>(&bytes).unwrap(), st);
    }
    // unknown discriminant rejected
    assert!(from_canonical_bytes::<StatementType>(&[9]).is_err());
}
