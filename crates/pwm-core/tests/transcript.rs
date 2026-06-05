// SPDX-License-Identifier: Apache-2.0
//! Tests for the native Fiat-Shamir transcript (specs.md §6, §7): prover and
//! verifier derive identical challenges from the same absorb history, the absorb
//! order is load-bearing, and any change to a public input changes the
//! challenges.

use pwm_core::field::M31;
use pwm_core::public_input::PublicInput;
use pwm_core::relation::StatementType;
use pwm_core::transcript::{init_transcript, Transcript, TRANSCRIPT_PROTOCOL_VERSION};

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
        goal_latent_public: Some(vec![M31::from_u32_unchecked(7)]),
        candidate_actions_commitment: Some([6u8; 32]),
        candidate_actions_public: None,
        claimed_output_commitment: [7u8; 32],
        selected_index: Some(3),
        selected_cost: None,
    }
}

/// Draw a short Freivalds challenge vector and an audit selector.
fn draw(t: &mut Transcript) -> (Vec<u64>, u64) {
    let r = t.challenge_fp61_vec(b"freivalds.r", 3);
    let sel = t.challenge_u64(b"audit.select");
    (r.iter().map(|f| f.0).collect(), sel)
}

#[test]
fn protocol_version_is_one() {
    assert_eq!(TRANSCRIPT_PROTOCOL_VERSION, 1);
}

#[test]
fn prover_and_verifier_derive_identical_challenges() {
    let pi = sample();
    let mut prover = init_transcript(&pi);
    let mut verifier = init_transcript(&pi);
    // Both absorb the same commitments in the same order before squeezing.
    for t in [&mut prover, &mut verifier] {
        t.absorb(b"trace_root", &[9u8; 32]);
    }
    assert_eq!(draw(&mut prover), draw(&mut verifier));
}

#[test]
fn changing_public_input_changes_challenges() {
    let base = {
        let mut t = init_transcript(&sample());
        draw(&mut t)
    };
    let mut other = sample();
    other.selected_index = Some(99);
    let changed = {
        let mut t = init_transcript(&other);
        draw(&mut t)
    };
    assert_ne!(changed, base);
}

#[test]
fn absorb_order_is_load_bearing() {
    let a = {
        let mut t = Transcript::new(b"t");
        t.absorb(b"x", &[1]);
        t.absorb(b"y", &[2]);
        draw(&mut t)
    };
    let b = {
        let mut t = Transcript::new(b"t");
        t.absorb(b"y", &[2]);
        t.absorb(b"x", &[1]);
        draw(&mut t)
    };
    assert_ne!(a, b, "a reordered transcript must diverge");
}

#[test]
fn sequential_squeezes_differ() {
    let mut t = Transcript::new(b"t");
    let first = t.challenge_fp61(b"r");
    let second = t.challenge_fp61(b"r");
    assert_ne!(first, second, "squeezing advances the transcript state");
}

#[test]
fn label_separates_challenges() {
    let mut t1 = Transcript::new(b"t");
    let mut t2 = Transcript::new(b"t");
    assert_ne!(t1.challenge_fp61(b"a"), t2.challenge_fp61(b"b"));
}
