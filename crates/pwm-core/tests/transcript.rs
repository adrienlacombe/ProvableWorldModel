// SPDX-License-Identifier: Apache-2.0
//! Tests for the Fiat-Shamir transcript (RFC-0014 §4): the shared T0–T3 absorb
//! ordering produces identical challenges on both sides, the ordering is
//! load-bearing, and any reorder changes the derived challenges.

use pwm_core::field::M31;
use pwm_core::public_input::PublicInput;
use pwm_core::relation::StatementType;
use pwm_core::transcript::{
    init_channel, public_input_digest, COMPONENT_SCHEDULE, TRANSCRIPT_PROTOCOL_VERSION,
};
use stwo::core::channel::{Blake2sChannel, Channel};

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

fn digest_u32s(pi: &PublicInput) -> [u32; 8] {
    let d = public_input_digest(pi);
    let mut out = [0u32; 8];
    for (i, w) in out.iter_mut().enumerate() {
        *w = u32::from_le_bytes([d[4 * i], d[4 * i + 1], d[4 * i + 2], d[4 * i + 3]]);
    }
    out
}

fn draw3(ch: &mut Blake2sChannel) -> [stwo::core::fields::qm31::SecureField; 3] {
    [
        ch.draw_secure_felt(),
        ch.draw_secure_felt(),
        ch.draw_secure_felt(),
    ]
}

#[test]
fn component_schedule_is_the_p2_order() {
    assert_eq!(COMPONENT_SCHEDULE.len(), 13);
    assert_eq!(COMPONENT_SCHEDULE[0], "range_check");
    assert_eq!(COMPONENT_SCHEDULE[1], "tensor_memory");
    assert_eq!(COMPONENT_SCHEDULE[12], "argmin");
    assert_eq!(TRANSCRIPT_PROTOCOL_VERSION, 1);
}

#[test]
fn prover_and_verifier_derive_identical_challenges() {
    let pi = sample();
    let sizes = [10u32, 9, 8, 8, 7];
    let mut prover = init_channel(&pi, &sizes);
    let mut verifier = init_channel(&pi, &sizes);
    assert_eq!(draw3(&mut prover), draw3(&mut verifier));
}

#[test]
fn changing_public_input_changes_challenges() {
    let sizes = [10u32, 9, 8];
    let base = draw3(&mut init_channel(&sample(), &sizes));

    let mut other = sample();
    other.selected_index = Some(99);
    assert_ne!(draw3(&mut init_channel(&other, &sizes)), base);
}

#[test]
fn changing_component_log_sizes_changes_challenges() {
    let pi = sample();
    let a = draw3(&mut init_channel(&pi, &[1, 2, 3]));
    let b = draw3(&mut init_channel(&pi, &[3, 2, 1]));
    let c = draw3(&mut init_channel(&pi, &[1, 2, 3, 4]));
    assert_ne!(a, b, "order of log sizes matters");
    assert_ne!(a, c, "count of log sizes matters");
}

#[test]
fn the_documented_absorb_order_is_what_init_channel_does() {
    let pi = sample();
    let sizes = [10u32, 9, 8];
    let pid = digest_u32s(&pi);

    // Rebuild the documented order T1 (version) -> T2 (pid) -> T3 (sizes).
    let mut correct = Blake2sChannel::default();
    correct.mix_u64(TRANSCRIPT_PROTOCOL_VERSION);
    correct.mix_u32s(&pid);
    correct.mix_u32s(&sizes);
    assert_eq!(
        draw3(&mut init_channel(&pi, &sizes)),
        draw3(&mut correct),
        "init_channel must match the documented T1->T2->T3 order"
    );
}

#[test]
fn reordered_transcript_yields_different_challenges() {
    let pi = sample();
    let sizes = [10u32, 9, 8];
    let pid = digest_u32s(&pi);

    let correct = draw3(&mut init_channel(&pi, &sizes));

    // Wrong order: absorb the pid before the protocol version.
    let mut wrong = Blake2sChannel::default();
    wrong.mix_u32s(&pid);
    wrong.mix_u64(TRANSCRIPT_PROTOCOL_VERSION);
    wrong.mix_u32s(&sizes);
    assert_ne!(
        draw3(&mut wrong),
        correct,
        "a reordered transcript must diverge"
    );

    // Wrong order: sizes before pid.
    let mut wrong2 = Blake2sChannel::default();
    wrong2.mix_u64(TRANSCRIPT_PROTOCOL_VERSION);
    wrong2.mix_u32s(&sizes);
    wrong2.mix_u32s(&pid);
    assert_ne!(draw3(&mut wrong2), correct);
}
