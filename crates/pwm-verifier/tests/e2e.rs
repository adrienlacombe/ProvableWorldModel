// SPDX-License-Identifier: Apache-2.0
//! End-to-end accept/reject tests for the commit-and-audit slice (specs.md §15):
//! a quantized feed-forward model is proven by `pwm-prover` and checked by
//! `pwm-verifier`. The accepting case verifies; each tampering case is rejected
//! with a specific, correct `VerifyError`.

use pwm_core::fixed_point::{BoundedInt, Rounding};
use pwm_core::tables::ActivationTable;
use pwm_core::tensor::{Dtype, Scale, Tensor};
use pwm_core::trace::OpRecord;
use pwm_export::reference::{LayerSpec, Model};
use pwm_prover::{prove_feedforward, prove_planning, OutputBinding};
use pwm_verifier::{
    verify, verify_interactive, verify_planning, verify_planning_batched, verify_sampled,
    VerifyError,
};

fn weight(id: u32, rows: u32, cols: u32, vals: &[i8]) -> Tensor {
    let data = vals
        .iter()
        .map(|&v| BoundedInt::new(v as i64, -128, 127).unwrap())
        .collect();
    Tensor::new(id, vec![rows, cols], 0, data).unwrap()
}

fn bias(id: u32, vals: &[i32]) -> Tensor {
    // Bounds must fit M31_SIGNED (±2^30); real biases are small, so bound exactly.
    let data = vals
        .iter()
        .map(|&v| BoundedInt::exact(v as i64).unwrap())
        .collect();
    Tensor::new(id, vec![vals.len() as u32], 2, data).unwrap()
}

/// Identity table over the full int8 domain (exercises the activation path).
fn identity_table(table_id: u32) -> ActivationTable {
    ActivationTable {
        table_id,
        lo: -128,
        outputs: (-128i64..=127).collect(),
    }
}

fn model() -> Model {
    let scales = vec![
        Scale {
            scale_id: 0,
            log2: 0,
            dtype: Dtype::I8,
        },
        Scale {
            scale_id: 1,
            log2: 0,
            dtype: Dtype::I8,
        },
        Scale {
            scale_id: 2,
            log2: 0,
            dtype: Dtype::I32,
        },
    ];
    let l1 = LayerSpec {
        pre_layernorm: None,
        linear_op_id: 100,
        weight: weight(10, 4, 3, &[1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1]),
        bias: Some(bias(11, &[1, -1, 0, 2])),
        requant_op_id: 101,
        shift: 0,
        zero_point: 0,
        clamp_lo: -128,
        clamp_hi: 127,
        rounding: Rounding::NearestTiesToEven,
        activation: Some((102, 0)),
    };
    let l2 = LayerSpec {
        pre_layernorm: None,
        linear_op_id: 103,
        weight: weight(12, 2, 4, &[1, 1, 1, 1, 1, -1, 1, -1]),
        bias: None,
        requant_op_id: 104,
        shift: 2,
        zero_point: 0,
        clamp_lo: -128,
        clamp_hi: 127,
        rounding: Rounding::NearestTiesToEven,
        activation: None,
    };
    Model {
        layers: vec![l1, l2],
        tables: vec![identity_table(0)],
        scales,
    }
}

fn out_binding() -> OutputBinding {
    OutputBinding {
        tensor_id: 200,
        scale_id: 1,
    }
}

fn input() -> Vec<i64> {
    vec![1, 2, 3]
}

#[test]
fn golden_vector_regression() {
    // Golden vector (T-701): the committed quantized model's artifact is pinned by
    // its exact output and the digest of its canonical bytes. A change to the
    // serialization, the model, or the proof shape changes this digest.
    use pwm_core::serialize::canonical_bytes;
    use pwm_core::transcript::blake2s256;
    let a = prove_feedforward(&model(), &input(), out_binding()).unwrap();
    let out: Vec<i64> = a.claimed_output.data().iter().map(|c| c.value()).collect();
    assert_eq!(out, vec![4, -1], "golden output");
    let digest = blake2s256(&canonical_bytes(&a));
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(
        hex, "7b88ea77a41ca3c4329c08649dd9e2a5188fe216e5a9c96d5f5e172a313ebff2",
        "golden artifact digest changed"
    );
}

#[test]
fn artifact_roundtrips_and_reproduces() {
    use pwm_core::audit::AuditArtifact;
    use pwm_core::serialize::{canonical_bytes, from_canonical_bytes};

    let a = prove_feedforward(&model(), &input(), out_binding()).unwrap();
    // Reproducibility (T-706): proving the same inputs twice is byte-identical.
    let a2 = prove_feedforward(&model(), &input(), out_binding()).unwrap();
    assert_eq!(canonical_bytes(&a), canonical_bytes(&a2));
    // Binary round-trip (PR-303): encode -> decode is the identity and still verifies.
    let bytes = canonical_bytes(&a);
    let decoded: AuditArtifact = from_canonical_bytes(&bytes).unwrap();
    assert_eq!(decoded, a);
    assert_eq!(verify(&decoded), Ok(()));
}

#[test]
fn accept_interactive_verifier_secret_mode() {
    use pwm_core::field::Fp61;
    let a = prove_feedforward(&model(), &input(), out_binding()).unwrap();
    // One verifier-secret challenge per Linear op in trace order (rows 4 then 2).
    let secret_r = vec![
        vec![Fp61::new(7), Fp61::new(11), Fp61::new(13), Fp61::new(17)],
        vec![Fp61::new(19), Fp61::new(23)],
    ];
    assert_eq!(verify_interactive(&a, &secret_r), Ok(()));

    // A tampered accumulator is rejected even with verifier-secret challenges.
    let mut b = a.clone();
    let idx = b
        .trace
        .iter()
        .position(|r| matches!(r, OpRecord::Linear(_)))
        .unwrap();
    if let OpRecord::Linear(r) = &mut b.trace[idx] {
        r.output[0] += 1;
    }
    assert!(matches!(
        verify_interactive(&b, &secret_r),
        Err(VerifyError::FreivaldsCheckFailed { .. })
    ));
}

#[test]
fn sampled_audit_full_and_zero_coverage() {
    let a = prove_feedforward(&model(), &input(), out_binding()).unwrap();
    // Full coverage: all (both) linear ops are Freivalds-checked.
    assert_eq!(verify_sampled(&a, 100, 7), Ok(2));
    // Zero coverage: no linear is Freivalds-checked, but structure/output still
    // pass on an honest proof.
    assert_eq!(verify_sampled(&a, 0, 7), Ok(0));
}

#[test]
fn sampled_audit_full_coverage_rejects_tamper() {
    let mut a = prove_feedforward(&model(), &input(), out_binding()).unwrap();
    let idx = a
        .trace
        .iter()
        .position(|r| matches!(r, OpRecord::Linear(_)))
        .unwrap();
    if let OpRecord::Linear(r) = &mut a.trace[idx] {
        r.output[0] += 1;
    }
    // At full coverage the tampered accumulator is Freivalds-rejected.
    assert!(matches!(
        verify_sampled(&a, 100, 7),
        Err(VerifyError::FreivaldsCheckFailed { .. })
    ));
}

#[test]
fn accept_valid_proof() {
    let artifact = prove_feedforward(&model(), &input(), out_binding()).unwrap();
    // Expected output: L1 -> [2,1,3,8] -> id -> L2 -> [14,-4] -> requant>>2 -> [4,-1].
    assert_eq!(
        artifact
            .claimed_output
            .data()
            .iter()
            .map(|c| c.value())
            .collect::<Vec<_>>(),
        vec![4, -1]
    );
    assert_eq!(verify(&artifact), Ok(()));
}

#[test]
fn reject_tampered_weight() {
    let mut a = prove_feedforward(&model(), &input(), out_binding()).unwrap();
    // Mutate a weight value -> weights_root changes -> model commitment mismatch.
    let w = &mut a.weights[0];
    let mut data: Vec<BoundedInt> = w.data().to_vec();
    data[0] = BoundedInt::new(7, -128, 127).unwrap();
    *w = Tensor::new(w.tensor_id(), w.shape().to_vec(), w.scale_id(), data).unwrap();
    assert_eq!(verify(&a), Err(VerifyError::CommitmentMismatch("model")));
}

#[test]
fn reject_tampered_linear_accumulator() {
    let mut a = prove_feedforward(&model(), &input(), out_binding()).unwrap();
    // Mutate a Linear accumulator -> Freivalds fails.
    let idx = a
        .trace
        .iter()
        .position(|r| matches!(r, OpRecord::Linear(_)))
        .unwrap();
    if let OpRecord::Linear(r) = &mut a.trace[idx] {
        r.output[0] += 1;
    }
    assert!(matches!(
        verify(&a),
        Err(VerifyError::FreivaldsCheckFailed { .. })
    ));
}

#[test]
fn reject_tampered_requant_output() {
    let mut a = prove_feedforward(&model(), &input(), out_binding()).unwrap();
    let idx = a
        .trace
        .iter()
        .position(|r| matches!(r, OpRecord::Requant(_)))
        .unwrap();
    if let OpRecord::Requant(r) = &mut a.trace[idx] {
        r.output[0] += 1;
    }
    // The next op's input no longer matches this op's output (wiring), or the
    // exact replay fails — either is a sound rejection.
    assert!(matches!(
        verify(&a),
        Err(VerifyError::ExactReplayMismatch { .. }) | Err(VerifyError::WiringMismatch { .. })
    ));
}

#[test]
fn reject_tampered_claimed_output() {
    let mut a = prove_feedforward(&model(), &input(), out_binding()).unwrap();
    let mut data: Vec<BoundedInt> = a.claimed_output.data().to_vec();
    data[0] = BoundedInt::exact(99).unwrap();
    a.claimed_output = Tensor::new(
        a.claimed_output.tensor_id(),
        a.claimed_output.shape().to_vec(),
        a.claimed_output.scale_id(),
        data,
    )
    .unwrap();
    assert_eq!(verify(&a), Err(VerifyError::OutputMismatch));
}

#[test]
fn reject_wrong_relation() {
    let mut a = prove_feedforward(&model(), &input(), out_binding()).unwrap();
    a.public_input.relation_id = [0u8; 32];
    assert_eq!(verify(&a), Err(VerifyError::UnsupportedRelation));
}

#[test]
fn reject_tampered_public_input() {
    let mut a = prove_feedforward(&model(), &input(), out_binding()).unwrap();
    // Change the public latent history -> first input no longer matches the trace.
    if let Some(hist) = a.public_input.latent_history_public.as_mut() {
        hist[0] = pwm_core::field::encode(42);
    }
    assert_eq!(verify(&a), Err(VerifyError::PublicInputMismatch));
}

// --- P2 fixed-candidate planning (the V0 headline) ---

fn candidates() -> Vec<Vec<i64>> {
    // costs vs goal [4,-1]: cand0 -> [4,-1] cost 0; cand1 -> [0,0] cost 17;
    // cand2 -> [4,-1] cost 0 (ties cand0, but cand0 is earlier).
    vec![vec![1, 2, 3], vec![0, 0, 0], vec![3, 2, 1]]
}

fn goal() -> Vec<i64> {
    vec![4, -1]
}

#[test]
fn accept_valid_planning_proof() {
    let proof = prove_planning(&model(), &candidates(), &goal(), out_binding()).unwrap();
    assert_eq!(proof.costs, vec![0, 17, 0]);
    assert_eq!(proof.selected_index, 0); // first minimum
    assert_eq!(proof.selected_cost, 0);
    assert_eq!(verify_planning(&proof), Ok(()));
}

#[test]
fn accept_batched_planning_equivalent_to_per_candidate() {
    let proof = prove_planning(&model(), &candidates(), &goal(), out_binding()).unwrap();
    // Batched (amortized) verification agrees with per-candidate verification.
    assert_eq!(verify_planning(&proof), Ok(()));
    assert_eq!(verify_planning_batched(&proof), Ok(()));
}

#[test]
fn reject_batched_planning_tampered_candidate() {
    let mut proof = prove_planning(&model(), &candidates(), &goal(), out_binding()).unwrap();
    let idx = proof.candidates[1]
        .trace
        .iter()
        .position(|r| matches!(r, OpRecord::Linear(_)))
        .unwrap();
    if let OpRecord::Linear(r) = &mut proof.candidates[1].trace[idx] {
        r.output[0] += 1;
    }
    assert!(matches!(
        verify_planning_batched(&proof),
        Err(VerifyError::Candidate { index: 1, .. })
    ));
}

#[test]
fn reject_planning_tie_break_violation() {
    let mut proof = prove_planning(&model(), &candidates(), &goal(), out_binding()).unwrap();
    // Candidate 2 also has cost 0, but candidate 0 is the earliest minimum.
    proof.selected_index = 2;
    proof.selected_cost = proof.costs[2];
    assert_eq!(verify_planning(&proof), Err(VerifyError::ArgminViolation));
}

#[test]
fn reject_planning_not_minimum() {
    let mut proof = prove_planning(&model(), &candidates(), &goal(), out_binding()).unwrap();
    proof.selected_index = 1; // cost 17 is not the minimum
    proof.selected_cost = proof.costs[1];
    assert_eq!(verify_planning(&proof), Err(VerifyError::ArgminViolation));
}

#[test]
fn reject_planning_forged_cost() {
    let mut proof = prove_planning(&model(), &candidates(), &goal(), out_binding()).unwrap();
    // Claim candidate 1 is cheap without changing its (verified) output.
    proof.costs[1] = -5;
    assert!(matches!(
        verify_planning(&proof),
        Err(VerifyError::CostMismatch { index: 1 })
    ));
}

#[test]
fn reject_planning_tampered_candidate() {
    let mut proof = prove_planning(&model(), &candidates(), &goal(), out_binding()).unwrap();
    // Tamper a candidate's claimed output -> its P0 proof fails to verify.
    let data = vec![
        BoundedInt::exact(123).unwrap(),
        BoundedInt::exact(0).unwrap(),
    ];
    let co = &proof.candidates[1].claimed_output;
    proof.candidates[1].claimed_output =
        Tensor::new(co.tensor_id(), co.shape().to_vec(), co.scale_id(), data).unwrap();
    assert!(matches!(
        verify_planning(&proof),
        Err(VerifyError::Candidate { index: 1, .. })
    ));
}
