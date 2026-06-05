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
use pwm_prover::{prove_feedforward, OutputBinding};
use pwm_verifier::{verify, VerifyError};

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
