// SPDX-License-Identifier: Apache-2.0
//! End-to-end test for the autoregressive rollout (P1, specs.md §9): a predictor
//! maps a 2-latent window (dim 2 each -> input dim 4) to the next latent (dim 2),
//! rolled out over a horizon. The valid rollout verifies; a broken recurrence is
//! rejected.

use pwm_core::fixed_point::{BoundedInt, Rounding};
use pwm_core::tensor::{Dtype, Scale, Tensor};
use pwm_export::reference::{LayerSpec, Model};
use pwm_prover::{prove_rollout, OutputBinding};
use pwm_verifier::{verify_rollout, VerifyError};

fn weight(id: u32, rows: u32, cols: u32, vals: &[i8]) -> Tensor {
    let data = vals
        .iter()
        .map(|&v| BoundedInt::new(v as i64, -128, 127).unwrap())
        .collect();
    Tensor::new(id, vec![rows, cols], 0, data).unwrap()
}

/// Predictor: next = W · flatten(window), W = [[1,0,1,0],[0,1,0,1]] (sum the two
/// latents in the window). Requant is identity (shift 0).
fn predictor() -> Model {
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
    ];
    let layer = LayerSpec {
        pre_layernorm: None,
        linear_op_id: 1,
        weight: weight(10, 2, 4, &[1, 0, 1, 0, 0, 1, 0, 1]),
        bias: None,
        requant_op_id: 2,
        shift: 0,
        zero_point: 0,
        clamp_lo: -128,
        clamp_hi: 127,
        rounding: Rounding::NearestTiesToEven,
        activation: None,
    };
    Model {
        layers: vec![layer],
        tables: vec![],
        scales,
    }
}

fn out_binding() -> OutputBinding {
    OutputBinding {
        tensor_id: 200,
        scale_id: 1,
    }
}

#[test]
fn accept_valid_rollout() {
    let init = vec![vec![1, 2], vec![3, 4]];
    let proof = prove_rollout(&predictor(), &init, 2, 2, out_binding()).unwrap();
    // step0: window [1,2,3,4] -> [1+3, 2+4] = [4,6].
    // step1: window [3,4,4,6] -> [3+4, 4+6] = [7,10].
    assert_eq!(proof.trajectory, vec![vec![4, 6], vec![7, 10]]);
    assert_eq!(verify_rollout(&proof), Ok(()));
}

#[test]
fn reject_tampered_trajectory() {
    let init = vec![vec![1, 2], vec![3, 4]];
    let mut proof = prove_rollout(&predictor(), &init, 2, 2, out_binding()).unwrap();
    // Corrupt the recorded trajectory for step 0 (without re-proving) -> the
    // step's verified output no longer matches the trajectory entry.
    proof.trajectory[0] = vec![99, 99];
    assert_eq!(
        verify_rollout(&proof),
        Err(VerifyError::RolloutWiring { step: 0 })
    );
}

#[test]
fn reject_broken_recurrence() {
    let init = vec![vec![1, 2], vec![3, 4]];
    let mut proof = prove_rollout(&predictor(), &init, 2, 2, out_binding()).unwrap();
    // Swap the two steps so step 1's window no longer follows from step 0.
    proof.steps.swap(0, 1);
    proof.trajectory.swap(0, 1);
    assert!(matches!(
        verify_rollout(&proof),
        Err(VerifyError::RolloutWiring { .. })
    ));
}
