// SPDX-License-Identifier: Apache-2.0
//! End-to-end test for the integrated LayerNorm predictor op (specs.md §8.3):
//! a model with a pre-LayerNorm (affine-free, committed inverse-sqrt table) ->
//! Linear -> Requant is proven and verified; tampering the normalized output is
//! rejected by exact replay.

use pwm_core::fixed_point::{BoundedInt, Rounding};
use pwm_core::tables::ActivationTable;
use pwm_core::tensor::{Dtype, Scale, Tensor};
use pwm_core::trace::OpRecord;
use pwm_export::reference::{LayerNormSpec, LayerSpec, Model};
use pwm_prover::{prove_feedforward, OutputBinding};
use pwm_verifier::{verify, VerifyError};

fn weight(id: u32, rows: u32, cols: u32, vals: &[i8]) -> Tensor {
    let data = vals
        .iter()
        .map(|&v| BoundedInt::new(v as i64, -128, 127).unwrap())
        .collect();
    Tensor::new(id, vec![rows, cols], 0, data).unwrap()
}

/// Identity inverse-sqrt table over a wide variance domain (so the test math is
/// transparent: inv_std == var); and an identity passthrough for the final clamp.
fn inv_sqrt_identity(table_id: u32) -> ActivationTable {
    ActivationTable {
        table_id,
        lo: 0,
        outputs: (0i64..=200).collect(),
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
    ];
    // Pre-LayerNorm over a 4-vector, then a 2x4 linear (identity-ish) + requant.
    let layer = LayerSpec {
        pre_layernorm: Some(LayerNormSpec {
            op_id: 1,
            table_id: 0,
            shift: 6, // keep (x-mean)*inv_std within int8 clamp
            clamp_lo: -128,
            clamp_hi: 127,
            rounding: Rounding::NearestTiesToEven,
        }),
        linear_op_id: 2,
        weight: weight(10, 2, 4, &[1, 1, 0, 0, 0, 0, 1, 1]),
        bias: None,
        requant_op_id: 3,
        shift: 0,
        zero_point: 0,
        clamp_lo: -128,
        clamp_hi: 127,
        rounding: Rounding::NearestTiesToEven,
        activation: None,
    };
    Model {
        layers: vec![layer],
        tables: vec![inv_sqrt_identity(0)],
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
fn accept_layernorm_block() {
    // input [2,4,6,8]: mean 5, centered [-3,-1,1,3], var round(20/4)=5, inv_std=5,
    // (x-mean)*inv_std = [-15,-5,5,15], >>6 nearest -> [0,0,0,0]... too coarse;
    // use a smaller spread so the normalized output is non-trivial.
    let artifact = prove_feedforward(&model(), &[10, 20, 30, 40], out_binding()).unwrap();
    assert_eq!(verify(&artifact), Ok(()));
    // The trace contains a LayerNorm op.
    assert!(artifact
        .trace
        .iter()
        .any(|r| matches!(r, OpRecord::LayerNorm(_))));
}

#[test]
fn reject_tampered_layernorm_output() {
    let mut a = prove_feedforward(&model(), &[10, 20, 30, 40], out_binding()).unwrap();
    let idx = a
        .trace
        .iter()
        .position(|r| matches!(r, OpRecord::LayerNorm(_)))
        .unwrap();
    if let OpRecord::LayerNorm(r) = &mut a.trace[idx] {
        r.output[0] += 1;
    }
    // Tampering the normalized output breaks exact replay or the downstream wiring.
    assert!(matches!(
        verify(&a),
        Err(VerifyError::ExactReplayMismatch { .. }) | Err(VerifyError::WiringMismatch { .. })
    ));
}
