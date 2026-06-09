// SPDX-License-Identifier: Apache-2.0
//! Prove the REAL le-wm `pred_proj` head from an exported bundle.
//!
//! `crates/pwm-export/python/scripts/export_lewm_v0.py` loads the pretrained
//! `quentinll/lewm-pusht` checkpoint, quantizes the V0 subgraph, folds the
//! `pred_proj` BatchNorm, and writes a JSON bundle. This module turns that bundle
//! into a `pwm-export` reference `Model` for the `pred_proj` head
//! (`Linear(192->2048) -> GELU -> Linear(2048->192)`) and proves it with the real
//! folded weights, so the verifier audits a real-checkpoint computation.
//!
//! The full 6-block attention predictor is quantized and committed by the same
//! export; wiring its 16-head graph into the prover is the next step (the op
//! kernels already exist in `pwm_prover::prove_block`).

use pwm_core::audit::AuditArtifact;
use pwm_core::fixed_point::{BoundedInt, Rounding};
use pwm_core::tables::ActivationTable;
use pwm_core::tensor::{Dtype, Scale, Tensor};
use pwm_export::reference::{LayerSpec, Model};
use pwm_prover::{prove_feedforward, OutputBinding};

use crate::bundle::{self, BundleError};

/// A loaded export bundle: the `pred_proj` head model and a real input latent.
pub struct LewmBundle {
    /// Human-readable description of the proven model.
    pub label: String,
    /// The two-linear `pred_proj` head as a reference model.
    pub model: Model,
    /// The quantized input latent (`dim` int8 values).
    pub input: Vec<i64>,
    /// Latent dimension.
    pub dim: usize,
    /// FFN hidden width.
    pub mlp: usize,
}

fn i8_tensor(
    id: u32,
    rows: u32,
    cols: u32,
    scale_id: u32,
    data: &[i64],
) -> Result<Tensor, BundleError> {
    let cells = data
        .iter()
        .map(|&v| {
            BoundedInt::new(v, -128, 127).map_err(|_| BundleError::WeightOutOfRange { value: v })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Tensor::new(id, vec![rows, cols], scale_id, cells).map_err(|_| BundleError::WrongType {
        field: "fc weight",
        expected: "a rows*cols tensor",
    })
}

/// Parse an export bundle (JSON) into a provable `pred_proj` model + input, or a
/// [`BundleError`] describing what was malformed.
pub fn load_bundle(json: &str) -> Result<LewmBundle, BundleError> {
    let b = bundle::parse(json)?;
    let dim = bundle::u64_at(&b, "dim")? as u32;
    let mlp = bundle::u64_at(&b, "mlp")? as u32;
    let fc1 = i8_tensor(
        10,
        mlp,
        dim,
        0,
        &bundle::ints_at(bundle::field(&b, "fc1")?, "data")?,
    )?;
    let fc2 = i8_tensor(
        11,
        dim,
        mlp,
        1,
        &bundle::ints_at(bundle::field(&b, "fc2")?, "data")?,
    )?;
    let input = bundle::ints_at(bundle::field(&b, "input")?, "data")?;
    let gt = bundle::field(&b, "gelu_table")?;
    let gelu = ActivationTable {
        table_id: bundle::u64_at(gt, "table_id")? as u32,
        lo: bundle::i64_at(gt, "lo")?,
        outputs: bundle::ints_at(gt, "outputs")?,
    };
    let fc1_shift = bundle::u64_at(&b, "fc1_shift")? as u32;
    let fc2_shift = bundle::u64_at(&b, "fc2_shift")? as u32;
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
    let l1 = LayerSpec {
        pre_layernorm: None,
        linear_op_id: 100,
        weight: fc1,
        bias: None,
        requant_op_id: 101,
        shift: fc1_shift,
        zero_point: 0,
        clamp_lo: -128,
        clamp_hi: 127,
        rounding: Rounding::NearestTiesToEven,
        activation: Some((102, gelu.table_id)),
    };
    let l2 = LayerSpec {
        pre_layernorm: None,
        linear_op_id: 103,
        weight: fc2,
        bias: None,
        requant_op_id: 104,
        shift: fc2_shift,
        zero_point: 0,
        clamp_lo: -128,
        clamp_hi: 127,
        rounding: Rounding::NearestTiesToEven,
        activation: None,
    };
    Ok(LewmBundle {
        label: bundle::str_at_or(&b, "model", "lewm pred_proj"),
        model: Model {
            layers: vec![l1, l2],
            tables: vec![gelu],
            scales,
        },
        input,
        dim: dim as usize,
        mlp: mlp as usize,
    })
}

/// Run the prover over the bundle's `pred_proj` head and the real input latent.
pub fn prove(bundle: &LewmBundle) -> AuditArtifact {
    prove_feedforward(
        &bundle.model,
        &bundle.input,
        OutputBinding {
            tensor_id: 200,
            scale_id: 1,
        },
    )
    .expect("prove lewm pred_proj head")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demo::tamper_accumulator;
    use pwm_verifier::{verify, VerifyError};

    // A tiny synthetic bundle (dim=2, mlp=2) exercising the loader + prove/verify
    // without the 3 MB real checkpoint bundle.
    fn tiny_bundle() -> String {
        serde_json::json!({
            "model": "tiny pred_proj",
            "dim": 2, "mlp": 2,
            "input": {"data": [1, 2], "log2": 0},
            "fc1": {"data": [1, 0, 0, 1], "rows": 2, "cols": 2, "log2": 0},
            "fc2": {"data": [1, 1, 1, -1], "rows": 2, "cols": 2, "log2": 0},
            "gelu_table": {"table_id": 0, "lo": -128, "outputs": (-128i64..=127).collect::<Vec<_>>()},
            "fc1_shift": 0, "fc2_shift": 0
        })
        .to_string()
    }

    #[test]
    fn loads_and_proves_pred_proj_bundle() {
        let bundle = load_bundle(&tiny_bundle()).expect("valid bundle");
        assert_eq!(bundle.dim, 2);
        assert!(verify(&prove(&bundle)).is_ok());
    }

    #[test]
    fn malformed_bundle_is_a_typed_error_not_a_panic() {
        assert!(matches!(
            load_bundle("not json"),
            Err(BundleError::Parse(_))
        ));
        assert!(matches!(
            load_bundle("{}"),
            Err(BundleError::MissingField("dim"))
        ));
    }

    #[test]
    fn tampered_pred_proj_is_rejected() {
        let bundle = load_bundle(&tiny_bundle()).expect("valid bundle");
        let mut a = prove(&bundle);
        let op = tamper_accumulator(&mut a).expect("a linear op");
        assert!(matches!(
            verify(&a),
            Err(VerifyError::FreivaldsCheckFailed { op_id }) if op_id == op
        ));
    }
}
