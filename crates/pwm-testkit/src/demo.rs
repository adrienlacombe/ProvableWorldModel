// SPDX-License-Identifier: Apache-2.0
//! Reusable logic for the `pwm` demo CLI: a small built-in quantized model, the
//! prove path, verifier-secret challenge vectors, and a structural tamper. Kept
//! in the library (not the binary) so the demo is unit-tested, not just runnable.
//!
//! The model is a two-layer quantized feed-forward net: `Linear(3->4)` + identity
//! activation table, then `Linear(4->2)` + requant. Proving it runs the exact
//! integer reference inference and records the trace; verifying it Freivalds-checks
//! the two linears and exactly replays the rest.

use pwm_core::audit::AuditArtifact;
use pwm_core::field::Fp61;
use pwm_core::fixed_point::{BoundedInt, Rounding};
use pwm_core::tables::ActivationTable;
use pwm_core::tensor::{Dtype, Scale, Tensor};
use pwm_core::trace::OpRecord;
use pwm_export::reference::{LayerSpec, Model};
use pwm_prover::{prove_feedforward, OutputBinding};

/// The demo input activation.
pub const DEMO_INPUT: [i64; 3] = [1, 2, 3];

fn weight(id: u32, rows: u32, cols: u32, vals: &[i8]) -> Tensor {
    let data = vals
        .iter()
        .map(|&v| BoundedInt::new(v as i64, -128, 127).unwrap())
        .collect();
    Tensor::new(id, vec![rows, cols], 0, data).unwrap()
}

/// The built-in quantized feed-forward model proven by the demo.
pub fn demo_model() -> Model {
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
        weight: weight(10, 4, 3, &[1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1]),
        bias: None,
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
        shift: 1,
        zero_point: 0,
        clamp_lo: -128,
        clamp_hi: 127,
        rounding: Rounding::NearestTiesToEven,
        activation: None,
    };
    Model {
        layers: vec![l1, l2],
        tables: vec![ActivationTable {
            table_id: 0,
            lo: -128,
            outputs: (-128i64..=127).collect(),
        }],
        scales,
    }
}

/// The output tensor binding for the demo model.
pub fn out_binding() -> OutputBinding {
    OutputBinding {
        tensor_id: 200,
        scale_id: 1,
    }
}

/// Run the prover over the demo model and input: real integer inference, trace,
/// commitments, and the assembled `AuditArtifact`.
pub fn prove_demo() -> AuditArtifact {
    prove_feedforward(&demo_model(), &DEMO_INPUT, out_binding()).expect("prove demo model")
}

/// One pseudo-random `Fp61` challenge vector per `Linear` op (length = that op's
/// rows), drawn from `seed` via an LCG and never derived from the transcript.
/// This is the verifier-secret `r` used by [`pwm_verifier::verify_interactive`]:
/// the prover commits its accumulators before any of these values exist.
pub fn secret_challenge(artifact: &AuditArtifact, seed: u64) -> Vec<Vec<Fp61>> {
    let mut s = seed | 1;
    let mut out = Vec::new();
    for rec in &artifact.trace {
        if let OpRecord::Linear(l) = rec {
            let mut v = Vec::with_capacity(l.output.len());
            for _ in 0..l.output.len() {
                // SplitMix-style LCG step; reduce the state into the audit field.
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                v.push(Fp61::new(s));
            }
            out.push(v);
        }
    }
    out
}

/// Structurally tamper the proof: bump the first `Linear` accumulator by one. This
/// is the canonical "the prover lied about a matmul output" forgery; the
/// committed `z` no longer equals `W*x`, so the Freivalds check `v*x == r*z` fails.
/// Returns the op id that will be rejected, or `None` if the trace has no linear.
pub fn tamper_accumulator(artifact: &mut AuditArtifact) -> Option<u32> {
    for rec in &mut artifact.trace {
        if let OpRecord::Linear(l) = rec {
            if let Some(first) = l.output.first_mut() {
                *first += 1;
            }
            return Some(l.op_id);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use pwm_verifier::{verify, verify_interactive, VerifyError};

    #[test]
    fn demo_proves_and_verifies() {
        assert!(verify(&prove_demo()).is_ok());
    }

    #[test]
    fn verifier_secret_challenge_accepts_honest_proof() {
        let a = prove_demo();
        let r = secret_challenge(&a, 0x00C0_FFEE);
        // One challenge vector per linear op, lengths matching the rows.
        assert_eq!(r.len(), 2);
        assert!(verify_interactive(&a, &r).is_ok());
    }

    #[test]
    fn tampered_accumulator_is_rejected() {
        let mut a = prove_demo();
        let op = tamper_accumulator(&mut a).expect("a linear op to tamper");
        assert!(matches!(
            verify(&a),
            Err(VerifyError::FreivaldsCheckFailed { op_id }) if op_id == op
        ));
    }

    #[test]
    fn tamper_also_defeats_the_secret_challenge() {
        let mut a = prove_demo();
        tamper_accumulator(&mut a);
        let r = secret_challenge(&a, 0x00C0_FFEE);
        assert!(verify_interactive(&a, &r).is_err());
    }
}
