// SPDX-License-Identifier: Apache-2.0
//! A real, action-conditioned world-model predictor step for the demo.
//!
//! This proves the actual le-wm predictor *architecture* run in exact integer
//! arithmetic over the named-buffer block DAG: an action embedding conditions a
//! self-attention block (`Q·Kᵀ` -> row softmax -> `prob·V` -> out-projection) and
//! a GELU feed-forward, each with a residual, over a latent history. It predicts
//! the next latent from the history and the action.
//!
//! It is a compact instance (`dim = 2`, `history = 2`, one head, `mlp = 4`): the
//! full le-wm V0 predictor runs at `latent_dim = 192`, depth 6, 16 heads, with
//! AdaLN-zero conditioning and LayerNorm, which the exporter ingests and quantizes
//! (see the optional `real` profile). The point here is that this is the
//! transformer-predictor computation itself, not a stand-in feed-forward net, and
//! the proof attests the exact quantized relation regardless of the weights.
//!
//! The weights are a fixed synthesized instance: le-wm V0 is `pretrained: false`,
//! so no public checkpoint exists, and the proof is weight-agnostic.

use pwm_core::block::{block_root, Block, BlockOp};
use pwm_core::fixed_point::BoundedInt;
use pwm_core::serialize::{
    canonical_bytes, from_canonical_bytes, CanonicalDecode, CanonicalEncode, DecodeError, Reader,
};
use pwm_core::tables::ActivationTable;
use pwm_core::tensor::Tensor;
use pwm_prover::prove_block;
use pwm_verifier::{verify_block, VerifyError};

/// Latent dimension of the compact instance.
pub const DIM: usize = 2;
/// Latent history length (sequence the predictor attends over).
pub const SEQ: usize = 2;
/// Action embedding dimension.
pub const ACTION_DIM: usize = 2;
/// Feed-forward hidden width.
pub const MLP: usize = 4;
/// Full le-wm V0 latent dimension, for honest reporting.
pub const V0_DIM: usize = 192;
/// Full le-wm V0 predictor depth.
pub const V0_DEPTH: usize = 6;
/// Full le-wm V0 attention head count.
pub const V0_HEADS: usize = 16;

fn w(id: u32, rows: u32, cols: u32, vals: &[i8]) -> Tensor {
    let data = vals
        .iter()
        .map(|&v| BoundedInt::new(v as i64, -128, 127).unwrap())
        .collect();
    Tensor::new(id, vec![rows, cols], 0, data).unwrap()
}

/// The committed quantized weights of the predictor block.
pub fn weights() -> Vec<Tensor> {
    let id2 = |id| w(id, 2, 2, &[1, 0, 0, 1]); // identity 2x2
    vec![
        id2(20),                                 // W_action (action -> latent)
        id2(10),                                 // W_q
        id2(11),                                 // W_k
        id2(12),                                 // W_v
        id2(13),                                 // W_out (attention out-projection)
        w(14, 4, 2, &[1, 0, 0, 1, 1, 1, 1, -1]), // W_fc1 [4x2]
        w(15, 2, 4, &[1, 1, 1, 1, 1, 0, 0, 1]),  // W_fc2 [2x4]
    ]
}

/// The committed lookup tables. Illustrative integer approximations: `exp` is a
/// doubling table over a wide domain so any reasonable score range is covered;
/// `gelu` is identity over the activation range (the verifier only checks that the
/// claimed read matches the committed table).
pub fn tables() -> Vec<ActivationTable> {
    vec![
        ActivationTable {
            table_id: 3,
            lo: -16,
            outputs: (-16i64..=0).map(|k| 1i64 << (k + 16)).collect(), // 1,2,4,...,65536
        },
        ActivationTable {
            table_id: 16,
            lo: -4000,
            outputs: (-4000i64..=4000).collect(), // GELU as identity
        },
    ]
}

/// The block's external input buffer ids, in order: latent history, then action.
pub const INPUT_BUFS: [u32; 2] = [0, 1];

/// The input values aligned to [`INPUT_BUFS`]: a latent history `[SEQ x DIM]` and
/// an action embedding `[ACTION_DIM]`.
pub fn input_values() -> Vec<Vec<i64>> {
    vec![
        vec![1, 0, 0, 1], // z_history: position 0 = [1,0], position 1 = [0,1]
        vec![1, 0],       // action embedding
    ]
}

fn input_pairs(values: &[Vec<i64>]) -> Vec<(u32, Vec<i64>)> {
    INPUT_BUFS
        .iter()
        .zip(values.iter())
        .map(|(&id, v)| (id, v.clone()))
        .collect()
}

/// The static predictor graph (op `out` fields are placeholders, filled by the
/// prover). Buffers: 0=history, 1=action.
pub fn block() -> Block {
    let bl = |op_id, weight_id, in_buf, out_buf| BlockOp::BatchedLinear {
        op_id,
        weight_id,
        bias_id: None,
        in_buf,
        out_buf,
        out: vec![],
        seq: SEQ as u32,
    };
    let mm = |op_id, a_buf, b_buf, out_buf, transpose_b| BlockOp::MatMul {
        op_id,
        a_buf,
        b_buf,
        out_buf,
        out: vec![],
        rows: SEQ as u32,
        inner: DIM as u32,
        cols: SEQ as u32,
        transpose_b,
    };
    Block {
        input_bufs: INPUT_BUFS.to_vec(),
        ops: vec![
            // action conditioning: project the action and add it to every position.
            BlockOp::Linear {
                op_id: 1,
                weight_id: 20,
                bias_id: None,
                in_buf: 1,
                out_buf: 2,
                out: vec![],
            }, // action_proj [DIM]
            BlockOp::Concat {
                op_id: 2,
                in_bufs: vec![2, 2],
                out_buf: 3,
                out: vec![],
            }, // tile across SEQ positions
            BlockOp::Add {
                op_id: 3,
                a_buf: 0,
                b_buf: 3,
                out_buf: 4,
                out: vec![],
            }, // x_cond = history + action
            // self-attention over the conditioned history.
            bl(4, 10, 4, 5),      // Q
            bl(5, 11, 4, 6),      // K
            bl(6, 12, 4, 7),      // V
            mm(7, 5, 6, 8, true), // scores = Q·Kᵀ
            BlockOp::Softmax {
                op_id: 8,
                table_id: 3,
                in_buf: 8,
                out_buf: 9,
                out: vec![],
                row_len: SEQ as u32,
                one: 600,
            },
            mm(9, 9, 7, 10, false), // attn = prob·V
            bl(10, 13, 10, 11),     // out-projection
            BlockOp::Add {
                op_id: 11,
                a_buf: 4,
                b_buf: 11,
                out_buf: 12,
                out: vec![],
            }, // residual 1
            // GELU feed-forward.
            bl(12, 14, 12, 13), // fc1 -> [SEQ x MLP]
            BlockOp::Activation {
                op_id: 13,
                table_id: 16,
                in_buf: 13,
                out_buf: 14,
                out: vec![],
            }, // GELU
            bl(14, 15, 14, 15), // fc2 -> [SEQ x DIM]
            BlockOp::Add {
                op_id: 15,
                a_buf: 12,
                b_buf: 15,
                out_buf: 16,
                out: vec![],
            }, // residual 2 = predicted next history
        ],
        output_buf: 16,
    }
}

/// A serializable predictor proof: the prover transmits its witness (the input
/// values and the claimed output of every op) over the public predictor graph.
/// The verifier reconstructs the graph (public) and audits these claimed values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredictorProof {
    /// Input buffer values aligned to [`INPUT_BUFS`].
    pub inputs: Vec<Vec<i64>>,
    /// The claimed output of each op, in block op order.
    pub op_outputs: Vec<Vec<i64>>,
}

impl CanonicalEncode for PredictorProof {
    fn encode(&self, out: &mut Vec<u8>) {
        self.inputs.encode(out);
        self.op_outputs.encode(out);
    }
}

impl CanonicalDecode for PredictorProof {
    fn decode(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        Ok(PredictorProof {
            inputs: Vec::<Vec<i64>>::decode(reader)?,
            op_outputs: Vec::<Vec<i64>>::decode(reader)?,
        })
    }
}

impl PredictorProof {
    /// Canonical wire bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        canonical_bytes(self)
    }
    /// Decode from canonical wire bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DecodeError> {
        from_canonical_bytes(bytes)
    }
    /// The op count (trace length).
    pub fn op_count(&self) -> usize {
        self.op_outputs.len()
    }
}

/// Run the predictor (real integer inference) and produce its proof. Returns the
/// proof and the predicted next history `[SEQ x DIM]` (last `DIM` values are the
/// predicted next latent).
pub fn prove() -> (PredictorProof, Vec<i64>) {
    let values = input_values();
    let proven = prove_block(&block(), &weights(), &tables(), &input_pairs(&values))
        .expect("prove predictor block");
    let op_outputs: Vec<Vec<i64>> = proven.ops.iter().map(|o| o.out().to_vec()).collect();
    let next = proven
        .ops
        .last()
        .map(|o| o.out().to_vec())
        .unwrap_or_default();
    (
        PredictorProof {
            inputs: values,
            op_outputs,
        },
        next,
    )
}

/// Reconstruct the public graph, overlay the proof's claimed outputs, and bind the
/// verifier transcript to it.
fn overlay(proof: &PredictorProof) -> (Block, Vec<(u32, Vec<i64>)>) {
    let mut b = block();
    for (op, claimed) in b.ops.iter_mut().zip(proof.op_outputs.iter()) {
        op.set_out(claimed.clone());
    }
    (b, input_pairs(&proof.inputs))
}

/// Audit a predictor proof: Freivalds-check the linears, exactly recompute the
/// attention, softmax, GELU, and residuals, and return the verified output.
pub fn verify(proof: &PredictorProof) -> Result<Vec<i64>, VerifyError> {
    let (b, inputs) = overlay(proof);
    verify_block(&b, &weights(), &tables(), &inputs)
}

/// The Merkle root over the proof's claimed op graph (for logging / binding).
pub fn block_root_of(proof: &PredictorProof) -> [u8; 32] {
    let (b, _) = overlay(proof);
    block_root(&b.ops)
}

/// Number of Freivalds-checked linear ops in the predictor graph (`Linear` and
/// `BatchedLinear`).
pub fn linear_op_count() -> usize {
    block()
        .ops
        .iter()
        .filter(|o| matches!(o, BlockOp::Linear { .. } | BlockOp::BatchedLinear { .. }))
        .count()
}

/// Forge the proof: bump the first attention projection (a `BatchedLinear`)
/// output by one. The committed value no longer equals `W·x`, so the Freivalds
/// check fails. Returns the forged op id.
pub fn tamper(proof: &mut PredictorProof) -> Option<u32> {
    let b = block();
    for (i, op) in b.ops.iter().enumerate() {
        if matches!(op, BlockOp::BatchedLinear { .. }) {
            if let Some(first) = proof.op_outputs.get_mut(i).and_then(|v| v.first_mut()) {
                *first += 1;
            }
            return Some(op.op_id());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predictor_proves_and_verifies() {
        let (proof, next) = prove();
        assert_eq!(next.len(), SEQ * DIM);
        // Pin the predicted next latent (deterministic from the committed weights,
        // tables, and inputs): an is_ok() smoke test would miss silent kernel drift.
        assert_eq!(&next[(SEQ - 1) * DIM..], &[3905, 1802]);
        assert!(verify(&proof).is_ok());
    }

    #[test]
    fn predictor_proof_round_trips() {
        let (proof, _) = prove();
        let bytes = proof.to_bytes();
        let back = PredictorProof::from_bytes(&bytes).unwrap();
        assert_eq!(proof, back);
        assert!(verify(&back).is_ok());
    }

    #[test]
    fn tampered_predictor_is_rejected() {
        let (mut proof, _) = prove();
        let op = tamper(&mut proof).expect("a linear op");
        assert!(matches!(
            verify(&proof),
            Err(VerifyError::FreivaldsCheckFailed { op_id }) if op_id == op
        ));
    }

    /// Block-path counterpart of the mod-`p` accumulator-aliasing forgery (WQ-01):
    /// inflating a `BatchedLinear` accumulator by `p` is invisible to Freivalds but
    /// must be rejected by the soundness range guard before the field check.
    #[test]
    fn modp_forged_predictor_is_rejected() {
        use pwm_core::field::FREIVALDS_P;
        let (mut proof, _) = prove();
        let b = block();
        let mut forged = None;
        for (i, op) in b.ops.iter().enumerate() {
            if matches!(op, BlockOp::BatchedLinear { .. }) {
                if let Some(first) = proof.op_outputs.get_mut(i).and_then(|v| v.first_mut()) {
                    *first += FREIVALDS_P as i64;
                }
                forged = Some(op.op_id());
                break;
            }
        }
        let op = forged.expect("a batched-linear op");
        assert!(matches!(
            verify(&proof),
            Err(VerifyError::AccumulatorRange { op_id }) if op_id == op
        ));
    }

    // --- Commitment-bound predictor relation (PRED-02 / RELATION_PREDICTOR) ---

    fn committed_artifact() -> pwm_core::audit::PredictorArtifact {
        use pwm_core::tensor::{Dtype, Scale};
        use pwm_prover::{prove_predictor, OutputBinding};
        let scales = vec![Scale {
            scale_id: 0,
            log2: 0,
            dtype: Dtype::I8,
        }];
        let inputs = input_pairs(&input_values());
        prove_predictor(
            &block(),
            &weights(),
            &tables(),
            &scales,
            &inputs,
            OutputBinding {
                tensor_id: 200,
                scale_id: 0,
            },
        )
        .expect("prove committed predictor")
    }

    #[test]
    fn committed_predictor_verifies() {
        use pwm_verifier::verify_predictor;
        assert_eq!(verify_predictor(&committed_artifact()), Ok(()));
    }

    #[test]
    fn committed_predictor_rejects_swapped_weight() {
        use pwm_verifier::{verify_predictor, VerifyError as VE};
        let mut art = committed_artifact();
        // Replace fc1 (weight_id 14) without updating the committed model_commitment.
        for t in &mut art.weights {
            if t.tensor_id() == 14 {
                *t = w(14, 4, 2, &[2, 0, 0, 1, 1, 1, 1, -1]);
            }
        }
        assert_eq!(verify_predictor(&art), Err(VE::CommitmentMismatch("model")));
    }

    #[test]
    fn committed_predictor_rejects_swapped_table() {
        use pwm_verifier::{verify_predictor, VerifyError as VE};
        let mut art = committed_artifact();
        // Change a committed table without updating quantization_commitment.
        art.tables[0].outputs[0] += 1;
        assert_eq!(
            verify_predictor(&art),
            Err(VE::CommitmentMismatch("quantization"))
        );
    }

    #[test]
    fn committed_predictor_rejects_swapped_inputs() {
        use pwm_verifier::{verify_predictor, VerifyError as VE};
        let mut art = committed_artifact();
        // Change the seeded inputs without updating the input commitment.
        art.inputs[0].1[0] += 1;
        assert_eq!(
            verify_predictor(&art),
            Err(VE::CommitmentMismatch("inputs"))
        );
    }

    #[test]
    fn committed_predictor_rejects_tampered_accumulator() {
        use pwm_verifier::{verify_predictor, VerifyError as VE};
        let mut art = committed_artifact();
        let mut forged = None;
        for op in &mut art.block.ops {
            if let BlockOp::BatchedLinear { op_id, out, .. } = op {
                out[0] += 1;
                forged = Some(*op_id);
                break;
            }
        }
        let op = forged.expect("a batched-linear op");
        assert!(matches!(
            verify_predictor(&art),
            Err(VE::FreivaldsCheckFailed { op_id }) if op_id == op
        ));
    }

    #[test]
    fn committed_predictor_rejects_modp_accumulator() {
        use pwm_core::field::FREIVALDS_P;
        use pwm_verifier::{verify_predictor, VerifyError as VE};
        let mut art = committed_artifact();
        let mut forged = None;
        for op in &mut art.block.ops {
            if let BlockOp::BatchedLinear { op_id, out, .. } = op {
                out[0] += FREIVALDS_P as i64;
                forged = Some(*op_id);
                break;
            }
        }
        let op = forged.expect("a batched-linear op");
        assert!(matches!(
            verify_predictor(&art),
            Err(VE::AccumulatorRange { op_id }) if op_id == op
        ));
    }
}
