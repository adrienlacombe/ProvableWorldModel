// SPDX-License-Identifier: Apache-2.0
//! End-to-end test for the named-buffer predictor block (specs.md §2.1, §5): a
//! real LeWorldModel AdaLN-gated FFN sub-path — SiLU(c) → AdaLN scale/shift/gate
//! linears, then LayerNorm → modulate → Linear → GELU → Linear → gate → residual
//! add — composed over the buffer DAG, proven by `prove_block` and checked by
//! `verify_block`. The valid block verifies; a tampered op output is rejected.

use pwm_core::block::{block_root, Block, BlockOp};
use pwm_core::fixed_point::BoundedInt;
use pwm_core::tables::ActivationTable;
use pwm_core::tensor::Tensor;
use pwm_core::transcript::Transcript;
use pwm_prover::prove_block;
use pwm_verifier::{verify_block, VerifyError};

const RND: u8 = 0; // Rounding::NearestTiesToEven discriminant

fn w(id: u32, rows: u32, cols: u32, vals: &[i8]) -> Tensor {
    let data = vals
        .iter()
        .map(|&v| BoundedInt::new(v as i64, -128, 127).unwrap())
        .collect();
    Tensor::new(id, vec![rows, cols], 0, data).unwrap()
}

fn ident_table(id: u32, lo: i64, hi: i64) -> ActivationTable {
    ActivationTable {
        table_id: id,
        lo,
        outputs: (lo..=hi).collect(),
    }
}

fn weights() -> Vec<Tensor> {
    vec![
        w(20, 2, 2, &[0, 0, 0, 0]),              // W_scale -> scale = [0,0]
        w(21, 2, 2, &[0, 0, 0, 0]),              // W_shift -> shift = [0,0]
        w(22, 2, 2, &[1, 0, 0, 1]),              // W_gate  -> gate = SiLU(c)
        w(23, 4, 2, &[1, 0, 0, 1, 1, 1, 1, -1]), // W_fc1
        w(24, 2, 4, &[1, 1, 1, 1, 1, 0, 0, 1]),  // W_fc2
    ]
}

fn tables() -> Vec<ActivationTable> {
    vec![
        ident_table(0, -128, 127),   // SiLU (identity)
        ident_table(1, 0, 1000),     // inverse-sqrt (identity)
        ident_table(2, -1000, 1000), // GELU (identity)
    ]
}

const CLO: i64 = -100000;
const CHI: i64 = 100000;

/// The structural block (op `out` fields are placeholders, filled by prove_block).
fn block_spec() -> Block {
    let lin = |op_id, weight_id, in_buf, out_buf| BlockOp::Linear {
        op_id,
        weight_id,
        bias_id: None,
        in_buf,
        out_buf,
        out: vec![],
    };
    Block {
        input_bufs: vec![0, 1], // 0 = x, 1 = c (conditioning)
        ops: vec![
            // AdaLN modulation generation: SiLU(c) then scale/shift/gate linears.
            BlockOp::Activation {
                op_id: 1,
                table_id: 0,
                in_buf: 1,
                out_buf: 2,
                out: vec![],
            },
            lin(2, 20, 2, 3), // scale
            lin(3, 21, 2, 4), // shift
            lin(4, 22, 2, 5), // gate
            // FFN sub-block over x with AdaLN modulation + gate + residual.
            BlockOp::LayerNorm {
                op_id: 5,
                table_id: 1,
                in_buf: 0,
                out_buf: 6,
                out: vec![],
                shift: 0,
                clamp_lo: CLO,
                clamp_hi: CHI,
                rounding: RND,
            },
            BlockOp::Modulate {
                op_id: 6,
                x_buf: 6,
                scale_buf: 3,
                shift_buf: 4,
                out_buf: 7,
                out: vec![],
                one: 1,
                shift_bits: 0,
                clamp_lo: CLO,
                clamp_hi: CHI,
                rounding: RND,
            },
            lin(7, 23, 7, 8), // fc1
            BlockOp::Activation {
                op_id: 8,
                table_id: 2,
                in_buf: 8,
                out_buf: 9,
                out: vec![],
            }, // GELU
            lin(9, 24, 9, 10), // fc2
            BlockOp::Gate {
                op_id: 10,
                gate_buf: 5,
                x_buf: 10,
                out_buf: 11,
                out: vec![],
                shift_bits: 0,
                clamp_lo: CLO,
                clamp_hi: CHI,
                rounding: RND,
            },
            BlockOp::Add {
                op_id: 11,
                a_buf: 0,
                b_buf: 11,
                out_buf: 12,
                out: vec![],
            },
        ],
        output_buf: 12,
    }
}

fn inputs() -> Vec<(u32, Vec<i64>)> {
    vec![(0, vec![4, 8]), (1, vec![1, 1])]
}

/// Build the verifier's transcript, bound to the proven block's ops + inputs.
fn transcript_for(block: &Block, inputs: &[(u32, Vec<i64>)]) -> Transcript {
    let mut t = Transcript::new(b"pwm.block.v1");
    t.absorb(b"block_root", &block_root(&block.ops));
    for (id, v) in inputs {
        t.absorb_u64(b"in_buf", *id as u64);
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        t.absorb(b"in_vals", &bytes);
    }
    t
}

#[test]
fn accept_adaln_gated_ffn_block() {
    let proven = prove_block(&block_spec(), &weights(), &tables(), &inputs()).unwrap();
    // x=[4,8]; LN->[-8,8]; modulate(scale0,shift0)->[-8,8]; fc1->[-8,8,0,-16];
    // GELU id->same; fc2->[-16,-24]; gate(=1)->[-16,-24]; +x residual -> [-12,-16].
    let mut t = transcript_for(&proven, &inputs());
    let out = verify_block(&proven, &weights(), &tables(), &inputs(), &mut t).unwrap();
    assert_eq!(out, vec![-12, -16]);
}

#[test]
fn reject_tampered_block_linear() {
    let mut proven = prove_block(&block_spec(), &weights(), &tables(), &inputs()).unwrap();
    // Tamper the fc1 linear output (op_id 7) -> Freivalds rejects.
    for op in proven.ops.iter_mut() {
        if let BlockOp::Linear { op_id: 7, out, .. } = op {
            out[0] += 1;
        }
    }
    let mut t = transcript_for(&proven, &inputs());
    assert!(matches!(
        verify_block(&proven, &weights(), &tables(), &inputs(), &mut t),
        Err(VerifyError::FreivaldsCheckFailed { op_id: 7 })
    ));
}

#[test]
fn reject_tampered_block_residual() {
    let mut proven = prove_block(&block_spec(), &weights(), &tables(), &inputs()).unwrap();
    // Tamper the residual add output (op_id 11) -> exact recompute rejects.
    for op in proven.ops.iter_mut() {
        if let BlockOp::Add { op_id: 11, out, .. } = op {
            out[0] += 5;
        }
    }
    let mut t = transcript_for(&proven, &inputs());
    assert!(matches!(
        verify_block(&proven, &weights(), &tables(), &inputs(), &mut t),
        Err(VerifyError::BlockOpMismatch { op_id: 11 })
    ));
}
