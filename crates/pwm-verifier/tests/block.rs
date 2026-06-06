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

// --- Single-head attention core: QKᵀ -> row softmax -> prob·V ---

fn attention_block() -> Block {
    Block {
        input_bufs: vec![0, 1, 2], // Q, K, V (each [S=2, d=2])
        ops: vec![
            // scores = Q · Kᵀ  ([2,2])
            BlockOp::MatMul {
                op_id: 1,
                a_buf: 0,
                b_buf: 1,
                out_buf: 3,
                out: vec![],
                rows: 2,
                inner: 2,
                cols: 2,
                transpose_b: true,
            },
            // prob = softmax(scores) per row
            BlockOp::Softmax {
                op_id: 2,
                table_id: 3,
                in_buf: 3,
                out_buf: 4,
                out: vec![],
                row_len: 2,
                one: 600,
            },
            // out = prob · V  ([2,2])
            BlockOp::MatMul {
                op_id: 3,
                a_buf: 4,
                b_buf: 2,
                out_buf: 5,
                out: vec![],
                rows: 2,
                inner: 2,
                cols: 2,
                transpose_b: false,
            },
        ],
        output_buf: 5,
    }
}

fn attention_inputs() -> Vec<(u32, Vec<i64>)> {
    vec![
        (0, vec![1, 0, 0, 1]), // Q = I
        (1, vec![1, 0, 0, 1]), // K = I
        (2, vec![1, 0, 0, 1]), // V = I
    ]
}

fn exp_tables() -> Vec<ActivationTable> {
    // exp over shifted scores in [-2,0]: exp(-2)=1, exp(-1)=2, exp(0)=4.
    vec![ActivationTable {
        table_id: 3,
        lo: -2,
        outputs: vec![1, 2, 4],
    }]
}

#[test]
fn accept_single_head_attention() {
    let proven = prove_block(&attention_block(), &[], &exp_tables(), &attention_inputs()).unwrap();
    let mut t = transcript_for(&proven, &attention_inputs());
    let out = verify_block(&proven, &[], &exp_tables(), &attention_inputs(), &mut t).unwrap();
    // scores=[[1,0],[0,1]]; softmax rows -> [[400,200],[200,400]]; prob·V(=I) -> same.
    assert_eq!(out, vec![400, 200, 200, 400]);
}

#[test]
fn reject_tampered_attention_scores() {
    let mut proven =
        prove_block(&attention_block(), &[], &exp_tables(), &attention_inputs()).unwrap();
    // Tamper the QKᵀ scores (op_id 1) -> exact matmul recompute rejects.
    for op in proven.ops.iter_mut() {
        if let BlockOp::MatMul { op_id: 1, out, .. } = op {
            out[0] += 1;
        }
    }
    let mut t = transcript_for(&proven, &attention_inputs());
    assert!(matches!(
        verify_block(&proven, &[], &exp_tables(), &attention_inputs(), &mut t),
        Err(VerifyError::BlockOpMismatch { op_id: 1 })
    ));
}

#[test]
fn accept_multiposition_attention_with_projection() {
    // Per-position Q projection (slice -> Freivalds linear -> concat) feeding the
    // attention core, with K and V supplied pre-projected. Wq = I, so Q = x.
    let wq = vec![w(30, 2, 2, &[1, 0, 0, 1])];
    let block = Block {
        input_bufs: vec![0, 6, 7], // x (2 positions x dim 2), K, V
        ops: vec![
            BlockOp::Slice {
                op_id: 1,
                in_buf: 0,
                out_buf: 1,
                start: 0,
                len: 2,
                out: vec![],
            },
            BlockOp::Linear {
                op_id: 2,
                weight_id: 30,
                bias_id: None,
                in_buf: 1,
                out_buf: 2,
                out: vec![],
            },
            BlockOp::Slice {
                op_id: 3,
                in_buf: 0,
                out_buf: 3,
                start: 2,
                len: 2,
                out: vec![],
            },
            BlockOp::Linear {
                op_id: 4,
                weight_id: 30,
                bias_id: None,
                in_buf: 3,
                out_buf: 4,
                out: vec![],
            },
            BlockOp::Concat {
                op_id: 5,
                in_bufs: vec![2, 4],
                out_buf: 5,
                out: vec![],
            }, // Q
            BlockOp::MatMul {
                op_id: 6,
                a_buf: 5,
                b_buf: 6,
                out_buf: 8,
                out: vec![],
                rows: 2,
                inner: 2,
                cols: 2,
                transpose_b: true,
            },
            BlockOp::Softmax {
                op_id: 7,
                table_id: 3,
                in_buf: 8,
                out_buf: 9,
                out: vec![],
                row_len: 2,
                one: 600,
            },
            BlockOp::MatMul {
                op_id: 8,
                a_buf: 9,
                b_buf: 7,
                out_buf: 10,
                out: vec![],
                rows: 2,
                inner: 2,
                cols: 2,
                transpose_b: false,
            },
        ],
        output_buf: 10,
    };
    let inputs = vec![
        (0, vec![1, 0, 0, 1]), // x
        (6, vec![1, 0, 0, 1]), // K
        (7, vec![1, 0, 0, 1]), // V
    ];
    let proven = prove_block(&block, &wq, &exp_tables(), &inputs).unwrap();
    let mut t = transcript_for(&proven, &inputs);
    let out = verify_block(&proven, &wq, &exp_tables(), &inputs, &mut t).unwrap();
    assert_eq!(out, vec![400, 200, 200, 400]);
}

// --- Standard transformer encoder block (ViT structure; D-804) via BatchedLinear ---

#[test]
fn accept_vit_encoder_block() {
    let id = |tid| w(tid, 2, 2, &[1, 0, 0, 1]); // identity 2x2
    let weights = vec![
        id(10),
        id(11),
        id(12),
        id(13),                                  // Wq, Wk, Wv, Wproj = I
        w(14, 4, 2, &[1, 0, 0, 1, 1, 1, 1, -1]), // fc1 [4x2]
        w(15, 2, 4, &[1, 1, 1, 1, 1, 0, 0, 1]),  // fc2 [2x4]
    ];
    let tables = vec![
        ActivationTable {
            table_id: 3,
            lo: -2,
            outputs: vec![1, 2, 4],
        }, // exp
        ActivationTable {
            table_id: 16,
            lo: -2000,
            outputs: (-2000i64..=2000).collect(),
        }, // GELU=id
    ];
    let bl = |op_id, weight_id, in_buf, out_buf| BlockOp::BatchedLinear {
        op_id,
        weight_id,
        bias_id: None,
        in_buf,
        out_buf,
        out: vec![],
        seq: 2,
    };
    let mm = |op_id, a_buf, b_buf, out_buf, transpose_b| BlockOp::MatMul {
        op_id,
        a_buf,
        b_buf,
        out_buf,
        out: vec![],
        rows: 2,
        inner: 2,
        cols: 2,
        transpose_b,
    };
    let block = Block {
        input_bufs: vec![0], // x: 2 patches x dim 2
        ops: vec![
            bl(1, 10, 0, 1),      // Q
            bl(2, 11, 0, 2),      // K
            bl(3, 12, 0, 3),      // V
            mm(4, 1, 2, 4, true), // scores = Q·Kᵀ
            BlockOp::Softmax {
                op_id: 5,
                table_id: 3,
                in_buf: 4,
                out_buf: 5,
                out: vec![],
                row_len: 2,
                one: 600,
            },
            mm(6, 5, 3, 6, false), // attn = prob·V
            bl(7, 13, 6, 7),       // out-proj
            BlockOp::Add {
                op_id: 8,
                a_buf: 0,
                b_buf: 7,
                out_buf: 8,
                out: vec![],
            }, // residual 1
            bl(9, 14, 8, 9),       // FFN fc1 -> [2,4]
            BlockOp::Activation {
                op_id: 10,
                table_id: 16,
                in_buf: 9,
                out_buf: 10,
                out: vec![],
            }, // GELU
            bl(11, 15, 10, 11),    // FFN fc2 -> [2,2]
            BlockOp::Add {
                op_id: 12,
                a_buf: 8,
                b_buf: 11,
                out_buf: 12,
                out: vec![],
            }, // residual 2
        ],
        output_buf: 12,
    };
    let inputs = vec![(0, vec![1, 0, 0, 1])];
    let proven = prove_block(&block, &weights, &tables, &inputs).unwrap();
    let mut t = transcript_for(&proven, &inputs);
    let out = verify_block(&proven, &weights, &tables, &inputs, &mut t).unwrap();
    assert_eq!(out, vec![1804, 802, 1201, 400]);
}

#[test]
fn reject_tampered_encoder_projection() {
    let id = |tid| w(tid, 2, 2, &[1, 0, 0, 1]);
    let weights = vec![w(10, 2, 2, &[1, 0, 0, 1]), id(11), id(12)];
    let tables = vec![ActivationTable {
        table_id: 3,
        lo: -2,
        outputs: vec![1, 2, 4],
    }];
    let block = Block {
        input_bufs: vec![0],
        ops: vec![BlockOp::BatchedLinear {
            op_id: 1,
            weight_id: 10,
            bias_id: None,
            in_buf: 0,
            out_buf: 1,
            out: vec![],
            seq: 2,
        }],
        output_buf: 1,
    };
    let inputs = vec![(0, vec![3, 4, 5, 6])];
    let mut proven = prove_block(&block, &weights, &tables, &inputs).unwrap();
    // Tamper one row of the batched projection -> Freivalds rejects.
    if let BlockOp::BatchedLinear { out, .. } = &mut proven.ops[0] {
        out[3] += 1;
    }
    let mut t = transcript_for(&proven, &inputs);
    assert!(matches!(
        verify_block(&proven, &weights, &tables, &inputs, &mut t),
        Err(VerifyError::FreivaldsCheckFailed { op_id: 1 })
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
