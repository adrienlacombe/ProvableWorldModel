// SPDX-License-Identifier: Apache-2.0
//! Pixel encoder proof — the **P4 pixel-to-plan** encode path (D-804, specs.md
//! §2/§5). A ViT-style image encoder mapping raw patches to a pooled latent,
//! proven over the named-buffer block DAG and checked by `verify_block`:
//!
//! 1. **Patch embedding** — one shared linear over every patch token
//!    (`BatchedLinear`, Freivalds-checked: "Freivalds on patch/linear").
//! 2. **Positional embedding** — residual `Add` of a committed position table.
//! 3. **Transformer encoder block** — Q/K/V projections (`BatchedLinear`),
//!    exact attention (`Q·Kᵀ` → row softmax → `prob·V`), out-projection,
//!    residual; then FFN (`fc1 → GELU → fc2`) with a second residual. Attention
//!    is recomputed *exactly* in integers — no bf16 hole.
//! 4. **Pooling + final norm** — slice the CLS token and `LayerNorm` it to the
//!    latent that a downstream planner (P2/P3) would consume.
//!
//! The instance below is scaled down so every value is hand-checkable
//! (`patch_dim = embed_dim = 2`, `seq = 2`, one head, one block, `ffn = 4`); it
//! is structurally identical to ViT-Tiny/14 (patch 14²·3 = 588 → embed 192,
//! depth 12, heads 3, head-dim 64, mlp 768) — only the dimensions differ, so the
//! same op vocabulary and Freivalds amortization carry over to `S = 256`.
//!
//! A valid encode verifies; tampering the patch embedding (Freivalds), the
//! attention scores (exact matmul), or the final norm (exact recompute) is
//! rejected.

use pwm_core::block::{block_root, Block, BlockOp};
use pwm_core::fixed_point::BoundedInt;
use pwm_core::relation::StatementType;
use pwm_core::tables::ActivationTable;
use pwm_core::tensor::Tensor;
use pwm_core::transcript::Transcript;
use pwm_prover::prove_block;
use pwm_verifier::{verify_block, VerifyError};

const RND: u8 = 0; // Rounding::NearestTiesToEven
const CLO: i64 = -100_000;
const CHI: i64 = 100_000;

fn w(id: u32, rows: u32, cols: u32, vals: &[i8]) -> Tensor {
    let data = vals
        .iter()
        .map(|&v| BoundedInt::new(v as i64, -128, 127).unwrap())
        .collect();
    Tensor::new(id, vec![rows, cols], 0, data).unwrap()
}

fn weights() -> Vec<Tensor> {
    let id2 = |tid| w(tid, 2, 2, &[1, 0, 0, 1]); // identity 2x2
    vec![
        id2(10),                                 // W_embed (patch embedding)
        id2(11),                                 // W_q
        id2(12),                                 // W_k
        id2(13),                                 // W_v
        id2(14),                                 // W_o (out-proj)
        w(15, 4, 2, &[1, 0, 0, 1, 1, 1, 1, -1]), // W_fc1 [4x2]
        w(16, 2, 4, &[1, 1, 1, 1, 1, 0, 0, 1]),  // W_fc2 [2x4]
    ]
}

fn tables() -> Vec<ActivationTable> {
    vec![
        // exp over shifted scores in [-2,0]: exp(-2)=1, exp(-1)=2, exp(0)=4.
        ActivationTable {
            table_id: 3,
            lo: -2,
            outputs: vec![1, 2, 4],
        },
        // GELU as identity over the activation range.
        ActivationTable {
            table_id: 30,
            lo: -3000,
            outputs: (-3000i64..=3000).collect(),
        },
        // Inverse-sqrt that returns unit scale over the variance domain (so the
        // final LayerNorm centers the pooled token); domain covers var=251001.
        ActivationTable {
            table_id: 31,
            lo: 0,
            outputs: vec![1; 300_001],
        },
    ]
}

/// The structural pixel encoder (op `out` fields are filled by `prove_block`).
fn encoder() -> Block {
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
    Block {
        input_bufs: vec![0, 1], // 0 = patches [seq,patch_dim], 1 = positional [seq,dim]
        ops: vec![
            bl(1, 10, 0, 2), // patch embedding (Freivalds) -> tokens [2,2]
            BlockOp::Add {
                op_id: 2,
                a_buf: 2,
                b_buf: 1,
                out_buf: 3,
                out: vec![],
            }, // + positional
            bl(3, 11, 3, 4),      // Q
            bl(4, 12, 3, 5),      // K
            bl(5, 13, 3, 6),      // V
            mm(6, 4, 5, 7, true), // scores = Q·Kᵀ
            BlockOp::Softmax {
                op_id: 7,
                table_id: 3,
                in_buf: 7,
                out_buf: 8,
                out: vec![],
                row_len: 2,
                one: 600,
            },
            mm(8, 8, 6, 9, false), // attn = prob·V
            bl(9, 14, 9, 10),      // out-projection
            BlockOp::Add {
                op_id: 10,
                a_buf: 3,
                b_buf: 10,
                out_buf: 11,
                out: vec![],
            }, // residual 1
            bl(11, 15, 11, 12), // FFN fc1 -> [2,4]
            BlockOp::Activation {
                op_id: 12,
                table_id: 30,
                in_buf: 12,
                out_buf: 13,
                out: vec![],
            }, // GELU
            bl(13, 16, 13, 14), // FFN fc2 -> [2,2]
            BlockOp::Add {
                op_id: 14,
                a_buf: 11,
                b_buf: 14,
                out_buf: 15,
                out: vec![],
            }, // residual 2
            BlockOp::Slice {
                op_id: 15,
                in_buf: 15,
                out_buf: 16,
                start: 0,
                len: 2,
                out: vec![],
            }, // pool CLS token
            BlockOp::LayerNorm {
                op_id: 16,
                table_id: 31,
                in_buf: 16,
                out_buf: 17,
                out: vec![],
                shift: 0,
                clamp_lo: CLO,
                clamp_hi: CHI,
                rounding: RND,
            }, // final norm -> latent
        ],
        output_buf: 17,
    }
}

fn inputs() -> Vec<(u32, Vec<i64>)> {
    vec![
        (0, vec![1, 0, 0, 1]), // patches (2 patches x dim 2)
        (1, vec![0, 0, 0, 0]), // positional embedding
    ]
}

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
fn p4_relation_id_is_pixel_to_plan() {
    // The encode path proven here is the P4 statement's front half.
    assert_eq!(StatementType::P4PixelToPlan.discriminant(), 4);
}

#[test]
fn accept_pixel_encoder_to_latent() {
    let proven = prove_block(&encoder(), &weights(), &tables(), &inputs()).unwrap();
    let mut t = transcript_for(&proven, &inputs());
    let latent = verify_block(&proven, &weights(), &tables(), &inputs(), &mut t).unwrap();
    // CLS token after both residuals is [1804, 802] (mean 1303); final LayerNorm
    // with unit inverse-std centers it -> [501, -501].
    assert_eq!(latent, vec![501, -501]);
}

#[test]
fn reject_tampered_patch_embedding() {
    let mut proven = prove_block(&encoder(), &weights(), &tables(), &inputs()).unwrap();
    // Tamper the patch-embedding projection (op_id 1) -> Freivalds rejects.
    if let BlockOp::BatchedLinear { out, .. } = &mut proven.ops[0] {
        out[0] += 1;
    }
    let mut t = transcript_for(&proven, &inputs());
    assert!(matches!(
        verify_block(&proven, &weights(), &tables(), &inputs(), &mut t),
        Err(VerifyError::FreivaldsCheckFailed { op_id: 1 })
    ));
}

#[test]
fn reject_tampered_attention_scores() {
    let mut proven = prove_block(&encoder(), &weights(), &tables(), &inputs()).unwrap();
    // Tamper the QKᵀ scores (op_id 6) -> exact matmul recompute rejects.
    for op in proven.ops.iter_mut() {
        if let BlockOp::MatMul { op_id: 6, out, .. } = op {
            out[0] += 1;
        }
    }
    let mut t = transcript_for(&proven, &inputs());
    assert!(matches!(
        verify_block(&proven, &weights(), &tables(), &inputs(), &mut t),
        Err(VerifyError::BlockOpMismatch { op_id: 6 })
    ));
}

#[test]
fn reject_tampered_final_norm() {
    let mut proven = prove_block(&encoder(), &weights(), &tables(), &inputs()).unwrap();
    // Tamper the final LayerNorm latent (op_id 16) -> exact recompute rejects.
    for op in proven.ops.iter_mut() {
        if let BlockOp::LayerNorm { op_id: 16, out, .. } = op {
            out[0] += 7;
        }
    }
    let mut t = transcript_for(&proven, &inputs());
    assert!(matches!(
        verify_block(&proven, &weights(), &tables(), &inputs(), &mut t),
        Err(VerifyError::BlockOpMismatch { op_id: 16 })
    ));
}
