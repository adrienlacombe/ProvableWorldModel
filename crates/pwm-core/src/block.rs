// SPDX-License-Identifier: Apache-2.0
//! Named-buffer (tensor-memory) block trace for the predictor DAG (specs.md §5,
//! §2.1).
//!
//! The LeWorldModel conditional block is a DAG, not a linear chain: residual adds
//! reuse an earlier activation, and AdaLN modulation/gate parameters branch off a
//! conditioning vector. This module models that as a list of [`BlockOp`]s over
//! **named buffers** — each op reads its inputs from buffer ids and writes its
//! claimed output to a buffer id. The verifier ([`crate::block`] consumers)
//! replays each op exactly (Linear via Freivalds, the rest via the integer
//! kernels in [`crate::predictor`] / [`crate::fixed_point`]) and threads the
//! buffer map, so residuals and AdaLN wiring are checked by buffer identity.
//!
//! Pure integer, `no_std`. Op outputs are stored for self-containment and Merkle
//! commitment; the verifier recomputes them from the (buffer-sourced) inputs.

use alloc::vec::Vec;

use crate::serialize::CanonicalEncode;
use crate::transcript::blake2s256;

/// Domain tag for a block-op Merkle leaf.
pub const TAG_BLOCK_LEAF: &[u8; 16] = b"pwm.bleaf.v1\0\0\0\0";
/// Domain tag for a block Merkle node.
pub const TAG_BLOCK_NODE: &[u8; 16] = b"pwm.bnode.v1\0\0\0\0";

/// One op of a named-buffer block. Inputs are read from buffer ids; the claimed
/// output is written to `out_buf`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockOp {
    /// `out = W·x + bias`, `x` read from `in_buf` (Freivalds-checked).
    Linear {
        /// Op id.
        op_id: u32,
        /// Weight matrix id.
        weight_id: u32,
        /// Optional bias id.
        bias_id: Option<u32>,
        /// Input buffer.
        in_buf: u32,
        /// Output buffer.
        out_buf: u32,
        /// Claimed `W·x + bias`.
        out: Vec<i64>,
    },
    /// Arithmetic-right-shift rescale.
    Requant {
        /// Op id.
        op_id: u32,
        /// Input buffer.
        in_buf: u32,
        /// Output buffer.
        out_buf: u32,
        /// Claimed output.
        out: Vec<i64>,
        /// Right-shift amount.
        shift: u32,
        /// Zero point.
        zero_point: i64,
        /// Clamp low.
        clamp_lo: i64,
        /// Clamp high.
        clamp_hi: i64,
        /// Rounding discriminant.
        rounding: u8,
    },
    /// Committed-table activation read (GELU / SiLU).
    Activation {
        /// Op id.
        op_id: u32,
        /// Table id.
        table_id: u32,
        /// Input buffer.
        in_buf: u32,
        /// Output buffer.
        out_buf: u32,
        /// Claimed output.
        out: Vec<i64>,
    },
    /// Affine-free LayerNorm.
    LayerNorm {
        /// Op id.
        op_id: u32,
        /// Inverse-sqrt table id.
        table_id: u32,
        /// Input buffer.
        in_buf: u32,
        /// Output buffer.
        out_buf: u32,
        /// Claimed output.
        out: Vec<i64>,
        /// Right-shift amount.
        shift: u32,
        /// Clamp low.
        clamp_lo: i64,
        /// Clamp high.
        clamp_hi: i64,
        /// Rounding discriminant.
        rounding: u8,
    },
    /// AdaLN modulation `out = requant(x·(one+scale)) + shift` (per channel),
    /// reading `x`, `scale`, `shift` from buffers.
    Modulate {
        /// Op id.
        op_id: u32,
        /// `x` buffer.
        x_buf: u32,
        /// Per-channel scale buffer.
        scale_buf: u32,
        /// Per-channel shift buffer.
        shift_buf: u32,
        /// Output buffer.
        out_buf: u32,
        /// Claimed output.
        out: Vec<i64>,
        /// Fixed-point unit.
        one: i64,
        /// Right-shift amount.
        shift_bits: u32,
        /// Clamp low.
        clamp_lo: i64,
        /// Clamp high.
        clamp_hi: i64,
        /// Rounding discriminant.
        rounding: u8,
    },
    /// AdaLN gate `out = requant(gate·x)` (per channel), reading `gate`, `x`.
    Gate {
        /// Op id.
        op_id: u32,
        /// Per-channel gate buffer.
        gate_buf: u32,
        /// `x` buffer.
        x_buf: u32,
        /// Output buffer.
        out_buf: u32,
        /// Claimed output.
        out: Vec<i64>,
        /// Right-shift amount.
        shift_bits: u32,
        /// Clamp low.
        clamp_lo: i64,
        /// Clamp high.
        clamp_hi: i64,
        /// Rounding discriminant.
        rounding: u8,
    },
    /// Residual add `out = a + b`, reading `a`, `b` from buffers.
    Add {
        /// Op id.
        op_id: u32,
        /// First addend buffer.
        a_buf: u32,
        /// Second addend buffer.
        b_buf: u32,
        /// Output buffer.
        out_buf: u32,
        /// Claimed output.
        out: Vec<i64>,
    },
    /// Data-dependent matmul (attention QKᵀ / prob·V), exactly recomputed. `a` is
    /// `[rows, inner]`; if `transpose_b`, `b` is `[cols, inner]` and the result is
    /// `a·bᵀ` (QKᵀ), else `b` is `[inner, cols]` and the result is `a·b` (prob·V).
    MatMul {
        /// Op id.
        op_id: u32,
        /// Left operand buffer (`[rows, inner]`, row-major).
        a_buf: u32,
        /// Right operand buffer.
        b_buf: u32,
        /// Output buffer (`[rows, cols]`, row-major).
        out_buf: u32,
        /// Claimed output.
        out: Vec<i64>,
        /// Output rows.
        rows: u32,
        /// Contraction dimension.
        inner: u32,
        /// Output columns.
        cols: u32,
        /// Whether `b` is transposed (`a·bᵀ`).
        transpose_b: bool,
    },
    /// Row-wise softmax over a `[rows, row_len]` buffer via a committed exp table,
    /// scaled by `one` (per-row `Σ ≈ one`), exactly recomputed.
    Softmax {
        /// Op id.
        op_id: u32,
        /// Committed exp table id.
        table_id: u32,
        /// Input buffer (`[rows, row_len]`, row-major).
        in_buf: u32,
        /// Output buffer.
        out_buf: u32,
        /// Claimed output.
        out: Vec<i64>,
        /// Row length (softmax is applied per row).
        row_len: u32,
        /// Fixed-point unit the row sums to.
        one: i64,
    },
    /// Batched linear: apply one weight `W` (`[rows, cols]`) to **every row** of a
    /// `[seq, cols]` sequence buffer, producing `[seq, rows]` (the ViT/transformer
    /// projection). The Freivalds `v = rᵀW` is precomputed once and checked per
    /// row — the amortized sequence linear.
    BatchedLinear {
        /// Op id.
        op_id: u32,
        /// Weight matrix id (`[rows, cols]`).
        weight_id: u32,
        /// Optional bias id (`[rows]`).
        bias_id: Option<u32>,
        /// Input buffer (`[seq, cols]`, row-major).
        in_buf: u32,
        /// Output buffer (`[seq, rows]`, row-major).
        out_buf: u32,
        /// Claimed output.
        out: Vec<i64>,
        /// Sequence length (number of rows).
        seq: u32,
    },
    /// Contiguous slice `out = in_buf[start..start+len]` (split a per-position
    /// row out of a packed buffer, or one AdaLN parameter out of the 6-way chunk).
    Slice {
        /// Op id.
        op_id: u32,
        /// Source buffer.
        in_buf: u32,
        /// Output buffer.
        out_buf: u32,
        /// Start offset.
        start: u32,
        /// Length.
        len: u32,
        /// Claimed output.
        out: Vec<i64>,
    },
    /// Concatenate `in_bufs` in order (recombine per-position projections).
    Concat {
        /// Op id.
        op_id: u32,
        /// Source buffers, concatenated in order.
        in_bufs: Vec<u32>,
        /// Output buffer.
        out_buf: u32,
        /// Claimed output.
        out: Vec<i64>,
    },
}

impl BlockOp {
    /// Op id.
    pub fn op_id(&self) -> u32 {
        match self {
            BlockOp::Linear { op_id, .. }
            | BlockOp::Requant { op_id, .. }
            | BlockOp::Activation { op_id, .. }
            | BlockOp::LayerNorm { op_id, .. }
            | BlockOp::Modulate { op_id, .. }
            | BlockOp::Gate { op_id, .. }
            | BlockOp::Add { op_id, .. }
            | BlockOp::MatMul { op_id, .. }
            | BlockOp::Softmax { op_id, .. }
            | BlockOp::BatchedLinear { op_id, .. }
            | BlockOp::Slice { op_id, .. }
            | BlockOp::Concat { op_id, .. } => *op_id,
        }
    }

    /// Output buffer id.
    pub fn out_buf(&self) -> u32 {
        match self {
            BlockOp::Linear { out_buf, .. }
            | BlockOp::Requant { out_buf, .. }
            | BlockOp::Activation { out_buf, .. }
            | BlockOp::LayerNorm { out_buf, .. }
            | BlockOp::Modulate { out_buf, .. }
            | BlockOp::Gate { out_buf, .. }
            | BlockOp::Add { out_buf, .. }
            | BlockOp::MatMul { out_buf, .. }
            | BlockOp::Softmax { out_buf, .. }
            | BlockOp::BatchedLinear { out_buf, .. }
            | BlockOp::Slice { out_buf, .. }
            | BlockOp::Concat { out_buf, .. } => *out_buf,
        }
    }

    /// Claimed output values.
    pub fn out(&self) -> &[i64] {
        match self {
            BlockOp::Linear { out, .. }
            | BlockOp::Requant { out, .. }
            | BlockOp::Activation { out, .. }
            | BlockOp::LayerNorm { out, .. }
            | BlockOp::Modulate { out, .. }
            | BlockOp::Gate { out, .. }
            | BlockOp::Add { out, .. }
            | BlockOp::MatMul { out, .. }
            | BlockOp::Softmax { out, .. }
            | BlockOp::BatchedLinear { out, .. }
            | BlockOp::Slice { out, .. }
            | BlockOp::Concat { out, .. } => out,
        }
    }

    const fn tag(&self) -> u8 {
        match self {
            BlockOp::Linear { .. } => 0,
            BlockOp::Requant { .. } => 1,
            BlockOp::Activation { .. } => 2,
            BlockOp::LayerNorm { .. } => 3,
            BlockOp::Modulate { .. } => 4,
            BlockOp::Gate { .. } => 5,
            BlockOp::Add { .. } => 6,
            BlockOp::MatMul { .. } => 7,
            BlockOp::Softmax { .. } => 8,
            BlockOp::Slice { .. } => 9,
            BlockOp::Concat { .. } => 10,
            BlockOp::BatchedLinear { .. } => 11,
        }
    }

    /// Merkle leaf hash of this op.
    pub fn leaf(&self) -> [u8; 32] {
        let mut p = Vec::new();
        p.extend_from_slice(TAG_BLOCK_LEAF);
        self.encode(&mut p);
        blake2s256(&p)
    }
}

impl CanonicalEncode for BlockOp {
    fn encode(&self, out: &mut Vec<u8>) {
        out.push(self.tag());
        match self {
            BlockOp::Linear {
                op_id,
                weight_id,
                bias_id,
                in_buf,
                out_buf,
                out: o,
            } => {
                op_id.encode(out);
                weight_id.encode(out);
                bias_id.encode(out);
                in_buf.encode(out);
                out_buf.encode(out);
                o.encode(out);
            }
            BlockOp::Requant {
                op_id,
                in_buf,
                out_buf,
                out: o,
                shift,
                zero_point,
                clamp_lo,
                clamp_hi,
                rounding,
            } => {
                op_id.encode(out);
                in_buf.encode(out);
                out_buf.encode(out);
                o.encode(out);
                (*shift as u64).encode(out);
                zero_point.encode(out);
                clamp_lo.encode(out);
                clamp_hi.encode(out);
                out.push(*rounding);
            }
            BlockOp::Activation {
                op_id,
                table_id,
                in_buf,
                out_buf,
                out: o,
            } => {
                op_id.encode(out);
                table_id.encode(out);
                in_buf.encode(out);
                out_buf.encode(out);
                o.encode(out);
            }
            BlockOp::LayerNorm {
                op_id,
                table_id,
                in_buf,
                out_buf,
                out: o,
                shift,
                clamp_lo,
                clamp_hi,
                rounding,
            } => {
                op_id.encode(out);
                table_id.encode(out);
                in_buf.encode(out);
                out_buf.encode(out);
                o.encode(out);
                (*shift as u64).encode(out);
                clamp_lo.encode(out);
                clamp_hi.encode(out);
                out.push(*rounding);
            }
            BlockOp::Modulate {
                op_id,
                x_buf,
                scale_buf,
                shift_buf,
                out_buf,
                out: o,
                one,
                shift_bits,
                clamp_lo,
                clamp_hi,
                rounding,
            } => {
                op_id.encode(out);
                x_buf.encode(out);
                scale_buf.encode(out);
                shift_buf.encode(out);
                out_buf.encode(out);
                o.encode(out);
                one.encode(out);
                (*shift_bits as u64).encode(out);
                clamp_lo.encode(out);
                clamp_hi.encode(out);
                out.push(*rounding);
            }
            BlockOp::Gate {
                op_id,
                gate_buf,
                x_buf,
                out_buf,
                out: o,
                shift_bits,
                clamp_lo,
                clamp_hi,
                rounding,
            } => {
                op_id.encode(out);
                gate_buf.encode(out);
                x_buf.encode(out);
                out_buf.encode(out);
                o.encode(out);
                (*shift_bits as u64).encode(out);
                clamp_lo.encode(out);
                clamp_hi.encode(out);
                out.push(*rounding);
            }
            BlockOp::Add {
                op_id,
                a_buf,
                b_buf,
                out_buf,
                out: o,
            } => {
                op_id.encode(out);
                a_buf.encode(out);
                b_buf.encode(out);
                out_buf.encode(out);
                o.encode(out);
            }
            BlockOp::MatMul {
                op_id,
                a_buf,
                b_buf,
                out_buf,
                out: o,
                rows,
                inner,
                cols,
                transpose_b,
            } => {
                op_id.encode(out);
                a_buf.encode(out);
                b_buf.encode(out);
                out_buf.encode(out);
                o.encode(out);
                rows.encode(out);
                inner.encode(out);
                cols.encode(out);
                transpose_b.encode(out);
            }
            BlockOp::Softmax {
                op_id,
                table_id,
                in_buf,
                out_buf,
                out: o,
                row_len,
                one,
            } => {
                op_id.encode(out);
                table_id.encode(out);
                in_buf.encode(out);
                out_buf.encode(out);
                o.encode(out);
                row_len.encode(out);
                one.encode(out);
            }
            BlockOp::Slice {
                op_id,
                in_buf,
                out_buf,
                start,
                len,
                out: o,
            } => {
                op_id.encode(out);
                in_buf.encode(out);
                out_buf.encode(out);
                start.encode(out);
                len.encode(out);
                o.encode(out);
            }
            BlockOp::Concat {
                op_id,
                in_bufs,
                out_buf,
                out: o,
            } => {
                op_id.encode(out);
                in_bufs.encode(out);
                out_buf.encode(out);
                o.encode(out);
            }
            BlockOp::BatchedLinear {
                op_id,
                weight_id,
                bias_id,
                in_buf,
                out_buf,
                out: o,
                seq,
            } => {
                op_id.encode(out);
                weight_id.encode(out);
                bias_id.encode(out);
                in_buf.encode(out);
                out_buf.encode(out);
                o.encode(out);
                seq.encode(out);
            }
        }
    }
}

/// A named-buffer block: an ordered op list plus the ids of the input buffers it
/// expects to be seeded (the block's external inputs).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Block {
    /// Buffer ids the block consumes as external input.
    pub input_bufs: Vec<u32>,
    /// The ordered ops.
    pub ops: Vec<BlockOp>,
    /// The buffer id holding the block's output.
    pub output_buf: u32,
}

/// The block Merkle root over the ordered op leaves (same shape as the trace root).
pub fn block_root(ops: &[BlockOp]) -> [u8; 32] {
    let mut level: Vec<[u8; 32]> = ops.iter().map(BlockOp::leaf).collect();
    if level.is_empty() {
        return blake2s256(TAG_BLOCK_NODE);
    }
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut i = 0;
        while i < level.len() {
            let a = level[i];
            let b = if i + 1 < level.len() {
                level[i + 1]
            } else {
                level[i]
            };
            let mut p = Vec::with_capacity(16 + 64);
            p.extend_from_slice(TAG_BLOCK_NODE);
            p.extend_from_slice(&a);
            p.extend_from_slice(&b);
            next.push(blake2s256(&p));
            i += 2;
        }
        level = next;
    }
    level[0]
}
