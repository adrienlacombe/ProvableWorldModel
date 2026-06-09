// SPDX-License-Identifier: Apache-2.0
//! The full le-wm conditional predictor block wired into `prove_block`.
//!
//! This builds the real le-wm predictor architecture as a named-buffer block DAG
//! and proves one P0 step in exact integer arithmetic: per ConditionalBlock,
//! AdaLN-zero conditioning (SiLU(c) -> Linear -> chunk6) modulates a multi-head
//! self-attention sub-block (per head: `Q.Kᵀ` -> softmax -> `prob.V`,
//! `heads=H`) and a GELU feed-forward, each with a residual gate. The multi-head
//! reshapes become Slice/Concat over the flat buffers.
//!
//! Quantization is a self-consistent integer scheme: every Linear, Modulate, Gate
//! and residual is requant-clamped back to int8, so all activations stay int8 and
//! the committed tables (inverse-sqrt, GELU, SiLU, a bounded monotonic softmax exp)
//! always cover. The proof attests this quantized relation. The graph is validated
//! at small dims and then instantiated at the real V0 dims (192/16/64).

use pwm_core::block::{Block, BlockOp};
use pwm_core::fixed_point::BoundedInt;
use pwm_core::tables::ActivationTable;
use pwm_core::tensor::Tensor;
use pwm_prover::prove_block;
use pwm_verifier::{verify_block, VerifyError};

use crate::bundle::{self, BundleError};

const CLO: i64 = -128;
const CHI: i64 = 127;
const RND: u8 = 0; // NearestTiesToEven

/// What a predictor build returns: the skeleton block, the weight tensors, the
/// committed tables, and the seeded input buffers, all ready for `prove_block`.
pub type Built = (
    Block,
    Vec<Tensor>,
    Vec<ActivationTable>,
    Vec<(u32, Vec<i64>)>,
);

/// Predictor dimensions.
#[derive(Clone, Copy)]
pub struct Dims {
    /// Latent dimension.
    pub d: usize,
    /// History length (sequence).
    pub s: usize,
    /// Attention heads.
    pub h: usize,
    /// Per-head dimension.
    pub dh: usize,
    /// FFN hidden width.
    pub mlp: usize,
    /// Number of predictor blocks.
    pub depth: usize,
}

impl Dims {
    /// `heads * dim_head` (the attention inner width).
    pub fn inner(&self) -> usize {
        self.h * self.dh
    }
}

/// Incrementally builds a block-op DAG, tracking fresh buffer and op ids.
pub struct Builder {
    ops: Vec<BlockOp>,
    next_buf: u32,
    next_op: u32,
    next_weight: u32,
    inputs: Vec<(u32, Vec<i64>)>,
    weights: Vec<Tensor>,
}

impl Builder {
    /// New empty builder.
    pub fn new() -> Self {
        Builder {
            ops: Vec::new(),
            next_buf: 0,
            next_op: 1,
            next_weight: 1000,
            inputs: Vec::new(),
            weights: Vec::new(),
        }
    }

    fn buf(&mut self) -> u32 {
        let b = self.next_buf;
        self.next_buf += 1;
        b
    }
    fn op(&mut self) -> u32 {
        let o = self.next_op;
        self.next_op += 1;
        o
    }

    /// Seed an external input buffer with values.
    pub fn input(&mut self, vals: Vec<i64>) -> u32 {
        let b = self.buf();
        self.inputs.push((b, vals));
        b
    }

    /// Register a quantized int8 weight matrix `[rows, cols]`; returns its id.
    pub fn weight(&mut self, rows: usize, cols: usize, vals: &[i64]) -> u32 {
        let id = self.next_weight;
        self.next_weight += 1;
        let data = vals
            .iter()
            .map(|&v| BoundedInt::new(v, -128, 127).expect("int8 weight"))
            .collect();
        self.weights
            .push(Tensor::new(id, vec![rows as u32, cols as u32], 0, data).expect("tensor"));
        id
    }

    fn requant(&mut self, in_buf: u32, shift: u32) -> u32 {
        let out = self.buf();
        let op = self.op();
        self.ops.push(BlockOp::Requant {
            op_id: op,
            in_buf,
            out_buf: out,
            out: vec![],
            shift,
            zero_point: 0,
            clamp_lo: CLO,
            clamp_hi: CHI,
            rounding: RND,
        });
        out
    }

    fn batched_linear(&mut self, in_buf: u32, weight_id: u32, seq: usize) -> u32 {
        let out = self.buf();
        let op = self.op();
        self.ops.push(BlockOp::BatchedLinear {
            op_id: op,
            weight_id,
            bias_id: None,
            in_buf,
            out_buf: out,
            out: vec![],
            seq: seq as u32,
        });
        out
    }

    fn activation(&mut self, in_buf: u32, table_id: u32) -> u32 {
        let out = self.buf();
        let op = self.op();
        self.ops.push(BlockOp::Activation {
            op_id: op,
            table_id,
            in_buf,
            out_buf: out,
            out: vec![],
        });
        out
    }

    fn layernorm(&mut self, in_buf: u32, table_id: u32) -> u32 {
        let out = self.buf();
        let op = self.op();
        self.ops.push(BlockOp::LayerNorm {
            op_id: op,
            table_id,
            in_buf,
            out_buf: out,
            out: vec![],
            shift: 0,
            clamp_lo: CLO,
            clamp_hi: CHI,
            rounding: RND,
        });
        out
    }

    fn modulate(&mut self, x: u32, scale: u32, shift: u32) -> u32 {
        let out = self.buf();
        let op = self.op();
        self.ops.push(BlockOp::Modulate {
            op_id: op,
            x_buf: x,
            scale_buf: scale,
            shift_buf: shift,
            out_buf: out,
            out: vec![],
            one: 1,
            shift_bits: 0,
            clamp_lo: CLO,
            clamp_hi: CHI,
            rounding: RND,
        });
        out
    }

    fn gate(&mut self, g: u32, x: u32) -> u32 {
        let out = self.buf();
        let op = self.op();
        self.ops.push(BlockOp::Gate {
            op_id: op,
            gate_buf: g,
            x_buf: x,
            out_buf: out,
            out: vec![],
            shift_bits: 0,
            clamp_lo: CLO,
            clamp_hi: CHI,
            rounding: RND,
        });
        out
    }

    fn add(&mut self, a: u32, b: u32) -> u32 {
        let out = self.buf();
        let op = self.op();
        self.ops.push(BlockOp::Add {
            op_id: op,
            a_buf: a,
            b_buf: b,
            out_buf: out,
            out: vec![],
        });
        out
    }

    fn matmul(&mut self, a: u32, b: u32, rows: usize, inner: usize, cols: usize, t: bool) -> u32 {
        let out = self.buf();
        let op = self.op();
        self.ops.push(BlockOp::MatMul {
            op_id: op,
            a_buf: a,
            b_buf: b,
            out_buf: out,
            out: vec![],
            rows: rows as u32,
            inner: inner as u32,
            cols: cols as u32,
            transpose_b: t,
        });
        out
    }

    fn softmax(&mut self, in_buf: u32, table_id: u32, row_len: usize, one: i64) -> u32 {
        let out = self.buf();
        let op = self.op();
        self.ops.push(BlockOp::Softmax {
            op_id: op,
            table_id,
            in_buf,
            out_buf: out,
            out: vec![],
            row_len: row_len as u32,
            one,
        });
        out
    }

    fn slice(&mut self, in_buf: u32, start: usize, len: usize) -> u32 {
        let out = self.buf();
        let op = self.op();
        self.ops.push(BlockOp::Slice {
            op_id: op,
            in_buf,
            out_buf: out,
            start: start as u32,
            len: len as u32,
            out: vec![],
        });
        out
    }

    fn concat(&mut self, bufs: Vec<u32>) -> u32 {
        let out = self.buf();
        let op = self.op();
        self.ops.push(BlockOp::Concat {
            op_id: op,
            in_bufs: bufs,
            out_buf: out,
            out: vec![],
        });
        out
    }

    /// Per-position affine-free LayerNorm over `[s, d]`: slice each row, normalize,
    /// concat. Returns `[s*d]`.
    fn ln_per_pos(&mut self, x: u32, s: usize, d: usize, invsqrt: u32) -> u32 {
        let rows: Vec<u32> = (0..s)
            .map(|p| {
                let r = self.slice(x, p * d, d);
                self.layernorm(r, invsqrt)
            })
            .collect();
        self.concat(rows)
    }

    /// Extract a column band `[col..col+len]` from every row of `[s, width]`,
    /// concatenated to `[s*len]`.
    fn extract_cols(&mut self, src: u32, s: usize, width: usize, col: usize, len: usize) -> u32 {
        let rows: Vec<u32> = (0..s)
            .map(|p| self.slice(src, p * width + col, len))
            .collect();
        self.concat(rows)
    }
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}

// --- table ids ---
const SILU: u32 = 1;
const GELU: u32 = 2;
const INVSQRT: u32 = 3;
const EXP: u32 = 4;
const SOFTMAX_ONE: i64 = 1000;

/// The committed integer tables. Self-consistent for the int8 activation scheme:
/// SiLU/GELU identity over int8, inverse-sqrt returning unit over the variance
/// domain, and a bounded monotonic softmax exp over the shifted-score domain.
pub fn tables() -> Vec<ActivationTable> {
    vec![
        ActivationTable {
            table_id: SILU,
            lo: -128,
            outputs: (-128i64..=127).collect(),
        },
        ActivationTable {
            table_id: GELU,
            lo: -128,
            outputs: (-128i64..=127).collect(),
        },
        ActivationTable {
            table_id: INVSQRT,
            lo: 0,
            outputs: vec![1; 70_001],
        },
        ActivationTable {
            table_id: EXP,
            lo: -255,
            outputs: (1i64..=256).collect(),
        },
    ]
}

/// Weight ids for one ConditionalBlock.
#[derive(Clone, Copy)]
pub struct BlockWeights {
    adaln: u32,
    qkv: u32,
    out: u32,
    fc1: u32,
    fc2: u32,
}

fn lin_shift(cols: usize) -> u32 {
    ((cols as f64) * 127.0).log2().ceil() as u32
}

/// A deterministic int8 weight matrix `[rows, cols]` (small, in {-1,0,1}).
fn synth(b: &mut Builder, rows: usize, cols: usize) -> u32 {
    let vals: Vec<i64> = (0..rows * cols).map(|i| (i as i64 % 3) - 1).collect();
    b.weight(rows, cols, &vals)
}

fn synth_block_weights(b: &mut Builder, d: Dims) -> BlockWeights {
    BlockWeights {
        adaln: synth(b, 6 * d.d, d.d),
        qkv: synth(b, 3 * d.inner(), d.d),
        out: synth(b, d.d, d.inner()),
        fc1: synth(b, d.mlp, d.d),
        fc2: synth(b, d.d, d.mlp),
    }
}

/// Multi-head self-attention sub-block (with the attn.norm affine-free LayerNorm).
fn attn(b: &mut Builder, x: u32, d: Dims, w: BlockWeights) -> u32 {
    let inner = d.inner();
    let a = b.ln_per_pos(x, d.s, d.d, INVSQRT);
    let qkv = b.batched_linear(a, w.qkv, d.s);
    let qkv = b.requant(qkv, lin_shift(d.d));
    let q = b.extract_cols(qkv, d.s, 3 * inner, 0, inner);
    let k = b.extract_cols(qkv, d.s, 3 * inner, inner, inner);
    let v = b.extract_cols(qkv, d.s, 3 * inner, 2 * inner, inner);
    let score_shift = lin_shift(d.dh);
    let oh_shift = ((SOFTMAX_ONE as f64).log2().ceil()) as u32;
    let mut head_outs = Vec::with_capacity(d.h);
    for h in 0..d.h {
        let qh = b.extract_cols(q, d.s, inner, h * d.dh, d.dh);
        let kh = b.extract_cols(k, d.s, inner, h * d.dh, d.dh);
        let vh = b.extract_cols(v, d.s, inner, h * d.dh, d.dh);
        let scores = b.matmul(qh, kh, d.s, d.dh, d.s, true);
        let scores = b.requant(scores, score_shift);
        let prob = b.softmax(scores, EXP, d.s, SOFTMAX_ONE);
        let oh = b.matmul(prob, vh, d.s, d.s, d.dh, false);
        head_outs.push(b.requant(oh, oh_shift));
    }
    // Reassemble [s, inner] = per position, concat the heads' [dh] slices.
    let mut parts = Vec::with_capacity(d.s * d.h);
    for p in 0..d.s {
        for oh in &head_outs {
            parts.push(b.slice(*oh, p * d.dh, d.dh));
        }
    }
    let attn_cat = b.concat(parts);
    let out = b.batched_linear(attn_cat, w.out, d.s);
    b.requant(out, lin_shift(inner))
}

/// GELU feed-forward sub-block (with the mlp.net.0 affine-free LayerNorm).
fn ffn(b: &mut Builder, x: u32, d: Dims, w: BlockWeights) -> u32 {
    let f = b.ln_per_pos(x, d.s, d.d, INVSQRT);
    let h1 = b.batched_linear(f, w.fc1, d.s);
    let h1 = b.requant(h1, lin_shift(d.d));
    let g = b.activation(h1, GELU);
    let h2 = b.batched_linear(g, w.fc2, d.s);
    b.requant(h2, lin_shift(d.mlp))
}

/// One AdaLN-zero ConditionalBlock: `x = x + gate_msa * attn(modulate(norm1(x)))`
/// then `x = x + gate_mlp * mlp(modulate(norm2(x)))`.
fn conditional_block(b: &mut Builder, x: u32, c: u32, d: Dims, w: BlockWeights) -> u32 {
    // AdaLN: SiLU(c) -> Linear -> chunk6 (per position).
    let silu_c = b.activation(c, SILU);
    let adaln = b.batched_linear(silu_c, w.adaln, d.s);
    let adaln = b.requant(adaln, lin_shift(d.d));
    let chunk = |b: &mut Builder, j: usize| b.extract_cols(adaln, d.s, 6 * d.d, j * d.d, d.d);
    let shift_msa = chunk(b, 0);
    let scale_msa = chunk(b, 1);
    let gate_msa = chunk(b, 2);
    let shift_mlp = chunk(b, 3);
    let scale_mlp = chunk(b, 4);
    let gate_mlp = chunk(b, 5);
    // Attention sub-block + gated residual.
    let n1 = b.ln_per_pos(x, d.s, d.d, INVSQRT);
    let m1 = b.modulate(n1, scale_msa, shift_msa);
    let a = attn(b, m1, d, w);
    let g1 = b.gate(gate_msa, a);
    let x = b.add(x, g1);
    let x = b.requant(x, 0);
    // FFN sub-block + gated residual.
    let n2 = b.ln_per_pos(x, d.s, d.d, INVSQRT);
    let m2 = b.modulate(n2, scale_mlp, shift_mlp);
    let f = ffn(b, m2, d, w);
    let g2 = b.gate(gate_mlp, f);
    let x = b.add(x, g2);
    b.requant(x, 0)
}

/// The real quantized int8 weights of one ConditionalBlock (row-major, matching
/// the shapes the builder expects).
pub struct RealBlock {
    /// Fused QKV `[3*inner, d]`.
    pub qkv: Vec<i64>,
    /// Attention out-projection `[d, inner]`.
    pub out: Vec<i64>,
    /// FFN fc1 `[mlp, d]`.
    pub fc1: Vec<i64>,
    /// FFN fc2 `[d, mlp]`.
    pub fc2: Vec<i64>,
    /// AdaLN modulation `[6*d, d]`.
    pub adaln: Vec<i64>,
}

/// Core builder: assemble the `depth`-block predictor over the given inputs, with
/// each block's weights supplied by `block_weights`. The graph is the real le-wm
/// architecture; the weight source (synthetic or real) is the caller's choice.
pub fn build_predictor_with(
    d: Dims,
    xv: Vec<i64>,
    cv: Vec<i64>,
    mut block_weights: impl FnMut(&mut Builder, Dims) -> BlockWeights,
) -> Built {
    let mut b = Builder::new();
    let mut x = b.input(xv);
    let c = b.input(cv);
    for _ in 0..d.depth {
        let w = block_weights(&mut b, d);
        x = conditional_block(&mut b, x, c, d, w);
    }
    // Final transformer norm (affine-free).
    let out = b.ln_per_pos(x, d.s, d.d, INVSQRT);
    let block = Block {
        input_bufs: b.inputs.iter().map(|(id, _)| *id).collect(),
        ops: b.ops,
        output_buf: out,
    };
    (block, b.weights, tables(), b.inputs)
}

/// Build the full predictor over **synthetic** int8 weights and inputs. The graph
/// is the real le-wm architecture, so this validates at small dims and runs at the
/// real V0 dims.
pub fn build_predictor(d: Dims) -> Built {
    let xv: Vec<i64> = (0..d.s * d.d).map(|i| (i as i64 % 5) - 2).collect();
    let cv: Vec<i64> = (0..d.s * d.d).map(|i| (i as i64 % 3) - 1).collect();
    build_predictor_with(d, xv, cv, synth_block_weights)
}

/// Build the full predictor over the **real** quantized checkpoint weights, a
/// quantized latent history `xv` and action embedding `cv`.
pub fn build_predictor_real(d: Dims, blocks: Vec<RealBlock>, xv: Vec<i64>, cv: Vec<i64>) -> Built {
    let mut it = blocks.into_iter();
    build_predictor_with(d, xv, cv, move |b, d| {
        let rb = it.next().expect("a RealBlock per depth");
        BlockWeights {
            adaln: b.weight(6 * d.d, d.d, &rb.adaln),
            qkv: b.weight(3 * d.inner(), d.d, &rb.qkv),
            out: b.weight(d.d, d.inner(), &rb.out),
            fc1: b.weight(d.mlp, d.d, &rb.fc1),
            fc2: b.weight(d.d, d.mlp, &rb.fc2),
        }
    })
}

/// Run the prover (the exact integer reference) over a predictor skeleton.
pub fn prove(
    skeleton: &Block,
    weights: &[Tensor],
    tabs: &[ActivationTable],
    inputs: &[(u32, Vec<i64>)],
) -> Block {
    prove_block(skeleton, weights, tabs, inputs).expect("prove predictor")
}

/// Audit a proven predictor block; returns the verified output.
pub fn verify(
    proven: &Block,
    weights: &[Tensor],
    tabs: &[ActivationTable],
    inputs: &[(u32, Vec<i64>)],
) -> Result<Vec<i64>, VerifyError> {
    verify_block(proven, weights, tabs, inputs)
}

/// Forge the first attention/FFN projection output; the Freivalds check rejects it.
pub fn tamper(proven: &mut Block) -> Option<u32> {
    for op in &mut proven.ops {
        if let BlockOp::BatchedLinear { op_id, out, .. } = op {
            if let Some(first) = out.first_mut() {
                *first += 1;
            }
            return Some(*op_id);
        }
    }
    None
}

/// A loaded predictor bundle: dims, per-block real weights, the quantized inputs,
/// and a description of where the inputs came from.
pub struct RealPredictor {
    /// Predictor dimensions.
    pub dims: Dims,
    /// Per-block real quantized weights.
    pub blocks: Vec<RealBlock>,
    /// Quantized latent history input.
    pub x: Vec<i64>,
    /// Quantized action embedding input.
    pub c: Vec<i64>,
    /// Provenance of the inputs (real observation vs synthetic).
    pub input_source: String,
}

/// Parse a full predictor export bundle, or a [`BundleError`] describing what was
/// malformed (so the CLI can report it cleanly instead of panicking).
pub fn load_real_predictor(json: &str) -> Result<RealPredictor, BundleError> {
    let v = bundle::parse(json)?;
    let dv = bundle::field(&v, "dims")?;
    let dims = Dims {
        d: bundle::u64_at(dv, "d")? as usize,
        s: bundle::u64_at(dv, "s")? as usize,
        h: bundle::u64_at(dv, "h")? as usize,
        dh: bundle::u64_at(dv, "dh")? as usize,
        mlp: bundle::u64_at(dv, "mlp")? as usize,
        depth: bundle::u64_at(dv, "depth")? as usize,
    };
    let blocks = bundle::field(&v, "blocks")?
        .as_array()
        .ok_or(BundleError::WrongType {
            field: "blocks",
            expected: "an array",
        })?
        .iter()
        .map(|bk| {
            Ok(RealBlock {
                qkv: bundle::ints_at(bk, "qkv")?,
                out: bundle::ints_at(bk, "out")?,
                fc1: bundle::ints_at(bk, "fc1")?,
                fc2: bundle::ints_at(bk, "fc2")?,
                adaln: bundle::ints_at(bk, "adaln")?,
            })
        })
        .collect::<Result<Vec<_>, BundleError>>()?;
    Ok(RealPredictor {
        dims,
        blocks,
        x: bundle::ints_at(&v, "x")?,
        c: bundle::ints_at(&v, "c")?,
        input_source: bundle::str_at_or(&v, "input_source", ""),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(d: Dims) {
        let (skeleton, weights, tabs, inputs) = build_predictor(d);
        let proven = prove(&skeleton, &weights, &tabs, &inputs);
        verify(&proven, &weights, &tabs, &inputs).expect("verify");
    }

    #[test]
    fn small_dims_one_block_verifies() {
        run(Dims {
            d: 8,
            s: 3,
            h: 2,
            dh: 4,
            mlp: 16,
            depth: 1,
        });
    }

    #[test]
    fn tampered_predictor_is_rejected() {
        let d = Dims {
            d: 8,
            s: 3,
            h: 2,
            dh: 4,
            mlp: 16,
            depth: 1,
        };
        let (sk, w, t, inp) = build_predictor(d);
        let mut p = prove(&sk, &w, &t, &inp);
        let op = tamper(&mut p).expect("a linear op");
        assert!(matches!(
            verify(&p, &w, &t, &inp),
            Err(VerifyError::FreivaldsCheckFailed { op_id }) if op_id == op
        ));
    }

    #[test]
    fn real_weights_path_verifies() {
        // Exercise build_predictor_real with synthetic RealBlocks at small dims.
        let d = Dims {
            d: 8,
            s: 3,
            h: 2,
            dh: 4,
            mlp: 16,
            depth: 2,
        };
        let inner = d.inner();
        let mk = |rows: usize, cols: usize| (0..rows * cols).map(|i| (i as i64 % 3) - 1).collect();
        let blocks: Vec<RealBlock> = (0..d.depth)
            .map(|_| RealBlock {
                qkv: mk(3 * inner, d.d),
                out: mk(d.d, inner),
                fc1: mk(d.mlp, d.d),
                fc2: mk(d.d, d.mlp),
                adaln: mk(6 * d.d, d.d),
            })
            .collect();
        let xv = vec![1i64; d.s * d.d];
        let cv = vec![0i64; d.s * d.d];
        let (sk, w, t, inp) = build_predictor_real(d, blocks, xv, cv);
        let proven = prove(&sk, &w, &t, &inp);
        verify(&proven, &w, &t, &inp).expect("verify real-weights predictor");
    }

    #[test]
    fn small_dims_six_blocks_verify() {
        run(Dims {
            d: 8,
            s: 3,
            h: 2,
            dh: 4,
            mlp: 16,
            depth: 6,
        });
    }

    // The real le-wm V0 dims (192/3/16/64/2048, depth 6, ~2.4k ops). Heavy in a
    // debug build, so it is `#[ignore]`d for the default `cargo test`; the CI test
    // job runs it in release via `cargo test --release -- --ignored` (~0.1 s there),
    // so the headline "the full 192-dim predictor verifies" claim is gated.
    #[test]
    #[ignore]
    fn real_v0_dims_full_predictor_verifies() {
        run(Dims {
            d: 192,
            s: 3,
            h: 16,
            dh: 64,
            mlp: 2048,
            depth: 6,
        });
    }
}
