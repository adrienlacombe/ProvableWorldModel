// SPDX-License-Identifier: Apache-2.0
//! The full le-wm conditional predictor block wired into the commitment-bound
//! predictor relation (`prove_predictor` / `verify_predictor`).
//!
//! This builds the real le-wm predictor architecture as a named-buffer block DAG
//! and proves one P0 step in exact integer arithmetic: per ConditionalBlock,
//! AdaLN-zero conditioning (SiLU(c) -> Linear -> chunk6) modulates a multi-head
//! self-attention sub-block (per head: `Q.Kᵀ` -> softmax -> `prob.V`,
//! `heads=H`) and a GELU feed-forward, each with a residual gate. The multi-head
//! reshapes become Slice/Concat over the flat buffers.
//!
//! Quantization is the **calibrated float-faithful** scheme from
//! [`pwm_export::predictor_quant`]: per-site power-of-two scales derived from the
//! float reference forward pass, real committed GELU / SiLU / inverse-sqrt /
//! softmax-exp tables generated at those scales, and per-op requant shifts that
//! carry each accumulator back to its calibrated frac. The proven integer output
//! therefore approximates the float predictor within a pinned tolerance, and the
//! proof binds the model / quantization / input / output commitments in the
//! artifact's [`PublicInput`]. The graph is validated at small dims and then
//! instantiated at the real V0 dims (192/16/64).
//!
//! [`PublicInput`]: pwm_core::public_input::PublicInput

use pwm_core::audit::PredictorArtifact;
use pwm_core::block::{Block, BlockOp};
use pwm_core::fixed_point::BoundedInt;
use pwm_core::tables::ActivationTable;
use pwm_core::tensor::{Dtype, Scale, Tensor};
use pwm_export::predictor_quant::{
    calibrate, derive, quantize_at, quantize_weights, scheme_tables, synth_float_predictor,
    BlockShifts, DerivedParams, PredictorDims, QuantScheme, TABLE_EXP, TABLE_GELU, TABLE_INVSQRT,
    TABLE_SILU,
};
use pwm_prover::{prove_predictor, prove_predictor_with_weights_root, OutputBinding, ProveError};
use pwm_verifier::{verify_predictor, VerifyError};

use crate::bundle::{self, BundleError};

const CLO: i64 = -128;
const CHI: i64 = 127;
const RND: u8 = 0; // NearestTiesToEven

/// Tensor id of the claimed predictor output (outside the weight id range).
const OUT_TENSOR_ID: u32 = 2000;
/// Scale id of the latent-history input (`log2 = -f_x`).
const SCALE_X: u32 = 0;
/// Scale id of the action-embedding input (`log2 = -f_c`).
const SCALE_C: u32 = 1;
/// Scale id of the LayerNorm outputs and the claimed output (`log2 = -f_ln`).
const SCALE_OUT: u32 = 2;
/// Weight tensor `k` (registration order) uses scale id `SCALE_W_BASE + k`.
const SCALE_W_BASE: u32 = 10;

/// Max |int·2^-f − float| tolerance pinned for the synthetic profile (the
/// deterministic float model quantized by [`build_predictor`]). The real
/// checkpoint bundle carries its own export-measured tolerance.
pub const SYNTH_TOLERANCE: f64 = 0.25;

/// Everything a predictor build produces, ready for the commitment-bound prover:
/// the skeleton block, the weight tensors, the committed calibrated tables, the
/// scale table (binding the scheme's fracs), and the seeded input buffers.
pub struct PredictorCircuit {
    /// The block-op DAG with empty (unproven) op outputs.
    pub block: Block,
    /// The quantized int8 weight tensors (their Merkle root binds the model).
    pub weights: Vec<Tensor>,
    /// The committed calibrated activation tables (bound by the quantization
    /// commitment).
    pub tables: Vec<ActivationTable>,
    /// The scale table binding the scheme: input/output fracs and the per-tensor
    /// weight scales (bound by the quantization commitment).
    pub scales: Vec<Scale>,
    /// The seeded public input buffers (latent history + action embedding).
    pub inputs: Vec<(u32, Vec<i64>)>,
    /// The calibrated fixed-point scheme the circuit instantiates.
    pub scheme: QuantScheme,
    /// The float reference output this circuit approximates (`[s, d]`).
    pub z_float: Option<Vec<f64>>,
    /// Max-abs error tolerance for `|z_int·2^-f_ln − z_float|`.
    pub tolerance: Option<f64>,
    /// The export-computed weights root carried by the bundle, if any. When set,
    /// it is bound into the model commitment *as carried* (not recomputed), so
    /// verification fails with `CommitmentMismatch(Model)` unless the circuit's
    /// weights reproduce the export's commitment bit-for-bit.
    pub export_weights_root: Option<[u8; 32]>,
}

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

    /// The same dims as the export crate's [`PredictorDims`].
    pub fn export(&self) -> PredictorDims {
        PredictorDims {
            d: self.d,
            s: self.s,
            h: self.h,
            dh: self.dh,
            mlp: self.mlp,
            depth: self.depth,
        }
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
    /// The `k`-th registered weight references scale id `SCALE_W_BASE + k`, so
    /// the quantization commitment binds its `log2` interpretation.
    pub fn weight(&mut self, rows: usize, cols: usize, vals: &[i64]) -> u32 {
        let id = self.next_weight;
        self.next_weight += 1;
        let scale_id = SCALE_W_BASE + self.weights.len() as u32;
        let data = vals
            .iter()
            .map(|&v| BoundedInt::new(v, -128, 127).expect("int8 weight"))
            .collect();
        self.weights
            .push(Tensor::new(id, vec![rows as u32, cols as u32], scale_id, data).expect("tensor"));
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

    fn layernorm(&mut self, in_buf: u32, table_id: u32, shift: u32) -> u32 {
        let out = self.buf();
        let op = self.op();
        self.ops.push(BlockOp::LayerNorm {
            op_id: op,
            table_id,
            in_buf,
            out_buf: out,
            out: vec![],
            shift,
            clamp_lo: CLO,
            clamp_hi: CHI,
            rounding: RND,
        });
        out
    }

    fn modulate(&mut self, x: u32, scale: u32, shift: u32, one: i64, shift_bits: u32) -> u32 {
        let out = self.buf();
        let op = self.op();
        self.ops.push(BlockOp::Modulate {
            op_id: op,
            x_buf: x,
            scale_buf: scale,
            shift_buf: shift,
            out_buf: out,
            out: vec![],
            one,
            shift_bits,
            clamp_lo: CLO,
            clamp_hi: CHI,
            rounding: RND,
        });
        out
    }

    fn gate(&mut self, g: u32, x: u32, shift_bits: u32) -> u32 {
        let out = self.buf();
        let op = self.op();
        self.ops.push(BlockOp::Gate {
            op_id: op,
            gate_buf: g,
            x_buf: x,
            out_buf: out,
            out: vec![],
            shift_bits,
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

    /// Per-position affine-free LayerNorm over `[s, d]`: slice each row, normalize
    /// with the scale-invariant inverse-sqrt table, requant `t_inv → f_ln`, concat.
    /// Returns `[s*d]`.
    fn ln_per_pos(&mut self, x: u32, s: usize, d: usize, shift: u32) -> u32 {
        let rows: Vec<u32> = (0..s)
            .map(|p| {
                let r = self.slice(x, p * d, d);
                self.layernorm(r, TABLE_INVSQRT, shift)
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

/// Weight ids for one ConditionalBlock.
#[derive(Clone, Copy)]
pub struct BlockWeights {
    adaln: u32,
    qkv: u32,
    out: u32,
    fc1: u32,
    fc2: u32,
}

/// Multi-head self-attention sub-block (with the attn.norm affine-free LayerNorm).
fn attn(
    b: &mut Builder,
    x: u32,
    d: Dims,
    w: BlockWeights,
    qp: &DerivedParams,
    bs: BlockShifts,
) -> u32 {
    let inner = d.inner();
    let a = b.ln_per_pos(x, d.s, d.d, qp.ln_shift);
    let qkv = b.batched_linear(a, w.qkv, d.s);
    let qkv = b.requant(qkv, bs.qkv);
    let q = b.extract_cols(qkv, d.s, 3 * inner, 0, inner);
    let k = b.extract_cols(qkv, d.s, 3 * inner, inner, inner);
    let v = b.extract_cols(qkv, d.s, 3 * inner, 2 * inner, inner);
    let mut head_outs = Vec::with_capacity(d.h);
    for h in 0..d.h {
        let qh = b.extract_cols(q, d.s, inner, h * d.dh, d.dh);
        let kh = b.extract_cols(k, d.s, inner, h * d.dh, d.dh);
        let vh = b.extract_cols(v, d.s, inner, h * d.dh, d.dh);
        let scores = b.matmul(qh, kh, d.s, d.dh, d.s, true);
        let scores = b.requant(scores, qp.score_shift);
        let prob = b.softmax(scores, TABLE_EXP, d.s, qp.softmax_one);
        let oh = b.matmul(prob, vh, d.s, d.s, d.dh, false);
        head_outs.push(b.requant(oh, qp.oh_shift));
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
    b.requant(out, bs.out)
}

/// GELU feed-forward sub-block (with the mlp.net.0 affine-free LayerNorm).
fn ffn(
    b: &mut Builder,
    x: u32,
    d: Dims,
    w: BlockWeights,
    qp: &DerivedParams,
    bs: BlockShifts,
) -> u32 {
    let f = b.ln_per_pos(x, d.s, d.d, qp.ln_shift);
    let h1 = b.batched_linear(f, w.fc1, d.s);
    let h1 = b.requant(h1, bs.fc1);
    let g = b.activation(h1, TABLE_GELU);
    let h2 = b.batched_linear(g, w.fc2, d.s);
    b.requant(h2, bs.fc2)
}

/// One AdaLN-zero ConditionalBlock: `x = x + gate_msa * attn(modulate(norm1(x)))`
/// then `x = x + gate_mlp * mlp(modulate(norm2(x)))`.
fn conditional_block(
    b: &mut Builder,
    x: u32,
    c: u32,
    d: Dims,
    w: BlockWeights,
    qp: &DerivedParams,
    bs: BlockShifts,
) -> u32 {
    // AdaLN: SiLU(c) -> Linear -> chunk6 (per position).
    let silu_c = b.activation(c, TABLE_SILU);
    let adaln = b.batched_linear(silu_c, w.adaln, d.s);
    let adaln = b.requant(adaln, bs.adaln);
    let chunk = |b: &mut Builder, j: usize| b.extract_cols(adaln, d.s, 6 * d.d, j * d.d, d.d);
    let shift_msa = chunk(b, 0);
    let scale_msa = chunk(b, 1);
    let gate_msa = chunk(b, 2);
    let shift_mlp = chunk(b, 3);
    let scale_mlp = chunk(b, 4);
    let gate_mlp = chunk(b, 5);
    // Attention sub-block + gated residual.
    let n1 = b.ln_per_pos(x, d.s, d.d, qp.ln_shift);
    let m1 = b.modulate(n1, scale_msa, shift_msa, qp.mod_one, qp.mod_bits);
    let a = attn(b, m1, d, w, qp, bs);
    let g1 = b.gate(gate_msa, a, qp.gate_msa_bits);
    let x = b.add(x, g1);
    let x = b.requant(x, 0);
    // FFN sub-block + gated residual.
    let n2 = b.ln_per_pos(x, d.s, d.d, qp.ln_shift);
    let m2 = b.modulate(n2, scale_mlp, shift_mlp, qp.mod_one, qp.mod_bits);
    let f = ffn(b, m2, d, w, qp, bs);
    let g2 = b.gate(gate_mlp, f, qp.gate_mlp_bits);
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

/// The scale table binding the scheme: the input/output fracs (`scale_id` 0/1/2)
/// and one entry per weight tensor (`SCALE_W_BASE + k`, `log2` from the scheme).
fn scheme_scales(s: &QuantScheme) -> Vec<Scale> {
    let mut v = vec![
        Scale {
            scale_id: SCALE_X,
            log2: -s.f_x,
            dtype: Dtype::I8,
        },
        Scale {
            scale_id: SCALE_C,
            log2: -s.f_c,
            dtype: Dtype::I8,
        },
        Scale {
            scale_id: SCALE_OUT,
            log2: -s.f_ln,
            dtype: Dtype::I8,
        },
    ];
    for (i, block) in s.w_log2.iter().enumerate() {
        for (j, &log2) in block.iter().enumerate() {
            v.push(Scale {
                scale_id: SCALE_W_BASE + (5 * i + j) as u32,
                log2,
                dtype: Dtype::I8,
            });
        }
    }
    v
}

/// Core builder: assemble the `depth`-block predictor over the given quantized
/// inputs and derived per-op parameters, with each block's weights supplied by
/// `block_weights`. The graph is the real le-wm architecture; the weight source
/// (synthetic or real) is the caller's choice.
pub fn build_predictor_with(
    d: Dims,
    xv: Vec<i64>,
    cv: Vec<i64>,
    scheme: QuantScheme,
    qp: &DerivedParams,
    tables: Vec<ActivationTable>,
    mut block_weights: impl FnMut(&mut Builder, Dims) -> BlockWeights,
) -> PredictorCircuit {
    assert_eq!(qp.blocks.len(), d.depth, "one BlockShifts per block");
    let mut b = Builder::new();
    let mut x = b.input(xv);
    let c = b.input(cv);
    for bs in &qp.blocks {
        let w = block_weights(&mut b, d);
        x = conditional_block(&mut b, x, c, d, w, qp, *bs);
    }
    // Final transformer norm (affine-free).
    let out = b.ln_per_pos(x, d.s, d.d, qp.ln_shift);
    let block = Block {
        input_bufs: b.inputs.iter().map(|(id, _)| *id).collect(),
        ops: b.ops,
        output_buf: out,
    };
    PredictorCircuit {
        block,
        weights: b.weights,
        tables,
        scales: scheme_scales(&scheme),
        inputs: b.inputs,
        scheme,
        z_float: None,
        tolerance: None,
        export_weights_root: None,
    }
}

/// Build the full predictor over a **synthetic** float model: deterministic
/// float weights and inputs, calibrated and quantized exactly like a real
/// export (real activation tables, per-site scales, float reference output).
/// The graph is the real le-wm architecture, so this validates at small dims
/// and runs at the real V0 dims.
pub fn build_predictor(d: Dims) -> PredictorCircuit {
    let pd = d.export();
    let (fblocks, xf, cf) = synth_float_predictor(pd);
    let (scheme, z_float) = calibrate(pd, &fblocks, &xf, &cf);
    let qp = derive(&scheme, pd).expect("synthetic scheme derives");
    let tables = scheme_tables(&scheme);
    let (qblocks, _) = quantize_weights(&fblocks);
    let xq = quantize_at(&xf, scheme.f_x);
    let cq = quantize_at(&cf, scheme.f_c);
    let mut it = qblocks.into_iter();
    let mut circuit = build_predictor_with(d, xq, cq, scheme, &qp, tables, move |b, d| {
        let qb = it.next().expect("a QuantBlock per depth");
        BlockWeights {
            adaln: b.weight(6 * d.d, d.d, &qb.adaln),
            qkv: b.weight(3 * d.inner(), d.d, &qb.qkv),
            out: b.weight(d.d, d.inner(), &qb.out),
            fc1: b.weight(d.mlp, d.d, &qb.fc1),
            fc2: b.weight(d.d, d.mlp, &qb.fc2),
        }
    });
    circuit.z_float = Some(z_float);
    circuit.tolerance = Some(SYNTH_TOLERANCE);
    circuit
}

/// Build the full predictor over the **real** quantized checkpoint weights, a
/// quantized latent history `xv` and action embedding `cv`, under the bundle's
/// calibrated scheme and committed tables. Fails with a typed error if the
/// scheme needs a negative requant shift or the block count mismatches.
pub fn build_predictor_real(
    d: Dims,
    blocks: Vec<RealBlock>,
    xv: Vec<i64>,
    cv: Vec<i64>,
    scheme: QuantScheme,
    tables: Vec<ActivationTable>,
) -> Result<PredictorCircuit, BundleError> {
    if scheme.w_log2.len() != d.depth || blocks.len() != d.depth {
        return Err(BundleError::WrongType {
            field: "quant.w_log2",
            expected: "one entry per block",
        });
    }
    let qp = derive(&scheme, d.export()).map_err(|e| BundleError::UnsoundScheme {
        site: e.site,
        shift: e.shift,
    })?;
    let mut it = blocks.into_iter();
    Ok(build_predictor_with(
        d,
        xv,
        cv,
        scheme,
        &qp,
        tables,
        move |b, d| {
            let rb = it.next().expect("a RealBlock per depth");
            BlockWeights {
                adaln: b.weight(6 * d.d, d.d, &rb.adaln),
                qkv: b.weight(3 * d.inner(), d.d, &rb.qkv),
                out: b.weight(d.d, d.inner(), &rb.out),
                fc1: b.weight(d.mlp, d.d, &rb.fc1),
                fc2: b.weight(d.d, d.mlp, &rb.fc2),
            }
        },
    ))
}

/// Prove one predictor step as a commitment-bound [`PredictorArtifact`]: run the
/// exact integer reference over the circuit and bind the model / quantization /
/// input / output commitments into the artifact's public input. If the circuit
/// carries an export-computed weights root, that carried value (not a recomputed
/// one) is bound into the model commitment, chaining the proof to the export.
pub fn prove(c: &PredictorCircuit) -> Result<PredictorArtifact, ProveError> {
    let out_binding = OutputBinding {
        tensor_id: OUT_TENSOR_ID,
        scale_id: SCALE_OUT,
    };
    match c.export_weights_root {
        Some(root) => prove_predictor_with_weights_root(
            &c.block,
            &c.weights,
            &c.tables,
            &c.scales,
            &c.inputs,
            out_binding,
            root,
        ),
        None => prove_predictor(
            &c.block,
            &c.weights,
            &c.tables,
            &c.scales,
            &c.inputs,
            out_binding,
        ),
    }
}

/// Audit a predictor artifact: recompute and check the model / quantization /
/// planner / input / output commitments, then run the full arithmetic audit
/// (Freivalds linear checks + exact recompute + wiring). Returns the verified
/// claimed output values.
pub fn verify(artifact: &PredictorArtifact) -> Result<Vec<i64>, VerifyError> {
    verify_predictor(artifact)?;
    Ok(artifact
        .claimed_output
        .data()
        .iter()
        .map(BoundedInt::value)
        .collect())
}

/// Forge the first attention/FFN projection output inside the artifact's proven
/// block; the Freivalds check rejects it (the commitments still match, because the
/// architecture commitment covers the ops with outputs cleared).
pub fn tamper(artifact: &mut PredictorArtifact) -> Option<u32> {
    for op in &mut artifact.block.ops {
        if let BlockOp::BatchedLinear { op_id, out, .. } = op {
            if let Some(first) = out.first_mut() {
                *first += 1;
            }
            return Some(*op_id);
        }
    }
    None
}

/// Dequantize the verified integer output at the scheme's output frac and return
/// the max absolute error against the float reference.
pub fn max_float_error(z_int: &[i64], f_out: i32, z_float: &[f64]) -> f64 {
    let scale = 2f64.powi(-f_out);
    z_int
        .iter()
        .zip(z_float.iter())
        .map(|(&q, &z)| (q as f64 * scale - z).abs())
        .fold(0.0, f64::max)
}

/// The bundle's quantization section: the calibrated scheme, the committed
/// tables generated by the export at those scales, the float reference output,
/// and the export-measured error tolerance.
pub struct BundleQuant {
    /// The calibrated fixed-point scheme.
    pub scheme: QuantScheme,
    /// The committed activation tables (SiLU/GELU/inverse-sqrt/exp).
    pub tables: Vec<ActivationTable>,
    /// The float reference output `[s, d]` on the bundle's inputs.
    pub z_out_float: Vec<f64>,
    /// Max-abs `|int·2^-f_ln − float|` tolerance the export measured.
    pub tolerance: f64,
}

/// A loaded predictor bundle: dims, per-block real weights, the quantized inputs,
/// the calibrated quantization section, a description of where the inputs came
/// from, and (for export-bound bundles) the predictor-scoped weight commitment
/// the export computed.
pub struct RealPredictor {
    /// Predictor dimensions.
    pub dims: Dims,
    /// Per-block real quantized weights.
    pub blocks: Vec<RealBlock>,
    /// Quantized latent history input (at `quant.scheme.f_x`).
    pub x: Vec<i64>,
    /// Quantized action embedding input (at `quant.scheme.f_c`).
    pub c: Vec<i64>,
    /// Provenance of the inputs (real observation vs synthetic).
    pub input_source: String,
    /// The export-computed `weights_root` over the proven block tensors (in the
    /// prover's canonical `tensor_id` order), if the bundle carries one.
    pub weights_root: Option<[u8; 32]>,
    /// The calibrated quantization section.
    pub quant: BundleQuant,
}

/// Build the predictor circuit from a loaded bundle: instantiate the graph under
/// the bundle's calibrated scheme and committed tables, and carry the bundle's
/// export-computed weights root (if present) into the circuit so [`prove`] binds
/// it into the model commitment.
pub fn build_predictor_bundle(r: RealPredictor) -> Result<PredictorCircuit, BundleError> {
    let mut c = build_predictor_real(r.dims, r.blocks, r.x, r.c, r.quant.scheme, r.quant.tables)?;
    c.export_weights_root = r.weights_root;
    c.z_float = Some(r.quant.z_out_float);
    c.tolerance = Some(r.quant.tolerance);
    Ok(c)
}

fn load_quant(v: &serde_json::Value, depth: usize) -> Result<BundleQuant, BundleError> {
    let q = bundle::field(v, "quant")?;
    let w_log2_rows = bundle::field(q, "w_log2")?
        .as_array()
        .ok_or(BundleError::WrongType {
            field: "w_log2",
            expected: "an array",
        })?;
    let mut w_log2 = Vec::with_capacity(depth);
    for row in w_log2_rows {
        let vals: Vec<i64> = row
            .as_array()
            .ok_or(BundleError::WrongType {
                field: "w_log2",
                expected: "an array of 5-element arrays",
            })?
            .iter()
            .map(|x| {
                x.as_i64().ok_or(BundleError::WrongType {
                    field: "w_log2",
                    expected: "integer log2 entries",
                })
            })
            .collect::<Result<_, _>>()?;
        let five: [i64; 5] = vals.try_into().map_err(|_| BundleError::WrongType {
            field: "w_log2",
            expected: "5 entries per block (adaln,qkv,out,fc1,fc2)",
        })?;
        w_log2.push(five.map(|x| x as i32));
    }
    let tables = bundle::field(q, "tables")?
        .as_array()
        .ok_or(BundleError::WrongType {
            field: "tables",
            expected: "an array",
        })?
        .iter()
        .map(|t| {
            Ok(ActivationTable {
                table_id: bundle::u64_at(t, "table_id")? as u32,
                lo: bundle::i64_at(t, "lo")?,
                outputs: bundle::ints_at(t, "outputs")?,
            })
        })
        .collect::<Result<Vec<_>, BundleError>>()?;
    let scheme = QuantScheme {
        f_x: bundle::i32_at(q, "f_x")?,
        f_c: bundle::i32_at(q, "f_c")?,
        f_ln: bundle::i32_at(q, "f_ln")?,
        f_qkv: bundle::i32_at(q, "f_qkv")?,
        f_score: bundle::i32_at(q, "f_score")?,
        f_p: bundle::i32_at(q, "f_p")?,
        f_att: bundle::i32_at(q, "f_att")?,
        f_a: bundle::i32_at(q, "f_a")?,
        f_g1: bundle::i32_at(q, "f_g1")?,
        f_g2: bundle::i32_at(q, "f_g2")?,
        f_f: bundle::i32_at(q, "f_f")?,
        t_inv: bundle::i32_at(q, "t_inv")?,
        f_e: bundle::i32_at(q, "f_e")?,
        w_log2,
    };
    Ok(BundleQuant {
        scheme,
        tables,
        z_out_float: bundle::floats_at(q, "z_out_float")?,
        tolerance: bundle::f64_at(q, "tolerance")?,
    })
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
        weights_root: bundle::hex32_at_opt(&v, "weights_root")?,
        quant: load_quant(&v, dims.depth)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pwm_core::audit::output_tensor;
    use pwm_core::commit::weights_root;
    use pwm_verifier::CommitmentKind;

    fn small() -> Dims {
        Dims {
            d: 8,
            s: 3,
            h: 2,
            dh: 4,
            mlp: 16,
            depth: 1,
        }
    }

    fn run(d: Dims) {
        let c = build_predictor(d);
        let artifact = prove(&c).expect("prove");
        let out = verify(&artifact).expect("verify");
        // The proven integer output approximates the float reference within the
        // pinned tolerance (the float-faithfulness gate).
        let err = max_float_error(&out, c.scheme.f_ln, c.z_float.as_ref().expect("z_float"));
        assert!(
            err <= c.tolerance.expect("tolerance"),
            "float-vs-int error {err} exceeds tolerance"
        );
    }

    #[test]
    fn small_dims_one_block_verifies() {
        run(small());
    }

    #[test]
    fn small_dims_six_blocks_verify_within_float_tolerance() {
        run(Dims {
            d: 8,
            s: 3,
            h: 2,
            dh: 4,
            mlp: 16,
            depth: 6,
        });
    }

    #[test]
    fn tables_are_calibrated_not_placeholders() {
        let c = build_predictor(small());
        assert_eq!(c.tables.len(), 4);
        let by_id = |id: u32| c.tables.iter().find(|t| t.table_id == id).unwrap();
        // No identity ramps, no constant inverse-sqrt, no linear exp.
        assert_ne!(
            by_id(TABLE_SILU).outputs,
            (-128i64..=127).collect::<Vec<_>>()
        );
        assert_ne!(
            by_id(TABLE_GELU).outputs,
            (-128i64..=127).collect::<Vec<_>>()
        );
        let inv = by_id(TABLE_INVSQRT);
        assert!(inv.outputs.iter().any(|&v| v != 1));
        assert!(inv.eval(1).unwrap() > inv.eval(4).unwrap());
        let exp = by_id(TABLE_EXP);
        assert_eq!(exp.eval(0), Some(1 << c.scheme.f_e));
        assert!(exp.eval(-1).unwrap() < exp.eval(0).unwrap());
    }

    #[test]
    fn tampered_predictor_is_rejected() {
        let c = build_predictor(small());
        let mut artifact = prove(&c).expect("prove");
        let op = tamper(&mut artifact).expect("a linear op");
        assert!(matches!(
            verify(&artifact),
            Err(VerifyError::FreivaldsCheckFailed { op_id }) if op_id == op
        ));
    }

    #[test]
    fn swapped_weight_is_a_model_commitment_mismatch() {
        let c = build_predictor(small());
        let mut artifact = prove(&c).expect("prove");
        let original = &artifact.weights[0];
        let mut data = original.data().to_vec();
        data[0] = BoundedInt::new(data[0].value() ^ 1, -128, 127).expect("int8");
        artifact.weights[0] = Tensor::new(
            original.tensor_id(),
            original.shape().to_vec(),
            original.scale_id(),
            data,
        )
        .expect("tensor");
        assert!(matches!(
            verify(&artifact),
            Err(VerifyError::CommitmentMismatch(CommitmentKind::Model))
        ));
    }

    #[test]
    fn swapped_table_is_a_quantization_commitment_mismatch() {
        let c = build_predictor(small());
        let mut artifact = prove(&c).expect("prove");
        artifact.tables[0].outputs[0] += 1;
        assert!(matches!(
            verify(&artifact),
            Err(VerifyError::CommitmentMismatch(
                CommitmentKind::Quantization
            ))
        ));
    }

    #[test]
    fn swapped_scale_is_a_quantization_commitment_mismatch() {
        let c = build_predictor(small());
        let mut artifact = prove(&c).expect("prove");
        artifact.scales[0].log2 += 1;
        assert!(matches!(
            verify(&artifact),
            Err(VerifyError::CommitmentMismatch(
                CommitmentKind::Quantization
            ))
        ));
    }

    #[test]
    fn swapped_input_is_an_input_commitment_mismatch() {
        let c = build_predictor(small());
        let mut artifact = prove(&c).expect("prove");
        artifact.inputs[0].1[0] += 1;
        assert!(matches!(
            verify(&artifact),
            Err(VerifyError::CommitmentMismatch(CommitmentKind::Inputs))
        ));
    }

    #[test]
    fn swapped_output_values_are_an_output_mismatch() {
        let c = build_predictor(small());
        let mut artifact = prove(&c).expect("prove");
        let mut vals: Vec<i64> = artifact
            .claimed_output
            .data()
            .iter()
            .map(BoundedInt::value)
            .collect();
        vals[0] += 1;
        artifact.claimed_output = output_tensor(
            artifact.claimed_output.tensor_id(),
            artifact.claimed_output.scale_id(),
            &vals,
        )
        .expect("output tensor");
        assert!(matches!(
            verify(&artifact),
            Err(VerifyError::OutputMismatch)
        ));
    }

    #[test]
    fn relabeled_output_tensor_is_an_output_commitment_mismatch() {
        let c = build_predictor(small());
        let mut artifact = prove(&c).expect("prove");
        let vals: Vec<i64> = artifact
            .claimed_output
            .data()
            .iter()
            .map(BoundedInt::value)
            .collect();
        // Same values, different tensor identity: the recomputed output
        // commitment no longer matches the public input's.
        artifact.claimed_output = output_tensor(artifact.claimed_output.tensor_id() + 1, 0, &vals)
            .expect("output tensor");
        assert!(matches!(
            verify(&artifact),
            Err(VerifyError::OutputCommitmentMismatch)
        ));
    }

    #[test]
    fn matching_export_root_verifies_and_reproduces_the_model_commitment() {
        let baseline = prove(&build_predictor(small())).expect("prove");
        let mut c = build_predictor(small());
        c.export_weights_root = Some(weights_root(&c.weights));
        let artifact = prove(&c).expect("prove");
        verify(&artifact).expect("verify export-bound predictor");
        // Binding the carried root produced exactly the commitment the prover
        // would compute over its own weights: the chain is bit-for-bit.
        assert_eq!(
            artifact.public_input.model_commitment,
            baseline.public_input.model_commitment
        );
    }

    #[test]
    fn wrong_export_root_is_a_model_commitment_mismatch() {
        let mut c = build_predictor(small());
        let mut root = weights_root(&c.weights);
        root[0] ^= 1;
        c.export_weights_root = Some(root);
        let artifact = prove(&c).expect("prove");
        assert!(matches!(
            verify(&artifact),
            Err(VerifyError::CommitmentMismatch(CommitmentKind::Model))
        ));
    }

    /// A full bundle JSON for the small synthetic model, exactly like the real
    /// export flow produces (quant section, tables, float reference, root).
    fn bundle_json(d: Dims, with_root: bool) -> String {
        let pd = d.export();
        let (fblocks, xf, cf) = synth_float_predictor(pd);
        let (scheme, z_float) = calibrate(pd, &fblocks, &xf, &cf);
        let tables = scheme_tables(&scheme);
        let (qblocks, _) = quantize_weights(&fblocks);
        let xq = quantize_at(&xf, scheme.f_x);
        let cq = quantize_at(&cf, scheme.f_c);
        // Recover the canonical weights/root by building the circuit once.
        let qp = derive(&scheme, pd).expect("derives");
        let mut it = qblocks.clone().into_iter();
        let pre = build_predictor_with(
            d,
            xq.clone(),
            cq.clone(),
            scheme.clone(),
            &qp,
            tables.clone(),
            move |b, d| {
                let qb = it.next().unwrap();
                BlockWeights {
                    adaln: b.weight(6 * d.d, d.d, &qb.adaln),
                    qkv: b.weight(3 * d.inner(), d.d, &qb.qkv),
                    out: b.weight(d.d, d.inner(), &qb.out),
                    fc1: b.weight(d.mlp, d.d, &qb.fc1),
                    fc2: b.weight(d.d, d.mlp, &qb.fc2),
                }
            },
        );
        let root = weights_root(&pre.weights);
        let hex: String = root.iter().map(|b| format!("{b:02x}")).collect();
        let mut obj = serde_json::json!({
            "dims": {"d": d.d, "s": d.s, "h": d.h, "dh": d.dh, "mlp": d.mlp, "depth": d.depth},
            "x": xq,
            "c": cq,
            "input_source": "synthetic calibrated test vectors",
            "blocks": qblocks.iter().map(|qb| serde_json::json!({
                "qkv": qb.qkv, "out": qb.out, "fc1": qb.fc1, "fc2": qb.fc2, "adaln": qb.adaln,
            })).collect::<Vec<_>>(),
            "quant": {
                "f_x": scheme.f_x, "f_c": scheme.f_c, "f_ln": scheme.f_ln,
                "f_qkv": scheme.f_qkv, "f_score": scheme.f_score, "f_p": scheme.f_p,
                "f_att": scheme.f_att, "f_a": scheme.f_a, "f_g1": scheme.f_g1,
                "f_g2": scheme.f_g2, "f_f": scheme.f_f, "t_inv": scheme.t_inv,
                "f_e": scheme.f_e,
                "w_log2": scheme.w_log2,
                "tables": tables.iter().map(|t| serde_json::json!({
                    "table_id": t.table_id, "lo": t.lo, "outputs": t.outputs,
                })).collect::<Vec<_>>(),
                "z_out_float": z_float,
                "tolerance": SYNTH_TOLERANCE,
            },
        });
        if with_root {
            obj["weights_root"] = serde_json::Value::String(hex);
        }
        obj.to_string()
    }

    #[test]
    fn export_bound_bundle_json_round_trips_and_verifies_float_faithfully() {
        let d = small();
        let json = bundle_json(d, true);
        let loaded = load_real_predictor(&json).expect("load");
        assert!(loaded.weights_root.is_some());
        let circuit = build_predictor_bundle(loaded).expect("build");
        let f_out = circuit.scheme.f_ln;
        let z_float = circuit.z_float.clone().expect("z_float");
        let tol = circuit.tolerance.expect("tolerance");
        let artifact = prove(&circuit).expect("prove");
        let out = verify(&artifact).expect("verify export-bound bundle");
        let err = max_float_error(&out, f_out, &z_float);
        assert!(err <= tol, "bundle float error {err} > {tol}");
    }

    #[test]
    fn bundle_without_root_is_not_export_bound_but_verifies() {
        let json = bundle_json(small(), false);
        let loaded = load_real_predictor(&json).expect("load");
        assert!(loaded.weights_root.is_none());
        let circuit = build_predictor_bundle(loaded).expect("build");
        let artifact = prove(&circuit).expect("prove");
        verify(&artifact).expect("verify");
    }

    #[test]
    fn malformed_bundle_weights_root_is_a_typed_error() {
        let base = bundle_json(small(), false);
        // Too short, non-hex, and non-string all reject with a typed error.
        for bad in [
            r#""abcd""#.to_string(),
            format!(r#""{}""#, "zz".repeat(32)),
            "7".to_string(),
        ] {
            let json = base.replacen('{', &format!(r#"{{"weights_root": {bad},"#), 1);
            assert!(matches!(
                load_real_predictor(&json),
                Err(BundleError::WrongType {
                    field: "weights_root",
                    ..
                })
            ));
        }
    }

    #[test]
    fn unsound_bundle_scheme_is_a_typed_error() {
        let json = bundle_json(small(), false);
        let mut loaded = load_real_predictor(&json).expect("load");
        // Force a negative qkv shift: f_qkv far above f_ln + f_w.
        loaded.quant.scheme.f_qkv = 60;
        assert!(matches!(
            build_predictor_bundle(loaded),
            Err(BundleError::UnsoundScheme { site: "qkv", .. })
        ));
    }

    // Cross-language pin of the predictor weight scheme (ids 1000+5i in
    // registration order adaln,qkv,out,fc1,fc2; per-tensor scale ids
    // SCALE_W_BASE+k; int8 bounds): the Python exporter's
    // predictor_weight_dicts must reproduce this exact root for the same dims
    // and deterministic weight pattern
    // (test_canonical_parity.py::test_predictor_weight_scheme_matches_rust).
    #[test]
    fn predictor_weight_scheme_parity_vector_is_pinned() {
        let d = Dims {
            d: 8,
            s: 3,
            h: 2,
            dh: 4,
            mlp: 16,
            depth: 2,
        };
        let inner = d.inner();
        let mk = |rows: usize, cols: usize| -> Vec<i64> {
            (0..rows * cols).map(|i| (i as i64 % 3) - 1).collect()
        };
        let mut b = Builder::new();
        for _ in 0..d.depth {
            b.weight(6 * d.d, d.d, &mk(6 * d.d, d.d));
            b.weight(3 * inner, d.d, &mk(3 * inner, d.d));
            b.weight(d.d, inner, &mk(d.d, inner));
            b.weight(d.mlp, d.d, &mk(d.mlp, d.d));
            b.weight(d.d, d.mlp, &mk(d.d, d.mlp));
        }
        let hex: String = weights_root(&b.weights)
            .iter()
            .map(|x| format!("{x:02x}"))
            .collect();
        assert_eq!(
            hex,
            "b935eb9908da38f4f23f6ff1db8922a87b79f2c5686717726ebb30920803a95b"
        );
    }

    // The real le-wm V0 dims (192/3/16/64/2048, depth 6, ~2.4k ops). Heavy in a
    // debug build, so it is `#[ignore]`d for the default `cargo test`; the CI test
    // job runs it in release via `cargo test --release -- --ignored` (~1 s there),
    // so the headline "the full 192-dim predictor verifies float-faithfully"
    // claim is gated.
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
