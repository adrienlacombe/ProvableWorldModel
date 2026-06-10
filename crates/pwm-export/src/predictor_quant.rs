// SPDX-License-Identifier: Apache-2.0
//! Float-faithful predictor quantization: calibration, scales, and tables.
//!
//! This module turns the float le-wm conditional predictor into the global
//! fixed-point scheme the integer circuit proves: it runs the float reference
//! forward pass (mirroring the prover's block DAG op for op), records per-site
//! absolute maxima, derives a [`QuantScheme`] of power-of-two fractional bits,
//! and generates the four committed activation tables (SiLU, GELU, inverse-sqrt,
//! softmax exp) from the real functions at the calibrated scales via
//! [`crate::tables_gen`].
//!
//! Scheme invariants the derivation enforces (each per-op requant shift must be
//! a nonnegative power-of-two rescale):
//!
//! - every LayerNorm output, and the AdaLN conditioning chunks, share one frac
//!   `f_ln`, so `Modulate`'s `x·(1+scale)+shift` is scale-consistent with
//!   `one = 2^f_ln` and `shift_bits = f_ln`;
//! - the inverse-sqrt table is **scale-invariant**: with `table(v) = 2^t_inv/√v`
//!   over the raw integer variance, `(x_q−mean_q)·table(var_q)` lands at frac
//!   `t_inv` regardless of the input frac (the input's `2^f` cancels through
//!   `√(2^{2f})`), so one table serves every LayerNorm site;
//! - attention's `1/√dh` folds exactly into the score requant shift, which
//!   requires `log2(dh)` to be even (true for the V0 `dh = 64` and the small-dims
//!   test `dh = 4`);
//! - the residual stream keeps one frac `f_x` end to end (residual requants are
//!   pure int8 re-clamps), so additions never mix scales.

use pwm_core::tables::ActivationTable;

use crate::tables_gen;

/// LayerNorm epsilon of the float reference (folded into the centered/variance
/// integer recipe only through calibration; the integer table is `1/√var` with
/// `table[0]` defined as `2^t_inv`, which is exact because a zero variance means
/// every centered value is zero).
pub const LN_EPS: f64 = 1e-5;

/// Committed table ids (shared with the testkit predictor builder).
pub const TABLE_SILU: u32 = 1;
/// GELU table id.
pub const TABLE_GELU: u32 = 2;
/// Inverse-sqrt (LayerNorm) table id.
pub const TABLE_INVSQRT: u32 = 3;
/// Softmax exp table id.
pub const TABLE_EXP: u32 = 4;

/// The largest integer variance a LayerNorm input can produce: inputs are either
/// int8 (|x| ≤ 127) or a Modulate output (|x| ≤ 255), so a centered value is at
/// most 510 in magnitude and the mean of squares at most 510².
pub const INVSQRT_DOMAIN_HI: i64 = 510 * 510;

/// Predictor dimensions (mirrors the testkit `Dims`; kept separate so the
/// dependency points testkit → export only).
#[derive(Clone, Copy, Debug)]
pub struct PredictorDims {
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

impl PredictorDims {
    /// `heads * dim_head` (the attention inner width).
    pub fn inner(&self) -> usize {
        self.h * self.dh
    }
}

/// One block's float weights, row-major, in the le-wm shapes the builder expects.
#[derive(Clone, Debug)]
pub struct FloatBlock {
    /// Fused QKV `[3*inner, d]`.
    pub qkv: Vec<f64>,
    /// Attention out-projection `[d, inner]`.
    pub out: Vec<f64>,
    /// FFN fc1 `[mlp, d]`.
    pub fc1: Vec<f64>,
    /// FFN fc2 `[d, mlp]`.
    pub fc2: Vec<f64>,
    /// AdaLN modulation `[6*d, d]`.
    pub adaln: Vec<f64>,
}

/// One block's quantized int8 weights (same shapes/order as [`FloatBlock`]).
#[derive(Clone, Debug)]
pub struct QuantBlock {
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

/// Per-block weight `log2` scales in registration order
/// `[adaln, qkv, out, fc1, fc2]` (a quantized weight `q` denotes `q · 2^log2`).
pub type BlockWeightLog2 = [i32; 5];

/// The global fixed-point scheme: fractional bits per site class (a buffer at
/// frac `f` stores `q = round(real · 2^f)`), the table output scales, and the
/// per-tensor weight scales. Derived by [`calibrate`]; every per-op requant
/// shift is a function of these (see [`derive()`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuantScheme {
    /// Residual-stream frac (the latent history input and every block output).
    pub f_x: i32,
    /// Action-embedding frac (the conditioning input and the SiLU output).
    pub f_c: i32,
    /// Frac shared by every LayerNorm output and the AdaLN chunks.
    pub f_ln: i32,
    /// Frac of the q/k/v projections.
    pub f_qkv: i32,
    /// Frac of the attention scores (after the folded `1/√dh`).
    pub f_score: i32,
    /// Softmax probability frac: `one = 2^f_p`.
    pub f_p: i32,
    /// Frac of the per-head `prob·V` values.
    pub f_att: i32,
    /// Frac of the attention out-projection (the gated sublayer output).
    pub f_a: i32,
    /// Frac of the pre-GELU fc1 output.
    pub f_g1: i32,
    /// Frac of the GELU output.
    pub f_g2: i32,
    /// Frac of the fc2 output (the gated FFN sublayer output).
    pub f_f: i32,
    /// Inverse-sqrt table output frac (`table(v) = round(2^t_inv / √v)`).
    pub t_inv: i32,
    /// Softmax exp table output frac (cancels in the `e·one/Σe` ratio; only
    /// precision).
    pub f_e: i32,
    /// Per-block weight `log2`s in registration order `[adaln, qkv, out, fc1, fc2]`.
    pub w_log2: Vec<BlockWeightLog2>,
}

/// A scheme violation: some site needs an upscale (negative requant shift),
/// which the integer circuit cannot express.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuantError {
    /// The site whose derived shift was negative.
    pub site: &'static str,
    /// The (negative) shift value.
    pub shift: i64,
}

impl core::fmt::Display for QuantError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "quant scheme needs a negative requant shift at {}: {}",
            self.site, self.shift
        )
    }
}

impl std::error::Error for QuantError {}

/// Per-block requant shifts for the weight-bearing linears, in registration
/// order semantics: `adaln`, `qkv`, `out` (attention projection), `fc1`, `fc2`.
#[derive(Clone, Copy, Debug)]
pub struct BlockShifts {
    /// AdaLN linear requant: `f_c + f_w(adaln) − f_ln`.
    pub adaln: u32,
    /// QKV linear requant: `f_ln + f_w(qkv) − f_qkv`.
    pub qkv: u32,
    /// Out-projection requant: `f_att + f_w(out) − f_a`.
    pub out: u32,
    /// fc1 requant: `f_ln + f_w(fc1) − f_g1`.
    pub fc1: u32,
    /// fc2 requant: `f_g2 + f_w(fc2) − f_f`.
    pub fc2: u32,
}

/// Every per-op parameter the circuit builder needs, derived from a
/// [`QuantScheme`] (all shifts validated nonnegative).
#[derive(Clone, Debug)]
pub struct DerivedParams {
    /// LayerNorm requant shift (`t_inv − f_ln`), every LN site.
    pub ln_shift: u32,
    /// Attention score requant shift (`2·f_qkv + log2(dh)/2 − f_score`).
    pub score_shift: u32,
    /// Per-head `prob·V` requant shift (`f_p + f_qkv − f_att`).
    pub oh_shift: u32,
    /// Gate shift for the attention residual (`f_ln + f_a − f_x`).
    pub gate_msa_bits: u32,
    /// Gate shift for the FFN residual (`f_ln + f_f − f_x`).
    pub gate_mlp_bits: u32,
    /// Modulate fixed-point unit (`2^f_ln`).
    pub mod_one: i64,
    /// Modulate rescale (`f_ln`).
    pub mod_bits: u32,
    /// Softmax integer unit (`2^f_p`).
    pub softmax_one: i64,
    /// Per-block weight-linear shifts.
    pub blocks: Vec<BlockShifts>,
}

fn nonneg(site: &'static str, v: i64) -> Result<u32, QuantError> {
    u32::try_from(v).map_err(|_| QuantError { site, shift: v })
}

/// Derive every per-op parameter from the scheme; rejects any site that would
/// need a negative shift. `log2(dh)` must be even so `1/√dh` is a power of two.
pub fn derive(scheme: &QuantScheme, dims: PredictorDims) -> Result<DerivedParams, QuantError> {
    assert!(
        dims.dh.is_power_of_two() && dims.dh.trailing_zeros() % 2 == 0,
        "dh must be a power of four so 1/sqrt(dh) folds into the score shift"
    );
    let half_log2_dh = i64::from(dims.dh.trailing_zeros() / 2);
    let s = scheme;
    let fw = |b: usize, i: usize| -i64::from(s.w_log2[b][i]);
    let mut blocks = Vec::with_capacity(s.w_log2.len());
    for b in 0..s.w_log2.len() {
        blocks.push(BlockShifts {
            adaln: nonneg("adaln", i64::from(s.f_c) + fw(b, 0) - i64::from(s.f_ln))?,
            qkv: nonneg("qkv", i64::from(s.f_ln) + fw(b, 1) - i64::from(s.f_qkv))?,
            out: nonneg("out", i64::from(s.f_att) + fw(b, 2) - i64::from(s.f_a))?,
            fc1: nonneg("fc1", i64::from(s.f_ln) + fw(b, 3) - i64::from(s.f_g1))?,
            fc2: nonneg("fc2", i64::from(s.f_g2) + fw(b, 4) - i64::from(s.f_f))?,
        });
    }
    Ok(DerivedParams {
        ln_shift: nonneg("layernorm", i64::from(s.t_inv) - i64::from(s.f_ln))?,
        score_shift: nonneg(
            "score",
            2 * i64::from(s.f_qkv) + half_log2_dh - i64::from(s.f_score),
        )?,
        oh_shift: nonneg(
            "prob_v",
            i64::from(s.f_p) + i64::from(s.f_qkv) - i64::from(s.f_att),
        )?,
        gate_msa_bits: nonneg(
            "gate_msa",
            i64::from(s.f_ln) + i64::from(s.f_a) - i64::from(s.f_x),
        )?,
        gate_mlp_bits: nonneg(
            "gate_mlp",
            i64::from(s.f_ln) + i64::from(s.f_f) - i64::from(s.f_x),
        )?,
        mod_one: 1i64 << s.f_ln,
        mod_bits: nonneg("modulate", i64::from(s.f_ln))?,
        softmax_one: 1i64 << s.f_p,
        blocks,
    })
}

/// Generate the four committed tables at the calibrated scales: SiLU over int8
/// at `f_c`, GELU over int8 `f_g1 → f_g2`, the scale-invariant inverse-sqrt over
/// the raw integer variance domain, and softmax exp over the shifted-score
/// domain `[-255, 0]` at `f_score → f_e`.
pub fn scheme_tables(s: &QuantScheme) -> Vec<ActivationTable> {
    vec![
        tables_gen::silu_table(TABLE_SILU, -128, 127, s.f_c, s.f_c),
        tables_gen::gelu_table(TABLE_GELU, -128, 127, s.f_g1, s.f_g2),
        inv_sqrt_norm_table(TABLE_INVSQRT, INVSQRT_DOMAIN_HI, s.t_inv),
        tables_gen::exp_table(TABLE_EXP, -255, s.f_score, s.f_e),
    ]
}

/// The scale-invariant LayerNorm inverse-sqrt table over the **raw integer
/// variance**: `table(v) = round(2^t_inv / √v)` for `v ≥ 1`, and
/// `table(0) = 2^t_inv` (exact: zero variance means every centered value is
/// zero, so the multiplier is irrelevant). Because `√(2^{2f})` is exactly `2^f`,
/// `(x_q − mean_q) · table(var_q)` lands at frac `t_inv` for *any* input frac —
/// one table serves every LayerNorm site.
pub fn inv_sqrt_norm_table(table_id: u32, hi: i64, t_inv: i32) -> ActivationTable {
    let out_scale = 2f64.powi(t_inv);
    let outputs = (0..=hi)
        .map(|q| {
            if q == 0 {
                out_scale.round() as i64
            } else {
                (out_scale / (q as f64).sqrt()).round() as i64
            }
        })
        .collect();
    ActivationTable {
        table_id,
        lo: 0,
        outputs,
    }
}

/// Quantize floats at frac `f` (`q = round(v · 2^f)`), clamped to int8.
pub fn quantize_at(vals: &[f64], f: i32) -> Vec<i64> {
    let scale = 2f64.powi(f);
    vals.iter()
        .map(|&v| ((v * scale).round() as i64).clamp(-128, 127))
        .collect()
}

fn absmax(vals: &[f64]) -> f64 {
    vals.iter().fold(0.0, |m, &v| m.max(v.abs()))
}

/// The per-tensor power-of-two weight scale: smallest `log2` with
/// `absmax / 2^log2 ≤ 127` (0 for an all-zero tensor) — the Rust mirror of the
/// Python exporter's `pow2_log2`.
pub fn weight_log2(w: &[f64]) -> i32 {
    let m = absmax(w);
    if m == 0.0 {
        0
    } else {
        (m / 127.0).log2().ceil() as i32
    }
}

fn quantize_weight(w: &[f64], log2: i32) -> Vec<i64> {
    let scale = 2f64.powi(-log2);
    w.iter()
        .map(|&v| ((v * scale).round() as i64).clamp(-128, 127))
        .collect()
}

/// Quantize every block's weights with per-tensor power-of-two scales; returns
/// the int8 blocks and the per-block `log2`s in registration order.
pub fn quantize_weights(blocks: &[FloatBlock]) -> (Vec<QuantBlock>, Vec<BlockWeightLog2>) {
    let mut qblocks = Vec::with_capacity(blocks.len());
    let mut log2s = Vec::with_capacity(blocks.len());
    for b in blocks {
        let l: BlockWeightLog2 = [
            weight_log2(&b.adaln),
            weight_log2(&b.qkv),
            weight_log2(&b.out),
            weight_log2(&b.fc1),
            weight_log2(&b.fc2),
        ];
        qblocks.push(QuantBlock {
            adaln: quantize_weight(&b.adaln, l[0]),
            qkv: quantize_weight(&b.qkv, l[1]),
            out: quantize_weight(&b.out, l[2]),
            fc1: quantize_weight(&b.fc1, l[3]),
            fc2: quantize_weight(&b.fc2, l[4]),
        });
        log2s.push(l);
    }
    (qblocks, log2s)
}

// --- float reference forward (mirrors the builder's block DAG op for op) ---

fn ln_rows(x: &[f64], s: usize, d: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(s * d);
    for p in 0..s {
        let row = &x[p * d..(p + 1) * d];
        let mean = row.iter().sum::<f64>() / d as f64;
        let var = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / d as f64;
        let inv = 1.0 / (var + LN_EPS).sqrt();
        out.extend(row.iter().map(|&v| (v - mean) * inv));
    }
    out
}

/// `out[p, r] = Σ_c W[r, c] · x[p, c]` for `x: [s, cols]`, `W: [rows, cols]`.
fn batched_linear(w: &[f64], x: &[f64], s: usize, rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; s * rows];
    for p in 0..s {
        for r in 0..rows {
            let mut acc = 0.0;
            for c in 0..cols {
                acc += w[r * cols + c] * x[p * cols + c];
            }
            out[p * rows + r] = acc;
        }
    }
    out
}

fn softmax_row(scores: &[f64]) -> Vec<f64> {
    let m = scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f64> = scores.iter().map(|&v| (v - m).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Per-site absolute maxima recorded by the calibration forward pass.
#[derive(Clone, Debug, Default)]
pub struct Calibration {
    /// Residual stream (input and every block output).
    pub x: f64,
    /// Action embedding (and its SiLU image).
    pub c: f64,
    /// LayerNorm outputs (all sites, including the final norm).
    pub ln: f64,
    /// AdaLN chunks (shift/scale/gate, both branches).
    pub adaln: f64,
    /// q/k/v projections.
    pub qkv: f64,
    /// Attention scores after `1/√dh`.
    pub score: f64,
    /// Per-head `prob·V` values.
    pub att: f64,
    /// Attention out-projection.
    pub attn_out: f64,
    /// Pre-GELU fc1 output.
    pub g1: f64,
    /// GELU output.
    pub g2: f64,
    /// fc2 output.
    pub ffn: f64,
    /// Modulate product `n·(1+scale)` before the additive shift (the part the
    /// integer kernel requant-clamps at `f_ln`).
    pub mod_prod: f64,
}

impl Calibration {
    fn see(field: &mut f64, vals: &[f64]) {
        *field = field.max(absmax(vals));
    }
}

/// Run the float reference forward pass, recording per-site maxima. Returns the
/// final normalized output `[s, d]` and the calibration record. The graph is the
/// builder's, op for op: per ConditionalBlock, AdaLN chunks from `SiLU(c)`
/// modulate a pre-norm multi-head attention and a pre-norm GELU FFN, each with a
/// gated residual; a final affine-free LayerNorm closes the stack.
pub fn float_forward(
    dims: PredictorDims,
    blocks: &[FloatBlock],
    x0: &[f64],
    c: &[f64],
) -> (Vec<f64>, Calibration) {
    let (d, s) = (dims.d, dims.s);
    let inner = dims.inner();
    let mut cal = Calibration::default();
    Calibration::see(&mut cal.x, x0);
    Calibration::see(&mut cal.c, c);
    let silu_c: Vec<f64> = c.iter().map(|&v| tables_gen::silu(v)).collect();
    Calibration::see(&mut cal.c, &silu_c);
    let mut x = x0.to_vec();
    for fb in blocks {
        // AdaLN: SiLU(c) -> Linear -> chunk6 (per position).
        let ada = batched_linear(&fb.adaln, &silu_c, s, 6 * d, d);
        Calibration::see(&mut cal.adaln, &ada);
        let chunk = |j: usize| -> Vec<f64> {
            let mut out = Vec::with_capacity(s * d);
            for p in 0..s {
                out.extend_from_slice(&ada[p * 6 * d + j * d..p * 6 * d + (j + 1) * d]);
            }
            out
        };
        let (sh_msa, sc_msa, g_msa) = (chunk(0), chunk(1), chunk(2));
        let (sh_mlp, sc_mlp, g_mlp) = (chunk(3), chunk(4), chunk(5));

        // Attention branch.
        let n1 = ln_rows(&x, s, d);
        Calibration::see(&mut cal.ln, &n1);
        let prod1: Vec<f64> = n1
            .iter()
            .zip(sc_msa.iter())
            .map(|(&n, &sc)| n * (1.0 + sc))
            .collect();
        Calibration::see(&mut cal.mod_prod, &prod1);
        let m1: Vec<f64> = prod1
            .iter()
            .zip(sh_msa.iter())
            .map(|(&p, &sh)| p + sh)
            .collect();
        let a0 = ln_rows(&m1, s, d);
        Calibration::see(&mut cal.ln, &a0);
        let qkv = batched_linear(&fb.qkv, &a0, s, 3 * inner, d);
        Calibration::see(&mut cal.qkv, &qkv);
        let col = |src: &[f64], width: usize, off: usize, len: usize| -> Vec<f64> {
            let mut out = Vec::with_capacity(s * len);
            for p in 0..s {
                out.extend_from_slice(&src[p * width + off..p * width + off + len]);
            }
            out
        };
        let q = col(&qkv, 3 * inner, 0, inner);
        let k = col(&qkv, 3 * inner, inner, inner);
        let v = col(&qkv, 3 * inner, 2 * inner, inner);
        let scale = 1.0 / (dims.dh as f64).sqrt();
        let mut heads = vec![0.0; s * inner];
        for h in 0..dims.h {
            let qh = col(&q, inner, h * dims.dh, dims.dh);
            let kh = col(&k, inner, h * dims.dh, dims.dh);
            let vh = col(&v, inner, h * dims.dh, dims.dh);
            for i in 0..s {
                let mut scores = vec![0.0; s];
                for (j, sc) in scores.iter_mut().enumerate() {
                    let mut acc = 0.0;
                    for t in 0..dims.dh {
                        acc += qh[i * dims.dh + t] * kh[j * dims.dh + t];
                    }
                    *sc = acc * scale;
                }
                Calibration::see(&mut cal.score, &scores);
                let prob = softmax_row(&scores);
                for t in 0..dims.dh {
                    let mut acc = 0.0;
                    for (j, &p) in prob.iter().enumerate() {
                        acc += p * vh[j * dims.dh + t];
                    }
                    heads[i * inner + h * dims.dh + t] = acc;
                }
            }
        }
        Calibration::see(&mut cal.att, &heads);
        let a = batched_linear(&fb.out, &heads, s, d, inner);
        Calibration::see(&mut cal.attn_out, &a);
        let gated1: Vec<f64> = g_msa.iter().zip(a.iter()).map(|(&g, &v)| g * v).collect();
        // The gate output is requant-clamped at f_x before the residual add.
        Calibration::see(&mut cal.x, &gated1);
        for i in 0..s * d {
            x[i] += gated1[i];
        }
        Calibration::see(&mut cal.x, &x);

        // FFN branch.
        let n2 = ln_rows(&x, s, d);
        Calibration::see(&mut cal.ln, &n2);
        let prod2: Vec<f64> = n2
            .iter()
            .zip(sc_mlp.iter())
            .map(|(&n, &sc)| n * (1.0 + sc))
            .collect();
        Calibration::see(&mut cal.mod_prod, &prod2);
        let m2: Vec<f64> = prod2
            .iter()
            .zip(sh_mlp.iter())
            .map(|(&p, &sh)| p + sh)
            .collect();
        let f0 = ln_rows(&m2, s, d);
        Calibration::see(&mut cal.ln, &f0);
        let h1 = batched_linear(&fb.fc1, &f0, s, dims.mlp, d);
        Calibration::see(&mut cal.g1, &h1);
        let g: Vec<f64> = h1.iter().map(|&v| tables_gen::gelu(v)).collect();
        Calibration::see(&mut cal.g2, &g);
        let f = batched_linear(&fb.fc2, &g, s, d, dims.mlp);
        Calibration::see(&mut cal.ffn, &f);
        let gated2: Vec<f64> = g_mlp.iter().zip(f.iter()).map(|(&g, &v)| g * v).collect();
        Calibration::see(&mut cal.x, &gated2);
        for i in 0..s * d {
            x[i] += gated2[i];
        }
        Calibration::see(&mut cal.x, &x);
    }
    let z = ln_rows(&x, s, d);
    Calibration::see(&mut cal.ln, &z);
    (z, cal)
}

/// Deterministic pseudo-random floats in `[-a, a]` (a 64-bit LCG; no rand
/// dependency). Used to synthesize float weights/inputs for the demo's
/// synthetic profile and the calibration tests.
pub fn noise(n: usize, a: f64, seed: u64) -> Vec<f64> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = ((state >> 33) as f64) / ((1u64 << 31) as f64); // [0, 2)
            (u - 1.0) * a
        })
        .collect()
}

/// Deterministic synthetic float predictor (weights and inputs) for the given
/// dims: the demo's synthetic profile, with realistic magnitudes so the
/// calibrated tables and shifts exercise the same code paths as a real export.
pub fn synth_float_predictor(dims: PredictorDims) -> (Vec<FloatBlock>, Vec<f64>, Vec<f64>) {
    let inner = dims.inner();
    let fan = |cols: usize| 0.4 / (cols as f64).sqrt();
    let blocks = (0..dims.depth)
        .map(|i| {
            let s = (i as u64 + 1) * 7919;
            FloatBlock {
                qkv: noise(3 * inner * dims.d, fan(dims.d), s),
                out: noise(dims.d * inner, fan(inner), s + 1),
                fc1: noise(dims.mlp * dims.d, fan(dims.d), s + 2),
                fc2: noise(dims.d * dims.mlp, fan(dims.mlp), s + 3),
                adaln: noise(6 * dims.d * dims.d, fan(dims.d), s + 4),
            }
        })
        .collect();
    let x = noise(dims.s * dims.d, 1.5, 99);
    let c = noise(dims.s * dims.d, 1.5, 100);
    (blocks, x, c)
}

/// The largest frac `f` with `absmax · 2^f ≤ margin` (int8 headroom; `margin`
/// slightly under 127 leaves room for rounding). Capped at 12 so near-zero
/// sites cannot blow the downstream requant shifts past the `MAX_SHIFT` guard.
fn frac_for(absmax: f64, margin: f64) -> i32 {
    if absmax <= 0.0 {
        7
    } else {
        ((margin / absmax).log2().floor() as i32).min(12)
    }
}

/// Calibrate the global scheme from the float weights and the actual inputs:
/// run the float forward, derive per-site fracs from the recorded maxima, and
/// pick the table precisions. Returns the scheme and the float reference output
/// (the value the quantized circuit must approximate).
pub fn calibrate(
    dims: PredictorDims,
    blocks: &[FloatBlock],
    x0: &[f64],
    c: &[f64],
) -> (QuantScheme, Vec<f64>) {
    let (z, cal) = float_forward(dims, blocks, x0, c);
    let (_, w_log2) = quantize_weights(blocks);
    const M: f64 = 126.0;
    let scheme = QuantScheme {
        f_x: frac_for(cal.x, M),
        f_c: frac_for(cal.c, M),
        // One frac for every LN output, the AdaLN chunks, and the requant-clamped
        // modulate product: take the tightest so none of those sites clips.
        f_ln: frac_for(cal.ln.max(cal.adaln).max(cal.mod_prod), M),
        f_qkv: frac_for(cal.qkv, M),
        f_score: frac_for(cal.score, M),
        f_p: 10,
        f_att: frac_for(cal.att, M),
        f_a: frac_for(cal.attn_out, M),
        f_g1: frac_for(cal.g1, M),
        f_g2: frac_for(cal.g2, M),
        f_f: frac_for(cal.ffn, M),
        t_inv: 16,
        f_e: 15,
        w_log2,
    };
    (scheme, z)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims() -> PredictorDims {
        PredictorDims {
            d: 8,
            s: 3,
            h: 2,
            dh: 4,
            mlp: 16,
            depth: 2,
        }
    }

    #[test]
    fn calibrated_scheme_derives_with_nonnegative_shifts() {
        let d = dims();
        let (blocks, x, c) = synth_float_predictor(d);
        let (scheme, z) = calibrate(d, &blocks, &x, &c);
        assert_eq!(z.len(), d.s * d.d);
        let p = derive(&scheme, d).expect("derivable scheme");
        assert_eq!(p.blocks.len(), d.depth);
        assert_eq!(p.mod_one, 1i64 << scheme.f_ln);
        assert_eq!(p.softmax_one, 1i64 << scheme.f_p);
    }

    #[test]
    fn tables_are_real_functions_not_placeholders() {
        let d = dims();
        let (blocks, x, c) = synth_float_predictor(d);
        let (scheme, _) = calibrate(d, &blocks, &x, &c);
        let tables = scheme_tables(&scheme);
        assert_eq!(tables.len(), 4);
        // SiLU/GELU fix 0 and are not the identity ramp.
        let silu = &tables[0];
        let gelu = &tables[1];
        assert_eq!(silu.eval(0), Some(0));
        assert_eq!(gelu.eval(0), Some(0));
        assert_ne!(silu.outputs, (-128i64..=127).collect::<Vec<_>>());
        assert_ne!(gelu.outputs, (-128i64..=127).collect::<Vec<_>>());
        // Inverse-sqrt is strictly decreasing over v >= 1 (until rounding floors)
        // and scale-invariant by construction: table(4v) = table(v)/2 exactly in
        // the reals; spot-check the rounded relation.
        let inv = &tables[2];
        assert_eq!(inv.eval(0), Some(1 << scheme.t_inv));
        assert_eq!(inv.eval(1), Some(1 << scheme.t_inv));
        assert_eq!(inv.eval(4), Some(1 << (scheme.t_inv - 1)));
        assert!(inv.eval(2).unwrap() > inv.eval(3).unwrap());
        // Exp is monotonically increasing up to e^0 = 2^f_e.
        let exp = &tables[3];
        assert_eq!(exp.eval(0), Some(1 << scheme.f_e));
        assert!(exp.eval(-1).unwrap() < exp.eval(0).unwrap());
        assert!(exp.eval(-255).unwrap() >= 0);
    }

    #[test]
    fn layernorm_table_is_scale_invariant_through_the_kernel() {
        // The same inv-sqrt table normalizes the same real vector quantized at
        // two different fracs to the same real output (up to quantization).
        use pwm_core::fixed_point::Rounding;
        use pwm_core::predictor::layernorm;
        let t = inv_sqrt_norm_table(TABLE_INVSQRT, INVSQRT_DOMAIN_HI, 16);
        let real = [1.5f64, -0.5, 2.0, -3.0, 0.25, 1.0, -1.25, 0.5];
        let f_out = 5i32;
        let shift = (16 - f_out) as u32;
        let mut outs = Vec::new();
        for f in [4i32, 6] {
            let q: Vec<i64> = real
                .iter()
                .map(|&v| (v * 2f64.powi(f)).round() as i64)
                .collect();
            let out = layernorm(&q, &t, shift, -128, 127, Rounding::NearestTiesToEven)
                .expect("in domain");
            outs.push(out);
        }
        for (a, b) in outs[0].iter().zip(outs[1].iter()) {
            assert!((a - b).abs() <= 1, "scale-invariance within one quantum");
        }
    }
}
