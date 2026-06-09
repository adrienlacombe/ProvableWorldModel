// SPDX-License-Identifier: Apache-2.0
//! `pwm`: the prover/verifier demo CLI for the commit-and-audit scheme.
//!
//! It proves a real, action-conditioned world-model predictor step (the le-wm
//! predictor architecture: self-attention plus a GELU feed-forward with residuals,
//! conditioned on an action, over a latent history) run in exact integer
//! arithmetic, then audits it. No floating point, no GPU, no external assets.
//!
//! ```text
//! pwm demo              # the whole story: infer -> prove -> challenge -> accept -> tamper -> reject
//! pwm prove  <file>     # run the predictor, write the proof to <file>
//! pwm verify <file>     # decode a proof and verify it (accept or reject)
//! pwm audit  <file>     # the verifier's story: challenge + accept, then tamper -> reject
//! pwm tamper <in> <out> # forge one matmul output (the canonical Freivalds-caught lie)
//! pwm prove-lewm <json> # prove the REAL le-wm pred_proj head from an export bundle
//! pwm prove-predictor [bundle]  # prove the full 6-block 16-head predictor (real or synthetic)
//! ```
//!
//! Add `--json` for machine-readable output. `docker compose up` runs
//! `prove-predictor` (the real architecture); `--profile real` proves the real
//! pretrained checkpoint.

use std::env;
use std::fs;
use std::process::exit;
use std::time::Instant;

use pwm_core::block::{block_architecture_commitment, block_root, BlockOp};
use pwm_core::commit::weights_root;
use pwm_core::serialize::canonical_bytes;
use pwm_core::tensor::Tensor;
use pwm_testkit::lewm_predictor::{self, Dims};
use pwm_testkit::predictor::{
    self, PredictorProof, ACTION_DIM, DIM, MLP, SEQ, V0_DEPTH, V0_DIM, V0_HEADS,
};
use pwm_testkit::{demo, lewm};
use pwm_verifier::verify as verify_artifact;
use serde_json::json;

// --- ANSI helpers (respect NO_COLOR) ---
fn color() -> bool {
    env::var_os("NO_COLOR").is_none()
}
fn paint(code: &str, s: &str) -> String {
    if color() {
        format!("\x1b[{code}m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}
fn ok(s: &str) -> String {
    paint("32;1", s)
}
fn bad(s: &str) -> String {
    paint("31;1", s)
}
fn dim(s: &str) -> String {
    paint("2", s)
}
fn tag(role: &str) -> String {
    paint("36;1", &format!("[{role}]"))
}
fn ms(d: std::time::Duration) -> String {
    format!("{:.3} ms", d.as_secs_f64() * 1000.0)
}
fn hex8(root: &[u8; 32]) -> String {
    root.iter()
        .take(4)
        .map(|b| format!("{b:02x}"))
        .collect::<String>()
        + "..."
}
// --- pipeline visuals (box-drawing tree, respects NO_COLOR) ---
fn commas(n: usize) -> String {
    let s = n.to_string();
    let b = s.as_bytes();
    let mut out = String::new();
    for (i, c) in b.iter().enumerate() {
        if i > 0 && (b.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(*c as char);
    }
    out
}
fn stage(n: u32, total: u32, name: &str, sub: &str) {
    println!(
        "\n{} {}  {}",
        paint("35;1", &format!("[stage {n}/{total}]")),
        paint("1", name),
        dim(sub)
    );
}
// tree-branch connector: mid (├) for inner rows, end (└) for the last
fn li(last: bool) -> String {
    dim(if last { "  \u{2514}" } else { "  \u{251c}" })
}
// continuation line under a branch (│)
fn cont() -> String {
    dim("  \u{2502}")
}
fn next_latent(next: &[i64]) -> Vec<i64> {
    next.iter().skip((SEQ - 1) * DIM).copied().collect()
}

// --- MLOps metrics over a proven block ---
/// Human-readable byte size (`1 B`, `4.2 KiB`, `10.27 MiB`).
fn bytes_human(n: usize) -> String {
    let x = n as f64;
    if x >= 1024.0 * 1024.0 {
        format!("{:.2} MiB", x / 1024.0 / 1024.0)
    } else if x >= 1024.0 {
        format!("{:.1} KiB", x / 1024.0)
    } else {
        format!("{n} B")
    }
}
/// A short display name for a block op (the predictor's primitive vocabulary).
fn op_kind(op: &BlockOp) -> &'static str {
    match op {
        BlockOp::Linear { .. } => "linear",
        BlockOp::BatchedLinear { .. } => "batched-linear",
        BlockOp::MatMul { .. } => "matmul",
        BlockOp::Softmax { .. } => "softmax",
        BlockOp::Activation { .. } => "gelu-table",
        BlockOp::LayerNorm { .. } => "layernorm",
        BlockOp::Modulate { .. } => "adaln-modulate",
        BlockOp::Gate { .. } => "adaln-gate",
        BlockOp::Add { .. } => "residual-add",
        BlockOp::Requant { .. } => "requant",
        BlockOp::Slice { .. } => "slice",
        BlockOp::Concat { .. } => "concat",
    }
}
/// Op-type histogram, most frequent first.
fn op_histogram(ops: &[BlockOp]) -> Vec<(&'static str, usize)> {
    let mut counts: Vec<(&'static str, usize)> = Vec::new();
    for op in ops {
        let k = op_kind(op);
        match counts.iter_mut().find(|(name, _)| *name == k) {
            Some((_, n)) => *n += 1,
            None => counts.push((k, 1)),
        }
    }
    counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    counts
}
/// Total multiply-accumulates in the forward pass: the linear projections
/// (`rows*cols`, times `seq` for batched) plus the attention matmuls
/// (`rows*inner*cols`). The exact integer compute the prover actually runs.
fn mac_count(ops: &[BlockOp], weights: &[Tensor]) -> u128 {
    let wshape = |id: u32| -> Option<(u128, u128)> {
        weights.iter().find(|t| t.tensor_id() == id).map(|t| {
            let s = t.shape();
            (s[0] as u128, s[1] as u128)
        })
    };
    let mut total: u128 = 0;
    for op in ops {
        match op {
            BlockOp::Linear { weight_id, .. } => {
                if let Some((r, c)) = wshape(*weight_id) {
                    total += r * c;
                }
            }
            BlockOp::BatchedLinear { weight_id, seq, .. } => {
                if let Some((r, c)) = wshape(*weight_id) {
                    total += r * c * (*seq as u128);
                }
            }
            BlockOp::MatMul {
                rows, inner, cols, ..
            } => total += (*rows as u128) * (*inner as u128) * (*cols as u128),
            _ => {}
        }
    }
    total
}
/// Format a multiply-accumulate count (`842 K`, `1.23 G`).
fn macs_human(n: u128) -> String {
    let x = n as f64;
    if x >= 1e9 {
        format!("{:.2} G", x / 1e9)
    } else if x >= 1e6 {
        format!("{:.2} M", x / 1e6)
    } else if x >= 1e3 {
        format!("{:.1} K", x / 1e3)
    } else {
        format!("{n}")
    }
}
/// Throughput of `n` units over a duration, e.g. `25.0 GMAC/s` or `49.0 Kop/s`.
fn rate(n: u128, d: std::time::Duration, unit: &str) -> String {
    let secs = d.as_secs_f64();
    if secs <= 0.0 {
        return format!("- {unit}/s");
    }
    let r = n as f64 / secs;
    if r >= 1e9 {
        format!("{:.1} G{unit}/s", r / 1e9)
    } else if r >= 1e6 {
        format!("{:.1} M{unit}/s", r / 1e6)
    } else if r >= 1e3 {
        format!("{:.1} K{unit}/s", r / 1e3)
    } else {
        format!("{r:.0} {unit}/s")
    }
}

fn log_model(role: &str) {
    println!(
        "{} {}  le-wm action-conditioned predictor block {}",
        tag(role),
        paint("1", "model "),
        dim("(self-attention + GELU FFN + residuals)")
    );
    println!(
        "{} config  dim={DIM}, history={SEQ}, heads=1, action_dim={ACTION_DIM}, mlp={MLP}  {}",
        tag(role),
        dim(&format!(
            "(compact instance; le-wm V0 is dim={V0_DIM}, depth={V0_DEPTH}, {V0_HEADS} heads)"
        ))
    );
    println!(
        "{} weights {} quantized int8 tensors, {} committed tables  {}",
        tag(role),
        predictor::weights().len(),
        predictor::tables().len(),
        dim("(synthesized V0-config instance; le-wm V0 is pretrained:false, no public checkpoint)")
    );
}

/// Prover side: run the predictor, log the inference, return the proof.
fn run_prove(role: &str) -> (PredictorProof, Vec<i64>, std::time::Duration) {
    let t0 = Instant::now();
    let (proof, next) = predictor::prove();
    let elapsed = t0.elapsed();
    let vals = predictor::input_values();
    println!(
        "{} infer   exact integer forward pass in {}",
        tag(role),
        ok(&ms(elapsed))
    );
    println!(
        "{}   z_history {:?}  action {:?}",
        tag(role),
        vals[0],
        vals[1]
    );
    println!(
        "{}   z_next    {:?}  {}",
        tag(role),
        next_latent(&next),
        dim("(predicted next latent)")
    );
    println!(
        "{} trace   {} ops, block_root {}",
        tag(role),
        proof.op_count(),
        dim(&hex8(&predictor::block_root_of(&proof)))
    );
    (proof, next, elapsed)
}

/// Verifier side: the Fiat-Shamir challenge + the audit, timed.
fn run_audit(role: &str, proof: &PredictorProof) -> (bool, Vec<i64>, std::time::Duration) {
    println!(
        "{} challenge  replayed the Fiat-Shamir transcript, derived Freivalds r for {} linear ops",
        tag(role),
        predictor::linear_op_count()
    );
    let t0 = Instant::now();
    let res = predictor::verify(proof);
    let elapsed = t0.elapsed();
    match res {
        Ok(out) => {
            println!(
                "{} checks     Freivalds {} on linears; exact recompute of QK\u{1d40}, softmax, prob\u{00b7}V, GELU, residuals",
                tag(role),
                dim("v\u{00b7}x == r\u{00b7}z")
            );
            println!(
                "{} {}     in {}   z_next {:?}",
                tag(role),
                ok("ACCEPT"),
                ms(elapsed),
                next_latent(&out)
            );
            (true, out, elapsed)
        }
        Err(e) => {
            println!("{} {}     {e:?}", tag(role), bad("REJECT"));
            (false, Vec::new(), elapsed)
        }
    }
}

fn run_tamper(role: &str, proof: &PredictorProof) -> String {
    let mut forged = proof.clone();
    let op = predictor::tamper(&mut forged).expect("a linear op");
    println!(
        "{} tamper     forged the output of matmul op {op} {}",
        tag(role),
        dim("(a fake projection result)")
    );
    match predictor::verify(&forged) {
        Ok(_) => {
            eprintln!("tamper went undetected (this is a bug)");
            exit(1);
        }
        Err(e) => {
            println!("{} {}     {e:?}", tag(role), bad("REJECT"));
            format!("{e:?}")
        }
    }
}

fn read_proof(path: &str) -> PredictorProof {
    let bytes = fs::read(path).unwrap_or_else(|e| {
        eprintln!("read {path} failed: {e}");
        exit(1);
    });
    PredictorProof::from_bytes(&bytes).unwrap_or_else(|e| {
        eprintln!("decode {path} failed: {e:?}");
        exit(1);
    })
}

/// Everything the `prove-predictor` command computes: the world model is loaded,
/// run, proven, audited, and tampered exactly once, and every metric the reporters
/// print is a field here. JSON and human output are then pure projections of this
/// single computation, with no proving logic interleaved in the rendering.
struct PredictorReport {
    label: &'static str,
    is_real: bool,
    input_source: String,
    dims: Dims,
    /// Per-head inner width times head count (`dims.inner()`), cached for the layout line.
    inner: usize,
    weight_tensors: usize,
    table_count: usize,
    params: usize,
    /// On-the-wire model size: int8 weights are one byte per parameter, so this equals `params`.
    model_bytes: usize,
    ops: usize,
    linear_ops: usize,
    witness_vals: usize,
    proof_bytes: usize,
    macs: u128,
    hist: Vec<(&'static str, usize)>,
    weights_root: [u8; 32],
    model_commitment: [u8; 32],
    trace_root: [u8; 32],
    infer: std::time::Duration,
    verify: std::time::Duration,
    /// Predicted next-latent head (first few entries of the verified output).
    z_out_head: Vec<i64>,
    /// First few entries of the quantized latent-history input.
    z_in_head: Vec<i64>,
    /// First few entries of the quantized action input.
    a_in_head: Vec<i64>,
    /// `None` if the honest proof verified; the rejection reason otherwise.
    verify_err: Option<String>,
    /// The op id forged by the tamper pass (`None` only if no linear op existed).
    forged_op: Option<u32>,
    /// The verifier's rejection reason for the forged proof (`None` means undetected, a bug).
    reject: Option<String>,
}

/// Load (or synthesize) the predictor, run the exact integer forward pass, prove,
/// audit, and tamper it once, and gather every metric into a [`PredictorReport`].
/// With a bundle path in `pos[1]` it proves the real 6-block 16-head checkpoint;
/// without one it proves the real V0 dims over synthetic weights.
fn build_predictor_report(pos: &[&str]) -> PredictorReport {
    let real = pos.get(1).map(|path| {
        let j = fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("read {path} failed: {e}");
            exit(1);
        });
        lewm_predictor::load_real_predictor(&j).unwrap_or_else(|e| {
            eprintln!("bad bundle {path}: {e}");
            exit(1);
        })
    });
    let is_real = real.is_some();
    let label = if is_real {
        "le-wm V0 predictor (6 blocks, 16 heads), REAL quantized checkpoint weights"
    } else {
        "le-wm V0 predictor (6 blocks, 16 heads), synthetic weights (pass a bundle for real)"
    };
    let input_source = real
        .as_ref()
        .map(|r| r.input_source.clone())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "synthetic quantized latents".to_string());
    let t0 = Instant::now();
    let (dims, skeleton, weights, tabs, inputs) = match real {
        Some(r) => {
            let (b, w, t, i) = lewm_predictor::build_predictor_real(r.dims, r.blocks, r.x, r.c);
            (r.dims, b, w, t, i)
        }
        None => {
            let d = Dims {
                d: 192,
                s: 3,
                h: 16,
                dh: 64,
                mlp: 2048,
                depth: 6,
            };
            let (b, w, t, i) = lewm_predictor::build_predictor(d);
            (d, b, w, t, i)
        }
    };
    let params: usize = weights.iter().map(|t| t.data().len()).sum();
    let proven = lewm_predictor::prove(&skeleton, &weights, &tabs, &inputs);
    let infer = t0.elapsed();
    let tv = Instant::now();
    let res = lewm_predictor::verify(&proven, &weights, &tabs, &inputs);
    let verify = tv.elapsed();
    let mut forged = proven.clone();
    let forged_op = lewm_predictor::tamper(&mut forged);
    let reject = lewm_predictor::verify(&forged, &weights, &tabs, &inputs)
        .err()
        .map(|e| format!("{e:?}"));
    let out = res.as_ref().ok().cloned().unwrap_or_default();

    // --- MLOps metrics over the proven block ---
    let hist = op_histogram(&proven.ops);
    let macs = mac_count(&proven.ops, &weights);
    let witness_vals: usize = proven.ops.iter().map(|op| op.out().len()).sum();
    let n_linear = proven
        .ops
        .iter()
        .filter(|o| matches!(o, BlockOp::Linear { .. } | BlockOp::BatchedLinear { .. }))
        .count();
    let z_in: &[i64] = inputs.first().map_or(&[], |(_, v)| v.as_slice());
    let a_in: &[i64] = inputs.get(1).map_or(&[], |(_, v)| v.as_slice());
    let head = |v: &[i64]| v[..v.len().min(6)].to_vec();

    PredictorReport {
        label,
        is_real,
        input_source,
        inner: dims.inner(),
        weight_tensors: weights.len(),
        table_count: tabs.len(),
        params,
        model_bytes: params, // int8 weights: one byte per parameter
        ops: proven.ops.len(),
        linear_ops: n_linear,
        witness_vals,
        proof_bytes: witness_vals * core::mem::size_of::<i64>(),
        macs,
        hist,
        weights_root: weights_root(&weights),
        model_commitment: block_architecture_commitment(&proven),
        trace_root: block_root(&proven.ops),
        infer,
        verify,
        z_out_head: head(&out),
        z_in_head: head(z_in),
        a_in_head: head(a_in),
        verify_err: res.err().map(|e| format!("{e:?}")),
        forged_op,
        reject,
        dims,
    }
}

/// Emit the machine-readable predictor report (one JSON object).
fn print_predictor_report_json(r: &PredictorReport) {
    println!(
        "{}",
        json!({
            "model": r.label,
            "real_weights": r.is_real,
            "input_source": r.input_source,
            "config": {"dim": r.dims.d, "history": r.dims.s, "heads": r.dims.h,
                       "dim_head": r.dims.dh, "mlp": r.dims.mlp, "depth": r.dims.depth},
            "weight_tensors": r.weight_tensors,
            "int8_params": r.params,
            "model_bytes": r.model_bytes,
            "ops": r.ops,
            "linear_ops": r.linear_ops,
            "macs": r.macs.to_string(),
            "op_histogram": r.hist.iter().map(|(k, n)| json!({"op": k, "count": n})).collect::<Vec<_>>(),
            "weights_root": hex8(&r.weights_root),
            "model_commitment": hex8(&r.model_commitment),
            "trace_root": hex8(&r.trace_root),
            "infer_ms": r.infer.as_secs_f64() * 1000.0,
            "verify_ms": r.verify.as_secs_f64() * 1000.0,
            "proof_bytes": r.proof_bytes,
            "z_out_head": r.z_out_head,
            "accepted": r.verify_err.is_none(),
            "tamper": {"forged_op": r.forged_op, "rejected_with": r.reject},
        })
    );
}

/// Render the staged, human-readable predictor pipeline (LOAD -> INFER -> COMMIT
/// -> VERIFY -> TAMPER). Exits non-zero if the honest proof was rejected or a
/// forged matmul slipped through, mirroring the demo's fail-closed contract.
fn print_predictor_report_human(r: &PredictorReport) {
    let dims = &r.dims;
    println!(
        "{}",
        paint(
            "36;1",
            "ProvableWorldModel  commit-and-audit over the le-wm world model"
        )
    );
    println!(
        "{}",
        dim("  pipeline   checkpoint -> quantize -> commit -> encode -> infer -> prove -> verify")
    );

    // --- stage 1: LOAD the committed quantized world model ---
    stage(1, 5, "LOAD", "the committed quantized world model");
    println!("{} model    {}", li(false), r.label);
    if r.is_real {
        println!(
            "{} source   quentinll/lewm-pusht {}",
            li(false),
            dim("(Hugging Face, MIT), int8-quantized V0 subgraph")
        );
    }
    println!(
        "{} config   dim={}, history={}, heads={}, dim_head={}, mlp={}, depth={}",
        li(false),
        dims.d,
        dims.s,
        dims.h,
        dims.dh,
        dims.mlp,
        dims.depth
    );
    println!(
        "{} tensors  {} int8 weight matrices, {} committed table(s)",
        li(false),
        r.weight_tensors,
        r.table_count
    );
    println!(
        "{}          {}",
        cont(),
        dim(&format!(
            "per block: qkv[{}x{}] out[{}x{}] fc1[{}x{}] fc2[{}x{}] adaln[{}x{}]",
            3 * r.inner,
            dims.d,
            dims.d,
            r.inner,
            dims.mlp,
            dims.d,
            dims.d,
            dims.mlp,
            6 * dims.d,
            dims.d
        ))
    );
    println!(
        "{} params   {} int8 weights  {}",
        li(false),
        commas(r.params),
        dim(&format!(
            "({} on the wire, 1 byte each)",
            bytes_human(r.model_bytes)
        ))
    );
    println!(
        "{} commit   weights_root {}   model {}",
        li(false),
        dim(&hex8(&r.weights_root)),
        dim(&hex8(&r.model_commitment))
    );
    println!(
        "{} inputs   z_history [{}x{}], action [{}x{}]  {}",
        li(false),
        dims.s,
        dims.d,
        dims.s,
        dims.d,
        dim(&r.input_source)
    );
    println!(
        "{}          {}",
        li(true),
        dim(&format!(
            "z[..6] {:?}   action[..6] {:?}",
            r.z_in_head, r.a_in_head
        ))
    );

    // --- stage 2: INFER (the world model actually runs) ---
    stage(
        2,
        5,
        "INFER",
        "exact integer forward pass (the world model runs)",
    );
    println!(
        "{} graph    {} ops over the named-buffer block DAG  {}",
        li(false),
        commas(r.ops),
        dim("(AdaLN-zero, 16-head attention, GELU FFN, gated residuals)")
    );
    let hist_str = r
        .hist
        .iter()
        .map(|(k, n)| format!("{} {}", commas(*n), k))
        .collect::<Vec<_>>()
        .join(" \u{00b7} ");
    println!("{} ops      {}", li(false), dim(&hist_str));
    println!(
        "{} compute  {} multiply-accumulates, exact integer  {}",
        li(false),
        macs_human(r.macs),
        dim("(no float, no GPU)")
    );
    println!(
        "{} latency  forward pass in {}  {}",
        li(false),
        ok(&ms(r.infer)),
        dim(&format!(
            "({}, {})",
            rate(r.ops as u128, r.infer, "op"),
            rate(r.macs, r.infer, "MAC")
        ))
    );
    println!(
        "{} z_next   {:?}  {}",
        li(true),
        r.z_out_head,
        dim("(predicted next-latent head, from the real forward pass)")
    );

    // --- stage 3: COMMIT (bind execution to a Fiat-Shamir transcript) ---
    stage(
        3,
        5,
        "COMMIT",
        "bind the execution to a Fiat-Shamir transcript",
    );
    println!(
        "{} witness  {} claimed op outputs ({})  trace_root {}",
        li(false),
        commas(r.witness_vals),
        bytes_human(r.proof_bytes),
        dim(&hex8(&r.trace_root))
    );
    println!(
        "{} bind     {}",
        li(true),
        dim("absorbed model + inputs + trace, then squeezed the Freivalds r (non-adaptive)")
    );

    // --- stage 4: VERIFY (no_std, float-free) ---
    stage(
        4,
        5,
        "VERIFY",
        "no_std, float-free, never re-runs the model",
    );
    println!(
        "{} challenge derived the Freivalds r for {} linear projections",
        li(false),
        commas(r.linear_ops)
    );
    println!(
        "{} checks   Freivalds {}  {}",
        li(false),
        dim("v\u{00b7}x == r\u{00b7}z"),
        dim("(soundness \u{2264} 1/p, p = 2\u{2076}\u{00b9}\u{2212}1; union over the checks ~2\u{207b}\u{2074}\u{2074})")
    );
    println!(
        "{}          {}",
        cont(),
        dim("exact recompute of attention, softmax, GELU, LayerNorm, residuals")
    );
    let speedup = if r.verify.as_secs_f64() > 0.0 {
        r.infer.as_secs_f64() / r.verify.as_secs_f64()
    } else {
        0.0
    };
    match &r.verify_err {
        None => println!(
            "{} verdict  {}  in {}  {}",
            li(true),
            ok("ACCEPT"),
            ms(r.verify),
            dim(&format!(
                "({speedup:.1}x faster than proving; audits arithmetic only)"
            ))
        ),
        Some(e) => {
            println!("{} verdict  {}  {e}", li(true), bad("REJECT"));
            exit(1);
        }
    }

    // --- stage 5: TAMPER (forge one matmul output) ---
    stage(5, 5, "TAMPER", "forge one matmul output");
    match &r.reject {
        Some(e) => println!(
            "{} forged matmul op {} -> {} {}  {}",
            li(true),
            r.forged_op.map(|i| i.to_string()).unwrap_or_default(),
            bad("REJECT"),
            e,
            dim("(caught)")
        ),
        None => {
            eprintln!("tamper undetected (bug)");
            exit(1);
        }
    }

    // --- MLOps metrics summary ---
    println!(
        "\n{}  infer {} \u{00b7} verify {} \u{00b7} {} int8 model \u{00b7} {} MAC \u{00b7} {} ops \u{00b7} {}",
        paint("35;1", "metrics"),
        ms(r.infer),
        ms(r.verify),
        bytes_human(r.model_bytes),
        macs_human(r.macs),
        commas(r.ops),
        ok("ACCEPT")
    );
    println!(
        "\n{}",
        ok("a real le-wm world-model forward pass, proven and audited; a forged matmul is caught.")
    );
    if !r.is_real {
        println!(
            "{}",
            dim(
                "  weights are synthetic (real architecture, real integer inference). For the REAL"
            )
        );
        println!(
            "{}",
            dim("  pretrained checkpoint end to end:  ./demo/run-real.sh   (docker compose --profile real up)")
        );
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let json = args.iter().any(|a| a == "--json");
    let pos: Vec<&str> = args
        .iter()
        .skip(1)
        .filter(|a| !a.starts_with("--"))
        .map(String::as_str)
        .collect();
    let cmd = pos.first().copied().unwrap_or("demo");

    match cmd {
        "demo" => {
            if json {
                let (proof, next) = predictor::prove();
                let accepted = predictor::verify(&proof).is_ok();
                let mut forged = proof.clone();
                let op = predictor::tamper(&mut forged);
                let reject = predictor::verify(&forged).err().map(|e| format!("{e:?}"));
                println!(
                    "{}",
                    json!({
                        "model": "le-wm action-conditioned predictor block",
                        "config": {"dim": DIM, "history": SEQ, "mlp": MLP},
                        "z_next": next_latent(&next),
                        "ops": proof.op_count(),
                        "linear_ops": predictor::linear_op_count(),
                        "block_root": hex8(&predictor::block_root_of(&proof)),
                        "accepted": accepted,
                        "tamper": {"forged_op": op, "rejected_with": reject},
                    })
                );
                return;
            }
            println!(
                "{}",
                paint("36;1", "ProvableWorldModel commit-and-audit demo\n")
            );
            log_model("prover");
            let (proof, _, _) = run_prove("prover");
            run_audit("verifier", &proof);
            run_tamper("verifier", &proof);
            println!(
                "\n{}",
                ok("a real world-model prediction, proven and audited; a forged matmul is caught.")
            );
        }
        "prove" => {
            let Some(path) = pos.get(1) else {
                eprintln!("usage: pwm prove <file>");
                exit(2);
            };
            if !json {
                log_model("prover");
            }
            let (proof, next, elapsed) = if json {
                let t0 = Instant::now();
                let (p, n) = predictor::prove();
                (p, n, t0.elapsed())
            } else {
                run_prove("prover")
            };
            let bytes = proof.to_bytes();
            fs::write(path, &bytes).unwrap_or_else(|e| {
                eprintln!("write {path} failed: {e}");
                exit(1);
            });
            if json {
                println!(
                    "{}",
                    json!({"wrote": path, "bytes": bytes.len(), "infer_ms": elapsed.as_secs_f64()*1000.0, "z_next": next_latent(&next), "ops": proof.op_count()})
                );
            } else {
                println!(
                    "{} wrote   proof ({} bytes) to {path}",
                    tag("prover"),
                    bytes.len()
                );
            }
        }
        "verify" => {
            let Some(path) = pos.get(1) else {
                eprintln!("usage: pwm verify <file>");
                exit(2);
            };
            let proof = read_proof(path);
            match predictor::verify(&proof) {
                Ok(out) => {
                    if json {
                        println!("{}", json!({"accepted": true, "z_next": next_latent(&out)}));
                    } else {
                        println!("{} ACCEPT  z_next {:?}", ok("\u{2713}"), next_latent(&out));
                    }
                }
                Err(e) => {
                    if json {
                        println!("{}", json!({"accepted": false, "error": format!("{e:?}")}));
                    } else {
                        println!("{} REJECT  {e:?}", bad("\u{2717}"));
                    }
                    exit(1);
                }
            }
        }
        "audit" => {
            let Some(path) = pos.get(1) else {
                eprintln!("usage: pwm audit <file>");
                exit(2);
            };
            let proof = read_proof(path);
            if json {
                let res = predictor::verify(&proof);
                let mut forged = proof.clone();
                predictor::tamper(&mut forged);
                let reject = predictor::verify(&forged).err().map(|e| format!("{e:?}"));
                println!(
                    "{}",
                    json!({
                        "ops": proof.op_count(),
                        "linear_ops": predictor::linear_op_count(),
                        "accepted": res.is_ok(),
                        "z_next": res.ok().map(|o| next_latent(&o)),
                        "tamper": {"rejected_with": reject},
                    })
                );
                return;
            }
            println!(
                "{} loaded     proof: {} claimed op outputs over the public predictor graph",
                tag("verifier"),
                proof.op_count()
            );
            run_audit("verifier", &proof);
            run_tamper("verifier", &proof);
            println!("\n{}", ok("honest proof accepted; forged proof rejected."));
        }
        "tamper" => {
            let (Some(inp), Some(outp)) = (pos.get(1), pos.get(2)) else {
                eprintln!("usage: pwm tamper <in> <out>");
                exit(2);
            };
            let mut proof = read_proof(inp);
            let op = predictor::tamper(&mut proof);
            fs::write(outp, proof.to_bytes()).unwrap_or_else(|e| {
                eprintln!("write {outp} failed: {e}");
                exit(1);
            });
            if json {
                println!("{}", json!({"forged_op": op, "wrote": outp}));
            } else {
                println!(
                    "{} forged matmul op {op:?}, wrote tampered proof to {outp}",
                    dim("\u{2022}")
                );
            }
        }
        "prove-lewm" => {
            let Some(path) = pos.get(1) else {
                eprintln!("usage: pwm prove-lewm <bundle.json>");
                exit(2);
            };
            let bundle_json = fs::read_to_string(path).unwrap_or_else(|e| {
                eprintln!("read {path} failed: {e}");
                exit(1);
            });
            let bundle = lewm::load_bundle(&bundle_json).unwrap_or_else(|e| {
                eprintln!("bad bundle {path}: {e}");
                exit(1);
            });
            let t0 = Instant::now();
            let artifact = lewm::prove(&bundle).unwrap_or_else(|e| {
                eprintln!("prove {path} failed: {e:?}");
                exit(1);
            });
            let infer = t0.elapsed();
            let bytes = canonical_bytes(&artifact).len();
            let out: Vec<i64> = artifact
                .claimed_output
                .data()
                .iter()
                .map(pwm_core::BoundedInt::value)
                .collect();
            let tv = Instant::now();
            let accepted = verify_artifact(&artifact).is_ok();
            let vtime = tv.elapsed();
            let mut forged = lewm::prove(&bundle).unwrap_or_else(|e| {
                eprintln!("prove {path} failed: {e:?}");
                exit(1);
            });
            let forged_op = demo::tamper_accumulator(&mut forged);
            let reject = verify_artifact(&forged).err().map(|e| format!("{e:?}"));
            if json {
                println!(
                    "{}",
                    json!({"model": bundle.label, "dim": bundle.dim, "mlp": bundle.mlp,
                        "proof_bytes": bytes, "accepted": accepted, "output_head": &out[..out.len().min(6)],
                        "tamper": {"forged_op": forged_op, "rejected_with": reject}})
                );
                return;
            }
            println!(
                "{}",
                paint(
                    "36;1",
                    "ProvableWorldModel: real le-wm checkpoint into the prover\n"
                )
            );
            println!(
                "{} {}  {}",
                tag("prover"),
                paint("1", "model "),
                bundle.label
            );
            println!(
                "{} config  dim={}, mlp={}  {}",
                tag("prover"),
                bundle.dim,
                bundle.mlp,
                dim("(real BN-folded pred_proj weights from quentinll/lewm-pusht)")
            );
            println!(
                "{} infer   exact integer forward pass in {}",
                tag("prover"),
                ok(&ms(infer))
            );
            println!(
                "{}   input  z[..6] {:?}   output z_proj[..6] {:?}",
                tag("prover"),
                &bundle.input[..bundle.input.len().min(6)],
                &out[..out.len().min(6)]
            );
            println!("{} proof   {} bytes", tag("prover"), bytes);
            if accepted {
                println!("{} {}  in {}", tag("verifier"), ok("ACCEPT"), ms(vtime));
            } else {
                println!("{} {}", tag("verifier"), bad("REJECT"));
                exit(1);
            }
            match reject {
                Some(e) => println!(
                    "{} tamper  forged matmul op {forged_op:?} -> {} {e}",
                    tag("verifier"),
                    bad("REJECT")
                ),
                None => {
                    eprintln!("tamper undetected (bug)");
                    exit(1);
                }
            }
            println!(
                "\n{}",
                ok("real le-wm pred_proj head proven and verified; a forged matmul is caught.")
            );
        }
        "prove-predictor" => {
            // With a bundle: the real 6-block 16-head predictor from the checkpoint.
            // Without: the real V0 dims (192/16/64, depth 6) over synthetic weights.
            let report = build_predictor_report(&pos);
            if json {
                print_predictor_report_json(&report);
            } else {
                print_predictor_report_human(&report);
            }
        }
        other => {
            eprintln!("unknown command {other:?}; usage: pwm [demo|prove <f>|verify <f>|audit <f>|tamper <in> <out>|prove-lewm <bundle>|prove-predictor [bundle]] [--json]");
            exit(2);
        }
    }
}
