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

use pwm_core::serialize::canonical_bytes;
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
            let bundle = lewm::load_bundle(&bundle_json);
            let t0 = Instant::now();
            let artifact = lewm::prove(&bundle);
            let infer = t0.elapsed();
            let bytes = canonical_bytes(&artifact).len();
            let out: Vec<i64> = artifact
                .claimed_output
                .data()
                .iter()
                .map(|c| c.value())
                .collect();
            let tv = Instant::now();
            let accepted = verify_artifact(&artifact).is_ok();
            let vtime = tv.elapsed();
            let mut forged = lewm::prove(&bundle);
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
            let real = pos.get(1).map(|path| {
                let j = fs::read_to_string(path).unwrap_or_else(|e| {
                    eprintln!("read {path} failed: {e}");
                    exit(1);
                });
                lewm_predictor::load_real_predictor(&j)
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
                    let (b, w, t, i) =
                        lewm_predictor::build_predictor_real(r.dims, r.blocks, r.x, r.c);
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
            let vtime = tv.elapsed();
            let mut forged = proven.clone();
            let forged_op = lewm_predictor::tamper(&mut forged);
            let reject = lewm_predictor::verify(&forged, &weights, &tabs, &inputs)
                .err()
                .map(|e| format!("{e:?}"));
            let out = res.as_ref().ok().cloned().unwrap_or_default();
            if json {
                println!(
                    "{}",
                    json!({"model": label, "ops": proven.ops.len(), "weight_tensors": weights.len(),
                        "accepted": res.is_ok(), "z_out_head": &out[..out.len().min(6)],
                        "tamper": {"forged_op": forged_op, "rejected_with": reject}})
                );
                return;
            }
            println!(
                "{}",
                paint(
                    "36;1",
                    "ProvableWorldModel  commit-and-audit over the le-wm world model"
                )
            );
            println!(
                "{}",
                dim("  pipeline   checkpoint -> quantize -> commit -> encode -> run -> prove -> verify")
            );

            // --- stage 1: export (offline, trusted) ---
            stage(1, 4, "EXPORT", "offline, trusted");
            println!("{} model    {}", li(false), label);
            if is_real {
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
                "{} weights  {} tensors, {} int8 params",
                li(false),
                weights.len(),
                commas(params)
            );
            println!(
                "{} inputs   z_history [{}x{}], action embedding [{}x{}]",
                li(false),
                dims.s,
                dims.d,
                dims.s,
                dims.d
            );
            println!("{} source   {}", li(true), dim(&input_source));

            // --- stage 2: prove (exact integer inference + commitment) ---
            stage(2, 4, "PROVE", "exact integer inference + commitment");
            println!(
                "{} graph    {} ops over the named-buffer block DAG",
                li(false),
                commas(proven.ops.len())
            );
            println!(
                "{}          {}",
                cont(),
                dim("per block: AdaLN-zero, 16-head attention, GELU FFN, gated residuals")
            );
            println!(
                "{} infer    exact integer forward pass in {}",
                li(false),
                ok(&ms(infer))
            );
            println!(
                "{} z_next   {:?}  {}",
                li(true),
                &out[..out.len().min(6)],
                dim("(predicted next-latent head)")
            );

            // --- stage 3: verify (no_std, float-free) ---
            stage(3, 4, "VERIFY", "no_std, float-free");
            println!(
                "{} challenge replayed the Fiat-Shamir transcript, derived the Freivalds r",
                li(false)
            );
            println!(
                "{} checks   Freivalds {} on every projection",
                li(false),
                dim("v\u{00b7}x == r\u{00b7}z")
            );
            println!(
                "{}          {}",
                cont(),
                dim("exact recompute of attention, softmax, GELU, LayerNorm, residuals")
            );
            match res {
                Ok(_) => println!("{} verdict  {}  in {}", li(true), ok("ACCEPT"), ms(vtime)),
                Err(e) => {
                    println!("{} verdict  {}  {e:?}", li(true), bad("REJECT"));
                    exit(1);
                }
            }

            // --- stage 4: tamper (forge one matmul output) ---
            stage(4, 4, "TAMPER", "forge one matmul output");
            match reject {
                Some(e) => println!(
                    "{} forged matmul op {} -> {} {}  {}",
                    li(true),
                    forged_op.map(|i| i.to_string()).unwrap_or_default(),
                    bad("REJECT"),
                    e,
                    dim("(caught)")
                ),
                None => {
                    eprintln!("tamper undetected (bug)");
                    exit(1);
                }
            }
            println!("\n{}", ok("the full 6-block 16-head predictor proven and verified; a forged matmul is caught."));
        }
        other => {
            eprintln!("unknown command {other:?}; usage: pwm [demo|prove <f>|verify <f>|audit <f>|tamper <in> <out>|prove-lewm <bundle>|prove-predictor [bundle]] [--json]");
            exit(2);
        }
    }
}
