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
//! ```
//!
//! Add `--json` for machine-readable output. The docker demo runs a prover service
//! (`prove`) and a verifier service (`audit`) over a shared proof file.

use std::env;
use std::fs;
use std::process::exit;
use std::time::Instant;

use pwm_testkit::predictor::{
    self, PredictorProof, ACTION_DIM, DIM, MLP, SEQ, V0_DEPTH, V0_DIM, V0_HEADS,
};
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
        other => {
            eprintln!("unknown command {other:?}; usage: pwm [demo|prove <f>|verify <f>|audit <f>|tamper <in> <out>] [--json]");
            exit(2);
        }
    }
}
