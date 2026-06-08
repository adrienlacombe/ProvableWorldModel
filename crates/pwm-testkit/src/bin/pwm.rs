// SPDX-License-Identifier: Apache-2.0
//! `pwm` — the prover/verifier demo CLI for the commit-and-audit scheme.
//!
//! It drives the full path over a small built-in quantized model (real integer
//! inference, no floating point, no external assets):
//!
//! ```text
//! pwm demo              # the whole story: prove -> challenge -> accept -> tamper -> reject
//! pwm prove  <file>     # run inference, write the canonical AuditArtifact bytes
//! pwm verify <file>     # decode an artifact and verify it (accept or reject)
//! pwm challenge <file>  # verifier-secret Freivalds challenge game (accept)
//! pwm audit  <file>     # the verifier's full story: challenge accept, then tamper -> reject
//! pwm tamper <in> <out> # forge one matmul output (the canonical Freivalds-caught lie)
//! ```
//!
//! Add `--json` to any command for machine-readable output. The docker demo runs
//! a prover service (`prove`) and a verifier service (`audit`) over a shared file.

use std::env;
use std::fs;
use std::process::exit;

use pwm_core::audit::AuditArtifact;
use pwm_core::serialize::{canonical_bytes, from_canonical_bytes};
use pwm_testkit::demo::{prove_demo, secret_challenge, tamper_accumulator, DEMO_INPUT};
use pwm_verifier::{verify, verify_interactive};
use serde_json::json;

/// Verifier-local challenge seed. It is drawn here, never from the Fiat-Shamir
/// transcript, so the prover cannot have known it when it committed the trace.
const SECRET_SEED: u64 = 0x00C0_FFEE_D00D;

// --- tiny ANSI helpers (respect NO_COLOR) ---
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
fn head(s: &str) -> String {
    paint("36;1", s)
}

fn output_of(a: &AuditArtifact) -> Vec<i64> {
    a.claimed_output.data().iter().map(|c| c.value()).collect()
}

fn read_artifact(path: &str) -> (Vec<u8>, AuditArtifact) {
    let bytes = fs::read(path).unwrap_or_else(|e| {
        eprintln!("read {path} failed: {e}");
        exit(1);
    });
    let a = from_canonical_bytes::<AuditArtifact>(&bytes).unwrap_or_else(|e| {
        eprintln!("decode {path} failed: {e:?}");
        exit(1);
    });
    (bytes, a)
}

/// Run the verifier-secret challenge game on an honest artifact. Returns the
/// number of linear ops challenged.
fn challenge(a: &AuditArtifact, json: bool) -> usize {
    let r = secret_challenge(a, SECRET_SEED);
    let n = r.len();
    match verify_interactive(a, &r) {
        Ok(()) => {
            if !json {
                println!(
                    "  {} verifier drew a {} challenge r ({} vector(s)); checked {} per linear op",
                    ok("✓"),
                    paint("1", "secret"),
                    n,
                    dim("v·x == r·z"),
                );
                println!("  {} ACCEPT  output {:?}", ok("✓"), output_of(a));
            }
            n
        }
        Err(e) => {
            if !json {
                println!("  {} REJECT  {e:?}", bad("✗"));
            }
            exit(1);
        }
    }
}

/// Forge one matmul output and show the verifier reject it.
fn tamper_and_verify(mut bad_artifact: AuditArtifact, json: bool) -> String {
    let op = tamper_accumulator(&mut bad_artifact).expect("a linear op");
    match verify(&bad_artifact) {
        Ok(()) => {
            eprintln!("tamper went undetected (this is a bug)");
            exit(1);
        }
        Err(e) => {
            if !json {
                println!(
                    "  {} forged the accumulator of linear op {op} (a fake matmul result)",
                    dim("•")
                );
                println!("  {} REJECT  {e:?}", bad("✗"));
            }
            format!("{e:?}")
        }
    }
}

fn run_demo(json: bool) {
    let a = prove_demo();
    let bytes = canonical_bytes(&a).len();
    if json {
        let r = secret_challenge(&a, SECRET_SEED);
        let accepted = verify_interactive(&a, &r).is_ok();
        let mut bad_artifact = prove_demo();
        let op = tamper_accumulator(&mut bad_artifact);
        let reject = verify(&bad_artifact).err().map(|e| format!("{e:?}"));
        println!(
            "{}",
            json!({
                "input": DEMO_INPUT,
                "artifact_bytes": bytes,
                "output": output_of(&a),
                "challenge": { "linear_ops": r.len(), "verifier_secret": true, "accepted": accepted },
                "tamper": { "forged_linear_op": op, "rejected_with": reject },
            })
        );
        return;
    }
    println!("{}", head("ProvableWorldModel commit-and-audit demo"));
    println!(
        "{}",
        dim("a quantized world-model step, proven and audited with no floating point\n")
    );
    println!(
        "{} prove   ran integer inference on input {:?}, committed the trace, built a {}-byte proof",
        head("1."),
        DEMO_INPUT,
        bytes
    );
    println!("{} challenge", head("2."));
    challenge(&a, false);
    println!("{} tamper", head("3."));
    tamper_and_verify(prove_demo(), false);
    println!(
        "\n{}",
        ok("the honest proof verifies; a single forged matmul is caught. that is the whole game.")
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let json = args.iter().any(|a| a == "--json");
    let positional: Vec<&str> = args
        .iter()
        .skip(1)
        .filter(|a| !a.starts_with("--"))
        .map(String::as_str)
        .collect();
    let cmd = positional.first().copied().unwrap_or("demo");

    match cmd {
        "demo" => run_demo(json),
        "prove" => {
            let Some(path) = positional.get(1) else {
                eprintln!("usage: pwm prove <file>");
                exit(2);
            };
            let a = prove_demo();
            let bytes = canonical_bytes(&a);
            fs::write(path, &bytes).unwrap_or_else(|e| {
                eprintln!("write {path} failed: {e}");
                exit(1);
            });
            if json {
                println!(
                    "{}",
                    json!({"wrote": path, "artifact_bytes": bytes.len(), "output": output_of(&a)})
                );
            } else {
                println!(
                    "{} prover ran inference and wrote a {}-byte proof to {path}",
                    ok("✓"),
                    bytes.len()
                );
            }
        }
        "verify" => {
            let Some(path) = positional.get(1) else {
                eprintln!("usage: pwm verify <file>");
                exit(2);
            };
            let (bytes, a) = read_artifact(path);
            match verify(&a) {
                Ok(()) => {
                    if json {
                        println!(
                            "{}",
                            json!({"accepted": true, "artifact_bytes": bytes.len(), "output": output_of(&a)})
                        );
                    } else {
                        println!(
                            "{} ACCEPT  ({} bytes, output {:?})",
                            ok("✓"),
                            bytes.len(),
                            output_of(&a)
                        );
                    }
                }
                Err(e) => {
                    if json {
                        println!("{}", json!({"accepted": false, "error": format!("{e:?}")}));
                    } else {
                        println!("{} REJECT  {e:?}", bad("✗"));
                    }
                    exit(1);
                }
            }
        }
        "challenge" => {
            let Some(path) = positional.get(1) else {
                eprintln!("usage: pwm challenge <file>");
                exit(2);
            };
            let (_, a) = read_artifact(path);
            let n = challenge(&a, json);
            if json {
                println!(
                    "{}",
                    json!({"accepted": true, "linear_ops": n, "verifier_secret": true, "output": output_of(&a)})
                );
            }
        }
        "audit" => {
            let Some(path) = positional.get(1) else {
                eprintln!("usage: pwm audit <file>");
                exit(2);
            };
            let (_, a) = read_artifact(path);
            if !json {
                println!("{} challenge", head("verifier:"));
            }
            let n = challenge(&a, json);
            // Tamper a fresh decode of the same proof and show the reject.
            let (_, bad_copy) = read_artifact(path);
            if !json {
                println!("{} tamper attempt", head("verifier:"));
            }
            let err = tamper_and_verify(bad_copy, json);
            if json {
                println!(
                    "{}",
                    json!({
                        "challenge": {"accepted": true, "linear_ops": n, "verifier_secret": true},
                        "tamper": {"rejected_with": err},
                        "output": output_of(&a),
                    })
                );
            } else {
                println!("\n{}", ok("honest proof accepted; forged proof rejected."));
            }
        }
        "tamper" => {
            let (Some(inp), Some(outp)) = (positional.get(1), positional.get(2)) else {
                eprintln!("usage: pwm tamper <in> <out>");
                exit(2);
            };
            let (_, mut a) = read_artifact(inp);
            let op = tamper_accumulator(&mut a);
            let bytes = canonical_bytes(&a);
            fs::write(outp, &bytes).unwrap_or_else(|e| {
                eprintln!("write {outp} failed: {e}");
                exit(1);
            });
            if json {
                println!("{}", json!({"forged_linear_op": op, "wrote": outp}));
            } else {
                println!(
                    "{} forged linear op {op:?} and wrote a tampered proof to {outp}",
                    dim("•")
                );
            }
        }
        other => {
            eprintln!(
                "unknown command {other:?}; usage: pwm [demo|prove <f>|verify <f>|challenge <f>|audit <f>|tamper <in> <out>] [--json]"
            );
            exit(2);
        }
    }
}
