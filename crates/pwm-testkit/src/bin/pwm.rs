// SPDX-License-Identifier: Apache-2.0
//! `pwm` — the prover/verifier demo CLI (backlog PR-304).
//!
//! Exercises the full commit-and-audit path over a small built-in quantized model:
//!
//! ```text
//! pwm demo            # prove + verify in-memory, print the result
//! pwm prove <file>    # prove and write the canonical artifact bytes to <file>
//! pwm verify <file>   # read an artifact from <file>, decode, and verify it
//! ```
//!
//! Reading an artifact back exercises the canonical binary decode (PR-303); the
//! verifier never re-runs the model.

use std::env;
use std::fs;
use std::process::exit;

use pwm_core::audit::AuditArtifact;
use pwm_core::fixed_point::{BoundedInt, Rounding};
use pwm_core::serialize::{canonical_bytes, from_canonical_bytes};
use pwm_core::tables::ActivationTable;
use pwm_core::tensor::{Dtype, Scale, Tensor};
use pwm_export::reference::{LayerSpec, Model};
use pwm_prover::{prove_feedforward, OutputBinding};
use pwm_verifier::verify;

fn weight(id: u32, rows: u32, cols: u32, vals: &[i8]) -> Tensor {
    let data = vals
        .iter()
        .map(|&v| BoundedInt::new(v as i64, -128, 127).unwrap())
        .collect();
    Tensor::new(id, vec![rows, cols], 0, data).unwrap()
}

/// A small built-in quantized feed-forward model: Linear(3->4) + GELU-table, then
/// Linear(4->2) + requant.
fn demo_model() -> Model {
    let scales = vec![
        Scale {
            scale_id: 0,
            log2: 0,
            dtype: Dtype::I8,
        },
        Scale {
            scale_id: 1,
            log2: 0,
            dtype: Dtype::I8,
        },
    ];
    let l1 = LayerSpec {
        pre_layernorm: None,
        linear_op_id: 100,
        weight: weight(10, 4, 3, &[1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1]),
        bias: None,
        requant_op_id: 101,
        shift: 0,
        zero_point: 0,
        clamp_lo: -128,
        clamp_hi: 127,
        rounding: Rounding::NearestTiesToEven,
        activation: Some((102, 0)),
    };
    let l2 = LayerSpec {
        pre_layernorm: None,
        linear_op_id: 103,
        weight: weight(12, 2, 4, &[1, 1, 1, 1, 1, -1, 1, -1]),
        bias: None,
        requant_op_id: 104,
        shift: 1,
        zero_point: 0,
        clamp_lo: -128,
        clamp_hi: 127,
        rounding: Rounding::NearestTiesToEven,
        activation: None,
    };
    Model {
        layers: vec![l1, l2],
        // Identity activation table over the int8 domain.
        tables: vec![ActivationTable {
            table_id: 0,
            lo: -128,
            outputs: (-128i64..=127).collect(),
        }],
        scales,
    }
}

fn out_binding() -> OutputBinding {
    OutputBinding {
        tensor_id: 200,
        scale_id: 1,
    }
}

fn prove_demo() -> AuditArtifact {
    prove_feedforward(&demo_model(), &[1, 2, 3], out_binding()).expect("prove")
}

fn report(artifact: &AuditArtifact, bytes: usize) {
    match verify(artifact) {
        Ok(()) => {
            let out: Vec<i64> = artifact
                .claimed_output
                .data()
                .iter()
                .map(|c| c.value())
                .collect();
            println!("verified OK  ({bytes} artifact bytes, output {out:?})");
        }
        Err(e) => {
            eprintln!("verification FAILED: {e:?}");
            exit(1);
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("demo");
    match cmd {
        "demo" => {
            let a = prove_demo();
            report(&a, canonical_bytes(&a).len());
        }
        "prove" => {
            let Some(path) = args.get(2) else {
                eprintln!("usage: pwm prove <file>");
                exit(2);
            };
            let a = prove_demo();
            let bytes = canonical_bytes(&a);
            fs::write(path, &bytes).unwrap_or_else(|e| {
                eprintln!("write {path} failed: {e}");
                exit(1);
            });
            println!("wrote {} artifact bytes to {path}", bytes.len());
        }
        "verify" => {
            let Some(path) = args.get(2) else {
                eprintln!("usage: pwm verify <file>");
                exit(2);
            };
            let bytes = fs::read(path).unwrap_or_else(|e| {
                eprintln!("read {path} failed: {e}");
                exit(1);
            });
            let a: AuditArtifact = from_canonical_bytes(&bytes).unwrap_or_else(|e| {
                eprintln!("decode {path} failed: {e:?}");
                exit(1);
            });
            report(&a, bytes.len());
        }
        other => {
            eprintln!("unknown command {other:?}; usage: pwm [demo|prove <file>|verify <file>]");
            exit(2);
        }
    }
}
