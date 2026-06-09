// SPDX-License-Identifier: Apache-2.0
//! Regression: the mod-`p` accumulator-aliasing forgery must be REJECTED (WQ-01).
//!
//! Freivalds checks `v·x == r·z` in Fp61 (mod `p = 2^61-1`), which only proves
//! `z ≡ Wx (mod p)`. Adding `p` to a claimed accumulator collides mod `p` for
//! EVERY challenge `r`, so without a range check the verifier would accept a
//! forged output with probability 1. The soundness range guard
//! (`pwm_core::freivalds::check_linear_biased` + `VerifyError::AccumulatorRange`)
//! bounds every operand to `[-SAFE_HI, SAFE_HI]`, so an accumulator inflated by a
//! multiple of `p` (`p ≫ SAFE_HI`) is rejected. This test reproduces the original
//! forgery and asserts it is now rejected; it must stay red if the guard regresses.

use pwm_core::audit::output_tensor;
use pwm_core::commit::claimed_output_commitment;
use pwm_core::field::FREIVALDS_P;
use pwm_core::fixed_point::{requantize, Rounding};
use pwm_core::trace::OpRecord;
use pwm_testkit::demo::prove_demo;
use pwm_verifier::{verify, VerifyError};

/// Forge the final linear accumulator by `+p` and rebuild a self-consistent proof
/// (forged requant output + forged claimed-output commitment), exactly as a
/// malicious prover would. Returns the verifier's verdict and the forged output.
fn forge_modp() -> (Result<(), VerifyError>, Vec<i64>, Vec<i64>) {
    let mut art = prove_demo();
    let honest: Vec<i64> = art
        .claimed_output
        .data()
        .iter()
        .map(pwm_core::BoundedInt::value)
        .collect();
    assert!(verify(&art).is_ok(), "honest proof must verify");

    let p = FREIVALDS_P as i64;
    let n = art.trace.len();

    // demo trace: [Linear(100), Activation(102), Linear(103), Requant(104)].
    // The last Linear feeds the final Requant (no domain check), so inject p there.
    match &mut art.trace[n - 2] {
        OpRecord::Linear(l) => l.output[0] += p,
        other => panic!("expected Linear at n-2, got {other:?}"),
    }
    let new_final = match &mut art.trace[n - 1] {
        OpRecord::Requant(rq) => {
            rq.input[0] += p; // keep the wiring (input == previous output)
            rq.output = rq
                .input
                .iter()
                .map(|&n| {
                    requantize(
                        n,
                        rq.shift,
                        rq.zero_point,
                        rq.clamp_lo,
                        rq.clamp_hi,
                        Rounding::NearestTiesToEven,
                    )
                })
                .collect();
            rq.output.clone()
        }
        other => panic!("expected Requant at n-1, got {other:?}"),
    };

    // The prover controls the public input, so it rebinds the claimed output too.
    let forged_tensor = output_tensor(
        art.claimed_output.tensor_id(),
        art.claimed_output.scale_id(),
        &new_final,
    )
    .unwrap();
    art.public_input.claimed_output_commitment =
        claimed_output_commitment(core::slice::from_ref(&forged_tensor));
    art.claimed_output = forged_tensor;

    (verify(&art), honest, new_final)
}

#[test]
fn modp_accumulator_forgery_is_rejected() {
    let (res, honest, forged) = forge_modp();
    // The forgery genuinely changes the output (it is a real attack, not a no-op)...
    assert_ne!(
        honest, forged,
        "the forgery must actually change the output"
    );
    // ...and the verifier must reject it with the accumulator-range code.
    assert_eq!(
        res,
        Err(VerifyError::AccumulatorRange { op_id: 103 }),
        "mod-p accumulator aliasing must be rejected, got {res:?}"
    );
}
