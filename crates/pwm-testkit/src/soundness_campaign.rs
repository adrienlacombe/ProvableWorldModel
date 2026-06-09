// SPDX-License-Identifier: Apache-2.0
//! A real soundness mutation campaign (replaces the empty skeleton for the
//! `SOUNDNESS_CRITICAL` components — INV-TEST-05, WQ-05).
//!
//! Each [`Mutant`] is a concrete weakening of a verifier soundness check, paired
//! with the forgery that weakening would let through. [`SoundnessCampaign::is_killed`]
//! constructs that forgery and runs the **live** verifier: the mutant is *killed*
//! iff the verifier still rejects (i.e. a reject test exists that would fail if the
//! check were removed). A surviving mutant in a soundness-critical component is a
//! direct unsoundness and fails the merge gate.
//!
//! The `range_check.modp_accumulator` mutant is the mod-`p` accumulator-aliasing
//! forgery (WQ-01): before the Freivalds range guard landed, the verifier accepted
//! it, so this campaign would have reported a soundness-critical survivor and failed
//! CI — the gate the green skeleton never provided.

use pwm_core::audit::output_tensor;
use pwm_core::commit::claimed_output_commitment;
use pwm_core::field::FREIVALDS_P;
use pwm_core::fixed_point::{requantize, Rounding};
use pwm_core::planning::verify_argmin;
use pwm_core::trace::OpRecord;
use pwm_verifier::{verify, VerifyError};

use crate::demo::prove_demo;
use crate::mutation::{Mutant, MutationCampaign, MutationOperator};

/// A campaign over the live commit-and-audit verifier's soundness checks.
#[derive(Debug, Default, Clone, Copy)]
pub struct SoundnessCampaign;

fn mutant(id: &str, component: &str, operator: MutationOperator, description: &str) -> Mutant {
    Mutant {
        id: id.into(),
        component: component.into(),
        operator,
        description: description.into(),
    }
}

/// True iff the mod-`p` accumulator-aliasing forgery (add `p` to a Freivalds
/// accumulator) is rejected by the live verifier (WQ-01 range guard).
fn modp_accumulator_rejected() -> bool {
    let mut art = prove_demo();
    let p = FREIVALDS_P as i64;
    let n = art.trace.len();
    match &mut art.trace[n - 2] {
        OpRecord::Linear(l) => l.output[0] += p,
        _ => return false,
    }
    let new_final = match &mut art.trace[n - 1] {
        OpRecord::Requant(rq) => {
            rq.input[0] += p;
            rq.output = rq
                .input
                .iter()
                .map(|&v| {
                    requantize(
                        v,
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
        _ => return false,
    };
    let forged = output_tensor(
        art.claimed_output.tensor_id(),
        art.claimed_output.scale_id(),
        &new_final,
    )
    .unwrap();
    art.public_input.claimed_output_commitment =
        claimed_output_commitment(core::slice::from_ref(&forged));
    art.claimed_output = forged;
    verify(&art).is_err()
}

/// True iff breaking the op-to-op wiring is rejected (the `tensor_memory`
/// component): bump a mid-trace record's first input cell while leaving the prior
/// record's output intact, so the threaded running activation no longer matches.
/// The verifier's wiring check (`VerifyError::WiringMismatch`) must fire — it runs
/// before each op's own check, so this is the dedicated guard for trace memory
/// consistency, which had no mutant before (INV-TEST-05).
fn wiring_tamper_rejected() -> bool {
    let mut art = prove_demo();
    for rec in art.trace.iter_mut().skip(1) {
        let input = match rec {
            OpRecord::Linear(r) => &mut r.input,
            OpRecord::Requant(r) => &mut r.input,
            OpRecord::Activation(r) => &mut r.input,
            OpRecord::LayerNorm(r) => &mut r.input,
        };
        if let Some(first) = input.first_mut() {
            *first += 1;
            return matches!(verify(&art), Err(VerifyError::WiringMismatch { .. }));
        }
    }
    false
}

/// True iff tampering an exactly-recomputed op output is rejected. `select` picks
/// the trace record to perturb (the requant or the activation).
fn tampered_op_rejected(select: fn(&OpRecord) -> bool) -> bool {
    let mut art = prove_demo();
    for rec in &mut art.trace {
        if select(rec) {
            match rec {
                OpRecord::Requant(r) => r.output[0] += 1,
                OpRecord::Activation(r) => r.output[0] += 1,
                _ => {}
            }
            return verify(&art).is_err();
        }
    }
    false
}

impl MutationCampaign for SoundnessCampaign {
    fn mutants(&self) -> Vec<Mutant> {
        vec![
            mutant(
                "range_check.modp_accumulator",
                "range_check",
                MutationOperator::DropConstraint,
                "drop the Freivalds accumulator range guard: add p to an accumulator (mod-p aliasing)",
            ),
            mutant(
                "tensor_memory.break_wiring",
                "tensor_memory",
                MutationOperator::DropConstraint,
                "drop the op-to-op wiring check: a record's input no longer equals the prior output",
            ),
            mutant(
                "requant.tamper_output",
                "requant",
                MutationOperator::OffByOneBound,
                "off-by-one a requantized output cell vs the exact recompute",
            ),
            mutant(
                "activation_lookup.tamper_output",
                "activation_lookup",
                MutationOperator::WeakenLookup,
                "claim a table read that does not match the committed lookup",
            ),
            mutant(
                "argmin.select_non_minimum",
                "argmin",
                MutationOperator::RelaxComparison,
                "select a candidate that is not the minimum cost",
            ),
            mutant(
                "argmin.relax_tiebreak",
                "argmin",
                MutationOperator::RelaxComparison,
                "select a later candidate that ties the minimum (violates smallest-index tie-break)",
            ),
        ]
    }

    fn is_killed(&self, mutant: &Mutant) -> bool {
        match mutant.id.as_str() {
            "range_check.modp_accumulator" => modp_accumulator_rejected(),
            "tensor_memory.break_wiring" => wiring_tamper_rejected(),
            "requant.tamper_output" => tampered_op_rejected(|r| matches!(r, OpRecord::Requant(_))),
            "activation_lookup.tamper_output" => {
                tampered_op_rejected(|r| matches!(r, OpRecord::Activation(_)))
            }
            // costs [5, 3]: selecting index 0 (cost 5) is not the minimum (index 1 is).
            "argmin.select_non_minimum" => verify_argmin(&[5, 3], 0, 5).is_err(),
            // costs [3, 3]: selecting index 1 violates smallest-index tie-break.
            "argmin.relax_tiebreak" => verify_argmin(&[3, 3], 1, 3).is_err(),
            // An unregistered mutant must not be silently scored as killed.
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mutation::{run_campaign, OVERALL_TARGET};

    #[test]
    fn soundness_campaign_kills_every_mutant() {
        let report = run_campaign(&SoundnessCampaign, &[]);
        assert!(
            report.total >= 5,
            "campaign must cover the soundness checks"
        );
        assert!(
            report.soundness_critical_survivors().is_empty(),
            "soundness-critical survivors: {:?}",
            report.soundness_critical_survivors()
        );
        assert!(report.passes(OVERALL_TARGET), "score {}", report.score());
    }
}
