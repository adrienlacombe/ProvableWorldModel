// SPDX-License-Identifier: Apache-2.0
//! Constraint-mutation runner entry point (CI job, layer 8).
//!
//! Runs every registered [`MutationCampaign`](pwm_testkit::mutation::MutationCampaign),
//! applies the INV-TEST-05 gate, and exits non-zero if the suite is inadequate.
//! The [`SoundnessCampaign`](pwm_testkit::soundness_campaign::SoundnessCampaign)
//! exercises the live verifier's soundness checks (range guard, requant/table
//! replay, argmin) against concrete forgeries; components append further campaigns
//! as their constraints land.

use std::process::ExitCode;

use pwm_testkit::mutation::{run_campaign, BaselineSurvivor, MutationReport, OVERALL_TARGET};
use pwm_testkit::soundness_campaign::SoundnessCampaign;

fn main() -> ExitCode {
    // Registered campaigns. Each component appends its own as its constraints land.
    let baseline: Vec<BaselineSurvivor> = Vec::new();
    let reports: Vec<(&str, MutationReport)> =
        vec![("soundness", run_campaign(&SoundnessCampaign, &baseline))];

    let mut all_pass = true;
    for (name, report) in &reports {
        let critical = report.soundness_critical_survivors();
        let pass = report.passes(OVERALL_TARGET);
        all_pass &= pass;
        println!(
            "campaign={name} total={} killed={} survivors={} score={:.4} critical_survivors={} -> {}",
            report.total,
            report.killed,
            report.survivors.len(),
            report.score(),
            critical.len(),
            if pass { "PASS" } else { "FAIL" },
        );
        for mutant in &report.survivors {
            println!(
                "  survivor: id={} component={} op={:?}",
                mutant.id, mutant.component, mutant.operator
            );
        }
    }

    if all_pass {
        println!(
            "mutation gate: PASS (target {OVERALL_TARGET:.2}, no soundness-critical survivors)"
        );
        ExitCode::SUCCESS
    } else {
        eprintln!(
            "mutation gate: FAIL (score below target or soundness-critical survivor present)"
        );
        ExitCode::FAILURE
    }
}
