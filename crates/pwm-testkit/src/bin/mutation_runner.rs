// SPDX-License-Identifier: Apache-2.0
//! Constraint-mutation runner entry point (CI job, layer 8).
//!
//! Runs every registered [`MutationCampaign`](pwm_testkit::mutation::MutationCampaign),
//! applies the INV-TEST-05 gate, and exits non-zero if the suite is inadequate.
//! On the skeleton the only registered campaign is [`EmptyCampaign`] (no
//! constraints exist yet), so the runner is wired into CI and green; components
//! register real campaigns as they land.

use std::process::ExitCode;

use pwm_testkit::mutation::{
    run_campaign, BaselineSurvivor, EmptyCampaign, MutationReport, OVERALL_TARGET,
};

fn main() -> ExitCode {
    // Registered campaigns. Each component appends its own as its constraints
    // land; the empty campaign keeps the gate wired and meaningful until then.
    let baseline: Vec<BaselineSurvivor> = Vec::new();
    let reports: Vec<(&str, MutationReport)> =
        vec![("empty", run_campaign(&EmptyCampaign, &baseline))];

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
