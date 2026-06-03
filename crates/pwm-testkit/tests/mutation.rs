// SPDX-License-Identifier: Apache-2.0
//! Tests for the mutation runner skeleton: scoring, the soundness-critical
//! 100%-kill bar, and baseline excusal.

use pwm_testkit::mutation::{
    run_campaign, BaselineSurvivor, EmptyCampaign, Mutant, MutationCampaign, MutationOperator,
    OVERALL_TARGET,
};

#[test]
fn empty_campaign_is_vacuously_adequate() {
    let report = run_campaign(&EmptyCampaign, &[]);
    assert_eq!(report.total, 0);
    assert_eq!(report.score(), 1.0);
    assert!(report.passes(OVERALL_TARGET));
}

/// A campaign whose kill decision is dictated by the test, so we can drive
/// survivors deterministically.
struct ScriptedCampaign {
    mutants: Vec<Mutant>,
    killed_ids: Vec<&'static str>,
}

impl MutationCampaign for ScriptedCampaign {
    fn mutants(&self) -> Vec<Mutant> {
        self.mutants.clone()
    }

    fn is_killed(&self, mutant: &Mutant) -> bool {
        self.killed_ids.contains(&mutant.id.as_str())
    }
}

fn mutant(id: &str, component: &str) -> Mutant {
    Mutant {
        id: id.to_string(),
        component: component.to_string(),
        operator: MutationOperator::DropTerm,
        description: format!("drop a term in {component}"),
    }
}

#[test]
fn score_is_killed_over_total_and_gate_uses_target() {
    // 9 killed of 10 in a non-critical component -> score 0.9, exactly the target.
    let mut mutants: Vec<Mutant> = (0..10)
        .map(|i| mutant(&format!("m{i}"), "linear"))
        .collect();
    let killed_ids: Vec<&'static str> = vec!["m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"];
    let campaign = ScriptedCampaign {
        mutants: std::mem::take(&mut mutants),
        killed_ids,
    };

    let report = run_campaign(&campaign, &[]);
    assert_eq!(report.total, 10);
    assert_eq!(report.killed, 9);
    assert_eq!(report.survivors.len(), 1);
    assert!((report.score() - 0.9).abs() < 1e-9);
    assert!(report.passes(OVERALL_TARGET), "0.9 meets the 0.90 target");
}

#[test]
fn any_soundness_critical_survivor_fails_the_gate() {
    // One survivor in `range_check` (soundness-critical) must fail even at score 0.99.
    let mut mutants: Vec<Mutant> = (0..99)
        .map(|i| mutant(&format!("ok{i}"), "linear"))
        .collect();
    mutants.push(mutant("crit", "range_check"));
    let killed_ids: Vec<&'static str> = (0..99)
        .map(|i| Box::leak(format!("ok{i}").into_boxed_str()) as &'static str)
        .collect();
    let campaign = ScriptedCampaign {
        mutants,
        killed_ids,
    };

    let report = run_campaign(&campaign, &[]);
    assert!(report.score() > 0.98);
    assert_eq!(report.soundness_critical_survivors().len(), 1);
    assert!(
        !report.passes(OVERALL_TARGET),
        "a critical survivor blocks merge"
    );
}

#[test]
fn baseline_excuses_a_justified_survivor() {
    let mutants = vec![mutant("equiv", "linear"), mutant("real", "linear")];
    let campaign = ScriptedCampaign {
        mutants,
        killed_ids: vec!["real"],
    };

    // Without baseline: `equiv` survives.
    let bare = run_campaign(&campaign, &[]);
    assert_eq!(bare.survivors.len(), 1);

    // With a justified baseline entry: `equiv` counts as killed.
    let baseline = vec![BaselineSurvivor {
        id: "equiv".to_string(),
        justification: "equivalent: commutative addends".to_string(),
    }];
    let excused = run_campaign(&campaign, &baseline);
    assert_eq!(excused.survivors.len(), 0);
    assert_eq!(excused.killed, 2);
    assert!(excused.passes(OVERALL_TARGET));
}
