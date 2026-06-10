// SPDX-License-Identifier: Apache-2.0
//! Constraint-mutation runner skeleton (layer 8, INV-TEST-05).
//!
//! Mutation testing measures whether the *constraint suite* is strong enough to
//! be sound (`specs.md §15`): perturb one
//! constraint at a time and confirm the reject suite notices. A mutant the suite
//! still accepts is a **surviving mutant** — a soundness gap.
//!
//! This module provides the runner and its types; components register a
//! [`MutationCampaign`] as their constraints land. The runner computes
//! `score = killed / (killed + surviving)` and applies the gate: overall score
//! at least [`OVERALL_TARGET`], with **zero** survivors in the
//! [`SOUNDNESS_CRITICAL`] components.

use std::collections::BTreeSet;

/// A single-constraint mutation operator. Exactly one is applied per run
/// (`specs.md §15`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MutationOperator {
    /// Remove an addend from a constraint polynomial.
    DropTerm,
    /// Change a range bound `hi` to `hi + 1` (or `lo` to `lo - 1`).
    OffByOneBound,
    /// Weaken a comparison, e.g. drop the `- 1` from a strict tie-break.
    RelaxComparison,
    /// Wire a constraint to the wrong trace column.
    SwapColumn,
    /// Delete an entire constraint row.
    DropConstraint,
    /// Invert an `is_first` / `is_last` / mask selector.
    FlipSelector,
    /// Drop a lookup relation or its multiplicity check.
    WeakenLookup,
}

/// A constraint mutant: one operator applied to one constraint of one component.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mutant {
    /// Stable identifier, unique within a campaign (used to match the baseline).
    pub id: String,
    /// The component the constraint belongs to (`pwm-core` / `pwm-prover` /
    /// `pwm-verifier`), e.g. `"freivalds"`, `"requant"`, `"argmin"`.
    pub component: String,
    /// The operator applied.
    pub operator: MutationOperator,
    /// Human-readable description of the perturbation.
    pub description: String,
}

/// Components held to a 100%-kill bar: a survivor in any of these is a direct
/// unsoundness (`specs.md §15`, INV-TEST-05).
pub const SOUNDNESS_CRITICAL: &[&str] = &[
    "requant",
    "range_check",
    "argmin",
    "tensor_memory",
    "activation_lookup",
];

/// Minimum overall mutation score required to merge (INV-TEST-05). Below 1.0
/// because some mutants are provably equivalent and cannot be killed.
pub const OVERALL_TARGET: f64 = 0.90;

/// A reviewed, justified survivor (e.g. a provably equivalent mutant) recorded in
/// the mutation baseline. Baseline survivors count as killed for scoring but must
/// carry a justification reviewed on every change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaselineSurvivor {
    /// The [`Mutant::id`] this entry excuses.
    pub id: String,
    /// Why this survivor is acceptable (e.g. "equivalent: commutative addends").
    pub justification: String,
}

/// A mutation campaign for one or more components. Implemented by each component
/// (or a composite) as its constraints land.
pub trait MutationCampaign {
    /// Enumerate every single-constraint mutant this campaign covers.
    fn mutants(&self) -> Vec<Mutant>;

    /// Decide whether `mutant` is killed: true iff applying it makes at least one
    /// rejecting test fail (`specs.md §15`).
    fn is_killed(&self, mutant: &Mutant) -> bool;
}

/// The result of a mutation sweep.
#[derive(Debug, Clone)]
pub struct MutationReport {
    /// Total mutants evaluated.
    pub total: usize,
    /// Mutants killed by the reject suite (plus justified baseline survivors).
    pub killed: usize,
    /// Mutants that survived and are not excused by the baseline — the gaps.
    pub survivors: Vec<Mutant>,
}

impl MutationReport {
    /// `killed / (killed + surviving)`. An empty sweep scores `1.0` (vacuously
    /// adequate): there are no constraints whose coverage could be missing.
    pub fn score(&self) -> f64 {
        let denom = self.killed + self.survivors.len();
        if denom == 0 {
            1.0
        } else {
            self.killed as f64 / denom as f64
        }
    }

    /// Survivors that fall in a [`SOUNDNESS_CRITICAL`] component — each is a
    /// direct unsoundness and is never tolerated, regardless of overall score.
    pub fn soundness_critical_survivors(&self) -> Vec<&Mutant> {
        self.survivors
            .iter()
            .filter(|m| SOUNDNESS_CRITICAL.contains(&m.component.as_str()))
            .collect()
    }

    /// Apply the merge gate: overall [`MutationReport::score`] at least `target`
    /// **and** no soundness-critical survivors (INV-TEST-05).
    pub fn passes(&self, target: f64) -> bool {
        self.score() >= target && self.soundness_critical_survivors().is_empty()
    }
}

/// Run a campaign against its baseline and produce a [`MutationReport`].
///
/// A mutant is counted as killed if the campaign kills it *or* the baseline
/// excuses it by id; otherwise it is a survivor.
pub fn run_campaign<C: MutationCampaign>(
    campaign: &C,
    baseline: &[BaselineSurvivor],
) -> MutationReport {
    let excused: BTreeSet<&str> = baseline.iter().map(|b| b.id.as_str()).collect();
    let mutants = campaign.mutants();
    let total = mutants.len();
    let mut killed = 0usize;
    let mut survivors = Vec::new();

    for mutant in mutants {
        if campaign.is_killed(&mutant) || excused.contains(mutant.id.as_str()) {
            killed += 1;
        } else {
            survivors.push(mutant);
        }
    }

    MutationReport {
        total,
        killed,
        survivors,
    }
}

/// An empty campaign: no constraints registered yet. The default the runner uses
/// on the skeleton, so the mutation gate is wired into CI and green before any
/// component lands. Replaced/extended as components register real campaigns.
#[derive(Debug, Default, Clone, Copy)]
pub struct EmptyCampaign;

impl MutationCampaign for EmptyCampaign {
    fn mutants(&self) -> Vec<Mutant> {
        Vec::new()
    }

    fn is_killed(&self, _mutant: &Mutant) -> bool {
        // Unreachable on the empty campaign; a real campaign decides per mutant.
        true
    }
}
