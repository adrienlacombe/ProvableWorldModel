// SPDX-License-Identifier: Apache-2.0
//! Fixed-candidate planning checks: goal-latent MSE cost and argmin selection
//! (specs.md §10; the P2 = V0 statement).
//!
//! These are the soundness-critical planning primitives. [`mse_cost`] is the exact
//! integer goal cost; [`verify_argmin`] enforces that the selected candidate has
//! minimum cost under deterministic smallest-index tie-breaking, and that **every**
//! candidate is scored (proving only the winner is unsound — NG9). Pure integer,
//! `no_std`.

use alloc::vec::Vec;

use crate::predictor::round_div;
use crate::tables::ActivationTable;
use crate::transcript::blake2s256;

/// Exact goal-latent cost `Σ_j (pred[j] − goal[j])²` over integers, or `None` if
/// the exact cost does not fit `i64`. Inputs must have equal length (the caller
/// guarantees shape via the trace/graph).
///
/// The sum is accumulated in **checked `i128`**: at the project's own latent size
/// (192-dim) with M31-range residuals a single squared term reaches `~2^62` and the
/// `i64` sum would overflow, letting the argmin select over wrapped costs. Checked
/// `i128` is overflow-safe even for an out-of-envelope goal, and `None` (returned
/// when the exact cost exceeds `i64`) signals rejection rather than a wrapped value
/// (OverflowPolicy::Reject — fail closed).
pub fn mse_cost(pred: &[i64], goal: &[i64]) -> Option<i64> {
    let mut acc: i128 = 0;
    for (&p, &g) in pred.iter().zip(goal.iter()) {
        let d = (p as i128) - (g as i128);
        acc = acc.checked_add(d.checked_mul(d)?)?;
    }
    i64::try_from(acc).ok()
}

/// Why an argmin selection was rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArgminError {
    /// `selected_index` is out of range for the candidate set.
    IndexOutOfRange {
        /// The offending index.
        index: usize,
        /// Number of candidates.
        count: usize,
    },
    /// `selected_cost` does not equal `costs[selected_index]`.
    SelectedCostMismatch,
    /// A candidate has strictly lower cost than the selected one.
    NotMinimum {
        /// Index of the cheaper candidate.
        index: usize,
    },
    /// An earlier candidate ties the selected cost (smallest-index tie-break
    /// requires the *first* minimum to be selected).
    TieBreakViolation {
        /// Index of the earlier tying candidate.
        index: usize,
    },
    /// The candidate set is empty.
    Empty,
}

/// Verify that `selected_index` is the minimum-cost candidate under deterministic
/// smallest-index tie-breaking, and that `selected_cost == costs[selected_index]`.
///
/// All candidates are scored (the full `costs` slice is checked); a lower
/// unselected cost, a wrong index, a mismatched selected cost, or an earlier tie
/// are each rejected (specs.md §10).
pub fn verify_argmin(
    costs: &[i64],
    selected_index: usize,
    selected_cost: i64,
) -> Result<(), ArgminError> {
    if costs.is_empty() {
        return Err(ArgminError::Empty);
    }
    if selected_index >= costs.len() {
        return Err(ArgminError::IndexOutOfRange {
            index: selected_index,
            count: costs.len(),
        });
    }
    if costs[selected_index] != selected_cost {
        return Err(ArgminError::SelectedCostMismatch);
    }
    for (i, &c) in costs.iter().enumerate() {
        if c < selected_cost {
            return Err(ArgminError::NotMinimum { index: i });
        }
        if i < selected_index && c == selected_cost {
            return Err(ArgminError::TieBreakViolation { index: i });
        }
    }
    Ok(())
}

/// The honest argmin (smallest index achieving the minimum cost) over a non-empty
/// `costs` slice — the prover's selection rule, mirrored by [`verify_argmin`].
pub fn argmin(costs: &[i64]) -> Option<(usize, i64)> {
    let mut best: Option<(usize, i64)> = None;
    for (i, &c) in costs.iter().enumerate() {
        match best {
            Some((_, bc)) if c >= bc => {}
            _ => best = Some((i, c)),
        }
    }
    best
}

// ---------------------------------------------------------------------------
// CEM planner step (P3 verifiable core; backlog D-803)
// ---------------------------------------------------------------------------

/// Why a top-k elite selection was rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopkError {
    /// `selected.len() != k`.
    WrongCount,
    /// A selected index is out of range.
    IndexOutOfRange,
    /// A selected index appears more than once.
    Duplicate,
    /// The selected set is not the `k` lowest-cost candidates under (cost, index)
    /// ordering.
    NotElite,
}

/// The honest elite set: the indices of the `k` lowest-cost candidates, ordered by
/// `(cost, index)` (the CEM selection rule), mirrored by [`verify_topk`].
pub fn topk(costs: &[i64], k: usize) -> Vec<u32> {
    let mut order: Vec<usize> = (0..costs.len()).collect();
    order.sort_by(|&a, &b| costs[a].cmp(&costs[b]).then(a.cmp(&b)));
    order.into_iter().take(k).map(|i| i as u32).collect()
}

/// Verify that `selected` is exactly the elite set of the `k` lowest-cost
/// candidates under deterministic `(cost, index)` tie-breaking (the CEM
/// elite-selection step). Every candidate is scored (the full `costs` slice is
/// used); selecting a non-elite candidate is rejected.
pub fn verify_topk(costs: &[i64], selected: &[u32], k: usize) -> Result<(), TopkError> {
    if selected.len() != k {
        return Err(TopkError::WrongCount);
    }
    let mut seen = alloc::collections::BTreeSet::new();
    for &s in selected {
        if (s as usize) >= costs.len() {
            return Err(TopkError::IndexOutOfRange);
        }
        if !seen.insert(s) {
            return Err(TopkError::Duplicate);
        }
    }
    let elite: alloc::collections::BTreeSet<u32> = topk(costs, k).into_iter().collect();
    if seen != elite {
        return Err(TopkError::NotElite);
    }
    Ok(())
}

/// CEM distribution update — per-dimension mean of the elite action vectors
/// (`rows`), rounded to nearest. Each row is one elite candidate's action vector;
/// all rows must share a length.
pub fn cem_mean(rows: &[alloc::vec::Vec<i64>]) -> alloc::vec::Vec<i64> {
    if rows.is_empty() {
        return alloc::vec::Vec::new();
    }
    let dim = rows[0].len();
    let n = rows.len() as i64;
    (0..dim)
        .map(|j| round_div(rows.iter().map(|r| r[j]).sum::<i64>(), n))
        .collect()
}

/// CEM distribution update — per-dimension variance of the elite action vectors
/// about `mean` (population variance, rounded to nearest).
pub fn cem_var(rows: &[alloc::vec::Vec<i64>], mean: &[i64]) -> alloc::vec::Vec<i64> {
    if rows.is_empty() {
        return alloc::vec::Vec::new();
    }
    let dim = rows[0].len();
    let n = rows.len() as i64;
    (0..dim)
        .map(|j| {
            let s: i64 = rows
                .iter()
                .map(|r| (r[j] - mean[j]) * (r[j] - mean[j]))
                .sum();
            round_div(s, n)
        })
        .collect()
}

/// Deterministic CEM candidate generation from a seed (backlog D-803, the
/// sampling side). For candidate `j`, dimension `k`:
/// `candidate[j][k] = mean[k] + round(z · std[k] / 2^std_shift)`, where
/// `z = table[ blake2s(seed‖j‖k) mod |table| ]` and `table` is the committed
/// standard-normal lookup. Deterministic and exactly replayable, so the verifier
/// reproduces the candidate set from the public seed + committed table — closing
/// the loop with the [`verify_topk`] / [`cem_mean`] update step.
pub fn cem_sample(
    seed: &[u8; 32],
    mean: &[i64],
    std: &[i64],
    num_samples: usize,
    table: &ActivationTable,
    std_shift: u32,
) -> Vec<Vec<i64>> {
    let dim = mean.len();
    let domain = (table.outputs.len() as u64).max(1);
    let divisor = 1i64 << std_shift;
    let mut out = Vec::with_capacity(num_samples);
    for j in 0..num_samples {
        let mut row = Vec::with_capacity(dim);
        for k in 0..dim {
            let mut buf = Vec::with_capacity(48);
            buf.extend_from_slice(seed);
            buf.extend_from_slice(&(j as u64).to_le_bytes());
            buf.extend_from_slice(&(k as u64).to_le_bytes());
            let h = blake2s256(&buf);
            let mut b8 = [0u8; 8];
            b8.copy_from_slice(&h[..8]);
            let idx = table.lo + (u64::from_le_bytes(b8) % domain) as i64;
            let z = table.eval(idx).unwrap_or(0);
            row.push(mean[k] + round_div(z * std[k], divisor));
        }
        out.push(row);
    }
    out
}

/// Verify a claimed CEM candidate set was generated by [`cem_sample`] from the
/// public seed, mean, std, and committed table (exact replay).
pub fn verify_cem_sample(
    seed: &[u8; 32],
    mean: &[i64],
    std: &[i64],
    claimed: &[Vec<i64>],
    table: &ActivationTable,
    std_shift: u32,
) -> bool {
    cem_sample(seed, mean, std, claimed.len(), table, std_shift) == claimed
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn mse_is_sum_of_squared_diffs() {
        assert_eq!(mse_cost(&[1, 2, 3], &[1, 0, 0]), Some(13)); // 0 + 4 + 9
        assert_eq!(mse_cost(&[5], &[2]), Some(9));
    }

    #[test]
    fn mse_cost_overflow_is_rejected_not_wrapped() {
        // Near-M31 residuals over a realistic latent dim exceed i64: the exact
        // cost must signal rejection (None), never a wrapped value (WQ-03).
        let big = (1i64 << 30) - 1; // P_HALF, the M31 envelope
        let pred = vec![big; 192];
        let goal = vec![-big; 192];
        assert_eq!(mse_cost(&pred, &goal), None);
        // A single near-max residual still fits i64 and computes exactly.
        let d = 2 * big; // (2^31 - 2)^2 ≈ 2^62 < i64::MAX
        assert_eq!(mse_cost(&[big], &[-big]), Some(d * d));
    }

    #[test]
    fn argmin_picks_first_minimum() {
        assert_eq!(argmin(&[3, 1, 1, 4]), Some((1, 1)));
        assert_eq!(argmin(&[7]), Some((0, 7)));
        assert_eq!(argmin(&[]), None);
    }

    #[test]
    fn verify_argmin_accepts_correct_selection() {
        assert_eq!(verify_argmin(&[3, 1, 4], 1, 1), Ok(()));
        // First minimum on a tie.
        assert_eq!(verify_argmin(&[2, 5, 2], 0, 2), Ok(()));
    }

    #[test]
    fn topk_selects_and_verifies_elite_set() {
        let costs = [5, 1, 3, 1, 9];
        // The 3 lowest by (cost,index): index1(1), index3(1), index2(3).
        assert_eq!(topk(&costs, 3), vec![1, 3, 2]);
        assert_eq!(verify_topk(&costs, &[1, 3, 2], 3), Ok(()));
        assert_eq!(verify_topk(&costs, &[2, 1, 3], 3), Ok(())); // order-insensitive set
                                                                // A non-elite member (index 0, cost 5) is rejected.
        assert_eq!(verify_topk(&costs, &[1, 3, 0], 3), Err(TopkError::NotElite));
        assert_eq!(verify_topk(&costs, &[1, 3], 3), Err(TopkError::WrongCount));
        assert_eq!(
            verify_topk(&costs, &[1, 1, 3], 3),
            Err(TopkError::Duplicate)
        );
    }

    #[test]
    fn cem_sample_is_deterministic_and_replayable() {
        let seed = [3u8; 32];
        let mean = [10i64, 20];
        let std = [1i64, 1];
        // Standard-normal-ish table over domain [0,3] -> z in {-2,-1,1,2}.
        let table = ActivationTable {
            table_id: 9,
            lo: 0,
            outputs: alloc::vec![-2, -1, 1, 2],
        };
        let cands = cem_sample(&seed, &mean, &std, 4, &table, 0);
        assert_eq!(cands.len(), 4);
        for row in &cands {
            // Each value is mean +/- a small z (std=1, shift=0): within [mean-2, mean+2].
            assert!(row[0] >= 8 && row[0] <= 12);
            assert!(row[1] >= 18 && row[1] <= 22);
        }
        // Replayable: the verifier reproduces the exact candidate set.
        assert!(verify_cem_sample(&seed, &mean, &std, &cands, &table, 0));
        // A tampered candidate is rejected.
        let mut bad = cands.clone();
        bad[1][0] += 1;
        assert!(!verify_cem_sample(&seed, &mean, &std, &bad, &table, 0));
        // A different seed gives a different set (overwhelmingly).
        let other = cem_sample(&[4u8; 32], &mean, &std, 4, &table, 0);
        assert_ne!(other, cands);
    }

    #[test]
    fn cem_mean_and_var_update() {
        let elite = alloc::vec![
            alloc::vec![2i64, 10],
            alloc::vec![4, 20],
            alloc::vec![6, 30]
        ];
        let mean = cem_mean(&elite);
        assert_eq!(mean, alloc::vec![4, 20]); // (2+4+6)/3=4, (10+20+30)/3=20
        let var = cem_var(&elite, &mean);
        // dim0: ((−2)²+0+2²)/3 = 8/3 -> round 3; dim1: (100+0+100)/3=200/3 -> 67.
        assert_eq!(var, alloc::vec![3, 67]);
    }

    #[test]
    fn verify_argmin_rejects_unsound_selections() {
        // A cheaper candidate exists.
        assert!(matches!(
            verify_argmin(&[3, 1, 4], 0, 3),
            Err(ArgminError::NotMinimum { .. })
        ));
        // Wrong selected cost.
        assert_eq!(
            verify_argmin(&[3, 1, 4], 1, 2),
            Err(ArgminError::SelectedCostMismatch)
        );
        // Index out of range.
        assert!(matches!(
            verify_argmin(&[3, 1], 5, 1),
            Err(ArgminError::IndexOutOfRange { .. })
        ));
        // An earlier candidate ties -> must pick the earlier one.
        assert!(matches!(
            verify_argmin(&[2, 5, 2], 2, 2),
            Err(ArgminError::TieBreakViolation { .. })
        ));
    }
}
