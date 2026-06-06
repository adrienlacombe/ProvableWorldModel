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

/// Exact goal-latent cost `Σ_j (pred[j] − goal[j])²` over integers. Inputs must
/// have equal length (the caller guarantees shape via the trace/graph).
pub fn mse_cost(pred: &[i64], goal: &[i64]) -> i64 {
    pred.iter()
        .zip(goal.iter())
        .map(|(&p, &g)| {
            let d = p - g;
            d * d
        })
        .sum()
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
    use crate::predictor::round_div;
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
    use crate::predictor::round_div;
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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn mse_is_sum_of_squared_diffs() {
        assert_eq!(mse_cost(&[1, 2, 3], &[1, 0, 0]), 13); // 0 + 4 + 9
        assert_eq!(mse_cost(&[5], &[2]), 9);
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
