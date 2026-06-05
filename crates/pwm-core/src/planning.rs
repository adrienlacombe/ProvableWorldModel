// SPDX-License-Identifier: Apache-2.0
//! Fixed-candidate planning checks: goal-latent MSE cost and argmin selection
//! (specs.md §10; the P2 = V0 statement).
//!
//! These are the soundness-critical planning primitives. [`mse_cost`] is the exact
//! integer goal cost; [`verify_argmin`] enforces that the selected candidate has
//! minimum cost under deterministic smallest-index tie-breaking, and that **every**
//! candidate is scored (proving only the winner is unsound — NG9). Pure integer,
//! `no_std`.

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

#[cfg(test)]
mod tests {
    use super::*;

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
