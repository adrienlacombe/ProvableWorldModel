// SPDX-License-Identifier: Apache-2.0
//! Tests for the accept/reject harness itself: a toy subject demonstrates that
//! accepting cases pass, reject cases must hit the exact error, and a suite with
//! no reject cases is itself a failure (INV-TEST-01).

use pwm_testkit::accept_reject::{run_suite, AcceptCase, RejectCase, Subject};

/// A toy subject: accepts an even value, rejects an odd one with a typed error.
struct Parity;

#[derive(Debug, PartialEq)]
enum ParityReject {
    Odd,
    Negative,
}

impl Subject for Parity {
    type Witness = i64;
    type Reject = ParityReject;

    fn check(&self, witness: &i64) -> Result<(), ParityReject> {
        if *witness < 0 {
            Err(ParityReject::Negative)
        } else if witness % 2 != 0 {
            Err(ParityReject::Odd)
        } else {
            Ok(())
        }
    }
}

#[test]
fn full_suite_passes_with_accept_and_reject_cases() {
    let accepts = vec![AcceptCase::new("zero", 0), AcceptCase::new("four", 4)];
    let rejects = vec![
        RejectCase::new("three_is_odd", 3, ParityReject::Odd),
        RejectCase::new("neg_two", -2, ParityReject::Negative),
    ];
    run_suite(&Parity, &accepts, &rejects).expect("suite passes");
}

#[test]
fn suite_without_reject_cases_fails_dual_test_rule() {
    let accepts = vec![AcceptCase::new("zero", 0)];
    let rejects: Vec<RejectCase<i64, ParityReject>> = vec![];
    let failures = run_suite(&Parity, &accepts, &rejects).expect_err("must fail");
    assert!(failures.iter().any(|m| m.contains("INV-TEST-01")));
}

#[test]
fn reject_case_with_wrong_expected_error_fails() {
    // 3 is rejected as Odd, but the case claims Negative — that is a test bug and
    // must surface, not silently pass.
    let rejects = vec![RejectCase::new("mislabeled", 3, ParityReject::Negative)];
    let failures = run_suite(&Parity, &[], &rejects).expect_err("must fail");
    assert!(failures.iter().any(|m| m.contains("expected Negative")));
}

#[test]
fn accept_case_that_is_rejected_fails() {
    let accepts = vec![AcceptCase::new("three_should_not_verify", 3)];
    // Provide a reject case so the failure is the accept case, not the dual-test rule.
    let rejects = vec![RejectCase::new("five", 5, ParityReject::Odd)];
    let failures = run_suite(&Parity, &accepts, &rejects).expect_err("must fail");
    assert!(failures
        .iter()
        .any(|m| m.contains("three_should_not_verify")));
}
