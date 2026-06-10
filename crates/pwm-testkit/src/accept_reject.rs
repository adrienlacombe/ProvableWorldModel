// SPDX-License-Identifier: Apache-2.0
//! The accept/reject (dual-test) harness.
//!
//! Implements the rule from `specs.md §15`
//! (INV-TEST-01): no component ships without **both** an accepting test (a valid
//! witness verifies) and at least one rejecting test (a witness violating the
//! component's contract is rejected with a *specific* typed error, INV-TEST-09).
//!
//! A component, verifier path, or reference primitive implements [`Subject`];
//! the runners below execute accept and reject cases against it and refuse a
//! suite that has no reject cases at all.

use core::fmt::Debug;

/// A checkable subject under the dual-test rule.
///
/// Implementors are AIR components, verifier paths, or fixed-point reference
/// primitives. [`Subject::check`] is the single decision procedure: it returns
/// `Ok(())` for a valid witness and a *specific* typed `Err` naming the contract
/// it violated. The harness never inspects anything but this result, mirroring
/// the verifier's trust boundary (it decides purely from the witness, never by
/// re-running a float reference; INV-TEST-11).
pub trait Subject {
    /// The witness type this subject checks (e.g. a trace slice, an artifact).
    type Witness;
    /// The typed rejection reason (e.g. a `VerifyError` variant). Compared by
    /// equality so a reject test pins the *exact* error, not a generic failure.
    type Reject: Debug + PartialEq;

    /// Decide a witness: `Ok(())` to accept, `Err(reject)` to reject.
    fn check(&self, witness: &Self::Witness) -> Result<(), Self::Reject>;
}

/// An accepting case: a valid witness the subject must accept (layer 5).
pub struct AcceptCase<W> {
    /// Human-readable case name, surfaced in failure messages.
    pub name: String,
    /// The witness expected to verify.
    pub witness: W,
}

impl<W> AcceptCase<W> {
    /// Construct an accepting case.
    pub fn new(name: impl Into<String>, witness: W) -> Self {
        Self {
            name: name.into(),
            witness,
        }
    }
}

/// A rejecting case: an invalid witness the subject must reject with exactly
/// `expected` (layer 6, INV-TEST-09).
pub struct RejectCase<W, R> {
    /// Human-readable case name, surfaced in failure messages.
    pub name: String,
    /// The tampered/invalid witness.
    pub witness: W,
    /// The exact rejection the subject must produce.
    pub expected: R,
}

impl<W, R> RejectCase<W, R> {
    /// Construct a rejecting case pinned to an exact expected rejection.
    pub fn new(name: impl Into<String>, witness: W, expected: R) -> Self {
        Self {
            name: name.into(),
            witness,
            expected,
        }
    }
}

/// Run one accepting case. `Ok(())` if accepted; otherwise a descriptive error.
pub fn run_accept<S: Subject>(subject: &S, case: &AcceptCase<S::Witness>) -> Result<(), String> {
    match subject.check(&case.witness) {
        Ok(()) => Ok(()),
        Err(got) => Err(format!(
            "accept case '{}' was rejected with {got:?}",
            case.name
        )),
    }
}

/// Run one rejecting case. `Ok(())` only if rejected with exactly the expected
/// error; an accept, or a different error, fails (INV-TEST-09).
pub fn run_reject<S: Subject>(
    subject: &S,
    case: &RejectCase<S::Witness, S::Reject>,
) -> Result<(), String> {
    match subject.check(&case.witness) {
        Ok(()) => Err(format!(
            "reject case '{}' was accepted; expected rejection {:?}",
            case.name, case.expected
        )),
        Err(got) if got == case.expected => Ok(()),
        Err(got) => Err(format!(
            "reject case '{}' was rejected with {got:?}; expected {:?}",
            case.name, case.expected
        )),
    }
}

/// Run a full dual-test suite for a subject.
///
/// Enforces INV-TEST-01/INV-TEST-10 structurally: a suite with **zero** reject
/// cases is itself a failure (`accepting tests alone are not enough`). Returns
/// all case failures so a test reports every gap at once, not just the first.
pub fn run_suite<S: Subject>(
    subject: &S,
    accepts: &[AcceptCase<S::Witness>],
    rejects: &[RejectCase<S::Witness, S::Reject>],
) -> Result<(), Vec<String>> {
    let mut failures = Vec::new();

    if rejects.is_empty() {
        failures.push(
            "dual-test violation (INV-TEST-01): suite has no reject cases; \
             accepting tests alone are insufficient"
                .to_string(),
        );
    }

    for case in accepts {
        if let Err(msg) = run_accept(subject, case) {
            failures.push(msg);
        }
    }
    for case in rejects {
        if let Err(msg) = run_reject(subject, case) {
            failures.push(msg);
        }
    }

    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures)
    }
}

/// Convenience wrapper for use inside `#[test]` functions: runs the suite and
/// panics with a readable report if any case (or the dual-test rule) fails.
pub fn assert_suite<S: Subject>(
    subject: &S,
    accepts: &[AcceptCase<S::Witness>],
    rejects: &[RejectCase<S::Witness, S::Reject>],
) {
    if let Err(failures) = run_suite(subject, accepts, rejects) {
        panic!("dual-test suite failed:\n  - {}", failures.join("\n  - "));
    }
}
