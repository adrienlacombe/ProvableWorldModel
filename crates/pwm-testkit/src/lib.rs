// SPDX-License-Identifier: Apache-2.0
//! `pwm-testkit` — shared test scaffolding for ProvableWorldModel.
//!
//! This crate is test infrastructure (it is never published) implementing the
//! reusable pieces of the test pyramid from
//! `specs.md §15`, so that every component lands its tests
//! against one harness rather than re-inventing them:
//!
//! - [`accept_reject`] — the dual-test harness (INV-TEST-01): a generic
//!   [`accept_reject::Subject`] that accepts valid witnesses and rejects invalid
//!   ones with a *specific* typed error, plus runners that enforce the rule that
//!   no component ships without at least one rejecting test.
//! - [`golden`] — the loader for committed golden-vector fixtures (layers 2/3),
//!   the integer-only canonical-JSON format from
//!   `specs.md §15`.
//! - [`mutation`] — the constraint-mutation runner skeleton (layer 8,
//!   INV-TEST-05): mutation operators, the campaign trait components implement as
//!   their constraints land, the score, and the soundness-critical 100%-kill bar.
//!
//! The harness is deliberately generic over each component's witness and error
//! types, so it bakes in no single component's surface; the prover
//! (`pwm-prover`) and the no_std verifier (`pwm-verifier`) plug their own types in.

pub mod accept_reject;
pub mod bundle;
pub mod demo;
pub mod golden;
pub mod lewm;
pub mod lewm_predictor;
pub mod mutation;
pub mod predictor;
pub mod soundness_campaign;
