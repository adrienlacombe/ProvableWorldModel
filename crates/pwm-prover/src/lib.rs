// SPDX-License-Identifier: Apache-2.0
//! `pwm-prover` — prover orchestration and CLI.
//!
//! Loads the manifest, verifies `hash(manifest) == model_commitment` and the
//! weight root *before* building any trace (INV-ARCH-05), runs the Rust
//! fixed-point reference inference, assembles the preprocessed/main/interaction
//! traces, drives the Stwo commit → challenge → interaction-trace → proof flow
//! in the canonical Fiat-Shamir order, and emits the `ProofArtifact` bundle.
//! For P2 it orchestrates the cost and argmin components over *all* candidates
//! (RFC-0009 forbids partial proofs).
//!
//! This crate is `std`; it sits at the top of the dependency DAG and is the
//! only first-party crate to depend on both `pwm-air` and `pwm-export`.
//!
//! Workspace skeleton (issue #23). The `cli`, `trace_builder`, `prove_rollout`,
//! and `prove_planning` modules land in #62, #64, and the per-statement issues.

// DAG edges (docs/spec/01-architecture.md#module-boundaries).
use pwm_air as _;
use pwm_circuits as _;
use pwm_core as _;
use pwm_export as _;
