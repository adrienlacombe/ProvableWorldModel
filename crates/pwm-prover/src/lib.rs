// SPDX-License-Identifier: Apache-2.0
//! `pwm-prover` — commit-and-audit prover orchestration and CLI.
//!
//! The prover runs the model normally; it does not generate a proving circuit.
//! The flow (specs.md §6, Phase 1):
//!
//! 1. verify `hash(manifest) == model_commitment` and the weight Merkle root
//!    *before* any work;
//! 2. canonicalize the public inputs and derive the public-input digest;
//! 3. run the **exact integer reference inference** over the exported graph (the
//!    `pwm-export` Rust reference), recording every accumulator and activation
//!    into the [`pwm_core::trace`] op records;
//! 4. Merkle-commit the trace (`trace_root`);
//! 5. drive the Fiat-Shamir [`pwm_core::transcript`] to squeeze the Freivalds
//!    challenge vectors and the audit selection;
//! 6. emit the `AuditArtifact { commitments, public_input, openings,
//!    claimed_outputs }`.
//!
//! This crate is `std` (host-side tooling) and sits at the top of the dependency
//! DAG; it depends on `pwm-core` and `pwm-export` (for the Rust reference) only —
//! no proving substrate. The `cli`, `trace_builder`, `prove_step`,
//! `prove_rollout`, and `prove_planning` modules land per the backlog (M3–M6).

// DAG edges (specs.md §11.1).
use pwm_core as _;
use pwm_export as _;
