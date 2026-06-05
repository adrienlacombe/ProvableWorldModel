// SPDX-License-Identifier: Apache-2.0
#![no_std]
//! `pwm-verifier` — the trust anchor (commit-and-audit, CPU, `no_std`, float-free).
//!
//! Verifies an `AuditArtifact` (specs.md §6, Phase 3):
//!
//! 1. recompute the public-input digest; check `relation_id` is supported and matches;
//! 2. check `model` / `quantization` / `planner` commitments and the `trace_root`
//!    against the opened leaves (Merkle proofs);
//! 3. replay the Fiat-Shamir [`pwm_core::transcript`] to re-derive the Freivalds
//!    challenges and the audit selection;
//! 4. **Freivalds-check** every fixed-weight matmul (`v·x == r·z`) via
//!    [`pwm_core::freivalds`];
//! 5. **exactly recompute** every requant / elementwise / LayerNorm / softmax /
//!    activation-table / attention-inner-product op from committed inputs;
//! 6. check the rollout recurrence, the MSE cost, and the argmin selection.
//!
//! It runs no model and contains no floating point. The audit surface for
//! accepting a proof is `pwm-core` (fields, fixed-point, commitments, transcript,
//! freivalds, trace) and this crate — nothing more. It depends on **neither
//! `pwm-export` nor any Python/PyTorch runtime** (INV-ARCH-02). Unsupported
//! semantics fail closed.
//!
//! The `verify()` entry point, public-input digest recomputation, the Freivalds
//! driver, the exact-replay checks, and the rollout/cost/argmin checks land per
//! the backlog (M4–M6).

// DAG edge (specs.md §11.1): the verifier depends only on pwm-core, and must
// never depend on pwm-export (INV-ARCH-02).
use pwm_core as _;
