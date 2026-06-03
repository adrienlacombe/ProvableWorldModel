// SPDX-License-Identifier: Apache-2.0
#![no_std]
//! `pwm-air` — the NN-specific AIR components.
//!
//! Defines each component's preprocessed/main/interaction trace columns and
//! constraint set: range checks, tensor memory, the weight table, linear,
//! matmul, requantization, activation lookups, layernorm, attention, MLP, the
//! action encoder, the predictor and rollout recurrences, cost, and argmin.
//! Dense operators are expressed directly against Stwo's constraint framework
//! (the efficient path); it calls into `pwm-circuits` for hashing and
//! public-input-binding glue. `no_std`-clean so the verifier can learn each
//! component's verification routine on the `no_std` path.
//!
//! Workspace skeleton (issue #23). The `components/*`, `proof`, and `verifier`
//! modules land in their per-component issues (#42 onward).

// DAG edges (docs/spec/01-architecture.md#module-boundaries).
use pwm_circuits as _;
use pwm_core as _;
