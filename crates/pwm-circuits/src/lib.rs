// SPDX-License-Identifier: Apache-2.0
#![no_std]
//! `pwm-circuits` — low-level scalar gates and witness builder.
//!
//! Provides thin wrappers over the vendored stwo-circuits ten-gate set, the
//! adapter surface to that vendored API, and the witness builder used for
//! commitment hashing, public-input binding, transcript glue, and recursion
//! plumbing. This is the irregular-glue substrate, **not** the dense-matmul hot
//! path (see `docs/spec/01-architecture.md#air-strategy`). `no_std`-clean so it
//! can sit beneath the `no_std` verifier.
//!
//! Workspace skeleton (issue #23). The `low_level_gates`,
//! `adapters_stwo_circuits`, and `witness_builder` modules land once Stwo and
//! stwo-circuits are vendored (#24, #25).

// DAG edge (docs/spec/01-architecture.md#module-boundaries): depends on pwm-core.
use pwm_core as _;
