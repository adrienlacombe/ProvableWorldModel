// SPDX-License-Identifier: Apache-2.0
#![no_std]
//! `pwm-verifier` — the trust anchor.
//!
//! Verifies a `ProofArtifact`: parses and digests the public input, rejects
//! unsupported or mismatched `relation_id`s, recomputes the public-input digest
//! and preprocessed-trace commitments from the committed manifest/config,
//! verifies the Stwo proof, and enforces application-level checks. It runs no
//! model.
//!
//! The crate is `no_std`-clean and depends on **neither `pwm-export` nor any
//! PyTorch/Python runtime** (INV-ARCH-02): the audit surface for accepting a
//! proof is `pwm-core` + `pwm-air` verification routines + the vendored Stwo
//! verifier + `pwm-circuits`, and nothing more.
//!
//! Workspace skeleton (issue #23). The `verify()` entry point, `public_input`
//! digest recomputation, and `recursive` aggregation land in #63 and later.

// DAG edges (docs/spec/01-architecture.md#module-boundaries). The verifier does
// not — and must never — depend on pwm-export (INV-ARCH-02).
use pwm_air as _;
use pwm_circuits as _;
use pwm_core as _;
