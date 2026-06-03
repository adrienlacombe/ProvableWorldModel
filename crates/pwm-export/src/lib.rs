// SPDX-License-Identifier: Apache-2.0
//! `pwm-export` — the export/quantize pipeline (Rust side).
//!
//! Holds the canonical Rust fixed-point reference inference — which must match
//! every `pwm-air` component bit-for-bit — and the Python↔Rust↔golden parity
//! harness. Graph extraction, quantization, and the manifest writer live in the
//! colocated Python subpackage; this Rust side never imports a Python runtime.
//! Export is an offline, trusted preprocessing step that produces a *committed*
//! manifest; neither the prover nor the verifier re-runs it.
//!
//! This crate is `std` (it is host-side tooling) and is a leaf of the
//! dependency DAG: nothing depends on it, which keeps PyTorch out of the
//! verifier (INV-ARCH-02).
//!
//! Workspace skeleton (issue #23). The `reference` and `parity_tests` modules
//! land in #37 and #39 onward.

// DAG edge (docs/spec/01-architecture.md#module-boundaries): depends on pwm-core.
use pwm_core as _;
