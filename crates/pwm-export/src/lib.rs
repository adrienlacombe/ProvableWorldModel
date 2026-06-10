// SPDX-License-Identifier: Apache-2.0
//! `pwm-export` — the export/quantize pipeline (Rust side).
//!
//! Holds the canonical Rust **integer reference inference** — which the prover's
//! trace builder replays and the verifier exactly recomputes (specs.md §3, §8) —
//! and the Python↔Rust↔golden parity harness. Graph extraction, quantization,
//! BatchNorm folding, the manifest writer, and the stable-worldmodel data adapter
//! live in the colocated Python subpackage; this Rust side never imports a Python
//! runtime. Export is an offline, trusted preprocessing step that produces a
//! *committed* manifest; neither the prover nor the verifier re-runs it.
//!
//! This crate is `std` (host-side tooling). The **verifier trust root does not
//! depend on it**: `pwm-verifier` pulls it in only as a dev-dependency for
//! end-to-end tests, which is what keeps PyTorch and the float reference out of the
//! verifier (INV-ARCH-02). `pwm-prover` and `pwm-testkit` depend on it normally for
//! the integer reference inference.
//!
//! The `reference` module (integer reference inference + trace builder) lands
//! here; quantization, the manifest writer, and the data adapter follow per the
//! backlog (M2).

pub mod manifest;
pub mod predictor_quant;
pub mod reference;
pub mod tables_gen;
