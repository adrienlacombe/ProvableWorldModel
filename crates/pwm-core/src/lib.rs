// SPDX-License-Identifier: Apache-2.0
#![no_std]
//! `pwm-core` — the dependency root of ProvableWorldModel.
//!
//! This crate owns the shared vocabulary every other crate speaks: field and
//! fixed-point types, tensors, manifest types, canonical serialization, the
//! Fiat-Shamir transcript, and `relation_id`. It carries **no proving
//! dependencies** (no Stwo prover, FRI, PCS, channel, or constraint framework)
//! and compiles under `no_std`, so it is linkable into the `no_std` verifier
//! path. See `docs/spec/01-architecture.md` (INV-ARCH-01).
//!
//! This is the workspace skeleton (issue #23). The modules listed in the
//! architecture (`field`, `fixed_point`, `tensor`, `manifest`, `transcript`,
//! `serialize`, `relation_id`) land in their own issues (#27 onward).
