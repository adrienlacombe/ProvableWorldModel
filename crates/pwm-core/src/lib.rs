// SPDX-License-Identifier: Apache-2.0
#![no_std]
//! `pwm-core` — the dependency root of ProvableWorldModel.
//!
//! This crate owns the shared vocabulary every other crate speaks: field and
//! fixed-point types, tensors, manifest types, canonical serialization, the
//! Fiat-Shamir transcript, `relation_id`, and the observability schema. It
//! carries **no proving dependencies** (no Stwo prover, FRI, PCS, channel, or
//! constraint framework) and compiles under `no_std`, so it is linkable into the
//! `no_std` verifier path. See `docs/spec/01-architecture.md` (INV-ARCH-01).
//!
//! Heap-allocating types use [`alloc`]; the crate does not require `std`.
//!
//! The remaining modules listed in the architecture (`field`, `fixed_point`,
//! `tensor`, `manifest`, `transcript`, `serialize`) land in their own issues
//! (#27 onward).

extern crate alloc;

pub mod field;
pub mod fixed_point;
pub mod limb;
pub mod obs;
pub mod public_input;
pub mod relation;
pub mod serialize;
pub mod tensor;
pub mod transcript;

pub use fixed_point::BoundedInt;
pub use public_input::PublicInput;
pub use relation::StatementType;
pub use tensor::Tensor;
