// SPDX-License-Identifier: Apache-2.0
#![no_std]
//! `pwm-core` — the dependency root of ProvableWorldModel.
//!
//! This crate owns the shared vocabulary every other crate speaks: the value
//! field ([`field::M31`]) and the audit field ([`field::Fp61`]), fixed-point
//! types, tensors, manifest types, canonical serialization, the Merkle
//! commitments, the native Fiat-Shamir [`transcript`], the [`freivalds`] check,
//! the execution-[`trace`] model, `relation_id`, and the observability schema.
//!
//! It carries **no proving substrate** (post-pivot the Stwo dependency was
//! removed; the fields are native) and compiles under `no_std`, so it is
//! linkable into the `no_std`, float-free verifier path (INV-ARCH-01; specs.md
//! §11).
//!
//! Heap-allocating types use [`alloc`]; the crate does not require `std`.

extern crate alloc;

pub mod audit;
pub mod commit;
pub mod field;
pub mod fixed_point;
pub mod freivalds;
pub mod graph;
pub mod limb;
pub mod manifest;
pub mod obs;
pub mod planning;
pub mod public_input;
pub mod relation;
pub mod serialize;
pub mod tables;
pub mod tensor;
pub mod trace;
pub mod transcript;

pub use fixed_point::BoundedInt;
pub use public_input::PublicInput;
pub use relation::StatementType;
pub use tensor::Tensor;
