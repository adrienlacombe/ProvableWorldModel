// SPDX-License-Identifier: Apache-2.0
//! Statement discriminant shared across the corpus.
//!
//! `StatementType` is the V0..V3 statement tier selector that appears in the
//! public input and in observability records. It is a small, stable, dependency-
//! free enum so the prover, verifier, and observability layer agree on the
//! statement vocabulary without sharing higher-level code. The full P0–P4 tier
//! definitions live in `specs.md §1`; the
//! relation-id registry and public-input binding are owned by later issues
//! (#33, #63).

/// The statement a proof attests to. The discriminant in `PublicInput` that
/// selects which AIR components must be present and verified
/// (`specs.md §11`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StatementType {
    /// P0: one predictor step (`z_next = PredProj(ARPredictor(...))`).
    P0Step,
    /// P1: a configurable-horizon autoregressive rollout.
    P1Rollout,
    /// P2: fixed-candidate planning — the V0 headline deliverable.
    P2FixedCandidatePlanning,
    /// P3: CEM planning (Future, out of V0 scope).
    P3Cem,
    /// P4: pixel-to-plan with the encoder (Future, out of V0 scope).
    P4PixelToPlan,
}

impl StatementType {
    /// The canonical, stable label used in serialized records (matches the
    /// variant name in `specs.md §11`).
    pub const fn as_str(self) -> &'static str {
        match self {
            StatementType::P0Step => "P0Step",
            StatementType::P1Rollout => "P1Rollout",
            StatementType::P2FixedCandidatePlanning => "P2FixedCandidatePlanning",
            StatementType::P3Cem => "P3Cem",
            StatementType::P4PixelToPlan => "P4PixelToPlan",
        }
    }

    /// The immutable canonical-serialization discriminant (RFC-0014 §1).
    /// Renumbering is a breaking change gated by `relation_id`.
    pub const fn discriminant(self) -> u8 {
        match self {
            StatementType::P0Step => 0,
            StatementType::P1Rollout => 1,
            StatementType::P2FixedCandidatePlanning => 2,
            StatementType::P3Cem => 3,
            StatementType::P4PixelToPlan => 4,
        }
    }

    /// Inverse of [`StatementType::discriminant`]; `None` for an unknown value.
    pub const fn from_discriminant(value: u8) -> Option<Self> {
        match value {
            0 => Some(StatementType::P0Step),
            1 => Some(StatementType::P1Rollout),
            2 => Some(StatementType::P2FixedCandidatePlanning),
            3 => Some(StatementType::P3Cem),
            4 => Some(StatementType::P4PixelToPlan),
            _ => None,
        }
    }
}
