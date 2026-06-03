// SPDX-License-Identifier: Apache-2.0
//! The public input and its canonical serialization (data-model `#public-input`,
//! RFC-0014 §1).
//!
//! `PublicInput` carries every public field a proof binds to: the relation id,
//! the model/quantization/planner commitments, the statement discriminant, the
//! input roles (each either public-in-the-clear or commitment-only, INV-DM-13),
//! the claimed-output commitment, and the planner selection. Its canonical bytes
//! feed the public-input digest (`crate::transcript::public_input_digest`).

use alloc::vec::Vec;

use crate::field::M31;
use crate::fixed_point::BoundedInt;
use crate::relation::StatementType;
use crate::serialize::{CanonicalDecode, CanonicalEncode, DecodeError, Reader};

/// The public input bound by a proof (data-model `#public-input`). Field order is
/// the canonical serialization schedule (RFC-0014 §1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicInput {
    /// Hash of the manifest's `relation_id` string.
    pub relation_id: [u8; 32],
    /// Binds architecture, weights, shapes, relation/serialization version.
    pub model_commitment: [u8; 32],
    /// Binds scales, rounding, overflow/clamp policy, tables, approximations.
    pub quantization_commitment: [u8; 32],
    /// Binds planner kind, cost, tie-break, horizon, action block, CEM params.
    pub planner_config_commitment: [u8; 32],
    /// Discriminates P0/P1/P2/P3/P4.
    pub statement_type: StatementType,
    /// Latent-history commitment when private (`None` when public).
    pub latent_history_commitment: Option<[u8; 32]>,
    /// Latent history in the clear when public.
    pub latent_history_public: Option<Vec<M31>>,
    /// Goal-latent commitment when private.
    pub goal_latent_commitment: Option<[u8; 32]>,
    /// Goal latent in the clear when public.
    pub goal_latent_public: Option<Vec<M31>>,
    /// Candidate-actions commitment when private.
    pub candidate_actions_commitment: Option<[u8; 32]>,
    /// Candidate actions in the clear when public.
    pub candidate_actions_public: Option<Vec<M31>>,
    /// Commitment to the claimed outputs.
    pub claimed_output_commitment: [u8; 32],
    /// The chosen candidate index (P2/P3).
    pub selected_index: Option<u32>,
    /// The cost of the selected candidate (P2/P3).
    pub selected_cost: Option<BoundedInt>,
}

impl CanonicalEncode for PublicInput {
    fn encode(&self, out: &mut Vec<u8>) {
        // Declaration order is the schedule (RFC-0014 §1); no field names on wire.
        self.relation_id.encode(out);
        self.model_commitment.encode(out);
        self.quantization_commitment.encode(out);
        self.planner_config_commitment.encode(out);
        self.statement_type.encode(out);
        self.latent_history_commitment.encode(out);
        self.latent_history_public.encode(out);
        self.goal_latent_commitment.encode(out);
        self.goal_latent_public.encode(out);
        self.candidate_actions_commitment.encode(out);
        self.candidate_actions_public.encode(out);
        self.claimed_output_commitment.encode(out);
        self.selected_index.encode(out);
        self.selected_cost.encode(out);
    }
}

impl CanonicalDecode for PublicInput {
    fn decode(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        Ok(PublicInput {
            relation_id: <[u8; 32]>::decode(reader)?,
            model_commitment: <[u8; 32]>::decode(reader)?,
            quantization_commitment: <[u8; 32]>::decode(reader)?,
            planner_config_commitment: <[u8; 32]>::decode(reader)?,
            statement_type: StatementType::decode(reader)?,
            latent_history_commitment: Option::<[u8; 32]>::decode(reader)?,
            latent_history_public: Option::<Vec<M31>>::decode(reader)?,
            goal_latent_commitment: Option::<[u8; 32]>::decode(reader)?,
            goal_latent_public: Option::<Vec<M31>>::decode(reader)?,
            candidate_actions_commitment: Option::<[u8; 32]>::decode(reader)?,
            candidate_actions_public: Option::<Vec<M31>>::decode(reader)?,
            claimed_output_commitment: <[u8; 32]>::decode(reader)?,
            selected_index: Option::<u32>::decode(reader)?,
            selected_cost: Option::<BoundedInt>::decode(reader)?,
        })
    }
}
