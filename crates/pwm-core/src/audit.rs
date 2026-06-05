// SPDX-License-Identifier: Apache-2.0
//! The commit-and-audit artifact and the shared challenge-derivation transcript
//! (specs.md §6, §12).
//!
//! [`AuditArtifact`] is the self-contained object the prover emits and the
//! verifier checks: the public input (which already binds the relation id and the
//! model/quantization/planner/output commitments), the committed weights and
//! activation tables, the scale table, the execution trace, and the claimed
//! output. [`audit_transcript`] is the single shared construction both sides use
//! to derive the Freivalds challenges, so the prover commits the trace *before*
//! any challenge is known (non-interactive Fiat-Shamir, specs.md §7).

use alloc::vec::Vec;

use crate::field::Fp61;
use crate::fixed_point::{BoundError, BoundedInt};
use crate::graph::GraphSpec;
use crate::public_input::PublicInput;
use crate::serialize::CanonicalEncode;
use crate::tables::ActivationTable;
use crate::tensor::{Scale, Tensor, TensorError};
use crate::trace::{trace_root, OpRecord};
use crate::transcript::{init_transcript, Transcript};

/// The current artifact wire-format version.
pub const ARTIFACT_VERSION: u32 = 1;

/// Relation id string for the quantized feed-forward statement (the working
/// vertical slice toward the full predictor relation `pwm.lewm.predictor_step.v1`).
pub const RELATION_MLP: &str = "pwm.lewm.mlp.v1";
/// Relation semantic version bound into the model commitment.
pub const RELATION_VERSION: u32 = 1;
/// Canonical serialization version bound into the model commitment.
pub const SERIALIZATION_VERSION: u32 = 1;

/// The `relation_id` is the Blake2s digest of the relation string.
pub fn relation_id(relation: &str) -> [u8; 32] {
    crate::transcript::blake2s256(relation.as_bytes())
}

/// A self-contained commit-and-audit proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditArtifact {
    /// Wire-format version.
    pub artifact_version: u32,
    /// All public fields the proof binds (relation id, commitments, selection).
    pub public_input: PublicInput,
    /// The static op graph (the committed wiring). Its commitment is the
    /// `architecture_commitment` inside the model commitment.
    pub graph: GraphSpec,
    /// Committed weight and bias tensors (public for V0). Their Merkle root must
    /// equal the model commitment's weight root.
    pub weights: Vec<Tensor>,
    /// Committed activation tables. Their commitment must equal the quantization
    /// commitment's `activation_tables_commitment`.
    pub tables: Vec<ActivationTable>,
    /// The scale table.
    pub scales: Vec<Scale>,
    /// The ordered execution trace (the witness the verifier audits).
    pub trace: Vec<OpRecord>,
    /// The claimed output tensor; its commitment must equal
    /// `public_input.claimed_output_commitment`.
    pub claimed_output: Tensor,
}

impl AuditArtifact {
    /// Recompute the trace Merkle root over this artifact's records.
    pub fn trace_root(&self) -> [u8; 32] {
        trace_root(&self.trace)
    }

    /// Look up a committed weight/bias tensor by id.
    pub fn weight(&self, id: u32) -> Option<&Tensor> {
        self.weights.iter().find(|t| t.tensor_id() == id)
    }

    /// Look up a committed activation table by id.
    pub fn table(&self, id: u32) -> Option<&ActivationTable> {
        self.tables.iter().find(|t| t.table_id == id)
    }
}

impl CanonicalEncode for AuditArtifact {
    fn encode(&self, out: &mut Vec<u8>) {
        self.artifact_version.encode(out);
        self.public_input.encode(out);
        self.graph.encode(out);
        self.weights.encode(out);
        self.tables.encode(out);
        self.scales.encode(out);
        self.trace.encode(out);
        self.claimed_output.encode(out);
    }
}

/// A fixed-candidate planning proof (P2 = V0): a P0 [`AuditArtifact`] per
/// candidate over the same committed model, the public goal latent, the per-
/// candidate goal costs, and the selected candidate. The verifier checks each
/// candidate proof, recomputes each cost from the verified output and the goal,
/// and checks the argmin selection (specs.md §10).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanningProof {
    /// Wire-format version.
    pub artifact_version: u32,
    /// One P0 proof per candidate (same model, different input).
    pub candidates: Vec<AuditArtifact>,
    /// The public goal latent (integer values).
    pub goal: Vec<i64>,
    /// Per-candidate goal-MSE costs (index-aligned with `candidates`).
    pub costs: Vec<i64>,
    /// The selected (minimum-cost) candidate index.
    pub selected_index: u32,
    /// The cost of the selected candidate.
    pub selected_cost: i64,
}

/// An autoregressive rollout proof (P1): one P0 [`AuditArtifact`] per step, where
/// each step predicts the next latent from the trailing `history_size`-window of
/// latents (initial history followed by previously predicted latents). The
/// verifier checks each step's P0 proof and the recurrence wiring — step `t`'s
/// input is exactly the flattened window of latents available at step `t`
/// (specs.md §9).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RolloutProof {
    /// Wire-format version.
    pub artifact_version: u32,
    /// One P0 proof per rollout step.
    pub steps: Vec<AuditArtifact>,
    /// The initial latent history (each entry a latent of dimension `D`).
    pub initial_latents: Vec<Vec<i64>>,
    /// The window size `H` (number of trailing latents fed to each step).
    pub history_size: u32,
    /// The predicted latents, one per step (the rolled-out trajectory).
    pub trajectory: Vec<Vec<i64>>,
}

/// Build the shared Fiat-Shamir transcript bound to the public input and the
/// trace root. The public-input digest already binds the relation id and the
/// model/quantization/planner/output commitments (they are fields of
/// [`PublicInput`]); absorbing `trace_root` binds the committed accumulators so a
/// later-derived Freivalds challenge cannot be anticipated (specs.md §6 Phase 1).
pub fn audit_transcript(public_input: &PublicInput, trace_root: &[u8; 32]) -> Transcript {
    let mut t = init_transcript(public_input);
    t.absorb(b"trace_root", trace_root);
    t
}

/// Squeeze the next Freivalds challenge vector of length `rows` from the audit
/// transcript. Called once per `Linear` record in trace order; sequential
/// squeezes differ because the transcript state advances.
pub fn next_freivalds_r(t: &mut Transcript, rows: usize) -> Vec<Fp61> {
    t.challenge_fp61_vec(b"freivalds.r", rows)
}

/// Failure building a claimed-output tensor from raw integer values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputTensorError {
    /// A value could not form a `BoundedInt`.
    Bound(BoundError),
    /// The tensor shape was invalid.
    Tensor(TensorError),
}

/// Build the canonical claimed-output [`Tensor`] from raw integer `values`, with
/// each cell bounded exactly (`[v, v]`). Prover and verifier both call this so
/// the `claimed_output_commitment` matches bit-for-bit.
pub fn output_tensor(
    tensor_id: u32,
    scale_id: u32,
    values: &[i64],
) -> Result<Tensor, OutputTensorError> {
    let mut data = Vec::with_capacity(values.len());
    for &v in values {
        data.push(BoundedInt::exact(v).map_err(OutputTensorError::Bound)?);
    }
    Tensor::new(tensor_id, alloc::vec![values.len() as u32], scale_id, data)
        .map_err(OutputTensorError::Tensor)
}
