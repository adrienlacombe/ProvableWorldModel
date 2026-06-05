// SPDX-License-Identifier: Apache-2.0
//! `pwm-prover` — commit-and-audit prover orchestration (specs.md §6, Phase 1).
//!
//! The prover runs the exact integer reference inference, records the execution
//! trace, computes the model / quantization / planner / output commitments, and
//! assembles a self-contained [`AuditArtifact`]. It generates no proving circuit:
//! the trace's accumulators are the true `W·x + bias` values, committed before any
//! Freivalds challenge exists (non-interactive Fiat-Shamir; the verifier derives
//! the challenge from the commitment).
//!
//! `pwm-prover` depends only on `pwm-core` and `pwm-export` (the Rust reference) —
//! no proving substrate. The trace builder, rollout, and planning provers extend
//! this per the backlog (M3–M6).

use pwm_core::audit::{
    output_tensor, relation_id, AuditArtifact, OutputTensorError, ARTIFACT_VERSION, RELATION_MLP,
    RELATION_VERSION, SERIALIZATION_VERSION,
};
use pwm_core::commit::{
    claimed_output_commitment, weights_root, ModelBinding, PlannerBinding, QuantBinding,
};
use pwm_core::field::{try_encode, OutOfRange};
use pwm_core::fixed_point::{OverflowPolicy, Rounding};
use pwm_core::public_input::PublicInput;
use pwm_core::relation::StatementType;
use pwm_core::tables::activation_tables_commitment;

use pwm_export::reference::{Model, ReferenceError};

/// Prover failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProveError {
    /// The reference inference failed.
    Reference(ReferenceError),
    /// An input value was outside the M31 representable range.
    InputEncoding(OutOfRange),
    /// The claimed output tensor could not be built.
    Output(OutputTensorError),
}

/// Parameters binding the output tensor identity (so prover and verifier agree).
#[derive(Debug, Clone, Copy)]
pub struct OutputBinding {
    /// Tensor id for the claimed output.
    pub tensor_id: u32,
    /// Scale id for the claimed output.
    pub scale_id: u32,
}

/// Prove the quantized feed-forward statement (`StatementType::P0Step`) over
/// `model` for `input`, returning a self-contained [`AuditArtifact`].
pub fn prove_feedforward(
    model: &Model,
    input: &[i64],
    out_binding: OutputBinding,
) -> Result<AuditArtifact, ProveError> {
    // 1. Exact integer reference inference -> trace, graph, output.
    let run = model.run(input).map_err(ProveError::Reference)?;

    // 2. Commitments (all via pwm-core, so the verifier recomputes identically).
    let architecture_commitment = run.graph.commitment();
    let weights = model.weight_tensors();
    let w_root = weights_root(&weights);
    let model_commitment = ModelBinding {
        architecture_commitment,
        weights_root: w_root,
        relation_version: RELATION_VERSION,
        serialization_version: SERIALIZATION_VERSION,
    }
    .commitment();

    let quantization_commitment = QuantBinding {
        default_rounding: Rounding::NearestTiesToEven,
        overflow_policy: OverflowPolicy::Reject,
        scales: model.scales.clone(),
        activation_tables_commitment: activation_tables_commitment(&model.tables),
    }
    .commitment();

    let planner_config_commitment = PlannerBinding {
        horizon: 0,
        action_block: 0,
        candidate_count: 0,
        tie_break_rule_id: 0,
    }
    .commitment();

    // 3. Claimed output tensor + commitment.
    let claimed_output = output_tensor(out_binding.tensor_id, out_binding.scale_id, &run.output)
        .map_err(ProveError::Output)?;
    let output_commitment = claimed_output_commitment(core::slice::from_ref(&claimed_output));

    // 4. Public input (input is public in the clear for the slice).
    let mut latent_history_public = Vec::with_capacity(input.len());
    for &v in input {
        latent_history_public.push(try_encode(v).map_err(ProveError::InputEncoding)?);
    }

    let public_input = PublicInput {
        relation_id: relation_id(RELATION_MLP),
        model_commitment,
        quantization_commitment,
        planner_config_commitment,
        statement_type: StatementType::P0Step,
        latent_history_commitment: None,
        latent_history_public: Some(latent_history_public),
        goal_latent_commitment: None,
        goal_latent_public: None,
        candidate_actions_commitment: None,
        candidate_actions_public: None,
        claimed_output_commitment: output_commitment,
        selected_index: None,
        selected_cost: None,
    };

    Ok(AuditArtifact {
        artifact_version: ARTIFACT_VERSION,
        public_input,
        graph: run.graph,
        weights,
        tables: model.tables.clone(),
        scales: model.scales.clone(),
        trace: run.trace,
        claimed_output,
    })
}
