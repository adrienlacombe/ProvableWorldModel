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
    output_tensor, relation_id, AuditArtifact, OutputTensorError, PlanningProof, RolloutProof,
    ARTIFACT_VERSION, RELATION_MLP, RELATION_VERSION, SERIALIZATION_VERSION,
};
use pwm_core::block::{Block, BlockOp};
use pwm_core::commit::{
    claimed_output_commitment, weights_root, ModelBinding, PlannerBinding, QuantBinding,
};
use pwm_core::field::{try_encode, OutOfRange};
use pwm_core::fixed_point::{requantize, OverflowPolicy, Rounding};
use pwm_core::planning::{argmin, mse_cost};
use pwm_core::predictor::{gate_vec, layernorm, modulate_vec, residual_add};
use pwm_core::public_input::PublicInput;
use pwm_core::relation::StatementType;
use pwm_core::tables::{activation_tables_commitment, ActivationTable};
use pwm_core::tensor::Tensor;

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

/// Failure building a planning or rollout proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanError {
    /// A candidate's / step's P0 proof could not be produced.
    Prove(ProveError),
    /// No candidate action sequences were supplied.
    NoCandidates,
    /// Fewer initial latents than the history window, or a zero window/horizon.
    BadHistory,
}

/// Prove an autoregressive rollout (`StatementType::P1Rollout`): predict the next
/// latent from the trailing `history_size`-window of latents, append it, and
/// repeat for `horizon` steps. Each step is a P0 proof; the window for step `t`
/// is the flattened latents available at step `t` (specs.md §9).
pub fn prove_rollout(
    model: &Model,
    initial_latents: &[Vec<i64>],
    history_size: usize,
    horizon: usize,
    out_binding: OutputBinding,
) -> Result<RolloutProof, PlanError> {
    if history_size == 0 || horizon == 0 || initial_latents.len() < history_size {
        return Err(PlanError::BadHistory);
    }
    let mut latents: Vec<Vec<i64>> = initial_latents.to_vec();
    let mut steps = Vec::with_capacity(horizon);
    let mut trajectory = Vec::with_capacity(horizon);
    for _ in 0..horizon {
        let window: Vec<i64> = latents[latents.len() - history_size..]
            .iter()
            .flatten()
            .copied()
            .collect();
        let artifact = prove_feedforward(model, &window, out_binding).map_err(PlanError::Prove)?;
        let next: Vec<i64> = artifact
            .claimed_output
            .data()
            .iter()
            .map(|c| c.value())
            .collect();
        latents.push(next.clone());
        trajectory.push(next);
        steps.push(artifact);
    }
    Ok(RolloutProof {
        artifact_version: ARTIFACT_VERSION,
        steps,
        initial_latents: initial_latents.to_vec(),
        history_size: history_size as u32,
        trajectory,
    })
}

/// Prove fixed-candidate planning (`StatementType::P2FixedCandidatePlanning`, the
/// V0 deliverable): roll each candidate input through the committed model, score
/// each final latent against `goal` by exact integer MSE, and select the minimum
/// under smallest-index tie-breaking. Composes one P0 proof per candidate.
pub fn prove_planning(
    model: &Model,
    candidate_inputs: &[Vec<i64>],
    goal: &[i64],
    out_binding: OutputBinding,
) -> Result<PlanningProof, PlanError> {
    if candidate_inputs.is_empty() {
        return Err(PlanError::NoCandidates);
    }
    let mut candidates = Vec::with_capacity(candidate_inputs.len());
    let mut costs = Vec::with_capacity(candidate_inputs.len());
    for input in candidate_inputs {
        let artifact = prove_feedforward(model, input, out_binding).map_err(PlanError::Prove)?;
        let output: Vec<i64> = artifact
            .claimed_output
            .data()
            .iter()
            .map(|c| c.value())
            .collect();
        costs.push(mse_cost(&output, goal));
        candidates.push(artifact);
    }
    let (selected_index, selected_cost) = argmin(&costs).expect("non-empty costs");
    Ok(PlanningProof {
        artifact_version: ARTIFACT_VERSION,
        candidates,
        goal: goal.to_vec(),
        costs,
        selected_index: selected_index as u32,
        selected_cost,
    })
}

/// Failure proving a predictor block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockError {
    /// A referenced buffer was not yet written/seeded.
    MissingBuffer(u32),
    /// A referenced weight/table binding was missing or malformed.
    MissingBinding,
    /// An activation/LayerNorm value fell outside the committed table domain.
    TableDomain,
    /// A shape mismatch between an op's buffers.
    Shape,
}

/// Run a named-buffer predictor block (specs.md §2.1): execute each [`BlockOp`]
/// over a buffer map using the exact integer kernels, filling each op's claimed
/// output. The input `block`'s op `out` fields are ignored and recomputed; the
/// returned block is verifiable by `pwm_verifier::verify_block`.
pub fn prove_block(
    block: &Block,
    weights: &[Tensor],
    tables: &[ActivationTable],
    inputs: &[(u32, Vec<i64>)],
) -> Result<Block, BlockError> {
    use std::collections::BTreeMap;
    let mut bufs: BTreeMap<u32, Vec<i64>> = BTreeMap::new();
    for (id, v) in inputs {
        bufs.insert(*id, v.clone());
    }
    let get = |bufs: &BTreeMap<u32, Vec<i64>>, id: u32| -> Result<Vec<i64>, BlockError> {
        bufs.get(&id).cloned().ok_or(BlockError::MissingBuffer(id))
    };
    let weight = |id: u32| {
        weights
            .iter()
            .find(|t| t.tensor_id() == id)
            .ok_or(BlockError::MissingBinding)
    };
    let table = |id: u32| {
        tables
            .iter()
            .find(|t| t.table_id == id)
            .ok_or(BlockError::MissingBinding)
    };

    let mut out_ops = Vec::with_capacity(block.ops.len());
    for op in &block.ops {
        let filled = match op.clone() {
            BlockOp::Linear {
                op_id,
                weight_id,
                bias_id,
                in_buf,
                out_buf,
                ..
            } => {
                let x = get(&bufs, in_buf)?;
                let w = weight(weight_id)?;
                let shape = w.shape();
                if shape.len() != 2 {
                    return Err(BlockError::Shape);
                }
                let (rows, cols) = (shape[0] as usize, shape[1] as usize);
                if x.len() != cols {
                    return Err(BlockError::Shape);
                }
                let wd = w.data();
                let bias: Vec<i64> = match bias_id {
                    Some(bid) => weight(bid)?.data().iter().map(|c| c.value()).collect(),
                    None => vec![0i64; rows],
                };
                let out: Vec<i64> = (0..rows)
                    .map(|r| {
                        let base = r * cols;
                        bias[r] + (0..cols).map(|c| wd[base + c].value() * x[c]).sum::<i64>()
                    })
                    .collect();
                bufs.insert(out_buf, out.clone());
                BlockOp::Linear {
                    op_id,
                    weight_id,
                    bias_id,
                    in_buf,
                    out_buf,
                    out,
                }
            }
            BlockOp::Requant {
                op_id,
                in_buf,
                out_buf,
                shift,
                zero_point,
                clamp_lo,
                clamp_hi,
                rounding,
                ..
            } => {
                let x = get(&bufs, in_buf)?;
                let mode =
                    Rounding::from_discriminant(rounding).ok_or(BlockError::MissingBinding)?;
                let out: Vec<i64> = x
                    .iter()
                    .map(|&n| requantize(n, shift, zero_point, clamp_lo, clamp_hi, mode))
                    .collect();
                bufs.insert(out_buf, out.clone());
                BlockOp::Requant {
                    op_id,
                    in_buf,
                    out_buf,
                    out,
                    shift,
                    zero_point,
                    clamp_lo,
                    clamp_hi,
                    rounding,
                }
            }
            BlockOp::Activation {
                op_id,
                table_id,
                in_buf,
                out_buf,
                ..
            } => {
                let x = get(&bufs, in_buf)?;
                let t = table(table_id)?;
                let mut out = Vec::with_capacity(x.len());
                for &xi in &x {
                    out.push(t.eval(xi).ok_or(BlockError::TableDomain)?);
                }
                bufs.insert(out_buf, out.clone());
                BlockOp::Activation {
                    op_id,
                    table_id,
                    in_buf,
                    out_buf,
                    out,
                }
            }
            BlockOp::LayerNorm {
                op_id,
                table_id,
                in_buf,
                out_buf,
                shift,
                clamp_lo,
                clamp_hi,
                rounding,
                ..
            } => {
                let x = get(&bufs, in_buf)?;
                let t = table(table_id)?;
                let mode =
                    Rounding::from_discriminant(rounding).ok_or(BlockError::MissingBinding)?;
                let out = layernorm(&x, t, shift, clamp_lo, clamp_hi, mode)
                    .ok_or(BlockError::TableDomain)?;
                bufs.insert(out_buf, out.clone());
                BlockOp::LayerNorm {
                    op_id,
                    table_id,
                    in_buf,
                    out_buf,
                    out,
                    shift,
                    clamp_lo,
                    clamp_hi,
                    rounding,
                }
            }
            BlockOp::Modulate {
                op_id,
                x_buf,
                scale_buf,
                shift_buf,
                out_buf,
                one,
                shift_bits,
                clamp_lo,
                clamp_hi,
                rounding,
                ..
            } => {
                let x = get(&bufs, x_buf)?;
                let scale = get(&bufs, scale_buf)?;
                let shift = get(&bufs, shift_buf)?;
                let mode =
                    Rounding::from_discriminant(rounding).ok_or(BlockError::MissingBinding)?;
                if scale.len() != x.len() || shift.len() != x.len() {
                    return Err(BlockError::Shape);
                }
                let out = modulate_vec(
                    &x, &scale, &shift, one, shift_bits, clamp_lo, clamp_hi, mode,
                );
                bufs.insert(out_buf, out.clone());
                BlockOp::Modulate {
                    op_id,
                    x_buf,
                    scale_buf,
                    shift_buf,
                    out_buf,
                    out,
                    one,
                    shift_bits,
                    clamp_lo,
                    clamp_hi,
                    rounding,
                }
            }
            BlockOp::Gate {
                op_id,
                gate_buf,
                x_buf,
                out_buf,
                shift_bits,
                clamp_lo,
                clamp_hi,
                rounding,
                ..
            } => {
                let g = get(&bufs, gate_buf)?;
                let x = get(&bufs, x_buf)?;
                let mode =
                    Rounding::from_discriminant(rounding).ok_or(BlockError::MissingBinding)?;
                if g.len() != x.len() {
                    return Err(BlockError::Shape);
                }
                let out = gate_vec(&g, &x, shift_bits, clamp_lo, clamp_hi, mode);
                bufs.insert(out_buf, out.clone());
                BlockOp::Gate {
                    op_id,
                    gate_buf,
                    x_buf,
                    out_buf,
                    out,
                    shift_bits,
                    clamp_lo,
                    clamp_hi,
                    rounding,
                }
            }
            BlockOp::Add {
                op_id,
                a_buf,
                b_buf,
                out_buf,
                ..
            } => {
                let a = get(&bufs, a_buf)?;
                let b = get(&bufs, b_buf)?;
                if a.len() != b.len() {
                    return Err(BlockError::Shape);
                }
                let out = residual_add(&a, &b);
                bufs.insert(out_buf, out.clone());
                BlockOp::Add {
                    op_id,
                    a_buf,
                    b_buf,
                    out_buf,
                    out,
                }
            }
        };
        out_ops.push(filled);
    }
    Ok(Block {
        input_bufs: block.input_bufs.clone(),
        ops: out_ops,
        output_buf: block.output_buf,
    })
}
