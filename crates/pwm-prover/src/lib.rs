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
//! `pwm-prover` depends only on `pwm-core` and the `pwm-export` Rust reference —
//! no proving substrate. It provides the P0 feed-forward prover
//! ([`prove_feedforward`]), the P1 autoregressive rollout ([`prove_rollout`]), the
//! P2 fixed-candidate planner ([`prove_planning`]), and the commitment-bound
//! predictor-block provers ([`prove_block`], [`prove_predictor`]).

use pwm_core::audit::{
    output_tensor, relation_id, AuditArtifact, OutputTensorError, PlanningProof, PredictorArtifact,
    RolloutProof, ARTIFACT_VERSION, RELATION_MLP, RELATION_PREDICTOR,
};
use pwm_core::block::{block_architecture_commitment, Block, BlockOp};
use pwm_core::commit::{
    claimed_output_commitment, predictor_inputs_commitment, weights_root, ModelBinding,
    PlannerBinding, QuantBinding,
};
use pwm_core::field::{try_encode, OutOfRange};
use pwm_core::fixed_point::{requantize, Rounding};
use pwm_core::planning::{argmin, mse_cost};
use pwm_core::predictor::{
    gate_vec, layernorm, linear, matmul, modulate_vec, residual_add, softmax_rows,
};
use pwm_core::public_input::PublicInput;
use pwm_core::relation::StatementType;
use pwm_core::tables::{activation_tables_commitment, ActivationTable};
use pwm_core::tensor::{Scale, Tensor};

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
    /// Running the predictor block failed.
    Block(BlockError),
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
    let weights = model.weight_tensors();
    let model_commitment =
        ModelBinding::v0(run.graph.commitment(), weights_root(&weights)).commitment();
    let quantization_commitment = QuantBinding::v0(
        model.scales.clone(),
        activation_tables_commitment(&model.tables),
    )
    .commitment();
    let planner_config_commitment = PlannerBinding::p0_sentinel().commitment();

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
    /// A candidate's exact goal-MSE cost did not fit `i64` (outside the V0 planning
    /// envelope; `mse_cost` returned `None`).
    CostOverflow,
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
            .map(pwm_core::BoundedInt::value)
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
            .map(pwm_core::BoundedInt::value)
            .collect();
        costs.push(mse_cost(&output, goal).ok_or(PlanError::CostOverflow)?);
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
    /// No weight/bias tensor was bound for this `weight_id`.
    MissingWeight(u32),
    /// No activation/LayerNorm table was bound for this `table_id`.
    MissingTable(u32),
    /// An op's serialized rounding-mode byte did not decode to a known [`Rounding`].
    InvalidRounding {
        /// The op whose rounding field was malformed.
        op_id: u32,
        /// The undecodable discriminant byte.
        discriminant: u8,
    },
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
            .ok_or(BlockError::MissingWeight(id))
    };
    let table = |id: u32| {
        tables
            .iter()
            .find(|t| t.table_id == id)
            .ok_or(BlockError::MissingTable(id))
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
                    Some(bid) => weight(bid)?
                        .data()
                        .iter()
                        .map(pwm_core::BoundedInt::value)
                        .collect(),
                    None => vec![0i64; rows],
                };
                // Delegate to the shared kernel so the prover trace is definitionally
                // identical to the reference model and the verifier's recompute.
                let out = linear(wd, &x, &bias, rows, cols);
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
                let mode = Rounding::from_discriminant(rounding).ok_or(
                    BlockError::InvalidRounding {
                        op_id,
                        discriminant: rounding,
                    },
                )?;
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
                let mode = Rounding::from_discriminant(rounding).ok_or(
                    BlockError::InvalidRounding {
                        op_id,
                        discriminant: rounding,
                    },
                )?;
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
                let mode = Rounding::from_discriminant(rounding).ok_or(
                    BlockError::InvalidRounding {
                        op_id,
                        discriminant: rounding,
                    },
                )?;
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
                let mode = Rounding::from_discriminant(rounding).ok_or(
                    BlockError::InvalidRounding {
                        op_id,
                        discriminant: rounding,
                    },
                )?;
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
            BlockOp::MatMul {
                op_id,
                a_buf,
                b_buf,
                out_buf,
                rows,
                inner,
                cols,
                transpose_b,
                ..
            } => {
                let a = get(&bufs, a_buf)?;
                let b = get(&bufs, b_buf)?;
                let out = matmul(
                    &a,
                    &b,
                    rows as usize,
                    inner as usize,
                    cols as usize,
                    transpose_b,
                )
                .ok_or(BlockError::Shape)?;
                bufs.insert(out_buf, out.clone());
                BlockOp::MatMul {
                    op_id,
                    a_buf,
                    b_buf,
                    out_buf,
                    out,
                    rows,
                    inner,
                    cols,
                    transpose_b,
                }
            }
            BlockOp::Softmax {
                op_id,
                table_id,
                in_buf,
                out_buf,
                row_len,
                one,
                ..
            } => {
                let x = get(&bufs, in_buf)?;
                let t = table(table_id)?;
                let out =
                    softmax_rows(&x, row_len as usize, t, one).ok_or(BlockError::TableDomain)?;
                bufs.insert(out_buf, out.clone());
                BlockOp::Softmax {
                    op_id,
                    table_id,
                    in_buf,
                    out_buf,
                    out,
                    row_len,
                    one,
                }
            }
            BlockOp::Slice {
                op_id,
                in_buf,
                out_buf,
                start,
                len,
                ..
            } => {
                let x = get(&bufs, in_buf)?;
                let s = start as usize;
                let e = s + len as usize;
                if e > x.len() {
                    return Err(BlockError::Shape);
                }
                let out = x[s..e].to_vec();
                bufs.insert(out_buf, out.clone());
                BlockOp::Slice {
                    op_id,
                    in_buf,
                    out_buf,
                    start,
                    len,
                    out,
                }
            }
            BlockOp::Concat {
                op_id,
                in_bufs,
                out_buf,
                ..
            } => {
                let mut out = Vec::new();
                for id in &in_bufs {
                    out.extend(get(&bufs, *id)?);
                }
                bufs.insert(out_buf, out.clone());
                BlockOp::Concat {
                    op_id,
                    in_bufs,
                    out_buf,
                    out,
                }
            }
            BlockOp::BatchedLinear {
                op_id,
                weight_id,
                bias_id,
                in_buf,
                out_buf,
                seq,
                ..
            } => {
                let x = get(&bufs, in_buf)?;
                let w = weight(weight_id)?;
                let shape = w.shape();
                if shape.len() != 2 {
                    return Err(BlockError::Shape);
                }
                let (rows, cols) = (shape[0] as usize, shape[1] as usize);
                let s = seq as usize;
                if x.len() != s * cols {
                    return Err(BlockError::Shape);
                }
                let wd = w.data();
                let bias: Vec<i64> = match bias_id {
                    Some(bid) => weight(bid)?
                        .data()
                        .iter()
                        .map(pwm_core::BoundedInt::value)
                        .collect(),
                    None => vec![0i64; rows],
                };
                // Same shared kernel as `Linear`, applied per sequence row.
                let mut out = Vec::with_capacity(s * rows);
                for tt in 0..s {
                    let xrow = &x[tt * cols..(tt + 1) * cols];
                    out.extend(linear(wd, xrow, &bias, rows, cols));
                }
                bufs.insert(out_buf, out.clone());
                BlockOp::BatchedLinear {
                    op_id,
                    weight_id,
                    bias_id,
                    in_buf,
                    out_buf,
                    out,
                    seq,
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

/// Prove one predictor step as a commitment-bound [`PredictorArtifact`]
/// (`RELATION_PREDICTOR`): run the block prover, then bind the model commitment
/// (block architecture + weight root), the quantization commitment (scales +
/// tables), the seeded-input commitment, and the claimed-output commitment into a
/// [`PublicInput`]. The verifier (`pwm_verifier::verify_predictor`) recomputes and
/// checks every commitment, so the proof attests the *committed* model on the
/// *committed* inputs — not merely some caller-supplied weights.
pub fn prove_predictor(
    skeleton: &Block,
    weights: &[Tensor],
    tables: &[ActivationTable],
    scales: &[Scale],
    inputs: &[(u32, Vec<i64>)],
    out_binding: OutputBinding,
) -> Result<PredictorArtifact, ProveError> {
    // 1. Run the exact integer reference over the block, filling each op's output.
    let proven = prove_block(skeleton, weights, tables, inputs).map_err(ProveError::Block)?;

    // 2. The block output is the claimed `out` of the op writing `output_buf`.
    let out_values: Vec<i64> = proven
        .ops
        .iter()
        .rev()
        .find(|op| op.out_buf() == proven.output_buf)
        .map(|op| op.out().to_vec())
        .ok_or(ProveError::Block(BlockError::MissingBuffer(
            proven.output_buf,
        )))?;

    // 3. Commitments (all via pwm-core, so the verifier recomputes identically).
    let model_commitment = ModelBinding::v0(
        block_architecture_commitment(&proven),
        weights_root(weights),
    )
    .commitment();
    let quantization_commitment =
        QuantBinding::v0(scales.to_vec(), activation_tables_commitment(tables)).commitment();
    let planner_config_commitment = PlannerBinding::p0_sentinel().commitment();
    let input_commitment = predictor_inputs_commitment(inputs);

    // 4. Claimed output tensor + commitment.
    let claimed_output = output_tensor(out_binding.tensor_id, out_binding.scale_id, &out_values)
        .map_err(ProveError::Output)?;
    let output_commitment = claimed_output_commitment(core::slice::from_ref(&claimed_output));

    let public_input = PublicInput {
        relation_id: relation_id(RELATION_PREDICTOR),
        model_commitment,
        quantization_commitment,
        planner_config_commitment,
        statement_type: StatementType::P0Step,
        // The predictor's seeded buffers (latent history + action) are public but
        // carried in the artifact; bind their commitment here.
        latent_history_commitment: Some(input_commitment),
        latent_history_public: None,
        goal_latent_commitment: None,
        goal_latent_public: None,
        candidate_actions_commitment: None,
        candidate_actions_public: None,
        claimed_output_commitment: output_commitment,
        selected_index: None,
        selected_cost: None,
    };

    Ok(PredictorArtifact {
        artifact_version: ARTIFACT_VERSION,
        public_input,
        block: proven,
        weights: weights.to_vec(),
        tables: tables.to_vec(),
        scales: scales.to_vec(),
        inputs: inputs.to_vec(),
        claimed_output,
    })
}
