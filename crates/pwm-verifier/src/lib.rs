// SPDX-License-Identifier: Apache-2.0
#![no_std]
//! `pwm-verifier` — the trust anchor (commit-and-audit, CPU, `no_std`, float-free).
//!
//! [`verify`] checks an [`AuditArtifact`] (specs.md §6, Phase 3): it recomputes
//! the public-input commitments, confirms the trace conforms to the committed op
//! graph, replays the Fiat-Shamir transcript to derive the Freivalds challenges,
//! and walks the trace threading a running activation vector — Freivalds-checking
//! every linear op and **exactly recomputing** every requant and committed-table
//! op — then checks the first input against the public input and the final output
//! against the claimed-output commitment.
//!
//! It runs no model and contains no floating point. The audit surface is
//! `pwm-core` (fields, fixed-point, commitments, transcript, freivalds, graph,
//! tables, trace) and this crate. It depends on **neither `pwm-export` nor any
//! Python/PyTorch runtime** (INV-ARCH-02). Unsupported semantics fail closed.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use pwm_core::block::{Block, BlockOp};
use pwm_core::predictor::{gate_vec, matmul, modulate_vec, residual_add, softmax_rows};
use pwm_core::tensor::Tensor;

use pwm_core::audit::{
    audit_transcript, next_freivalds_r, output_tensor, relation_id, AuditArtifact, PlanningProof,
    RolloutProof, RELATION_MLP, RELATION_VERSION, SERIALIZATION_VERSION,
};
use pwm_core::commit::{
    claimed_output_commitment, weights_root, ModelBinding, PlannerBinding, QuantBinding,
};
use pwm_core::field::Fp61;
use pwm_core::fixed_point::{requantize, OverflowPolicy, Rounding};
use pwm_core::freivalds::{check_linear_biased, precompute_v};
use pwm_core::graph::OpSpec;
use pwm_core::planning::{mse_cost, verify_argmin};
use pwm_core::predictor::layernorm;
use pwm_core::relation::StatementType;
use pwm_core::tables::activation_tables_commitment;
use pwm_core::trace::OpRecord;
use pwm_core::transcript::Transcript;

/// Why a proof was rejected. Every rejection has a stable, specific code
/// (specs.md §12); unsupported semantics fail closed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyError {
    /// `artifact_version` is not supported.
    UnsupportedArtifactVersion(u32),
    /// `relation_id` is not a relation this verifier supports.
    UnsupportedRelation,
    /// The statement type is wrong for the relation.
    RelationMismatch,
    /// A recomputed commitment did not match the public input.
    CommitmentMismatch(&'static str),
    /// The trace does not conform to the committed op graph.
    GraphMismatch {
        /// Index in the op list where the mismatch was found.
        index: usize,
    },
    /// A referenced weight or table was missing from the artifact.
    MissingBinding(&'static str),
    /// The first trace input did not match the public input.
    PublicInputMismatch,
    /// A record's input did not equal the previous record's output.
    WiringMismatch {
        /// Trace index of the offending record.
        index: usize,
    },
    /// A Freivalds linear check failed (`v·x + r·bias != r·out`).
    FreivaldsCheckFailed {
        /// Op id of the offending linear.
        op_id: u32,
    },
    /// An exactly-recomputed op (requant / table) did not match its record.
    ExactReplayMismatch {
        /// Op id of the offending op.
        op_id: u32,
    },
    /// An activation input fell outside the committed table domain.
    ActivationDomain {
        /// The table id.
        table_id: u32,
    },
    /// The claimed output did not equal the computed output.
    OutputMismatch,
    /// The claimed-output commitment did not match the public input.
    OutputCommitmentMismatch,
    /// A value could not be represented (e.g. building the output tensor).
    Encoding,
    /// A planning proof had no candidates.
    NoCandidates,
    /// A candidate's P0 proof failed to verify.
    Candidate {
        /// Candidate index.
        index: usize,
    },
    /// A recomputed candidate cost did not match the claimed cost.
    CostMismatch {
        /// Candidate index.
        index: usize,
    },
    /// The argmin selection was unsound (lower cost, wrong index, or tie-break).
    ArgminViolation,
    /// A rollout step's input window or predicted latent broke the recurrence.
    RolloutWiring {
        /// The offending rollout step.
        step: usize,
    },
    /// A block op read a buffer that was not yet written / seeded.
    MissingBuffer {
        /// The missing buffer id.
        buf: u32,
    },
    /// A block op's claimed output disagreed with its exact recompute.
    BlockOpMismatch {
        /// Op id of the offending block op.
        op_id: u32,
    },
}

/// Verify a commit-and-audit proof. Returns `Ok(())` iff every check passes.
pub fn verify(artifact: &AuditArtifact) -> Result<(), VerifyError> {
    let pi = &artifact.public_input;

    // 1. Version + relation gating.
    if artifact.artifact_version != pwm_core::audit::ARTIFACT_VERSION {
        return Err(VerifyError::UnsupportedArtifactVersion(
            artifact.artifact_version,
        ));
    }
    if pi.relation_id != relation_id(RELATION_MLP) {
        return Err(VerifyError::UnsupportedRelation);
    }
    if pi.statement_type != StatementType::P0Step {
        return Err(VerifyError::RelationMismatch);
    }

    // 2. Recompute and check the bound commitments.
    let architecture_commitment = artifact.graph.commitment();
    let w_root = weights_root(&artifact.weights);
    let model_commitment = ModelBinding {
        architecture_commitment,
        weights_root: w_root,
        relation_version: RELATION_VERSION,
        serialization_version: SERIALIZATION_VERSION,
    }
    .commitment();
    if model_commitment != pi.model_commitment {
        return Err(VerifyError::CommitmentMismatch("model"));
    }

    let quantization_commitment = QuantBinding {
        default_rounding: Rounding::NearestTiesToEven,
        overflow_policy: OverflowPolicy::Reject,
        scales: artifact.scales.clone(),
        activation_tables_commitment: activation_tables_commitment(&artifact.tables),
    }
    .commitment();
    if quantization_commitment != pi.quantization_commitment {
        return Err(VerifyError::CommitmentMismatch("quantization"));
    }

    let planner_config_commitment = PlannerBinding {
        horizon: 0,
        action_block: 0,
        candidate_count: 0,
        tie_break_rule_id: 0,
    }
    .commitment();
    if planner_config_commitment != pi.planner_config_commitment {
        return Err(VerifyError::CommitmentMismatch("planner"));
    }

    // 3. The trace must conform to the committed op graph (same kinds/ids/dims).
    check_graph_conformance(artifact)?;

    // 4. Replay the transcript to derive Freivalds challenges (bound to the
    //    recomputed trace root, so challenges depend on the committed accumulators).
    let troot = artifact.trace_root();
    let mut transcript = audit_transcript(pi, &troot);

    // 5. Walk the trace, threading the running activation.
    let mut current: Vec<i64> = decode_public_input(pi)?;
    if let Some(first) = artifact.trace.first() {
        if first.input() != current.as_slice() {
            return Err(VerifyError::PublicInputMismatch);
        }
    }

    for (index, record) in artifact.trace.iter().enumerate() {
        if record.input() != current.as_slice() {
            return Err(VerifyError::WiringMismatch { index });
        }
        match record {
            OpRecord::Linear(r) => check_linear(artifact, r, &mut transcript)?,
            OpRecord::Requant(r) => check_requant(r)?,
            OpRecord::Activation(r) => check_activation(artifact, r)?,
            OpRecord::LayerNorm(r) => check_layernorm(artifact, r)?,
        }
        current = record.output().to_vec();
    }

    // 6. The computed output must equal the claimed output, and the claimed-output
    //    commitment must match the public input.
    if artifact.claimed_output.data().len() != current.len() {
        return Err(VerifyError::OutputMismatch);
    }
    for (cell, &v) in artifact.claimed_output.data().iter().zip(current.iter()) {
        if cell.value() != v {
            return Err(VerifyError::OutputMismatch);
        }
    }
    // Rebuild the canonical output tensor from the computed values and compare its
    // commitment to the public input (binds tensor id, scale, and values).
    let rebuilt = output_tensor(
        artifact.claimed_output.tensor_id(),
        artifact.claimed_output.scale_id(),
        &current,
    )
    .map_err(|_| VerifyError::Encoding)?;
    let computed_commitment = claimed_output_commitment(core::slice::from_ref(&rebuilt));
    if computed_commitment != pi.claimed_output_commitment {
        return Err(VerifyError::OutputCommitmentMismatch);
    }

    Ok(())
}

/// Verify a fixed-candidate planning proof (P2 = V0): every candidate's P0 proof
/// verifies, every cost is the exact goal-MSE of that candidate's verified output,
/// and the selected candidate is the minimum under smallest-index tie-breaking
/// (specs.md §10). All candidates are scored — proving only the winner is unsound.
pub fn verify_planning(proof: &PlanningProof) -> Result<(), VerifyError> {
    if proof.candidates.is_empty() {
        return Err(VerifyError::NoCandidates);
    }
    if proof.costs.len() != proof.candidates.len() {
        return Err(VerifyError::ArgminViolation);
    }
    for (index, candidate) in proof.candidates.iter().enumerate() {
        // 1. Each candidate is a sound P0 proof over the committed model.
        verify(candidate).map_err(|_| VerifyError::Candidate { index })?;
        // 2. Its cost is the exact goal-MSE of its (now verified) output.
        let output: Vec<i64> = candidate
            .claimed_output
            .data()
            .iter()
            .map(|c| c.value())
            .collect();
        if output.len() != proof.goal.len() {
            return Err(VerifyError::CostMismatch { index });
        }
        if mse_cost(&output, &proof.goal) != proof.costs[index] {
            return Err(VerifyError::CostMismatch { index });
        }
    }
    // 3. The selection is the sound argmin.
    verify_argmin(
        &proof.costs,
        proof.selected_index as usize,
        proof.selected_cost,
    )
    .map_err(|_| VerifyError::ArgminViolation)
}

/// Verify an autoregressive rollout proof (P1): every step is a sound P0 proof,
/// and the recurrence wiring holds — step `t`'s input is exactly the flattened
/// trailing `history_size`-window of latents (initial history + previously
/// predicted latents), and each step's output is the recorded trajectory latent
/// (specs.md §9).
pub fn verify_rollout(proof: &RolloutProof) -> Result<(), VerifyError> {
    let h = proof.history_size as usize;
    if h == 0 || proof.initial_latents.len() < h || proof.steps.len() != proof.trajectory.len() {
        return Err(VerifyError::RolloutWiring { step: 0 });
    }
    let mut latents = proof.initial_latents.clone();
    for (step, artifact) in proof.steps.iter().enumerate() {
        // 1. The step is a sound P0 proof over the committed model.
        verify(artifact).map_err(|_| VerifyError::Candidate { index: step })?;
        // 2. Its input is the flattened trailing window of latents at this step.
        let window: Vec<i64> = latents[latents.len() - h..]
            .iter()
            .flatten()
            .copied()
            .collect();
        let got = decode_public_input(&artifact.public_input)?;
        if got != window {
            return Err(VerifyError::RolloutWiring { step });
        }
        // 3. Its output is the recorded trajectory latent; append for the next step.
        let next: Vec<i64> = artifact
            .claimed_output
            .data()
            .iter()
            .map(|c| c.value())
            .collect();
        if next != proof.trajectory[step] {
            return Err(VerifyError::RolloutWiring { step });
        }
        latents.push(next);
    }
    Ok(())
}

/// Verify a named-buffer predictor block (specs.md §2.1, §5): replay each
/// [`BlockOp`] over a buffer map, threading inputs by buffer id — Linear via
/// Freivalds, the rest by exact integer recompute (LayerNorm / activation tables /
/// AdaLN modulate / gate / residual add). Returns the block's output buffer. The
/// `transcript` must already be bound to the block's ops and inputs by the caller;
/// `weights` and `tables` are the committed bindings.
pub fn verify_block(
    block: &Block,
    weights: &[Tensor],
    tables: &[pwm_core::tables::ActivationTable],
    inputs: &[(u32, Vec<i64>)],
    transcript: &mut Transcript,
) -> Result<Vec<i64>, VerifyError> {
    let mut bufs: BTreeMap<u32, Vec<i64>> = BTreeMap::new();
    for (id, v) in inputs {
        bufs.insert(*id, v.clone());
    }
    let get = |bufs: &BTreeMap<u32, Vec<i64>>, id: u32| -> Result<Vec<i64>, VerifyError> {
        bufs.get(&id)
            .cloned()
            .ok_or(VerifyError::MissingBuffer { buf: id })
    };
    let find_weight = |id: u32| -> Result<&Tensor, VerifyError> {
        weights
            .iter()
            .find(|t| t.tensor_id() == id)
            .ok_or(VerifyError::MissingBinding("block_weight"))
    };
    let find_table = |id: u32| -> Result<&pwm_core::tables::ActivationTable, VerifyError> {
        tables
            .iter()
            .find(|t| t.table_id == id)
            .ok_or(VerifyError::MissingBinding("block_table"))
    };

    for op in &block.ops {
        match op {
            BlockOp::Linear {
                op_id,
                weight_id,
                bias_id,
                in_buf,
                out_buf,
                out,
            } => {
                let x = get(&bufs, *in_buf)?;
                let w = find_weight(*weight_id)?;
                let shape = w.shape();
                if shape.len() != 2 {
                    return Err(VerifyError::MissingBinding("block_weight_shape"));
                }
                let rows = shape[0] as usize;
                let cols = shape[1] as usize;
                if x.len() != cols || out.len() != rows {
                    return Err(VerifyError::BlockOpMismatch { op_id: *op_id });
                }
                let w_i8: Vec<i8> = w.data().iter().map(|c| c.value() as i8).collect();
                let bias: Vec<i64> = match bias_id {
                    Some(bid) => find_weight(*bid)?
                        .data()
                        .iter()
                        .map(|c| c.value())
                        .collect(),
                    None => alloc::vec![0i64; rows],
                };
                let r: Vec<Fp61> = next_freivalds_r(transcript, rows);
                let v = precompute_v(&r, &w_i8, rows, cols);
                if !check_linear_biased(&v, &x, &bias, &r, out) {
                    return Err(VerifyError::FreivaldsCheckFailed { op_id: *op_id });
                }
                bufs.insert(*out_buf, out.clone());
            }
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
            } => {
                let x = get(&bufs, *in_buf)?;
                let mode = Rounding::from_discriminant(*rounding)
                    .ok_or(VerifyError::BlockOpMismatch { op_id: *op_id })?;
                if x.len() != out.len() {
                    return Err(VerifyError::BlockOpMismatch { op_id: *op_id });
                }
                for (&n, &claimed) in x.iter().zip(out.iter()) {
                    if requantize(n, *shift, *zero_point, *clamp_lo, *clamp_hi, mode) != claimed {
                        return Err(VerifyError::BlockOpMismatch { op_id: *op_id });
                    }
                }
                bufs.insert(*out_buf, out.clone());
            }
            BlockOp::Activation {
                op_id,
                table_id,
                in_buf,
                out_buf,
                out,
            } => {
                let x = get(&bufs, *in_buf)?;
                let table = find_table(*table_id)?;
                if x.len() != out.len() {
                    return Err(VerifyError::BlockOpMismatch { op_id: *op_id });
                }
                for (&xi, &claimed) in x.iter().zip(out.iter()) {
                    let e = table.eval(xi).ok_or(VerifyError::ActivationDomain {
                        table_id: *table_id,
                    })?;
                    if e != claimed {
                        return Err(VerifyError::BlockOpMismatch { op_id: *op_id });
                    }
                }
                bufs.insert(*out_buf, out.clone());
            }
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
            } => {
                let x = get(&bufs, *in_buf)?;
                let table = find_table(*table_id)?;
                let mode = Rounding::from_discriminant(*rounding)
                    .ok_or(VerifyError::BlockOpMismatch { op_id: *op_id })?;
                let expected = layernorm(&x, table, *shift, *clamp_lo, *clamp_hi, mode).ok_or(
                    VerifyError::ActivationDomain {
                        table_id: *table_id,
                    },
                )?;
                if &expected != out {
                    return Err(VerifyError::BlockOpMismatch { op_id: *op_id });
                }
                bufs.insert(*out_buf, out.clone());
            }
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
            } => {
                let x = get(&bufs, *x_buf)?;
                let scale = get(&bufs, *scale_buf)?;
                let shift = get(&bufs, *shift_buf)?;
                let mode = Rounding::from_discriminant(*rounding)
                    .ok_or(VerifyError::BlockOpMismatch { op_id: *op_id })?;
                if scale.len() != x.len() || shift.len() != x.len() {
                    return Err(VerifyError::BlockOpMismatch { op_id: *op_id });
                }
                let expected = modulate_vec(
                    &x,
                    &scale,
                    &shift,
                    *one,
                    *shift_bits,
                    *clamp_lo,
                    *clamp_hi,
                    mode,
                );
                if &expected != out {
                    return Err(VerifyError::BlockOpMismatch { op_id: *op_id });
                }
                bufs.insert(*out_buf, out.clone());
            }
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
            } => {
                let g = get(&bufs, *gate_buf)?;
                let x = get(&bufs, *x_buf)?;
                let mode = Rounding::from_discriminant(*rounding)
                    .ok_or(VerifyError::BlockOpMismatch { op_id: *op_id })?;
                if g.len() != x.len() {
                    return Err(VerifyError::BlockOpMismatch { op_id: *op_id });
                }
                let expected = gate_vec(&g, &x, *shift_bits, *clamp_lo, *clamp_hi, mode);
                if &expected != out {
                    return Err(VerifyError::BlockOpMismatch { op_id: *op_id });
                }
                bufs.insert(*out_buf, out.clone());
            }
            BlockOp::Add {
                op_id,
                a_buf,
                b_buf,
                out_buf,
                out,
            } => {
                let a = get(&bufs, *a_buf)?;
                let b = get(&bufs, *b_buf)?;
                if a.len() != b.len() {
                    return Err(VerifyError::BlockOpMismatch { op_id: *op_id });
                }
                if &residual_add(&a, &b) != out {
                    return Err(VerifyError::BlockOpMismatch { op_id: *op_id });
                }
                bufs.insert(*out_buf, out.clone());
            }
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
            } => {
                let a = get(&bufs, *a_buf)?;
                let b = get(&bufs, *b_buf)?;
                let expected = matmul(
                    &a,
                    &b,
                    *rows as usize,
                    *inner as usize,
                    *cols as usize,
                    *transpose_b,
                )
                .ok_or(VerifyError::BlockOpMismatch { op_id: *op_id })?;
                if &expected != out {
                    return Err(VerifyError::BlockOpMismatch { op_id: *op_id });
                }
                bufs.insert(*out_buf, out.clone());
            }
            BlockOp::Softmax {
                op_id,
                table_id,
                in_buf,
                out_buf,
                out,
                row_len,
                one,
            } => {
                let x = get(&bufs, *in_buf)?;
                let t = find_table(*table_id)?;
                let expected = softmax_rows(&x, *row_len as usize, t, *one).ok_or(
                    VerifyError::ActivationDomain {
                        table_id: *table_id,
                    },
                )?;
                if &expected != out {
                    return Err(VerifyError::BlockOpMismatch { op_id: *op_id });
                }
                bufs.insert(*out_buf, out.clone());
            }
        }
    }
    get(&bufs, block.output_buf)
}

/// Each trace record must match the graph op at the same index by kind, ids, and
/// dimensions.
fn check_graph_conformance(artifact: &AuditArtifact) -> Result<(), VerifyError> {
    let ops = &artifact.graph.ops;
    if ops.len() != artifact.trace.len() {
        return Err(VerifyError::GraphMismatch {
            index: ops.len().min(artifact.trace.len()),
        });
    }
    for (index, (spec, rec)) in ops.iter().zip(artifact.trace.iter()).enumerate() {
        let ok = match (spec, rec) {
            (
                OpSpec::Linear {
                    op_id,
                    weight_id,
                    bias_id,
                    rows,
                    cols,
                },
                OpRecord::Linear(r),
            ) => {
                r.op_id == *op_id
                    && r.weight_id == *weight_id
                    && r.bias_id == *bias_id
                    && r.input.len() == *cols as usize
                    && r.output.len() == *rows as usize
            }
            (
                OpSpec::Requant {
                    op_id,
                    shift,
                    zero_point,
                    clamp_lo,
                    clamp_hi,
                    rounding,
                },
                OpRecord::Requant(r),
            ) => {
                r.op_id == *op_id
                    && r.shift == *shift
                    && r.zero_point == *zero_point
                    && r.clamp_lo == *clamp_lo
                    && r.clamp_hi == *clamp_hi
                    && r.rounding == *rounding
            }
            (OpSpec::Activation { op_id, table_id }, OpRecord::Activation(r)) => {
                r.op_id == *op_id && r.table_id == *table_id
            }
            (
                OpSpec::LayerNorm {
                    op_id,
                    table_id,
                    shift,
                    clamp_lo,
                    clamp_hi,
                    rounding,
                },
                OpRecord::LayerNorm(r),
            ) => {
                r.op_id == *op_id
                    && r.table_id == *table_id
                    && r.shift == *shift
                    && r.clamp_lo == *clamp_lo
                    && r.clamp_hi == *clamp_hi
                    && r.rounding == *rounding
            }
            _ => false,
        };
        if !ok {
            return Err(VerifyError::GraphMismatch { index });
        }
    }
    Ok(())
}

/// Decode the public latent history (M31) into the integer input vector.
fn decode_public_input(pi: &pwm_core::public_input::PublicInput) -> Result<Vec<i64>, VerifyError> {
    let hist = pi
        .latent_history_public
        .as_ref()
        .ok_or(VerifyError::PublicInputMismatch)?;
    Ok(hist.iter().map(|&m| pwm_core::field::decode(m)).collect())
}

/// Freivalds-check `out = W·x + bias` (specs.md §7).
fn check_linear(
    artifact: &AuditArtifact,
    r: &pwm_core::trace::LinearRec,
    transcript: &mut Transcript,
) -> Result<(), VerifyError> {
    let w_tensor = artifact
        .weight(r.weight_id)
        .ok_or(VerifyError::MissingBinding("weight"))?;
    let shape = w_tensor.shape();
    if shape.len() != 2 {
        return Err(VerifyError::MissingBinding("weight_shape"));
    }
    let rows = shape[0] as usize;
    let cols = shape[1] as usize;
    // Weight as i8 (values are in i8 range by the scale dtype invariant).
    let w_i8: Vec<i8> = w_tensor.data().iter().map(|c| c.value() as i8).collect();
    let bias: Vec<i64> = match r.bias_id {
        Some(id) => {
            let b = artifact
                .weight(id)
                .ok_or(VerifyError::MissingBinding("bias"))?;
            b.data().iter().map(|c| c.value()).collect()
        }
        None => alloc::vec![0i64; rows],
    };
    let challenge: Vec<Fp61> = next_freivalds_r(transcript, rows);
    let v = precompute_v(&challenge, &w_i8, rows, cols);
    if check_linear_biased(&v, &r.input, &bias, &challenge, &r.output) {
        Ok(())
    } else {
        Err(VerifyError::FreivaldsCheckFailed { op_id: r.op_id })
    }
}

/// Exactly recompute the requantization and compare to the record (specs.md §8.1).
fn check_requant(r: &pwm_core::trace::RequantRec) -> Result<(), VerifyError> {
    let mode = Rounding::from_discriminant(r.rounding)
        .ok_or(VerifyError::ExactReplayMismatch { op_id: r.op_id })?;
    if r.input.len() != r.output.len() {
        return Err(VerifyError::ExactReplayMismatch { op_id: r.op_id });
    }
    for (&n, &claimed) in r.input.iter().zip(r.output.iter()) {
        let expected = requantize(n, r.shift, r.zero_point, r.clamp_lo, r.clamp_hi, mode);
        if expected != claimed {
            return Err(VerifyError::ExactReplayMismatch { op_id: r.op_id });
        }
    }
    Ok(())
}

/// Exactly recompute the affine-free LayerNorm and compare to the record
/// (specs.md §8.3, integer mean/variance + committed inverse-sqrt table).
fn check_layernorm(
    artifact: &AuditArtifact,
    r: &pwm_core::trace::LayerNormRec,
) -> Result<(), VerifyError> {
    let table = artifact
        .table(r.table_id)
        .ok_or(VerifyError::MissingBinding("layernorm_table"))?;
    let mode = Rounding::from_discriminant(r.rounding)
        .ok_or(VerifyError::ExactReplayMismatch { op_id: r.op_id })?;
    let expected = layernorm(&r.input, table, r.shift, r.clamp_lo, r.clamp_hi, mode).ok_or(
        VerifyError::ActivationDomain {
            table_id: r.table_id,
        },
    )?;
    if expected != r.output {
        return Err(VerifyError::ExactReplayMismatch { op_id: r.op_id });
    }
    Ok(())
}

/// Exactly replay the committed-table read and compare to the record (specs.md §8.3).
fn check_activation(
    artifact: &AuditArtifact,
    r: &pwm_core::trace::ActivationRec,
) -> Result<(), VerifyError> {
    let table = artifact
        .table(r.table_id)
        .ok_or(VerifyError::MissingBinding("table"))?;
    if r.input.len() != r.output.len() {
        return Err(VerifyError::ExactReplayMismatch { op_id: r.op_id });
    }
    for (&x, &claimed) in r.input.iter().zip(r.output.iter()) {
        let expected = table.eval(x).ok_or(VerifyError::ActivationDomain {
            table_id: r.table_id,
        })?;
        if expected != claimed {
            return Err(VerifyError::ExactReplayMismatch { op_id: r.op_id });
        }
    }
    Ok(())
}
