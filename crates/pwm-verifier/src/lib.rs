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

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use pwm_core::block::{block_architecture_commitment, block_transcript, Block, BlockOp};
use pwm_core::predictor::{gate_vec, matmul, modulate_vec, residual_add, softmax_rows};
use pwm_core::tensor::Tensor;

use pwm_core::audit::{
    audit_transcript, next_freivalds_r, output_tensor, predictor_transcript, relation_id,
    AuditArtifact, PlanningProof, PredictorArtifact, RolloutProof, ARTIFACT_VERSION, RELATION_MLP,
    RELATION_PREDICTOR,
};
use pwm_core::commit::{
    claimed_output_commitment, predictor_inputs_commitment, weights_root, ModelBinding,
    PlannerBinding, QuantBinding,
};
use pwm_core::field::Fp61;
use pwm_core::fixed_point::{requantize, valid_shift, Rounding};
use pwm_core::freivalds::{
    check_linear_biased, dims_within_soundness_margin, precompute_v, within_operand_bound,
};
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
    /// A Freivalds operand (input, bias, or claimed accumulator) escaped the
    /// sound integer range `[-SAFE_HI, SAFE_HI]`, so the mod-`p` check could not
    /// certify integer equality (the accumulator-aliasing guard, specs.md §7 /
    /// INV-FP-10). Fail-closed: an out-of-envelope accumulator is rejected.
    AccumulatorRange {
        /// Op id of the offending linear.
        op_id: u32,
    },
    /// An exactly-recomputed op (requant / table) did not match its record.
    ExactReplayMismatch {
        /// Op id of the offending op.
        op_id: u32,
    },
    /// A proof-supplied right-shift exceeded [`pwm_core::fixed_point::MAX_SHIFT`]
    /// (would make `1i64 << r` negative or overflow). Fail-closed (WQ-08).
    InvalidShift {
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
    /// A candidate's P0 proof failed to verify; `cause` is the inner rejection.
    Candidate {
        /// Candidate index.
        index: usize,
        /// The underlying P0 rejection (preserved for diagnosis).
        cause: alloc::boxed::Box<VerifyError>,
    },
    /// A rollout step's P0 proof failed to verify; `cause` is the inner rejection.
    RolloutStep {
        /// Rollout step index.
        step: usize,
        /// The underlying P0 rejection (preserved for diagnosis).
        cause: alloc::boxed::Box<VerifyError>,
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

/// Source of Freivalds challenge vectors. `Fiat` derives them non-interactively
/// from the committed trace (computational soundness under the hash); `Secret`
/// uses verifier-supplied vectors the prover never sees (unconditional Freivalds
/// soundness — the interactive mode, backlog D-802).
enum Challenges<'a> {
    /// Fiat-Shamir transcript bound to the public input and trace root.
    Fiat(Transcript),
    /// Verifier-secret challenge vectors, one per `Linear` op in trace order.
    Secret {
        /// The secret challenge vectors.
        rs: &'a [Vec<Fp61>],
        /// Cursor into `rs`.
        idx: usize,
    },
    /// Batched/amortized mode (backlog D-806): `r` and the precomputed `v = rᵀW`
    /// are keyed by `weight_id` and **shared across all candidates**, so `v` is
    /// computed once per distinct weight matrix instead of once per instance.
    Batched {
        /// `weight_id -> challenge vector r` (one per distinct weight).
        rmap: &'a BTreeMap<u32, Vec<Fp61>>,
        /// `weight_id -> precomputed v = rᵀW`.
        vmap: &'a BTreeMap<u32, Vec<Fp61>>,
    },
    /// Routine/sampled audit mode (backlog D-801): Freivalds-check only the sampled
    /// subset of linear ops (statistical, sub-linear coverage). Unsampled linears
    /// are trusted; non-linear ops are still exactly recomputed.
    Sampled {
        /// Transcript used to derive `r` for the sampled ops.
        t: Transcript,
        /// Indices (in linear-op order) that are checked.
        sampled: BTreeSet<usize>,
        /// Cursor over linear ops.
        linear_idx: usize,
    },
}

/// A Freivalds challenge for one linear op: `(r, v)` where `v = rᵀW`.
type LinearChallenge = (Vec<Fp61>, Vec<Fp61>);

/// Look up a weight tensor as a flat `i8` matrix with its `(rows, cols)`.
fn weight_i8(
    artifact: &AuditArtifact,
    weight_id: u32,
) -> Result<(Vec<i8>, usize, usize), VerifyError> {
    let w = artifact
        .weight(weight_id)
        .ok_or(VerifyError::MissingBinding("weight"))?;
    let shape = w.shape();
    if shape.len() != 2 {
        return Err(VerifyError::MissingBinding("weight_shape"));
    }
    let i8s = w.data().iter().map(|c| c.value() as i8).collect();
    Ok((i8s, shape[0] as usize, shape[1] as usize))
}

impl Challenges<'_> {
    /// The Freivalds challenge `r` and precomputed `v = rᵀW` for a linear op, or
    /// `None` to skip the check (only in `Sampled` mode — the op is trusted). In
    /// `Fiat`/`Secret` mode `v` is computed per call; in `Batched` mode both are
    /// looked up from the shared per-weight maps (computed once).
    fn linear_step(
        &mut self,
        artifact: &AuditArtifact,
        rec: &pwm_core::trace::LinearRec,
    ) -> Result<Option<LinearChallenge>, VerifyError> {
        match self {
            Challenges::Fiat(t) => {
                let (w, rows, cols) = weight_i8(artifact, rec.weight_id)?;
                let r = next_freivalds_r(t, rows);
                let v = precompute_v(&r, &w, rows, cols);
                Ok(Some((r, v)))
            }
            Challenges::Secret { rs, idx } => {
                let (w, rows, cols) = weight_i8(artifact, rec.weight_id)?;
                let r = rs
                    .get(*idx)
                    .filter(|r| r.len() == rows)
                    .ok_or(VerifyError::FreivaldsCheckFailed { op_id: rec.op_id })?
                    .clone();
                *idx += 1;
                let v = precompute_v(&r, &w, rows, cols);
                Ok(Some((r, v)))
            }
            Challenges::Batched { rmap, vmap } => {
                let r = rmap
                    .get(&rec.weight_id)
                    .ok_or(VerifyError::FreivaldsCheckFailed { op_id: rec.op_id })?
                    .clone();
                let v = vmap
                    .get(&rec.weight_id)
                    .ok_or(VerifyError::FreivaldsCheckFailed { op_id: rec.op_id })?
                    .clone();
                Ok(Some((r, v)))
            }
            Challenges::Sampled {
                t,
                sampled,
                linear_idx,
            } => {
                let this = *linear_idx;
                *linear_idx += 1;
                if !sampled.contains(&this) {
                    return Ok(None); // trusted (statistical coverage)
                }
                let (w, rows, cols) = weight_i8(artifact, rec.weight_id)?;
                let r = next_freivalds_r(t, rows);
                let v = precompute_v(&r, &w, rows, cols);
                Ok(Some((r, v)))
            }
        }
    }
}

/// Verify a commit-and-audit proof (non-interactive Fiat-Shamir). Returns `Ok(())`
/// iff every check passes.
pub fn verify(artifact: &AuditArtifact) -> Result<(), VerifyError> {
    let troot = artifact.trace_root();
    let mut ch = Challenges::Fiat(audit_transcript(&artifact.public_input, &troot));
    verify_with(artifact, &mut ch)
}

/// Verify a proof in **interactive (verifier-secret) mode** (backlog D-802): the
/// Freivalds challenge vectors `secret_r` (one per `Linear` op in trace order) are
/// chosen by the verifier and never revealed to the prover, giving unconditional
/// (not merely computational) Freivalds soundness. All other checks are identical.
pub fn verify_interactive(
    artifact: &AuditArtifact,
    secret_r: &[Vec<Fp61>],
) -> Result<(), VerifyError> {
    let mut ch = Challenges::Secret {
        rs: secret_r,
        idx: 0,
    };
    verify_with(artifact, &mut ch)
}

/// Verify a proof in **routine/sampled audit mode** (backlog D-801): Freivalds-check
/// only a sampled subset of `sample` linear ops (chosen deterministically from
/// `seed` and the trace root); the remaining linears are *trusted*. All non-linear
/// ops, the wiring, and the output/commitment checks are still exact and complete.
///
/// This is **statistical, sub-linear coverage**, not full soundness — an undetected
/// wrong accumulator in an unsampled linear can slip through. Returns the number of
/// linear ops actually Freivalds-checked so the caller can reason about coverage.
/// Use [`verify`] for full soundness.
pub fn verify_sampled(
    artifact: &AuditArtifact,
    sample: usize,
    seed: u64,
) -> Result<usize, VerifyError> {
    // Count linear ops in the trace.
    let n_linear = artifact
        .trace
        .iter()
        .filter(|r| matches!(r, OpRecord::Linear(_)))
        .count();
    // Deterministically choose `min(sample, n_linear)` linear-op indices from a
    // transcript bound to the trace root and the seed.
    let troot = artifact.trace_root();
    let mut sel = audit_transcript(&artifact.public_input, &troot);
    sel.absorb_u64(b"audit.seed", seed);
    let mut sampled = BTreeSet::new();
    let want = sample.min(n_linear);
    let mut guard = 0u32;
    while sampled.len() < want && guard < 100_000 {
        let idx = (sel.challenge_u64(b"audit.pick") as usize) % n_linear.max(1);
        sampled.insert(idx);
        guard += 1;
    }
    let checked = sampled.len();
    let mut ch = Challenges::Sampled {
        t: audit_transcript(&artifact.public_input, &troot),
        sampled,
        linear_idx: 0,
    };
    verify_with(artifact, &mut ch)?;
    Ok(checked)
}

/// The shared verification body, parameterized by the Freivalds challenge source.
fn verify_with(artifact: &AuditArtifact, ch: &mut Challenges<'_>) -> Result<(), VerifyError> {
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
    let model_commitment =
        ModelBinding::v0(artifact.graph.commitment(), weights_root(&artifact.weights)).commitment();
    if model_commitment != pi.model_commitment {
        return Err(VerifyError::CommitmentMismatch("model"));
    }

    let quantization_commitment = QuantBinding::v0(
        artifact.scales.clone(),
        activation_tables_commitment(&artifact.tables),
    )
    .commitment();
    if quantization_commitment != pi.quantization_commitment {
        return Err(VerifyError::CommitmentMismatch("quantization"));
    }

    if PlannerBinding::p0_sentinel().commitment() != pi.planner_config_commitment {
        return Err(VerifyError::CommitmentMismatch("planner"));
    }

    // 3. The trace must conform to the committed op graph (same kinds/ids/dims).
    check_graph_conformance(artifact)?;

    // 4. Walk the trace, threading the running activation; Freivalds challenges
    //    come from `ch` (Fiat-Shamir-from-the-committed-trace, or verifier-secret).
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
            OpRecord::Linear(r) => {
                if let Some((challenge, v)) = ch.linear_step(artifact, r)? {
                    check_linear(artifact, r, &v, &challenge)?;
                }
            }
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
        verify(candidate).map_err(|cause| VerifyError::Candidate {
            index,
            cause: alloc::boxed::Box::new(cause),
        })?;
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
        if mse_cost(&output, &proof.goal) != Some(proof.costs[index]) {
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

/// Verify fixed-candidate planning in **batched/amortized mode** (backlog D-806).
///
/// All candidates share the same committed model, so the Freivalds `v = rᵀW` is
/// precomputed **once per distinct weight matrix** and reused across every
/// candidate × instance, instead of once per instance. The shared challenges are
/// bound to *all* candidate trace roots (so they depend on every committed
/// accumulator). Verdict-equivalent to [`verify_planning`]; only cheaper. Falls
/// back to per-candidate verification if the candidates do not share a model.
pub fn verify_planning_batched(proof: &PlanningProof) -> Result<(), VerifyError> {
    if proof.candidates.is_empty() {
        return Err(VerifyError::NoCandidates);
    }
    if proof.costs.len() != proof.candidates.len() {
        return Err(VerifyError::ArgminViolation);
    }
    let c0 = &proof.candidates[0];
    for c in &proof.candidates[1..] {
        let same = c.graph == c0.graph
            && c.weights == c0.weights
            && c.tables == c0.tables
            && c.public_input.model_commitment == c0.public_input.model_commitment
            && c.public_input.quantization_commitment == c0.public_input.quantization_commitment;
        if !same {
            return verify_planning(proof);
        }
    }

    // Bind the shared challenges to every candidate's committed trace root.
    let mut t = Transcript::new(b"pwm.batched.v1");
    t.absorb_u64(b"candidates", proof.candidates.len() as u64);
    for c in &proof.candidates {
        t.absorb(b"trace_root", &c.trace_root());
    }
    // One r + precomputed v per distinct weight matrix (sorted, deterministic).
    let mut weight_ids: Vec<u32> = c0
        .graph
        .ops
        .iter()
        .filter_map(|op| match op {
            OpSpec::Linear { weight_id, .. } => Some(*weight_id),
            _ => None,
        })
        .collect();
    weight_ids.sort_unstable();
    weight_ids.dedup();
    let mut rmap: BTreeMap<u32, Vec<Fp61>> = BTreeMap::new();
    let mut vmap: BTreeMap<u32, Vec<Fp61>> = BTreeMap::new();
    for wid in weight_ids {
        let (w, rows, cols) = weight_i8(c0, wid)?;
        let r = t.challenge_fp61_vec(b"freivalds.r", rows);
        let v = precompute_v(&r, &w, rows, cols); // computed once per weight
        rmap.insert(wid, r);
        vmap.insert(wid, v);
    }

    for (index, candidate) in proof.candidates.iter().enumerate() {
        let mut ch = Challenges::Batched {
            rmap: &rmap,
            vmap: &vmap,
        };
        verify_with(candidate, &mut ch).map_err(|cause| VerifyError::Candidate {
            index,
            cause: alloc::boxed::Box::new(cause),
        })?;
        let output: Vec<i64> = candidate
            .claimed_output
            .data()
            .iter()
            .map(|c| c.value())
            .collect();
        if output.len() != proof.goal.len()
            || mse_cost(&output, &proof.goal) != Some(proof.costs[index])
        {
            return Err(VerifyError::CostMismatch { index });
        }
    }

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
        verify(artifact).map_err(|cause| VerifyError::RolloutStep {
            step,
            cause: alloc::boxed::Box::new(cause),
        })?;
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
/// AdaLN modulate / gate / residual add). Returns the block's output buffer.
///
/// The Fiat-Shamir transcript is built **internally** from the proven block and its
/// inputs via [`block_transcript`], which absorbs the block root (binding every
/// claimed `out`) before any challenge is squeezed. Building it here, rather than
/// taking it from the caller, makes the non-adaptive binding impossible to get
/// wrong. `weights` and `tables` are the committed bindings.
pub fn verify_block(
    block: &Block,
    weights: &[Tensor],
    tables: &[pwm_core::tables::ActivationTable],
    inputs: &[(u32, Vec<i64>)],
) -> Result<Vec<i64>, VerifyError> {
    let mut transcript = block_transcript(block, inputs);
    audit_block(block, weights, tables, inputs, &mut transcript)
}

/// The shared block-walk audit, parameterized by a caller-built Fiat-Shamir
/// transcript: the standalone [`block_transcript`] for [`verify_block`], or the
/// commitment-bound [`predictor_transcript`] for [`verify_predictor`]. Private so
/// the transcript binding cannot be supplied incorrectly from outside the crate.
fn audit_block(
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
                range_guard_linear(*op_id, cols, &x, &bias, out)?;
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
                if !valid_shift(*shift) {
                    return Err(VerifyError::InvalidShift { op_id: *op_id });
                }
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
                if !valid_shift(*shift) {
                    return Err(VerifyError::InvalidShift { op_id: *op_id });
                }
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
                if !valid_shift(*shift_bits) {
                    return Err(VerifyError::InvalidShift { op_id: *op_id });
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
                if !valid_shift(*shift_bits) {
                    return Err(VerifyError::InvalidShift { op_id: *op_id });
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
            BlockOp::Slice {
                op_id,
                in_buf,
                out_buf,
                start,
                len,
                out,
            } => {
                let x = get(&bufs, *in_buf)?;
                let s = *start as usize;
                let e = s + *len as usize;
                if e > x.len() || x[s..e] != out[..] {
                    return Err(VerifyError::BlockOpMismatch { op_id: *op_id });
                }
                bufs.insert(*out_buf, out.clone());
            }
            BlockOp::Concat {
                op_id,
                in_bufs,
                out_buf,
                out,
            } => {
                let mut expected = Vec::new();
                for id in in_bufs {
                    expected.extend(get(&bufs, *id)?);
                }
                if &expected != out {
                    return Err(VerifyError::BlockOpMismatch { op_id: *op_id });
                }
                bufs.insert(*out_buf, out.clone());
            }
            BlockOp::BatchedLinear {
                op_id,
                weight_id,
                bias_id,
                in_buf,
                out_buf,
                out,
                seq,
            } => {
                let x = get(&bufs, *in_buf)?;
                let w = find_weight(*weight_id)?;
                let shape = w.shape();
                if shape.len() != 2 {
                    return Err(VerifyError::MissingBinding("block_weight_shape"));
                }
                let rows = shape[0] as usize;
                let cols = shape[1] as usize;
                let w_i8s: Vec<i8> = w.data().iter().map(|c| c.value() as i8).collect();
                let s = *seq as usize;
                if x.len() != s * cols || out.len() != s * rows {
                    return Err(VerifyError::BlockOpMismatch { op_id: *op_id });
                }
                let bias: Vec<i64> = match bias_id {
                    Some(bid) => find_weight(*bid)?
                        .data()
                        .iter()
                        .map(|c| c.value())
                        .collect(),
                    None => alloc::vec![0i64; rows],
                };
                // Soundness range guard over every row's input and accumulator.
                range_guard_linear(*op_id, cols, &x, &bias, out)?;
                // One challenge + one precomputed v, reused across all `seq` rows.
                let r = next_freivalds_r(transcript, rows);
                let v = precompute_v(&r, &w_i8s, rows, cols);
                for tt in 0..s {
                    let xrow = &x[tt * cols..(tt + 1) * cols];
                    let orow = &out[tt * rows..(tt + 1) * rows];
                    if !check_linear_biased(&v, xrow, &bias, &r, orow) {
                        return Err(VerifyError::FreivaldsCheckFailed { op_id: *op_id });
                    }
                }
                bufs.insert(*out_buf, out.clone());
            }
        }
    }
    get(&bufs, block.output_buf)
}

/// Verify a commitment-bound predictor proof (`RELATION_PREDICTOR`): the
/// first-class, commitment-wrapped counterpart of [`verify_block`].
///
/// Unlike a bare `verify_block` (which audits arithmetic against caller-supplied
/// weights), this recomputes the model commitment (block architecture + weight
/// root), the quantization commitment (scales + tables), the seeded-input
/// commitment, and the claimed-output commitment, and checks each against the public
/// input — so accepting the proof means the *committed* model ran on the *committed*
/// inputs. The Fiat-Shamir transcript is the commitment-bound [`predictor_transcript`]
/// (public input + block witness), built internally, so the challenge is
/// statement-bound and non-adaptive.
pub fn verify_predictor(artifact: &PredictorArtifact) -> Result<(), VerifyError> {
    let pi = &artifact.public_input;

    // 1. Version + relation gating.
    if artifact.artifact_version != ARTIFACT_VERSION {
        return Err(VerifyError::UnsupportedArtifactVersion(
            artifact.artifact_version,
        ));
    }
    if pi.relation_id != relation_id(RELATION_PREDICTOR) {
        return Err(VerifyError::UnsupportedRelation);
    }
    if pi.statement_type != StatementType::P0Step {
        return Err(VerifyError::RelationMismatch);
    }

    // 2. Recompute and check the bound commitments.
    let model_commitment = ModelBinding::v0(
        block_architecture_commitment(&artifact.block),
        weights_root(&artifact.weights),
    )
    .commitment();
    if model_commitment != pi.model_commitment {
        return Err(VerifyError::CommitmentMismatch("model"));
    }
    let quantization_commitment = QuantBinding::v0(
        artifact.scales.clone(),
        activation_tables_commitment(&artifact.tables),
    )
    .commitment();
    if quantization_commitment != pi.quantization_commitment {
        return Err(VerifyError::CommitmentMismatch("quantization"));
    }
    if PlannerBinding::p0_sentinel().commitment() != pi.planner_config_commitment {
        return Err(VerifyError::CommitmentMismatch("planner"));
    }
    if pi.latent_history_commitment != Some(predictor_inputs_commitment(&artifact.inputs)) {
        return Err(VerifyError::CommitmentMismatch("inputs"));
    }

    // 3. Audit the block under the commitment-bound transcript (binds the public
    //    input and the block witness before any challenge is squeezed).
    let mut transcript = predictor_transcript(pi, &artifact.block, &artifact.inputs);
    let computed = audit_block(
        &artifact.block,
        &artifact.weights,
        &artifact.tables,
        &artifact.inputs,
        &mut transcript,
    )?;

    // 4. The computed output must equal the claimed output, and its commitment must
    //    match the public input.
    if artifact.claimed_output.data().len() != computed.len() {
        return Err(VerifyError::OutputMismatch);
    }
    for (cell, &v) in artifact.claimed_output.data().iter().zip(computed.iter()) {
        if cell.value() != v {
            return Err(VerifyError::OutputMismatch);
        }
    }
    let rebuilt = output_tensor(
        artifact.claimed_output.tensor_id(),
        artifact.claimed_output.scale_id(),
        &computed,
    )
    .map_err(|_| VerifyError::Encoding)?;
    if claimed_output_commitment(core::slice::from_ref(&rebuilt)) != pi.claimed_output_commitment {
        return Err(VerifyError::OutputCommitmentMismatch);
    }

    Ok(())
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

/// Freivalds-check `out = W·x + bias` (specs.md §7) with the supplied challenge
/// `r` and the precomputed `v = rᵀW` (lengths must match the op's dimensions).
fn check_linear(
    artifact: &AuditArtifact,
    rec: &pwm_core::trace::LinearRec,
    v: &[Fp61],
    r: &[Fp61],
) -> Result<(), VerifyError> {
    let (_, rows, cols) = weight_i8(artifact, rec.weight_id)?;
    if r.len() != rows || v.len() != cols || rec.output.len() != rows || rec.input.len() != cols {
        return Err(VerifyError::FreivaldsCheckFailed { op_id: rec.op_id });
    }
    let bias: Vec<i64> = match rec.bias_id {
        Some(id) => {
            let b = artifact
                .weight(id)
                .ok_or(VerifyError::MissingBinding("bias"))?;
            b.data().iter().map(|c| c.value()).collect()
        }
        None => alloc::vec![0i64; rows],
    };
    // Soundness range guard (specs.md §7 / INV-FP-10): bound the operands so the
    // mod-`p` Freivalds check certifies integer equality, not mere congruence.
    range_guard_linear(rec.op_id, cols, &rec.input, &bias, &rec.output)?;
    if check_linear_biased(v, &rec.input, &bias, r, &rec.output) {
        Ok(())
    } else {
        Err(VerifyError::FreivaldsCheckFailed { op_id: rec.op_id })
    }
}

/// The Freivalds soundness range guard, emitted as a dedicated [`VerifyError::AccumulatorRange`]
/// for a clear rejection code. Mirrors the backstop folded into
/// [`pwm_core::freivalds::check_linear_biased`] so the bound lives in one place
/// (`pwm-core`) and cannot drift between the trace, block, and batched paths.
fn range_guard_linear(
    op_id: u32,
    cols: usize,
    x: &[i64],
    bias: &[i64],
    out: &[i64],
) -> Result<(), VerifyError> {
    if dims_within_soundness_margin(cols)
        && within_operand_bound(x)
        && within_operand_bound(bias)
        && within_operand_bound(out)
    {
        Ok(())
    } else {
        Err(VerifyError::AccumulatorRange { op_id })
    }
}

/// Exactly recompute the requantization and compare to the record (specs.md §8.1).
fn check_requant(r: &pwm_core::trace::RequantRec) -> Result<(), VerifyError> {
    let mode = Rounding::from_discriminant(r.rounding)
        .ok_or(VerifyError::ExactReplayMismatch { op_id: r.op_id })?;
    if !valid_shift(r.shift) {
        return Err(VerifyError::InvalidShift { op_id: r.op_id });
    }
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
    if !valid_shift(r.shift) {
        return Err(VerifyError::InvalidShift { op_id: r.op_id });
    }
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
