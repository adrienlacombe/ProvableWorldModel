// SPDX-License-Identifier: Apache-2.0
//! Blake2s commitments for the model, quantization, and planner config, and the
//! weight Merkle root (RFC-0014 §3).
//!
//! V0 fixes one hash primitive (Blake2s-256) for every commitment. Blake2s is the
//! native sponge of the Fiat-Shamir transcript, so the prover and the `no_std`
//! verifier share exactly one hash implementation. Each commitment is
//! **domain-separated and length-prefixed** so a model digest can never collide
//! with a quantization,
//! planner, output, or tensor digest. Each commitment binds exactly the fields
//! the soundness argument requires (`specs.md §13`):
//! changing any bound field changes the commitment.

use alloc::vec::Vec;

use crate::fixed_point::{OverflowPolicy, Rounding};
use crate::serialize::{canonical_bytes, CanonicalEncode};
use crate::tensor::{Scale, Tensor};
use crate::transcript::blake2s256;

/// Domain tag for `model_commitment`.
pub const TAG_MODEL: &[u8; 16] = b"pwm.model.v1\0\0\0\0";
/// Domain tag for `quantization_commitment`.
pub const TAG_QUANT: &[u8; 16] = b"pwm.quant.v1\0\0\0\0";
/// Domain tag for `planner_config_commitment`.
pub const TAG_PLANNER: &[u8; 16] = b"pwm.plan.v1\0\0\0\0\0";
/// Domain tag for `claimed_output_commitment`.
pub const TAG_OUTPUT: &[u8; 16] = b"pwm.out.v1\0\0\0\0\0\0";
/// Domain tag for a committed input tensor.
pub const TAG_TENSOR: &[u8; 16] = b"pwm.tensor.v1\0\0\0";
/// Domain tag for the committed predictor input buffers.
pub const TAG_PINPUTS: &[u8; 16] = b"pwm.pinput.v1\0\0\0";
/// Domain tag for a weight Merkle leaf.
pub const TAG_WLEAF: &[u8; 16] = b"pwm.wleaf.v1\0\0\0\0";
/// Domain tag for a weight Merkle node.
pub const TAG_WNODE: &[u8; 16] = b"pwm.wnode.v1\0\0\0\0";

/// Domain-separated, length-prefixed commitment (RFC-0014 §3):
/// `blake2s256(domain_tag || u64_le(payload.len()) || payload)`.
pub fn commit(domain_tag: &[u8; 16], payload: &[u8]) -> [u8; 32] {
    let mut input = Vec::with_capacity(16 + 8 + payload.len());
    input.extend_from_slice(domain_tag);
    input.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    input.extend_from_slice(payload);
    blake2s256(&input)
}

/// Commit a single committed input tensor: `commit(TAG_TENSOR, canonical_bytes(t))`.
pub fn commit_tensor(tensor: &Tensor) -> [u8; 32] {
    commit(TAG_TENSOR, &canonical_bytes(tensor))
}

/// Commit the claimed output tensors in declared order:
/// `commit(TAG_OUTPUT, canonical_bytes(outputs))`.
pub fn claimed_output_commitment(outputs: &[Tensor]) -> [u8; 32] {
    let outputs = outputs.to_vec();
    commit(TAG_OUTPUT, &canonical_bytes(&outputs))
}

/// Commit the predictor's seeded input buffers in declared order (the public latent
/// history and action embedding): `commit(TAG_PINPUTS, count ‖ (buf_id ‖ values)*)`.
/// Binds the inputs a committed predictor statement is about, so a proof cannot be
/// replayed against different inputs.
pub fn predictor_inputs_commitment(inputs: &[(u32, Vec<i64>)]) -> [u8; 32] {
    let mut payload = Vec::new();
    (inputs.len() as u32).encode(&mut payload);
    for (id, values) in inputs {
        id.encode(&mut payload);
        values.encode(&mut payload);
    }
    commit(TAG_PINPUTS, &payload)
}

/// The weight Merkle root (`blake2s_merkle_v1`, RFC-0014 §3): leaves are
/// `blake2s256(TAG_WLEAF || u32(tensor_id) || canonical_bytes(Tensor))`, **sorted
/// by ascending `tensor_id`**, folded with
/// `node(a,b) = blake2s256(TAG_WNODE || a || b)`; an odd level duplicates its last
/// node. The empty set hashes the node tag as a fixed sentinel.
pub fn weights_root(tensors: &[Tensor]) -> [u8; 32] {
    let mut leaves: Vec<(u32, [u8; 32])> = tensors
        .iter()
        .map(|t| {
            let mut payload = Vec::new();
            payload.extend_from_slice(TAG_WLEAF);
            t.tensor_id().encode(&mut payload);
            payload.extend_from_slice(&canonical_bytes(t));
            (t.tensor_id(), blake2s256(&payload))
        })
        .collect();
    leaves.sort_by_key(|(id, _)| *id);

    let mut level: Vec<[u8; 32]> = leaves.into_iter().map(|(_, h)| h).collect();
    if level.is_empty() {
        return blake2s256(TAG_WNODE);
    }
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut i = 0;
        while i < level.len() {
            let a = level[i];
            let b = if i + 1 < level.len() {
                level[i + 1]
            } else {
                level[i]
            };
            let mut payload = Vec::with_capacity(16 + 64);
            payload.extend_from_slice(TAG_WNODE);
            payload.extend_from_slice(&a);
            payload.extend_from_slice(&b);
            next.push(blake2s256(&payload));
            i += 2;
        }
        level = next;
    }
    level[0]
}

/// The model binding (RFC-0014 §3, security B2–B4/B15): everything
/// `model_commitment` binds. The architecture/ops sub-structure is committed by
/// the manifest writer (#36/#38) into `architecture_commitment`; this binds that
/// sub-commitment together with the weight root and the version fields, so any
/// change to architecture, weights, or versions changes the commitment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelBinding {
    /// Sub-commitment over the architecture + ops + shapes structure (#36).
    pub architecture_commitment: [u8; 32],
    /// The weight Merkle root ([`weights_root`]).
    pub weights_root: [u8; 32],
    /// The relation version this model serves.
    pub relation_version: u32,
    /// The canonical serialization version.
    pub serialization_version: u32,
}

impl CanonicalEncode for ModelBinding {
    fn encode(&self, out: &mut Vec<u8>) {
        self.architecture_commitment.encode(out);
        self.weights_root.encode(out);
        self.relation_version.encode(out);
        self.serialization_version.encode(out);
    }
}

impl ModelBinding {
    /// The V0 model binding for a given architecture sub-commitment and weight root,
    /// pinning the V0 relation and serialization versions the prover and verifier
    /// must agree on. The single constructor both sides call, so a version bump
    /// changes one place instead of drifting across call sites.
    pub fn v0(architecture_commitment: [u8; 32], weights_root: [u8; 32]) -> Self {
        ModelBinding {
            architecture_commitment,
            weights_root,
            relation_version: crate::audit::RELATION_VERSION,
            serialization_version: crate::audit::SERIALIZATION_VERSION,
        }
    }

    /// `commit(TAG_MODEL, canonical_bytes(self))`.
    pub fn commitment(&self) -> [u8; 32] {
        commit(TAG_MODEL, &canonical_bytes(self))
    }
}

/// The quantization binding (RFC-0014 §3, security B5–B10): rounding, overflow
/// policy, the scale table, and the activation/lookup tables commitment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantBinding {
    /// The single active rounding mode.
    pub default_rounding: Rounding,
    /// The overflow policy (`Reject` in V0).
    pub overflow_policy: OverflowPolicy,
    /// The scale table.
    pub scales: Vec<Scale>,
    /// Commitment over the committed activation/lookup tables.
    pub activation_tables_commitment: [u8; 32],
}

impl CanonicalEncode for QuantBinding {
    fn encode(&self, out: &mut Vec<u8>) {
        self.default_rounding.encode(out);
        self.overflow_policy.encode(out);
        self.scales.encode(out);
        self.activation_tables_commitment.encode(out);
    }
}

impl QuantBinding {
    /// The V0 quantization binding: round-nearest-ties-even and overflow=reject (the
    /// only V0 policies), for the given scale table and committed-tables commitment.
    /// Pins the two policy fields so they cannot diverge between prover and verifier.
    pub fn v0(scales: Vec<Scale>, activation_tables_commitment: [u8; 32]) -> Self {
        QuantBinding {
            default_rounding: Rounding::NearestTiesToEven,
            overflow_policy: OverflowPolicy::Reject,
            scales,
            activation_tables_commitment,
        }
    }

    /// `commit(TAG_QUANT, canonical_bytes(self))`.
    pub fn commitment(&self) -> [u8; 32] {
        commit(TAG_QUANT, &canonical_bytes(self))
    }
}

/// The planner-config binding (RFC-0014 §3, security B11): horizon, action block,
/// candidate count, and the tie-break rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlannerBinding {
    /// Rollout horizon.
    pub horizon: u32,
    /// Action block length.
    pub action_block: u32,
    /// Candidate count `S`.
    pub candidate_count: u32,
    /// Identifier of the tie-break rule.
    pub tie_break_rule_id: u32,
}

impl CanonicalEncode for PlannerBinding {
    fn encode(&self, out: &mut Vec<u8>) {
        self.horizon.encode(out);
        self.action_block.encode(out);
        self.candidate_count.encode(out);
        self.tie_break_rule_id.encode(out);
    }
}

impl PlannerBinding {
    /// The P0/V0 planner sentinel: an all-zero config (no planner). The single
    /// definition of the "no planner" binding the P0 prover and verifier both bind.
    pub fn p0_sentinel() -> Self {
        PlannerBinding {
            horizon: 0,
            action_block: 0,
            candidate_count: 0,
            tie_break_rule_id: 0,
        }
    }

    /// `commit(TAG_PLANNER, canonical_bytes(self))`.
    pub fn commitment(&self) -> [u8; 32] {
        commit(TAG_PLANNER, &canonical_bytes(self))
    }
}
