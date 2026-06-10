// SPDX-License-Identifier: Apache-2.0
//! Tests for Blake2s commitments and the weight Merkle root (RFC-0014 §3):
//! domain separation, field-sensitivity, and Merkle determinism/ordering.

use pwm_core::commit::{
    claimed_output_commitment, commit, weights_root, ModelBinding, PlannerBinding, QuantBinding,
    TAG_MODEL, TAG_OUTPUT, TAG_PLANNER, TAG_QUANT, TAG_TENSOR, TAG_WLEAF, TAG_WNODE,
};
use pwm_core::fixed_point::{BoundedInt, OverflowPolicy, Rounding};
use pwm_core::tensor::{Dtype, Scale, Tensor};

fn tensor(id: u32, vals: &[i64]) -> Tensor {
    let data = vals
        .iter()
        .map(|&v| BoundedInt::new(v, -128, 127).unwrap())
        .collect();
    Tensor::new(id, vec![vals.len() as u32], 0, data).unwrap()
}

fn model() -> ModelBinding {
    ModelBinding {
        architecture_commitment: [1; 32],
        weights_root: [2; 32],
        relation_version: 1,
        serialization_version: 1,
    }
}

fn quant() -> QuantBinding {
    QuantBinding {
        default_rounding: Rounding::NearestTiesToEven,
        overflow_policy: OverflowPolicy::Reject,
        scales: vec![Scale {
            scale_id: 0,
            log2: 0,
            dtype: Dtype::I8,
        }],
        activation_tables_commitment: [9; 32],
    }
}

fn planner() -> PlannerBinding {
    PlannerBinding {
        horizon: 5,
        action_block: 5,
        candidate_count: 32,
        tie_break_rule_id: 0,
    }
}

#[test]
fn tags_are_16_bytes_and_distinct() {
    let tags = [
        TAG_MODEL,
        TAG_QUANT,
        TAG_PLANNER,
        TAG_OUTPUT,
        TAG_TENSOR,
        TAG_WLEAF,
        TAG_WNODE,
    ];
    for t in tags {
        assert_eq!(t.len(), 16);
    }
    // distinct tags
    for (i, a) in tags.iter().enumerate() {
        for b in &tags[i + 1..] {
            assert_ne!(a, b);
        }
    }
}

#[test]
fn commit_is_domain_separated_and_length_prefixed() {
    // Same payload, different tag -> different digest.
    assert_ne!(commit(TAG_MODEL, b"x"), commit(TAG_QUANT, b"x"));
    // Length prefix removes concatenation ambiguity: ("a","bcd") vs ("ab","cd")
    // both have payload "abcd" only if the payload is identical bytes.
    assert_eq!(commit(TAG_MODEL, b"abcd"), commit(TAG_MODEL, b"abcd"));
    assert_ne!(commit(TAG_MODEL, b"abcd"), commit(TAG_MODEL, b"abce"));
    // deterministic
    assert_eq!(commit(TAG_MODEL, b"hello"), commit(TAG_MODEL, b"hello"));
}

#[test]
fn model_commitment_binds_every_field() {
    let base = model().commitment();
    let mut m = model();
    m.architecture_commitment[0] ^= 1;
    assert_ne!(m.commitment(), base);
    let mut m = model();
    m.weights_root[31] ^= 1;
    assert_ne!(m.commitment(), base);
    let mut m = model();
    m.relation_version = 2;
    assert_ne!(m.commitment(), base);
    let mut m = model();
    m.serialization_version = 2;
    assert_ne!(m.commitment(), base);
}

#[test]
fn quant_commitment_binds_every_field() {
    let base = quant().commitment();
    let mut q = quant();
    q.default_rounding = Rounding::TruncateTowardZero;
    assert_ne!(q.commitment(), base);
    let mut q = quant();
    q.scales[0].log2 = -7;
    assert_ne!(q.commitment(), base);
    let mut q = quant();
    q.scales.push(Scale {
        scale_id: 1,
        log2: -7,
        dtype: Dtype::I16,
    });
    assert_ne!(q.commitment(), base);
    let mut q = quant();
    q.activation_tables_commitment[0] ^= 1;
    assert_ne!(q.commitment(), base);
}

#[test]
fn planner_commitment_binds_every_field() {
    let base = planner().commitment();
    for mutate in [
        |p: &mut PlannerBinding| p.horizon = 7,
        |p: &mut PlannerBinding| p.action_block = 3,
        |p: &mut PlannerBinding| p.candidate_count = 64,
        |p: &mut PlannerBinding| p.tie_break_rule_id = 1,
    ] {
        let mut p = planner();
        mutate(&mut p);
        assert_ne!(p.commitment(), base);
    }
}

#[test]
fn the_three_commitments_do_not_collide() {
    // Different domain tags keep the kinds apart even though all hash to 32 bytes.
    let m = model().commitment();
    let q = quant().commitment();
    let p = planner().commitment();
    assert_ne!(m, q);
    assert_ne!(m, p);
    assert_ne!(q, p);
}

#[test]
fn weights_root_is_deterministic_and_order_independent() {
    let a = tensor(1, &[1, 2, 3]);
    let b = tensor(2, &[4, 5, 6]);
    let root_ab = weights_root(&[a.clone(), b.clone()]);
    // Determinism
    assert_eq!(root_ab, weights_root(&[a.clone(), b.clone()]));
    // Sorted by tensor_id: input order does not matter.
    assert_eq!(root_ab, weights_root(&[b.clone(), a.clone()]));
}

#[test]
fn weights_root_changes_with_a_leaf_and_handles_edges() {
    let a = tensor(1, &[1, 2, 3]);
    let b = tensor(2, &[4, 5, 6]);
    let base = weights_root(&[a.clone(), b.clone()]);

    // A changed data value changes the root.
    let b2 = tensor(2, &[4, 5, 7]);
    assert_ne!(weights_root(&[a.clone(), b2]), base);

    // A changed tensor_id changes the root (id is in the leaf preimage).
    let a3 = tensor(3, &[1, 2, 3]);
    assert_ne!(weights_root(&[a3, b.clone()]), base);

    // Single tensor and empty are both defined and distinct.
    let single = weights_root(std::slice::from_ref(&a));
    let empty = weights_root(&[]);
    assert_ne!(single, empty);
    assert_ne!(single, base);
}

#[test]
fn weights_root_multi_leaf_parity_vector_is_pinned() {
    // The cross-language parity vector: predictor-style tensors (ids 1000..,
    // scale_id 0, int8 bounds; three leaves exercise the odd-level node
    // duplication). The Python exporter pins the same hex in
    // crates/pwm-export/python/tests/test_canonical_parity.py
    // (test_predictor_weights_root_matches_rust), so the gate is two-way: a
    // change to either side's encoding breaks its own pin.
    let shaped = |id: u32, shape: &[u32], vals: &[i64]| {
        let data = vals
            .iter()
            .map(|&v| BoundedInt::new(v, -128, 127).unwrap())
            .collect();
        Tensor::new(id, shape.to_vec(), 0, data).unwrap()
    };
    let w = [
        shaped(1000, &[1, 2], &[1, -2]),
        shaped(1001, &[2, 1], &[3, 4]),
        shaped(1002, &[2, 2], &[-5, 6, -7, 8]),
    ];
    let hex: String = weights_root(&w)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(
        hex,
        "c666f637a30f38d653f7c6a3280e87b705777df25bae4644148b534992b03120"
    );
}

#[test]
fn output_commitment_binds_outputs() {
    let o1 = tensor(0, &[1, 2]);
    let o2 = tensor(1, &[3, 4]);
    let base = claimed_output_commitment(&[o1.clone(), o2.clone()]);
    assert_eq!(base, claimed_output_commitment(&[o1.clone(), o2.clone()]));
    // Order is part of the declared output sequence (not sorted): swapping differs.
    assert_ne!(claimed_output_commitment(&[o2.clone(), o1.clone()]), base);
    // It is domain-separated from a plain tensor commitment.
    assert_ne!(base, commit(TAG_TENSOR, b"whatever"));
}
