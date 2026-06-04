// SPDX-License-Identifier: Apache-2.0
//! Tests for the model manifest (data-model `#model-manifest`, RFC-0001): canonical
//! round-trip, schema-version validation, required-binding-field validation, and
//! the self-hash.

use pwm_core::fixed_point::{OverflowPolicy, Rounding};
use pwm_core::manifest::{
    Architecture, Arithmetic, BaseField, BlockConfig, ClampPolicy, CommitmentScheme, CostKind,
    ExtensionField, FieldConfig, ManifestError, ManifestModel, MaxAbsPolicy, Op, PlannerConfig,
    PlannerKind, PredictorConfig, QuantConfig, SerializationConfig, SignedEncoding, TieBreak,
    Visibility, WeightsConfig, MANIFEST_VERSION_V1, SERIALIZATION_VERSION_V1,
};
use pwm_core::serialize::{canonical_bytes, from_canonical_bytes};
use pwm_core::tensor::{Dtype, Scale};

fn scale(id: u32, log2: i32, dtype: Dtype) -> Scale {
    Scale {
        scale_id: id,
        log2,
        dtype,
    }
}

fn linear_op(id: &str) -> Op {
    Op {
        id: id.to_string(),
        op: "linear".to_string(),
        in_scale_id: 1,
        weight_scale_id: Some(0),
        acc_scale_id: Some(3),
        out_scale_id: 1,
        activation: Some("gelu_lookup_v1".to_string()),
        lookup_table: None,
        weight_commitment: Some([1u8; 32]),
    }
}

fn sample() -> ManifestModel {
    let mut m = ManifestModel {
        manifest_version: MANIFEST_VERSION_V1.to_string(),
        model_family: "lewm".to_string(),
        relation_id: "pwm.lewm.fixed_candidate_planning.v1".to_string(),
        field: FieldConfig {
            base: BaseField::M31,
            extension: ExtensionField::Qm31,
            signed_encoding: SignedEncoding::CenteredModP,
            max_abs_value: MaxAbsPolicy::PerTensor,
        },
        quantization: QuantConfig {
            arithmetic: Arithmetic::FixedPoint,
            default_rounding: Rounding::NearestTiesToEven,
            overflow_policy: OverflowPolicy::Reject,
            clamp_policy: ClampPolicy::Explicit,
            activation_tables_commitment: [0xAA; 32],
        },
        scales: vec![
            scale(0, 0, Dtype::I8),
            scale(1, -7, Dtype::I8),
            scale(3, -16, Dtype::I32),
        ],
        architecture: Architecture {
            latent_dim: 192,
            history_size: 3,
            predictor: PredictorConfig {
                depth: 6,
                heads: 16,
                dim_head: 64,
                mlp_dim: 2048,
                block: BlockConfig {
                    modulation_activation_silu: true,
                    ffn_activation_gelu: true,
                    positional_embedding_learned: true,
                },
            },
            action_encoder_silu: true,
            encoder: None,
        },
        weights: WeightsConfig {
            visibility: Visibility::Public,
            commitment_scheme: CommitmentScheme::Blake2sMerkleV1,
            root: [0xBB; 32],
        },
        planner: PlannerConfig {
            kind: PlannerKind::FixedCandidate,
            cost: CostKind::MseGoalLatent,
            tie_break: TieBreak::SmallestIndex,
            horizon: 5,
            action_block: 5,
            cem: None,
        },
        ops: vec![
            linear_op("action_encoder.embed"),
            Op {
                id: "predictor.block0.attn.softmax".to_string(),
                op: "softmax_approx_v1".to_string(),
                in_scale_id: 1,
                weight_scale_id: None,
                acc_scale_id: None,
                out_scale_id: 1,
                activation: None,
                lookup_table: Some("softmax_table_v1".to_string()),
                weight_commitment: None,
            },
        ],
        serialization: SerializationConfig {
            serialization_version: SERIALIZATION_VERSION_V1.to_string(),
            canonical_json_hash: [0u8; 32],
        },
    };
    m.seal_self_hash();
    m
}

#[test]
fn valid_manifest_validates() {
    sample().validate().expect("sample manifest is valid");
}

#[test]
fn round_trips_canonically() {
    let m = sample();
    let bytes = canonical_bytes(&m);
    assert_eq!(bytes, canonical_bytes(&m), "encoding is deterministic");
    let decoded: ManifestModel = from_canonical_bytes(&bytes).expect("decodes");
    assert_eq!(decoded, m);
    decoded
        .validate()
        .expect("decoded manifest still validates");
}

#[test]
fn rejects_unknown_manifest_version() {
    let mut m = sample();
    m.manifest_version = "pwm-model-manifest-v99".to_string();
    assert_eq!(
        m.validate(),
        Err(ManifestError::UnknownManifestVersion(
            "pwm-model-manifest-v99".to_string()
        ))
    );
}

#[test]
fn rejects_unknown_serialization_version() {
    let mut m = sample();
    m.serialization.serialization_version = "bad".to_string();
    assert!(matches!(
        m.validate(),
        Err(ManifestError::UnknownSerializationVersion(_))
    ));
}

#[test]
fn rejects_missing_required_binding_field() {
    // A linear op without its weight_commitment is missing a required binding.
    let mut m = sample();
    m.ops[0].weight_commitment = None;
    assert_eq!(
        m.validate(),
        Err(ManifestError::MissingBindingField {
            op_id: "action_encoder.embed".to_string(),
            field: "weight_commitment",
        })
    );
}

#[test]
fn rejects_op_referencing_undeclared_scale() {
    let mut m = sample();
    m.ops[0].in_scale_id = 99;
    assert_eq!(
        m.validate(),
        Err(ManifestError::UndeclaredScale {
            op_id: "action_encoder.embed".to_string(),
            scale_id: 99,
        })
    );
}

#[test]
fn rejects_planner_kind_mismatch() {
    let mut m = sample();
    m.planner.kind = PlannerKind::Cem; // but cem config is None
    assert_eq!(m.validate(), Err(ManifestError::PlannerKindMismatch));
}

#[test]
fn rejects_malformed_relation_id() {
    let mut m = sample();
    m.relation_id = "not-a-relation".to_string();
    assert!(matches!(
        m.validate(),
        Err(ManifestError::MalformedRelationId(_))
    ));
}

#[test]
fn self_hash_detects_tampering() {
    // Tamper a field not otherwise validated, without re-sealing.
    let mut m = sample();
    m.model_family = "tampered".to_string();
    assert!(matches!(
        m.validate(),
        Err(ManifestError::SelfHashMismatch { .. })
    ));

    // Re-sealing makes it valid again.
    m.seal_self_hash();
    m.validate().expect("re-sealed manifest validates");
}
