// SPDX-License-Identifier: Apache-2.0
//! The model manifest: typed schema, canonical (de)serialization, validation,
//! and self-hash (data-model `#model-manifest`, RFC-0001, RFC-0014).
//!
//! The manifest is the canonical artifact that defines the exact model proven.
//! On disk it is YAML, but the **binding** representation is its canonical bytes
//! (RFC-0014 §1): the YAML is parsed into [`ManifestModel`], validated, and
//! re-encoded, and that byte string feeds the model/quantization/planner
//! commitments. This module owns the typed schema and the validation rules
//! (INV-DM-09/11, schema-version and required-binding-field checks). The
//! YAML↔type ingest and the byte-identical writer live in the export pipeline
//! (#37/#38).

use alloc::string::String;
use alloc::vec::Vec;

use crate::fixed_point::{OverflowPolicy, Rounding};
use crate::serialize::{impl_enum_u8, CanonicalDecode, CanonicalEncode, DecodeError, Reader};
use crate::tensor::Scale;
use crate::transcript::blake2s256;

/// The only manifest schema version recognized in V0.
pub const MANIFEST_VERSION_V1: &str = "pwm-model-manifest-v1";
/// The only serialization-format version recognized in V0.
pub const SERIALIZATION_VERSION_V1: &str = "pwm-serialization-v1";
/// Domain tag for the manifest self-hash (`canonical_json_hash`).
pub const TAG_MANIFEST_SELF_HASH: &[u8; 16] = b"pwm.mhash.v1\0\0\0\0";

macro_rules! fieldless_enum {
    ($(#[$m:meta])* $vis:vis enum $name:ident { $($(#[$vm:meta])* $variant:ident = $disc:literal),+ $(,)? }) => {
        $(#[$m])*
        // Variants are trivial manifest config discriminants; the enum and its
        // purpose are documented above.
        #[allow(missing_docs)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        $vis enum $name { $($(#[$vm])* $variant),+ }
        impl $name {
            /// Immutable canonical-serialization discriminant (RFC-0014 §1).
            pub const fn discriminant(self) -> u8 { match self { $($name::$variant => $disc),+ } }
            /// Inverse of `discriminant`; `None` for an unknown value.
            pub const fn from_discriminant(value: u8) -> Option<Self> {
                match value { $($disc => Some($name::$variant),)+ _ => None }
            }
        }
        impl_enum_u8!($name, stringify!($name));
    };
}

fieldless_enum!(
    /// Base field identifier.
    pub enum BaseField {
        M31 = 0,
    }
);
fieldless_enum!(
    /// Extension (secure) field identifier.
    pub enum ExtensionField {
        Qm31 = 0,
    }
);
fieldless_enum!(
    /// Signed encoding scheme.
    pub enum SignedEncoding {
        CenteredModP = 0,
    }
);
fieldless_enum!(
    /// Global magnitude-cap policy (per-tensor bounds always take precedence).
    pub enum MaxAbsPolicy {
        PerTensor = 0,
    }
);
fieldless_enum!(
    /// Arithmetic kind.
    pub enum Arithmetic {
        FixedPoint = 0,
    }
);
fieldless_enum!(
    /// Clamp policy (every clamp range declared per op).
    pub enum ClampPolicy {
        Explicit = 0,
    }
);
fieldless_enum!(
    /// Weight visibility.
    pub enum Visibility {
        Public = 0,
        PrivateCommitted = 1,
    }
);
fieldless_enum!(
    /// Weight commitment scheme (V0: blake2s Merkle).
    pub enum CommitmentScheme {
        Blake2sMerkleV1 = 0,
    }
);
fieldless_enum!(
    /// Planner kind.
    pub enum PlannerKind {
        FixedCandidate = 0,
        Cem = 1,
    }
);
fieldless_enum!(
    /// Cost function.
    pub enum CostKind {
        MseGoalLatent = 0,
    }
);
fieldless_enum!(
    /// Argmin tie-break rule.
    pub enum TieBreak {
        SmallestIndex = 0,
    }
);

/// Field and signed-integer encoding configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldConfig {
    /// Base field (M31).
    pub base: BaseField,
    /// Extension field (QM31, challenges only).
    pub extension: ExtensionField,
    /// Signed encoding scheme.
    pub signed_encoding: SignedEncoding,
    /// Global magnitude-cap policy.
    pub max_abs_value: MaxAbsPolicy,
}

/// Quantization policy (one active mode).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantConfig {
    /// Arithmetic kind.
    pub arithmetic: Arithmetic,
    /// The single active rounding mode (INV-DM-09).
    pub default_rounding: Rounding,
    /// Overflow policy (`Reject` in V0).
    pub overflow_policy: OverflowPolicy,
    /// Clamp policy.
    pub clamp_policy: ClampPolicy,
    /// Commitment over all activation/lookup tables.
    pub activation_tables_commitment: [u8; 32],
}

/// One conditional-block configuration (the load-bearing activations).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockConfig {
    /// AdaLN modulation activation (SiLU in V0); per-module, not uniform.
    pub modulation_activation_silu: bool,
    /// FFN activation (GELU in V0).
    pub ffn_activation_gelu: bool,
    /// Whether positional embeddings are learned.
    pub positional_embedding_learned: bool,
}

/// Predictor (ARPredictor) configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredictorConfig {
    /// Block depth.
    pub depth: u32,
    /// Attention head count.
    pub heads: u32,
    /// Per-head dimension.
    pub dim_head: u32,
    /// FFN hidden dimension.
    pub mlp_dim: u32,
    /// Conditional-block configuration.
    pub block: BlockConfig,
}

/// Optional pixel encoder (declared and bound but NOT proven in V0).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncoderConfig {
    /// Hidden dimension.
    pub hidden: u32,
    /// Input image size.
    pub image_size: u32,
    /// Patch size.
    pub patch_size: u32,
}

/// Architecture description (bound by `model_commitment`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Architecture {
    /// Latent/embed dimension.
    pub latent_dim: u32,
    /// History window size.
    pub history_size: u32,
    /// Predictor configuration.
    pub predictor: PredictorConfig,
    /// Action-encoder (Embedder) activation is SiLU in V0.
    pub action_encoder_silu: bool,
    /// Optional pixel encoder (P4; declared, not proven in V0).
    pub encoder: Option<EncoderConfig>,
}

/// Weights configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightsConfig {
    /// Public or private-committed.
    pub visibility: Visibility,
    /// Commitment scheme (V0: blake2s Merkle).
    pub commitment_scheme: CommitmentScheme,
    /// The weight Merkle root.
    pub root: [u8; 32],
}

/// CEM-only planner parameters (P3/V2; absent for fixed-candidate).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CemConfig {
    /// Samples per iteration.
    pub num_samples: u32,
    /// Iterations.
    pub n_steps: u32,
    /// Top-k elites.
    pub topk: u32,
    /// Variance scale, in thousandths (floats are forbidden in canonical bytes).
    pub var_scale_milli: u32,
    /// Seed.
    pub seed: u64,
}

/// Planner configuration (bound by `planner_config_commitment`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlannerConfig {
    /// Planner kind.
    pub kind: PlannerKind,
    /// Cost function.
    pub cost: CostKind,
    /// Tie-break rule.
    pub tie_break: TieBreak,
    /// Rollout horizon.
    pub horizon: u32,
    /// Actions applied per step.
    pub action_block: u32,
    /// CEM parameters (present iff `kind == Cem`).
    pub cem: Option<CemConfig>,
}

/// One ordered op entry (binds its scales and per-op weight commitment).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Op {
    /// Stable op id, e.g. `predictor.block0.attn.qkv`.
    pub id: String,
    /// Versioned op-type identifier, e.g. `linear`, `softmax_approx_v1`.
    pub op: String,
    /// Input scale id.
    pub in_scale_id: u32,
    /// Weight scale id (linear/matmul ops).
    pub weight_scale_id: Option<u32>,
    /// Accumulator scale id (linear/matmul ops).
    pub acc_scale_id: Option<u32>,
    /// Output scale id.
    pub out_scale_id: u32,
    /// Activation id, e.g. `gelu_lookup_v1`.
    pub activation: Option<String>,
    /// Lookup-table id for table-based ops.
    pub lookup_table: Option<String>,
    /// Per-op weight commitment (required for weight-bearing ops).
    pub weight_commitment: Option<[u8; 32]>,
}

/// Serialization self-hash block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SerializationConfig {
    /// RFC-0014 byte-format version.
    pub serialization_version: String,
    /// Self-hash over the canonical serialization of the manifest (see
    /// [`ManifestModel::compute_self_hash`]).
    pub canonical_json_hash: [u8; 32],
}

/// The full model manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestModel {
    /// Schema version of this document (must be [`MANIFEST_VERSION_V1`]).
    pub manifest_version: String,
    /// Model family identifier.
    pub model_family: String,
    /// Relation id, e.g. `pwm.lewm.fixed_candidate_planning.v1`.
    pub relation_id: String,
    /// Field and encoding configuration.
    pub field: FieldConfig,
    /// Quantization policy.
    pub quantization: QuantConfig,
    /// The scale table.
    pub scales: Vec<Scale>,
    /// Architecture description.
    pub architecture: Architecture,
    /// Weights configuration.
    pub weights: WeightsConfig,
    /// Planner configuration.
    pub planner: PlannerConfig,
    /// The ordered op list.
    pub ops: Vec<Op>,
    /// Serialization self-hash.
    pub serialization: SerializationConfig,
}

/// A manifest validation failure (data-model `#model-manifest` failure modes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManifestError {
    /// `manifest_version` is not a recognized schema version.
    UnknownManifestVersion(String),
    /// `serialization.serialization_version` is not recognized.
    UnknownSerializationVersion(String),
    /// `relation_id` is empty or malformed.
    MalformedRelationId(String),
    /// The scale table is empty.
    NoScales,
    /// An op references a `scale_id` absent from `scales[]` (INV-DM-11).
    UndeclaredScale {
        /// The op id.
        op_id: String,
        /// The undeclared scale id.
        scale_id: u32,
    },
    /// A weight-bearing op is missing a required binding field.
    MissingBindingField {
        /// The op id.
        op_id: String,
        /// The missing field name.
        field: &'static str,
    },
    /// `planner.kind` is inconsistent with the presence of `cem` config.
    PlannerKindMismatch,
    /// The stored `canonical_json_hash` does not match the computed self-hash.
    SelfHashMismatch {
        /// The stored hash.
        stored: [u8; 32],
        /// The recomputed hash.
        computed: [u8; 32],
    },
}

// --- canonical codecs for the composite structs ---

impl CanonicalEncode for FieldConfig {
    fn encode(&self, out: &mut Vec<u8>) {
        self.base.encode(out);
        self.extension.encode(out);
        self.signed_encoding.encode(out);
        self.max_abs_value.encode(out);
    }
}
impl CanonicalDecode for FieldConfig {
    fn decode(r: &mut Reader<'_>) -> Result<Self, DecodeError> {
        Ok(FieldConfig {
            base: BaseField::decode(r)?,
            extension: ExtensionField::decode(r)?,
            signed_encoding: SignedEncoding::decode(r)?,
            max_abs_value: MaxAbsPolicy::decode(r)?,
        })
    }
}

impl CanonicalEncode for QuantConfig {
    fn encode(&self, out: &mut Vec<u8>) {
        self.arithmetic.encode(out);
        self.default_rounding.encode(out);
        self.overflow_policy.encode(out);
        self.clamp_policy.encode(out);
        self.activation_tables_commitment.encode(out);
    }
}
impl CanonicalDecode for QuantConfig {
    fn decode(r: &mut Reader<'_>) -> Result<Self, DecodeError> {
        Ok(QuantConfig {
            arithmetic: Arithmetic::decode(r)?,
            default_rounding: Rounding::decode(r)?,
            overflow_policy: OverflowPolicy::decode(r)?,
            clamp_policy: ClampPolicy::decode(r)?,
            activation_tables_commitment: <[u8; 32]>::decode(r)?,
        })
    }
}

impl CanonicalEncode for BlockConfig {
    fn encode(&self, out: &mut Vec<u8>) {
        self.modulation_activation_silu.encode(out);
        self.ffn_activation_gelu.encode(out);
        self.positional_embedding_learned.encode(out);
    }
}
impl CanonicalDecode for BlockConfig {
    fn decode(r: &mut Reader<'_>) -> Result<Self, DecodeError> {
        Ok(BlockConfig {
            modulation_activation_silu: bool::decode(r)?,
            ffn_activation_gelu: bool::decode(r)?,
            positional_embedding_learned: bool::decode(r)?,
        })
    }
}

impl CanonicalEncode for PredictorConfig {
    fn encode(&self, out: &mut Vec<u8>) {
        self.depth.encode(out);
        self.heads.encode(out);
        self.dim_head.encode(out);
        self.mlp_dim.encode(out);
        self.block.encode(out);
    }
}
impl CanonicalDecode for PredictorConfig {
    fn decode(r: &mut Reader<'_>) -> Result<Self, DecodeError> {
        Ok(PredictorConfig {
            depth: u32::decode(r)?,
            heads: u32::decode(r)?,
            dim_head: u32::decode(r)?,
            mlp_dim: u32::decode(r)?,
            block: BlockConfig::decode(r)?,
        })
    }
}

impl CanonicalEncode for EncoderConfig {
    fn encode(&self, out: &mut Vec<u8>) {
        self.hidden.encode(out);
        self.image_size.encode(out);
        self.patch_size.encode(out);
    }
}
impl CanonicalDecode for EncoderConfig {
    fn decode(r: &mut Reader<'_>) -> Result<Self, DecodeError> {
        Ok(EncoderConfig {
            hidden: u32::decode(r)?,
            image_size: u32::decode(r)?,
            patch_size: u32::decode(r)?,
        })
    }
}

impl CanonicalEncode for Architecture {
    fn encode(&self, out: &mut Vec<u8>) {
        self.latent_dim.encode(out);
        self.history_size.encode(out);
        self.predictor.encode(out);
        self.action_encoder_silu.encode(out);
        self.encoder.encode(out);
    }
}
impl CanonicalDecode for Architecture {
    fn decode(r: &mut Reader<'_>) -> Result<Self, DecodeError> {
        Ok(Architecture {
            latent_dim: u32::decode(r)?,
            history_size: u32::decode(r)?,
            predictor: PredictorConfig::decode(r)?,
            action_encoder_silu: bool::decode(r)?,
            encoder: Option::<EncoderConfig>::decode(r)?,
        })
    }
}

impl CanonicalEncode for WeightsConfig {
    fn encode(&self, out: &mut Vec<u8>) {
        self.visibility.encode(out);
        self.commitment_scheme.encode(out);
        self.root.encode(out);
    }
}
impl CanonicalDecode for WeightsConfig {
    fn decode(r: &mut Reader<'_>) -> Result<Self, DecodeError> {
        Ok(WeightsConfig {
            visibility: Visibility::decode(r)?,
            commitment_scheme: CommitmentScheme::decode(r)?,
            root: <[u8; 32]>::decode(r)?,
        })
    }
}

impl CanonicalEncode for CemConfig {
    fn encode(&self, out: &mut Vec<u8>) {
        self.num_samples.encode(out);
        self.n_steps.encode(out);
        self.topk.encode(out);
        self.var_scale_milli.encode(out);
        self.seed.encode(out);
    }
}
impl CanonicalDecode for CemConfig {
    fn decode(r: &mut Reader<'_>) -> Result<Self, DecodeError> {
        Ok(CemConfig {
            num_samples: u32::decode(r)?,
            n_steps: u32::decode(r)?,
            topk: u32::decode(r)?,
            var_scale_milli: u32::decode(r)?,
            seed: u64::decode(r)?,
        })
    }
}

impl CanonicalEncode for PlannerConfig {
    fn encode(&self, out: &mut Vec<u8>) {
        self.kind.encode(out);
        self.cost.encode(out);
        self.tie_break.encode(out);
        self.horizon.encode(out);
        self.action_block.encode(out);
        self.cem.encode(out);
    }
}
impl CanonicalDecode for PlannerConfig {
    fn decode(r: &mut Reader<'_>) -> Result<Self, DecodeError> {
        Ok(PlannerConfig {
            kind: PlannerKind::decode(r)?,
            cost: CostKind::decode(r)?,
            tie_break: TieBreak::decode(r)?,
            horizon: u32::decode(r)?,
            action_block: u32::decode(r)?,
            cem: Option::<CemConfig>::decode(r)?,
        })
    }
}

impl CanonicalEncode for Op {
    fn encode(&self, out: &mut Vec<u8>) {
        self.id.encode(out);
        self.op.encode(out);
        self.in_scale_id.encode(out);
        self.weight_scale_id.encode(out);
        self.acc_scale_id.encode(out);
        self.out_scale_id.encode(out);
        self.activation.encode(out);
        self.lookup_table.encode(out);
        self.weight_commitment.encode(out);
    }
}
impl CanonicalDecode for Op {
    fn decode(r: &mut Reader<'_>) -> Result<Self, DecodeError> {
        Ok(Op {
            id: String::decode(r)?,
            op: String::decode(r)?,
            in_scale_id: u32::decode(r)?,
            weight_scale_id: Option::<u32>::decode(r)?,
            acc_scale_id: Option::<u32>::decode(r)?,
            out_scale_id: u32::decode(r)?,
            activation: Option::<String>::decode(r)?,
            lookup_table: Option::<String>::decode(r)?,
            weight_commitment: Option::<[u8; 32]>::decode(r)?,
        })
    }
}

impl CanonicalEncode for SerializationConfig {
    fn encode(&self, out: &mut Vec<u8>) {
        self.serialization_version.encode(out);
        self.canonical_json_hash.encode(out);
    }
}
impl CanonicalDecode for SerializationConfig {
    fn decode(r: &mut Reader<'_>) -> Result<Self, DecodeError> {
        Ok(SerializationConfig {
            serialization_version: String::decode(r)?,
            canonical_json_hash: <[u8; 32]>::decode(r)?,
        })
    }
}

impl CanonicalEncode for ManifestModel {
    fn encode(&self, out: &mut Vec<u8>) {
        self.manifest_version.encode(out);
        self.model_family.encode(out);
        self.relation_id.encode(out);
        self.field.encode(out);
        self.quantization.encode(out);
        self.scales.encode(out);
        self.architecture.encode(out);
        self.weights.encode(out);
        self.planner.encode(out);
        self.ops.encode(out);
        self.serialization.encode(out);
    }
}
impl CanonicalDecode for ManifestModel {
    fn decode(r: &mut Reader<'_>) -> Result<Self, DecodeError> {
        Ok(ManifestModel {
            manifest_version: String::decode(r)?,
            model_family: String::decode(r)?,
            relation_id: String::decode(r)?,
            field: FieldConfig::decode(r)?,
            quantization: QuantConfig::decode(r)?,
            scales: Vec::<Scale>::decode(r)?,
            architecture: Architecture::decode(r)?,
            weights: WeightsConfig::decode(r)?,
            planner: PlannerConfig::decode(r)?,
            ops: Vec::<Op>::decode(r)?,
            serialization: SerializationConfig::decode(r)?,
        })
    }
}

impl ManifestModel {
    /// Compute the self-hash over the manifest with `canonical_json_hash` zeroed,
    /// domain-separated by [`TAG_MANIFEST_SELF_HASH`]. Avoids self-reference.
    pub fn compute_self_hash(&self) -> [u8; 32] {
        let mut bare = self.clone();
        bare.serialization.canonical_json_hash = [0u8; 32];
        let mut input = Vec::new();
        input.extend_from_slice(TAG_MANIFEST_SELF_HASH);
        crate::serialize::canonical_bytes(&bare)
            .iter()
            .for_each(|b| input.push(*b));
        blake2s256(&input)
    }

    /// Set `serialization.canonical_json_hash` to the computed self-hash.
    pub fn seal_self_hash(&mut self) {
        self.serialization.canonical_json_hash = self.compute_self_hash();
    }

    /// True iff a scale with `scale_id` is declared.
    fn scale_declared(&self, scale_id: u32) -> bool {
        self.scales.iter().any(|s| s.scale_id == scale_id)
    }

    /// Validate the manifest: schema versions, relation-id form, declared scales,
    /// required binding fields, planner consistency, and the self-hash.
    pub fn validate(&self) -> Result<(), ManifestError> {
        if self.manifest_version != MANIFEST_VERSION_V1 {
            return Err(ManifestError::UnknownManifestVersion(
                self.manifest_version.clone(),
            ));
        }
        if self.serialization.serialization_version != SERIALIZATION_VERSION_V1 {
            return Err(ManifestError::UnknownSerializationVersion(
                self.serialization.serialization_version.clone(),
            ));
        }
        // relation_id form: "pwm.<family>.<statement>.v<N>".
        if !self.relation_id.starts_with("pwm.") || self.relation_id.matches('.').count() < 3 {
            return Err(ManifestError::MalformedRelationId(self.relation_id.clone()));
        }
        if self.scales.is_empty() {
            return Err(ManifestError::NoScales);
        }

        for op in &self.ops {
            for scale_id in [
                Some(op.in_scale_id),
                Some(op.out_scale_id),
                op.weight_scale_id,
                op.acc_scale_id,
            ]
            .into_iter()
            .flatten()
            {
                if !self.scale_declared(scale_id) {
                    return Err(ManifestError::UndeclaredScale {
                        op_id: op.id.clone(),
                        scale_id,
                    });
                }
            }
            // Weight-bearing ops must carry their binding fields.
            if matches!(op.op.as_str(), "linear" | "matmul") {
                if op.weight_commitment.is_none() {
                    return Err(ManifestError::MissingBindingField {
                        op_id: op.id.clone(),
                        field: "weight_commitment",
                    });
                }
                if op.weight_scale_id.is_none() {
                    return Err(ManifestError::MissingBindingField {
                        op_id: op.id.clone(),
                        field: "weight_scale_id",
                    });
                }
                if op.acc_scale_id.is_none() {
                    return Err(ManifestError::MissingBindingField {
                        op_id: op.id.clone(),
                        field: "acc_scale_id",
                    });
                }
            }
        }

        // CEM config present iff the planner is CEM.
        if (self.planner.kind == PlannerKind::Cem) != self.planner.cem.is_some() {
            return Err(ManifestError::PlannerKindMismatch);
        }

        let computed = self.compute_self_hash();
        if self.serialization.canonical_json_hash != computed {
            return Err(ManifestError::SelfHashMismatch {
                stored: self.serialization.canonical_json_hash,
                computed,
            });
        }
        Ok(())
    }
}
