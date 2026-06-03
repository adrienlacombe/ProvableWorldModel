// SPDX-License-Identifier: Apache-2.0
//! The structured log record schema and per-stage required-field policy.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use super::json;
use super::value::LoggableValue;
use crate::relation::StatementType;

/// Log schema version defined by this spec (`docs/spec/05-observability.md`).
pub const LOG_SCHEMA_VERSION: u32 = 1;

/// Severity level. Ordered most-severe first; see [`Level::should_emit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    /// Terminal failure of the run; always paired with an `error_code`.
    Error,
    /// Recoverable anomaly or a fallback that changes cost, not correctness.
    Warn,
    /// Stage transitions and per-stage summaries (the default threshold).
    Info,
    /// Per-component detail (per-op counts, per-layer timings).
    Debug,
    /// Per-row / per-cell control-flow labels. Never values.
    Trace,
}

/// The default emit threshold (`docs/spec/05-observability.md#level-policy`).
pub const DEFAULT_LEVEL: Level = Level::Info;

impl Level {
    /// The canonical uppercase label used in serialized records.
    pub const fn as_str(self) -> &'static str {
        match self {
            Level::Error => "ERROR",
            Level::Warn => "WARN",
            Level::Info => "INFO",
            Level::Debug => "DEBUG",
            Level::Trace => "TRACE",
        }
    }

    /// Whether an event at `self` is emitted under the configured `threshold`.
    ///
    /// A lower threshold reveals *more events*, never *more value content*
    /// (INV-OBS-09): redaction is governed by the type system, not the level.
    pub fn should_emit(self, threshold: Level) -> bool {
        // `Error` is the smallest discriminant (most severe); an event is emitted
        // when it is at least as severe as the threshold.
        self <= threshold
    }
}

/// The pipeline stage emitting a record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Graph + weight extraction from the checkpoint.
    Export,
    /// Per-tensor quantization of the exported graph.
    Quantize,
    /// Deterministic fixed-point reference inference.
    ReferenceInference,
    /// Manifest + commitments written.
    ManifestWrite,
    /// Per-component main-trace construction.
    TraceBuild,
    /// Merkle commitment of the preprocessed + main traces.
    Commit,
    /// Fiat-Shamir challenge derivation.
    FiatShamir,
    /// LogUp interaction-trace construction.
    InteractionTrace,
    /// FRI commit + query phase, proof serialization.
    Fri,
    /// Prover root stage (terminal prove record).
    Prove,
    /// Verifier root stage (terminal verify record).
    Verify,
}

impl Stage {
    /// The canonical label used in serialized records.
    pub const fn as_str(self) -> &'static str {
        match self {
            Stage::Export => "Export",
            Stage::Quantize => "Quantize",
            Stage::ReferenceInference => "ReferenceInference",
            Stage::ManifestWrite => "ManifestWrite",
            Stage::TraceBuild => "TraceBuild",
            Stage::Commit => "Commit",
            Stage::FiatShamir => "FiatShamir",
            Stage::InteractionTrace => "InteractionTrace",
            Stage::Fri => "Fri",
            Stage::Prove => "Prove",
            Stage::Verify => "Verify",
        }
    }
}

/// Terminal outcome flag, set on terminal events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// The unit of work completed successfully.
    Ok,
    /// The unit of work failed (paired with an `error_code`).
    Error,
}

impl Outcome {
    /// The canonical label used in serialized records.
    pub const fn as_str(self) -> &'static str {
        match self {
            Outcome::Ok => "ok",
            Outcome::Error => "error",
        }
    }
}

/// The shape of a tensor touched by a stage. Structural metadata only — never the
/// tensor's values (`docs/spec/05-observability.md#redaction`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    /// Logical tensor identifier.
    pub tensor_id: u32,
    /// Tensor dimensions.
    pub dims: Vec<u32>,
    /// Fixed-point scale identifier of the tensor.
    pub scale_id: u32,
}

/// Structural counts a record may carry. Each is derived from witness
/// *structure* (shapes, op counts, table sizes), never witness *values*
/// (INV-OBS-03). Field names match the named metric set
/// (`docs/spec/05-observability.md#metrics`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Counts {
    /// Number of exported tensors.
    pub tensor_count: Option<u64>,
    /// Number of requantization ops.
    pub requant_op_count: Option<u64>,
    /// Main-trace row count.
    pub trace_rows: Option<u64>,
    /// Trace column count.
    pub trace_cols: Option<u64>,
    /// Number of values wired to a range relation.
    pub range_check_count: Option<u64>,
    /// Sum of LogUp multiplicities over all lookups.
    pub lookup_multiplicity_total: Option<u64>,
    /// Number of Fiat-Shamir challenges derived.
    pub challenge_count: Option<u64>,
    /// Number of FRI queries performed.
    pub fri_query_count: Option<u64>,
    /// Number of FRI layers.
    pub fri_layers: Option<u64>,
    /// Serialized proof size in bytes.
    pub proof_size_bytes: Option<u64>,
}

/// One structured log record, serialized as a single line of JSON
/// (`docs/spec/05-observability.md#record-schema`). No field of this struct may
/// ever hold a private witness value (INV-OBS-01); structural detail lives in
/// typed [`Counts`]/[`Shape`] and in `fields`, constrained to [`LoggableValue`].
#[derive(Debug, Clone, PartialEq)]
pub struct LogRecord {
    /// Schema version (= [`LOG_SCHEMA_VERSION`]).
    pub log_schema_version: u32,
    /// RFC3339 UTC timestamp.
    pub ts: String,
    /// Severity level.
    pub level: Level,
    /// The stage emitting this record.
    pub stage: Stage,
    /// Tracing span id; joins logs to spans.
    pub span_id: u64,
    /// Parent span id, when nested.
    pub parent_span_id: Option<u64>,
    /// ULID of this process invocation (emitted as 32 hex chars).
    pub run_id: [u8; 16],
    /// Relation id once known.
    pub relation_id: Option<String>,
    /// Statement discriminant once known.
    pub statement_type: Option<StatementType>,
    /// Model commitment.
    pub model_commitment: Option<[u8; 32]>,
    /// Quantization commitment.
    pub quantization_commitment: Option<[u8; 32]>,
    /// Planner-config commitment.
    pub planner_config_commitment: Option<[u8; 32]>,
    /// AIR component name, when stage-specific.
    pub component: Option<String>,
    /// Shapes of tensors touched; values never included.
    pub shapes: Option<Vec<Shape>>,
    /// Structural counts.
    pub counts: Option<Counts>,
    /// Wall time of the unit of work, if terminal.
    pub timing_ms: Option<f64>,
    /// Terminal outcome.
    pub outcome: Option<Outcome>,
    /// Error code from the taxonomy, on failure.
    pub error_code: Option<String>,
    /// Value-free human-readable message.
    pub message: String,
    /// Structured extras, redaction-checked (only [`LoggableValue`]s).
    pub fields: BTreeMap<String, LoggableValue>,
}

impl LogRecord {
    /// Construct a minimal record with the always-present fields populated and
    /// every optional field empty. Callers set the stage-specific fields.
    pub fn new(
        level: Level,
        stage: Stage,
        span_id: u64,
        run_id: [u8; 16],
        message: impl Into<String>,
    ) -> Self {
        LogRecord {
            log_schema_version: LOG_SCHEMA_VERSION,
            ts: String::new(),
            level,
            stage,
            span_id,
            parent_span_id: None,
            run_id,
            relation_id: None,
            statement_type: None,
            model_commitment: None,
            quantization_commitment: None,
            planner_config_commitment: None,
            component: None,
            shapes: None,
            counts: None,
            timing_ms: None,
            outcome: None,
            error_code: None,
            message: message.into(),
            fields: BTreeMap::new(),
        }
    }

    /// Validate that a *terminal* record carries every field required for its
    /// stage (`docs/spec/05-observability.md#per-stage-required-fields`). Returns
    /// the list of missing requirements, empty when the record is complete.
    pub fn missing_terminal_fields(&self) -> Vec<RequiredField> {
        let mut missing = Vec::new();
        for required in required_terminal_fields(self.stage) {
            if !required.is_present(self) {
                missing.push(*required);
            }
        }
        // The Verify stage additionally requires an error_code on a rejection.
        if self.stage == Stage::Verify
            && self.outcome == Some(Outcome::Error)
            && self.error_code.is_none()
        {
            missing.push(RequiredField::ErrorCode);
        }
        missing
    }

    /// Serialize this record as a single line of JSON (newline-delimited form).
    /// Optional fields that are `None` are omitted. The output is value-free by
    /// construction: every field is a commitment, hash, shape, count, timing,
    /// enum label, error code, or value-free message.
    pub fn to_json(&self) -> String {
        let mut out = String::with_capacity(256);
        out.push('{');
        let mut first = true;

        comma_key(&mut out, "log_schema_version", &mut first);
        json::push_u64(&mut out, self.log_schema_version as u64);

        comma_key(&mut out, "ts", &mut first);
        json::push_string(&mut out, &self.ts);

        comma_key(&mut out, "level", &mut first);
        json::push_string(&mut out, self.level.as_str());

        comma_key(&mut out, "stage", &mut first);
        json::push_string(&mut out, self.stage.as_str());

        comma_key(&mut out, "span_id", &mut first);
        json::push_u64(&mut out, self.span_id);

        if let Some(parent) = self.parent_span_id {
            comma_key(&mut out, "parent_span_id", &mut first);
            json::push_u64(&mut out, parent);
        }

        comma_key(&mut out, "run_id", &mut first);
        push_run_id(&mut out, &self.run_id);

        if let Some(ref r) = self.relation_id {
            comma_key(&mut out, "relation_id", &mut first);
            json::push_string(&mut out, r);
        }
        if let Some(st) = self.statement_type {
            comma_key(&mut out, "statement_type", &mut first);
            json::push_string(&mut out, st.as_str());
        }
        if let Some(ref c) = self.model_commitment {
            comma_key(&mut out, "model_commitment", &mut first);
            json::push_hex32(&mut out, c);
        }
        if let Some(ref c) = self.quantization_commitment {
            comma_key(&mut out, "quantization_commitment", &mut first);
            json::push_hex32(&mut out, c);
        }
        if let Some(ref c) = self.planner_config_commitment {
            comma_key(&mut out, "planner_config_commitment", &mut first);
            json::push_hex32(&mut out, c);
        }
        if let Some(ref c) = self.component {
            comma_key(&mut out, "component", &mut first);
            json::push_string(&mut out, c);
        }
        if let Some(ref shapes) = self.shapes {
            comma_key(&mut out, "shapes", &mut first);
            push_shapes(&mut out, shapes);
        }
        if let Some(ref counts) = self.counts {
            comma_key(&mut out, "counts", &mut first);
            push_counts(&mut out, counts);
        }
        if let Some(t) = self.timing_ms {
            comma_key(&mut out, "timing_ms", &mut first);
            json::push_f64(&mut out, t);
        }
        if let Some(o) = self.outcome {
            comma_key(&mut out, "outcome", &mut first);
            json::push_string(&mut out, o.as_str());
        }
        if let Some(ref e) = self.error_code {
            comma_key(&mut out, "error_code", &mut first);
            json::push_string(&mut out, e);
        }

        comma_key(&mut out, "message", &mut first);
        json::push_string(&mut out, &self.message);

        if !self.fields.is_empty() {
            comma_key(&mut out, "fields", &mut first);
            out.push('{');
            let mut field_first = true;
            for (k, v) in &self.fields {
                if field_first {
                    field_first = false;
                } else {
                    out.push(',');
                }
                json::push_key(&mut out, k);
                v.write_json(&mut out);
            }
            out.push('}');
        }

        out.push('}');
        out
    }
}

fn comma_key(out: &mut String, key: &str, first: &mut bool) {
    if *first {
        *first = false;
    } else {
        out.push(',');
    }
    json::push_key(out, key);
}

fn push_run_id(out: &mut String, run_id: &[u8; 16]) {
    out.push('"');
    for byte in run_id {
        out.push(char::from_digit((byte >> 4) as u32, 16).unwrap_or('0'));
        out.push(char::from_digit((byte & 0xf) as u32, 16).unwrap_or('0'));
    }
    out.push('"');
}

fn push_shapes(out: &mut String, shapes: &[Shape]) {
    out.push('[');
    for (i, s) in shapes.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str("{\"tensor_id\":");
        json::push_u64(out, s.tensor_id as u64);
        out.push_str(",\"dims\":[");
        for (j, d) in s.dims.iter().enumerate() {
            if j > 0 {
                out.push(',');
            }
            json::push_u64(out, *d as u64);
        }
        out.push_str("],\"scale_id\":");
        json::push_u64(out, s.scale_id as u64);
        out.push('}');
    }
    out.push(']');
}

fn push_counts(out: &mut String, counts: &Counts) {
    out.push('{');
    let mut first = true;
    let entries: [(&str, Option<u64>); 10] = [
        ("tensor_count", counts.tensor_count),
        ("requant_op_count", counts.requant_op_count),
        ("trace_rows", counts.trace_rows),
        ("trace_cols", counts.trace_cols),
        ("range_check_count", counts.range_check_count),
        (
            "lookup_multiplicity_total",
            counts.lookup_multiplicity_total,
        ),
        ("challenge_count", counts.challenge_count),
        ("fri_query_count", counts.fri_query_count),
        ("fri_layers", counts.fri_layers),
        ("proof_size_bytes", counts.proof_size_bytes),
    ];
    for (name, value) in entries {
        if let Some(v) = value {
            if first {
                first = false;
            } else {
                out.push(',');
            }
            json::push_key(out, name);
            json::push_u64(out, v);
        }
    }
    out.push('}');
}

/// A field a terminal record must carry for its stage. The `fields`-carried
/// variant names a key required in [`LogRecord::fields`] as a commitment/hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequiredField {
    /// `model_commitment` is set.
    ModelCommitment,
    /// `quantization_commitment` is set.
    QuantizationCommitment,
    /// `planner_config_commitment` is set.
    PlannerConfigCommitment,
    /// `relation_id` is set.
    RelationId,
    /// `statement_type` is set.
    StatementType,
    /// `component` is set.
    Component,
    /// `shapes` is set.
    Shapes,
    /// `timing_ms` is set.
    TimingMs,
    /// `outcome` is set.
    Outcome,
    /// `error_code` is set (Verify-on-reject).
    ErrorCode,
    /// A named [`Counts`] field is set.
    Count(CountField),
    /// A named commitment/hash is present in `fields`.
    Field(&'static str),
}

/// A named structural count required on a terminal record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CountField {
    /// `counts.tensor_count`.
    TensorCount,
    /// `counts.requant_op_count`.
    RequantOpCount,
    /// `counts.trace_rows`.
    TraceRows,
    /// `counts.trace_cols`.
    TraceCols,
    /// `counts.range_check_count`.
    RangeCheckCount,
    /// `counts.lookup_multiplicity_total`.
    LookupMultiplicityTotal,
    /// `counts.challenge_count`.
    ChallengeCount,
    /// `counts.fri_query_count`.
    FriQueryCount,
    /// `counts.fri_layers`.
    FriLayers,
    /// `counts.proof_size_bytes`.
    ProofSizeBytes,
}

impl RequiredField {
    fn is_present(self, record: &LogRecord) -> bool {
        match self {
            RequiredField::ModelCommitment => record.model_commitment.is_some(),
            RequiredField::QuantizationCommitment => record.quantization_commitment.is_some(),
            RequiredField::PlannerConfigCommitment => record.planner_config_commitment.is_some(),
            RequiredField::RelationId => record.relation_id.is_some(),
            RequiredField::StatementType => record.statement_type.is_some(),
            RequiredField::Component => record.component.is_some(),
            RequiredField::Shapes => record.shapes.is_some(),
            RequiredField::TimingMs => record.timing_ms.is_some(),
            RequiredField::Outcome => record.outcome.is_some(),
            RequiredField::ErrorCode => record.error_code.is_some(),
            RequiredField::Count(c) => c.is_present(record.counts.as_ref()),
            RequiredField::Field(key) => record.fields.contains_key(key),
        }
    }
}

impl CountField {
    fn is_present(self, counts: Option<&Counts>) -> bool {
        let Some(counts) = counts else { return false };
        match self {
            CountField::TensorCount => counts.tensor_count.is_some(),
            CountField::RequantOpCount => counts.requant_op_count.is_some(),
            CountField::TraceRows => counts.trace_rows.is_some(),
            CountField::TraceCols => counts.trace_cols.is_some(),
            CountField::RangeCheckCount => counts.range_check_count.is_some(),
            CountField::LookupMultiplicityTotal => counts.lookup_multiplicity_total.is_some(),
            CountField::ChallengeCount => counts.challenge_count.is_some(),
            CountField::FriQueryCount => counts.fri_query_count.is_some(),
            CountField::FriLayers => counts.fri_layers.is_some(),
            CountField::ProofSizeBytes => counts.proof_size_bytes.is_some(),
        }
    }
}

/// The fields a terminal record must carry for `stage`, per the per-stage table
/// in `docs/spec/05-observability.md#per-stage-required-fields`. The Verify
/// `error_code`-on-reject requirement is applied separately by
/// [`LogRecord::missing_terminal_fields`].
pub fn required_terminal_fields(stage: Stage) -> &'static [RequiredField] {
    use CountField::*;
    use RequiredField::*;
    match stage {
        Stage::Export => &[
            ModelCommitment,
            RelationId,
            Count(TensorCount),
            Shapes,
            TimingMs,
            Outcome,
        ],
        Stage::Quantize => &[
            ModelCommitment,
            QuantizationCommitment,
            Count(RequantOpCount),
            TimingMs,
            Outcome,
        ],
        Stage::ReferenceInference => &[
            ModelCommitment,
            RelationId,
            StatementType,
            Field("claimed_output_commitment"),
            TimingMs,
            Outcome,
        ],
        Stage::ManifestWrite => &[
            ModelCommitment,
            QuantizationCommitment,
            PlannerConfigCommitment,
            RelationId,
            TimingMs,
            Outcome,
        ],
        Stage::TraceBuild => &[
            Component,
            Count(TraceRows),
            Count(TraceCols),
            Count(RangeCheckCount),
            Count(LookupMultiplicityTotal),
            TimingMs,
            Outcome,
        ],
        Stage::Commit => &[
            Count(TraceRows),
            Count(TraceCols),
            Field("root"),
            TimingMs,
            Outcome,
        ],
        Stage::FiatShamir => &[
            RelationId,
            Field("public_input_digest"),
            Count(ChallengeCount),
            TimingMs,
            Outcome,
        ],
        Stage::InteractionTrace => &[
            Count(LookupMultiplicityTotal),
            Count(TraceCols),
            TimingMs,
            Outcome,
        ],
        Stage::Fri => &[
            Count(FriQueryCount),
            Count(FriLayers),
            Count(ProofSizeBytes),
            TimingMs,
            Outcome,
        ],
        Stage::Prove => &[
            RelationId,
            ModelCommitment,
            Field("claimed_output_commitment"),
            Count(ProofSizeBytes),
            Field("determinism_digest"),
            TimingMs,
            Outcome,
        ],
        Stage::Verify => &[
            RelationId,
            ModelCommitment,
            Field("claimed_output_commitment"),
            Field("determinism_digest"),
            Outcome,
            TimingMs,
        ],
    }
}
