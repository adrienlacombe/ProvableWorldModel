// SPDX-License-Identifier: Apache-2.0
//! Tests for the observability schema, redaction guard, required-field policy,
//! metric set, and span scaffolding. Gated on the default `obs` feature (the
//! schema it exercises is excluded from the no_std verifier trust root).
#![cfg(feature = "obs")]

use std::collections::BTreeMap;

use pwm_core::obs::record::{required_terminal_fields, CountField, Counts, RequiredField};
use pwm_core::obs::span::{trace_build_span, TRACE_BUILD_COMPONENTS};
use pwm_core::obs::value::{commitment, label, LoggableValue};
use pwm_core::obs::{metric_def, Level, LogRecord, Outcome, Stage, METRIC_SET};
use pwm_core::StatementType;

fn complete_export_record() -> LogRecord {
    let mut rec = LogRecord::new(Level::Info, Stage::Export, 1, [0u8; 16], "export complete");
    rec.model_commitment = Some([0xab; 32]);
    rec.relation_id = Some("pwm.lewm.predictor_step.v1".to_string());
    rec.shapes = Some(vec![pwm_core::obs::Shape {
        tensor_id: 0,
        dims: vec![3, 192],
        scale_id: 7,
    }]);
    rec.counts = Some(Counts {
        tensor_count: Some(42),
        ..Counts::default()
    });
    rec.timing_ms = Some(12.5);
    rec.outcome = Some(Outcome::Ok);
    rec
}

#[test]
fn level_threshold_changes_events_not_content() {
    // Default threshold INFO emits ERROR/WARN/INFO, suppresses DEBUG/TRACE.
    assert!(Level::Error.should_emit(Level::Info));
    assert!(Level::Info.should_emit(Level::Info));
    assert!(!Level::Debug.should_emit(Level::Info));
    assert!(!Level::Trace.should_emit(Level::Info));
    // Lowering the threshold reveals more events.
    assert!(Level::Trace.should_emit(Level::Trace));
}

#[test]
fn complete_terminal_record_has_no_missing_fields() {
    let rec = complete_export_record();
    assert!(
        rec.missing_terminal_fields().is_empty(),
        "export record should be complete"
    );
}

#[test]
fn incomplete_terminal_record_reports_missing_fields() {
    let mut rec = complete_export_record();
    rec.counts = None; // drop tensor_count
    rec.timing_ms = None;
    let missing = rec.missing_terminal_fields();
    assert!(missing.contains(&RequiredField::Count(CountField::TensorCount)));
    assert!(missing.contains(&RequiredField::TimingMs));
}

#[test]
fn verify_reject_requires_error_code() {
    let mut rec = LogRecord::new(
        Level::Error,
        Stage::Verify,
        3,
        [1u8; 16],
        "argmin tie-break failed",
    );
    rec.relation_id = Some("pwm.lewm.fixed_candidate_planning.v1".to_string());
    rec.model_commitment = Some([0x3a; 32]);
    rec.fields.insert(
        "claimed_output_commitment".to_string(),
        commitment(&[0x77; 32]),
    );
    rec.fields
        .insert("determinism_digest".to_string(), commitment(&[0xe1; 32]));
    rec.outcome = Some(Outcome::Error);
    rec.timing_ms = Some(1.0);
    // error_code missing -> reported.
    assert!(rec
        .missing_terminal_fields()
        .contains(&RequiredField::ErrorCode));
    rec.error_code = Some("PWM-VERIFY-0007".to_string());
    assert!(rec.missing_terminal_fields().is_empty());
}

#[test]
fn every_stage_has_required_fields_defined() {
    for stage in [
        Stage::Export,
        Stage::Quantize,
        Stage::ReferenceInference,
        Stage::ManifestWrite,
        Stage::TraceBuild,
        Stage::Commit,
        Stage::FiatShamir,
        Stage::InteractionTrace,
        Stage::Fri,
        Stage::Prove,
        Stage::Verify,
    ] {
        assert!(
            !required_terminal_fields(stage).is_empty(),
            "{stage:?} must require fields"
        );
    }
}

#[test]
fn record_serializes_to_well_formed_value_free_json() {
    let mut rec = complete_export_record();
    rec.statement_type = Some(StatementType::P0Step);
    rec.fields.insert(
        "claimed_output_commitment".to_string(),
        commitment(&[0x77; 32]),
    );
    rec.fields.insert("component".to_string(), label("linear"));

    let json = rec.to_json();
    // Single line.
    assert!(!json.contains('\n'));
    // Parses as valid JSON.
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("emitted JSON is valid");

    assert_eq!(parsed["log_schema_version"], 1);
    assert_eq!(parsed["level"], "INFO");
    assert_eq!(parsed["stage"], "Export");
    assert_eq!(parsed["statement_type"], "P0Step");
    assert_eq!(parsed["model_commitment"], "ab".repeat(32));
    assert_eq!(parsed["counts"]["tensor_count"], 42);
    assert_eq!(parsed["shapes"][0]["dims"][1], 192);
    assert_eq!(
        parsed["fields"]["claimed_output_commitment"],
        "77".repeat(32)
    );
    assert_eq!(parsed["message"], "export complete");
}

#[test]
fn loggable_value_only_carries_redaction_safe_shapes() {
    // The only constructors yield counts, hex commitments, labels, durations,
    // bools — there is no path from a witness integer to a LoggableValue.
    let v = commitment(&[0u8; 32]);
    assert!(matches!(v, LoggableValue::Bytes32Hex(ref s) if s.len() == 64));
    assert!(matches!(label("attention"), LoggableValue::Label(_)));
    assert!(matches!(pwm_core::obs::count(7), LoggableValue::Count(7)));
}

#[test]
fn metric_set_names_are_unique_and_resolvable() {
    let mut seen = BTreeMap::new();
    for def in METRIC_SET {
        assert!(
            seen.insert(def.name, ()).is_none(),
            "duplicate metric {}",
            def.name
        );
        assert!(!def.labels.is_empty(), "metric {} has labels", def.name);
        assert!(metric_def(def.name).is_some());
    }
    // Spot-check a few names match the spec.
    assert!(metric_def("trace_rows").is_some());
    assert!(metric_def("prove_time_seconds").is_some());
    assert!(metric_def("not_a_metric").is_none());
}

#[test]
fn trace_build_span_names_match_component_set() {
    assert_eq!(
        trace_build_span("linear").as_deref(),
        Some("trace_build.linear")
    );
    assert_eq!(
        trace_build_span("argmin").as_deref(),
        Some("trace_build.argmin")
    );
    // An off-spec name does not mint a span.
    assert_eq!(trace_build_span("cem"), None); // P3, out of V0 scope
    assert_eq!(trace_build_span("typo"), None);
    assert_eq!(TRACE_BUILD_COMPONENTS.len(), 13);
}
