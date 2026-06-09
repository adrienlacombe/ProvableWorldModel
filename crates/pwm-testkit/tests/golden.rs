// SPDX-License-Identifier: Apache-2.0
//! Tests for the golden-vector loader: it reads the committed format and
//! enforces the integer-only and structural rules.

use std::path::PathBuf;

use pwm_testkit::golden::{load_fixture, GoldenError, GoldenFixture};

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name)
}

#[test]
fn loads_committed_sample_fixture() {
    let fixture =
        load_fixture(fixture_path("requantize.ntte.sample.json")).expect("sample fixture loads");

    assert_eq!(fixture.primitive, "requantize");
    assert_eq!(fixture.header.relation_id, "pwm.lewm.predictor_step.v1");
    assert_eq!(fixture.header.manifest_version, 1);
    assert_eq!(fixture.domain.lo, -2048);
    assert_eq!(fixture.domain.hi, 2047);
    assert_eq!(fixture.vectors.len(), 5);

    // Spot-check the first vector and that labels survive the round trip.
    let exact = &fixture.vectors[0];
    assert_eq!(exact.label.as_deref(), Some("exact"));
    assert_eq!(exact.inputs, vec![256, 3]);
    assert_eq!(exact.outputs, vec![32]);
}

#[test]
fn committed_requantize_vectors_match_the_primitive() {
    // The fixture is the differential oracle (specs.md §15, INV-TEST-02/06): run each
    // committed vector through the real Rust primitive, not just parse it. This catches
    // any silent divergence between the JSON and `requantize` (the companion test in
    // pwm-core/tests/requantize.rs hardcodes the same values, so without this they could
    // drift apart undetected).
    use pwm_core::fixed_point::{requantize, Rounding::NearestTiesToEven};
    let fixture =
        load_fixture(fixture_path("requantize.ntte.sample.json")).expect("sample fixture loads");
    assert_eq!(fixture.primitive, "requantize");
    for v in &fixture.vectors {
        // requantize fixtures encode `inputs` as [n, shift] and `outputs` as [result].
        assert_eq!(v.inputs.len(), 2, "requantize vector needs [n, shift]");
        assert_eq!(v.outputs.len(), 1, "requantize vector needs one output");
        let (n, shift) = (v.inputs[0], v.inputs[1]);
        let got = requantize(
            n,
            u32::try_from(shift).expect("non-negative shift"),
            0,
            i64::MIN,
            i64::MAX,
            NearestTiesToEven,
        );
        assert_eq!(
            got, v.outputs[0],
            "fixture vector {:?}: requantize({n}, {shift}) = {got}, want {}",
            v.label, v.outputs[0]
        );
    }
}

#[test]
fn rejects_float_in_fixed_point_section() {
    // The fixed-point section is integers only; a float must not silently coerce.
    let json = r#"{
        "header": {
            "relation_id": "r", "manifest_version": 1,
            "quantization_commitment": "0x00", "export_tool_version": "x",
            "generator_seed": 0
        },
        "primitive": "requantize",
        "domain": { "lo": 0, "hi": 10 },
        "vectors": [ { "inputs": [1.5], "outputs": [2] } ]
    }"#;
    let err = GoldenFixture::from_json_str(json).expect_err("float input must be rejected");
    assert!(matches!(err, GoldenError::Parse(_)), "got {err:?}");
}

#[test]
fn rejects_empty_and_inverted_domain() {
    let empty = r#"{
        "header": { "relation_id": "r", "manifest_version": 1,
            "quantization_commitment": "0x00", "export_tool_version": "x", "generator_seed": 0 },
        "primitive": "p", "domain": { "lo": 0, "hi": 10 }, "vectors": []
    }"#;
    assert!(matches!(
        GoldenFixture::from_json_str(empty),
        Err(GoldenError::Empty)
    ));

    let inverted = r#"{
        "header": { "relation_id": "r", "manifest_version": 1,
            "quantization_commitment": "0x00", "export_tool_version": "x", "generator_seed": 0 },
        "primitive": "p", "domain": { "lo": 10, "hi": 0 },
        "vectors": [ { "inputs": [1], "outputs": [1] } ]
    }"#;
    assert!(matches!(
        GoldenFixture::from_json_str(inverted),
        Err(GoldenError::DomainInverted { lo: 10, hi: 0 })
    ));
}

#[test]
fn missing_file_is_an_io_error() {
    let err = load_fixture(fixture_path("does-not-exist.json")).expect_err("missing file errors");
    assert!(matches!(err, GoldenError::Io { .. }), "got {err:?}");
}
