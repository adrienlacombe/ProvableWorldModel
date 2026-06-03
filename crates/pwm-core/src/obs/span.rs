// SPDX-License-Identifier: Apache-2.0
//! Span-tree scaffolding: the canonical, normative span names and nesting.
//!
//! The pipeline is instrumented with spans whose names and nesting are normative
//! (`docs/spec/05-observability.md#span-tree`), so a flamegraph reads identically
//! across runs and contributors. This module owns the names and the canonical
//! `trace_build.<component>` execution order; the `tracing` instrumentation that
//! opens/closes them lives on the prove/verify path (#67). Span attributes obey
//! the same redaction rules as log fields (INV-OBS-01) — shapes and counts,
//! never values.

/// Root span of the offline export path.
pub const SPAN_EXPORT: &str = "export";
/// Per-tensor quantization of the exported graph (child of `export`).
pub const SPAN_QUANTIZE: &str = "quantize";
/// Deterministic fixed-point reference run (child of `quantize`).
pub const SPAN_REFERENCE_INFERENCE: &str = "reference_inference";

/// Prover root span for the planning statement (P2).
pub const SPAN_PROVE_PLANNING: &str = "prove_planning";
/// Prover root span for the rollout statements (P0/P1).
pub const SPAN_PROVE_ROLLOUT: &str = "prove_rollout";

/// Commit span: Merkle commitment of preprocessed + main traces.
pub const SPAN_COMMIT: &str = "commit";
/// Fiat-Shamir challenge derivation span.
pub const SPAN_FIAT_SHAMIR: &str = "fiat_shamir";
/// LogUp interaction-trace construction span.
pub const SPAN_INTERACTION_TRACE: &str = "interaction_trace";
/// FRI commit + query + proof serialization span.
pub const SPAN_FRI: &str = "fri";

/// Verifier root span.
pub const SPAN_VERIFY: &str = "verify";
/// Public-input digest recomputation span (child of `verify`).
pub const SPAN_PUBLIC_INPUT_DIGEST: &str = "public_input_digest";
/// Application-level checks span (child of `verify`).
pub const SPAN_APP_CHECKS: &str = "app_checks";

/// The prefix every per-component trace-build span name shares.
pub const TRACE_BUILD_PREFIX: &str = "trace_build.";

/// The AIR components, in canonical trace-build execution order
/// (`docs/spec/01-architecture.md#component-model`). `cem` is P3-only and out of
/// V0 scope, so it is excluded here.
pub const TRACE_BUILD_COMPONENTS: &[&str] = &[
    "range_check",
    "tensor_memory",
    "linear",
    "matmul",
    "requant",
    "activation_lookup",
    "layernorm",
    "attention",
    "mlp",
    "predictor",
    "rollout",
    "cost",
    "argmin",
];

/// The canonical span name for a component's trace-build span, e.g.
/// `trace_build.linear`. Returns `None` for a name not in
/// [`TRACE_BUILD_COMPONENTS`], so a typo cannot mint an off-spec span.
pub fn trace_build_span(component: &str) -> Option<alloc::string::String> {
    if TRACE_BUILD_COMPONENTS.contains(&component) {
        let mut name =
            alloc::string::String::with_capacity(TRACE_BUILD_PREFIX.len() + component.len());
        name.push_str(TRACE_BUILD_PREFIX);
        name.push_str(component);
        Some(name)
    } else {
        None
    }
}
