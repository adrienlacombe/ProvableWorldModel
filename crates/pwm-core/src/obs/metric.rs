// SPDX-License-Identifier: Apache-2.0
//! The named metric set and the metric sample/snapshot types.
//!
//! Metrics are a fixed, named set (`docs/legacy-stark/spec/05-observability.md (archived)`).
//! Every metric is derived from witness *structure* or public timings/sizes,
//! never witness *values* (INV-OBS-03); label values are enum labels, component
//! names, or relation ids, never values. The wiring of real samples on the prove
//! path is #67; this module owns the names, kinds, units, label domains, and the
//! wire types.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use super::record::Outcome;

/// The aggregation kind of a metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricKind {
    /// A point-in-time value.
    Gauge,
    /// A monotonic-within-a-run total.
    Counter,
    /// A distribution over the run.
    Histogram,
}

impl MetricKind {
    /// The canonical label used in serialized samples.
    pub const fn as_str(self) -> &'static str {
        match self {
            MetricKind::Gauge => "gauge",
            MetricKind::Counter => "counter",
            MetricKind::Histogram => "histogram",
        }
    }
}

/// The definition of one named metric: its name, kind, unit, and the label keys
/// it may carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetricDef {
    /// Stable snake_case metric name.
    pub name: &'static str,
    /// Aggregation kind.
    pub kind: MetricKind,
    /// Unit string (`"rows"`, `"bytes"`, `"seconds"`, …).
    pub unit: &'static str,
    /// The label keys this metric carries.
    pub labels: &'static [&'static str],
}

/// The complete named metric set (`docs/legacy-stark/spec/05-observability.md (archived)`).
/// This is the single source of truth the registry validates samples against.
pub const METRIC_SET: &[MetricDef] = &[
    MetricDef {
        name: "trace_rows",
        kind: MetricKind::Gauge,
        unit: "rows",
        labels: &["component", "relation_id"],
    },
    MetricDef {
        name: "trace_cols",
        kind: MetricKind::Gauge,
        unit: "columns",
        labels: &["component", "trace_kind"],
    },
    MetricDef {
        name: "proof_size_bytes",
        kind: MetricKind::Gauge,
        unit: "bytes",
        labels: &["relation_id"],
    },
    MetricDef {
        name: "prove_time_seconds",
        kind: MetricKind::Histogram,
        unit: "seconds",
        labels: &["relation_id", "phase"],
    },
    MetricDef {
        name: "verify_time_seconds",
        kind: MetricKind::Histogram,
        unit: "seconds",
        labels: &["relation_id", "phase"],
    },
    MetricDef {
        name: "peak_memory_bytes",
        kind: MetricKind::Gauge,
        unit: "bytes",
        labels: &["process"],
    },
    MetricDef {
        name: "lookup_multiplicity_total",
        kind: MetricKind::Counter,
        unit: "entries",
        labels: &["table", "component"],
    },
    MetricDef {
        name: "range_check_count",
        kind: MetricKind::Counter,
        unit: "checks",
        labels: &["range", "component"],
    },
    MetricDef {
        name: "fri_query_count",
        kind: MetricKind::Gauge,
        unit: "queries",
        labels: &["relation_id"],
    },
];

/// Look up the definition of a metric by name.
pub fn metric_def(name: &str) -> Option<&'static MetricDef> {
    METRIC_SET.iter().find(|m| m.name == name)
}

/// The `table` label domain for `lookup_multiplicity_total`.
pub const LOOKUP_TABLES: &[&str] = &[
    "u8",
    "i8",
    "u16",
    "bounded_limb",
    "gelu",
    "silu",
    "softmax",
    "inv_sqrt",
    "tensor_memory",
    "weight_table",
];

/// The `range` label domain for `range_check_count`.
pub const RANGE_TABLES: &[&str] = &["u8", "i8", "u16", "bounded_limb"];

/// The `phase` label domain for `prove_time_seconds`.
pub const PROVE_PHASES: &[&str] = &[
    "trace_build",
    "commit",
    "fiat_shamir",
    "interaction_trace",
    "fri",
    "total",
];

/// The `phase` label domain for `verify_time_seconds`.
pub const VERIFY_PHASES: &[&str] = &[
    "deserialize",
    "public_input_digest",
    "fri",
    "app_checks",
    "total",
];

/// The `process` label domain for `peak_memory_bytes`.
pub const PROCESSES: &[&str] = &["export", "prover", "verifier"];

/// The wire form of one metric observation
/// (`docs/legacy-stark/spec/05-observability.md (archived)`).
#[derive(Debug, Clone, PartialEq)]
pub struct MetricSample {
    /// One of the names in [`METRIC_SET`].
    pub name: String,
    /// The metric's kind.
    pub kind: MetricKind,
    /// The metric's unit.
    pub unit: String,
    /// Label keys and values; keys restricted to the metric's label set, values
    /// always enum labels / names — never witness values.
    pub labels: BTreeMap<String, String>,
    /// Gauge/counter scalar value.
    pub value: f64,
    /// Histogram buckets as `(upper_bound, cumulative_count)`.
    pub buckets: Option<Vec<(f64, u64)>>,
}

/// A snapshot of the registry serialized at process exit — written on **both**
/// success and failure paths (INV-OBS-04).
#[derive(Debug, Clone, PartialEq)]
pub struct MetricsSnapshot {
    /// Process invocation id.
    pub run_id: [u8; 16],
    /// Relation id, once known.
    pub relation_id: Option<String>,
    /// Terminal outcome, set even when the run failed.
    pub outcome: Outcome,
    /// The collected samples.
    pub samples: Vec<MetricSample>,
}
