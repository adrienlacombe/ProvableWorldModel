// SPDX-License-Identifier: Apache-2.0
//! Observability: the structured-logging schema, the redaction guard, the named
//! metric set, and the span-tree scaffolding.
//!
//! This module owns the canonical observability types (design archived under
//! `docs/legacy-stark/spec/05-observability.md`):
//!
//! - [`record`](crate::obs::record) — the [`LogRecord`](crate::obs::record::LogRecord)
//!   schema, the [`Level`](crate::obs::record::Level) policy, and the per-stage
//!   required-field contract.
//! - [`value`](crate::obs::value) — [`LoggableValue`](crate::obs::value::LoggableValue)
//!   and the redaction guard. Redaction is a type-level guarantee (INV-OBS-02): a
//!   witness value is not *representable* as a loggable field, so a careless log
//!   call is a compile error, not a leak.
//! - [`metric`](crate::obs::metric) — the named metric set, label domains, and the
//!   sample/snapshot wire types.
//! - [`span`](crate::obs::span) — the normative span names and the canonical
//!   `trace_build.<component>` execution order.
//!
//! The types live here, in the `no_std` dependency-free trust root, so the
//! prover and verifier share one definition. The *emission* — wiring real
//! samples and `tracing` spans through the prove path — is #67; this module is
//! the schema and the guard those emitters are bound by.

pub mod metric;
pub mod record;
pub mod span;
pub mod value;

mod json;

pub use metric::{metric_def, MetricDef, MetricKind, MetricSample, MetricsSnapshot, METRIC_SET};
pub use record::{
    required_terminal_fields, CountField, Counts, Level, LogRecord, Outcome, RequiredField, Shape,
    Stage, DEFAULT_LEVEL, LOG_SCHEMA_VERSION,
};
pub use value::{commitment, count, label, LogShape, LoggableValue};
