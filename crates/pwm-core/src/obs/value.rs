// SPDX-License-Identifier: Apache-2.0
//! Loggable values and the redaction guard.
//!
//! Redaction is a **type-level** guarantee (INV-OBS-02), not reviewer etiquette:
//! the only thing that can be placed in a log field, span attribute, or metric
//! label is a [`LoggableValue`], and there is deliberately **no** conversion from
//! a `BoundedInt`, `Tensor`, `M31`, `TensorCell.value`, or raw byte blob into a
//! `LoggableValue`. A developer who tries to log a witness value gets a compile
//! error, not a runtime leak (`docs/spec/05-observability.md#redaction`).

use alloc::string::String;

use super::json;

/// The only value variants permitted in an observability record.
///
/// There is intentionally no `BoundedInt`, `Tensor`, `M31`, or raw-bytes variant.
/// A private witness value cannot be *constructed* as a `LoggableValue`, so it
/// cannot reach a log line, span attribute, or metric label (INV-OBS-01/02).
#[derive(Debug, Clone, PartialEq)]
pub enum LoggableValue {
    /// A structural count (rows, columns, lookups, candidates, …).
    Count(u64),
    /// A 32-byte commitment or hash, already reduced to lowercase hex. The only
    /// sanctioned way to construct one is [`commitment`].
    Bytes32Hex(String),
    /// An enum label, op name, or `relation_id` — a *name*, never a value.
    Label(String),
    /// A duration in milliseconds.
    Duration(f64),
    /// A boolean flag.
    Bool(bool),
}

impl LoggableValue {
    /// Append this value as a bare JSON scalar.
    pub(crate) fn write_json(&self, out: &mut String) {
        match self {
            LoggableValue::Count(n) => json::push_u64(out, *n),
            LoggableValue::Bytes32Hex(s) => json::push_string(out, s),
            LoggableValue::Label(s) => json::push_string(out, s),
            LoggableValue::Duration(ms) => json::push_f64(out, *ms),
            LoggableValue::Bool(b) => json::push_bool(out, *b),
        }
    }
}

/// The sanctioned redaction helper: turn a 32-byte commitment into a loggable
/// hex value. This is the *only* way a private object enters a record — by its
/// commitment, never its contents (`docs/spec/05-observability.md#enforcement-not-etiquette`,
/// the `commit_for_log` rule). The caller computes the commitment with the
/// committed scheme (RFC-0014, #35); this function only encodes the already-bound
/// 32-byte digest.
pub fn commitment(digest: &[u8; 32]) -> LoggableValue {
    let mut hex = String::with_capacity(64);
    for byte in digest {
        push_hex_byte(&mut hex, *byte);
    }
    LoggableValue::Bytes32Hex(hex)
}

/// Construct a label value from a name (enum label, op name, relation id).
pub fn label(name: impl Into<String>) -> LoggableValue {
    LoggableValue::Label(name.into())
}

/// Construct a count value.
pub fn count(n: u64) -> LoggableValue {
    LoggableValue::Count(n)
}

fn push_hex_byte(out: &mut String, byte: u8) {
    out.push(nibble(byte >> 4));
    out.push(nibble(byte & 0xf));
}

fn nibble(n: u8) -> char {
    char::from_digit(n as u32, 16).unwrap_or('0')
}

/// Structural metadata a witness type may expose to the observability layer.
///
/// A witness struct implements `LogShape` to yield shape/count/scale metadata —
/// and *only* that. The trait has no method that can return a value, so the
/// redaction boundary holds even for types that opt into logging their structure
/// (`docs/spec/05-observability.md#enforcement-not-etiquette`).
pub trait LogShape {
    /// The tensor/op shapes touched, as `(tensor_id, dims, scale_id)` triples.
    /// Dimensions and ids are structural metadata; no cell value is exposed.
    fn log_shapes(&self) -> alloc::vec::Vec<super::record::Shape>;

    /// A structural count summarizing the object (e.g. element count). Never a
    /// value; a count of how many elements exist reveals none of them.
    fn log_count(&self) -> u64;
}
