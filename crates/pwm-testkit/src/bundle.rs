// SPDX-License-Identifier: Apache-2.0
//! Typed error and JSON accessors for the demo CLI's export-bundle loaders.
//!
//! The `pwm prove-lewm` / `pwm prove-predictor` commands ingest a JSON bundle
//! emitted by the Python exporter. These helpers turn malformed user input into a
//! typed [`BundleError`] so the CLI can print a clean message and exit, instead of
//! panicking on a stray `.expect()` — matching how every other CLI failure (file
//! read, proof decode, bad argument) is already handled.

use std::fmt;

use serde_json::Value;

/// Why an export bundle could not be loaded.
#[derive(Debug)]
pub enum BundleError {
    /// The text was not valid JSON.
    Parse(serde_json::Error),
    /// A required field was absent.
    MissingField(&'static str),
    /// A field held the wrong JSON type.
    WrongType {
        /// The field name.
        field: &'static str,
        /// What was expected (for the message).
        expected: &'static str,
    },
    /// A weight value was outside the int8 range `[-128, 127]`.
    WeightOutOfRange {
        /// The offending value.
        value: i64,
    },
}

impl fmt::Display for BundleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BundleError::Parse(e) => write!(f, "invalid bundle JSON: {e}"),
            BundleError::MissingField(field) => write!(f, "bundle is missing field '{field}'"),
            BundleError::WrongType { field, expected } => {
                write!(f, "bundle field '{field}' is not {expected}")
            }
            BundleError::WeightOutOfRange { value } => {
                write!(f, "bundle weight {value} is outside int8 [-128, 127]")
            }
        }
    }
}

impl std::error::Error for BundleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            BundleError::Parse(e) => Some(e),
            _ => None,
        }
    }
}

/// Parse the bundle text into a JSON value.
pub fn parse(json: &str) -> Result<Value, BundleError> {
    serde_json::from_str(json).map_err(BundleError::Parse)
}

/// The sub-value at `name`, or [`BundleError::MissingField`].
pub fn field<'a>(v: &'a Value, name: &'static str) -> Result<&'a Value, BundleError> {
    v.get(name).ok_or(BundleError::MissingField(name))
}

/// A `u64` field (a dimension, id, or shift).
pub fn u64_at(v: &Value, name: &'static str) -> Result<u64, BundleError> {
    field(v, name)?.as_u64().ok_or(BundleError::WrongType {
        field: name,
        expected: "an unsigned integer",
    })
}

/// An `i64` field.
pub fn i64_at(v: &Value, name: &'static str) -> Result<i64, BundleError> {
    field(v, name)?.as_i64().ok_or(BundleError::WrongType {
        field: name,
        expected: "an integer",
    })
}

/// A string field, defaulting to `default` when absent or non-string.
pub fn str_at_or(v: &Value, name: &'static str, default: &str) -> String {
    v.get(name)
        .and_then(Value::as_str)
        .unwrap_or(default)
        .to_string()
}

/// An integer-array field (every element must be an integer).
pub fn ints_at(v: &Value, name: &'static str) -> Result<Vec<i64>, BundleError> {
    field(v, name)?
        .as_array()
        .ok_or(BundleError::WrongType {
            field: name,
            expected: "an array",
        })?
        .iter()
        .map(|x| {
            x.as_i64().ok_or(BundleError::WrongType {
                field: name,
                expected: "an integer array",
            })
        })
        .collect()
}
