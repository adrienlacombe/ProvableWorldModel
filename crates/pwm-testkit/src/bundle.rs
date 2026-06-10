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
    /// The bundle's quantization scheme cannot be represented by the integer
    /// circuit because some site would need an upscale.
    UnsoundScheme {
        /// The derived quantization site.
        site: &'static str,
        /// The negative shift value.
        shift: i64,
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
            BundleError::UnsoundScheme { site, shift } => write!(
                f,
                "bundle quantization scheme needs a negative shift at {site}: {shift}"
            ),
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

/// An `i32` field.
pub fn i32_at(v: &Value, name: &'static str) -> Result<i32, BundleError> {
    let x = i64_at(v, name)?;
    i32::try_from(x).map_err(|_| BundleError::WrongType {
        field: name,
        expected: "a signed 32-bit integer",
    })
}

/// An `f64` field.
pub fn f64_at(v: &Value, name: &'static str) -> Result<f64, BundleError> {
    field(v, name)?.as_f64().ok_or(BundleError::WrongType {
        field: name,
        expected: "a finite number",
    })
}

/// A string field, defaulting to `default` when absent or non-string.
pub fn str_at_or(v: &Value, name: &'static str, default: &str) -> String {
    v.get(name)
        .and_then(Value::as_str)
        .unwrap_or(default)
        .to_string()
}

/// An optional 32-byte hex field (e.g. an export-computed commitment). Absent
/// fields are `Ok(None)`; present fields must be exactly 64 lowercase/uppercase
/// hex characters.
pub fn hex32_at_opt(v: &Value, name: &'static str) -> Result<Option<[u8; 32]>, BundleError> {
    let Some(raw) = v.get(name) else {
        return Ok(None);
    };
    let wrong = || BundleError::WrongType {
        field: name,
        expected: "a 64-character hex string",
    };
    let s = raw.as_str().ok_or_else(wrong)?;
    if s.len() != 64 || !s.is_ascii() {
        return Err(wrong());
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).map_err(|_| wrong())?;
    }
    Ok(Some(out))
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

/// A floating-point array field used only for offline export-vs-integer
/// faithfulness checks. The verifier never consumes these values.
pub fn floats_at(v: &Value, name: &'static str) -> Result<Vec<f64>, BundleError> {
    field(v, name)?
        .as_array()
        .ok_or(BundleError::WrongType {
            field: name,
            expected: "an array",
        })?
        .iter()
        .map(|x| {
            x.as_f64().ok_or(BundleError::WrongType {
                field: name,
                expected: "a numeric array",
            })
        })
        .collect()
}
