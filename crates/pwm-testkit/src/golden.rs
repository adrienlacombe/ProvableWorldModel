// SPDX-License-Identifier: Apache-2.0
//! Loader for committed golden-vector fixtures.
//!
//! Golden vectors are the differential oracle of the test stack
//! (`docs/spec/07-testing-strategy.md#golden-vectors`): the Python fixed-point
//! reference (layer 2) emits them and the Rust fixed-point reference (layer 3)
//! must reproduce every integer bit-for-bit (INV-TEST-02/06, tolerance zero).
//!
//! Fixtures are committed as canonical JSON with **integers only** in the
//! fixed-point section. This loader parses that format and enforces the
//! integer-only rule structurally — every numeric value deserializes into an
//! integer type, so a stray float in a fixture is a parse error, not a silently
//! coerced value. Generation of fixtures is the export pipeline's job (#39); this
//! is the read side every layer-3 test shares.

use std::fmt;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Header recorded at the top of every committed golden fixture.
///
/// Pins the provenance of the vectors so a fixture is meaningful only for the
/// exact relation and quantization it was generated against
/// (`docs/spec/07-testing-strategy.md#generation-storage-and-versioning`).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct GoldenHeader {
    /// Immutable relation identity, e.g. `pwm.lewm.predictor_step.v1`.
    pub relation_id: String,
    /// Manifest schema version the vectors were generated against.
    pub manifest_version: u32,
    /// Hex commitment binding the quantization (scales, rounding, tables).
    pub quantization_commitment: String,
    /// Version of the export tool that produced the fixture.
    pub export_tool_version: String,
    /// Deterministic generator seed (INV-TEST-03); recorded for reproducibility.
    pub generator_seed: u64,
}

/// Inclusive integer input domain for a primitive's golden vectors.
///
/// "Explicit domain bounds" per the spec: the `[lo, hi]` range each input is
/// drawn from, so a reader knows which boundary/one-past-boundary cases the
/// fixture is expected to cover.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub struct Domain {
    /// Inclusive lower bound.
    pub lo: i64,
    /// Inclusive upper bound.
    pub hi: i64,
}

/// One golden test vector: integer inputs mapped to integer outputs.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct GoldenVector {
    /// Optional label for this vector (e.g. `"tie"`, `"hi+1"`).
    #[serde(default)]
    pub label: Option<String>,
    /// Integer inputs to the primitive.
    pub inputs: Vec<i64>,
    /// Expected integer outputs (the committed reference result).
    pub outputs: Vec<i64>,
}

/// A committed per-primitive golden fixture (e.g. `requantize.ntte.json`).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct GoldenFixture {
    /// Provenance header.
    pub header: GoldenHeader,
    /// The primitive these vectors exercise, e.g. `"requantize"`, `"gelu_q"`.
    pub primitive: String,
    /// The declared inclusive input domain the vectors are drawn from.
    pub domain: Domain,
    /// The committed input/output vectors. Never empty in a valid fixture.
    pub vectors: Vec<GoldenVector>,
}

impl GoldenFixture {
    /// Parse a fixture from canonical-JSON bytes and validate its structure.
    ///
    /// Fails with [`GoldenError::Parse`] if any numeric value is not an integer
    /// (the integer-only rule), and with the structural variants if the domain
    /// is inverted or the fixture carries no vectors.
    pub fn from_slice(bytes: &[u8]) -> Result<Self, GoldenError> {
        let fixture: GoldenFixture = serde_json::from_slice(bytes).map_err(GoldenError::Parse)?;
        fixture.validate()?;
        Ok(fixture)
    }

    /// Parse a fixture from a canonical-JSON string. See [`Self::from_slice`].
    ///
    /// Named `from_json_str` rather than `from_str` deliberately: this is not the
    /// `core::str::FromStr` contract (it takes a JSON document, not a `Display`
    /// round-trip), so it does not implement that trait.
    pub fn from_json_str(text: &str) -> Result<Self, GoldenError> {
        Self::from_slice(text.as_bytes())
    }

    /// Validate structural invariants: a non-inverted domain and at least one
    /// vector. The integer-only rule is enforced earlier, at parse time.
    pub fn validate(&self) -> Result<(), GoldenError> {
        if self.domain.lo > self.domain.hi {
            return Err(GoldenError::DomainInverted {
                lo: self.domain.lo,
                hi: self.domain.hi,
            });
        }
        if self.vectors.is_empty() {
            return Err(GoldenError::Empty);
        }
        Ok(())
    }
}

/// Load and validate a golden fixture from a path on disk.
pub fn load_fixture(path: impl AsRef<Path>) -> Result<GoldenFixture, GoldenError> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|source| GoldenError::Io {
        path: path.display().to_string(),
        source,
    })?;
    GoldenFixture::from_slice(&bytes)
}

/// Errors from loading or validating a golden fixture.
#[derive(Debug)]
pub enum GoldenError {
    /// The fixture file could not be read.
    Io {
        /// The path that failed to read.
        path: String,
        /// The underlying I/O error.
        source: std::io::Error,
    },
    /// The bytes were not valid canonical JSON, or carried a non-integer value
    /// in the fixed-point section (a float where an integer is required).
    Parse(serde_json::Error),
    /// The fixture declared an inverted domain (`lo > hi`).
    DomainInverted {
        /// Declared lower bound.
        lo: i64,
        /// Declared upper bound.
        hi: i64,
    },
    /// The fixture carried no vectors.
    Empty,
}

impl fmt::Display for GoldenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GoldenError::Io { path, source } => {
                write!(f, "reading golden fixture '{path}': {source}")
            }
            GoldenError::Parse(e) => write!(f, "parsing golden fixture: {e}"),
            GoldenError::DomainInverted { lo, hi } => {
                write!(f, "golden fixture has inverted domain: lo={lo} > hi={hi}")
            }
            GoldenError::Empty => write!(f, "golden fixture has no vectors"),
        }
    }
}

impl std::error::Error for GoldenError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GoldenError::Io { source, .. } => Some(source),
            GoldenError::Parse(e) => Some(e),
            _ => None,
        }
    }
}
