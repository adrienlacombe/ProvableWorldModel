// SPDX-License-Identifier: Apache-2.0
//! The static operator graph (the committed op wiring; specs.md §2.1, §4).
//!
//! The graph is the witness-independent description of the model: the ordered op
//! list with kinds, shapes, weight/table bindings, and requant parameters. Its
//! commitment is the `architecture_commitment` inside the model commitment, so a
//! prover cannot change the op wiring without changing the model commitment. The
//! verifier checks that the execution [`crate::trace`] conforms to this graph
//! (same length, same op kinds/ids/dims) before auditing the witness values.

use alloc::vec::Vec;

use crate::commit::commit;
use crate::serialize::{canonical_bytes, CanonicalDecode, CanonicalEncode, DecodeError, Reader};

/// Domain tag for the architecture (op-graph) commitment.
pub const TAG_GRAPH: &[u8; 16] = b"pwm.graph.v1\0\0\0\0";

/// One static op in the graph (no witness values; only kinds, ids, dims, params).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpSpec {
    /// `out = W·x + bias`; `W` has shape `[rows, cols]`.
    Linear {
        /// Stable op id.
        op_id: u32,
        /// Weight tensor id.
        weight_id: u32,
        /// Optional bias tensor id.
        bias_id: Option<u32>,
        /// Output dimension.
        rows: u32,
        /// Input dimension.
        cols: u32,
    },
    /// Arithmetic-right-shift rescale with the locked params.
    Requant {
        /// Stable op id.
        op_id: u32,
        /// Right-shift amount.
        shift: u32,
        /// Zero point.
        zero_point: i64,
        /// Clamp lower bound.
        clamp_lo: i64,
        /// Clamp upper bound.
        clamp_hi: i64,
        /// Rounding-mode discriminant.
        rounding: u8,
    },
    /// Committed-table read bound to `table_id`.
    Activation {
        /// Stable op id.
        op_id: u32,
        /// Committed table id.
        table_id: u32,
    },
    /// Affine-free LayerNorm bound to an inverse-sqrt `table_id`.
    LayerNorm {
        /// Stable op id.
        op_id: u32,
        /// Committed inverse-sqrt table id.
        table_id: u32,
        /// Right-shift amount.
        shift: u32,
        /// Clamp lower bound.
        clamp_lo: i64,
        /// Clamp upper bound.
        clamp_hi: i64,
        /// Rounding-mode discriminant.
        rounding: u8,
    },
}

impl OpSpec {
    const fn tag(&self) -> u8 {
        match self {
            OpSpec::Linear { .. } => 0,
            OpSpec::Requant { .. } => 1,
            OpSpec::Activation { .. } => 2,
            OpSpec::LayerNorm { .. } => 3,
        }
    }

    /// Stable op id.
    pub const fn op_id(&self) -> u32 {
        match *self {
            OpSpec::Linear { op_id, .. } => op_id,
            OpSpec::Requant { op_id, .. } => op_id,
            OpSpec::Activation { op_id, .. } => op_id,
            OpSpec::LayerNorm { op_id, .. } => op_id,
        }
    }
}

impl CanonicalEncode for OpSpec {
    fn encode(&self, out: &mut Vec<u8>) {
        out.push(self.tag());
        match *self {
            OpSpec::Linear {
                op_id,
                weight_id,
                bias_id,
                rows,
                cols,
            } => {
                op_id.encode(out);
                weight_id.encode(out);
                bias_id.encode(out);
                rows.encode(out);
                cols.encode(out);
            }
            OpSpec::Requant {
                op_id,
                shift,
                zero_point,
                clamp_lo,
                clamp_hi,
                rounding,
            } => {
                op_id.encode(out);
                (shift as u64).encode(out);
                zero_point.encode(out);
                clamp_lo.encode(out);
                clamp_hi.encode(out);
                out.push(rounding);
            }
            OpSpec::Activation { op_id, table_id } => {
                op_id.encode(out);
                table_id.encode(out);
            }
            OpSpec::LayerNorm {
                op_id,
                table_id,
                shift,
                clamp_lo,
                clamp_hi,
                rounding,
            } => {
                op_id.encode(out);
                table_id.encode(out);
                (shift as u64).encode(out);
                clamp_lo.encode(out);
                clamp_hi.encode(out);
                out.push(rounding);
            }
        }
    }
}

impl CanonicalDecode for OpSpec {
    fn decode(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        let tag = u8::decode(reader)?;
        Ok(match tag {
            0 => OpSpec::Linear {
                op_id: u32::decode(reader)?,
                weight_id: u32::decode(reader)?,
                bias_id: Option::<u32>::decode(reader)?,
                rows: u32::decode(reader)?,
                cols: u32::decode(reader)?,
            },
            1 => OpSpec::Requant {
                op_id: u32::decode(reader)?,
                shift: u64::decode(reader)? as u32,
                zero_point: i64::decode(reader)?,
                clamp_lo: i64::decode(reader)?,
                clamp_hi: i64::decode(reader)?,
                rounding: u8::decode(reader)?,
            },
            2 => OpSpec::Activation {
                op_id: u32::decode(reader)?,
                table_id: u32::decode(reader)?,
            },
            3 => OpSpec::LayerNorm {
                op_id: u32::decode(reader)?,
                table_id: u32::decode(reader)?,
                shift: u64::decode(reader)? as u32,
                clamp_lo: i64::decode(reader)?,
                clamp_hi: i64::decode(reader)?,
                rounding: u8::decode(reader)?,
            },
            v => {
                return Err(DecodeError::InvalidDiscriminant {
                    type_name: "OpSpec",
                    value: v,
                })
            }
        })
    }
}

/// The static op graph: an ordered op list.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GraphSpec {
    /// The ordered ops.
    pub ops: Vec<OpSpec>,
}

impl GraphSpec {
    /// `commit(TAG_GRAPH, canonical_bytes(self))` — the `architecture_commitment`.
    pub fn commitment(&self) -> [u8; 32] {
        commit(TAG_GRAPH, &canonical_bytes(self))
    }
}

impl CanonicalEncode for GraphSpec {
    fn encode(&self, out: &mut Vec<u8>) {
        self.ops.encode(out);
    }
}

impl CanonicalDecode for GraphSpec {
    fn decode(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        Ok(GraphSpec {
            ops: Vec::<OpSpec>::decode(reader)?,
        })
    }
}
