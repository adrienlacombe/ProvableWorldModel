// SPDX-License-Identifier: Apache-2.0
//! Execution-trace data model (specs.md §5).
//!
//! The trace is an **ordered list of op records**, one per exported op, that the
//! prover fills from the integer reference inference and the verifier audits. The
//! verifier threads a running activation vector through the records: each record
//! states its input and output, the verifier checks the record is internally
//! correct (Linear via Freivalds, the rest by exact integer recompute) and that
//! its input equals the previous record's output (wiring).
//!
//! Records are pure integer data (`no_std`); activations and accumulators are
//! stored uniformly as `i64` (every int8 / int16 / i32 value fits). Leaves of the
//! `trace_root` Merkle tree are the canonical bytes of each record.

use alloc::vec::Vec;

use crate::serialize::CanonicalEncode;
use crate::transcript::blake2s256;

/// Domain tag for a trace-record Merkle leaf.
pub const TAG_TRACE_LEAF: &[u8; 16] = b"pwm.tleaf.v1\0\0\0\0";
/// Domain tag for a trace Merkle node.
pub const TAG_TRACE_NODE: &[u8; 16] = b"pwm.tnode.v1\0\0\0\0";

/// One linear op `out = W·x + bias` over integers. `W`/`bias` are committed
/// weights referenced by id; `output` is the exact i32-range accumulator (kept
/// un-requantized so the Freivalds check is exact, specs.md §7).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearRec {
    /// Stable op id (bound by the architecture commitment).
    pub op_id: u32,
    /// Weight matrix id (index into the artifact weight set), shape `[rows, cols]`.
    pub weight_id: u32,
    /// Optional bias vector id (length `rows`); `None` means no bias.
    pub bias_id: Option<u32>,
    /// Input activation `x` (length `cols`).
    pub input: Vec<i64>,
    /// Claimed `out = W·x + bias` (length `rows`).
    pub output: Vec<i64>,
}

/// Rescale an accumulator by an arithmetic right shift, round, add zero point, and
/// clamp — the exact `requantize` reference (specs.md §8.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequantRec {
    /// Stable op id.
    pub op_id: u32,
    /// Input accumulator (length `n`).
    pub input: Vec<i64>,
    /// Claimed requantized output (length `n`).
    pub output: Vec<i64>,
    /// Right-shift amount `r` (rescale by `2^-r`).
    pub shift: u32,
    /// Zero point added after rounding.
    pub zero_point: i64,
    /// Clamp lower bound.
    pub clamp_lo: i64,
    /// Clamp upper bound.
    pub clamp_hi: i64,
    /// Rounding mode discriminant (0 = nearest-ties-even, 1 = truncate).
    pub rounding: u8,
}

/// A committed lookup-table read (GELU / SiLU / etc.) verified by exact replay
/// against the table (specs.md §8.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivationRec {
    /// Stable op id.
    pub op_id: u32,
    /// Committed table id (index into the artifact table set).
    pub table_id: u32,
    /// Input values (table domain).
    pub input: Vec<i64>,
    /// Claimed per-element table outputs.
    pub output: Vec<i64>,
}

/// One op of the execution trace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpRecord {
    /// `out = W·x + bias` (Freivalds-checked).
    Linear(LinearRec),
    /// Arithmetic-right-shift rescale (exact recompute).
    Requant(RequantRec),
    /// Committed-table read (exact replay).
    Activation(ActivationRec),
}

impl OpRecord {
    /// The op's input activation vector (wiring: must equal the prior op output).
    pub fn input(&self) -> &[i64] {
        match self {
            OpRecord::Linear(r) => &r.input,
            OpRecord::Requant(r) => &r.input,
            OpRecord::Activation(r) => &r.input,
        }
    }

    /// The op's output activation vector (fed to the next op).
    pub fn output(&self) -> &[i64] {
        match self {
            OpRecord::Linear(r) => &r.output,
            OpRecord::Requant(r) => &r.output,
            OpRecord::Activation(r) => &r.output,
        }
    }

    /// Stable op id.
    pub fn op_id(&self) -> u32 {
        match self {
            OpRecord::Linear(r) => r.op_id,
            OpRecord::Requant(r) => r.op_id,
            OpRecord::Activation(r) => r.op_id,
        }
    }

    /// Variant tag for canonical encoding.
    const fn tag(&self) -> u8 {
        match self {
            OpRecord::Linear(_) => 0,
            OpRecord::Requant(_) => 1,
            OpRecord::Activation(_) => 2,
        }
    }

    /// The Merkle leaf hash of this record.
    pub fn leaf(&self) -> [u8; 32] {
        let mut payload = Vec::new();
        payload.extend_from_slice(TAG_TRACE_LEAF);
        self.encode(&mut payload);
        blake2s256(&payload)
    }
}

impl CanonicalEncode for LinearRec {
    fn encode(&self, out: &mut Vec<u8>) {
        self.op_id.encode(out);
        self.weight_id.encode(out);
        self.bias_id.encode(out);
        self.input.encode(out);
        self.output.encode(out);
    }
}

impl CanonicalEncode for RequantRec {
    fn encode(&self, out: &mut Vec<u8>) {
        self.op_id.encode(out);
        self.input.encode(out);
        self.output.encode(out);
        (self.shift as u64).encode(out);
        self.zero_point.encode(out);
        self.clamp_lo.encode(out);
        self.clamp_hi.encode(out);
        out.push(self.rounding);
    }
}

impl CanonicalEncode for ActivationRec {
    fn encode(&self, out: &mut Vec<u8>) {
        self.op_id.encode(out);
        self.table_id.encode(out);
        self.input.encode(out);
        self.output.encode(out);
    }
}

impl CanonicalEncode for OpRecord {
    fn encode(&self, out: &mut Vec<u8>) {
        out.push(self.tag());
        match self {
            OpRecord::Linear(r) => r.encode(out),
            OpRecord::Requant(r) => r.encode(out),
            OpRecord::Activation(r) => r.encode(out),
        }
    }
}

/// The trace Merkle root over the ordered record leaves (`blake2s_merkle`, the
/// same shape as the weight root): an odd level duplicates its last node; the
/// empty trace hashes the node tag as a fixed sentinel.
pub fn trace_root(records: &[OpRecord]) -> [u8; 32] {
    let mut level: Vec<[u8; 32]> = records.iter().map(OpRecord::leaf).collect();
    if level.is_empty() {
        return blake2s256(TAG_TRACE_NODE);
    }
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut i = 0;
        while i < level.len() {
            let a = level[i];
            let b = if i + 1 < level.len() {
                level[i + 1]
            } else {
                level[i]
            };
            let mut payload = Vec::with_capacity(16 + 64);
            payload.extend_from_slice(TAG_TRACE_NODE);
            payload.extend_from_slice(&a);
            payload.extend_from_slice(&b);
            next.push(blake2s256(&payload));
            i += 2;
        }
        level = next;
    }
    level[0]
}
