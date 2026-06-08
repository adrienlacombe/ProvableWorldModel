// SPDX-License-Identifier: Apache-2.0
//! Committed integer lookup tables for nonlinearities (specs.md §3, §8.3).
//!
//! GELU, SiLU, softmax-exp, and inverse-sqrt are realized as **committed integer
//! tables**: a contiguous output array over an input domain `[lo, lo+len)`. The
//! exporter generates them by quantizing the reference nonlinearity; the prover
//! records the looked-up outputs; the verifier replays the lookup exactly and
//! checks the table commitment is bound by `quantization_commitment`. There is no
//! floating point and no approximation gap at verify time — the table *is* the
//! definition of the quantized nonlinearity.

use alloc::vec::Vec;

use crate::commit::commit;
use crate::serialize::{canonical_bytes, CanonicalDecode, CanonicalEncode, DecodeError, Reader};

/// Domain tag for an activation-table commitment.
pub const TAG_TABLE: &[u8; 16] = b"pwm.table.v1\0\0\0\0";

/// A committed integer lookup table over the contiguous input domain
/// `[lo, lo + outputs.len())`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivationTable {
    /// Stable table id (bound by `quantization_commitment`).
    pub table_id: u32,
    /// Inclusive lower bound of the input domain.
    pub lo: i64,
    /// `outputs[i]` is the table value for input `lo + i`.
    pub outputs: Vec<i64>,
}

impl ActivationTable {
    /// Exact table read: the output for input `x`, or `None` if `x` is outside the
    /// committed domain (an out-of-domain read is a verifier rejection).
    pub fn eval(&self, x: i64) -> Option<i64> {
        if x < self.lo {
            return None;
        }
        let idx = (x - self.lo) as usize;
        self.outputs.get(idx).copied()
    }

    /// Inclusive upper bound of the input domain.
    pub fn hi(&self) -> i64 {
        self.lo + self.outputs.len() as i64 - 1
    }

    /// `commit(TAG_TABLE, canonical_bytes(self))`.
    pub fn commitment(&self) -> [u8; 32] {
        commit(TAG_TABLE, &canonical_bytes(self))
    }
}

impl CanonicalEncode for ActivationTable {
    fn encode(&self, out: &mut Vec<u8>) {
        self.table_id.encode(out);
        self.lo.encode(out);
        self.outputs.encode(out);
    }
}

impl CanonicalDecode for ActivationTable {
    fn decode(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        Ok(ActivationTable {
            table_id: u32::decode(reader)?,
            lo: i64::decode(reader)?,
            outputs: Vec::<i64>::decode(reader)?,
        })
    }
}

/// The commitment over an ordered set of activation tables (the
/// `activation_tables_commitment` bound by `QuantBinding`): sorted by ascending
/// `table_id`, encoded as a `Vec`, committed under [`TAG_TABLE`].
pub fn activation_tables_commitment(tables: &[ActivationTable]) -> [u8; 32] {
    let mut sorted = tables.to_vec();
    sorted.sort_by_key(|t| t.table_id);
    commit(TAG_TABLE, &canonical_bytes(&sorted))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_in_and_out_of_domain() {
        let t = ActivationTable {
            table_id: 1,
            lo: -2,
            outputs: alloc::vec![10, 20, 30, 40, 50],
        };
        assert_eq!(t.eval(-2), Some(10));
        assert_eq!(t.eval(0), Some(30));
        assert_eq!(t.eval(2), Some(50));
        assert_eq!(t.hi(), 2);
        assert_eq!(t.eval(-3), None);
        assert_eq!(t.eval(3), None);
    }

    #[test]
    fn commitment_binds_contents() {
        let a = ActivationTable {
            table_id: 1,
            lo: 0,
            outputs: alloc::vec![1, 2, 3],
        };
        let mut b = a.clone();
        b.outputs[1] = 99;
        assert_ne!(a.commitment(), b.commitment());
    }
}
