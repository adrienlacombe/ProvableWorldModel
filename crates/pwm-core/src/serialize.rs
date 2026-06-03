// SPDX-License-Identifier: Apache-2.0
//! Canonical byte serialization (RFC-0014 §1).
//!
//! Serialization is a total function from a typed value to a byte string with
//! **exactly one** valid encoding per value: a length-prefixed, fixed-endian,
//! tag-free binary form. Decoding rejects any non-canonical encoding, so
//! decode-then-reencode is the identity on bytes. We never hash YAML/JSON text —
//! those permit whitespace/key-order/number freedom that destroys determinism;
//! typed values are re-encoded with [`canonical_bytes`] and that is the binding
//! representation feeding every commitment.
//!
//! This module provides the primitive codec ([`CanonicalEncode`]/
//! [`CanonicalDecode`]) and the encodings for the types that exist today (`M31`,
//! [`BoundedInt`], [`Tensor`]). `PublicInput`, `ProofArtifact`, and the manifest
//! reuse this framework as those types land (#33, #64, #36).

use alloc::string::String;
use alloc::vec::Vec;

use crate::field::{M31, P};
use crate::fixed_point::{BoundError, BoundedInt, OverflowPolicy, Rounding};
use crate::relation::StatementType;
use crate::tensor::{Dtype, Scale, Tensor, TensorError};

/// A type with a canonical byte encoding.
pub trait CanonicalEncode {
    /// Append this value's canonical bytes to `out`.
    fn encode(&self, out: &mut Vec<u8>);
}

/// A type that can be decoded from canonical bytes, rejecting any non-canonical
/// encoding.
pub trait CanonicalDecode: Sized {
    /// Decode one value from `reader`, advancing it past the consumed bytes.
    fn decode(reader: &mut Reader<'_>) -> Result<Self, DecodeError>;
}

/// Serialize `value` to its canonical byte string.
pub fn canonical_bytes<T: CanonicalEncode>(value: &T) -> Vec<u8> {
    let mut out = Vec::new();
    value.encode(&mut out);
    out
}

/// Deserialize a value from canonical bytes, requiring **all** bytes to be
/// consumed (trailing bytes are a non-canonical encoding).
pub fn from_canonical_bytes<T: CanonicalDecode>(bytes: &[u8]) -> Result<T, DecodeError> {
    let mut reader = Reader::new(bytes);
    let value = T::decode(&mut reader)?;
    reader.finish()?;
    Ok(value)
}

/// A forward-only cursor over a byte slice.
pub struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    /// Construct a reader over `bytes`.
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], DecodeError> {
        let end = self.pos.checked_add(n).ok_or(DecodeError::LengthOverflow)?;
        if end > self.bytes.len() {
            return Err(DecodeError::UnexpectedEof {
                needed: n,
                have: self.bytes.len() - self.pos,
            });
        }
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    /// Assert the reader is exhausted; trailing bytes are non-canonical.
    pub fn finish(self) -> Result<(), DecodeError> {
        let remaining = self.bytes.len() - self.pos;
        if remaining == 0 {
            Ok(())
        } else {
            Err(DecodeError::TrailingBytes { remaining })
        }
    }
}

/// Error decoding canonical bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// Not enough bytes remained to decode the next value.
    UnexpectedEof {
        /// Bytes the decoder needed.
        needed: usize,
        /// Bytes that remained.
        have: usize,
    },
    /// Bytes remained after a complete value was decoded.
    TrailingBytes {
        /// Number of unconsumed bytes.
        remaining: usize,
    },
    /// A length prefix did not fit the platform `usize`.
    LengthOverflow,
    /// A `bool` byte was neither `0x00` nor `0x01`.
    InvalidBool(u8),
    /// An `Option` tag byte was neither `0x00` nor `0x01`.
    InvalidOptionTag(u8),
    /// An `M31` residue was `>= p` (not canonical).
    NonCanonicalM31(u32),
    /// An enum discriminant was not a defined variant.
    InvalidDiscriminant {
        /// The type whose discriminant was invalid.
        type_name: &'static str,
        /// The offending discriminant.
        value: u8,
    },
    /// A `String` field was not valid UTF-8.
    InvalidUtf8,
    /// A decoded `BoundedInt` violated its bound invariants.
    Bound(BoundError),
    /// A decoded `Tensor` violated its shape invariants.
    Tensor(TensorError),
}

// --- primitive integers (little-endian, fixed width) ---

macro_rules! impl_le_int {
    ($ty:ty, $n:expr) => {
        impl CanonicalEncode for $ty {
            fn encode(&self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
        }
        impl CanonicalDecode for $ty {
            fn decode(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
                let bytes = reader.take($n)?;
                let mut buf = [0u8; $n];
                buf.copy_from_slice(bytes);
                Ok(<$ty>::from_le_bytes(buf))
            }
        }
    };
}

impl_le_int!(u8, 1);
impl_le_int!(u16, 2);
impl_le_int!(u32, 4);
impl_le_int!(u64, 8);
impl_le_int!(i32, 4);
impl_le_int!(i64, 8);

impl CanonicalEncode for bool {
    fn encode(&self, out: &mut Vec<u8>) {
        out.push(if *self { 1 } else { 0 });
    }
}

impl CanonicalDecode for bool {
    fn decode(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        match reader.take(1)?[0] {
            0 => Ok(false),
            1 => Ok(true),
            b => Err(DecodeError::InvalidBool(b)),
        }
    }
}

impl CanonicalEncode for [u8; 32] {
    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(self);
    }
}

impl CanonicalDecode for [u8; 32] {
    fn decode(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        let mut buf = [0u8; 32];
        buf.copy_from_slice(reader.take(32)?);
        Ok(buf)
    }
}

impl CanonicalEncode for M31 {
    fn encode(&self, out: &mut Vec<u8>) {
        self.0.encode(out);
    }
}

impl CanonicalDecode for M31 {
    fn decode(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        let residue = u32::decode(reader)?;
        if residue >= P {
            return Err(DecodeError::NonCanonicalM31(residue));
        }
        Ok(M31::from_u32_unchecked(residue))
    }
}

impl<T: CanonicalEncode> CanonicalEncode for Option<T> {
    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            None => out.push(0),
            Some(value) => {
                out.push(1);
                value.encode(out);
            }
        }
    }
}

impl<T: CanonicalDecode> CanonicalDecode for Option<T> {
    fn decode(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        match reader.take(1)?[0] {
            0 => Ok(None),
            1 => Ok(Some(T::decode(reader)?)),
            b => Err(DecodeError::InvalidOptionTag(b)),
        }
    }
}

/// Encode a length as a `u32` prefix; panics in debug if it exceeds `u32::MAX`
/// (no canonical structure in V0 is that large).
fn encode_len(len: usize, out: &mut Vec<u8>) {
    debug_assert!(len <= u32::MAX as usize, "length exceeds u32 prefix");
    (len as u32).encode(out);
}

impl<T: CanonicalEncode> CanonicalEncode for Vec<T> {
    fn encode(&self, out: &mut Vec<u8>) {
        encode_len(self.len(), out);
        for item in self {
            item.encode(out);
        }
    }
}

impl<T: CanonicalDecode> CanonicalDecode for Vec<T> {
    fn decode(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        let len = u32::decode(reader)? as usize;
        let mut items = Vec::with_capacity(len.min(4096));
        for _ in 0..len {
            items.push(T::decode(reader)?);
        }
        Ok(items)
    }
}

impl CanonicalEncode for String {
    fn encode(&self, out: &mut Vec<u8>) {
        let bytes = self.as_bytes();
        encode_len(bytes.len(), out);
        out.extend_from_slice(bytes);
    }
}

impl CanonicalDecode for String {
    fn decode(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        let len = u32::decode(reader)? as usize;
        let bytes = reader.take(len)?;
        core::str::from_utf8(bytes)
            .map(String::from)
            .map_err(|_| DecodeError::InvalidUtf8)
    }
}

// --- composite domain types ---

impl CanonicalEncode for BoundedInt {
    fn encode(&self, out: &mut Vec<u8>) {
        // value || lo || hi (RFC-0014 §1): the bounds are load-bearing bytes.
        self.value().encode(out);
        self.lo().encode(out);
        self.hi().encode(out);
    }
}

impl CanonicalDecode for BoundedInt {
    fn decode(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        let value = i64::decode(reader)?;
        let lo = i64::decode(reader)?;
        let hi = i64::decode(reader)?;
        BoundedInt::new(value, lo, hi).map_err(DecodeError::Bound)
    }
}

impl CanonicalEncode for Tensor {
    fn encode(&self, out: &mut Vec<u8>) {
        self.tensor_id().encode(out);
        self.scale_id().encode(out);
        encode_len(self.shape().len(), out); // rank
        for &dim in self.shape() {
            dim.encode(out);
        }
        encode_len(self.data().len(), out); // len == product(shape)
        for cell in self.data() {
            cell.encode(out);
        }
    }
}

impl CanonicalDecode for Tensor {
    fn decode(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        let tensor_id = u32::decode(reader)?;
        let scale_id = u32::decode(reader)?;
        let rank = u32::decode(reader)? as usize;
        let mut shape = Vec::with_capacity(rank.min(8));
        for _ in 0..rank {
            shape.push(u32::decode(reader)?);
        }
        let len = u32::decode(reader)? as usize;
        let mut data = Vec::with_capacity(len.min(4096));
        for _ in 0..len {
            data.push(BoundedInt::decode(reader)?);
        }
        // Tensor::new re-checks len == product(shape): a mismatched encoded len
        // (a non-canonical Tensor) is rejected here.
        Tensor::new(tensor_id, shape, scale_id, data).map_err(DecodeError::Tensor)
    }
}

impl CanonicalEncode for StatementType {
    fn encode(&self, out: &mut Vec<u8>) {
        out.push(self.discriminant());
    }
}

impl CanonicalDecode for StatementType {
    fn decode(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        let value = reader.take(1)?[0];
        StatementType::from_discriminant(value).ok_or(DecodeError::InvalidDiscriminant {
            type_name: "StatementType",
            value,
        })
    }
}

/// Helper: encode/decode a fieldless enum as a `u8` discriminant.
macro_rules! impl_enum_u8 {
    ($ty:ty, $name:literal) => {
        impl CanonicalEncode for $ty {
            fn encode(&self, out: &mut Vec<u8>) {
                out.push(self.discriminant());
            }
        }
        impl CanonicalDecode for $ty {
            fn decode(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
                let value = reader.take(1)?[0];
                <$ty>::from_discriminant(value).ok_or(DecodeError::InvalidDiscriminant {
                    type_name: $name,
                    value,
                })
            }
        }
    };
}

impl_enum_u8!(Rounding, "Rounding");
impl_enum_u8!(OverflowPolicy, "OverflowPolicy");
impl_enum_u8!(Dtype, "Dtype");

impl CanonicalEncode for Scale {
    fn encode(&self, out: &mut Vec<u8>) {
        self.scale_id.encode(out);
        self.log2.encode(out);
        self.dtype.encode(out);
    }
}

impl CanonicalDecode for Scale {
    fn decode(reader: &mut Reader<'_>) -> Result<Self, DecodeError> {
        Ok(Scale {
            scale_id: u32::decode(reader)?,
            log2: i32::decode(reader)?,
            dtype: Dtype::decode(reader)?,
        })
    }
}
