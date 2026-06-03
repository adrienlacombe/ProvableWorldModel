// SPDX-License-Identifier: Apache-2.0
//! Dense row-major tensors of bounded integers (RFC-0002, RFC-0001).
//!
//! A [`Tensor`] is a row-major array of [`BoundedInt`] bound to a single
//! `scale_id` (per-tensor quantization, the V0 policy). The structural invariant
//! `len(data) == product(shape)` (INV-DM-01) and the ≤4-dimension limit are
//! enforced at construction; the scale invariants (the `scale_id` is declared,
//! and every element's bound fits the scale's dtype — INV-DM-05/INV-DM-11) are
//! checked against a scale table via [`Tensor::validate_scale`].

use alloc::vec::Vec;

use crate::fixed_point::BoundedInt;

/// The storage dtype of a scale (`#scale-table`). Determines the value range a
/// tensor element at that scale may occupy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    /// 8-bit signed.
    I8,
    /// 16-bit signed.
    I16,
    /// 32-bit signed (accumulators / costs).
    I32,
}

impl Dtype {
    /// The inclusive storage range `[lo, hi]` of this dtype.
    pub const fn range(self) -> (i64, i64) {
        match self {
            Dtype::I8 => (i8::MIN as i64, i8::MAX as i64),
            Dtype::I16 => (i16::MIN as i64, i16::MAX as i64),
            Dtype::I32 => (i32::MIN as i64, i32::MAX as i64),
        }
    }

    /// Immutable canonical-serialization discriminant (RFC-0014 §1).
    pub const fn discriminant(self) -> u8 {
        match self {
            Dtype::I8 => 0,
            Dtype::I16 => 1,
            Dtype::I32 => 2,
        }
    }

    /// Inverse of [`Dtype::discriminant`]; `None` for an unknown value.
    pub const fn from_discriminant(value: u8) -> Option<Self> {
        match value {
            0 => Some(Dtype::I8),
            1 => Some(Dtype::I16),
            2 => Some(Dtype::I32),
            _ => None,
        }
    }
}

/// One entry of the manifest scale table: a `scale_id`, its power-of-two exponent
/// `log2` (value = `q * 2^log2`), and its storage `dtype`. The authoritative
/// table lives in the manifest (#36); this is the shared type both sides bind to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scale {
    /// Index used by tensors and ops to reference this scale.
    pub scale_id: u32,
    /// Power-of-two exponent: a value `q` denotes `q * 2^log2`.
    pub log2: i32,
    /// Storage dtype.
    pub dtype: Dtype,
}

/// A dense row-major tensor of bounded integers with a per-tensor scale id. The
/// fields are private so the `len(data) == product(shape)` invariant cannot be
/// violated after construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor {
    tensor_id: u32,
    shape: Vec<u32>,
    scale_id: u32,
    data: Vec<BoundedInt>,
}

/// The maximum number of tensor dimensions addressable in V0 (the
/// `TensorCell.index: [u32; 4]`); more must be reshaped by the exporter.
pub const MAX_DIMS: usize = 4;

/// Error constructing or validating a [`Tensor`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorError {
    /// `len(data) != product(shape)` (INV-DM-01).
    ShapeDataMismatch {
        /// `product(shape)`.
        product: u64,
        /// `data.len()`.
        data_len: u64,
    },
    /// `len(shape) > MAX_DIMS` — not representable in V0 (RFC-0004).
    TooManyDims {
        /// The offending dimension count.
        dims: usize,
    },
    /// `shape` product overflowed `u64` — a pathologically large tensor.
    ShapeOverflow,
    /// `scale_id` is not present in the scale table (INV-DM-11).
    UndeclaredScale {
        /// The undeclared scale id.
        scale_id: u32,
    },
    /// An element's declared bound exceeds the scale dtype's range (INV-DM-05).
    ElementExceedsDtype {
        /// Index of the offending element.
        index: usize,
        /// Element lower bound.
        lo: i64,
        /// Element upper bound.
        hi: i64,
        /// The scale dtype that was exceeded.
        dtype: Dtype,
    },
}

/// `product(shape)`, or `None` on `u64` overflow. The empty shape (a scalar) has
/// product 1.
fn shape_product(shape: &[u32]) -> Option<u64> {
    let mut product: u64 = 1;
    for &dim in shape {
        product = product.checked_mul(dim as u64)?;
    }
    Some(product)
}

impl Tensor {
    /// Construct a tensor, enforcing `len(data) == product(shape)` (INV-DM-01)
    /// and `len(shape) <= MAX_DIMS`. The scale invariants are checked separately
    /// by [`Tensor::validate_scale`] once a scale table is available.
    pub fn new(
        tensor_id: u32,
        shape: Vec<u32>,
        scale_id: u32,
        data: Vec<BoundedInt>,
    ) -> Result<Self, TensorError> {
        if shape.len() > MAX_DIMS {
            return Err(TensorError::TooManyDims { dims: shape.len() });
        }
        let product = shape_product(&shape).ok_or(TensorError::ShapeOverflow)?;
        if product != data.len() as u64 {
            return Err(TensorError::ShapeDataMismatch {
                product,
                data_len: data.len() as u64,
            });
        }
        Ok(Self {
            tensor_id,
            shape,
            scale_id,
            data,
        })
    }

    /// The stable tensor identifier.
    pub fn tensor_id(&self) -> u32 {
        self.tensor_id
    }

    /// The row-major shape.
    pub fn shape(&self) -> &[u32] {
        &self.shape
    }

    /// The scale id (index into the manifest scale table).
    pub fn scale_id(&self) -> u32 {
        self.scale_id
    }

    /// The dense row-major payload.
    pub fn data(&self) -> &[BoundedInt] {
        &self.data
    }

    /// The element count (`== product(shape)`).
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True iff the tensor has no elements (only possible with a zero dimension).
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Validate the tensor against a scale table: the `scale_id` must be declared
    /// (INV-DM-11) and every element's bound must fit the scale's dtype range
    /// (INV-DM-05). Returns the resolved [`Scale`] on success.
    pub fn validate_scale<'a>(&self, scales: &'a [Scale]) -> Result<&'a Scale, TensorError> {
        let scale = scales.iter().find(|s| s.scale_id == self.scale_id).ok_or(
            TensorError::UndeclaredScale {
                scale_id: self.scale_id,
            },
        )?;
        let (lo, hi) = scale.dtype.range();
        for (index, element) in self.data.iter().enumerate() {
            if element.lo() < lo || element.hi() > hi {
                return Err(TensorError::ElementExceedsDtype {
                    index,
                    lo: element.lo(),
                    hi: element.hi(),
                    dtype: scale.dtype,
                });
            }
        }
        Ok(scale)
    }
}
