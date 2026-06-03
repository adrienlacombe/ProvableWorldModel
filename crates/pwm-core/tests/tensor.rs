// SPDX-License-Identifier: Apache-2.0
//! Tests for `Tensor` shape/scale invariants (INV-DM-01, INV-DM-05, INV-DM-11).

use pwm_core::fixed_point::BoundedInt;
use pwm_core::tensor::{Dtype, Scale, Tensor, TensorError, MAX_DIMS};

fn cells(values: &[i64], lo: i64, hi: i64) -> Vec<BoundedInt> {
    values
        .iter()
        .map(|&v| BoundedInt::new(v, lo, hi).unwrap())
        .collect()
}

#[test]
fn constructs_with_matching_shape_and_data() {
    let t = Tensor::new(0, vec![2, 3], 1, cells(&[1, 2, 3, 4, 5, 6], -8, 8)).unwrap();
    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(t.len(), 6);
    assert_eq!(t.scale_id(), 1);
    assert!(!t.is_empty());
}

#[test]
fn scalar_tensor_has_product_one() {
    // Empty shape (scalar): product == 1, so exactly one element.
    let t = Tensor::new(7, vec![], 0, cells(&[42], 0, 100)).unwrap();
    assert_eq!(t.len(), 1);
    assert!(Tensor::new(7, vec![], 0, cells(&[1, 2], 0, 100)).is_err());
}

#[test]
fn rejects_shape_data_mismatch() {
    // The acceptance negative test: product(shape) != len(data).
    let err = Tensor::new(0, vec![2, 3], 1, cells(&[1, 2, 3], -8, 8)).unwrap_err();
    assert_eq!(
        err,
        TensorError::ShapeDataMismatch {
            product: 6,
            data_len: 3
        }
    );
}

#[test]
fn rejects_more_than_four_dims() {
    let data = cells(&[1; 1], 0, 1);
    let err = Tensor::new(0, vec![1, 1, 1, 1, 1], 0, data).unwrap_err();
    assert_eq!(err, TensorError::TooManyDims { dims: 5 });
    // exactly MAX_DIMS is allowed
    assert!(Tensor::new(0, vec![1, 1, 1, 1], 0, cells(&[9], 0, 9)).is_ok());
    assert_eq!(MAX_DIMS, 4);
}

#[test]
fn validate_scale_accepts_declared_and_fitting() {
    let scales = [
        Scale {
            scale_id: 0,
            log2: 0,
            dtype: Dtype::I8,
        },
        Scale {
            scale_id: 3,
            log2: -16,
            dtype: Dtype::I32,
        },
    ];
    // int8 element bounds within [-128, 127]
    let t = Tensor::new(0, vec![2], 0, cells(&[10, -20], -127, 127)).unwrap();
    let s = t.validate_scale(&scales).unwrap();
    assert_eq!(s.dtype, Dtype::I8);
}

#[test]
fn validate_scale_rejects_undeclared_scale() {
    let scales = [Scale {
        scale_id: 0,
        log2: 0,
        dtype: Dtype::I8,
    }];
    let t = Tensor::new(0, vec![1], 5, cells(&[1], 0, 1)).unwrap();
    assert_eq!(
        t.validate_scale(&scales),
        Err(TensorError::UndeclaredScale { scale_id: 5 })
    );
}

#[test]
fn validate_scale_rejects_element_exceeding_dtype() {
    let scales = [Scale {
        scale_id: 0,
        log2: 0,
        dtype: Dtype::I8,
    }];
    // bound [0, 200] exceeds i8 max (127) -> INV-DM-05.
    let t = Tensor::new(0, vec![1], 0, cells(&[100], 0, 200)).unwrap();
    let err = t.validate_scale(&scales).unwrap_err();
    assert!(matches!(
        err,
        TensorError::ElementExceedsDtype {
            index: 0,
            hi: 200,
            dtype: Dtype::I8,
            ..
        }
    ));
}

#[test]
fn dtype_ranges() {
    assert_eq!(Dtype::I8.range(), (-128, 127));
    assert_eq!(Dtype::I16.range(), (-32768, 32767));
    assert_eq!(Dtype::I32.range(), (-2147483648, 2147483647));
}
