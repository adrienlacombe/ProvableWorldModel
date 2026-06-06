// SPDX-License-Identifier: Apache-2.0
//! The committed model manifest and its commitments (specs.md §4; backlog E-205).
//!
//! A [`Manifest`] is the witness-independent description of a committed model: its
//! op graph, weights, activation tables, scale table, and version fields. It
//! produces the `model_commitment` and `quantization_commitment` that the prover
//! binds into the public input, and serializes to **byte-identical** canonical
//! bytes (re-exporting the same manifest yields the same bytes and commitments —
//! the reproducibility contract). Building a manifest needs no checkpoint; the
//! exporter constructs it from the quantized graph + weights.

use pwm_core::commit::{weights_root, ModelBinding, QuantBinding};
use pwm_core::fixed_point::{OverflowPolicy, Rounding};
use pwm_core::graph::GraphSpec;
use pwm_core::serialize::{canonical_bytes, CanonicalEncode};
use pwm_core::tables::{activation_tables_commitment, ActivationTable};
use pwm_core::tensor::{Scale, Tensor};

/// The committed model manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    /// The static op graph.
    pub graph: GraphSpec,
    /// The committed weight and bias tensors.
    pub weights: Vec<Tensor>,
    /// The committed activation tables.
    pub tables: Vec<ActivationTable>,
    /// The scale table.
    pub scales: Vec<Scale>,
    /// The relation semantic version.
    pub relation_version: u32,
    /// The canonical serialization version.
    pub serialization_version: u32,
}

impl Manifest {
    /// The `model_commitment` (architecture + weights + versions).
    pub fn model_commitment(&self) -> [u8; 32] {
        ModelBinding {
            architecture_commitment: self.graph.commitment(),
            weights_root: weights_root(&self.weights),
            relation_version: self.relation_version,
            serialization_version: self.serialization_version,
        }
        .commitment()
    }

    /// The `quantization_commitment` (rounding, overflow policy, scales, tables).
    /// V0 uses nearest-ties-to-even rounding and reject-on-overflow.
    pub fn quantization_commitment(&self) -> [u8; 32] {
        QuantBinding {
            default_rounding: Rounding::NearestTiesToEven,
            overflow_policy: OverflowPolicy::Reject,
            scales: self.scales.clone(),
            activation_tables_commitment: activation_tables_commitment(&self.tables),
        }
        .commitment()
    }

    /// The byte-identical canonical serialization of the manifest.
    pub fn to_bytes(&self) -> Vec<u8> {
        canonical_bytes(self)
    }
}

impl CanonicalEncode for Manifest {
    fn encode(&self, out: &mut Vec<u8>) {
        self.graph.encode(out);
        self.weights.encode(out);
        self.tables.encode(out);
        self.scales.encode(out);
        self.relation_version.encode(out);
        self.serialization_version.encode(out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pwm_core::fixed_point::BoundedInt;
    use pwm_core::graph::OpSpec;
    use pwm_core::tensor::Dtype;

    fn sample() -> Manifest {
        let weight = Tensor::new(
            10,
            vec![2, 2],
            0,
            vec![
                BoundedInt::new(1, -128, 127).unwrap(),
                BoundedInt::new(2, -128, 127).unwrap(),
                BoundedInt::new(3, -128, 127).unwrap(),
                BoundedInt::new(4, -128, 127).unwrap(),
            ],
        )
        .unwrap();
        Manifest {
            graph: GraphSpec {
                ops: vec![OpSpec::Linear {
                    op_id: 1,
                    weight_id: 10,
                    bias_id: None,
                    rows: 2,
                    cols: 2,
                }],
            },
            weights: vec![weight],
            tables: vec![ActivationTable {
                table_id: 0,
                lo: -4,
                outputs: vec![0, 0, 0, 1, 2, 3, 4, 5, 6],
            }],
            scales: vec![Scale {
                scale_id: 0,
                log2: 0,
                dtype: Dtype::I8,
            }],
            relation_version: 1,
            serialization_version: 1,
        }
    }

    #[test]
    fn re_export_is_byte_identical() {
        let m = sample();
        assert_eq!(m.to_bytes(), m.clone().to_bytes());
        assert_eq!(m.model_commitment(), sample().model_commitment());
        assert_eq!(
            m.quantization_commitment(),
            sample().quantization_commitment()
        );
    }

    #[test]
    fn mutating_any_bound_field_changes_a_commitment() {
        let m = sample();
        let mut m2 = sample();
        // Change a weight value -> model commitment changes.
        m2.weights[0] = Tensor::new(
            10,
            vec![2, 2],
            0,
            vec![
                BoundedInt::new(9, -128, 127).unwrap(),
                BoundedInt::new(2, -128, 127).unwrap(),
                BoundedInt::new(3, -128, 127).unwrap(),
                BoundedInt::new(4, -128, 127).unwrap(),
            ],
        )
        .unwrap();
        assert_ne!(m.model_commitment(), m2.model_commitment());
        // Change a table -> quantization commitment changes.
        let mut m3 = sample();
        m3.tables[0].outputs[0] = 99;
        assert_ne!(m.quantization_commitment(), m3.quantization_commitment());
    }
}
