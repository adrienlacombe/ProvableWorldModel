// SPDX-License-Identifier: Apache-2.0
//! The integer reference inference and trace builder (specs.md §3, §5; backlog
//! E-206, PR-301).
//!
//! A [`Model`] is a sequence of quantized layers, each `Linear(W·x + bias) →
//! Requant → optional Activation(table)`, evaluated in exact integer fixed-point
//! using `pwm_core::fixed_point`. [`Model::run`] produces the ordered execution
//! [`OpRecord`] trace, the static [`GraphSpec`], and the final output — the exact
//! values the prover commits and the verifier audits. There is no floating point
//! and no PyTorch dependency on this path.

use pwm_core::fixed_point::{requantize, Rounding};
use pwm_core::graph::{GraphSpec, OpSpec};
use pwm_core::predictor::layernorm;
use pwm_core::tables::ActivationTable;
use pwm_core::tensor::{Scale, Tensor};
use pwm_core::trace::{ActivationRec, LayerNormRec, LinearRec, OpRecord, RequantRec};

/// An affine-free LayerNorm applied at the start of a layer (pre-norm).
#[derive(Debug, Clone)]
pub struct LayerNormSpec {
    /// Op id.
    pub op_id: u32,
    /// Committed inverse-sqrt table id.
    pub table_id: u32,
    /// Right-shift amount for the `(x − mean)·inv_std` requant.
    pub shift: u32,
    /// Clamp lower bound.
    pub clamp_lo: i64,
    /// Clamp upper bound.
    pub clamp_hi: i64,
    /// Rounding mode.
    pub rounding: Rounding,
}

/// One quantized layer: a biased linear, a requantization, and an optional
/// committed-table activation.
#[derive(Debug, Clone)]
pub struct LayerSpec {
    /// Optional affine-free LayerNorm applied before the linear (pre-norm).
    pub pre_layernorm: Option<LayerNormSpec>,
    /// Op id of the linear.
    pub linear_op_id: u32,
    /// Weight matrix `W`, shape `[rows, cols]`, int8 values.
    pub weight: Tensor,
    /// Optional bias vector, shape `[rows]`, int32 values.
    pub bias: Option<Tensor>,
    /// Op id of the requantization.
    pub requant_op_id: u32,
    /// Right-shift amount.
    pub shift: u32,
    /// Zero point.
    pub zero_point: i64,
    /// Clamp lower bound.
    pub clamp_lo: i64,
    /// Clamp upper bound.
    pub clamp_hi: i64,
    /// Rounding mode.
    pub rounding: Rounding,
    /// Optional activation: `(op_id, table_id)`.
    pub activation: Option<(u32, u32)>,
}

/// A quantized feed-forward model.
#[derive(Debug, Clone)]
pub struct Model {
    /// The ordered layers.
    pub layers: Vec<LayerSpec>,
    /// Committed activation tables referenced by the layers.
    pub tables: Vec<ActivationTable>,
    /// The scale table.
    pub scales: Vec<Scale>,
}

/// The result of running the reference: the trace, the static graph, and output.
#[derive(Debug, Clone)]
pub struct RunOutput {
    /// The ordered execution trace.
    pub trace: Vec<OpRecord>,
    /// The static op graph (committed by `architecture_commitment`).
    pub graph: GraphSpec,
    /// The final output activation.
    pub output: Vec<i64>,
}

/// Reference-inference failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReferenceError {
    /// A weight tensor was not rank-2.
    BadWeightShape {
        /// Op id of the offending linear.
        op_id: u32,
    },
    /// The running activation length did not match a layer's input dimension.
    ShapeMismatch {
        /// Op id of the offending op.
        op_id: u32,
        /// Expected input length.
        expected: usize,
        /// Actual input length.
        got: usize,
    },
    /// A layer referenced an activation table that is not in the model.
    MissingTable {
        /// The missing table id.
        table_id: u32,
    },
    /// An activation input fell outside the committed table domain.
    TableDomain {
        /// The table id.
        table_id: u32,
        /// The out-of-domain input.
        x: i64,
    },
    /// A LayerNorm variance fell outside the committed inverse-sqrt table domain.
    LayerNormDomain {
        /// The op id.
        op_id: u32,
    },
}

impl Model {
    /// All committed weight and bias tensors, in layer order (weights then biases).
    pub fn weight_tensors(&self) -> Vec<Tensor> {
        let mut out = Vec::new();
        for l in &self.layers {
            out.push(l.weight.clone());
            if let Some(b) = &l.bias {
                out.push(b.clone());
            }
        }
        out
    }

    /// Run exact integer inference over `input`, recording the trace and graph.
    pub fn run(&self, input: &[i64]) -> Result<RunOutput, ReferenceError> {
        let mut current: Vec<i64> = input.to_vec();
        let mut ops: Vec<OpSpec> = Vec::new();
        let mut trace: Vec<OpRecord> = Vec::new();

        for layer in &self.layers {
            // --- Optional pre-LayerNorm (affine-free) ---
            if let Some(ln) = &layer.pre_layernorm {
                let table = self
                    .tables
                    .iter()
                    .find(|t| t.table_id == ln.table_id)
                    .ok_or(ReferenceError::MissingTable {
                        table_id: ln.table_id,
                    })?;
                let ln_in = current.clone();
                let ln_out = layernorm(
                    &ln_in,
                    table,
                    ln.shift,
                    ln.clamp_lo,
                    ln.clamp_hi,
                    ln.rounding,
                )
                .ok_or(ReferenceError::LayerNormDomain { op_id: ln.op_id })?;
                ops.push(OpSpec::LayerNorm {
                    op_id: ln.op_id,
                    table_id: ln.table_id,
                    shift: ln.shift,
                    clamp_lo: ln.clamp_lo,
                    clamp_hi: ln.clamp_hi,
                    rounding: ln.rounding.discriminant(),
                });
                trace.push(OpRecord::LayerNorm(LayerNormRec {
                    op_id: ln.op_id,
                    table_id: ln.table_id,
                    input: ln_in,
                    output: ln_out.clone(),
                    shift: ln.shift,
                    clamp_lo: ln.clamp_lo,
                    clamp_hi: ln.clamp_hi,
                    rounding: ln.rounding.discriminant(),
                }));
                current = ln_out;
            }

            // --- Linear: out = W·x + bias ---
            let shape = layer.weight.shape();
            if shape.len() != 2 {
                return Err(ReferenceError::BadWeightShape {
                    op_id: layer.linear_op_id,
                });
            }
            let rows = shape[0] as usize;
            let cols = shape[1] as usize;
            if current.len() != cols {
                return Err(ReferenceError::ShapeMismatch {
                    op_id: layer.linear_op_id,
                    expected: cols,
                    got: current.len(),
                });
            }
            let w = layer.weight.data();
            let bias: Vec<i64> = match &layer.bias {
                Some(b) => b.data().iter().map(|c| c.value()).collect(),
                None => vec![0i64; rows],
            };
            let out_acc = pwm_core::predictor::linear(w, &current, &bias, rows, cols);
            ops.push(OpSpec::Linear {
                op_id: layer.linear_op_id,
                weight_id: layer.weight.tensor_id(),
                bias_id: layer.bias.as_ref().map(|b| b.tensor_id()),
                rows: rows as u32,
                cols: cols as u32,
            });
            trace.push(OpRecord::Linear(LinearRec {
                op_id: layer.linear_op_id,
                weight_id: layer.weight.tensor_id(),
                bias_id: layer.bias.as_ref().map(|b| b.tensor_id()),
                input: current.clone(),
                output: out_acc.clone(),
            }));
            current = out_acc;

            // --- Requant ---
            let rq_in = current.clone();
            let rq_out: Vec<i64> = rq_in
                .iter()
                .map(|&n| {
                    requantize(
                        n,
                        layer.shift,
                        layer.zero_point,
                        layer.clamp_lo,
                        layer.clamp_hi,
                        layer.rounding,
                    )
                })
                .collect();
            ops.push(OpSpec::Requant {
                op_id: layer.requant_op_id,
                shift: layer.shift,
                zero_point: layer.zero_point,
                clamp_lo: layer.clamp_lo,
                clamp_hi: layer.clamp_hi,
                rounding: layer.rounding.discriminant(),
            });
            trace.push(OpRecord::Requant(RequantRec {
                op_id: layer.requant_op_id,
                input: rq_in,
                output: rq_out.clone(),
                shift: layer.shift,
                zero_point: layer.zero_point,
                clamp_lo: layer.clamp_lo,
                clamp_hi: layer.clamp_hi,
                rounding: layer.rounding.discriminant(),
            }));
            current = rq_out;

            // --- Activation (committed table) ---
            if let Some((op_id, table_id)) = layer.activation {
                let table = self
                    .tables
                    .iter()
                    .find(|t| t.table_id == table_id)
                    .ok_or(ReferenceError::MissingTable { table_id })?;
                let act_in = current.clone();
                let mut act_out = Vec::with_capacity(act_in.len());
                for &x in &act_in {
                    let y = table
                        .eval(x)
                        .ok_or(ReferenceError::TableDomain { table_id, x })?;
                    act_out.push(y);
                }
                ops.push(OpSpec::Activation { op_id, table_id });
                trace.push(OpRecord::Activation(ActivationRec {
                    op_id,
                    table_id,
                    input: act_in,
                    output: act_out.clone(),
                }));
                current = act_out;
            }
        }

        Ok(RunOutput {
            trace,
            graph: GraphSpec { ops },
            output: current,
        })
    }
}
