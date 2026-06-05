// SPDX-License-Identifier: Apache-2.0
//! Execution-trace data model (specs.md §5).
//!
//! The trace is an **ordered list of op records**, one per exported op, that the
//! prover fills from the integer reference inference and the verifier audits.
//! Leaves of the `trace_root` Merkle tree are the canonical bytes of these
//! records (or per-tensor cell groups, to allow selective opening).
//!
//! This module defines the stable identifiers and the op taxonomy. The concrete
//! per-variant record structs and their `CanonicalEncode` impls land with the
//! trace builder (backlog C-103); they are intentionally thin data, free of any
//! proving substrate, so both `pwm-prover` and `pwm-verifier` share one model.

/// Stable identifier of an exported op (e.g. `predictor.block0.attn.qkv`),
/// assigned by the exporter and bound by `model_commitment`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OpId(pub u32);

/// Stable identifier of a committed lookup table (GELU / SiLU / softmax-exp /
/// inverse-sqrt), bound by `quantization_commitment`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TableId(pub u32);

/// The elementwise op kind carried by an [`OpKind::Elementwise`] record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EwKind {
    /// Residual / tensor add.
    Add,
    /// AdaLN modulation `t * (1 + scale) + shift`.
    Modulate,
    /// Per-channel gate `t * gate`.
    Gate,
}

/// The op taxonomy of a trace record. Each variant names how the verifier audits
/// that op (specs.md §6–§10):
///
/// - [`OpKind::Linear`] — fixed-weight matmul, verified by **Freivalds** (§7).
/// - [`OpKind::AttnScore`] / [`OpKind::AttnApply`] — data-dependent attention
///   inner products, **exactly recomputed** (§8.2).
/// - all others — **exactly recomputed** integer ops / committed-table reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpKind {
    /// `z = W·x` (i32 accumulators kept un-requantized for Freivalds).
    Linear,
    /// Arithmetic-right-shift rescale with quotient/remainder witnesses.
    Requant,
    /// Committed lookup-table read (GELU / SiLU).
    Activation,
    /// Affine-free LayerNorm: integer mean/variance + inverse-sqrt table.
    LayerNorm,
    /// `score = QKᵀ` per head (recomputed exactly).
    AttnScore,
    /// Row softmax via the committed exp table + reciprocal witness.
    Softmax,
    /// `out = prob · V` per head (recomputed exactly).
    AttnApply,
    /// Add / modulate / gate (see [`EwKind`]).
    Elementwise,
    /// One autoregressive rollout step (windowing + predictor output).
    RolloutStep,
    /// Goal-latent MSE for one candidate.
    Cost,
    /// Selected-cost ≤ every candidate cost + smallest-index tie-break.
    Argmin,
}

impl OpKind {
    /// Whether this op is audited by Freivalds (true) or exact recompute (false).
    pub const fn is_freivalds(self) -> bool {
        matches!(self, OpKind::Linear)
    }
}
