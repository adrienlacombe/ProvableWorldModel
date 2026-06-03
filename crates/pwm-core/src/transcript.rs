// SPDX-License-Identifier: Apache-2.0
//! Hashing primitive and the public-input digest (RFC-0014 §2).
//!
//! V0 fixes one hash primitive for digests and commitments: Blake2s with 256-bit
//! output. The **public-input digest** `pid` is a deterministic function of the
//! public input alone, domain-separated so it cannot collide with other hashed
//! structures. The verifier recomputes `pid` and the AIR binds it, so a proof is
//! valid only for its exact public input (and `relation_id`, which is the first
//! field hashed). The Fiat-Shamir channel built on this primitive is #34.

use blake2::{Blake2s256, Digest};
use stwo::core::channel::{Blake2sChannel, Channel};

use crate::public_input::PublicInput;
use crate::serialize::canonical_bytes;

/// Domain-separation tag for the public-input digest (RFC-0014 §2), 16 bytes.
pub const PID_DOMAIN_TAG: [u8; 16] = *b"pwm.pid.v1\0\0\0\0\0\0";

/// Blake2s-256 of `bytes`, the V0 commitment/digest primitive.
pub fn blake2s256(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Blake2s256::new();
    hasher.update(bytes);
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

/// The public-input digest: `blake2s256(PID_DOMAIN_TAG || canonical_bytes(pi))`
/// (RFC-0014 §2). Deterministic in the public input alone; changing **any**
/// public field changes the digest, because the field's canonical bytes are part
/// of the preimage.
pub fn public_input_digest(pi: &PublicInput) -> [u8; 32] {
    let mut hasher = Blake2s256::new();
    hasher.update(PID_DOMAIN_TAG);
    hasher.update(canonical_bytes(pi));
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

/// Transcript protocol version (T1). Bumping it invalidates old proofs by
/// changing every derived challenge (RFC-0014 §4).
pub const TRANSCRIPT_PROTOCOL_VERSION: u64 = 1;

/// The V0/P2 component schedule (RFC-0014 §4.1): `component_log_sizes` (T3) and
/// the LogUp challenge draws are ordered by this fixed topological order. The
/// authoritative table is frozen in `pwm-air`; it is reproduced here as the
/// documented ordering the transcript depends on. Any reorder mints a new
/// `relation_id`.
pub const COMPONENT_SCHEDULE: &[&str] = &[
    "range_check",
    "tensor_memory",
    "linear",
    "matmul",
    "requant",
    "activation_lookup",
    "layernorm",
    "attention",
    "mlp",
    "predictor",
    "rollout",
    "cost",
    "argmin",
];

/// Split a 32-byte digest into eight little-endian `u32` words for channel mixing.
fn digest_as_u32s(digest: &[u8; 32]) -> [u32; 8] {
    let mut out = [0u32; 8];
    for (i, word) in out.iter_mut().enumerate() {
        *word = u32::from_le_bytes([
            digest[4 * i],
            digest[4 * i + 1],
            digest[4 * i + 2],
            digest[4 * i + 3],
        ]);
    }
    out
}

/// Initialize the Fiat-Shamir channel with the ProvableWorldModel transcript
/// prefix T0–T3 (RFC-0014 §4), then hand the channel to the Stwo commitment
/// scheme (which performs T4–T15). **This is the single shared entry point**:
/// prover (`pwm-prover`) and verifier (`pwm-verifier`) both call it, so the
/// absorb ordering cannot diverge (INV-ARCH-06).
///
/// The ordering is the soundness boundary — every value the prover could choose
/// adaptively is absorbed before the challenge that depends on it:
///
/// - **T0** `Blake2sChannel::default()` — fixed initial state.
/// - **T1** `mix_u64(TRANSCRIPT_PROTOCOL_VERSION)`.
/// - **T2** mix the public-input digest (`public_input_digest`), as eight LE
///   `u32` words, so `relation_id` and every public field bind before any
///   prover-chosen commitment.
/// - **T3** mix `component_log_sizes`, ordered by [`COMPONENT_SCHEDULE`].
///
/// `component_log_sizes` must already be ordered per the schedule by the caller.
///
/// Note on the trust boundary: `pwm-core` uses the vendored Stwo `Blake2sChannel`
/// (the no_std verifier-side channel type) solely to host this one shared
/// transcript, as the architecture's module table and RFC-0014 §4 intend — there
/// is exactly one transcript implementation and both sides call it.
pub fn init_channel(public_input: &PublicInput, component_log_sizes: &[u32]) -> Blake2sChannel {
    let mut channel = Blake2sChannel::default(); // T0
    channel.mix_u64(TRANSCRIPT_PROTOCOL_VERSION); // T1
    channel.mix_u32s(&digest_as_u32s(&public_input_digest(public_input))); // T2
    channel.mix_u32s(component_log_sizes); // T3
    channel
}
