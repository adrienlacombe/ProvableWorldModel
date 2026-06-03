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
