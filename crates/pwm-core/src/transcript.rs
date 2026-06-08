// SPDX-License-Identifier: Apache-2.0
//! Hashing primitive, the public-input digest, and the native Fiat-Shamir
//! transcript (specs.md §6, §7).
//!
//! V0 fixes one hash primitive for digests, commitments, and the transcript:
//! Blake2s with 256-bit output. The **public-input digest** `pid` is a
//! deterministic function of the public input alone, domain-separated so it
//! cannot collide with other hashed structures.
//!
//! Post-pivot the transcript is a small native Blake2s sponge (no Stwo channel).
//! It is the single shared entry point: prover (`pwm-prover`) and verifier
//! (`pwm-verifier`) absorb the same values in the same order, then squeeze the
//! Freivalds challenge vectors and the audit selection. Because challenges are
//! squeezed **after** the trace commitment is absorbed, the prover commits its
//! claimed accumulators before any challenge is known (non-interactive
//! Fiat-Shamir; specs.md §7).

use alloc::vec::Vec;

use blake2::{Blake2s256, Digest};

use crate::field::Fp61;
use crate::public_input::PublicInput;
use crate::serialize::canonical_bytes;

/// Domain-separation tag for the public-input digest, 16 bytes.
pub const PID_DOMAIN_TAG: [u8; 16] = *b"pwm.pid.v1\0\0\0\0\0\0";

/// Transcript protocol version. Bumping it invalidates old proofs by changing
/// every derived challenge.
pub const TRANSCRIPT_PROTOCOL_VERSION: u64 = 1;

/// Blake2s-256 of `bytes`, the V0 commitment/digest primitive.
pub fn blake2s256(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Blake2s256::new();
    hasher.update(bytes);
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

/// The public-input digest: `blake2s256(PID_DOMAIN_TAG || canonical_bytes(pi))`.
/// Deterministic in the public input alone; changing **any** public field
/// changes the digest.
pub fn public_input_digest(pi: &PublicInput) -> [u8; 32] {
    let mut hasher = Blake2s256::new();
    hasher.update(PID_DOMAIN_TAG);
    hasher.update(canonical_bytes(pi));
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

/// A native Blake2s Fiat-Shamir transcript. Absorb binds values into the running
/// state; squeeze derives challenges from the state. Domain-separated so absorbs,
/// squeezes, and expansion never collide.
#[derive(Debug, Clone)]
pub struct Transcript {
    state: [u8; 32],
}

impl Transcript {
    /// Start a transcript with a domain string.
    pub fn new(domain: &[u8]) -> Self {
        let mut buf = Vec::with_capacity(18 + domain.len());
        buf.extend_from_slice(b"pwm.transcript.v1\0");
        buf.extend_from_slice(domain);
        Transcript {
            state: blake2s256(&buf),
        }
    }

    /// Absorb a labelled byte string: `state = H("absorb" || state || len(label)
    /// || label || len(data) || data)`. Lengths are length-prefixed so distinct
    /// (label, data) pairs never alias.
    pub fn absorb(&mut self, label: &[u8], data: &[u8]) {
        let mut buf = Vec::with_capacity(8 + 32 + 8 + label.len() + 8 + data.len());
        buf.extend_from_slice(b"absorb\0\0");
        buf.extend_from_slice(&self.state);
        buf.extend_from_slice(&(label.len() as u64).to_le_bytes());
        buf.extend_from_slice(label);
        buf.extend_from_slice(&(data.len() as u64).to_le_bytes());
        buf.extend_from_slice(data);
        self.state = blake2s256(&buf);
    }

    /// Absorb a labelled `u64` (little-endian).
    pub fn absorb_u64(&mut self, label: &[u8], value: u64) {
        self.absorb(label, &value.to_le_bytes());
    }

    /// Advance the state binding `label`, then expand `out.len()` bytes from the
    /// advanced state via a counter. Advancing means sequential squeezes differ
    /// and later absorbs depend on earlier draws.
    fn squeeze_into(&mut self, label: &[u8], out: &mut [u8]) {
        self.absorb(b"squeeze", label);
        let seed = self.state;
        let mut pos = 0usize;
        let mut ctr = 0u64;
        while pos < out.len() {
            let mut buf = Vec::with_capacity(8 + 32 + 8);
            buf.extend_from_slice(b"expand\0\0");
            buf.extend_from_slice(&seed);
            buf.extend_from_slice(&ctr.to_le_bytes());
            let block = blake2s256(&buf);
            let take = core::cmp::min(32, out.len() - pos);
            out[pos..pos + take].copy_from_slice(&block[..take]);
            pos += take;
            ctr += 1;
        }
    }

    /// Squeeze `out.len()` challenge bytes under `label`.
    pub fn challenge_bytes(&mut self, label: &[u8], out: &mut [u8]) {
        self.squeeze_into(label, out);
    }

    /// Squeeze a challenge `u64` (e.g. for audit selection).
    pub fn challenge_u64(&mut self, label: &[u8]) -> u64 {
        let mut b = [0u8; 8];
        self.squeeze_into(label, &mut b);
        u64::from_le_bytes(b)
    }

    /// Squeeze a single audit-field challenge.
    pub fn challenge_fp61(&mut self, label: &[u8]) -> Fp61 {
        let mut b = [0u8; 8];
        self.squeeze_into(label, &mut b);
        Fp61::new(u64::from_le_bytes(b))
    }

    /// Squeeze a length-`n` vector of audit-field challenges (a Freivalds `r`).
    pub fn challenge_fp61_vec(&mut self, label: &[u8], n: usize) -> Vec<Fp61> {
        let mut bytes = alloc::vec![0u8; n * 8];
        self.squeeze_into(label, &mut bytes);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut w = [0u8; 8];
            w.copy_from_slice(&bytes[i * 8..i * 8 + 8]);
            out.push(Fp61::new(u64::from_le_bytes(w)));
        }
        out
    }
}

/// The shared transcript prefix both sides bind before any commitment-derived
/// challenge: protocol version, then the public-input digest (which already binds
/// `relation_id` and every public field). The caller then absorbs the model /
/// quantization / planner / trace commitments (in that order) before squeezing
/// Freivalds vectors — see specs.md §6. This is the single point that keeps the
/// prover and verifier absorb order identical (INV-ARCH-06).
pub fn init_transcript(public_input: &PublicInput) -> Transcript {
    let mut t = Transcript::new(b"pwm.fs.v1");
    t.absorb_u64(b"protocol_version", TRANSCRIPT_PROTOCOL_VERSION);
    t.absorb(b"public_input_digest", &public_input_digest(public_input));
    t
}
