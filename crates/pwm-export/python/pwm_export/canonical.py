# SPDX-License-Identifier: Apache-2.0
"""Canonical serialization + commitments, byte-identical to pwm-core (RFC-0014 /
specs.md §4). This is the Python side of the export pipeline (E-205): it must
reproduce the Rust `CanonicalEncode` / commitment bytes exactly so a Python-
exported manifest yields the same `model_commitment` / `quantization_commitment`
the Rust prover and verifier compute. No PyTorch dependency — pure encoding.
"""
import hashlib
import struct

# --- primitive encoders (little-endian, fixed width; matches pwm-core::serialize) ---


def u32(x: int) -> bytes:
    return struct.pack("<I", x & 0xFFFFFFFF)


def i32(x: int) -> bytes:
    return struct.pack("<i", x)


def u64(x: int) -> bytes:
    return struct.pack("<Q", x)


def i64(x: int) -> bytes:
    return struct.pack("<q", x)


def vec(items: list[bytes]) -> bytes:
    return u32(len(items)) + b"".join(items)


def blake2s256(b: bytes) -> bytes:
    return hashlib.blake2s(b, digest_size=32).digest()


# --- domain tags (16 bytes each, matching pwm-core) ---
TAG_MODEL = b"pwm.model.v1\0\0\0\0"
TAG_QUANT = b"pwm.quant.v1\0\0\0\0"
TAG_GRAPH = b"pwm.graph.v1\0\0\0\0"
TAG_TABLE = b"pwm.table.v1\0\0\0\0"
TAG_WLEAF = b"pwm.wleaf.v1\0\0\0\0"
TAG_WNODE = b"pwm.wnode.v1\0\0\0\0"

# Dtype discriminants: I8=0, I16=1, I32=2.
DTYPE = {"i8": 0, "i16": 1, "i32": 2}
# Rounding: NearestTiesToEven=0, TruncateTowardZero=1. OverflowPolicy: Reject=0.


def commit(tag16: bytes, payload: bytes) -> bytes:
    """`blake2s256(tag || u64_le(len) || payload)` (pwm-core::commit::commit)."""
    return blake2s256(tag16 + u64(len(payload)) + payload)


# --- composite types (dicts; field order is the canonical schedule) ---


def enc_bounded(cell: tuple[int, int, int]) -> bytes:
    """BoundedInt = value || lo || hi (three i64)."""
    v, lo, hi = cell
    return i64(v) + i64(lo) + i64(hi)


def enc_tensor(t: dict) -> bytes:
    out = u32(t["tensor_id"]) + u32(t["scale_id"])
    out += u32(len(t["shape"])) + b"".join(u32(d) for d in t["shape"])
    out += u32(len(t["data"])) + b"".join(enc_bounded(c) for c in t["data"])
    return out


def enc_scale(s: dict) -> bytes:
    return u32(s["scale_id"]) + i32(s["log2"]) + bytes([DTYPE[s["dtype"]]])


def enc_table(tbl: dict) -> bytes:
    return u32(tbl["table_id"]) + i64(tbl["lo"]) + vec([i64(x) for x in tbl["outputs"]])


def table_commitment(tbl: dict) -> bytes:
    return commit(TAG_TABLE, enc_table(tbl))


def activation_tables_commitment(tables: list[dict]) -> bytes:
    srt = sorted(tables, key=lambda t: t["table_id"])
    return commit(TAG_TABLE, vec([enc_table(t) for t in srt]))


def enc_opspec(op: dict) -> bytes:
    k = op["kind"]
    if k == "linear":
        bias = b"\x00" if op["bias_id"] is None else b"\x01" + u32(op["bias_id"])
        return (
            bytes([0])
            + u32(op["op_id"])
            + u32(op["weight_id"])
            + bias
            + u32(op["rows"])
            + u32(op["cols"])
        )
    if k == "requant":
        return (
            bytes([1])
            + u32(op["op_id"])
            + u64(op["shift"])
            + i64(op["zero_point"])
            + i64(op["clamp_lo"])
            + i64(op["clamp_hi"])
            + bytes([op["rounding"]])
        )
    if k == "activation":
        return bytes([2]) + u32(op["op_id"]) + u32(op["table_id"])
    if k == "layernorm":
        return (
            bytes([3])
            + u32(op["op_id"])
            + u32(op["table_id"])
            + u64(op["shift"])
            + i64(op["clamp_lo"])
            + i64(op["clamp_hi"])
            + bytes([op["rounding"]])
        )
    raise ValueError(f"unknown op kind {k!r}")


def enc_graph(ops: list[dict]) -> bytes:
    return vec([enc_opspec(o) for o in ops])


def graph_commitment(ops: list[dict]) -> bytes:
    return commit(TAG_GRAPH, enc_graph(ops))


def weights_root(weights: list[dict]) -> bytes:
    """blake2s Merkle over weight leaves sorted by tensor_id (pwm-core::commit)."""
    leaves = []
    for t in sorted(weights, key=lambda t: t["tensor_id"]):
        leaf = blake2s256(TAG_WLEAF + u32(t["tensor_id"]) + enc_tensor(t))
        leaves.append(leaf)
    if not leaves:
        return blake2s256(TAG_WNODE)
    level = leaves
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            a = level[i]
            b = level[i + 1] if i + 1 < len(level) else level[i]
            nxt.append(blake2s256(TAG_WNODE + a + b))
        level = nxt
    return level[0]


def model_commitment(
    ops: list[dict], weights: list[dict], relation_version: int, serialization_version: int
) -> bytes:
    binding = (
        graph_commitment(ops)
        + weights_root(weights)
        + u32(relation_version)
        + u32(serialization_version)
    )
    return commit(TAG_MODEL, binding)


def quantization_commitment(scales: list[dict], tables: list[dict]) -> bytes:
    # default_rounding=NearestTiesToEven(0), overflow_policy=Reject(0).
    payload = (
        bytes([0])
        + bytes([0])
        + vec([enc_scale(s) for s in scales])
        + activation_tables_commitment(tables)
    )
    return commit(TAG_QUANT, payload)
