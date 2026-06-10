# SPDX-License-Identifier: Apache-2.0
"""Cross-language parity (E-207 bridge): the Python canonical encoding must
reproduce the Rust commitments byte-for-byte. The reference hexes below are the
values pwm-core / pwm-export compute for the fixed manifest in
`crates/pwm-export/src/manifest.rs::tests::sample()` and a fixed activation table.
If the Rust serialization changes, these must be regenerated.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pwm_export import canonical as c  # noqa: E402

# References captured from the Rust implementation (see commit message).
REF_MODEL = "f8be78a5aa2cfa510e46291102f3de0525205c33afc3855420674f21e5c3b930"
REF_QUANT = "32e5f6008d692d5b3f5bce831141b5fdd112b6642524852e743c55a0b4a7b31a"
REF_TABLE = "159172fe49bb7220060e9a20746824faba4987f079ec5e52b1b7e6a075ad27eb"
# Multi-leaf weights_root over predictor-style tensors (ids 1000.., scale_id 0,
# int8 bounds; three leaves exercise the odd-level node duplication). The same
# vector is pinned Rust-side in pwm-core/tests/commit.rs
# (weights_root_multi_leaf_parity_vector_is_pinned), so this gate is two-way.
REF_PRED_WROOT = "c666f637a30f38d653f7c6a3280e87b705777df25bae4644148b534992b03120"

# Mirrors the small generic multi-leaf weight-root parity vector. The full
# predictor-bundle scheme has its own test below because it uses per-tensor
# scale ids.
PRED_WEIGHTS = [
    {"tensor_id": 1000, "scale_id": 0, "shape": [1, 2],
     "data": [(1, -128, 127), (-2, -128, 127)]},
    {"tensor_id": 1001, "scale_id": 0, "shape": [2, 1],
     "data": [(3, -128, 127), (4, -128, 127)]},
    {"tensor_id": 1002, "scale_id": 0, "shape": [2, 2],
     "data": [(-5, -128, 127), (6, -128, 127), (-7, -128, 127), (8, -128, 127)]},
]

# The fixed sample manifest (mirrors manifest.rs::tests::sample()).
SAMPLE_OPS = [
    {"kind": "linear", "op_id": 1, "weight_id": 10, "bias_id": None, "rows": 2, "cols": 2}
]
SAMPLE_WEIGHTS = [
    {
        "tensor_id": 10,
        "scale_id": 0,
        "shape": [2, 2],
        "data": [(1, -128, 127), (2, -128, 127), (3, -128, 127), (4, -128, 127)],
    }
]
SAMPLE_TABLES = [{"table_id": 0, "lo": -4, "outputs": [0, 0, 0, 1, 2, 3, 4, 5, 6]}]
SAMPLE_SCALES = [{"scale_id": 0, "log2": 0, "dtype": "i8"}]


def test_table_commitment_matches_rust():
    tbl = {"table_id": 7, "lo": -2, "outputs": [1, 2, 4]}
    assert c.table_commitment(tbl).hex() == REF_TABLE


def test_model_commitment_matches_rust():
    got = c.model_commitment(SAMPLE_OPS, SAMPLE_WEIGHTS, 1, 1)
    assert got.hex() == REF_MODEL


def test_quantization_commitment_matches_rust():
    got = c.quantization_commitment(SAMPLE_SCALES, SAMPLE_TABLES)
    assert got.hex() == REF_QUANT


def test_predictor_weights_root_matches_rust():
    # The export-computed predictor weights_root carried in the bundle must be
    # the exact value the Rust prover binds into the model commitment.
    assert c.weights_root(PRED_WEIGHTS).hex() == REF_PRED_WROOT
    # Sorted by tensor_id: registration order must not matter.
    assert c.weights_root(list(reversed(PRED_WEIGHTS))).hex() == REF_PRED_WROOT


def test_predictor_weight_scheme_matches_rust():
    # The full bundle weight scheme (ids 1000+5i in registration order
    # adaln,qkv,out,fc1,fc2; scale_id 10+k; shapes from dims) must reproduce
    # the root the Rust builder computes for the same dims and the deterministic
    # (i % 3) - 1 weight pattern. Rust pin: lewm_predictor.rs
    # tests::predictor_weight_scheme_parity_vector_is_pinned.
    from pwm_export.export import predictor_weight_dicts

    d, heads, dim_head, mlp, depth = 8, 2, 4, 16, 2
    inner = heads * dim_head

    def mk(rows, cols):
        return [(i % 3) - 1 for i in range(rows * cols)]

    blocks = [
        {
            "qkv": mk(3 * inner, d),
            "out": mk(d, inner),
            "fc1": mk(mlp, d),
            "fc2": mk(d, mlp),
            "adaln": mk(6 * d, d),
        }
        for _ in range(depth)
    ]
    weights = predictor_weight_dicts(blocks, d, heads, dim_head, mlp)
    assert [w["tensor_id"] for w in weights] == list(range(1000, 1010))
    assert [w["scale_id"] for w in weights] == list(range(10, 20))
    got = c.weights_root(weights).hex()
    assert got == "b935eb9908da38f4f23f6ff1db8922a87b79f2c5686717726ebb30920803a95b"


if __name__ == "__main__":
    # Allow running without pytest.
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
    print("all parity checks passed")
