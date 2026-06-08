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


if __name__ == "__main__":
    # Allow running without pytest.
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
    print("all parity checks passed")
