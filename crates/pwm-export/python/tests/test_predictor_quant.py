# SPDX-License-Identifier: Apache-2.0
"""Predictor-bundle calibration and integer replay without checkpoint assets."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pwm_export import export  # noqa: E402
from pwm_export import predictor_quant as pq  # noqa: E402


def _blocks(dims, seed=0):
    rng = np.random.default_rng(seed)
    blocks = []
    for _ in range(dims.depth):
        blocks.append(
            {
                "qkv": rng.standard_normal((3 * dims.inner, dims.d)) * 0.4,
                "out": rng.standard_normal((dims.d, dims.inner)) * 0.4,
                "fc1": rng.standard_normal((dims.mlp, dims.d)) * 0.4,
                "fc2": rng.standard_normal((dims.d, dims.mlp)) * 0.4,
                "adaln": rng.standard_normal((6 * dims.d, dims.d)) * 0.4,
            }
        )
    return blocks


def test_predictor_bundle_quant_is_self_consistent():
    dims = pq.PredictorDims(d=8, s=3, h=2, dh=4, mlp=16, depth=2)
    rng = np.random.default_rng(1)
    quant, qblocks, xq, cq, tables = pq.bundle_quant(
        dims,
        _blocks(dims),
        rng.standard_normal(dims.s * dims.d),
        rng.standard_normal(dims.s * dims.d),
    )

    assert len(qblocks) == dims.depth
    assert len(xq) == dims.s * dims.d
    assert len(cq) == dims.s * dims.d
    assert len(quant["w_log2"]) == dims.depth
    assert len(quant["z_out_float"]) == dims.s * dims.d
    assert len(quant["z_out_int"]) == dims.s * dims.d
    assert quant["error"] <= quant["tolerance"]

    by_id = {t["table_id"]: t for t in tables}
    assert set(by_id) == {pq.TABLE_SILU, pq.TABLE_GELU, pq.TABLE_INVSQRT, pq.TABLE_EXP}
    assert by_id[pq.TABLE_SILU]["outputs"] != list(range(-128, 128))
    assert by_id[pq.TABLE_GELU]["outputs"] != list(range(-128, 128))
    assert by_id[pq.TABLE_INVSQRT]["outputs"][0] == 1 << quant["t_inv"]
    assert by_id[pq.TABLE_EXP]["outputs"][-1] == 1 << quant["f_e"]

    weights = export.predictor_weight_dicts(qblocks, dims.d, dims.h, dims.dh, dims.mlp)
    assert [w["tensor_id"] for w in weights] == list(range(1000, 1000 + 5 * dims.depth))
    assert [w["scale_id"] for w in weights] == list(range(10, 10 + 5 * dims.depth))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
    print("all predictor-quant checks passed")
