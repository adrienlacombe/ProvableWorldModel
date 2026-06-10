# SPDX-License-Identifier: Apache-2.0
"""Export pipeline on synthetic le-wm-shaped data (E-201/202/203/208).

Exercises the NumPy cores end-to-end with no torch / checkpoint: ingest dim
validation, int8 / pow-2 quantization with MAC bounds, BatchNorm folding
equivalence + determinism, the committed manifest, and the P2 input bundle. Real
assets only change the *source* of the arrays, not this logic.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pwm_export import data_adapter, export, fold, ingest, lewm, quantize  # noqa: E402


def _rng(seed):
    return np.random.default_rng(seed)


# --- E-201: ingest / V0 subgraph extraction ---------------------------------


def _synthetic_v0_arrays(dims, action_dim=4, seed=0):
    g = _rng(seed)
    return {name: g.standard_normal(shape) for name, shape in ingest.v0_param_shapes(dims, action_dim).items()}


def test_ingest_asserts_reference_dims():
    sub = ingest.extract_v0_subgraph(_synthetic_v0_arrays(ingest.V0_DIMS))
    assert sub["dims"].as_tuple() == (192, 3, 6, 16, 64, 2048)
    assert sub["dims"].attn_inner == 1024
    # all 6 predictor blocks present with attention + ffn weights
    assert sum(1 for k in sub["params"] if k.endswith(".attn.wq")) == 6
    assert sub["params"]["predictor.blocks.0.ffn.fc1"].shape == (2048, 192)
    assert sub["params"]["predictor.blocks.0.attn.wproj"].shape == (192, 1024)


def test_ingest_rejects_wrong_shape():
    arrays = _synthetic_v0_arrays(ingest.V0_DIMS)
    arrays["pred_proj.fc1"] = np.zeros((192, 191))  # off by one
    try:
        ingest.extract_v0_subgraph(arrays)
        raise AssertionError("expected shape mismatch")
    except ValueError as e:
        assert "pred_proj.fc1" in str(e)


def test_ingest_rejects_missing_param():
    arrays = _synthetic_v0_arrays(ingest.V0_DIMS)
    del arrays["predictor.blocks.3.attn.wv"]
    try:
        ingest.extract_v0_subgraph(arrays)
        raise AssertionError("expected missing param")
    except KeyError as e:
        assert "blocks.3.attn.wv" in str(e)


# --- E-202: quantization -----------------------------------------------------


def test_quantize_int8_range_and_pow2_scale():
    w = _rng(1).standard_normal((8, 16)) * 5.0
    q, log2 = quantize.quantize_array(w)
    assert all(-128 <= v <= 127 for v in q)  # int8 storage range
    assert len(q) == 8 * 16
    # dequant error is bounded by half a scale step
    err = np.abs(quantize.dequantize(q, log2).reshape(8, 16) - w)
    assert err.max() <= 2.0**log2 / 2 + 1e-9


def test_quantize_zero_tensor():
    q, log2 = quantize.quantize_array(np.zeros((4, 4)))
    assert set(q) == {0} and log2 == 0


def test_mac_bound_int32():
    # spec §3: 2048-wide int8 MAC fits int32; a huge inner does not.
    assert quantize.mac_bound(2048) == 33_032_192
    assert quantize.mac_fits_int32(2048)
    assert not quantize.mac_fits_int32(200_000)


# --- E-203: BatchNorm folding ------------------------------------------------


def test_fold_equals_linear_then_bn():
    g = _rng(2)
    out_dim, in_dim = 5, 7
    W = g.standard_normal((out_dim, in_dim))
    b = g.standard_normal(out_dim)
    gamma, beta = g.standard_normal(out_dim), g.standard_normal(out_dim)
    rmean = g.standard_normal(out_dim)
    rvar = np.abs(g.standard_normal(out_dim)) + 0.1
    x = g.standard_normal((3, in_dim))

    W2, b2 = fold.fold_linear_bn(W, b, gamma, beta, rmean, rvar)
    folded = fold.linear(x, W2, b2)
    reference = fold.linear_then_bn(x, W, b, gamma, beta, rmean, rvar)
    assert np.allclose(folded, reference, atol=1e-10)


def test_fold_is_byte_deterministic():
    g = _rng(3)
    args = (g.standard_normal((4, 4)), g.standard_normal(4), g.standard_normal(4),
            g.standard_normal(4), g.standard_normal(4), np.abs(g.standard_normal(4)) + 0.1)
    a = fold.fold_linear_bn(*args)
    b = fold.fold_linear_bn(*args)
    assert a[0].tobytes() == b[0].tobytes() and a[1].tobytes() == b[1].tobytes()


# --- E-205 driver: synthetic graph -> committed manifest ---------------------


def test_export_graph_manifest_is_deterministic():
    g = _rng(4)
    named = [
        (1, "fc1", g.standard_normal((2048, 192))),
        (2, "fc2", g.standard_normal((192, 2048))),
    ]
    tables = [{"table_id": 0, "lo": -4, "outputs": [0, 0, 0, 1, 2, 3, 4, 5, 6]}]
    a = export.export_graph(named, tables)
    b = export.export_graph(named, tables)
    assert a["manifest"] == b["manifest"]
    # commitments are 32-byte hex strings
    for k in ("model_commitment", "quantization_commitment", "graph_commitment", "weights_root"):
        assert len(a["manifest"][k]) == 64
    assert len(a["scales"]) == 2 and a["ops"][0]["cols"] == 192


# --- E-208: P2 input bundle --------------------------------------------------


def test_build_p2_bundle_shapes_and_int8():
    g = _rng(5)
    H, L, A, S, T = 3, 192, 4, 8, 5  # history, latent, action_dim, candidates, horizon
    bundle = data_adapter.build_p2_bundle(
        z_history=g.standard_normal((H, L)),
        action=g.standard_normal((H, A)),
        z_next=g.standard_normal(L),
        goal=g.standard_normal(L),
        candidate_actions=g.standard_normal((S, T, A)),
    )
    assert bundle["history_size"] == H and bundle["latent_dim"] == L
    assert bundle["num_candidates"] == S and bundle["horizon"] == T
    assert len(bundle["z_history"]["data"]) == H * L
    assert all(-128 <= v <= 127 for v in bundle["z_history"]["data"])
    assert len(bundle["candidate_actions"]) == S and len(bundle["candidate_actions"][0]) == T * A


# --- E-201: real le-wm key adapter (synthetic params, exercises the mapping) ---


def _synthetic_lewm_params(dims, seed=7):
    g = _rng(seed)
    return {name: g.standard_normal(shape) for name, shape in lewm.v0_linear_shapes(dims).items()}


def test_lewm_adapter_extracts_v0_linears():
    linears = lewm.extract_v0_linears(_synthetic_lewm_params(ingest.V0_DIMS))
    # 2 action-encoder + 6 blocks × 5 + 2 pred_proj = 34 Freivalds linears.
    assert len(linears) == 34
    names = {n for n, _ in linears}
    assert "predictor.transformer.layers.0.attn.to_qkv.weight" in names
    assert "predictor.transformer.layers.5.adaLN_modulation.1.weight" in names
    assert "pred_proj.net.0.weight" in names
    # qkv is fused [3·heads·dim_head, latent] = [3072, 192].
    qkv = dict(linears)["predictor.transformer.layers.0.attn.to_qkv.weight"]
    assert qkv.shape == (3072, 192)


def test_lewm_adapter_rejects_wrong_shape():
    params = _synthetic_lewm_params(ingest.V0_DIMS)
    params["pred_proj.net.3.weight"] = np.zeros((192, 2047))
    try:
        lewm.extract_v0_linears(params)
        raise AssertionError("expected shape mismatch")
    except ValueError as e:
        assert "pred_proj.net.3.weight" in str(e)


if __name__ == "__main__":
    # Allow running without pytest.
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok: {_name}")
    print("all export-pipeline checks passed")
