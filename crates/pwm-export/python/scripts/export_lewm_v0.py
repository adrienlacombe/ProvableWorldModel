#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export the REAL pretrained le-wm V0 checkpoint into the prover.

Loads the `quentinll/lewm-pusht` checkpoint (a plain state_dict of named tensors,
so no stable-worldmodel install is needed), extracts the V0 proven subgraph
(`action_encoder -> predictor x6 -> pred_proj`), quantizes every linear to int8
with power-of-two scales, folds the `pred_proj` BatchNorm into the preceding
Linear, asserts the spec-2 dims (192/3/6/16/64/2048), and emits:

1. the committed manifest (model / quantization / graph commitments), and
2. a JSON bundle the Rust prover ingests to prove + verify the `pred_proj` head
   (`Linear -> GELU -> Linear`) on a real latent, with the real folded weights.

The full 6-block attention predictor is quantized and committed here; proving it
end to end in the Rust prover is the next integration step (the op kernels exist;
it needs the 16-head 6-block graph wired and its activation scales calibrated).

Usage:
    pip install torch numpy
    # weights.pt + config.json from https://huggingface.co/quentinll/lewm-pusht
    LEWM_WEIGHTS=weights.pt python scripts/export_lewm_v0.py [out.json]
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pwm_export import export, fold, quantize

DIM, HEADS, DIM_HEAD, MLP, DEPTH, HIST = 192, 16, 64, 2048, 6, 3
INNER = HEADS * DIM_HEAD  # 1024


def load_state_dict() -> dict:
    path = os.environ.get("LEWM_WEIGHTS", "weights.pt")
    if not Path(path).exists():
        raise SystemExit(
            f"checkpoint not found at {path}. Download weights.pt from "
            "https://huggingface.co/quentinll/lewm-pusht and set LEWM_WEIGHTS."
        )
    sd = torch.load(path, map_location="cpu", weights_only=True)
    return {k: v.float().numpy() for k, v in sd.items()}


def assert_dims(sd: dict) -> None:
    p = "predictor.transformer.layers.0"
    assert sd[f"{p}.attn.to_qkv.weight"].shape == (3 * INNER, DIM)
    assert sd[f"{p}.attn.to_out.0.weight"].shape == (DIM, INNER)
    assert sd[f"{p}.mlp.net.1.weight"].shape == (MLP, DIM)
    assert sd[f"{p}.adaLN_modulation.1.weight"].shape == (6 * DIM, DIM)
    assert sd["predictor.pos_embedding"].shape == (1, HIST, DIM)
    n_blocks = sum(1 for k in sd if k.endswith(".attn.to_qkv.weight"))
    assert n_blocks == DEPTH, n_blocks


def v0_linears(sd: dict) -> dict:
    out = {}
    for i in range(DEPTH):
        p = f"predictor.transformer.layers.{i}"
        out[f"{p}.attn.to_qkv"] = sd[f"{p}.attn.to_qkv.weight"]
        out[f"{p}.attn.to_out"] = sd[f"{p}.attn.to_out.0.weight"]
        out[f"{p}.mlp.fc1"] = sd[f"{p}.mlp.net.1.weight"]
        out[f"{p}.mlp.fc2"] = sd[f"{p}.mlp.net.4.weight"]
        out[f"{p}.adaln"] = sd[f"{p}.adaLN_modulation.1.weight"]
    out["action_encoder.embed0"] = sd["action_encoder.embed.0.weight"]
    out["action_encoder.embed2"] = sd["action_encoder.embed.2.weight"]
    # pred_proj: fold BatchNorm1d (net.1) into net.0, then both linears.
    w0f, _b0f = fold.fold_linear_bn(
        sd["pred_proj.net.0.weight"], sd["pred_proj.net.0.bias"],
        sd["pred_proj.net.1.weight"], sd["pred_proj.net.1.bias"],
        sd["pred_proj.net.1.running_mean"], sd["pred_proj.net.1.running_var"],
    )
    out["pred_proj.fc1"] = w0f
    out["pred_proj.fc2"] = sd["pred_proj.net.3.weight"]
    return out


def quantize_to_dict(name: str, w: np.ndarray, idx: int) -> tuple[dict, dict]:
    out_dim, inner = w.shape
    if not quantize.mac_fits_int32(inner):
        raise ValueError(f"{name}: inner={inner} overflows int32 MAC")
    q, log2 = quantize.quantize_array(w)
    tensor = export.tensor_dict(idx, idx, q, [out_dim, inner])
    scale = {"scale_id": idx, "log2": log2, "dtype": "i8"}
    return tensor, scale


def gelu_int8_table() -> dict:
    """Integer GELU over the int8 activation domain [-128, 127], unit fixed point.

    The committed lookup table the verifier recomputes exactly; the proof attests
    the relation with this table, whatever it is.
    """
    outputs = [round(0.5 * x * (1.0 + math.erf(x / math.sqrt(2)))) for x in range(-128, 128)]
    return {"table_id": 0, "lo": -128, "outputs": outputs}


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/lewm_v0_export.json"
    sd = load_state_dict()
    assert_dims(sd)
    linears = v0_linears(sd)

    # Full V0 quantization + committed manifest.
    ops, weights, scales = [], [], []
    for idx, (name, w) in enumerate(sorted(linears.items())):
        tensor, scale = quantize_to_dict(name, w, idx)
        weights.append(tensor)
        scales.append(scale)
        ops.append({"kind": "linear", "op_id": idx, "weight_id": idx,
                    "bias_id": None, "rows": w.shape[0], "cols": w.shape[1]})
    tables = [gelu_int8_table()]
    manifest = export.build_manifest(ops, weights, tables, scales)
    total = sum(len(t["data"]) for t in weights)
    print("[export] checkpoint  quentinll/lewm-pusht (Hugging Face, MIT)")
    print(f"[export] config      latent_dim={DIM}, history={HIST}, depth={DEPTH}, heads={HEADS}, "
          f"dim_head={DIM_HEAD}, mlp_dim={MLP}")
    print(f"[export] quantize    {len(linears)} linears -> {total:,} int8 params (power-of-two scales)")
    print("[export] commitments (Blake2s-256):")
    for k, v in manifest.items():
        print(f"  {k:24} = {v[:24]}...")

    # --- Rust prover bundle: the real pred_proj head on a real latent. ---
    # pred_proj = Linear(192->2048, BN-folded) -> GELU -> Linear(2048->192).
    rng = np.random.default_rng(0)
    z = rng.standard_normal(DIM)  # a stand-in latent; any latent works (proof is weight-agnostic).
    z_q, z_log2 = quantize.quantize_array(z)
    fc1_q, fc1_log2 = quantize.quantize_array(linears["pred_proj.fc1"])
    fc2_q, fc2_log2 = quantize.quantize_array(linears["pred_proj.fc2"])

    # Calibrate the requant shifts from the actual integer accumulators so the
    # output is non-trivial (each shift brings the int32 accumulator into int8).
    gelu = gelu_int8_table()
    def calib_shift(acc: np.ndarray) -> int:
        return max(0, int(np.ceil(np.log2(max(1.0, float(np.abs(acc).max())) / 127.0))))
    zi = np.asarray(z_q, dtype=np.int64)
    acc1 = np.asarray(fc1_q, dtype=np.int64).reshape(MLP, DIM) @ zi
    fc1_shift = calib_shift(acc1)
    h1 = np.clip(np.round(acc1 / 2.0**fc1_shift), -128, 127).astype(np.int64)
    g1 = np.asarray([gelu["outputs"][int(v) - gelu["lo"]] for v in h1], dtype=np.int64)
    acc2 = np.asarray(fc2_q, dtype=np.int64).reshape(DIM, MLP) @ g1
    fc2_shift = calib_shift(acc2)

    bundle = {
        "model": "lewm-pusht pred_proj head (Linear -> GELU -> Linear), real BN-folded weights",
        "dim": DIM, "mlp": MLP,
        "input": {"data": z_q, "log2": z_log2},
        "fc1": {"data": fc1_q, "rows": MLP, "cols": DIM, "log2": fc1_log2},
        "fc2": {"data": fc2_q, "rows": DIM, "cols": MLP, "log2": fc2_log2},
        "gelu_table": gelu,
        "fc1_shift": fc1_shift,
        "fc2_shift": fc2_shift,
    }
    print(f"calibrated requant shifts: fc1={fc1_shift}, fc2={fc2_shift}")
    Path(out_path).write_text(json.dumps(bundle))
    print(f"\nwrote pred_proj bundle ({Path(out_path).stat().st_size} bytes) to {out_path}")
    print("verify it: cargo run -p pwm-testkit --bin pwm --release -- prove-lewm " + out_path)

    # --- Full predictor bundle: the real 6-block, 16-head attention predictor. ---
    def q8(w: np.ndarray) -> list:
        return quantize.quantize_array(w)[0]
    blocks = []
    for i in range(DEPTH):
        p = f"predictor.transformer.layers.{i}"
        blocks.append({
            "qkv": q8(sd[f"{p}.attn.to_qkv.weight"]),
            "out": q8(sd[f"{p}.attn.to_out.0.weight"]),
            "fc1": q8(sd[f"{p}.mlp.net.1.weight"]),
            "fc2": q8(sd[f"{p}.mlp.net.4.weight"]),
            "adaln": q8(sd[f"{p}.adaLN_modulation.1.weight"]),
        })
    # Inputs, in order of preference:
    #   LEWM_LEROBOT=1  consistent real (observation, action) from a lerobot/pusht
    #                   expert episode: real frames -> encoder, real 2D action+state.
    #   LEWM_GIF=<path> real observation frames (encoder) + a stand-in action.
    #   otherwise       synthetic latents.
    gif = os.environ.get("LEWM_GIF")
    if os.environ.get("LEWM_LEROBOT"):
        from pwm_export import encode, lerobot_pusht
        tsd = torch.load(os.environ.get("LEWM_WEIGHTS", "weights.pt"),
                         map_location="cpu", weights_only=True)
        frames, a10, n = lerobot_pusht.load_episode(history=HIST, frameskip=5)
        emb = encode.encode_observation(tsd, frames)
        act_emb = encode.encode_action(tsd, torch.tensor(a10)).numpy()
        x_q = quantize.quantize_array(emb.flatten())[0]
        c_q = quantize.quantize_array(act_emb.flatten())[0]
        input_source = (f"real PushT expert episode (lerobot/pusht): {n} frames @ frameskip 5 "
                        f"-> ViT encoder; real 2D action + agent state")
        print(f"[encode] {input_source}")
    elif gif and Path(gif).exists():
        from pwm_export import encode
        tsd = torch.load(os.environ.get("LEWM_WEIGHTS", "weights.pt"),
                         map_location="cpu", weights_only=True)
        emb, act_emb, total = encode.encode_history(tsd, gif, HIST, action_dim=10)
        x_q = quantize.quantize_array(emb.flatten())[0]
        c_q = quantize.quantize_array(act_emb.flatten())[0]
        input_source = (f"real PushT observation: {HIST} frames from {Path(gif).name} "
                        f"({total} total) -> ViT encoder -> projector (action is a stand-in)")
        print(f"[encode] {input_source}")
    else:
        x_q = quantize.quantize_array(rng.standard_normal(HIST * DIM))[0]
        c_q = quantize.quantize_array(rng.standard_normal(HIST * DIM))[0]
        input_source = "synthetic quantized latents (set LEWM_LEROBOT=1 for real obs+action)"
    pred_bundle = {
        "model": "lewm-pusht full predictor (6 blocks, 16 heads), real quantized weights",
        "input_source": input_source,
        "dims": {"d": DIM, "s": HIST, "h": HEADS, "dh": DIM_HEAD, "mlp": MLP, "depth": DEPTH},
        "x": x_q,
        "c": c_q,
        "blocks": blocks,
    }
    pred_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/lewm_predictor.json"
    Path(pred_path).write_text(json.dumps(pred_bundle))
    print(f"wrote predictor bundle ({Path(pred_path).stat().st_size} bytes) to {pred_path}")
    print("verify it: cargo run -p pwm-testkit --bin pwm --release -- prove-predictor " + pred_path)


if __name__ == "__main__":
    main()
