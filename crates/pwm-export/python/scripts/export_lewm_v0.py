#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export the REAL pretrained le-wm V0 checkpoint into the prover.

Loads the `quentinll/lewm-pusht` checkpoint (a plain state_dict of named tensors,
so no stable-worldmodel install is needed), extracts the V0 proven subgraph
(`action_encoder -> predictor x6 -> pred_proj`), quantizes every linear to int8
with power-of-two scales, folds the `pred_proj` BatchNorm into the preceding
Linear, asserts the spec-2 dims (192/3/6/16/64/2048), and emits:

1. the committed manifest (model / quantization / graph commitments), and
2. a JSON bundle the Rust prover ingests to prove + verify the full 6-block
   predictor on a real latent/action input, with calibrated activation tables
   and a predictor-scoped weight commitment.

The `pred_proj` head bundle is still emitted as a compact smoke artifact, but the
main demo path is the commitment-bound full predictor bundle.

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
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pwm_export import canonical, export, fold, predictor_quant, quantize

DIM, HEADS, DIM_HEAD, MLP, DEPTH, HIST = 192, 16, 64, 2048, 6, 3
INNER = HEADS * DIM_HEAD  # 1024


# --- observability helpers (a clear staged log; respects NO_COLOR) ---
def _c(code: str, s: str) -> str:
    return s if os.environ.get("NO_COLOR") is not None else f"\x1b[{code}m{s}\x1b[0m"


def _ms(dt: float) -> str:
    return f"{dt * 1000:.0f} ms" if dt < 1.0 else f"{dt:.2f} s"


def _human_bytes(n: int) -> str:
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if x < 1024.0 or unit == "GiB":
            return f"{int(x)} B" if unit == "B" else f"{x:.1f} {unit}"
        x /= 1024.0
    return f"{n} B"


def stage(name: str, sub: str = "") -> None:
    head = _c("35;1", f"[{name}]")
    print(f"\n{head}  {_c('2', sub)}" if sub else f"\n{head}")


def log(label: str, text: str) -> None:
    print(f"  {_c('2', '└')} {label:10} {text}")


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


def predictor_float_blocks(sd: dict) -> list[dict]:
    blocks = []
    for i in range(DEPTH):
        p = f"predictor.transformer.layers.{i}"
        blocks.append({
            "qkv": sd[f"{p}.attn.to_qkv.weight"],
            "out": sd[f"{p}.attn.to_out.0.weight"],
            "fc1": sd[f"{p}.mlp.net.1.weight"],
            "fc2": sd[f"{p}.mlp.net.4.weight"],
            "adaln": sd[f"{p}.adaLN_modulation.1.weight"],
        })
    return blocks


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
    t_start = time.perf_counter()
    out_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/lewm_v0_export.json"
    pred_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/lewm_predictor.json"

    print(_c("36;1", "ProvableWorldModel  export the real le-wm checkpoint into the prover"))
    print(_c("2", "  pipeline   load -> extract V0 -> fold BN -> quantize int8 -> commit -> encode inputs -> bundle"))

    # --- LOAD: the pretrained checkpoint (real torch state_dict). ---
    ckpt = os.environ.get("LEWM_WEIGHTS", "weights.pt")
    t = time.perf_counter()
    sd = load_state_dict()
    t_load = time.perf_counter() - t
    n_tensors = len(sd)
    float_params = sum(int(np.prod(v.shape)) for v in sd.values())
    ckpt_size = Path(ckpt).stat().st_size if Path(ckpt).exists() else 0
    stage("LOAD", "the pretrained checkpoint")
    log("checkpoint", f"quentinll/lewm-pusht  ({Path(ckpt).name}, {_human_bytes(ckpt_size)}, Hugging Face MIT)")
    log("state_dict", f"{n_tensors} tensors, {float_params:,} params (float32) loaded in {_ms(t_load)}")

    # --- EXTRACT: the V0 proven subgraph. ---
    assert_dims(sd)
    linears = v0_linears(sd)
    stage("EXTRACT", "the V0 proven subgraph")
    log("subgraph", "action_encoder -> predictor x6 -> pred_proj")
    log("config", f"latent={DIM}, history={HIST}, depth={DEPTH}, heads={HEADS}, dim_head={DIM_HEAD}, mlp={MLP}")
    log("per block",
        f"to_qkv[{3 * INNER}x{DIM}] to_out[{DIM}x{INNER}] mlp.fc1[{MLP}x{DIM}] "
        f"mlp.fc2[{DIM}x{MLP}] adaLN[{6 * DIM}x{DIM}]")
    log("fold", "pred_proj BatchNorm1d folded into the preceding Linear")

    # --- QUANTIZE every linear to int8 + build the committed manifest. ---
    t = time.perf_counter()
    ops, weights, scales = [], [], []
    for idx, (name, w) in enumerate(sorted(linears.items())):
        tensor, scale = quantize_to_dict(name, w, idx)
        weights.append(tensor)
        scales.append(scale)
        ops.append({"kind": "linear", "op_id": idx, "weight_id": idx,
                    "bias_id": None, "rows": w.shape[0], "cols": w.shape[1]})
    tables = [gelu_int8_table()]
    manifest = export.build_manifest(ops, weights, tables, scales)
    t_quant = time.perf_counter() - t
    int8_params = sum(len(t_["data"]) for t_ in weights)
    log2s = [s["log2"] for s in scales]
    stage("QUANTIZE", "every linear to int8 with per-tensor power-of-two scales")
    log("linears", f"{len(linears)} matrices -> {int8_params:,} int8 params in {_ms(t_quant)}")
    log("compression",
        f"float32 {_human_bytes(int8_params * 4)} -> int8 {_human_bytes(int8_params)} (4.0x smaller)")
    log("scales", f"per-tensor log2 in [{min(log2s)}, {max(log2s)}]")

    stage("COMMIT", "Blake2s-256 over the canonical manifest")
    for k, v in manifest.items():
        log(k, f"{v[:32]}...")

    # --- pred_proj head bundle: the real BN-folded head on a real latent. ---
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
    Path(out_path).write_text(json.dumps(bundle))

    # --- ENCODE the model inputs (latent history + action). ---
    # Inputs, in order of preference:
    #   LEWM_LEROBOT=1  consistent real (observation, action) from a lerobot/pusht
    #                   expert episode: real frames -> encoder, real 2D action+state.
    #   LEWM_GIF=<path> real observation frames (encoder) + a stand-in action.
    #   otherwise       synthetic latents.
    t = time.perf_counter()
    gif = os.environ.get("LEWM_GIF")
    if os.environ.get("LEWM_LEROBOT"):
        from pwm_export import encode, lerobot_pusht
        tsd = torch.load(os.environ.get("LEWM_WEIGHTS", "weights.pt"),
                         map_location="cpu", weights_only=True)
        frames, a10, n = lerobot_pusht.load_episode(history=HIST, frameskip=5)
        emb = encode.encode_observation(tsd, frames)
        act_emb = encode.encode_action(tsd, torch.tensor(a10)).numpy()
        x_float = np.asarray(emb, dtype=np.float64).flatten()
        c_float = np.asarray(act_emb, dtype=np.float64).flatten()
        input_source = (f"real PushT expert episode (lerobot/pusht): {n} frames @ frameskip 5 "
                        f"-> ViT encoder; real 2D action + agent state")
    elif gif and Path(gif).exists():
        from pwm_export import encode
        tsd = torch.load(os.environ.get("LEWM_WEIGHTS", "weights.pt"),
                         map_location="cpu", weights_only=True)
        emb, act_emb, n_frames = encode.encode_history(tsd, gif, HIST, action_dim=10)
        x_float = np.asarray(emb, dtype=np.float64).flatten()
        c_float = np.asarray(act_emb, dtype=np.float64).flatten()
        input_source = (f"real PushT observation: {HIST} frames from {Path(gif).name} "
                        f"({n_frames} total) -> ViT encoder -> projector (action is a stand-in)")
    else:
        x_float = rng.standard_normal(HIST * DIM)
        c_float = rng.standard_normal(HIST * DIM)
        input_source = "synthetic quantized latents (set LEWM_LEROBOT=1 for real obs+action)"
    t_encode = time.perf_counter() - t

    # --- CALIBRATE the full predictor proof relation. ---
    t = time.perf_counter()
    pred_dims = predictor_quant.PredictorDims(DIM, HIST, HEADS, DIM_HEAD, MLP, DEPTH)
    fblocks = predictor_float_blocks(sd)
    pred_quant, blocks, x_q, c_q, _tables = predictor_quant.bundle_quant(
        pred_dims, fblocks, x_float, c_float
    )
    t_calib = time.perf_counter() - t

    # --- Predictor-scoped weight commitment: chain the bundle to the prover. ---
    # predictor_weight_dicts mirrors the Rust prover's tensor registration
    # (lewm_predictor.rs Builder::weight) exactly; the Rust side binds this
    # carried root into the model commitment, so verification fails unless the
    # proven weights reproduce it bit-for-bit.
    pred_weights = export.predictor_weight_dicts(blocks, DIM, HEADS, DIM_HEAD, MLP)
    pred_weights_root = canonical.weights_root(pred_weights).hex()

    pred_bundle = {
        "model": "lewm-pusht full predictor (6 blocks, 16 heads), real quantized weights",
        "input_source": input_source,
        "dims": {"d": DIM, "s": HIST, "h": HEADS, "dh": DIM_HEAD, "mlp": MLP, "depth": DEPTH},
        "weights_root": pred_weights_root,
        "x": x_q,
        "c": c_q,
        "blocks": blocks,
        "quant": pred_quant,
    }
    Path(pred_path).write_text(json.dumps(pred_bundle))

    stage("ENCODE", "the model inputs (latent history + action)")
    log("source", input_source)
    log("z_history", f"[{HIST}x{DIM}] -> int8 ({len(x_q)} values)")
    log("action", f"[{HIST}x{DIM}] -> int8 ({len(c_q)} values)  in {_ms(t_encode)}")

    stage("CALIBRATE", "real activation tables and float-faithful tolerance")
    log("scales", f"f_x={pred_quant['f_x']} f_c={pred_quant['f_c']} f_ln={pred_quant['f_ln']} "
        f"f_qkv={pred_quant['f_qkv']} f_score={pred_quant['f_score']}")
    log("tables", "SiLU, GELU, inverse-sqrt, softmax-exp committed in the bundle")
    log("error", f"max |int - float| = {pred_quant['error']:.6g} "
        f"(tolerance {pred_quant['tolerance']:.6g}) in {_ms(t_calib)}")

    # --- BUNDLE the prover-ingestible artifacts. ---
    stage("BUNDLE", "the prover-ingestible artifacts")
    log("pred_proj", f"head bundle  {_human_bytes(Path(out_path).stat().st_size)} -> {out_path}")
    log("predictor", f"full bundle  {_human_bytes(Path(pred_path).stat().st_size)} -> {pred_path}")
    log("bind", f"predictor weights_root {pred_weights_root[:32]}...  "
        "(carried in the bundle; the prover must reproduce it bit-for-bit)")
    log("calib", f"requant shifts fc1={fc1_shift}, fc2={fc2_shift}")
    log("prove", f"pwm prove-predictor {pred_path}")

    print(_c("32;1",
             f"\nreal le-wm checkpoint exported, quantized, and committed in "
             f"{_ms(time.perf_counter() - t_start)}; ready to prove."))


if __name__ == "__main__":
    main()
