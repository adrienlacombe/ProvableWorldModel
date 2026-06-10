# SPDX-License-Identifier: Apache-2.0
"""Calibrated quantization for the full le-wm predictor bundle.

This mirrors `pwm_export::predictor_quant` on the Rust side closely enough for
the trusted Python export to emit a self-contained predictor bundle: per-site
fixed-point scales, the real activation tables, quantized predictor weights and
inputs, the float reference output, and a measured int-vs-float tolerance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import quantize

LN_EPS = 1e-5
TABLE_SILU = 1
TABLE_GELU = 2
TABLE_INVSQRT = 3
TABLE_EXP = 4
INVSQRT_DOMAIN_HI = 510 * 510
CLO, CHI = -128, 127


@dataclass(frozen=True)
class PredictorDims:
    d: int
    s: int
    h: int
    dh: int
    mlp: int
    depth: int

    @property
    def inner(self) -> int:
        return self.h * self.dh


def _as_block(block: dict) -> dict:
    return {k: np.asarray(block[k], dtype=np.float64) for k in ("adaln", "qkv", "out", "fc1", "fc2")}


def gelu(x):
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def silu(x):
    x = np.asarray(x, dtype=np.float64)
    return x / (1.0 + np.exp(-x))


def _round_half_away(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0.0, np.floor(x + 0.5), np.ceil(x - 0.5)).astype(np.int64)


def _table(table_id: int, lo: int, hi: int, in_frac: int, out_frac: int, fn) -> dict:
    xs = np.arange(lo, hi + 1, dtype=np.float64) * (2.0 ** -in_frac)
    ys = _round_half_away(fn(xs) * (2.0**out_frac)).astype(np.int64)
    return {"table_id": table_id, "lo": lo, "outputs": ys.tolist()}


def inv_sqrt_norm_table(table_id: int, hi: int, t_inv: int) -> dict:
    qs = np.arange(0, hi + 1, dtype=np.float64)
    out = np.empty(hi + 1, dtype=np.int64)
    out[0] = 1 << t_inv
    out[1:] = _round_half_away((2.0**t_inv) / np.sqrt(qs[1:]))
    return {"table_id": table_id, "lo": 0, "outputs": out.tolist()}


def scheme_tables(scheme: dict) -> list[dict]:
    return [
        _table(TABLE_SILU, -128, 127, scheme["f_c"], scheme["f_c"], silu),
        _table(TABLE_GELU, -128, 127, scheme["f_g1"], scheme["f_g2"], gelu),
        inv_sqrt_norm_table(TABLE_INVSQRT, INVSQRT_DOMAIN_HI, scheme["t_inv"]),
        _table(TABLE_EXP, -255, 0, scheme["f_score"], scheme["f_e"], np.exp),
    ]


def quantize_at(vals, frac: int) -> list[int]:
    q = _round_half_away(np.asarray(vals, dtype=np.float64) * (2.0**frac))
    return np.clip(q, -128, 127).astype(np.int64).flatten().tolist()


def quantize_weights(blocks: list[dict]) -> tuple[list[dict], list[list[int]]]:
    qblocks = []
    w_log2 = []
    for raw in blocks:
        b = _as_block(raw)
        row = []
        q = {}
        for key in ("adaln", "qkv", "out", "fc1", "fc2"):
            vals, log2 = quantize.quantize_array(b[key])
            q[key] = vals
            row.append(int(log2))
        qblocks.append(q)
        w_log2.append(row)
    return qblocks, w_log2


def _ln_rows(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=1, keepdims=True)
    return (x - mean) / np.sqrt(var + LN_EPS)


def _batched_linear(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return x @ w.T


def _cols(x: np.ndarray, off: int, length: int) -> np.ndarray:
    return x[:, off : off + length]


def _softmax(x: np.ndarray) -> np.ndarray:
    y = x - x.max(axis=1, keepdims=True)
    e = np.exp(y)
    return e / e.sum(axis=1, keepdims=True)


def float_forward(dims: PredictorDims, blocks: list[dict], x0, c) -> tuple[np.ndarray, dict]:
    x = np.asarray(x0, dtype=np.float64).reshape(dims.s, dims.d).copy()
    c = np.asarray(c, dtype=np.float64).reshape(dims.s, dims.d)
    cal = {k: 0.0 for k in ("x", "c", "ln", "adaln", "qkv", "score", "att", "attn_out", "g1", "g2", "ffn", "mod_prod")}

    def see(k: str, arr) -> None:
        a = np.asarray(arr, dtype=np.float64)
        if a.size:
            cal[k] = max(cal[k], float(np.abs(a).max()))

    see("x", x)
    see("c", c)
    silu_c = silu(c)
    see("c", silu_c)
    scale = 1.0 / math.sqrt(float(dims.dh))

    for raw in blocks:
        b = _as_block(raw)
        ada = _batched_linear(silu_c, b["adaln"])
        see("adaln", ada)
        sh_msa, sc_msa, g_msa, sh_mlp, sc_mlp, g_mlp = [ada[:, i * dims.d : (i + 1) * dims.d] for i in range(6)]

        n1 = _ln_rows(x)
        see("ln", n1)
        prod1 = n1 * (1.0 + sc_msa)
        see("mod_prod", prod1)
        m1 = prod1 + sh_msa

        a0 = _ln_rows(m1)
        see("ln", a0)
        qkv = _batched_linear(a0, b["qkv"])
        see("qkv", qkv)
        q = _cols(qkv, 0, dims.inner)
        k = _cols(qkv, dims.inner, dims.inner)
        v = _cols(qkv, 2 * dims.inner, dims.inner)
        heads = np.zeros((dims.s, dims.inner), dtype=np.float64)
        for head in range(dims.h):
            off = head * dims.dh
            qh = _cols(q, off, dims.dh)
            kh = _cols(k, off, dims.dh)
            vh = _cols(v, off, dims.dh)
            scores = (qh @ kh.T) * scale
            see("score", scores)
            heads[:, off : off + dims.dh] = _softmax(scores) @ vh
        see("att", heads)
        a = _batched_linear(heads, b["out"])
        see("attn_out", a)
        gated1 = g_msa * a
        see("x", gated1)
        x = x + gated1
        see("x", x)

        n2 = _ln_rows(x)
        see("ln", n2)
        prod2 = n2 * (1.0 + sc_mlp)
        see("mod_prod", prod2)
        m2 = prod2 + sh_mlp
        f0 = _ln_rows(m2)
        see("ln", f0)
        h1 = _batched_linear(f0, b["fc1"])
        see("g1", h1)
        g = gelu(h1)
        see("g2", g)
        f = _batched_linear(g, b["fc2"])
        see("ffn", f)
        gated2 = g_mlp * f
        see("x", gated2)
        x = x + gated2
        see("x", x)

    z = _ln_rows(x)
    see("ln", z)
    return z.reshape(-1), cal


def _frac_for(absmax: float, margin: float = 126.0) -> int:
    if absmax <= 0.0:
        return 7
    return min(int(math.floor(math.log2(margin / absmax))), 12)


def calibrate(dims: PredictorDims, blocks: list[dict], x0, c) -> tuple[dict, np.ndarray]:
    z, cal = float_forward(dims, blocks, x0, c)
    _, w_log2 = quantize_weights(blocks)
    scheme = {
        "f_x": _frac_for(cal["x"]),
        "f_c": _frac_for(cal["c"]),
        "f_ln": _frac_for(max(cal["ln"], cal["adaln"], cal["mod_prod"])),
        "f_qkv": _frac_for(cal["qkv"]),
        "f_score": _frac_for(cal["score"]),
        "f_p": 10,
        "f_att": _frac_for(cal["att"]),
        "f_a": _frac_for(cal["attn_out"]),
        "f_g1": _frac_for(cal["g1"]),
        "f_g2": _frac_for(cal["g2"]),
        "f_f": _frac_for(cal["ffn"]),
        "t_inv": 16,
        "f_e": 15,
        "w_log2": w_log2,
    }
    return scheme, z


def derive(scheme: dict, dims: PredictorDims) -> dict:
    if dims.dh & (dims.dh - 1) != 0 or int(math.log2(dims.dh)) % 2 != 0:
        raise ValueError("dh must be a power of four")
    half_log2_dh = int(math.log2(dims.dh)) // 2

    def nonneg(site: str, v: int) -> int:
        if v < 0:
            raise ValueError(f"{site} needs negative shift {v}")
        return int(v)

    blocks = []
    for row in scheme["w_log2"]:
        fw = [-int(x) for x in row]
        blocks.append(
            {
                "adaln": nonneg("adaln", scheme["f_c"] + fw[0] - scheme["f_ln"]),
                "qkv": nonneg("qkv", scheme["f_ln"] + fw[1] - scheme["f_qkv"]),
                "out": nonneg("out", scheme["f_att"] + fw[2] - scheme["f_a"]),
                "fc1": nonneg("fc1", scheme["f_ln"] + fw[3] - scheme["f_g1"]),
                "fc2": nonneg("fc2", scheme["f_g2"] + fw[4] - scheme["f_f"]),
            }
        )
    return {
        "ln_shift": nonneg("layernorm", scheme["t_inv"] - scheme["f_ln"]),
        "score_shift": nonneg("score", 2 * scheme["f_qkv"] + half_log2_dh - scheme["f_score"]),
        "oh_shift": nonneg("prob_v", scheme["f_p"] + scheme["f_qkv"] - scheme["f_att"]),
        "gate_msa_bits": nonneg("gate_msa", scheme["f_ln"] + scheme["f_a"] - scheme["f_x"]),
        "gate_mlp_bits": nonneg("gate_mlp", scheme["f_ln"] + scheme["f_f"] - scheme["f_x"]),
        "mod_one": 1 << scheme["f_ln"],
        "mod_bits": nonneg("modulate", scheme["f_ln"]),
        "softmax_one": 1 << scheme["f_p"],
        "blocks": blocks,
    }


def _requant(x, shift: int):
    arr = np.asarray(x, dtype=np.int64)
    if shift == 0:
        rounded = arr
    else:
        div = np.int64(1 << shift)
        q = np.floor_divide(arr, div)
        rem = np.mod(arr, div)
        half = np.int64(1 << (shift - 1))
        rounded = np.where(rem < half, q, np.where(rem > half, q + 1, q + (q & 1)))
    return np.clip(rounded, CLO, CHI).astype(np.int64)


def _round_div(a: int, n: int) -> int:
    if a >= 0:
        return (a + n // 2) // n
    return -((-a + n // 2) // n)


def _eval_table(table: dict, x):
    arr = np.asarray(x, dtype=np.int64)
    idx = arr - int(table["lo"])
    outs = np.asarray(table["outputs"], dtype=np.int64)
    if np.any(idx < 0) or np.any(idx >= len(outs)):
        raise ValueError(f"table {table['table_id']} lookup out of domain")
    return outs[idx]


def _ln_rows_int(x: np.ndarray, inv_table: dict, shift: int) -> np.ndarray:
    rows = []
    for row in np.asarray(x, dtype=np.int64):
        n = int(row.size)
        mean = _round_div(int(row.sum()), n)
        centered = row - mean
        var = _round_div(int((centered * centered).sum()), n)
        inv = int(_eval_table(inv_table, var))
        rows.append(_requant(centered * inv, shift))
    return np.vstack(rows).astype(np.int64)


def _linear_int(x: np.ndarray, w_flat, rows: int, cols: int) -> np.ndarray:
    w = np.asarray(w_flat, dtype=np.int64).reshape(rows, cols)
    return np.asarray(x, dtype=np.int64) @ w.T


def _softmax_rows_int(scores: np.ndarray, exp_table: dict, one: int) -> np.ndarray:
    out = []
    for row in np.asarray(scores, dtype=np.int64):
        shifted = row - int(row.max())
        exps = _eval_table(exp_table, shifted)
        total = int(exps.sum())
        if total == 0:
            raise ValueError("softmax exp sum is zero")
        out.append((exps * one) // total)
    return np.vstack(out).astype(np.int64)


def integer_forward(dims: PredictorDims, qblocks: list[dict], xq, cq, scheme: dict, tables: list[dict]) -> np.ndarray:
    p = derive(scheme, dims)
    by_id = {t["table_id"]: t for t in tables}
    x = np.asarray(xq, dtype=np.int64).reshape(dims.s, dims.d).copy()
    c = np.asarray(cq, dtype=np.int64).reshape(dims.s, dims.d)
    silu_c = _eval_table(by_id[TABLE_SILU], c)
    for qb, bs in zip(qblocks, p["blocks"], strict=True):
        ada = _requant(_linear_int(silu_c, qb["adaln"], 6 * dims.d, dims.d), bs["adaln"])
        chunks = [ada[:, i * dims.d : (i + 1) * dims.d] for i in range(6)]
        sh_msa, sc_msa, g_msa, sh_mlp, sc_mlp, g_mlp = chunks

        n1 = _ln_rows_int(x, by_id[TABLE_INVSQRT], p["ln_shift"])
        m1 = _requant(n1 * (p["mod_one"] + sc_msa), p["mod_bits"]) + sh_msa
        a0 = _ln_rows_int(m1, by_id[TABLE_INVSQRT], p["ln_shift"])
        qkv = _requant(_linear_int(a0, qb["qkv"], 3 * dims.inner, dims.d), bs["qkv"])
        q = qkv[:, 0 : dims.inner]
        k = qkv[:, dims.inner : 2 * dims.inner]
        v = qkv[:, 2 * dims.inner : 3 * dims.inner]
        heads = np.zeros((dims.s, dims.inner), dtype=np.int64)
        for head in range(dims.h):
            off = head * dims.dh
            qh = q[:, off : off + dims.dh]
            kh = k[:, off : off + dims.dh]
            vh = v[:, off : off + dims.dh]
            scores = _requant(qh @ kh.T, p["score_shift"])
            prob = _softmax_rows_int(scores, by_id[TABLE_EXP], p["softmax_one"])
            heads[:, off : off + dims.dh] = _requant(prob @ vh, p["oh_shift"])
        a = _requant(_linear_int(heads, qb["out"], dims.d, dims.inner), bs["out"])
        x = _requant(x + _requant(g_msa * a, p["gate_msa_bits"]), 0)

        n2 = _ln_rows_int(x, by_id[TABLE_INVSQRT], p["ln_shift"])
        m2 = _requant(n2 * (p["mod_one"] + sc_mlp), p["mod_bits"]) + sh_mlp
        f0 = _ln_rows_int(m2, by_id[TABLE_INVSQRT], p["ln_shift"])
        h1 = _requant(_linear_int(f0, qb["fc1"], dims.mlp, dims.d), bs["fc1"])
        gelu_h = _eval_table(by_id[TABLE_GELU], h1)
        f = _requant(_linear_int(gelu_h, qb["fc2"], dims.d, dims.mlp), bs["fc2"])
        x = _requant(x + _requant(g_mlp * f, p["gate_mlp_bits"]), 0)

    return _ln_rows_int(x, by_id[TABLE_INVSQRT], p["ln_shift"]).reshape(-1)


def bundle_quant(
    dims: PredictorDims,
    blocks: list[dict],
    x_float,
    c_float,
) -> tuple[dict, list[dict], list[int], list[int], list[dict]]:
    scheme, z_float = calibrate(dims, blocks, x_float, c_float)
    tables = scheme_tables(scheme)
    qblocks, _ = quantize_weights(blocks)
    xq = quantize_at(x_float, scheme["f_x"])
    cq = quantize_at(c_float, scheme["f_c"])
    z_int = integer_forward(dims, qblocks, xq, cq, scheme, tables)
    err = max_float_error(z_int, scheme["f_ln"], z_float)
    tolerance = math.ceil((err + 1e-9) * 1_000_000.0) / 1_000_000.0
    quant = {
        **{k: v for k, v in scheme.items() if k != "w_log2"},
        "w_log2": scheme["w_log2"],
        "tables": tables,
        "z_out_float": [float(v) for v in z_float.reshape(-1)],
        "z_out_int": z_int.astype(np.int64).tolist(),
        "error": float(err),
        "tolerance": float(tolerance),
    }
    return quant, qblocks, xq, cq, tables


def max_float_error(z_int, f_out: int, z_float) -> float:
    zi = np.asarray(z_int, dtype=np.float64) * (2.0 ** -f_out)
    zf = np.asarray(z_float, dtype=np.float64).reshape(-1)
    return float(np.max(np.abs(zi - zf))) if zi.size else 0.0
