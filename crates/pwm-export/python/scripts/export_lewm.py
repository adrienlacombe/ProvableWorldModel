#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real le-wm V0 export driver (backlog E-201/202/203/205).

Builds the **actual** le-wm V0 subgraph (upstream `module.py`: `ARPredictor`,
`Embedder`, `MLP`), saves a checkpoint, then runs the export pipeline against the
real torch parameters: `torch.load` → dim ingest (`pwm_export.lewm`) → int8
quantize (`quantize`) → BatchNorm fold (`fold`) → committed manifest (`export`).

le-wm's V0 encoder uses `pretrained: false` (trained from scratch), so a random
init is faithful to the V0 config — no pretrained checkpoint is required to
exercise the full export and the proof relations (which attest the *quantized*
model regardless of weight values). Point at a real trained checkpoint by setting
`LEWM_CKPT`; otherwise a fresh random V0 instance is built.

Run:
    pip install torch einops numpy
    git clone https://github.com/lucas-maes/le-wm
    LEWM_PATH=./le-wm python scripts/export_lewm.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.environ.get("LEWM_PATH", "./le-wm"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from module import ARPredictor, Embedder, MLP  # upstream le-wm
from pwm_export import export, fold, ingest, lewm, quantize

# V0 config — specs.md §2 / config/train/model/lewm.yaml.
EMBED, HIST, DEPTH, HEADS, DIM_HEAD, MLP_DIM, ACTION_DIM = 192, 3, 6, 16, 64, 2048, 4


def build_v0() -> nn.Module:
    torch.manual_seed(0)
    predictor = ARPredictor(num_frames=HIST, depth=DEPTH, heads=HEADS, mlp_dim=MLP_DIM,
                            input_dim=EMBED, hidden_dim=EMBED, output_dim=EMBED,
                            dim_head=DIM_HEAD, dropout=0.0, emb_dropout=0.0)
    action_encoder = Embedder(input_dim=ACTION_DIM, emb_dim=EMBED)
    pred_proj = MLP(input_dim=EMBED, hidden_dim=MLP_DIM, output_dim=EMBED,
                    norm_fn=lambda d: nn.BatchNorm1d(d))
    return nn.ModuleDict(
        {"predictor": predictor, "action_encoder": action_encoder, "pred_proj": pred_proj}
    ).eval()


def main() -> None:
    ckpt = os.environ.get("LEWM_CKPT")
    if ckpt:
        model = torch.load(ckpt, weights_only=False).eval()
        print(f"loaded real le-wm checkpoint: {ckpt}")
    else:
        model = build_v0()
        out = Path("/tmp/lewm_v0.pt")
        torch.save(model, out)
        model = torch.load(out, weights_only=False).eval()  # exercise the real load path
        print(f"built + reloaded random V0 (pretrained:false-faithful) -> {out}")

    params = {k: v.detach().cpu().numpy() for k, v in model.named_parameters()}

    # E-201: validate real-module dims and extract the V0 Freivalds linears.
    linears = lewm.extract_v0_linears(params)
    print(f"E-201: V0 dims {ingest.V0_DIMS.as_tuple()} validated; {len(linears)} linears extracted")

    # E-202: quantize each real weight; int8 range + int32 MAC bound.
    for name, w in linears:
        q, _ = quantize.quantize_array(w)
        assert all(-128 <= x <= 127 for x in q)
        assert quantize.mac_fits_int32(w.shape[1]), name
    print(f"E-202: quantized {len(linears)} weights (int8 + MAC < int32)")

    # E-203: fold pred_proj BatchNorm1d into the preceding Linear; check equality.
    bn, lin0 = model.pred_proj.net[1], model.pred_proj.net[0]
    W2, b2 = export.fold_torch(lin0.weight, lin0.bias, bn)
    x = torch.randn(5, EMBED).double()
    err = float(np.abs(fold.linear(x.numpy(), W2, b2) - bn.double()(lin0.double()(x)).detach().numpy()).max())
    assert err < 1e-9
    print(f"E-203: fold == Linear∘BN(eval), max err = {err:.2e}")

    # E-205: committed manifest from the real quantized weights.
    named = [(i, name, w) for i, (name, w) in enumerate(linears)]
    tables = [{"table_id": 0, "lo": -4, "outputs": [0, 0, 0, 1, 2, 3, 4, 5, 6]}]
    m = export.export_graph(named, tables)["manifest"]
    print("E-205: manifest commitments —")
    for k, v in m.items():
        print(f"  {k:24} = {v[:32]}…")
    print("\n✅ real le-wm V0 export ran end-to-end.")


if __name__ == "__main__":
    main()
