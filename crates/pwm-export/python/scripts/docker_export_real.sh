#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Download the real le-wm checkpoint and export the quantized V0 bundles to /shared.
set -e
CKPT=https://huggingface.co/quentinll/lewm-pusht/resolve/main/weights.pt
echo "[export] downloading quentinll/lewm-pusht checkpoint (weights.pt)..."
curl -sSL -o /shared/weights.pt "$CKPT"
echo "[export] $(wc -c < /shared/weights.pt) bytes; quantizing the full V0 subgraph..."
LEWM_WEIGHTS=/shared/weights.pt python scripts/export_lewm_v0.py \
  /shared/lewm_pred_proj.json /shared/lewm_predictor.json
