#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Download the real le-wm checkpoint, encode a real lerobot/pusht expert episode
# (consistent real observation + action) through the checkpoint's encoder, and
# export the quantized V0 bundles to /shared.
set -e
CKPT=https://huggingface.co/quentinll/lewm-pusht/resolve/main/weights.pt
echo "[export] downloading the quentinll/lewm-pusht checkpoint..."
curl -sSL -o /shared/weights.pt "$CKPT"
echo "[export] $(wc -c < /shared/weights.pt) bytes; encoding a real lerobot/pusht episode + quantizing the full V0..."
LEWM_WEIGHTS=/shared/weights.pt LEWM_LEROBOT=1 python scripts/export_lewm_v0.py \
  /shared/lewm_pred_proj.json /shared/lewm_predictor.json
