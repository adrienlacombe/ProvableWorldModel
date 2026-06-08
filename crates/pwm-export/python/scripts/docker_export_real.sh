#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Download the real le-wm checkpoint and a PushT observation GIF, encode real
# latents through the checkpoint's ViT encoder, and export the quantized V0
# bundles to /shared.
set -e
CKPT=https://huggingface.co/quentinll/lewm-pusht/resolve/main/weights.pt
GIF=https://raw.githubusercontent.com/lucas-maes/le-wm/main/assets/lewm.gif
echo "[export] downloading the quentinll/lewm-pusht checkpoint and a PushT observation GIF..."
curl -sSL -o /shared/weights.pt "$CKPT"
curl -sSL -o /shared/lewm.gif "$GIF"
echo "[export] $(wc -c < /shared/weights.pt) bytes; encoding real frames + quantizing the full V0..."
LEWM_WEIGHTS=/shared/weights.pt LEWM_GIF=/shared/lewm.gif python scripts/export_lewm_v0.py \
  /shared/lewm_pred_proj.json /shared/lewm_predictor.json
