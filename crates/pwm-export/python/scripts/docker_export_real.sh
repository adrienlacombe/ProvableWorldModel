#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Download the real le-wm checkpoint, encode a real lerobot/pusht expert episode
# (consistent real observation + action) through the checkpoint's encoder, and
# export the quantized V0 bundles to /shared. This is stage 1/2 of the real
# end-to-end demo; stage 2/2 is the `predictor-real` service proving the bundle.
set -e
CKPT=https://huggingface.co/quentinll/lewm-pusht/resolve/main/weights.pt

printf '\n=== ProvableWorldModel  real end-to-end demo   [stage 1/2: EXPORT service] ===\n'
echo "[download] pretrained checkpoint  quentinll/lewm-pusht (Hugging Face, MIT)"
echo "[download]   $CKPT"
t0=$(date +%s)
curl -fSL --progress-bar -o /shared/weights.pt "$CKPT"
sz=$(wc -c < /shared/weights.pt)
echo "[download] got $sz bytes (~$((sz / 1024 / 1024)) MiB) in $(($(date +%s) - t0))s"
echo "[encode]   real lerobot/pusht expert episode -> ViT encoder; quantizing the full 192-dim V0 ..."

LEWM_WEIGHTS=/shared/weights.pt LEWM_LEROBOT=1 python scripts/export_lewm_v0.py \
  /shared/lewm_pred_proj.json /shared/lewm_predictor.json

printf '\n[stage 1/2: EXPORT] complete; prover bundle written to /shared/lewm_predictor.json.\n'
printf '          next: the predictor-real service runs the exact-integer inference + proof.\n'
