#!/usr/bin/env bash
# The REAL pretrained quentinll/lewm-pusht checkpoint, end to end: download,
# quantize the full V0, then prove + verify the real-weight 192-dim predictor.
# Heavy: pulls a PyTorch image and downloads a ~70 MB checkpoint the first time.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "ProvableWorldModel real demo  ·  REAL pretrained checkpoint, end to end"
echo "  stage 1/2  export service (PyTorch): download quentinll/lewm-pusht (~70 MiB),"
echo "             encode a real lerobot/pusht episode, quantize the full 192-dim V0, commit."
echo "  stage 2/2  prover (pure Rust): exact-integer inference on the REAL weights + audit."
echo
exec docker compose --profile real up --build export predictor-real
