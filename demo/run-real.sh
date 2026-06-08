#!/usr/bin/env bash
# The REAL pretrained quentinll/lewm-pusht checkpoint, end to end: download,
# quantize the full V0, then prove + verify the real-weight 192-dim predictor.
# Heavy: pulls a PyTorch image and downloads a ~70 MB checkpoint the first time.
set -euo pipefail
cd "$(dirname "$0")/.."
exec docker compose --profile real up --build export predictor-real
