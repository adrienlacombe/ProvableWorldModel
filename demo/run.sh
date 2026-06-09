#!/usr/bin/env bash
# The real le-wm predictor architecture (192-dim, 16 heads, 6 blocks), proven and
# verified in exact integer arithmetic. Pure Rust, fast, offline (synthetic weights).
set -euo pipefail
cd "$(dirname "$0")/.."
echo "ProvableWorldModel demo  ·  real le-wm architecture, exact-integer inference + proof"
echo "  weights are synthetic (real architecture, real integer inference; fast, no download)."
echo "  for the REAL pretrained checkpoint end to end, run:  ./demo/run-real.sh"
echo
exec docker compose up --build
