#!/usr/bin/env bash
# The real le-wm predictor architecture (192-dim, 16 heads, 6 blocks), proven and
# verified in exact integer arithmetic. Pure Rust, fast, offline (synthetic weights).
set -euo pipefail
cd "$(dirname "$0")/.."
exec docker compose up --build
