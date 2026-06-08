#!/usr/bin/env bash
# One command: build the image and play the commit-and-audit challenge game.
# The prover writes a proof to a shared volume; the verifier accepts it, then
# tries to forge a matmul and gets rejected. Exits with the verifier's code.
set -euo pipefail
cd "$(dirname "$0")/.."
exec docker compose up --build --exit-code-from verifier
