# Agent context

Context for coding agents working in this repository. Keep it current when public
behavior changes.

## What this is

ProvableWorldModel is a commit-and-audit proof system: anyone can verify, on a CPU
with no floating point, that a committed quantized
[le-wm](https://github.com/lucas-maes/le-wm) JEPA world model was run exactly as
claimed. It adapts the [CommitLLM](https://github.com/lambdaclass/CommitLLM)
scheme (Freivalds matmul checks plus exact integer replay, Merkle commitments, a
Fiat-Shamir transcript) from language models to a world model, and because the
model runs in exact integer fixed point it closes CommitLLM's one open hole:
non-reproducible attention. There is no proving circuit and no arithmetization.

## Current state

- The full le-wm V0 predictor (latent_dim 192, history 3, depth 6, 16 heads,
  dim_head 64, mlp 2048; 2,437 ops) proves and verifies in the Rust prover. Per
  ConditionalBlock: AdaLN-zero conditioning, 16-head self-attention, GELU
  feed-forward, gated residuals, with the multi-head reshapes as Slice/Concat over
  a named-buffer block DAG (`pwm-core::block`, `prove_block`/`verify_block`).
- The exporter ingests the real `quentinll/lewm-pusht` checkpoint, takes a real
  PushT expert episode from `lerobot/pusht`, encodes the real observation frames
  through the checkpoint's own ViT encoder plus projector and the real expert
  action through the action encoder, quantizes the full 192-dim V0 subgraph, folds
  BatchNorm, and commits the manifest. `pwm prove-predictor <bundle>` then proves
  and verifies the real weights on the real observation and action.
- Tiers P0 (one predictor step), P1 (rollout), P2 (fixed-candidate planning) are
  implemented and tested. P3 (full CEM planner) and P4 (pixel-to-plan, the ViT
  encoder inside the proof) are deferred. The image encoder is the trusted offline
  step today.
- The proof attests the exact integer (quantized) relation, not float or PyTorch
  equivalence. Per-tensor activation-scale calibration for float-faithful outputs
  is a further refinement; activations stay int8 throughout the current scheme.
- The argmin uniqueness check and the Freivalds probability bound are formally
  verified in Lean 4 (no `sorry`), under `lean/`.

## Layout

Five small crates, one trust anchor. The verifier depends on neither the exporter
nor any Python or float runtime.

- `pwm-core`: fields (M31 value, Fp61 audit), fixed point, tensors, Merkle
  commitments, Fiat-Shamir transcript, the Freivalds check, the block DAG, the
  trace model. `no_std`.
- `pwm-export`: checkpoint to quantized integer graph, manifest, golden vectors,
  the Rust integer reference, the lerobot/pusht data adapter (Python under
  `crates/pwm-export/python`).
- `pwm-prover`: run the reference inference, commit the trace, derive challenges,
  emit the `AuditArtifact`.
- `pwm-verifier`: CPU, `no_std`, float-free. Freivalds-check the linears,
  recompute the rest, check rollout, cost, argmin.
- `pwm-testkit`: golden vectors, accept and reject suites, the mutation harness,
  the `pwm` demo CLI (`crates/pwm-testkit/src/bin/pwm.rs`).

## Run the demo

```bash
docker compose up --build                                  # full predictor, synthetic weights
docker compose --profile real up --build export predictor-real   # real pretrained checkpoint
docker compose --profile compact up --build prover verifier      # tiny two-party handoff
```

Without Docker:

```bash
cargo run -p pwm-testkit --bin pwm --release -- prove-predictor          # synthetic
cargo run -p pwm-testkit --bin pwm --release -- prove-predictor <bundle> # real weights
cargo test --workspace                                                   # accept + reject suites
```

The `pwm` CLI prints a four-stage pipeline (EXPORT, PROVE, VERIFY, TAMPER); it
honors `NO_COLOR` and `--json`.

## Conventions

- Stable Rust toolchain (MSRV 1.85). No nightly. The verifier is `no_std` and
  float-free; keep it that way.
- Prose (README, demo docs, website, printed CLI output) uses no em-dashes. Rust
  doc-comments may keep them per repo style.
- Commits and PRs carry no tool attribution and no `Co-Authored-By` lines.
- One logical change per branch and PR. Branch from `main`, open a PR, keep CI
  green (fmt, clippy `-D warnings`, the full test suite, SPDX headers, link check,
  the no-em-dash check). Confirm before any destructive git operation.
- Keep complexity and file size as low as the task allows. Small, focused files.

## Pointers

- [README.md](README.md): overview and quickstart.
- [demo/README.md](demo/README.md): the three demo modes and the real-checkpoint steps.
- [specs.md](specs.md): the normative commit-and-audit specification.
- [roadmap.md](roadmap.md): the plan, the pivot rationale, and the sequencing.
- [website/](website/): the interactive explainer, deployed to GitHub Pages on
  push to `main` via `.github/workflows/pages.yml`.
