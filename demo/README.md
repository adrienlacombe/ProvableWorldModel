# The demo: the le-wm world model, proven

This plays ProvableWorldModel end to end on your machine. The prover runs the
le-wm predictor in exact integer arithmetic, commits to its execution trace, and
the no_std verifier replays the Fiat-Shamir challenge, Freivalds-checks every
matmul, exactly recomputes the attention, softmax, GELU, LayerNorm, and residuals,
and accepts. Then a single matmul output is forged and the verifier rejects it.
No GPU, no floating point.

## 1. The real architecture (default, fast)

```bash
docker compose up --build
```

Runs the **real le-wm predictor architecture** at the real V0 dims (`latent_dim
192`, `history 3`, `depth 6`, `16 heads`, `dim_head 64`, `mlp 2048`): AdaLN-zero
conditioning, 16-head self-attention, GELU feed-forward, residuals, ~2437 ops.
Pure Rust, offline, weights are synthetic.

```
[prover] model   le-wm V0 predictor (6 blocks, 16 heads), synthetic weights (pass a bundle for real)
[prover] config  dim=192, history=3, heads=16, dim_head=64, mlp=2048, depth=6
[prover] weights 30 tensors, 10764288 int8 params
[prover] graph   2437 ops over the named-buffer block DAG
[prover] infer   exact integer forward pass in 36 ms
[verifier] ACCEPT  in 26 ms
[verifier] tamper  forged matmul op Some(2) -> REJECT FreivaldsCheckFailed { op_id: 2 }
```

## 2. The real pretrained checkpoint (real weights)

```bash
docker compose --profile real up --build export predictor-real
# or: ./demo/run-real.sh
```

This downloads the real [`quentinll/lewm-pusht`](https://huggingface.co/quentinll/lewm-pusht)
checkpoint and a PushT observation GIF, **encodes real observation frames through
the checkpoint's own ViT encoder + projector into a real latent history**, quantizes
the full 192-dim V0 subgraph, and proves the predictor with the **real quantized
weights on the real latents**. Heavy: it pulls a PyTorch image and downloads a
~70 MB checkpoint the first time.

```
[export] checkpoint  quentinll/lewm-pusht (Hugging Face, MIT)
[export] config      latent_dim=192, history=3, depth=6, heads=16, dim_head=64, mlp_dim=2048
[export] quantize    34 linears -> 11,705,856 int8 params (power-of-two scales)
[encode] real PushT observation: 3 frames from lewm.gif (1029 total) -> ViT encoder -> projector
[prover] model   le-wm V0 predictor (6 blocks, 16 heads), REAL quantized checkpoint weights
[prover] inputs  z_history [3x192], action embedding [3x192]
[prover]   source  real PushT observation: 3 frames from lewm.gif -> ViT encoder -> projector (action is a stand-in)
[prover] infer   exact integer forward pass in 37 ms
[prover]   z_next[..6] [-19, 41, -30, 1, 2, 12]  (predicted next-latent head)
[verifier] ACCEPT  in 25 ms
[verifier] tamper  forged matmul op Some(2) -> REJECT FreivaldsCheckFailed { op_id: 2 }
```

This is the most end-to-end path: a **real PushT observation** is encoded through
the checkpoint's own ViT encoder + projector into a real latent history, which the
real predictor then consumes. The export (download, encode, quantize, commit) is
the trusted offline step; the prover runs the exact integer inference and the
no_std verifier audits it. The action is a stand-in (the GIF has no action labels),
and the full image-encoder pass is the float reference (P4-deferred for proving).

## 3. The tiny two-party handoff (teaching)

```bash
docker compose --profile compact up --build prover verifier
```

A small built-in model where a `prover` container writes a proof to a shared
volume and a `verifier` container accepts it, then forges a matmul and rejects it.
Good for seeing the prover/verifier split; not the real model.

## Run it without Docker

```bash
cargo run -p pwm-testkit --bin pwm --release -- prove-predictor   # real architecture, synthetic
cargo run -p pwm-testkit --bin pwm --release -- prove-predictor <bundle>  # real checkpoint weights
cargo run -p pwm-testkit --bin pwm --release                      # the tiny compact story
```

To produce the real-checkpoint bundle locally (with real observation latents):

```bash
pip install torch numpy pillow
# download weights.pt from the model page and a PushT GIF, then
LEWM_WEIGHTS=weights.pt LEWM_GIF=path/to/lewm.gif \
  python crates/pwm-export/python/scripts/export_lewm_v0.py \
  /tmp/lewm_pred_proj.json /tmp/lewm_predictor.json
cargo run -p pwm-testkit --bin pwm --release -- prove-predictor /tmp/lewm_predictor.json
```

Omit `LEWM_GIF` to use synthetic input latents instead of encoding real frames.

## What is proven

The proof attests the exact integer (quantized) relation of the committed model:
every fixed-weight matmul is Freivalds-checked (`v.x == r.z`, error `<= 1/p` with
`p = 2^61-1`), and the attention dot products, softmax, GELU, SiLU, LayerNorm, and
residuals are recomputed exactly. It does not claim float/PyTorch equivalence. The
quantization here keeps activations int8 throughout; per-tensor activation-scale
calibration for float-faithful outputs is a further refinement.
