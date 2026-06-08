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
checkpoint, quantizes the full 192-dim V0 subgraph, and proves the predictor with
the **real quantized weights**. Heavy: it pulls a PyTorch image and downloads a
~70 MB checkpoint the first time.

```
[export] checkpoint  quentinll/lewm-pusht (Hugging Face, MIT)
[export] config      latent_dim=192, history=3, depth=6, heads=16, dim_head=64, mlp_dim=2048
[export] quantize    34 linears -> 11,705,856 int8 params (power-of-two scales)
[export] commitments (Blake2s-256): model / quantization / graph / weights_root
[prover] model   le-wm V0 predictor (6 blocks, 16 heads), REAL quantized checkpoint weights
[prover] source  quentinll/lewm-pusht (Hugging Face, MIT), int8-quantized V0 subgraph
[prover] inputs  z_history [3x192], action embedding [3x192]  (committed quantized latents)
[prover] infer   exact integer forward pass in 28 ms
[prover]   z_next[..6] [37, 0, -7, 55, -39, -17]  (predicted next-latent head)
[verifier] ACCEPT  in 25 ms
[verifier] tamper  forged matmul op Some(2) -> REJECT FreivaldsCheckFailed { op_id: 2 }
```

The export is the trusted offline step (it ingests the checkpoint, quantizes, and
commits). The prover then runs the exact integer inference and the no_std verifier
audits it. The input latents are committed stand-ins; real latents would come from
the image encoder on an observation (the encoder is P4-deferred).

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

To produce the real-checkpoint bundle locally:

```bash
pip install torch numpy
# download weights.pt from the model page, then
LEWM_WEIGHTS=weights.pt python crates/pwm-export/python/scripts/export_lewm_v0.py \
  /tmp/lewm_pred_proj.json /tmp/lewm_predictor.json
cargo run -p pwm-testkit --bin pwm --release -- prove-predictor /tmp/lewm_predictor.json
```

## What is proven

The proof attests the exact integer (quantized) relation of the committed model:
every fixed-weight matmul is Freivalds-checked (`v.x == r.z`, error `<= 1/p` with
`p = 2^61-1`), and the attention dot products, softmax, GELU, SiLU, LayerNorm, and
residuals are recomputed exactly. It does not claim float/PyTorch equivalence. The
quantization here keeps activations int8 throughout; per-tensor activation-scale
calibration for float-faithful outputs is a further refinement.
