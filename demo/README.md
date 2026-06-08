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
conditioning, 16-head self-attention, GELU feed-forward, residuals, 2,437 ops.
Pure Rust, offline, weights are synthetic.

```
ProvableWorldModel  commit-and-audit over the le-wm world model
  pipeline   checkpoint -> quantize -> commit -> encode -> run -> prove -> verify

[stage 1/4] EXPORT  offline, trusted
  ├ model    le-wm V0 predictor (6 blocks, 16 heads), synthetic weights (pass a bundle for real)
  ├ config   dim=192, history=3, heads=16, dim_head=64, mlp=2048, depth=6
  ├ weights  30 tensors, 10,764,288 int8 params
  ├ inputs   z_history [3x192], action embedding [3x192]
  └ source   synthetic quantized latents

[stage 2/4] PROVE  exact integer inference + commitment
  ├ graph    2,437 ops over the named-buffer block DAG
  │          per block: AdaLN-zero, 16-head attention, GELU FFN, gated residuals
  ├ infer    exact integer forward pass in 131 ms
  └ z_next   [-2, -1, 0, 1, 2, -2]  (predicted next-latent head)

[stage 3/4] VERIFY  no_std, float-free
  ├ challenge replayed the Fiat-Shamir transcript, derived the Freivalds r
  ├ checks   Freivalds v·x == r·z on every projection
  │          exact recompute of attention, softmax, GELU, LayerNorm, residuals
  └ verdict  ACCEPT  in 39 ms

[stage 4/4] TAMPER  forge one matmul output
  └ forged matmul op 2 -> REJECT FreivaldsCheckFailed { op_id: 2 }  (caught)
```

## 2. The real pretrained checkpoint (real weights)

```bash
docker compose --profile real up --build export predictor-real
# or: ./demo/run-real.sh
```

This downloads the real [`quentinll/lewm-pusht`](https://huggingface.co/quentinll/lewm-pusht)
checkpoint, takes a **real PushT expert episode from
[`lerobot/pusht`](https://huggingface.co/datasets/lerobot/pusht)** (a consistent
real observation and action), **encodes the real observation frames through the
checkpoint's own ViT encoder + projector** and the **real expert action through
the action encoder**, quantizes the full 192-dim V0 subgraph, and proves the
predictor on the real weights and the real inputs. Heavy: it pulls a PyTorch image
and downloads a ~70 MB checkpoint the first time.

```
[export] checkpoint  quentinll/lewm-pusht (Hugging Face, MIT)
[export] quantize    34 linears -> 11,705,856 int8 params (power-of-two scales)
[encode] real PushT expert episode (lerobot/pusht): 3 frames @ frameskip 5 -> ViT encoder; real 2D action + agent state

[stage 1/4] EXPORT  offline, trusted
  ├ model    le-wm V0 predictor (6 blocks, 16 heads), REAL quantized checkpoint weights
  ├ source   quentinll/lewm-pusht (Hugging Face, MIT), int8-quantized V0 subgraph
  ├ config   dim=192, history=3, heads=16, dim_head=64, mlp=2048, depth=6
  ├ weights  30 tensors, 10,764,288 int8 params
  ├ inputs   z_history [3x192], action embedding [3x192]
  └ source   real PushT expert episode (lerobot/pusht): 3 frames @ frameskip 5 -> ViT encoder; real 2D action + agent state

[stage 2/4] PROVE  exact integer inference + commitment
  ├ graph    2,437 ops over the named-buffer block DAG
  │          per block: AdaLN-zero, 16-head attention, GELU FFN, gated residuals
  ├ infer    exact integer forward pass in 130 ms
  └ z_next   [11, 55, 32, -73, -57, 13]  (predicted next-latent head)

[stage 3/4] VERIFY  no_std, float-free
  └ verdict  ACCEPT  in 38 ms

[stage 4/4] TAMPER  forge one matmul output
  └ forged matmul op 2 -> REJECT FreivaldsCheckFailed { op_id: 2 }  (caught)
```

This is fully end to end: a **real PushT expert episode** (consistent observation
and action from one rollout) is encoded through the checkpoint's own encoders into
the latent history and action embedding the predictor consumes. The export
(download, encode, quantize, commit) is the trusted offline step; the prover runs
the exact integer inference and the no_std verifier audits it. The action is the
real 2D expert control plus the agent state, normalized to `[-1, 1]`; the remaining
proprio dims (block pose) and the exact le-wm 10-dim layout live in the 13 GB
lewm-pusht dataset, which is impractical to ship in a demo. The full image encoder
is the float reference (P4-deferred for proving). Set `LEWM_GIF=<path>` instead of
the lerobot episode to encode frames from the le-wm rollout GIF with a stand-in
action.

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

To produce the real-checkpoint bundle locally (with real observation + action):

```bash
pip install torch numpy pillow pyarrow imageio imageio-ffmpeg
# download weights.pt from the model page, then
LEWM_WEIGHTS=weights.pt LEWM_LEROBOT=1 \
  python crates/pwm-export/python/scripts/export_lewm_v0.py \
  /tmp/lewm_pred_proj.json /tmp/lewm_predictor.json
cargo run -p pwm-testkit --bin pwm --release -- prove-predictor /tmp/lewm_predictor.json
```

`LEWM_LEROBOT=1` pulls a real lerobot/pusht episode (observation + action). Use
`LEWM_GIF=<path>` instead for le-wm rollout frames with a stand-in action, or omit
both for synthetic inputs.

## What is proven

The proof attests the exact integer (quantized) relation of the committed model:
every fixed-weight matmul is Freivalds-checked (`v.x == r.z`, error `<= 1/p` with
`p = 2^61-1`), and the attention dot products, softmax, GELU, SiLU, LayerNorm, and
residuals are recomputed exactly. It does not claim float/PyTorch equivalence. The
quantization here keeps activations int8 throughout; per-tensor activation-scale
calibration for float-faithful outputs is a further refinement.
