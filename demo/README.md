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
  pipeline   checkpoint -> quantize -> commit -> encode -> infer -> prove -> verify

[stage 1/5] LOAD  the committed quantized world model
  ├ model    le-wm V0 predictor (6 blocks, 16 heads), synthetic weights (pass a bundle for real)
  ├ config   dim=192, history=3, heads=16, dim_head=64, mlp=2048, depth=6
  ├ tensors  30 int8 weight matrices, 4 committed table(s)
  │          per block: qkv[3072x192] out[192x1024] fc1[2048x192] fc2[192x2048] adaln[1152x192]
  ├ params   10,764,288 int8 weights  (10.27 MiB on the wire, 1 byte each)
  ├ commit   weights_root 227e79e5...   model 3ab3d1f6...
  ├ inputs   z_history [3x192], action [3x192]  synthetic quantized latents
  └          z[..6] [-2, -1, 0, 1, 2, -2]   action[..6] [-1, 0, 1, -1, 0, 1]

[stage 2/5] INFER  exact integer forward pass (the world model runs)
  ├ graph    2,437 ops over the named-buffer block DAG
  ├ ops      1,389 slice · 373 concat · 234 requant · 192 matmul · 96 softmax · 75 layernorm · 30 batched-linear · ...
  ├ compute  32.40 M multiply-accumulates, exact integer  (no float, no GPU)
  ├ latency  forward pass in 49.0 ms  (49.7 Kop/s, 661 MMAC/s)
  └ z_next   [-2, -1, 0, 1, 2, -2]  (predicted next-latent head, from the real forward pass)

[stage 3/5] COMMIT  bind the execution to a Fiat-Shamir transcript
  ├ witness  689,760 claimed op outputs (5.26 MiB)  trace_root 56bc38fc...
  └ bind     absorbed model + inputs + trace, then squeezed the Freivalds r (non-adaptive)

[stage 4/5] VERIFY  no_std, float-free, never re-runs the model
  ├ challenge derived the Freivalds r for 30 linear projections
  ├ checks   Freivalds v·x == r·z  (soundness ≤ 1/p, p = 2⁶¹−1; union over the checks ~2⁻⁴⁴)
  └ verdict  ACCEPT  in 28.6 ms  (1.7x faster than proving; audits arithmetic only)

[stage 5/5] TAMPER  forge one matmul output
  └ forged matmul op 2 -> REJECT FreivaldsCheckFailed { op_id: 2 }  (caught)

metrics  infer 49.0 ms · verify 28.6 ms · 10.27 MiB int8 model · 32.40 M MAC · 2,437 ops · ACCEPT
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

Stage 1/2, the **export service** (PyTorch), loads the real checkpoint and logs it:

```
ProvableWorldModel  export the real le-wm checkpoint into the prover
  pipeline   load -> extract V0 -> fold BN -> quantize int8 -> commit -> encode inputs -> bundle

[LOAD]      checkpoint  quentinll/lewm-pusht  (weights.pt, 70.0 MiB, Hugging Face MIT)
            state_dict  loaded; float32 params extracted in 0.4 s
[EXTRACT]   action_encoder -> predictor x6 -> pred_proj   (latent=192, depth=6, heads=16, mlp=2048)
[QUANTIZE]  34 matrices -> 11,705,856 int8 params   (float32 44.7 MiB -> int8 11.2 MiB, 4.0x smaller)
[COMMIT]    model_commitment / quantization_commitment / graph_commitment  (Blake2s-256)
[ENCODE]    real PushT expert episode (lerobot/pusht): 3 frames @ frameskip 5 -> ViT encoder; real 2D action
```

Stage 2/2, the **prover** (pure Rust), runs the exact-integer inference on those
real weights and the no_std verifier audits it (abridged):

```
[stage 1/5] LOAD  the committed quantized world model
  ├ model    le-wm V0 predictor (6 blocks, 16 heads), REAL quantized checkpoint weights
  ├ source   quentinll/lewm-pusht (Hugging Face, MIT), int8-quantized V0 subgraph
  ├ params   10,764,288 int8 weights  (10.27 MiB on the wire, 1 byte each)
  └ inputs   real PushT expert episode (lerobot/pusht): 3 frames @ frameskip 5 -> ViT encoder; real 2D action

[stage 2/5] INFER  exact integer forward pass (the world model runs)
  ├ compute  32.40 M multiply-accumulates, exact integer  (no float, no GPU)
  ├ latency  forward pass in 49.0 ms  (49.7 Kop/s, 661 MMAC/s)
  └ z_next   [11, 55, 32, -73, -57, 13]  (predicted next-latent head, from the real forward pass)

[stage 4/5] VERIFY  no_std, float-free, never re-runs the model
  └ verdict  ACCEPT  in 28.6 ms  (1.7x faster than proving; audits arithmetic only)

[stage 5/5] TAMPER  forge one matmul output
  └ forged matmul op 2 -> REJECT FreivaldsCheckFailed { op_id: 2 }  (caught)

metrics  infer 49.0 ms · verify 28.6 ms · 10.27 MiB int8 model · 32.40 M MAC · 2,437 ops · ACCEPT
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
