# The demo: the whole scheme as a challenge game

This plays ProvableWorldModel end to end on your machine. A prover runs a real
world-model predictor step in exact integer arithmetic, commits to its execution
trace, and hands a verifier a proof. The verifier replays the Fiat-Shamir
challenge, Freivalds-checks every matmul, exactly recomputes the attention,
softmax, GELU, and residuals, and accepts. Then someone forges a single matmul
output, and the verifier rejects it. No GPU, no floating point, no network.

The model is the le-wm predictor architecture: an action embedding conditions a
self-attention block and a GELU feed-forward, each with a residual, over a latent
history, predicting the next latent. It runs as a compact instance (`dim = 2`,
`history = 2`, one head, `mlp = 4`). The full le-wm V0 predictor runs at
`latent_dim = 192`, depth 6, 16 heads, which the exporter ingests and quantizes
(see the optional `real` profile below). The weights are a fixed synthesized
instance, since le-wm V0 is `pretrained: false` and the proof attests the exact
quantized relation regardless of the weights.

## Run it

```bash
docker compose up --build
```

or:

```bash
./demo/run.sh
```

You will see two parties:

```
prover-1    | [prover] model   le-wm action-conditioned predictor block (self-attention + GELU FFN + residuals)
prover-1    | [prover] config  dim=2, history=2, heads=1, action_dim=2, mlp=4
prover-1    | [prover] infer   exact integer forward pass in 0.027 ms
prover-1    | [prover]   z_history [1, 0, 0, 1]  action [1, 0]
prover-1    | [prover]   z_next    [3905, 1802]  (predicted next latent)
prover-1    | [prover] trace   15 ops, block_root 99ba60b3...
prover-1    | [prover] wrote   proof (652 bytes) to /shared/artifact.bin
verifier-1  | [verifier] challenge  replayed the Fiat-Shamir transcript, derived Freivalds r for 7 linear ops
verifier-1  | [verifier] ACCEPT     in 0.048 ms   z_next [3905, 1802]
verifier-1  | [verifier] tamper     forged the output of matmul op 4 (a fake projection result)
verifier-1  | [verifier] REJECT     FreivaldsCheckFailed { op_id: 4 }
```

The `prover` service runs `pwm prove`: it runs the predictor, logs the inference,
and writes the proof (the input values plus the claimed output of every op) to a
shared volume. The `verifier` service runs `pwm audit`: it loads the proof,
reconstructs the public predictor graph, replays the Fiat-Shamir transcript to
derive the Freivalds challenge, accepts the honest proof, then forges one matmul
output and shows the typed rejection.

## What each step means

| Step | What happens | Why it is sound |
|---|---|---|
| infer | The prover runs the exact integer predictor forward pass, recording every matmul accumulator and op into a trace, and Merkle-commits it. | The committed trace is fixed before any challenge exists. |
| challenge | The verifier replays the Fiat-Shamir transcript to derive `r`, forms `v = rᵀW` from the committed weights, and checks `v·x == r·z` for each linear. | A wrong accumulator `z != Wx` passes with probability at most `1/p` (`p = 2⁶¹−1`). |
| accept | Every Freivalds check passes and the attention, softmax, GELU, and residuals recompute exactly. | Nothing the relation depends on is left unchecked. |
| tamper | One matmul output is bumped by 1, a forged projection result. | `z` no longer equals `Wx`. |
| reject | `v·x != r·z`, so the verifier returns `FreivaldsCheckFailed`. | The forgery is caught with overwhelming probability. |

## Run it without Docker

```bash
cargo run -p pwm-testkit --bin pwm --release            # the full story
cargo run -p pwm-testkit --bin pwm --release -- audit <proof>   # verifier side
cargo run -p pwm-testkit --bin pwm --release -- --json         # machine-readable
```

`pwm demo` plays infer, challenge, accept, tamper, and reject in one process.

## Optional: export a real le-wm model (heavy)

The default demo proves the predictor architecture so the image stays tiny and
runs offline. To run the real le-wm export pipeline (PyTorch, multi-GB image), use
the opt-in profile:

```bash
docker compose --profile real up --build real-export
```

This builds the real le-wm V0 subgraph (`ARPredictor`/`Embedder`/`MLP`), loads it
through `torch.load`, quantizes every linear, folds the BatchNorm, and emits the
committed manifest. It is the export side (quantize, fold, commit), not the Rust
prover. It needs no pretrained checkpoint: le-wm V0 is `pretrained: false`, so a
fresh instance is faithful to the V0 config.
