# The demo: the whole scheme as a challenge game

This plays ProvableWorldModel end to end on your machine. A prover runs a real
quantized world-model step in exact integer arithmetic, commits to its execution
trace, and hands a verifier a proof. The verifier draws a secret challenge,
Freivalds-checks every matmul, and accepts. Then someone forges a single matmul
output, and the verifier rejects it. No GPU, no floating point, no network.

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
prover-1    | ✓ prover ran inference and wrote a 3417-byte proof to /shared/artifact.bin
verifier-1  | verifier: challenge
verifier-1  |   ✓ verifier drew a secret challenge r (2 vector(s)); checked v·x == r·z per linear op
verifier-1  |   ✓ ACCEPT  output [6, -2]
verifier-1  | verifier: tamper attempt
verifier-1  |   • forged the accumulator of linear op 100 (a fake matmul result)
verifier-1  |   ✗ REJECT  FreivaldsCheckFailed { op_id: 100 }
verifier-1  | honest proof accepted; forged proof rejected.
```

The `prover` service runs `pwm prove` and writes the `AuditArtifact` to a shared
volume. The `verifier` service runs `pwm audit`: it loads the proof, runs the
verifier-secret Freivalds challenge (the challenge `r` is drawn by the verifier
and never seen by the prover), accepts the honest proof, then forges one
accumulator and shows the typed rejection.

## What each step means

| Step | What happens | Why it is sound |
|---|---|---|
| prove | The prover runs the exact integer reference inference, records every matmul accumulator and op into a trace, Merkle-commits it, and emits the proof. | The committed trace is fixed before any challenge exists. |
| challenge | The verifier draws a secret random `r`, forms `v = rᵀW` from the committed weights, and checks `v·x == r·z` for each linear. | A wrong accumulator `z ≠ Wx` passes with probability at most `1/p` (`p = 2⁶¹−1`). |
| accept | Every Freivalds check passes and every other op (requant, activation, output) recomputes exactly. | Nothing the relation depends on is left unchecked. |
| tamper | One linear accumulator is bumped by 1, a forged matmul result. | `z` no longer equals `Wx`. |
| reject | `v·x != r·z`, so the verifier returns `FreivaldsCheckFailed`. | The forgery is caught with overwhelming probability. |

## Run it without Docker

```bash
cargo run -p pwm-testkit --bin pwm --release            # the full story
cargo run -p pwm-testkit --bin pwm --release -- audit <proof>   # verifier side
cargo run -p pwm-testkit --bin pwm --release -- --json         # machine-readable
```

`pwm demo` plays prove, challenge, accept, tamper, and reject in one process.

## Optional: export a real le-wm model (heavy)

The default demo proves a small built-in model so the image stays tiny and runs
offline. To run the real le-wm export pipeline (PyTorch, multi-GB image), use the
opt-in profile:

```bash
docker compose --profile real up --build real-export
```

This builds the real le-wm V0 subgraph (`ARPredictor`/`Embedder`/`MLP`), loads it
through `torch.load`, quantizes every linear, folds the BatchNorm, and emits the
committed manifest. It needs no pretrained checkpoint: le-wm V0 is configured
`pretrained: false`, so a fresh instance is faithful to the V0 config, and the
proof attests the quantized model whatever its weights are.
