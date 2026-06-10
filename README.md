# ProvableWorldModel

[![CI](https://github.com/AbdelStark/ProvableWorldModel/actions/workflows/ci.yml/badge.svg)](https://github.com/AbdelStark/ProvableWorldModel/actions/workflows/ci.yml)
[![Real E2E](https://github.com/AbdelStark/ProvableWorldModel/actions/workflows/real-e2e.yml/badge.svg)](https://github.com/AbdelStark/ProvableWorldModel/actions/workflows/real-e2e.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-stable%20(MSRV%201.85)-orange.svg)](rust-toolchain.toml)
[![verifier](https://img.shields.io/badge/verifier-no__std%20%C2%B7%20float--free-5eead4.svg)](crates/pwm-verifier)

Cryptographic proof that a quantized world model planned exactly as claimed.

ProvableWorldModel lets anyone verify, on a CPU with no floating point, that a
committed quantized [LeWorldModel](https://github.com/lucas-maes/le-wm) (an
action-conditioned JEPA world model) was run exactly as specified: rolling out
candidate action plans, scoring them against a goal, and picking the best. It
Freivalds-checks the large matmuls and exactly re-executes everything else, all
bound by Merkle commitments and a Fiat-Shamir transcript. No proving circuit, no
arithmetization, stable Rust.

It adapts the [CommitLLM](https://github.com/lambdaclass/CommitLLM)
commit-and-audit scheme from language models to a world model, and because the
model runs in exact integer fixed point rather than bf16 on a GPU, it closes the
one honest hole CommitLLM leaves open: non-reproducible attention.

**Interactive explainer:** https://abdelstark.github.io/ProvableWorldModel/

### What it does not claim

It proves an exact integer arithmetic relation. For predictor bundles, the export
also carries a float reference output and a measured tolerance, and the demo CLI
checks the verified integer output against that bound. The verifier still does
not prove the full PyTorch program, the physical truth of the predictions, zero
knowledge, or that the checkpoint bytes were exported honestly. Public claims
must be a subset of the proven statement.

## Quickstart

Prove the real le-wm predictor architecture (192-dim, 16 heads, 6 blocks) in exact
integer arithmetic, then watch a forged matmul get rejected.

```bash
git clone https://github.com/AbdelStark/ProvableWorldModel
cd ProvableWorldModel
docker compose up --build
```

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
  ├ graph    2,437 ops over the named-buffer block DAG  (AdaLN-zero, 16-head attention, GELU FFN, gated residuals)
  ├ ops      1,389 slice · 373 concat · 234 requant · 192 matmul · 96 softmax · 75 layernorm · 30 batched-linear · ...
  ├ compute  32.40 M multiply-accumulates, exact integer  (no float, no GPU)
  ├ latency  forward pass in 49.0 ms  (49.7 Kop/s, 661 MMAC/s)
  └ z_next   [-2, -1, 0, 1, 2, -2]  (predicted next-latent head, from the real forward pass)

[stage 3/5] COMMIT  bind the execution to a Fiat-Shamir transcript
  ├ witness  689,760 claimed op outputs (5.26 MiB)  trace_root 56bc38fc...
  └ bind     absorbed model + inputs + trace, then squeezed the Freivalds r (non-adaptive)

[stage 4/5] VERIFY  no_std, float-free, commitment-bound, never re-runs the model
  ├ binding  recomputed model, quantization, planner, input + output commitments
  ├ challenge derived the Freivalds r for 30 linear projections
  ├ checks   Freivalds v·x == r·z  (soundness ≤ 1/p, p = 2⁶¹−1; union over the checks ~2⁻⁴⁴)
  │          exact recompute of attention, softmax, GELU, LayerNorm, residuals
  ├ faith    max |int - float| <= bundle tolerance  (offline reference from the export bundle)
  └ verdict  ACCEPT  in 28.6 ms  (1.7x faster than proving; commitments + arithmetic audited)

[stage 5/5] TAMPER  forge one matmul output
  └ forged matmul op 2 -> REJECT FreivaldsCheckFailed { op_id: 2 }  (caught)

metrics  infer 49.0 ms · verify 28.6 ms · 10.27 MiB int8 model · 32.40 M MAC · 2,437 ops · ACCEPT
```

That default is the real architecture with synthetic weights (fast, no checkpoint).
For the **real pretrained checkpoint** end to end (download, quantize, prove the
real weights):

```bash
docker compose --profile real up --build export predictor-real   # or ./demo/run-real.sh
```

The same real checkpoint path is also covered by the `Real E2E` GitHub Actions
workflow. It builds the exporter and prover images, downloads the
`quentinll/lewm-pusht` checkpoint plus a `lerobot/pusht` episode, exports the
bundle, runs `pwm --json prove-predictor /shared/lewm_predictor.json`, checks that
the proof is accepted and export-bound, and uploads the logs and bundles. It is a
separate workflow that runs on every push, every PR update, on demand, and
weekly because it depends on external assets and a large PyTorch image.

Without Docker:

```bash
cargo run -p pwm-testkit --bin pwm --release -- prove-predictor   # real architecture, synthetic
cargo run -p pwm-testkit --bin pwm --release                      # the tiny compact story
cargo test --workspace                                            # the accept and reject suites
```

See [demo/README.md](demo/README.md) for all three demo modes and the real-checkpoint steps.

### Prove the real pretrained checkpoint

The exporter ingests the real [`quentinll/lewm-pusht`](https://huggingface.co/quentinll/lewm-pusht)
checkpoint, takes a real PushT expert episode from
[`lerobot/pusht`](https://huggingface.co/datasets/lerobot/pusht), encodes the real
observation frames through the checkpoint's own ViT encoder + projector and the
real expert action through the action encoder, quantizes the full 192-dim V0
subgraph, folds the BatchNorm, and commits the manifest.

```bash
pip install torch numpy pillow pyarrow imageio imageio-ffmpeg
# download weights.pt from the model page, then
LEWM_WEIGHTS=weights.pt LEWM_LEROBOT=1 \
  python crates/pwm-export/python/scripts/export_lewm_v0.py \
  /tmp/lewm_pred_proj.json /tmp/lewm_predictor.json
```

The full **6-block, 16-head attention predictor** then proves and verifies in the
Rust prover on the real weights and the real observation and action:

```bash
cargo run -p pwm-testkit --bin pwm --release -- prove-predictor /tmp/lewm_predictor.json
```

The export stage loads the real checkpoint with PyTorch and logs exactly what it
did (used by the `--profile real` Docker path; here run locally):

```
ProvableWorldModel  export the real le-wm checkpoint into the prover
  pipeline   load -> extract V0 -> fold BN -> quantize int8 -> commit -> encode inputs -> bundle

[LOAD]      checkpoint  quentinll/lewm-pusht  (weights.pt, 70.0 MiB, Hugging Face MIT)
            state_dict  loaded; float32 params extracted in 0.4 s
[EXTRACT]   action_encoder -> predictor x6 -> pred_proj   (latent=192, depth=6, heads=16, mlp=2048)
[QUANTIZE]  34 matrices -> 11,705,856 int8 params   (float32 44.7 MiB -> int8 11.2 MiB, 4.0x smaller)
[COMMIT]    model_commitment / quantization_commitment / graph_commitment  (Blake2s-256)
[ENCODE]    real PushT expert episode (lerobot/pusht): 3 frames @ frameskip 5 -> ViT encoder; real 2D action
[CALIBRATE] real SiLU/GELU/inverse-sqrt/softmax-exp tables + measured predictor tolerance
```

The prover then runs the exact-integer inference on those real weights and audits it:

```
[stage 1/5] LOAD  the committed quantized world model
  ├ model    le-wm V0 predictor (6 blocks, 16 heads), REAL quantized checkpoint weights
  ├ source   quentinll/lewm-pusht (Hugging Face, MIT), int8-quantized V0 subgraph
  ├ params   10,764,288 int8 weights  (10.27 MiB on the wire, 1 byte each)
  └ inputs   real PushT expert episode (lerobot/pusht): 3 frames @ frameskip 5 -> ViT encoder; real 2D action

[stage 2/5] INFER  exact integer forward pass (the world model runs)
  ├ graph    2,437 ops over the named-buffer block DAG
  ├ compute  32.40 M multiply-accumulates, exact integer  (no float, no GPU)
  ├ latency  forward pass in 49.0 ms  (49.7 Kop/s, 661 MMAC/s)
  └ z_next   [11, 55, 32, -73, -57, 13]  (predicted next-latent head, from the real forward pass)

[stage 4/5] VERIFY  no_std, float-free, commitment-bound, never re-runs the model
  ├ faith    max |int - float| <= bundle tolerance  (offline reference from the export bundle)
  └ verdict  ACCEPT  in 28.6 ms  (1.7x faster than proving; commitments + arithmetic audited)

[stage 5/5] TAMPER  forge one matmul output
  └ forged matmul op 2 -> REJECT FreivaldsCheckFailed { op_id: 2 }  (caught)

metrics  infer 49.0 ms · verify 28.6 ms · 10.27 MiB int8 model · 32.40 M MAC · 2,437 ops · ACCEPT
```

A real expert episode (consistent observation and action) goes through the
checkpoint's own encoders to produce the latent history and action embedding the
predictor consumes. The encode is the trusted offline step (the image encoder is
P4-deferred for proving), and the predictor step is what the no_std verifier
audits. The bundle also carries the export-computed `weights_root` over the 30
proven block tensors; the prover binds that carried value into the model
commitment, so the verifier rejects unless the proven weights reproduce the
export's commitment bit-for-bit (bundle ⇄ export bound; checkpoint ⇄ export
trusted). The same bundle carries the calibrated SiLU, GELU, inverse-sqrt, and
softmax-exp tables plus the float predictor output on the same encoded input; the
CLI checks the verified integer `z_next` against the bundle's measured tolerance.
The action is the real 2D expert control plus the agent state; the full
le-wm 10-dim action layout lives in the 13 GB lewm-pusht dataset.

The predictor is built as the real le-wm architecture over the named-buffer block
DAG: per ConditionalBlock, AdaLN-zero conditioning (`SiLU(c) -> Linear -> chunk6`)
modulates a 16-head self-attention sub-block (per head: `Q.Kᵀ -> softmax ->
prob.V`) and a GELU feed-forward, each with a gated residual, with the multi-head
reshapes expressed as Slice/Concat. The verifier Freivalds-checks every projection
and exactly recomputes the attention, softmax, GELU, LayerNorm, and residuals.
`prove-lewm` proves the smaller `pred_proj` head on its own.

## How it works

```text
+=== PROVER  (trusted / offline) ============================================+
|                                                                            |
|  +----------+  +----------+  +----------+  +------------------+            |
|  |CHECKPOINT|  |  EXPORT  |  |  ENCODE  |  |      COMMIT      |            |
|  | le-wm    |->| quantize |->| PushT    |->| canonicalize     |            |
|  | JEPA     |  | to exact |  | frames ->|  | latent +         |            |
|  | weights +|  | int graph|  | ViT enc +|  | action;          |            |
|  | config   |  | fold     |  | projector|  | check the        |            |
|  | (lewm-   |  | BatchNorm|  | -> latent|  | manifest         |            |
|  | pusht)   |  | manifest |  | history; |  | hash vs          |            |
|  |          |  | + LUTs   |  | act->emb |  | model            |            |
|  +----------+  +----------+  +----------+  +---------+--------+            |
|     src: quentinll/lewm-pusht   src: lerobot/pusht   |                     |
|  +---------------------------------------------------+                     |
|  | PROVE  (z_next = predictor(latent, action))       |                     |
|  | exact integer inference, full predictor:          |                     |
|  | 192-dim, hist 3, depth 6, 16 heads x 64,          |                     |
|  | mlp 2048; 2,437 ops: AdaLN-zero, MHA,             |                     |
|  | GELU FFN, gated residuals;                        |                     |
|  | trace -> Merkle commit -> Fiat-Shamir r           |                     |
|  +-------------------------+-------------------------+                     |
|                            |                                               |
+============================+===============================================+
                             v
         +-------------------+------------------+
         | ARTIFACT  (the TRUST BOUNDARY)       |
         | AuditArtifact = commitments + trace  |  the ONLY thing that
         | + claimed z_next                     |  crosses prover -> verifier
         +-------------------+------------------+
                             |
+============================+===============================================+
|                            v        VERIFIER  (no_std / float-free)        |
|  +------------------------------------------------------------+            |
|  | VERIFY  (NEVER re-runs the model; audits bindings + trace) |            |
|  | replay transcript; Freivalds-check every fixed matmul:     |            |
|  | v.x == r.z  because  rT(W x) = (rT W) x                    |            |
|  | soundness err <= 1/p, p = 2^61-1;                          |            |
|  | union over ~1e5 checks stays ~ 2^-44                       |            |
|  | recompute attn, softmax, GELU, LayerNorm, residuals;       |            |
|  | check rollout, cost, argmin                                |            |
|  +--------------+------------------------------+--------------+            |
|                 | all checks pass              | any value mutated         |
|                 v                              v                           |
|   +-------------------------+     +-----------------------------------+    |
|   | VERDICT: ACCEPT (Ok)    |     | VERDICT: REJECT (VerifyError)     |    |
|   | every check passed      |     | forge one matmul output ->        |    |
|   |                         |     | FreivaldsCheckFailed              |    |
|   +-------------------------+     +-----------------------------------+    |
|                                                                            |
+============================================================================+

exact integer fixed point makes attention reproducible: the one hole
CommitLLM leaves open is closed here. The verifier audits the committed
statement and the exact arithmetic trace.
```

The prover runs the model normally and commits to its execution trace. The
verifier never re-runs the model. For each fixed weight matrix it precomputes
`v = rᵀW` once and checks `v·x == r·z` per use, which holds because
`rᵀ(Wx) = (rᵀW)x`. The cheap, deterministic ops are recomputed exactly from
committed data: attention dot products, requant, the nonlinear table reads, the
cost, and the argmin.

### Soundness

A wrong accumulator `z ≠ Wx` passes one random Freivalds check with probability at
most `1/p`, with `p = 2⁶¹ − 1`. A union bound over `N ≈ 10⁵` checked instances
stays around `2⁻⁴⁴`. The same `W` is reused across every candidate, rollout step,
and transformer block, so `v = rᵀW` is computed once per weight matrix and reused
about `S × horizon` times: that reuse is the whole point.

That `1/p` bound is only the random-challenge escape. The Freivalds equation lives
in `F_p`, so it proves `z ≡ Wx (mod p)`, not `z = Wx` over the integers, and a
prover who adds a multiple of `p` to an accumulator passes *every* challenge. So the
verifier range-checks every input and claimed accumulator to the single-M31
envelope (`|·| ≤ 2³⁰ − 1`, far above any honest int8 dot product); the honest
`Wx` and a `±k·p` forgery cannot both fit one length-`p` window, so the forgery is
rejected. An executed reject test (`+p` on a real accumulator) and a soundness
mutation gate pin this property.

Mutating any load-bearing value changes a commitment or fails an exact check, and
the proof is rejected. That covers weights, scales, the rounding mode, tables, op
order, planner config, public inputs, claimed outputs, and every trace cell.

## What gets proven

| Tier | Claim | Status |
|---|---|---|
| **P0** | One predictor step: `z_next = PredProj(ARPredictor(z_hist, ActEnc(actions)))`. | implemented and tested; the full 192-dim 6-block 16-head predictor proves and verifies with real checkpoint weights |
| **P1** | Autoregressive rollout: each step feeds the next; the recurrence wiring is checked. | implemented and tested |
| **P2** | Fixed-candidate planning (V0): roll out all `S` candidates, score by goal MSE, prove the selected is the argmin. | implemented and tested |
| P3 | Full CEM planner (sampling, elites, distribution updates). | deferred |
| P4 | Pixel to plan, including the ViT encoder. | deferred |

What is implemented and tested today (175 tests, including an executed mod-`p`
accumulator forgery that the verifier now rejects and a real soundness mutation
gate): the commit-and-audit protocol (Freivalds with the accumulator range guard,
exact replay, Merkle commitments, Fiat-Shamir), the full predictor op vocabulary
(attention, AdaLN, GELU and SiLU tables, LayerNorm, residuals, softmax), the
rollout recurrence, the MSE cost, and the argmin with tie-break, each with accept
and reject tests. The exporter ingests the real `quentinll/lewm-pusht` checkpoint
and quantizes the full 192-dim V0 subgraph, and the Rust prover proves and
verifies the full 6-block, 16-head, 192-dim predictor with the real quantized
weights (`pwm prove-predictor`), plus the `pred_proj` head on its own
(`pwm prove-lewm`). The proof attests the exact integer (quantized) relation, and
predictor bundles now include calibrated activation tables plus an offline
float-reference tolerance that the CLI enforces after verification. For P2, all
`S` candidate costs must be proven, not only the winner:
proving only the selected candidate would be unsound.

Binding status, precisely: both the P0 feed-forward statement (`AuditArtifact`) and
the full named-buffer predictor (`RELATION_PREDICTOR`, `pwm_verifier::verify_predictor`)
are **commitment-bound**. In each case the verifier recomputes the model commitment
(architecture + weight Merkle root), the quantization commitment (scales + tables),
and the input/output commitments, and checks them against the public input, so
accepting a proof means the *committed* model ran on the *committed* inputs, not just
some caller-supplied weights. The predictor's Fiat-Shamir transcript binds that public
input and the committed block witness (the block Merkle root, which commits every
claimed accumulator) before any challenge is squeezed, so the challenge is
statement-bound and non-adaptive. A bare `verify_block` still exists for standalone
arithmetic audits; the committed relation is the one to use for a real statement.

The trust boundary, exactly: **bundle ⇄ export is cryptographically bound;
checkpoint ⇄ export is trusted preprocessing.** The export computes a
predictor-scoped `weights_root` over exactly the proven block tensors (in the
prover's canonical `tensor_id` order) and carries it in the bundle; the prover
binds that carried value, not a recomputed one, into the model commitment, so
`verify_predictor` accepts only if the proven weights reproduce the export's
commitment bit-for-bit (`CommitmentMismatch(Model)` otherwise). The artifact alone
therefore certifies *which* weight set was proven, with no out-of-band weights.
What remains trusted is the export step itself: that the quantized tensors
faithfully derive from the `quentinll/lewm-pusht` checkpoint bytes (download,
float-to-int8 quantization, and input encoding are offline preprocessing, outside
the proven relation).

## Architecture

Five small crates, one trust anchor. The verifier depends on neither the exporter
nor any Python or float runtime; it verifies arithmetic only, sharing one
Freivalds and trace implementation with the prover through `pwm-core`.

| Crate | Role |
|---|---|
| [`pwm-core`](crates/pwm-core) | Fields (M31 value, Fp61 audit), fixed-point reference, tensors, Merkle commitments, Fiat-Shamir transcript, the Freivalds check, the trace model. `no_std`. |
| [`pwm-export`](crates/pwm-export) | le-wm checkpoint to quantized integer graph, manifest, golden vectors, the Rust integer reference, the data adapter. |
| [`pwm-prover`](crates/pwm-prover) | Run the reference inference, commit the trace, derive challenges, emit the `AuditArtifact`. |
| [`pwm-verifier`](crates/pwm-verifier) | CPU, `no_std`, float-free. Freivalds-check the linears, recompute the rest, check rollout, cost, argmin. |
| [`pwm-testkit`](crates/pwm-testkit) | Golden vectors, the accept and reject suites, the mutation harness, the `pwm` demo CLI. |

```text
pwm-prover --> pwm-export --> pwm-core <-- pwm-verifier
                                  ^
              pwm-testkit --> (pwm-prover, pwm-verifier, pwm-export, pwm-core)
```

The argmin uniqueness check and the Freivalds probability bound are also formally
verified in Lean 4 (no `sorry`); see [lean/](lean).

## The model

LeWorldModel is a JEPA-style action-conditioned world model: it predicts the next
latent, not pixels. The target V0 subgraph the exporter ingests is
`action_encoder -> predictor -> pred_proj` at `latent_dim = 192`,
`history_size = 3`, depth `6`, `16` heads, `dim_head = 64`, `mlp_dim = 2048`, which
is bit-deterministic in eval mode. The full 192-dim predictor proves and verifies
in the Rust prover with the real quantized weights (`pwm prove-predictor`). The
pixel encoder (ViT-Tiny/14) is deferred to P4; V0 takes latents as inputs.

## Documentation

- [Interactive explainer](https://abdelstark.github.io/ProvableWorldModel/): the visual walkthrough.
- [specs.md](specs.md): the normative specification of the commit-and-audit design.
- [roadmap.md](roadmap.md): the plan, the pivot rationale, and the sequencing.
- [demo/README.md](demo/README.md): the local demo.
- [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
