# pwm-export (Python side)

The offline, trusted export pipeline that turns a [le-wm](https://github.com/lucas-maes/le-wm)
checkpoint into a committed quantized manifest + golden vectors, and prepares
proving inputs from [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel).

| Module | Backlog | Needs |
|---|---|---|
| `canonical.py` | E-205 / E-207 bridge | nothing (pure stdlib) — **unit-tested for byte-identical parity with the Rust verifier** |
| `export.py` | E-201 ingest, E-202 quantize, E-203 BatchNorm fold | `torch` + a le-wm checkpoint |
| `data_adapter.py` | E-208 | `stable-worldmodel` + a checkpoint |

The crux — that a Python-exported manifest produces the same `model_commitment` /
`quantization_commitment` the Rust prover and verifier compute — is handled by
`canonical.py` and verified without any heavy dependency:

```bash
python3 crates/pwm-export/python/tests/test_canonical_parity.py
```

`export.py` and `data_adapter.py` implement the checkpoint/data logic on top of
that bridge; they require the model assets (PyTorch, the le-wm checkpoint, a GPU
for encoding, stable-worldmodel) to actually run, which is why their
checkpoint-dependent drivers are exercised in a GPU environment rather than in CI.
The reference hexes in the parity test are regenerated from the Rust
`crates/pwm-export/src/manifest.rs` sample if the serialization ever changes.
