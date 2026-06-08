# Real-model export scripts

End-to-end drivers that run the export pipeline against the **actual** le-wm /
ViT torch modules (not the synthetic fixtures the unit tests use). They need
heavy, optional deps, so they live here as scripts rather than CI tests; the
torch-free cores they call are unit-tested in `../tests/`.

le-wm V0 uses `pretrained: false` (encoder trained from scratch), so a random
init is faithful to the V0 config, and no pretrained checkpoint is needed to
exercise the export or the proof relations, which attest the *quantized* model
regardless of weight values. Point `export_lewm.py` at a real trained checkpoint
with `LEWM_CKPT=...`.

## `export_lewm.py`: E-201/202/203/205 (real le-wm export)

```bash
pip install torch einops numpy
git clone https://github.com/lucas-maes/le-wm
LEWM_PATH=./le-wm python scripts/export_lewm.py
```

Builds the real le-wm V0 subgraph (`ARPredictor`/`Embedder`/`MLP`), saves +
reloads a checkpoint via `torch.load`, then: validates the spec §2 dims
(192/3/6/16/64/2048) on the real module, quantizes all 34 Freivalds linears
(int8 + int32 MAC bound), folds the `pred_proj` BatchNorm1d into the preceding
Linear (exact, ≈2e-15), and emits the committed manifest. Verified output:

```
E-201: V0 dims (192, 3, 6, 16, 64, 2048) validated; 34 linears extracted
E-202: quantized 34 weights (int8 + MAC < int32)
E-203: fold == Linear∘BN(eval), max err = 2.00e-15
E-205: manifest commitments: model/quantization/graph/weights_root
✅ real le-wm V0 export ran end-to-end.
```

## `encode_vit.py`: D-804/E-208 (real ViT encode + P2 bundle)

```bash
pip install torch transformers numpy
python scripts/encode_vit.py
```

Instantiates the real HF ViT-Tiny/14 (faithful to `pretrained: false`), runs a
real pixel encode, and feeds the real patch latents into the E-208 P2 bundle
builder. The pixel-encoder *proof* relation is checked in
`crates/pwm-verifier/tests/pixel.rs`. Verified output:

```
D-804: encode (3, 3, 70, 70) -> latents (3, 26, 192)
E-208: bundle latent_dim=192 candidates=8 horizon=5 z_history int8 in [-92, 89]
✅ real ViT pixel encode + E-208 bundle ran end-to-end.
```
