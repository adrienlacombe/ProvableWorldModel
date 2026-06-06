#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real ViT-Tiny/14 pixel encode + E-208 bundle (backlog D-804/E-208).

le-wm's V0 encoder is `vit_hf(size=tiny, patch_size=14, pretrained=false)` — a HF
ViT trained from scratch (V0 uses **no** DINO weights; that is the V3 option). We
instantiate the real HF ViT (random init, faithful to `pretrained: false`), run a
real forward on an image batch, and feed the real patch latents into the repo's
E-208 P2 bundle builder. The pixel-encoder *proof* relation (patch-embed Freivalds
+ exact attention) is checked in `pwm-verifier/tests/pixel.rs`.

Run:
    pip install torch transformers numpy
    python scripts/encode_vit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import ViTConfig, ViTModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pwm_export import data_adapter

EMBED, HIST, ACTION_DIM = 192, 3, 4


def main() -> None:
    torch.manual_seed(0)
    # ViT-Tiny/14: hidden 192, heads 3, mlp 768; few layers for a fast real forward.
    cfg = ViTConfig(hidden_size=EMBED, num_hidden_layers=4, num_attention_heads=3,
                    intermediate_size=768, patch_size=14, image_size=70,
                    num_channels=3, qkv_bias=True)
    vit = ViTModel(cfg).eval()
    print(f"real HF ViT-Tiny/14: {sum(p.numel() for p in vit.parameters()):,} params, "
          f"{cfg.num_hidden_layers} layers")

    # D-804: real pixel encode -> patch latents.
    images = torch.randn(HIST, 3, 70, 70)
    with torch.no_grad():
        latents = vit(images).last_hidden_state  # [HIST, 1+patches, 192]
    print(f"D-804: encode {tuple(images.shape)} -> latents {tuple(latents.shape)}")

    # E-208: real latents -> committed P2 bundle.
    z_history = latents[:, 0, :].numpy()          # CLS per frame
    bundle = data_adapter.build_p2_bundle(
        z_history=z_history,
        action=torch.randn(HIST, ACTION_DIM).numpy(),
        z_next=latents[-1, 0, :].numpy(),
        goal=z_history[0],
        candidate_actions=torch.randn(8, 5, ACTION_DIM).numpy(),
    )
    assert bundle["latent_dim"] == EMBED
    assert all(-128 <= v <= 127 for v in bundle["z_history"]["data"])
    print(f"E-208: bundle latent_dim={bundle['latent_dim']} "
          f"candidates={bundle['num_candidates']} horizon={bundle['horizon']} "
          f"z_history int8 in [{min(bundle['z_history']['data'])}, "
          f"{max(bundle['z_history']['data'])}]")
    print("\n✅ real ViT pixel encode + E-208 bundle ran end-to-end.")


if __name__ == "__main__":
    main()
