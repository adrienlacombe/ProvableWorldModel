# SPDX-License-Identifier: Apache-2.0
"""Real le-wm encode: pixels -> latent via the checkpoint's ViT encoder + projector.

Turns real observation frames into the real latent history the predictor consumes,
exactly as le-wm does (`jepa.py`): the ViT encoder's CLS token through the
projector MLP. The ViT forward is reconstructed by hand from the checkpoint's
`encoder.*` weights (no transformers version dependency), and the projector and
action encoder from `projector.*` / `action_encoder.*`. Torch + Pillow; used only
by the real-export path (not in CI).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageSequence

# le-wm uses ImageNet stats for the image preprocessor (utils.get_img_preprocessor).
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _ln(x: torch.Tensor, sd: dict, p: str) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), sd[f"encoder.{p}.weight"], sd[f"encoder.{p}.bias"], 1e-12)


def _lin(x: torch.Tensor, sd: dict, p: str) -> torch.Tensor:
    return x @ sd[f"encoder.{p}.weight"].T + sd[f"encoder.{p}.bias"]


@torch.no_grad()
def vit_cls(sd: dict, pixels: torch.Tensor, heads: int = 3, dim_head: int = 64, depth: int = 12) -> torch.Tensor:
    """Run the checkpoint's ViT encoder on `[B, 3, 224, 224]`, return the CLS token."""
    b = pixels.size(0)
    pe = F.conv2d(
        pixels,
        sd["encoder.embeddings.patch_embeddings.projection.weight"],
        sd["encoder.embeddings.patch_embeddings.projection.bias"],
        stride=14,
    )
    pe = pe.flatten(2).transpose(1, 2)
    cls = sd["encoder.embeddings.cls_token"].expand(b, -1, -1)
    x = torch.cat([cls, pe], dim=1) + sd["encoder.embeddings.position_embeddings"]
    for i in range(depth):
        p = f"encoder.layer.{i}"
        h = _ln(x, sd, f"{p}.layernorm_before")
        q = _lin(h, sd, f"{p}.attention.attention.query").view(b, -1, heads, dim_head).transpose(1, 2)
        k = _lin(h, sd, f"{p}.attention.attention.key").view(b, -1, heads, dim_head).transpose(1, 2)
        v = _lin(h, sd, f"{p}.attention.attention.value").view(b, -1, heads, dim_head).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, -1, heads * dim_head)
        x = x + _lin(a, sd, f"{p}.attention.output.dense")
        h2 = _ln(x, sd, f"{p}.layernorm_after")
        h2 = F.gelu(_lin(h2, sd, f"{p}.intermediate.dense"))
        x = x + _lin(h2, sd, f"{p}.output.dense")
    return _ln(x, sd, "layernorm")[:, 0]


def _mlp(sd: dict, x: torch.Tensor, prefix: str) -> torch.Tensor:
    """le-wm MLP: Linear -> BatchNorm1d(eval) -> GELU -> Linear."""
    h = x @ sd[f"{prefix}.net.0.weight"].T + sd[f"{prefix}.net.0.bias"]
    g = sd[f"{prefix}.net.1.weight"]
    be = sd[f"{prefix}.net.1.bias"]
    m = sd[f"{prefix}.net.1.running_mean"]
    v = sd[f"{prefix}.net.1.running_var"]
    h = (h - m) / torch.sqrt(v + 1e-5) * g + be
    h = F.gelu(h)
    return h @ sd[f"{prefix}.net.3.weight"].T + sd[f"{prefix}.net.3.bias"]


@torch.no_grad()
def encode_action(sd: dict, action: torch.Tensor) -> torch.Tensor:
    """le-wm Embedder: Conv1d(k=1) -> Linear -> SiLU -> Linear. `action` is `[T, A]`."""
    x = action.float().t().unsqueeze(0)  # [1, A, T]
    x = F.conv1d(x, sd["action_encoder.patch_embed.weight"], sd["action_encoder.patch_embed.bias"])
    x = x.squeeze(0).t()  # [T, 10]
    x = x @ sd["action_encoder.embed.0.weight"].T + sd["action_encoder.embed.0.bias"]
    x = F.silu(x)
    return x @ sd["action_encoder.embed.2.weight"].T + sd["action_encoder.embed.2.bias"]


def _prep(im: Image.Image) -> torch.Tensor:
    arr = np.asarray(im.convert("RGB").resize((224, 224))).astype(np.float32) / 255.0
    return (torch.from_numpy(arr).permute(2, 0, 1) - IMNET_MEAN) / IMNET_STD


def load_frames(gif_path: str, n: int) -> tuple[list, int]:
    """`n` evenly spaced RGB frames from a GIF, plus the total frame count."""
    gif = Image.open(gif_path)
    frames = [f.copy() for f in ImageSequence.Iterator(gif)]
    idx = [int(round(i)) for i in np.linspace(0, len(frames) - 1, n)]
    return [frames[i] for i in idx], len(frames)


@torch.no_grad()
def encode_history(sd: dict, gif_path: str, history: int, action_dim: int = 10):
    """Encode `history` real observation frames into a real latent history `[history,
    192]` and an action embedding `[history, 192]`. The GIF has no action labels, so
    the action is a small stand-in; the latents are real. Returns `(emb, act_emb,
    total_frames)` as NumPy arrays."""
    frames, total = load_frames(gif_path, history)
    pixels = torch.stack([_prep(f) for f in frames])
    cls = vit_cls(sd, pixels)
    emb = _mlp(sd, cls, "projector")
    act_emb = encode_action(sd, torch.zeros(history, action_dim))
    return emb.numpy(), act_emb.numpy(), total
