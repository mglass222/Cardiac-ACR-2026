#!/usr/bin/env python
# coding: utf-8

"""
Custom LoRA (Low-Rank Adaptation) wrappers for UNI2-h fine-tuning.

We wrap target ``nn.Linear`` modules in ``LoRALinear`` and inject them
into the last N transformer blocks' attention projections, leaving the
bulk of UNI2-h frozen. The wrapped output is::

    out = base(x) + lora_B(dropout(lora_A(x))) * (alpha / rank)

with ``lora_B`` initialized to zero so the wrapped module is the
identity at construction — required to preserve a warm-started head
during the first training step.

Why custom rather than ``peft``? The dependency would be heavy
(`peft` pulls in `transformers`, `accelerate`, `safetensors`, etc.)
for ~50 lines of behavior. Keeping the implementation explicit and
in-tree also makes the saved checkpoint format trivial — just save
the named ``lora_*`` parameters and a small config dict alongside the
head state dict.

Sanity check (forward output unchanged at init)::

    from cardiac_acr.backends.uni.backbone import UNIBackbone
    from cardiac_acr.backends.uni.lora import apply_lora_to_uni
    import torch
    a, b = UNIBackbone(compile=False), UNIBackbone(compile=False)
    apply_lora_to_uni(b, target_blocks=4, rank=8, alpha=32)
    x = torch.randn(2, 3, 224, 224, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        ya, yb = a.model(x), b.model(x)
    assert torch.allclose(ya, yb, atol=1e-3)
"""

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Frozen ``nn.Linear`` with a low-rank parallel adapter.

    Parameters
    ----------
    base : nn.Linear
        The pretrained linear to keep frozen. Weights and bias are
        taken over by the wrapper; the underlying tensors are not
        copied.
    rank : int
        Bottleneck dimension. Higher = more capacity, more params.
    alpha : float
        Scaling factor. The effective adapter contribution is
        ``(alpha / rank) * lora_B(lora_A(x))``. ``alpha`` is the
        commonly-tuned knob; ``rank`` is structural.
    dropout : float
        Dropout applied between ``lora_A`` and ``lora_B``.
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 32.0,
                 dropout: float = 0.05):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)}")
        self.base = base
        # Defensive: explicitly freeze the base regardless of caller state.
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Match the base linear's device. The base may be on CUDA
        # (frozen UNI2-h backbone); the new LoRA matrices need to live
        # on the same device or the forward fails. Master weights stay
        # in fp32 — autocast handles the cast on the matmul.
        device = base.weight.device
        self.lora_A = nn.Linear(base.in_features, rank, bias=False,
                                device=device, dtype=torch.float32)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False,
                                device=device, dtype=torch.float32)
        self.lora_dropout = nn.Dropout(dropout)

        # lora_A: kaiming uniform (LoRA paper); lora_B: zero so the
        # adapter is the identity at step 0.
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_dropout(self.lora_A(x))) * self.scaling


def apply_lora_to_uni(backbone, target_blocks: int = 4, rank: int = 8,
                      alpha: float = 32.0, dropout: float = 0.05,
                      targets=("qkv",)):
    """Wrap target Linears in the last ``target_blocks`` ViT blocks.

    Parameters
    ----------
    backbone : UNIBackbone
        Built with ``compile=False``; the ``torch.compile`` graph
        cannot survive submodule replacement.
    target_blocks : int
        Number of trailing blocks to wrap (1..depth). Default 4 of 24.
    rank, alpha, dropout : LoRA hyperparameters (see ``LoRALinear``).
    targets : tuple[str, ...]
        Subset of ``("qkv", "proj")``. ``"qkv"`` wraps each block's
        attention QKV projection (Linear 1536 -> 4608). ``"proj"``
        wraps the attention output projection (Linear 1536 -> 1536).
        SwiGLU MLP is not currently supported.

    Returns
    -------
    list[torch.nn.Parameter]
        The trainable LoRA parameters, ready to hand to an optimizer.
    """
    blocks = backbone.model.blocks
    if target_blocks < 1 or target_blocks > len(blocks):
        raise ValueError(
            f"target_blocks must be in [1, {len(blocks)}], got {target_blocks}"
        )
    valid_targets = {"qkv", "proj"}
    if not set(targets).issubset(valid_targets):
        raise ValueError(
            f"targets must be a subset of {valid_targets}, got {targets}"
        )

    trainable = []
    for block in blocks[-target_blocks:]:
        attn = block.attn
        if "qkv" in targets:
            attn.qkv = LoRALinear(attn.qkv, rank=rank, alpha=alpha,
                                  dropout=dropout)
            trainable.extend(p for p in attn.qkv.parameters() if p.requires_grad)
        if "proj" in targets:
            attn.proj = LoRALinear(attn.proj, rank=rank, alpha=alpha,
                                   dropout=dropout)
            trainable.extend(p for p in attn.proj.parameters() if p.requires_grad)
    return trainable


def lora_state_dict(backbone) -> dict:
    """Return only the trainable LoRA parameters from a wrapped backbone.

    Suitable for serialization alongside a head checkpoint. Frozen
    base weights are excluded (UNI2-h is loaded fresh from HF Hub at
    inference time, so persisting them would be wasteful).
    """
    return {
        name: param.detach().cpu()
        for name, param in backbone.model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }
