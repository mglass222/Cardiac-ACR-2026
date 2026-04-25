#!/usr/bin/env python
# coding: utf-8

"""
Frozen UNI2-h backbone.

Loads the MahmoodLab/UNI2-h pathology foundation model via timm's
HuggingFace Hub integration and exposes a simple ``.encode(imgs)``
API returning 1536-dim CLS-token embeddings.

The model card's exact create_model arguments are reproduced here —
UNI2-h uses a non-default ViT configuration (patch 14, depth 24,
heads 24, embed 1536, SwiGLU FFN, 8 register tokens) that needs the
full spec.
"""

import os

import torch
import torch.nn as nn
import timm
from timm.layers import SwiGLUPacked

from cardiac_acr.backends.uni import config as uni_cfg


class UNIBackbone(nn.Module):
    """Frozen UNI2-h encoder with a convenience ``.encode()`` method."""

    def __init__(self, device=None, dtype=None, compile=True):
        super().__init__()
        self.device = device or _default_device()
        self.dtype = dtype if dtype is not None else _default_autocast_dtype(self.device)
        self.model = _build_uni2h()
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.to(self.device)
        # ``compile`` is opt-out for callers that need to mutate the
        # module tree afterwards (e.g. injecting LoRA wrappers — the
        # Dynamo graph cannot survive submodule replacement).
        if compile and self.device.type == "cuda":
            self.model = torch.compile(self.model)

    @torch.no_grad()
    def encode(self, images):
        """Encode a batch of preprocessed tensors.

        Parameters
        ----------
        images : torch.Tensor
            Shape ``(B, 3, 224, 224)``, already ImageNet-normalized.

        Returns
        -------
        torch.Tensor
            Shape ``(B, EMBED_DIM)`` (1536), **float32 on CPU** so the
            downstream head trains in fp32 regardless of the autocast
            dtype used for the forward pass.
        """
        images = images.to(self.device, non_blocking=True)
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            feats = self.model(images)
        return feats.float().cpu()


def _default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _default_autocast_dtype(device):
    # Turing (sm_7x) tensor cores accelerate fp16 but not bf16, so bf16
    # falls back to CUDA cores and tanks throughput ~10x. Ampere+ (sm_8x+)
    # added bf16 tensor cores, so bf16 is the safer default there.
    if device.type != "cuda":
        return torch.float32
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def _build_uni2h():
    """Instantiate UNI2-h per the MahmoodLab model card."""
    _require_hf_auth()
    # init_values=1e-5 is required so LayerScale ls1/ls2.gamma keys load
    # cleanly — the published UNI2-h weights include them even though the
    # model card's create_model snippet omits the arg.
    return timm.create_model(
        uni_cfg.UNI_MODEL_ID,
        pretrained=True,
        img_size=uni_cfg.INPUT_SIZE,
        patch_size=14,
        depth=24,
        num_heads=24,
        embed_dim=uni_cfg.EMBED_DIM,
        mlp_ratio=2.66667 * 2,
        num_classes=0,
        no_embed_class=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        reg_tokens=8,
        dynamic_img_size=False,
        init_values=1e-5,
    )


def _require_hf_auth():
    """Fail early with an actionable message if HF auth isn't set up."""
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return
    token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.isfile(token_path) and os.path.getsize(token_path) > 0:
        return
    raise RuntimeError(
        "No HuggingFace credentials found. Run `hf auth login` "
        "(or export HF_TOKEN) before loading UNI2-h. The token must have "
        "read access and the account must have been granted access to "
        "MahmoodLab/UNI2-h."
    )
