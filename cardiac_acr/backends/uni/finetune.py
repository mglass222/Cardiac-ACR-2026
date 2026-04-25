#!/usr/bin/env python
# coding: utf-8

"""
LoRA fine-tune of the UNI2-h tail.

Wraps ``backend.attn.qkv`` in the last ``LORA_TARGET_BLOCKS`` blocks
with rank-r LoRA adapters, then trains those adapters together with a
warm-started head on raw patches with per-batch random augmentation.
The bulk of UNI2-h stays frozen.

Reads from ``ImageFolder`` directly (no cached features), unlike
``train.py`` which works on the static feature cache.

Usage:
    python -m cardiac_acr.backends.uni.finetune
"""

import copy
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cardiac_acr.backends.uni import config as uni_cfg
from cardiac_acr.backends.uni.backbone import UNIBackbone
from cardiac_acr.backends.uni.head import build_head
from cardiac_acr.backends.uni.lora import apply_lora_to_uni
from cardiac_acr.backends.uni.train import (
    _class_weights,
    _cosine_with_warmup,
    _save_checkpoint,
)


def _build_train_transform():
    """Per-batch random augmentation. Approximation of the ResNet
    pipeline that hit ~0.96 (RandomRotation(180) + ColorJitter), with
    softer ColorJitter since UNI is more sensitive to hue shifts than
    ResNet was, and adding flips since they're free under D4
    invariance."""
    return transforms.Compose([
        transforms.Resize(uni_cfg.INPUT_SIZE),
        transforms.CenterCrop(uni_cfg.INPUT_SIZE),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(uni_cfg.IMAGENET_MEAN, uni_cfg.IMAGENET_STD),
    ])


def _build_eval_transform():
    return transforms.Compose([
        transforms.Resize(uni_cfg.INPUT_SIZE),
        transforms.CenterCrop(uni_cfg.INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(uni_cfg.IMAGENET_MEAN, uni_cfg.IMAGENET_STD),
    ])


def _load_warm_start_head(head_type, device):
    """Load the existing trained head as a warm start. Required: the
    LoRA adapters start as the identity, so the first forward pass is
    exactly the existing 0.94 model. Without warm-start the adapters
    train against a from-scratch head and the experiment is
    uninterpretable."""
    head = build_head(head_type)
    path = os.path.join(uni_cfg.MODEL_DIR, f"uni2h_{head_type}_head.pt")
    if not os.path.isfile(path):
        raise SystemExit(
            f"Warm-start head checkpoint not found: {path}\n"
            "Run `python -m cardiac_acr.backends.uni.train` first."
        )
    blob = torch.load(path, weights_only=False)
    head.load_state_dict(blob["state_dict"])
    return head.to(device), list(blob["classes"])


def finetune(seed=0, save=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    head_type = uni_cfg.HEAD_TYPE
    print(f"device: {device}  seed: {seed}")

    # Build datasets / loaders
    train_set = datasets.ImageFolder(
        uni_cfg.TRAIN_DIR, transform=_build_train_transform(), allow_empty=True,
    )
    val_set = datasets.ImageFolder(
        uni_cfg.VALID_DIR, transform=_build_eval_transform(), allow_empty=True,
    )
    classes = train_set.classes
    print(f"train patches: {len(train_set)}  val patches: {len(val_set)}")
    print(f"classes: {classes}")

    train_loader = DataLoader(
        train_set, batch_size=uni_cfg.LORA_BATCH_SIZE, shuffle=True,
        num_workers=uni_cfg.LORA_NUM_WORKERS, pin_memory=True,
        persistent_workers=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=uni_cfg.LORA_BATCH_SIZE, shuffle=False,
        num_workers=uni_cfg.LORA_NUM_WORKERS, pin_memory=True,
        persistent_workers=True,
    )

    # Backbone (compile=False is mandatory for LoRA replacement)
    backbone = UNIBackbone(device=device, compile=False)
    backbone.model.set_grad_checkpointing(True)
    backbone.eval()
    lora_params = apply_lora_to_uni(
        backbone,
        target_blocks=uni_cfg.LORA_TARGET_BLOCKS,
        rank=uni_cfg.LORA_RANK,
        alpha=uni_cfg.LORA_ALPHA,
        dropout=uni_cfg.LORA_DROPOUT,
        targets=uni_cfg.LORA_TARGETS,
    )
    # Only the LoRA submodules go into train() so frozen LayerNorms
    # and dropout in untouched blocks don't drift.
    for module in backbone.model.modules():
        from cardiac_acr.backends.uni.lora import LoRALinear
        if isinstance(module, LoRALinear):
            module.lora_A.train()
            module.lora_B.train()
            module.lora_dropout.train()
    print(f"trainable LoRA params: {sum(p.numel() for p in lora_params):,}")

    # Head warm-start
    head, head_classes = _load_warm_start_head(head_type, device)
    if head_classes != classes:
        raise SystemExit(
            f"Class mismatch between head checkpoint and ImageFolder: "
            f"{head_classes} vs {classes}"
        )
    head.train()

    # Class-weighted CE with weight clamp
    train_labels = torch.tensor([y for _, y in train_set.samples])
    weights = _class_weights(train_labels, len(classes)).to(device)
    weights = torch.clamp(weights, max=uni_cfg.LORA_CLASS_WEIGHT_CLIP)
    print(f"class weights (clamped <= {uni_cfg.LORA_CLASS_WEIGHT_CLIP}): "
          f"{weights.cpu().tolist()}")
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Two AdamW param groups: head + LoRA at different LRs
    optimizer = torch.optim.AdamW([
        {"params": head.parameters(), "lr": uni_cfg.LORA_HEAD_LR,
         "weight_decay": uni_cfg.TRAIN_WEIGHT_DECAY,
         "name": "head", "base_lr": uni_cfg.LORA_HEAD_LR},
        {"params": lora_params, "lr": uni_cfg.LORA_LR,
         "weight_decay": 0.0,
         "name": "lora", "base_lr": uni_cfg.LORA_LR},
    ])

    # GradScaler is mandatory at fp16 on Turing — without it lora_B's
    # zero-init gradients underflow and the run looks dead.
    scaler = torch.cuda.amp.GradScaler()
    autocast_dtype = backbone.dtype  # fp16 on Turing, bf16 on Ampere+

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * uni_cfg.LORA_NUM_EPOCHS
    warmup_steps = steps_per_epoch * uni_cfg.LORA_WARMUP_EPOCHS

    print(
        f"\nLoRA fine-tune | head: {head_type} | epochs: {uni_cfg.LORA_NUM_EPOCHS} "
        f"| batch: {uni_cfg.LORA_BATCH_SIZE} | head_lr: {uni_cfg.LORA_HEAD_LR} "
        f"| lora_lr: {uni_cfg.LORA_LR} | autocast: {autocast_dtype}\n"
    )

    best_acc = 0.0
    best_state = {
        "head": copy.deepcopy(head.state_dict()),
        "lora": {
            n: p.detach().cpu().clone()
            for n, p in backbone.model.named_parameters()
            if "lora_A" in n or "lora_B" in n
        },
    }
    epochs_since_improvement = 0
    step = 0
    t0 = time.time()

    for epoch in range(uni_cfg.LORA_NUM_EPOCHS):
        head.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            scale = _cosine_with_warmup(step, total_steps, warmup_steps)
            for g in optimizer.param_groups:
                g["lr"] = g["base_lr"] * scale

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                feats = backbone.model(images)  # NOT encode() — we need grad
                logits = head(feats)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                head.parameters(), uni_cfg.LORA_GRAD_CLIP
            )
            torch.nn.utils.clip_grad_norm_(
                lora_params, uni_cfg.LORA_GRAD_CLIP
            )
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * images.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += images.size(0)
            step += 1

        train_loss = train_loss_sum / max(1, train_total)
        train_acc = train_correct / max(1, train_total)

        # Validation
        head.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    feats = backbone.model(images)
                    logits = head(feats)
                    loss = criterion(logits, labels)
                val_loss_sum += loss.item() * images.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += images.size(0)

        val_loss = val_loss_sum / max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        head_lr = optimizer.param_groups[0]["lr"]
        lora_lr = optimizer.param_groups[1]["lr"]
        print(
            f"epoch {epoch+1:>3}/{uni_cfg.LORA_NUM_EPOCHS} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"head_lr {head_lr:.2e} lora_lr {lora_lr:.2e}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {
                "head": copy.deepcopy(head.state_dict()),
                "lora": {
                    n: p.detach().cpu().clone()
                    for n, p in backbone.model.named_parameters()
                    if "lora_A" in n or "lora_B" in n
                },
            }
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        # Hard abort: val acc collapses well below the warm-start
        # floor (0.94) in the first 2 epochs. The threshold is
        # deliberately loose (0.90, not 0.93) — a 1-pp slip is normal
        # mid-warmup and may recover; only large drops indicate the
        # real failure modes (LR way too high, GradScaler off, or LoRA
        # params not in the optimizer).
        if epoch < 2 and val_acc < 0.90:
            print(f"\nABORT: val acc {val_acc:.4f} < 0.90 at epoch {epoch+1}. "
                  "Likely catastrophic LR or optimizer misconfig. Killing run.")
            return None, None

        if epochs_since_improvement >= uni_cfg.LORA_EARLY_STOP_PATIENCE:
            print(f"\nEarly stop: no improvement for {epochs_since_improvement} epochs.")
            break

    elapsed = time.time() - t0
    print(f"\nFine-tune complete in {elapsed:.1f}s. Best val acc: {best_acc:.4f}")

    # Restore the best LoRA + head state
    head.load_state_dict(best_state["head"])
    for name, param in backbone.model.named_parameters():
        if name in best_state["lora"]:
            with torch.no_grad():
                param.copy_(best_state["lora"][name].to(param.device))

    if save:
        lora_config = {
            "rank": uni_cfg.LORA_RANK,
            "alpha": uni_cfg.LORA_ALPHA,
            "target_blocks": uni_cfg.LORA_TARGET_BLOCKS,
            "targets": list(uni_cfg.LORA_TARGETS),
            "dropout": uni_cfg.LORA_DROPOUT,
        }
        _save_checkpoint(head, head_type, classes,
                         lora_backbone=backbone, lora_config=lora_config)

    return head, best_acc


def main():
    finetune()


if __name__ == "__main__":
    main()
