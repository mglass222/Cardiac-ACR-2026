# LoRA fine-tune of UNI2-h tail (close the val-acc gap)

## Context

The UNI head plateaus at ~0.94 validation accuracy. Today's two
ablations ruled out the leading hypotheses:

- **D4 multi-view augmentation** (shipped 2026-04-24): +0.58 pp
  (0.9344 → 0.9402). Real but marginal; the model overfit harder on
  near-duplicate features.
- **Hyperparameter sweep** (shipped 2026-04-24): no config beats the
  current default by more than the 0.005 pp run-to-run noise floor.

The remaining hypothesis is that **the frozen backbone is the
ceiling**. The prior ResNet-50 pipeline reached ~0.96 by unfreezing
`layer3`/`layer4`. The ViT analog is to inject small trainable LoRA
adapters into the last few transformer blocks' attention projections
while keeping the bulk of UNI2-h frozen — the "B" option from the
earlier discussion. Mid-effort, modern best practice for small-data
fine-tuning, fits in 8 GB VRAM, and explicitly avoids the
catastrophic-forgetting risk of a full unfreeze.

Goal: close as much of the remaining ~2 pp gap as possible without
adding heavy dependencies (`peft`, `transformers`) and without
breaking the existing WSI inference path.

## Approach

Custom 50-line LoRA module (no `peft`). Wrap `block.attn.qkv` in the
last 4 of UNI2-h's 24 transformer blocks with a low-rank parallel
adapter. Train the adapters together with a warm-started head on raw
patches with per-batch random augmentation. Save adapters alongside
the head checkpoint; teach the WSI inference loader to re-apply the
wrapper when loading.

Key validation finding: confirmed via `timm/layers/attention.py:115`
that the standard `Attention.forward` calls `self.qkv(x)` as a
function (not `qkv.weight` directly), so the wrapper works
transparently.

## Files to create/modify

### NEW: `cardiac_acr/backends/uni/lora.py`

Two pieces:

1. `class LoRALinear(nn.Module)` — wraps an existing `nn.Linear`. Two
   trainable rank-r matrices (`lora_A`, `lora_B`) in **fp32 master
   weights** (autocast handles fp16 cast on the matmul). `lora_A` uses
   Kaiming init, `lora_B` is zero-init so the adapter starts as the
   identity (critical for warm-start preservation). Explicitly sets
   `self.base.weight.requires_grad_(False)` and the bias if present.
   Exposes `.in_features` / `.out_features` so reflective code keeps
   working.
2. `def apply_lora_to_uni(backbone, target_blocks=4, rank=8, alpha=32, dropout=0.05, targets=("qkv",))`
   — walks `backbone.model.blocks[-target_blocks:]` and replaces each
   `block.attn.qkv` (and `block.attn.proj` if `"proj"` in targets) with
   `LoRALinear(...)`. Returns the list of trainable LoRA parameters.

`alpha=32` (scaling=4) compensates for UNI2-h's `init_values=1e-5`
LayerScale, which dampens any residual-stream perturbation by ~1e-5.

### MODIFY: `cardiac_acr/backends/uni/backbone.py`

Add `compile: bool = True` to `__init__`. When `False`, skip the
`torch.compile(self.model)` line. Required because compile + LoRA
module replacement = recompile per step + a known crash surface with
grad checkpointing. Default stays `True` so encode-time and
non-LoRA inference are unaffected.

### NEW: `cardiac_acr/backends/uni/finetune.py`

Distinct from `train.py` so the cached-features pipeline stays
shippable as a baseline. Structure:

- Load `ImageFolder(TRAIN_DIR)` with **per-batch random** transforms
  (not D4 deterministic). Augmentation list: `RandomRotation(180)` +
  `RandomHorizontalFlip(0.5)` + `RandomVerticalFlip(0.5)` +
  `ColorJitter(0.2, 0.2, 0.2, 0.05)` + `Resize(224)` + `CenterCrop(224)`
  + `ToTensor` + `Normalize(IMAGENET)`. Validation uses the same
  Resize/CenterCrop/ToTensor/Normalize pipeline (no augmentation).
- Instantiate `UNIBackbone(compile=False)`, set the whole backbone to
  `eval()`, call `apply_lora_to_uni(...)`, then put **only the LoRA
  submodules** in `train()` (so frozen LayerNorms and dropout don't
  drift).
- Build the head via `build_head(...)`, **warm-start from
  `uni2h_mlp_head.pt`** (the current 0.9402 checkpoint).
- AdamW with two param groups: head at `lr=5e-5, weight_decay=1e-4`,
  LoRA at `lr=1e-4, weight_decay=0.0`. Cosine schedule + 2-epoch
  warmup scales both proportionally.
- Custom forward: `self.model(images)` directly (not `encode()` —
  it's `@torch.no_grad()`). Keep features on GPU through the head.
- `torch.cuda.amp.GradScaler` + `torch.autocast(device_type="cuda", dtype=fp16)`.
  Required at fp16 on Turing — without scaler, `lora_B`'s zero-init
  gradients underflow and the run looks dead.
- `gradient_clip_norm=1.0` (essential at fp16).
- Loss: class-weighted CE with **per-class weight clipped to max 5.0**
  so the 13.3× Hemorrhage weight × tiny batch doesn't dominate single
  updates.
- Best-val-acc checkpointing. Early stop if no improvement in 5 epochs.
- 15 epochs, batch 16 (or 32 if memory allows), grad-accum 2 if batch
  drops below 16. Grad checkpointing on
  (`backbone.model.set_grad_checkpointing(True)`).

### MODIFY: `cardiac_acr/backends/uni/train.py`

Extend `_save_checkpoint` to optionally accept `backbone=None`. When
provided, walk `backbone.model.named_parameters()` and save any
parameter whose name contains `"lora_"` into a `lora_state_dict`
field. Also save `lora_config` (rank, alpha, target_blocks, targets).
Existing checkpoints without LoRA continue to work — `finetune.py`
calls the extended path; `train.py:main()` keeps its current call.

### MODIFY: `cardiac_acr/backends/uni/classifier.py`

Update `load_classifier` to:
1. Load the head + checkpoint blob.
2. Check `blob.get("lora_config")`. If present, instantiate
   `UNIBackbone(compile=False)`, call `apply_lora_to_uni(...)` with
   the saved config, then `backbone.model.load_state_dict(blob["lora_state_dict"], strict=False)`.
3. Assert `unexpected` keys are empty (config mismatch fails loudly).
4. If `lora_config` is absent, behave exactly as today
   (instantiate `UNIBackbone()` with compile=True, no wrapper).

This keeps WSI inference (`wsi/diagnose.py`, `wsi/count_1r2.py`)
unchanged — they just call `classifier.classify(batch)`.

### MODIFY: `cardiac_acr/backends/uni/config.py`

Add a small block at the end:

```python
# LoRA fine-tune defaults (used by backends/uni/finetune.py).
LORA_RANK = 8
LORA_ALPHA = 32
LORA_TARGET_BLOCKS = 4
LORA_TARGETS = ("qkv",)
LORA_DROPOUT = 0.05
LORA_LR = 1e-4
LORA_HEAD_LR = 5e-5
LORA_NUM_EPOCHS = 15
LORA_BATCH_SIZE = 16
LORA_WARMUP_EPOCHS = 2
LORA_GRAD_CLIP = 1.0
LORA_CLASS_WEIGHT_CLIP = 5.0
```

### NEW: `docs/UNI_LORA_PLAN.md`

Copy of this plan. Same pattern as `UNI_MULTIVIEW_PLAN.md` —
preserves the design rationale in the repo before any code lands.

### MODIFY: `docs/DEVELOPMENT_LOG.md`

Append a new dated entry **before any code changes** with the heading
"2026-04-24 — Experimenting with LoRA fine-tune of the UNI2-h tail
(in progress)". The user explicitly asked for the log to be updated
to indicate experimentation. The entry states the hypothesis (frozen
backbone is the ceiling), the approach (LoRA on QKV in last 4
blocks), the explicit success bar (3 seeds, mean ≥ 0.955, all ≥ 0.95
to claim the gap is closed), and the failure modes to abort on
(val < 0.93 in first 2 epochs → kill, indicates LR or GradScaler
misconfig). Marks the entry as **speculative — to be updated with
results**.

## Reused components

- `UNIBackbone._build_uni2h()` (`backends/uni/backbone.py:78`) —
  unchanged; the LoRA path just adds `compile=False` and applies
  wrappers afterwards.
- `build_head()` and `LinearHead`/`MLPHead` (`backends/uni/head.py`)
  — used unchanged; warm-started from existing checkpoint.
- `_class_weights()` and `_cosine_with_warmup()`
  (`backends/uni/train.py:32, 48`) — reused; will need to add
  `torch.clamp(weights, max=clip)` step for `LORA_CLASS_WEIGHT_CLIP`.
- `load_head_checkpoint()` (`backends/uni/evaluate.py:35`) — reused
  unchanged; `classifier.py` already calls it.
- `datasets.ImageFolder` + `DataLoader` pattern from
  `backends/uni/encode_patches.py:106-148` — same shape, different
  transforms.

## Verification

**Step 1 — apply_lora and forward sanity check**:
```python
from cardiac_acr.backends.uni.backbone import UNIBackbone
from cardiac_acr.backends.uni.lora import apply_lora_to_uni
b = UNIBackbone(compile=False)
params = apply_lora_to_uni(b, target_blocks=4, rank=8, alpha=32)
print(f"trainable LoRA params: {sum(p.numel() for p in params)}")
# expect ~196K (8 * (1536 + 4608) * 4 blocks = ~196K)

# Forward through frozen+wrapped backbone should produce same output
# as plain UNI for the canonical untrained adapters (lora_B init=0).
import torch
x = torch.randn(2, 3, 224, 224, device="cuda")
b2 = UNIBackbone(compile=False)
with torch.autocast("cuda", dtype=torch.float16):
    y_plain = b2.model(x)
    y_lora  = b.model(x)
assert torch.allclose(y_plain, y_lora, atol=1e-3), "LoRA wrapper changed forward at init"
```

**Step 2 — fine-tune run**:
```bash
python -m cardiac_acr.backends.uni.finetune
```

Expected log shape:
- `epoch  1/15 | train loss X | val loss Y | val acc Z | head_lr 5e-5 | lora_lr 1e-4`
- val acc starting near 0.94 (because head is warm-started) and
  trending up over the first ~5 epochs.
- **Abort signal**: val acc drops below 0.93 in epochs 1–2 → kill the
  run; LoRA LR is too high or GradScaler isn't engaged.

Memory expectation: ~5–6 GB on 2070 SUPER at batch 16 + grad
checkpointing. Drop to batch 8 with `grad_accum_steps=2` if OOM.

**Step 3 — three-seed reproducibility** (after step 2 succeeds):
Re-run `finetune.py` 2 more times with different `torch.manual_seed`
values. Capture best val acc per run. **Success bar**: mean ≥ 0.955
*and* every individual run ≥ 0.95. Anything below is in the noise
band of the existing 0.94 plateau.

**Step 4 — WSI inference regression**:
```bash
python -m cardiac_acr.wsi.diagnose --backend uni
```
on slide 139. Expected slide-level dx still `2R`. Patch probabilities
will shift but the slide-level decision should be stable. If it
flips, investigate before declaring the fine-tune a win.

**Step 5 — evaluate metrics dump**:
```bash
python -m cardiac_acr.backends.uni.evaluate
```
Capture per-class P/R/F1, AUROC, confusion matrix. Compare to the
post-D4 numbers logged on 2026-04-24. The hard pairs (1R1A↔1R2,
Healing↔Normal) are where most of the gain should land if it lands
anywhere.

## Commit plan

1. **First commit (before any code changes):**
   - Copy this plan to `docs/UNI_LORA_PLAN.md`.
   - Append the "Experimenting with LoRA fine-tune (in progress)"
     entry to `docs/DEVELOPMENT_LOG.md`.
2. **Second commit:** `feat(uni): LoRA module + apply_lora_to_uni
   helper`. Adds `lora.py` and the `compile=` flag on `backbone.py`.
   Includes the forward-sanity-check snippet in a docstring.
3. **Third commit:** `feat(uni): LoRA fine-tune training loop
   (finetune.py)`. Adds the training script, config knobs, and
   extended `_save_checkpoint`/`load_classifier`.
4. **Fourth commit (after results):** dev-log update with measured
   val acc across 3 seeds, per-class deltas, slide-level regression
   check, and a verdict (gap closed / partially closed / not closed).

## Out of scope (revisit only if v1 partially closes the gap)

- LoRA on `attn.proj` (would double trainable params, smaller
  marginal benefit per literature).
- LoRA on the SwiGLU MLP — `SwiGLUPacked` fuses gate+value into one
  Linear with non-standard shape; needs a custom wrapper, not worth
  it for v1.
- DoRA, prompt tuning, or other PEFT variants.
- Adding `peft` as a dependency.
- More target blocks (5+) — diminishing returns and OOM risk.
- Test-time D4 logit averaging at WSI inference — orthogonal change,
  separate plan.

## Risk register

- **Memory**: 8 GB at fp16 + grad checkpointing should fit batch 16.
  If OOM, drop to batch 8 + grad accum.
- **GradScaler omission**: silent failure mode (loss looks normal,
  weights don't update). Mandatory at fp16 on Turing.
- **Compile + wrapper interaction**: addressed by `compile=False` on
  the LoRA path. Don't try to be clever and re-enable.
- **Catastrophic forgetting**: aggressive aug + small dataset on a
  large backbone. Mitigations: warm-start head, conservative LRs,
  early stopping at 5 epochs no-improvement, alpha=32 keeps the LoRA
  delta small in absolute terms.
- **WSI inference break**: addressed by `lora_config`-gated loading
  in `classifier.py`. Existing checkpoints without LoRA continue to
  work unchanged.
