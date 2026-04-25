#!/usr/bin/env python
# coding: utf-8

"""
UNI2-h backend configuration.

Re-exports the shared top-level config so modules in this subpackage can
keep a single ``from cardiac_acr.backends.uni import config as uni_cfg``
import and reach both shared paths (``TRAIN_DIR``, ``IMAGENET_MEAN`` …)
and UNI-specific hyperparameters (``UNI_MODEL_ID``, ``EMBED_DIM``,
head-training rates …) plus the per-backend output locations below.
"""

import os

from cardiac_acr.config import *  # noqa: F401,F403  (re-export shared constants)
from cardiac_acr.config import BACKEND_DIR, DATA_DIR, WSI_DIR


#####################################################################
# UNI-specific output locations
#####################################################################

FEATURE_DIR = os.path.join(DATA_DIR, "Features")
MODEL_DIR = os.path.join(DATA_DIR, "Saved_Models", "UNI_Head")

# Per-backend WSI diagnosis outputs. The UNI backend owns the
# unsuffixed names; the ResNet backend nests under a Weighted_Loss/
# subfolder in its own config so the two don't collide.
SAVED_DATABASE_DIR = os.path.join(BACKEND_DIR, "Saved_Databases")
SLIDE_DX_DIR = os.path.join(BACKEND_DIR, "Slide_Dx")
ANNOTATED_PNG_DIR = os.path.join(BACKEND_DIR, "Annotated_Test_Slides")
TEST_SLIDE_PREDICTIONS_DIR = os.path.join(BACKEND_DIR, "Test_Slide_Predictions")
TEST_SLIDE_ANNOTATIONS_DIR = os.path.join(WSI_DIR, "TEST_SLIDE_ANNOTATIONS")


#####################################################################
# UNI backbone
#####################################################################

UNI_MODEL_ID = "hf-hub:MahmoodLab/UNI2-h"
EMBED_DIM = 1536                 # UNI2-h CLS-token dimension
INPUT_SIZE = 224                 # UNI2-h expects 224x224


#####################################################################
# UNI feature encoding
#####################################################################

ENCODE_BATCH_SIZE = 32           # fits in ~5 GB on 8 GB cards in fp16/bf16; drop if OOM
ENCODE_NUM_WORKERS = 8

# Number of D4-symmetry views per training patch. 1 = canonical only
# (legacy); 8 = full dihedral group (4 rotations x 2 flips). Validation
# always uses 1 view so val-acc stays comparable across runs.
NUM_TRAIN_VIEWS = 8


# One cached tensor per split. Each file stores
# {"features": FloatTensor [N, EMBED_DIM], "labels": LongTensor [N], "classes": list[str]}.
# Training cache holds NUM_TRAIN_VIEWS encodings per source patch (one
# per D4 symmetry); validation always holds one encoding per patch.
TRAINING_FEATURES_PATH = os.path.join(
    FEATURE_DIR, f"training_views{NUM_TRAIN_VIEWS}.pt"
)
VALIDATION_FEATURES_PATH = os.path.join(FEATURE_DIR, "validation.pt")


#####################################################################
# Classifier head
#####################################################################

HEAD_TYPE = "mlp"                # "linear" | "mlp"
HEAD_DROPOUT = 0.4               # only used when HEAD_TYPE == "mlp"
HEAD_HIDDEN_DIM = 512            # only used when HEAD_TYPE == "mlp"


#####################################################################
# Head training
#####################################################################

TRAIN_BATCH_SIZE = 512           # features are tiny — large batches are fine
TRAIN_LEARNING_RATE = 1e-3
TRAIN_WEIGHT_DECAY = 1e-4
TRAIN_NUM_EPOCHS = 50
TRAIN_COSINE_WARMUP_EPOCHS = 2


#####################################################################
# LoRA fine-tune (used by backends/uni/finetune.py)
#####################################################################

# LoRA structure
LORA_RANK = 8
LORA_ALPHA = 32                  # scaling=alpha/rank=4; compensates for
                                 # init_values=1e-5 LayerScale damping.
LORA_TARGET_BLOCKS = 4           # last N of 24 ViT-H blocks
LORA_TARGETS = ("qkv",)          # ("qkv",) or ("qkv", "proj")
LORA_DROPOUT = 0.05

# LoRA training hyperparameters
LORA_NUM_EPOCHS = 15
LORA_BATCH_SIZE = 16             # 32 if memory allows; drop to 8 + grad accum if OOM
LORA_LR = 5e-5                   # LoRA params. First trial at 1e-4 caused
                                 # mild val-acc regression (0.94 -> 0.93) at
                                 # half-warmup; halved on the second pass.
LORA_HEAD_LR = 1e-5              # warm-started head: very gentle. The head
                                 # is already at 0.94 — it needs nudging to
                                 # absorb LoRA's residual-stream changes,
                                 # not retraining.
LORA_WARMUP_EPOCHS = 2
LORA_GRAD_CLIP = 1.0             # essential at fp16
LORA_CLASS_WEIGHT_CLIP = 5.0     # clamp class-weighted CE so the 13.3x
                                 # Hemorrhage weight doesn't dominate updates
LORA_EARLY_STOP_PATIENCE = 5     # epochs without val improvement -> stop
LORA_NUM_WORKERS = 8
