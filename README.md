# Cardiac-ACR

Cardiac allograft rejection diagnosis from whole-slide images. Shared
preprocessing, 1R2 focus counting, WSI inference, and annotation code;
**two interchangeable patch-classification backends**:

- **`uni`** — frozen `MahmoodLab/UNI2-h` (ViT-H/14, 1536-dim CLS
  embeddings) plus a trainable linear or MLP head. Feature-cache
  workflow: encode every patch once, train the head on the cache.
- **`resnet`** — the original 2019–2021 ResNet-50 patch classifier,
  trained end-to-end with a two-phase (FC+BN then `layer3`/`layer4`)
  unfreezing schedule and class-balanced cross-entropy.

Pick the backend at train / evaluate / diagnose time with
`--backend {uni,resnet}`. The WSI-level inference loop is
backend-agnostic — the same `wsi/diagnose.py` drives either one.

See `docs/DEVELOPMENT_LOG.md` for design rationale, the backbone survey
that landed on UNI2-h, and the refactor notes that merged the
historical `Cardiac-ACR-UNI` and `Cardiac-ACR-Resnet` projects into
this single package.

See `docs/DEVELOPMENT_LOG.md` for design rationale, alternatives
surveyed, and implementation notes.

---

## Requirements

### Hardware
- NVIDIA GPU with **≥ 6 GB VRAM** for UNI2-h inference (tested on
  RTX 2070 SUPER 8 GB — peak 2.9 GB at encode batch 4, ~5 GB at the
  default batch 16).
- ~5 GB free disk for the UNI2-h weights cache (one-time, shared
  across runs).
- Room for the patch library (tens of GB depending on slide count)
  and whatever raw WSI data you're working with.

### Software
- Python ≥ 3.10
- Linux or macOS (Windows works in principle via
  `openslide_compat.py`'s DLL helper, but not tested for this repo).
- System OpenSlide library — on Debian/Ubuntu:
  ```bash
  sudo apt install libopenslide0 openslide-tools
  ```

### Python packages (installed via pyproject)
- `torch`, `torchvision` (CUDA build matching your driver)
- `timm ≥ 1.0.0` (needed for `timm.layers.SwiGLUPacked`)
- `huggingface_hub`
- `numpy`, `Pillow`, `opencv-python`, `scipy`
- `scikit-image`, `scikit-learn`
- `openslide-python`

### HuggingFace access
UNI2-h is a **gated** model. Two requirements:

1. Your HuggingFace account must have **access** to
   [`MahmoodLab/UNI2-h`](https://huggingface.co/MahmoodLab/UNI2-h). The
   gate explicitly rejects applications from `@gmail`, `@hotmail`,
   `@qq` addresses — add an **institutional email** as your account's
   primary email, then apply on the model page.
2. Your local shell must have a valid HF token with **Read** access
   to gated repos. Run once:
   ```bash
   hf auth login
   ```
   (Older docs reference `huggingface-cli login` — deprecated; use
   `hf auth login` from `huggingface_hub ≥ 1.0`.)

---

## Setup

```bash
# 1. Native OpenSlide
sudo apt install libopenslide0 openslide-tools

# 2. Clone
git clone <this-repo-url> Cardiac-ACR
cd Cardiac-ACR

# 3. Venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 4. CUDA-matched torch (adjust the index URL for your driver;
#    cu124 works with NVIDIA driver ≥ 550)
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision

# 5. Install this package editably
pip install -e .

# 6. HuggingFace auth (UNI2-h must be accessible to your account)
hf auth login

# 7. Smoke-test
python -m cardiac_acr.utils.check_dependencies
python -c "from cardiac_acr.backends.uni.backbone import UNIBackbone; UNIBackbone(); print('OK')"
```

The last smoke test triggers a ~2.7 GB download of the UNI2-h weights
on first run. Subsequent loads are cached in
`~/.cache/huggingface/hub/`.

---

## Data layout

By default, all data lives under `Cardiac-ACR/data/`:

```
data/
├── WSI/
│   ├── Training/                 .svs files + matching .xml annotations
│   └── Test/                     .svs files for inference
├── Patches/                      (auto-created by extract_patches + create_training_sets)
│   ├── Openslide_Output/<class>/
│   └── Training_Sets/
│       ├── Training/<class>/
│       └── Validation/<class>/
├── DeepHistoPath/                (auto-created by preprocessing)
│   ├── training_png/
│   ├── filter_png/
│   ├── tiles_png/
│   └── tiles_png_split/
├── Features/                     (auto-created by encode_patches)
│   ├── training.pt
│   └── validation.pt
├── Saved_Models/UNI_Head/        (auto-created by train)
│   └── uni2h_<head>_head.pt
└── Backend/                      (auto-created by diagnose_wsi)
    ├── Saved_Databases/          per-slide prediction pickles
    └── Slide_Dx/                 per-slide rejection grade pickles
```

Before running anything, place your data:
- **Training WSIs** (`.svs` + matching ImageScope `.xml` annotations)
  → `data/WSI/Training/`
- **Inference WSIs** (`.svs` only) → `data/WSI/Test/`

If you already have these in another location (e.g. an existing
Cardiac-ACR data tree) and don't want to duplicate hundreds of GB, a
single symlink keeps everything in place:
```bash
ln -s /path/to/existing/data /home/you/Documents/Code/Cardiac-ACR/data
```
Every path in `cardiac_acr/config.py` is a single-level constant,
so you can also edit those directly if your layout is non-standard.

---

## Running the pipeline

```bash
cd ~/Documents/Code/Cardiac-ACR
source .venv/bin/activate
```

### 1. Extract and split the patch library (one-time)

```bash
python -m cardiac_acr.preprocessing.extract_patches
python -m cardiac_acr.preprocessing.create_training_sets
```

`extract_patches` crops SVS + XML annotations into per-class PNGs;
`create_training_sets` does the 80/20 per-slide split. Output lands
under `data/Patches/`.

### 2. Encode patches with UNI2-h (one-time)

```bash
python -m cardiac_acr.backends.uni.encode_patches
```

Walks `Training_Sets/Training/` and `Training_Sets/Validation/`,
encodes every PNG patch in bf16 autocast (no gradients), and writes:

- `data/Features/training.pt`
- `data/Features/validation.pt`

Each file is ~6 KB per patch (1536 floats × 4 bytes). Progress is
printed at each batch. Expected throughput on a 2070 SUPER at batch
16: somewhere in the 100–200 img/s range depending on IO.

### 3. Train the classifier head

```bash
python -m cardiac_acr.backends.uni.train
```

Loads the feature caches, trains `Linear(1536, 6)` with AdamW +
cosine-with-warmup + class-balanced cross-entropy, saves the
best-validation checkpoint to
`data/Saved_Models/UNI_Head/uni2h_linear_head.pt`.

Fast — the whole training run typically finishes in under a minute on
GPU. To try a 2-layer MLP instead, edit
`cardiac_acr/backends/uni/config.py`:

```python
HEAD_TYPE = "mlp"
```

The MLP checkpoint is written alongside the linear one with a
different filename, so you can keep both.

### 4. Evaluate on the validation split

```bash
python -m cardiac_acr.backends.uni.evaluate
```

Prints:
- per-class precision / recall / F1 / support
- one-vs-rest AUROC per class (with a `n/a` for single-label classes
  like Hemorrhage, which is annotated on only one training slide)
- a confusion matrix

### 5. Run end-to-end WSI inference

```bash
# Disk-based (default): materializes intermediate tile + patch PNGs
python -m cardiac_acr.wsi.diagnose --backend uni

# Streaming: reads 224×224 patches directly from the SVS via OpenSlide,
# no intermediate patch PNGs on disk. Preprocessing drops from several
# minutes to <10 seconds per slide.
python -m cardiac_acr.wsi.diagnose --backend uni --streaming
```

For each SVS file in `data/WSI/Test/`:

1. Extract PNG + tissue-filtered image (`slide` + `filter` modules).
2. Score tiles. In disk mode, also split into 224×224 patches on disk
   (`tiles` + `tileset_utils`). In streaming mode, scoring is
   in-memory per-slide.
3. Drop patches with < 50% tissue (inside the classify DataLoader).
4. Encode surviving patches with UNI2-h and classify with the head.
5. Threshold predictions (keep any patch whose top softmax exceeds
   `PREDICTION_THRESHOLD`, default 0.99).
6. Count 1R2 focuses using the dedicated segmentation pipeline
   (`count_1r2`).
7. Aggregate into a slide-level rejection grade
   (`0R`, `1R1A`, `1R2`, `2R`).

Outputs:

| File | Content |
|---|---|
| `data/Backend/Saved_Databases/model_predictions_dict_<slide>.pickle` | Dict of patch path → 6-vector of class probabilities |
| `data/Backend/Saved_Databases/model_predictions_dict_<slide>_filtered.pickle` | Same, after threshold filtering |
| `data/Backend/Slide_Dx/slide_dx_dict_99_pct.pickle` | Dict of slide number → final rejection grade |

**V1 limitation**: `wsi/diagnose.py` does not yet invoke
`annotate_png` / `annotate_svs` — the slide-level grade is written,
but per-patch PNG overlays and SVS region annotations are deferred to
V2. The annotation functions are available in this package and read
from the same `Saved_Databases/` pickles, so wiring them in is a
follow-up call in the main loop.

---

## Configuration

Shared paths and preprocessing constants live in
`cardiac_acr/config.py`; UNI-backbone and head-training
hyperparameters live in `cardiac_acr/backends/uni/config.py`. The
most commonly changed:

| Field | Default | Notes |
|---|---|---|
| `ENCODE_BATCH_SIZE` | 16 | Lower to 8 or 4 if you see CUDA OOM during `encode_patches` |
| `ENCODE_NUM_WORKERS` | 8 | DataLoader workers; lower if IO is a bottleneck |
| `HEAD_TYPE` | `"linear"` | Switch to `"mlp"` for the 2-layer fallback |
| `HEAD_HIDDEN_DIM` | 512 | MLP hidden layer size — ignored for linear |
| `HEAD_DROPOUT` | 0.4 | MLP dropout — ignored for linear |
| `TRAIN_BATCH_SIZE` | 512 | Features are small; can go up to 2048+ |
| `TRAIN_LEARNING_RATE` | 1e-3 | AdamW base LR, scaled by cosine-with-warmup |
| `TRAIN_NUM_EPOCHS` | 50 | Linear probe converges fast — 20–50 is plenty |
| `TRAIN_COSINE_WARMUP_EPOCHS` | 2 | Linear warmup length |
| `PREDICTION_THRESHOLD` | 0.99 | Minimum softmax top-prob to count a patch |

Path fields (`TRAIN_DIR`, `VALID_DIR`, `FEATURE_DIR`, `MODEL_DIR`,
`BACKEND_DIR`, …) are all derived from a single `PROJECT_ROOT` —
override any individually if you want to route outputs elsewhere.

---

## Troubleshooting

**`RuntimeError: Error(s) in loading state_dict ... ls1.gamma ...`**
You're building `UNIBackbone` without `init_values=1e-5`. The shipped
`backends/uni/backbone.py` includes the fix; if you've edited it and removed
the argument, put it back.

**`OSError: Not authenticated` or a 401 from HuggingFace on first
encode**
Either `hf auth login` hasn't been run, or your token was revoked.
Run `hf auth whoami` to confirm. The shell your Python process
inherits must see the token at `~/.cache/huggingface/token` or in
`$HF_TOKEN`.

**`ValueError: 'MahmoodLab/UNI2-h' not found` / 403**
Your account doesn't have the model grant. Visit
<https://huggingface.co/MahmoodLab/UNI2-h> with your institutional-email
account and accept the access agreement.

**CUDA out of memory during `encode_patches`**
Drop `ENCODE_BATCH_SIZE` in `config.py`. UNI2-h is a ViT-H — batch 8
is fine on 6 GB cards, batch 4 is fine on 4 GB cards.

**`FileNotFoundError: ... Training_Sets/Training`**
You haven't run `extract_patches` + `create_training_sets`.
`encode_patches` consumes their output.

**`FileNotFoundError` for `.svs` or `.xml` during `extract_patches`**
Put the SVS + matching XML files in `data/WSI/Training/`, or edit
`config.TRAIN_SLIDE_DIR` to point wherever they live.

**`FileNotFoundError: ... training.pt`**
You're trying to `train` / `evaluate` before `encode_patches` has
produced the feature cache. Run `encode_patches` first.

**`FileNotFoundError: ... uni2h_<head>_head.pt`**
You're trying to `evaluate` / `diagnose_wsi` before `train` has
written the head checkpoint. Run `train` first.

---

## Project layout

```
Cardiac-ACR/
├── README.md                           (this file)
├── pyproject.toml
├── .gitignore
├── cardiac_acr/
│   ├── __init__.py
│   ├── config.py                       Shared paths + preprocessing constants
│   │
│   ├── preprocessing/                  Backend-agnostic slide/tile/patch pipeline
│   │   ├── extract_patches.py          SVS + XML → per-class PNGs
│   │   ├── create_training_sets.py     80/20 per-slide train/val split
│   │   ├── preprocess_data_utils.py    Patch-library count helpers
│   │   ├── slide.py                    WSI loading and PNG extraction
│   │   ├── filter.py                   Tissue filtering (green, grays, pen marks)
│   │   ├── tiles.py                    Tile scoring, summaries, and extraction
│   │   ├── tileset_utils.py            Split tiles into 224×224 patches
│   │   ├── filter_patches.py           Filter patches by tissue content
│   │   └── openslide_compat.py         OpenSlide import + Windows DLL setup
│   │
│   ├── wsi/                            WSI-level inference and annotation
│   │   ├── diagnose.py                 End-to-end WSI inference (backend-aware)
│   │   ├── annotate_png.py             Color-coded patch annotations on PNGs
│   │   ├── annotate_svs.py             XML annotation generation for SVS viewers
│   │   └── count_1r2.py                1R2 rejection focus counting via segmentation
│   │
│   ├── utils/                          Shared helpers
│   │   ├── cardiac_utils.py
│   │   ├── util.py                     Low-level image/array utilities
│   │   └── check_dependencies.py       Runtime dependency check
│   │
│   └── backends/
│       └── uni/                        UNI2-h frozen backbone + linear/MLP head
│           ├── config.py               UNI-specific hyperparameters + shared re-export
│           ├── backbone.py             Frozen UNI2-h + .encode()
│           ├── encode_patches.py       One-shot feature cache builder
│           ├── features_dataset.py     FeatureCache + TensorDataset wrapper
│           ├── head.py                 LinearHead / MLPHead / build_head()
│           ├── train.py                Head training (AdamW + cosine + warmup)
│           └── evaluate.py             Per-class P/R/F1 + AUROC + confusion matrix
│
├── docs/
│   └── DEVELOPMENT_LOG.md              Design decisions + implementation notes
└── data/                               (gitignored)
    ├── WSI/
    ├── Patches/
    ├── DeepHistoPath/
    ├── Features/
    ├── Saved_Models/
    └── Backend/
```
