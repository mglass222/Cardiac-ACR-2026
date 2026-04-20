# Cardiac-ACR: Automated Cardiac Allograft Rejection Grading

Automated diagnosis of acute cellular rejection (ACR) in cardiac transplant biopsies using deep learning on whole-slide images (WSI).

The system processes `.svs` whole-slide images through a multi-stage pipeline — tissue extraction, filtering, tiling, patch classification with a ResNet-50 model, and slide-level diagnosis — to produce an ISHLT rejection grade (0R, 1R1A, 1R2, or 2R).

## Pipeline Overview

```
Input: .svs whole-slide images
                │
                ▼
┌─────────────────────────────┐
│  1. Slide Extraction        │  Convert SVS to scaled-down PNG images
│     (slide.py)              │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  2. Tissue Filtering        │  Remove background, pen marks, gray areas
│     (filter.py)             │  to isolate tissue regions
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  3. Tiling                  │  Divide filtered image into scored tiles,
│     (tiles.py)              │  save top-scoring tissue tiles
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  4. Patch Splitting         │  Split tiles into 224x224 patches
│     (tileset_utils.py)      │  for neural network input
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  5. Patch Filtering         │  Remove patches with <50% tissue
│     (filter_patches.py)     │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  6. Classification          │  ResNet-50 classifies each patch into
│     (cardiac_acr_diagnose_  │  one of 6 classes: 1R1A, 1R2, Healing,
│      wsi.py)                │  Hemorrhage, Normal, Quilty
└���─────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  7. Diagnosis               │  Aggregate patch predictions into
│     (cardiac_acr_diagnose_  │  slide-level ISHLT rejection grade
│      wsi.py + count_1r2.py) │  (0R, 1R1A, 1R2, 2R)
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  8. Annotation              │  Annotate PNG slides and generate
│     (annotate_png.py,       │  XML annotations for SVS viewer
│      annotate_svs.py)       │
└──────────────┘
```

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- OpenSlide (+ Python bindings)
- OpenCV (`cv2`)
- NumPy, Pillow, scikit-image, scipy, scikit-learn
- matplotlib (used by slide extraction)

### Install dependencies

```bash
pip install torch torchvision openslide-python opencv-python numpy Pillow scikit-image scipy scikit-learn matplotlib
```

### OpenSlide

OpenSlide has two parts:

1. The native OpenSlide library installed on your operating system
2. The Python bindings installed with `pip install openslide-python`

Install both pieces for your platform:

#### macOS

Install the native library with Homebrew:

```bash
brew install openslide
```

Then install the Python bindings:

```bash
pip install openslide-python
```

#### Linux

Install the native library with your package manager. On Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install libopenslide0 openslide-tools
```

Then install the Python bindings:

```bash
pip install openslide-python
```

If you are using another Linux distribution, install the equivalent
OpenSlide system package for your distro, then install
`openslide-python` with `pip`.

#### Windows

1. Install the Python bindings:

```powershell
pip install openslide-python
```

2. Download the OpenSlide Windows binaries from the OpenSlide project.
3. Extract them somewhere stable, for example:

```text
C:\openslide\
```

4. Set `OPENSLIDE_BIN_PATH` to the extracted `bin` directory:

```powershell
$env:OPENSLIDE_BIN_PATH="C:\openslide\bin"
```

For Command Prompt instead of PowerShell:

```bat
set OPENSLIDE_BIN_PATH=C:\openslide\bin
```

If you want the variable to persist across terminal sessions, set it in
your system environment settings instead of only in the current shell.

To verify the OpenSlide runtime before running the full pipeline:

```bash
python -m cardiac_acr.check_dependencies
```

This command checks the core inference dependencies, reports
platform-specific OpenSlide setup status, and prints clearer error
messages if a Python package or native library is missing.

Common setup failures:

- `No module named 'openslide'`: install `openslide-python`
- Native-library load error on macOS/Linux: install the system OpenSlide package
- Native-library load error on Windows: verify `OPENSLIDE_BIN_PATH` points to the folder containing the OpenSlide DLLs

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/mglass222/Cardiac-ACR-2026.git
cd Cardiac-ACR-2026
```

### 2. Place your data

```
data/
├── WSI/Test/                    Place .svs slide files here
├── Saved_Models/Weighted_Loss/  Place trained model (resnet50_ft) here
```

All other directories under `data/` are created automatically or populated during processing.

### 3. Place a font file

The tile summary visualizations require a TrueType font:

```
fonts/arial.ttf
```

Copy `Arial.ttf` from your system fonts, or use any compatible `.ttf` font.

### 4. Run

```bash
python -m cardiac_acr
```

The pipeline will:
1. Find all `.svs` files in `data/WSI/Test/`
2. Process each slide through the full pipeline
3. Output a rejection grade for each slide
4. Save annotated PNG images and SVS-compatible XML annotations

## Project Structure

```
Cardiac-ACR-2026/
│
├── cardiac_acr/
│   ├── __main__.py                   Package entry point for `python -m cardiac_acr`
│   ├── cardiac_acr_diagnose_wsi.py   Main inference pipeline
│   ├── cardiac_globals.py            Configuration — all paths and parameters
│   ├── cardiac_utils.py              Shared utility functions
│   │
│   ├── slide.py                      WSI loading and PNG extraction
│   ├── filter.py                     Tissue filtering (green channel, grays, pen marks)
│   ├── tiles.py                      Tile scoring, summaries, and extraction
│   ├── tileset_utils.py              Split tiles into 224x224 patches
│   ├── filter_patches.py             Filter patches by tissue content
│   ├── count_1r2.py                  1R2 rejection focus counting via segmentation
│   │
│   ├── annotate_png.py               Color-coded patch annotations on PNG slides
│   ├── annotate_svs.py               XML annotation generation for SVS viewers
│   │
│   ├── openslide_compat.py           Centralized OpenSlide import + Windows DLL setup
│   ├── util.py                       Low-level image/array utilities (from DeepHistoPath)
│   │
│   ├── training/                     Neural network training pipeline
│   │   ├── extract_patches.py        Extract labeled patches from annotated WSIs
│   │   ├── data_utils.py             Dataset helpers, class weights, dataloaders
│   │   ├── model.py                  ResNet-50 factory + staged unfreezing
│   │   ├── train.py                  Two-stage training (FC-only → fine-tune)
│   │   └── cross_validation.py       5-fold cross-validation
│   │
│   └── stats/                        Statistical analysis scripts
│       ├── _stats_utils.py           Shared helpers for slide-level stats
│       ├── dump_training_predictions.py
│       │                             Run trained model over training patches
│       ├── patch_level_stats.py      6-class confusion matrix + per-class ROC
│       ├── training_set_stats.py     Slide-level threshold-grid + 2R ROC (excl. AMR)
│       └── test_set_stats.py         Slide-level threshold-grid + 2R ROC + 4-class
│
├── data/                             All input/output data (not tracked in git)
├── docs/                             Planning notes, traces, and development docs
├── fonts/                            Font files for tile visualizations
├── .gitignore
└── README.md
```

Project notes and internal references now live under `docs/` so the
repository root stays focused on source, data, and the primary readme.

## Training and Statistics

The primary pipeline handles inference only. Two additional subpackages
under `cardiac_acr/` support model training and post-hoc evaluation.

### `cardiac_acr/training/` — Neural network training

Two-stage transfer learning on top of an ImageNet-pretrained ResNet-50:

1. **FC-only stage** — freeze the backbone, train a new
   `Dropout(0.5) → Linear` head until the FC+BN-only loss plateaus.
2. **Fine-tune stage** — unlock `layer3` and `layer4` with staged
   learning rates (`lr`, `lr/9`, `lr/3`) and continue training.

Class imbalance is handled with `CrossEntropyLoss(weight=class_weights)`
using sklearn's balanced formula. Augmentation is `ColorJitter` +
`RandomRotation` on the training split only.

```bash
# One-off patch extraction from annotated SVS files
python -m cardiac_acr.training.extract_patches

# Main training entry point — writes resnet50_ft into MODEL_DIR
python -m cardiac_acr.training.train

# 5-fold cross-validation
python -m cardiac_acr.training.cross_validation
```

### `cardiac_acr/stats/` — Evaluation

The stats scripts consume the trained model and the per-threshold
slide-classifier outputs to produce confusion matrices and ROC curves:

```bash
# Dump patch-level predictions across the training set (feeds the next script)
python -m cardiac_acr.stats.dump_training_predictions

# 6-class patch-level confusion matrix + one-vs-rest ROC curves
python -m cardiac_acr.stats.patch_level_stats

# Slide-level threshold-grid + 2R-vs-not-2R ROC (excludes AMR slides)
python -m cardiac_acr.stats.training_set_stats

# Slide-level threshold-grid + 2R ROC + 4-class confusion (test set)
python -m cardiac_acr.stats.test_set_stats
```

All four stats scripts share slide-level helpers in `_stats_utils.py`
and read their paths from `cardiac_globals.py`.

## Configuration

All configurable parameters are in `cardiac_acr/cardiac_globals.py`:

| Parameter | Default | Description |
|---|---|---|
| `ANNOTATION_SIZE` | 224 | Patch size (pixels) for neural network input |
| `SCALE_FACTOR` | 40 | Downscale factor for slide-to-PNG conversion |
| `PREDICTION_THRESHOLD` | 0.99 | Minimum confidence for a patch prediction to count |
| `BATCH_SIZE` | 200 | Batch size for GPU inference |
| `_1R2_DILATION_ITERS` | 28 | Dilation iterations for 1R2 focus counting |

## Classification Classes

| Class | Index | Description |
|---|---|---|
| 1R1A | 0 | Mild acute cellular rejection (focal) |
| 1R2 | 1 | Moderate acute cellular rejection (multifocal) |
| Healing | 2 | Healing rejection |
| Hemorrhage | 3 | Hemorrhage |
| Normal | 4 | Normal tissue |
| Quilty | 5 | Quilty lesion (benign endocardial infiltrate) |

## Diagnosis Logic

The slide-level ISHLT grade is determined by aggregating patch predictions:

| Condition | Grade |
|---|---|
| No 1R1A and no 1R2 foci | **0R** (no rejection) |
| 1R1A patches present, no 1R2 foci | **1R** (mild) |
| 1 focus of 1R2 | **1R** (moderate, focal) |
| 2+ foci of 1R2 | **2R** (moderate, multifocal) |

The 1R2 focus count uses morphological segmentation (`count_1r2.py`) rather than simple patch counting — nearby 1R2 patches are grouped into foci using dilation and connected component analysis.

## Multiprocessing

The pipeline uses Python `multiprocessing` at multiple stages for parallel processing across CPU cores:
- Slide extraction (one process per slide)
- Tissue filtering (one process per image chunk)
- Tile generation (one process per slide)
- Patch splitting (one process per tile chunk)
- Patch filtering (one process per patch chunk)

GPU inference (patch classification) uses batch processing on a single GPU.

## Output

After processing, results are saved to:

| Output | Location |
|---|---|
| Scaled PNG slides | `data/DeepHistoPath/training_png/` |
| Filtered images | `data/DeepHistoPath/filter_png/` |
| Tile images | `data/DeepHistoPath/tiles_png/` |
| 224x224 patches | `data/DeepHistoPath/tiles_png_split/` |
| Model predictions | `data/Backend/Saved_Databases/Weighted_Loss/` |
| Slide diagnoses | `data/Backend/Slide_Dx/Weighted_Loss/` |
| Annotated PNGs | `data/Backend/Annotated_Test_Slides/Weighted_Loss/` |
| SVS XML annotations | `data/WSI/TEST_SLIDE_ANNOTATIONS/Weighted_Loss/` |
