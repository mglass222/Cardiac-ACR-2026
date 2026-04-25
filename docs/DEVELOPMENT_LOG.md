# Cardiac-ACR-UNI Development Log

Development history, design decisions, and implementation notes for the
Cardiac-ACR-UNI project.

---

## Project genesis

Created as a successor to the original ResNet-50-based Cardiac-ACR
patch classifier (the 2019–2021 codebase that was later refactored as
`Cardiac-ACR-2026`). Motivation: pathology-specific foundation models
published since 2024 outperform ImageNet-pretrained ResNet-50 by
5–15 AUROC points on patch classification, especially on rare classes
where single-split validation gives no signal (notably Hemorrhage,
which only appears on slide 150 in the current training set).

The design took two phases:

1. **Original plan (library reuse)** — depend on the 2026 repo as a
   pip-installable dependency, reuse its preprocessing modules, add
   UNI-specific modules on top. This got the first end-to-end
   pipeline built quickly.
2. **Refactor to standalone** — copy the needed preprocessing /
   annotation / counting modules into this repo, rewrite imports, and
   drop the 2026 dependency entirely. The project is now self-contained.

Both phases are captured below.

---

## Design decisions

### Backbone: UNI2-h (not UNI v1, not DINOv3)

- **UNI2-h** (`MahmoodLab/UNI2-h`): ViT-H/14, 681M params, 1536-dim
  CLS embeddings. Trained on 200M H&E/IHC patches from 300k
  Mass General Brigham slides with the full DINOv2 recipe.
- **UNI v1** would have been a valid fallback (ViT-L/16, 300M params,
  1024-dim) but UNI2-h is the current SOTA with no access or hardware
  obstacles for this project.
- **DINOv3** (Meta, general-purpose) was considered and rejected: the
  arXiv benchmark *"Does DINOv3 Set a New Medical Vision Standard?"*
  (2509.06467) explicitly documents that DINOv3's natural-image
  features degrade on WSIs, and no pathology-DINOv3 variant had been
  released as of April 2026.
- **Virchow2 / Phikon v2 / H0-mini / GenBio-PathFM** were surveyed but
  not chosen — UNI2-h has the clearest track record and access was
  already granted.

### Standalone architecture (no external package dependencies)

The final layout keeps every module this project uses inside the
`cardiac_acr/` package. Benefits:

- The project can be cloned and run with a single `pip install -e .`
  — no out-of-tree source dependency.
- The patch extraction script, training data split, preprocessing
  filters, tile scoring, patch filtering, 1R2 segmentation, and
  annotation writers are all in one place and evolve together.
- Shared config: one `config.py` replaces the original project's
  `cardiac_globals.py` and holds both preprocessing paths *and* UNI
  hyperparameters.

Drawback: this repo and the older 2026 repo are now source-level
siblings, not linked. A bugfix in one has to be consciously ported to
the other if it's still active. That's acceptable because the
ResNet-50 baseline in the 2026 repo is an artifact to compare against,
not a live pipeline that needs ongoing maintenance.

### Frozen backbone + head (not fine-tuning)

For UNI-family backbones on pathology downstream tasks, the published
evaluation protocol is: freeze the backbone, extract features once,
train a linear probe on the cached features. This:

- Matches the canonical benchmark numbers so results are comparable.
- Avoids the memory cost of backprop through ViT-H (would not fit on
  an 8 GB 2070 SUPER at any useful batch size).
- Makes training fast — minutes on CPU, seconds on GPU.
- Decouples the expensive step (encoding) from the cheap iterative
  step (head training), so head hyperparameter sweeps are free.

### Head: linear probe first, MLP as a diagnostic fallback

Default: `Linear(1536, 6)`. Chosen because:

- It's the canonical UNI benchmark setup — reported numbers in the
  literature are directly comparable.
- Minimal overfitting risk on a small dataset.
- If linear underperforms, training a 2-layer MLP
  (`Linear(1536→512) → ReLU → Dropout(0.4) → Linear(512→6)`) is a
  2-minute diagnostic that tells us whether the features need
  non-linear combination (MLP >> linear) or the backbone is the
  ceiling (MLP ≈ linear). Both cases are useful signals.

Selectable via `HEAD_TYPE` in `config.py`.

### Feature cache format

Per-split `.pt` files at `data/Features/{training,validation}.pt`:

```python
{
    "features": FloatTensor [N, 1536],
    "labels":   LongTensor  [N],
    "classes":  list[str]               # index-aligned with labels
}
```

Features encoded in bf16 autocast during the forward pass, cast back
to fp32 before leaving GPU, and stored as fp32. Reason: bf16 saves
memory during encoding (the expensive step) but fp32 is the right
dtype for head optimization (the cheap, sensitive step). Storage cost
of fp32 is minimal — at 1536 × 4 bytes = 6 KB per patch, even
hundreds of thousands of patches fit easily in memory.

---

## Implementation notes

### Backbone loading quirk: `init_values=1e-5` needed

The UNI2-h model card's `timm.create_model(...)` snippet omits the
`init_values` argument. The downloaded weights, however, contain
LayerScale parameters (`blocks.N.ls{1,2}.gamma`). Loading without
`init_values` produces:

```
RuntimeError: Error(s) in loading state_dict for VisionTransformer:
    Unexpected key(s) in state_dict: "blocks.0.ls1.gamma", ...
```

Adding `init_values=1e-5` (the same value UNI v1's card specifies)
fixes it. Noted in `uni_backbone.py` with a comment because the
upstream model card does not.

### VRAM at inference

Smoke-tested with `torch.randn(4, 3, 224, 224)` on an RTX 2070 SUPER:
peak VRAM 2.88 GB in bf16 autocast. Plenty of headroom — the default
encode batch size is 16, which lands around 4–5 GB. If that ever
bites, drop `ENCODE_BATCH_SIZE` in `config.py` to 8 or 4.

### `allow_empty=True` for ImageFolder

The Hemorrhage class is only annotated on one slide (slide 150 in the
current training set, 147 regions). With the paper's 80/20 per-slide
split, that slide goes to training and the Validation/Hemorrhage
folder ends up empty. Without `allow_empty=True`,
`torchvision.datasets.ImageFolder` raises
`FileNotFoundError: Found no valid file for the classes Hemorrhage`.
The flag makes the loader accept empty class folders — the head still
learns Hemorrhage from training patches, but has no validation signal
for that class (which is an honest consequence of the data, not a bug).

### Class-balanced loss respects empty classes

`_class_weights()` in `train.py` uses the sklearn balanced formula
(`total / (num_classes * count_i)`) but assigns weight 0 to classes
with zero training patches. This prevents division by zero while
still giving reasonable weights to well-populated classes. For the
current split, all six classes have training patches — the zero-weight
path is defensive, not active.

### AdamW + cosine schedule + warmup

Upgrade from the original repo's plain Adam + fixed LR. AdamW +
cosine-with-warmup is the 2024–2026 default, simpler than staged
unfreezing, and well-suited to the small number of iterations we need
for a linear probe.

### Consolidated `config.py`

In the original Cardiac-ACR codebase, paths lived in
`cardiac_globals.py` and training hyperparameters were scattered
across `train.py` / notebook cells. Here they're all in one
`config.py`, grouped by purpose (preprocessing paths, patch library,
UNI outputs, classes, backbone, encoding, head, training). A single
file to change when retargeting the project to different data or
hardware.

### V1 scope: diagnosis only, no annotation writing in diagnose_wsi

`diagnose_wsi.py` stops after writing the per-slide diagnosis pickle.
The `annotate_png` and `annotate_svs` modules in this package are
fully ported from the original codebase and read from
`config.SAVED_DATABASE_DIR` — which is the same directory
`diagnose_wsi` writes to — so adding them to the main loop is a
one-line follow-up. Left out of V1 so the first end-to-end run
focuses on producing the diagnostic grade.

### Standalone refactor mechanics

Turning the library-dependent first build into a self-contained
package took five concrete steps:

1. **Copy preprocessing modules flat** into `cardiac_acr/` — no
   `core/` subpackage, no `preprocessing/` subpackage. Files live at
   the top level next to the UNI-specific modules. Modules copied:
   `slide`, `filter`, `tiles`, `tileset_utils`, `filter_patches`,
   `count_1r2`, `annotate_png`, `annotate_svs`, `openslide_compat`,
   `util`, `cardiac_utils`, `check_dependencies`, `extract_patches`,
   `create_training_sets`. The patch-library count helpers (the old
   `training/data_utils.py`) were renamed to `preprocess_data_utils.py`
   so the filename clearly reflects its role.

2. **Merge `cardiac_globals.py` into `config.py`** — not a second
   file. The original project's globals module contained only
   constants and paths, and keeping it separate would have meant
   every module had to choose between `from X import config` and
   `from X import globals`. One file is simpler.

3. **Rewrite imports via `sed`** — a small set of substitutions
   covers every form actually used in the codebase:
   ```
   from cardiac_acr import cardiac_globals as cg  →  from cardiac_acr import config as cg
   from cardiac_acr.training import X              →  from cardiac_acr import X
   from cardiac_acr.training.X import Y            →  from cardiac_acr.X import Y
   from cardiac_acr.X import Y                     →  from cardiac_acr.X import Y  (e.g. util.Time)
   from cardiac_acr import X                       →  from cardiac_acr import X
   import cardiac_acr.X                            →  import cardiac_acr.X
   ```
   The submodule form (`from cardiac_acr.util import Time`) was the
   easiest to miss — the first sed pass only handled leading-line
   `from cardiac_acr import …` and let that pattern through. Caught
   on the import-walk verification.

4. **Drop the `cardiac-acr` dependency** from `pyproject.toml`; add
   the previously transitive deps (`opencv-python`, `scipy`,
   `scikit-image`, `openslide-python`) explicitly.

5. **Verify isolation** with two tests: (a) `pkgutil.walk_packages`
   imports every submodule under `cardiac_acr` with no errors,
   (b) `grep -rn 'cardiac_acr[^_]'` on the entire package directory
   returns no matches. Both passed on the final pass.

A gotcha worth remembering for future refactors of this kind: running
`pip uninstall cardiac-acr` (the name of the legacy package) via an
alias/typo can also uninstall `cardiac-acr-uni` if pip is ever given
a prefix match; after this refactor the UNI package was silently
uninstalled and had to be reinstalled. The `pip install -e .` line
in the README setup is the canonical recovery.

---

## Execution flow

### Module-level on import
`uni_backbone.UNIBackbone.__init__` → `timm.create_model("hf-hub:MahmoodLab/UNI2-h", ...)`
→ triggers `huggingface_hub.snapshot_download` if the weights aren't cached.
The model is moved to GPU and `eval()`'d; its parameters have
`requires_grad = False`.

### Patch preparation
```
extract_patches.main()              ← SVS + XML → PNGs under config.OPENSLIDE_DIR
create_training_sets.main()         ← split into config.TRAIN_DIR / config.VALID_DIR
```
Run in that order; they share no state beyond the on-disk patch tree.

### `encode_patches.main()` call chain
```
main()
├── UNIBackbone()                                   ← loads UNI2-h, ~5 GB VRAM
├── for split in [Training, Validation]:
│   ├── _build_transform()                          ← Resize → CenterCrop → ToTensor → Normalize(ImageNet)
│   ├── datasets.ImageFolder(split_dir, allow_empty=True)
│   ├── DataLoader(batch=ENCODE_BATCH_SIZE, num_workers=ENCODE_NUM_WORKERS)
│   └── for batch in loader:
│       └── backbone.encode(batch)                  ← bf16 autocast forward, returns fp32 on CPU
│   └── torch.save({features, labels, classes} → data/Features/<split>.pt)
```

### `train.main()` call chain
```
main() → train_head()
├── FeatureCache.load(TRAINING_FEATURES_PATH)       ← entire split loaded into memory
├── FeatureCache.load(VALIDATION_FEATURES_PATH)
├── _class_weights(labels)                          ← balanced CE weights, 0 for empty classes
├── build_head(HEAD_TYPE)                           ← LinearHead or MLPHead
├── AdamW + cosine-with-warmup LR schedule
├── for epoch in range(TRAIN_NUM_EPOCHS):
│   ├── training loop  (model.train(), backprop through head only)
│   └── validation loop (model.eval(), torch.no_grad())
└── _save_checkpoint(best model by val_acc → data/Saved_Models/UNI_Head/uni2h_<head>_head.pt)
```

### `evaluate.main()` call chain
```
main() → evaluate()
├── FeatureCache.load(VALIDATION_FEATURES_PATH)
├── load_head_checkpoint()
├── _predict(model, features, device)
├── sklearn.metrics.classification_report          ← per-class P/R/F1
├── _one_vs_rest_auroc()                            ← per-class AUROC
└── sklearn.metrics.confusion_matrix
```

### `diagnose_wsi.main()` call chain
```
main()
├── cardiac_utils.get_test_slide_numbers()
├── slide.multiprocess_training_slides_to_images()          ← SVS → PNG
├── filter.multiprocess_apply_filters_to_images()           ← tissue filtering
├── tiles.multiprocess_filtered_images_to_tiles()           ← tile scoring / extraction
├── UNIBackbone()
├── load_head_checkpoint()
├── for slide in slides_to_process:
│   ├── tileset_utils.process_tilesets_multiprocess()       ← 224x224 patches
│   ├── filter_patches_multiprocess()                       ← <50% tissue removal
│   ├── classify_patches()                                  ← UNI2-h encode + head + softmax
│   ├── threshold_predictions()                             ← drop below PREDICTION_THRESHOLD
│   └── diagnose()
│       └── count_1r2.main()                                ← 1R2 focus segmentation
└── (V2 TODO: annotate_png.main() / annotate_svs.main())
```

---

## Repository layout

```
Cardiac-ACR-UNI/
├── README.md
├── pyproject.toml
├── .gitignore
├── cardiac_acr/
│   ├── __init__.py
│   ├── config.py                       Consolidated paths + hyperparameters
│   │
│   ├── extract_patches.py              SVS + XML → per-class PNGs
│   ├── create_training_sets.py         80/20 per-slide train/val split
│   ├── preprocess_data_utils.py        Patch-library count helpers (used by create_training_sets)
│   │
│   ├── slide.py                        WSI loading and PNG extraction
│   ├── filter.py                       Tissue filtering (green channel, grays, pen marks)
│   ├── tiles.py                        Tile scoring, summaries, and extraction
│   ├── tileset_utils.py                Split tiles into 224×224 patches
│   ├── filter_patches.py               Filter patches by tissue content
│   ├── count_1r2.py                    1R2 rejection focus counting via segmentation
│   │
│   ├── annotate_png.py                 Color-coded patch annotations on PNG slides
│   ├── annotate_svs.py                 XML annotation generation for SVS viewers
│   │
│   ├── openslide_compat.py             OpenSlide import + Windows DLL setup
│   ├── util.py                         Low-level image/array utilities
│   ├── cardiac_utils.py                Shared utility functions
│   ├── check_dependencies.py           Runtime dependency check
│   │
│   ├── uni_backbone.py                 Frozen UNI2-h + .encode()
│   ├── encode_patches.py               One-shot feature cache builder
│   ├── features_dataset.py             FeatureCache + TensorDataset wrapper
│   ├── head.py                         LinearHead / MLPHead / build_head()
│   ├── train.py                        Head training (AdamW + cosine + warmup)
│   ├── evaluate.py                     Per-class P/R/F1 + AUROC + confusion matrix
│   └── diagnose_wsi.py                 End-to-end WSI inference
│
├── docs/
│   └── DEVELOPMENT_LOG.md              (this file)
└── data/                               (gitignored — all inputs and outputs live here)
```

---

## Session timeline

All dates 2026-04-21 unless noted.

1. **Design phase** — chose frozen backbone, linear probe first,
   UNI2-h over DINOv3 / UNI v1 / Virchow2 / Phikon v2 / GenBio-PathFM.
   Plan captured and iterated in a shared notes file.
2. **HF access granted** — user updated HF primary email to Duke
   institutional (UNI2-h rejects @gmail applicants), re-ran
   `hf auth login` after a token-in-chat incident → token rotated.
3. **Build phase (library-dependent form)** — nine tracked tasks:
   - Install timm + huggingface_hub
   - Verify HF auth + prime model download (2.7 GB, 5 files)
   - Add `pyproject.toml` to the 2026 repo for editable install
   - Scaffold `Cardiac-ACR-UNI/` with pyproject + gitignore + package
   - Write `uni_backbone.py` + smoke test (hit + fixed `init_values` omission)
   - Write `encode_patches.py`
   - Write `head.py` + `features_dataset.py` + `train.py`
   - Write `evaluate.py`
   - Write `diagnose_wsi.py`
4. **Standalone refactor** — user requested removing the external
   dependency so the project is self-contained. Six tracked tasks:
   - Copy preprocessing / annotation / counting modules flat into
     `cardiac_acr/` (no subpackage — per user direction).
   - Rewrite imports in the copied modules (initial sed missed the
     `from cardiac_acr.util import Time` submodule form — caught on
     the import walk, fixed with a targeted second pass).
   - Rewrite imports in the top-level UNI modules (`config`,
     `prepare_patches`, `diagnose_wsi`).
   - Drop the `cardiac-acr` dependency from `pyproject.toml`; pull in
     previously transitive deps (`opencv-python`, `scipy`,
     `scikit-image`, `openslide-python`) directly.
   - Verify: `pkgutil.walk_packages` imports every submodule cleanly;
     `grep -rn 'cardiac_acr[^_]'` returns no matches in the package.
   - Update README and DEVELOPMENT_LOG for the new architecture.
   Smoke-test post-refactor: UNI2-h backbone still loads (681M params,
   (4, 1536) output, 2.88 GB peak VRAM in bf16).

---

## What's next (V2 candidates — pre-unification snapshot)

This list reflects the project's state *before* the unification
refactor. Items 3–6 still stand. Items 1–2 were partly addressed by
the refactor; the updated follow-up list lives at the bottom of this
document under "Post-unification follow-ups".

1. **Wire up `annotate_png` / `annotate_svs`** in the per-slide loop
   so PNG overlays and SVS region annotations get written alongside
   the diagnosis pickles. *(Partly addressed: the slot exists in
   `wsi/diagnose.run()` but the annotation modules still read paths
   from shared `config` instead of the per-backend `BackendClassifier`;
   see follow-ups.)*
2. **Head-to-head evaluation** against the original ResNet-50 checkpoint
   on the same validation split. *(Now trivially runnable: both
   backends live in the same package and share the validation split;
   just hasn't been executed yet.)*
3. **Per-slide confidence calibration** — UNI features + linear probe
   may produce very peaked softmax distributions, so
   `PREDICTION_THRESHOLD = 0.99` may need re-tuning.
4. **Stain-jitter or Macenko normalization** as an encoding-time
   augmentation (the only place augmentation matters in the frozen
   pipeline). Typically gives modest AUROC gains in H&E.
5. **Multi-crop / test-time augmentation** at encode time: average
   features over 4–8 random crops per patch. Very cheap with a frozen
   backbone; sometimes moves the needle.
6. **Periodic check for pathology-DINOv3 variants** — no such model
   existed as of this session. If one ships, swapping it in is a
   one-line change to `backends/uni/backbone.py` plus an `EMBED_DIM`
   update in `backends/uni/config.py`.

---

## 2026-04-21 — Shared `Image-Data/` across projects

Consolidated the large, backend-agnostic inputs (`WSI/`, `DeepHistoPath/`,
`Patches/`) into a sibling directory `../Image-Data/` used by both
`Cardiac-ACR-UNI` and `Cardiac-ACR-Resnet` (the renamed `Cardiac-ACR-2026`
repo). Each project's local `data/`
still owns its *outputs* — `Backend/`, `Saved_Models/`, and (UNI only)
`Features/`.

Both configs (`cardiac_acr/config.py`,
`cardiac_acr/cardiac_globals.py`) now define:

```python
SHARED_DATA_DIR = os.environ.get(
    "CARDIAC_ACR_SHARED_DATA_DIR",
    os.path.join(os.path.dirname(PROJECT_ROOT), "Image-Data"),
)
```

and root `WSI_DIR`, `DEEP_HISTO_DIR`, `PATCH_DIR` there. Everything else
(models, pickles, annotated slides, feature caches) stays under local
`DATA_DIR`. Env var override lets the shared tree be relocated without
editing code.

Rationale: raw WSIs and the derived patch library are large and shared
by both projects — duplicating them wasted disk. Outputs diverge per
project and must stay separate.

---

## Plan: unify into a single project with pluggable backends

The preprocessing pipeline (`slide`, `tiles`, `filter`,
`filter_patches`, `extract_patches`, `create_training_sets`), the
WSI-level inference/annotation code (`diagnose_wsi`, `annotate_png`,
`annotate_svs`, `count_1r2`), and the utilities (`cardiac_utils`,
`util`, `check_dependencies`, `openslide_compat`, `tileset_utils`) are
duplicated nearly verbatim between `Cardiac-ACR-UNI` and
`Cardiac-ACR-Resnet`. Only training, evaluation, and the model
definition meaningfully differ:

- **2026 / ResNet-50**: end-to-end patch training with weighted
  cross-entropy, `ImageFolder` data loading, 5-fold cross-validation,
  per-threshold slide-level stats.
- **UNI**: frozen UNI2-h backbone → cached per-patch feature tensors
  → linear or MLP head trained in seconds on cached tensors.

**Plan**: collapse the two repos into a single package, `cardiac_acr`,
with a `backends/{resnet,uni}/` subpackage per model and a thin backend
contract so the WSI-level code is model-agnostic.

### Target layout

```
cardiac_acr/
├── __init__.py  __main__.py
├── config.py                         # shared: paths, CLASS_NAMES, IMAGENET_*,
│                                     #         PATCH_SIZE, PREDICTION_THRESHOLD,
│                                     #         SHARED_DATA_DIR
├── preprocessing/                    # backend-agnostic
│   ├── slide.py  tiles.py  tileset_utils.py
│   ├── filter.py  filter_patches.py
│   ├── extract_patches.py  create_training_sets.py
│   └── openslide_compat.py
├── wsi/                              # WSI-level; consumes a classifier callable
│   ├── diagnose.py
│   ├── annotate_png.py  annotate_svs.py
│   └── count_1r2.py
├── utils/
│   └── cardiac_utils.py  util.py  check_dependencies.py
├── backends/
│   ├── __init__.py                   # Backend protocol + registry
│   ├── resnet/
│   │   ├── config.py  model.py  data_utils.py
│   │   ├── train.py  cross_validation.py  evaluate.py
│   │   └── classifier.py
│   └── uni/
│       ├── config.py  backbone.py  head.py
│       ├── features_dataset.py  encode_patches.py
│       ├── train.py  evaluate.py
│       └── classifier.py
└── stats/
```

### Backend contract

Each backend exposes:

```python
def train(args) -> None: ...
def evaluate(args) -> dict: ...
def load_classifier(ckpt_path) -> Callable[[Tensor[B,3,224,224]], Tensor[B,C]]
```

`wsi/diagnose.py` takes a classifier callable as input and knows
nothing about whether it's a ResNet end-to-end forward pass or a UNI
backbone → head chain. Annotation, ROI filtering, and per-slide pickle
writing are untouched.

### CLI

```
python -m cardiac_acr preprocess {tiles,filter,extract-patches,split}
python -m cardiac_acr train        --backend {resnet,uni}
python -m cardiac_acr evaluate     --backend {resnet,uni} --checkpoint PATH
python -m cardiac_acr diagnose-wsi --backend {resnet,uni} --checkpoint PATH
python -m cardiac_acr count-1r2    ...
```

### Config split

- Top-level `config.py`: paths (`WSI_DIR`, `DEEP_HISTO_DIR`,
  `PATCH_DIR`, `BACKEND_DIR`, `SHARED_DATA_DIR` + env override),
  `CLASS_NAMES`, `NUM_CLASSES`, `IMAGENET_MEAN/STD`, `PATCH_SIZE`,
  `ANNOTATION_SIZE`, `SCALE_FACTOR`, `PREDICTION_THRESHOLD`,
  `BATCH_SIZE`, `_1R2_DILATION_ITERS`, `FONT_PATH`, `OPENSLIDE_BIN_PATH`.
- `backends/resnet/config.py`: `TRAIN_INPUT_SIZE`, `TRAIN_BATCH_SIZE`,
  `TRAIN_LEARNING_RATE`, `TRAIN_NUM_EPOCHS`, `TRAIN_DEFAULT_MODEL_NAME`,
  `MODEL_DIR`, `SAVED_DATABASE_DIR`, cross-val dirs, stats spreadsheet
  paths.
- `backends/uni/config.py`: `UNI_MODEL_ID`, `EMBED_DIM`, `INPUT_SIZE`,
  `ENCODE_BATCH_SIZE`, `ENCODE_NUM_WORKERS`, `HEAD_TYPE`,
  `HEAD_DROPOUT`, `HEAD_HIDDEN_DIM`, `TRAIN_*` (head-training),
  `FEATURE_DIR`, `MODEL_DIR`, `*_FEATURES_PATH`.

### Migration order

Each step must leave a working tree.

1. **Start from `Cardiac-ACR-UNI`.** It's already standalone with
   cleaner imports and post-refactor module boundaries.
2. **Carve out `backends/uni/`.** Move `uni_backbone.py` → `backbone.py`,
   `head.py`, `encode_patches.py`, `features_dataset.py`, `train.py`,
   `evaluate.py` into `backends/uni/`. Split UNI-specific constants out
   of `config.py` into `backends/uni/config.py`. Fix imports. Smoke-test
   UNI end-to-end.
3. **Reorganize shared code** into `preprocessing/`, `wsi/`, `utils/`.
   Pure rename + import-rewrite commit; no logic changes.
4. **Port `backends/resnet/`** from `Cardiac-ACR-Resnet`:
   `training/model.py`, `training/train.py`,
   `training/cross_validation.py`, `training/data_utils.py`,
   `training/extract_patches.py` (diff against UNI's copy — should be
   identical or near-identical), `stats/*`.
5. **Make `wsi/diagnose.py` backend-agnostic.** Refactor to take a
   `classify: Callable[[Tensor], Tensor]` argument. Provide
   `load_classifier` in both backends. Wire up the CLI dispatch.
6. **Rename** the project directory to `Cardiac-ACR` and archive
   `Cardiac-ACR-Resnet`. `Image-Data/` is already shared; no data
   movement required.

### Tradeoffs / risks

- One-time refactor touching ~20+ files; will break any out-of-tree
  scripts pointing into either repo.
- The train/evaluate paths stay as two separate files per backend;
  abstraction only lives at the WSI-inference boundary where it's
  cheap. This avoids a premature "universal trainer".
- If ResNet later wants the cached-feature flow (train a head on
  pre-computed features), that's a future add, not a blocker.
- Neither project has tests, so verification is smoke-test only:
  UNI encode+train+diagnose one slide; ResNet train one epoch on a
  small split and diagnose one slide.

---

## 2026-04-21 — Step 2 complete: `backends/uni/` carved out

Cardiac-ACR-Resnet is the renamed Cardiac-ACR-2026. That naming is the
final one going into the unified project.

UNI-specific modules moved into `cardiac_acr/backends/uni/`:

- `uni_backbone.py` → `backends/uni/backbone.py` (`UNIBackbone`)
- `head.py` → `backends/uni/head.py`
- `encode_patches.py` → `backends/uni/encode_patches.py`
- `features_dataset.py` → `backends/uni/features_dataset.py`
- `train.py` → `backends/uni/train.py`
- `evaluate.py` → `backends/uni/evaluate.py`

UNI-specific hyperparameters and paths moved out of `config.py` into
`backends/uni/config.py`: `UNI_MODEL_ID`, `EMBED_DIM`, `INPUT_SIZE`,
`ENCODE_BATCH_SIZE`, `ENCODE_NUM_WORKERS`, `HEAD_TYPE`, `HEAD_DROPOUT`,
`HEAD_HIDDEN_DIM`, `TRAIN_*` (head-training), `FEATURE_DIR`, `MODEL_DIR`,
`TRAINING_FEATURES_PATH`, `VALIDATION_FEATURES_PATH`.

`backends/uni/config.py` re-exports shared constants via
`from cardiac_acr.config import *` so modules can keep a single
import (`config as uni_cfg`) and reach both shared and UNI-specific
names. Downstream callers that previously did
`from cardiac_acr import config as uni_cfg` now do
`from cardiac_acr.backends.uni import config as uni_cfg` and
everything else is unchanged.

`diagnose_wsi.py` imports updated for the backend-scoped paths but
otherwise untouched; the backend-agnostic refactor (step 4) is next.

User-facing command paths updated in README and module docstrings:

```
python -m cardiac_acr.backends.uni.encode_patches
python -m cardiac_acr.backends.uni.train
python -m cardiac_acr.backends.uni.evaluate
```

Verification: `pkgutil.walk_packages` imports all 27 modules clean;
`uni_cfg.EMBED_DIM == 1536`, shared `WSI_DIR` re-exports via `uni_cfg`,
UNI-specific `MODEL_DIR` still points at `data/Saved_Models/UNI_Head/`.

---

## 2026-04-21 — Step 3 complete: shared code moved into subpackages

Backend-agnostic modules regrouped under three subpackages:

- `preprocessing/` — `slide`, `tiles`, `tileset_utils`, `filter`,
  `filter_patches`, `extract_patches`, `create_training_sets`,
  `preprocess_data_utils`, `openslide_compat`.
- `wsi/` — `diagnose` (was `diagnose_wsi`), `annotate_png`,
  `annotate_svs`, `count_1r2`.
- `utils/` — `cardiac_utils`, `util`, `check_dependencies`.

`diagnose_wsi.py` renamed to `wsi/diagnose.py`. No logic changes — only
imports and the `-m` command paths were rewritten. Config stays at the
package top level as `cardiac_acr.config`.

Import rewrites ran mechanically across 13 files using a one-shot
regex pass; user-facing command strings in 5 files updated too.
`pkgutil.walk_packages` imports all 30 modules clean.

New command paths:

```
python -m cardiac_acr.preprocessing.extract_patches
python -m cardiac_acr.preprocessing.create_training_sets
python -m cardiac_acr.backends.uni.encode_patches
python -m cardiac_acr.backends.uni.train
python -m cardiac_acr.backends.uni.evaluate
python -m cardiac_acr.wsi.diagnose
python -m cardiac_acr.utils.check_dependencies
```

README project-layout tree and command examples updated to match.

---

## 2026-04-21 — Step 4 complete: `backends/resnet/` ported

Cardiac-ACR-Resnet's training code copied into
`cardiac_acr/backends/resnet/`:

- `model.py` — `build_resnet()`, `unfreeze_layers()` (ResNet 18/34/50/101/152 factory with the modern `weights=` API).
- `train.py` — two-phase training (FC+BN only → unfreeze `layer3`/`layer4` with lr/9 and lr/3).
- `cross_validation.py` — 5-fold CV runner, writes per-fold checkpoints and aggregated predictions pickle.
- `data_utils.py` — `count_classes/patches`, `epoch_steps`, `class_weights`, `datasets_normalization`, `dataloaders` (the ResNet path trains on an `ImageFolder`, so the DataLoader helpers UNI dropped still matter here).
- `stats/` — `_stats_utils.py`, `dump_training_predictions.py`, `patch_level_stats.py`, `test_set_stats.py`, `training_set_stats.py`. Threshold-sweep CSVs and ROC/AUROC plots are written out under `TRAIN_SET_ANALYSIS_DIR` / `TEST_SET_ANALYSIS_DIR`.

`backends/resnet/config.py` re-exports shared constants
(`from cardiac_acr.config import *`) and adds ResNet-specific paths
and hyperparameters. All ResNet outputs nest under a `Weighted_Loss/`
subfolder so they don't collide with UNI outputs in the same `data/`
tree: `Saved_Models/Weighted_Loss/`,
`Backend/{Saved_Databases,Slide_Dx,Annotated_Test_Slides,Test_Slide_Predictions}/Weighted_Loss/`,
`WSI/TEST_SLIDE_ANNOTATIONS/Weighted_Loss/`. Cross-val workspace
(`CROSS_VAL_DIR`), spreadsheets, and patch-level prediction pickle are
ResNet-only and stay unsuffixed.

`extract_patches.py`, `create_training_sets.py`,
`preprocess_data_utils.py` are **not** re-ported — UNI's
`preprocessing/` copies are the canonical versions and both backends
share them. The ResNet repo's `cardiac_globals.py` -> `config.py`
mapping took care of every old import with a mechanical regex pass.

`matplotlib` added to `pyproject.toml` dependencies (needed by
ResNet stats plots).

Verification: `pkgutil.walk_packages` imports all 42 modules clean.
`rcfg.MODEL_DIR` resolves to
`data/Saved_Models/Weighted_Loss`; shared `TRAIN_DIR` resolves into
`../Image-Data/Patches/Training_Sets/Training` as expected.

Deferred to step 5 (backend-agnostic WSI diagnose): the ported
`cardiac_acr_diagnose_wsi.py` from the ResNet repo is not yet merged
into `wsi/diagnose.py`. That's the whole point of the next step — one
`diagnose.py` module that takes a classifier callable instead of
hard-wiring a specific backend.

---

## 2026-04-21 — Step 5 complete: backend-agnostic WSI diagnose + CLI

### Backend contract

`cardiac_acr/backends/__init__.py` now defines a
`BackendClassifier` dataclass and a `load_classifier(name, device,
checkpoint_path)` dispatcher:

```python
@dataclass
class BackendClassifier:
    name: str
    classify: Callable[[torch.Tensor], torch.Tensor]  # (B,3,H,W) → (B,C) logits
    classes: Sequence[str]
    transform: Callable                               # PIL → Tensor[3,H,W]
    device: torch.device
    saved_database_dir: str
    slide_dx_dir: str
    annotated_png_dir: str
    test_slide_predictions_dir: str
```

Each backend exposes a `classifier.py` with `load_classifier(device,
checkpoint_path=None) -> BackendClassifier`:

- `backends/uni/classifier.py`: loads UNI2-h + the trained head,
  returns a `classify` closure that runs backbone.encode + head.
- `backends/resnet/classifier.py`: `torch.load`s the full ResNet model
  pickle and returns a `classify` closure that calls it directly.

Both fill `BackendClassifier.{saved_database_dir, slide_dx_dir,
annotated_png_dir, test_slide_predictions_dir}` with the paths from
their own `config.py` so UNI writes unsuffixed and ResNet writes
under `Weighted_Loss/`.

### Refactored `wsi/diagnose.py`

- `run(backend, checkpoint_path=None)` — new entry point that loads
  a `BackendClassifier` and drives the whole pipeline.
- `classify_patches(slide_number, classifier)` — uses
  `classifier.classify` and `classifier.transform`; no backbone / head
  knowledge left.
- `diagnose(slide_number, classifier)` — reads / writes from
  `classifier.saved_database_dir` / `.slide_dx_dir`; iterates
  `classifier.classes` instead of hard-coding the 6 names.
- `count_1r2.main(slide_number, saved_database_dir)` — count_1r2 was
  the one shared module that read `cg.SAVED_DATABASE_DIR` directly;
  now it takes the dir as an explicit arg, passed through from
  `diagnose()`.

Kept the legacy `cg.SAVED_DATABASE_DIR` / `SLIDE_DX_DIR` /
`ANNOTATED_PNG_DIR` / `TEST_SLIDE_PREDICTIONS_DIR` /
`TEST_SLIDE_ANNOTATIONS_DIR` defaults in `config.py` because
`wsi/annotate_png.py`, `wsi/annotate_svs.py`, and
`utils/cardiac_utils.py` still read them. Those modules aren't yet
wired into `diagnose.run()` (V1 limitation), so the stale read is
harmless for now. When annotations are wired in they'll need the same
treatment as `count_1r2`.

### Unified CLI

New `cardiac_acr/__main__.py` exposes a single entry point:

```
python -m cardiac_acr preprocess {extract-patches, split}
python -m cardiac_acr train        --backend {uni, resnet}
python -m cardiac_acr evaluate     --backend {uni, resnet} [--checkpoint PATH]
python -m cardiac_acr diagnose-wsi --backend {uni, resnet} [--checkpoint PATH]
python -m cardiac_acr check-deps
```

Direct module entry points (`python -m cardiac_acr.backends.uni.train`
etc.) still work — `__main__.py` just forwards to them. Keeping both
so the CLI is convenient and the module paths remain addressable for
scripts / debugging.

### Verification

`pkgutil.walk_packages` imports all 45 modules clean.
`load_classifier("bogus")` raises `ValueError` as expected.
`python -m cardiac_acr --help` lists every subcommand.
`python -m cardiac_acr diagnose-wsi --help` exposes `--backend` and
`--checkpoint`.

---

## 2026-04-21 — Step 6 complete: package renamed to `cardiac_acr`

The Python package was renamed in place:

- `cardiac_acr_uni/` → `cardiac_acr/` (directory rename).
- Every `cardiac_acr_uni` reference in source, docs, README, and
  pyproject rewritten to `cardiac_acr` via a one-shot regex pass (40
  files touched).
- `pyproject.toml`: `name = "cardiac-acr"`, `version = "0.2.0"`,
  description updated to mention both backends.
- `cardiac_acr/__init__.py` and README title updated to reflect the
  unified scope.
- Venv re-installed the package: `pip install -e .` picked up the new
  name. Import of the old `cardiac_acr_uni` now fails, as expected.

**What was intentionally not renamed**: the repo's containing
directory is still `Cardiac-ACR-UNI/`. Renaming it would have broken
the in-place `.venv` (absolute paths in activate scripts, shebangs,
etc.). The README still references `Cardiac-ACR-UNI` as the repo
directory — user can rename at their leisure and recreate the venv.

**Archive decision**: `Cardiac-ACR-Resnet/` is kept in place as a
sibling directory. It's no longer imported by anything in the new
`cardiac_acr` package — the ResNet code was copied in (not linked) —
so it's functionally archived already. Deletion / relocation is left
to the user to avoid touching directories outside the current project
without explicit authorization.

Final package layout:

```
cardiac_acr/
├── __init__.py  __main__.py
├── config.py                  # shared paths + preprocessing constants
├── preprocessing/             # slide / tile / patch pipeline
├── wsi/                       # backend-agnostic diagnosis + annotation
├── utils/                     # cardiac_utils, util, check_dependencies
└── backends/
    ├── __init__.py            # BackendClassifier + load_classifier()
    ├── uni/                   # UNI2-h frozen backbone + head
    └── resnet/                # ResNet-50 end-to-end
        └── stats/             # threshold-sweep CSVs, ROC/AUROC plots
```

45 modules import clean. The full refactor is complete.

---

## 2026-04-21 — Repo directory rename

User attempted renaming the containing repo directory from
`Cardiac-ACR-UNI/` to `Cardiac-ACR/`, then reverted it to finish this
dev-log update. The `.venv` inside would need recreation after a rename
because absolute paths are baked into activate scripts and shebangs:

```bash
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

No code changes are needed — nothing inside the package references the
containing directory name.

---

## 2026-04-21 — Repo directory renamed

User completed the rename: `Cardiac-ACR-UNI/` → `Cardiac-ACR/`. README
references updated (clone target, data-path examples, `cd` command,
project-layout tree). No code changes required. `.venv` needs
recreation per the prior entry if the rename was done in place.

The stale `cardiac_acr_uni.egg-info/` at the repo root is a leftover
from the pre-rename editable install; `pip install -e .` after venv
recreation will replace it with `cardiac_acr.egg-info/`, and the old
directory can be removed.

---

## 2026-04-21 — Venv rebuilt, tech-debt items 1–3 resolved

**Venv recreated** against the renamed directory — old `.venv` had
`Cardiac-ACR-UNI` baked into `activate`/shebangs. After
`rm -rf .venv cardiac_acr_uni.egg-info`, new venv + `cu124` torch +
`pip install -e .`, all 31 modules import clean; torch sees the
2070 SUPER.

**Tech debt 1 + 2 (parametrize annotate/CSV helpers).**

- `wsi/annotate_png.py`: `annotate_png(slide_number, saved_database_dir, annotated_png_dir)` and `main(...)` mirror the `count_1r2.main` signature. Reads pickle from `saved_database_dir`, writes PNG under `annotated_png_dir/CONFIDENCE <pct>%/<slide>.png`.
- `wsi/annotate_svs.py`: `annotate_slide(slide_number, saved_database_dir, test_slide_annotations_dir)` and `load_diagnoses(slide_number, saved_database_dir)` take explicit paths.
- `utils/cardiac_utils.py`: `model_prediction_dict_to_csv(slide_number, saved_database_dir, test_slide_predictions_dir)` and `slide_dx_to_csv(slide_dx_dict, filename, slide_dx_dir)` take explicit paths.
- Fixed several pre-existing `cg.<DIR> + filename` concatenation bugs in the same functions (missing path separator) — they were dormant because the helpers weren't being called, but the parametrized versions would hit the same bugs the first time they were turned on.

**`BackendClassifier`** gained a `test_slide_annotations_dir` field (alongside the existing `saved_database_dir`, `slide_dx_dir`, `annotated_png_dir`, `test_slide_predictions_dir`), so the annotate_svs wiring is ready whenever `wsi/diagnose.run` is extended to call it.

**Missing HSV helpers in `preprocessing/filter.py`.** `tiles.py:978-1000`
(`rgb_to_hues`, `hsv_saturation_and_value_factor`) calls
`filter.filter_rgb_to_hsv`, `filter_hsv_to_h`, `filter_hsv_to_s`,
`filter_hsv_to_v`. None of the four were defined in either the UNI or
the ResNet project's `filter.py`, so the first real run of
`wsi/diagnose` tripped an `AttributeError` inside `score_tile`. The
authoritative implementations were found in the archived
`/media/.../Cardiac_ACR/Py_Files/filter.py` (original DeepHistoPath
derivation) and ported verbatim — skimage-based (`sk_color.rgb2hsv`),
matching the rest of filter.py's skimage style. Added
`import skimage.color as sk_color` to the header.

Reference: `/media/mglass222/AI Stuff - EVO 860/Cardiac_ACR/Py_Files/`
holds the pre-refactor, pre-package-ification originals for every
module. Useful when a call site references a function that neither
of the merged projects retained.

**Tech debt 3 (move backend paths out of shared config).**

- Dropped `SAVED_DATABASE_DIR`, `SLIDE_DX_DIR`, `ANNOTATED_PNG_DIR`, `TEST_SLIDE_PREDICTIONS_DIR`, `TEST_SLIDE_ANNOTATIONS_DIR` from top-level `cardiac_acr/config.py`.
- Added them explicitly to `backends/uni/config.py` (defaults match the previous shared values — UNI owns the unsuffixed tree). `backends/resnet/config.py` already defined them under `Weighted_Loss/`.
- No consumers still read these from the shared config; the annotate/CSV helpers get them as args via the `BackendClassifier`.

---

## Post-unification state

### What the codebase looks like now

One package, two backends, one CLI:

```
Cardiac-ACR/                        # repo directory
├── pyproject.toml                  # name = "cardiac-acr", version 0.2.0
├── README.md                       # unified docs
├── docs/DEVELOPMENT_LOG.md         # (this file)
└── cardiac_acr/
    ├── __init__.py  __main__.py    # `python -m cardiac_acr <cmd>`
    ├── config.py                   # shared paths + preprocessing constants
    ├── preprocessing/              # slide/tile/patch pipeline (shared)
    ├── wsi/                        # backend-agnostic diagnose, annotate,
    │                               #   count_1r2 (takes saved_database_dir)
    ├── utils/                      # cardiac_utils, util, check_dependencies
    └── backends/
        ├── __init__.py             # BackendClassifier + load_classifier()
        ├── uni/                    # UNI2-h frozen backbone + linear/MLP head
        │   ├── config.py  backbone.py  head.py
        │   ├── features_dataset.py  encode_patches.py
        │   ├── train.py  evaluate.py
        │   └── classifier.py       # → BackendClassifier
        └── resnet/                 # ResNet-50 end-to-end
            ├── config.py  model.py  data_utils.py
            ├── train.py  cross_validation.py
            ├── classifier.py       # → BackendClassifier
            └── stats/              # threshold-sweep CSVs, ROC/AUROC plots
```

Shared input data (`WSI/`, `DeepHistoPath/`, `Patches/`) lives in the
sibling `../Image-Data/` directory, resolved via `SHARED_DATA_DIR`
(env-overridable with `CARDIAC_ACR_SHARED_DATA_DIR`). Each backend's
outputs land under the local `data/` tree — UNI unsuffixed, ResNet
under a `Weighted_Loss/` subfolder — so both backends can run against
the same working copy without colliding.

### Deliberate deferrals

- **`Cardiac-ACR-Resnet/` on disk.** Untouched. Its training code was
  *copied* into `backends/resnet/`, not linked, so it's functionally
  archived. Deletion / relocation is a user call.

### Known tech debt (post-refactor)

1. ~~**`wsi/annotate_png.py` and `wsi/annotate_svs.py`** still read
   `cg.SAVED_DATABASE_DIR` / `cg.ANNOTATED_PNG_DIR` /
   `cg.TEST_SLIDE_ANNOTATIONS_DIR` from the top-level `config`.~~
   **Resolved 2026-04-21.** Both modules now take `saved_database_dir`
   (and `annotated_png_dir` / `test_slide_annotations_dir`) as args.
2. ~~**`utils/cardiac_utils.py` CSV helpers**~~
   **Resolved 2026-04-21.** `model_prediction_dict_to_csv` and
   `slide_dx_to_csv` take path args. Pre-existing
   `cg.<DIR> + filename` concatenation bugs fixed.
3. ~~**Shared config still carries backend-specific defaults.**~~
   **Resolved 2026-04-21.** `SAVED_DATABASE_DIR`, `SLIDE_DX_DIR`,
   `ANNOTATED_PNG_DIR`, `TEST_SLIDE_PREDICTIONS_DIR`,
   `TEST_SLIDE_ANNOTATIONS_DIR` now live only in each backend's own
   `config.py`.
4. **Minor duplication between `preprocessing/preprocess_data_utils.py`
   and `backends/resnet/data_utils.py`.** Both define
   `count_classes` / `count_patches`. The ResNet copy has additional
   DataLoader-shaped helpers (`class_weights`, `datasets_normalization`,
   `dataloaders`, `epoch_steps`) that UNI dropped when it moved to
   feature-cache training. Acceptable for now; a one-line re-export
   from the backend side would clean it up.
5. **No tests.** Verification so far has been import smoke-tests +
   CLI `--help`. Real runs will produce the first end-to-end signal.
   Minimum viable test: run `preprocess` then `train --backend uni`
   on a small slide subset; then the same with `--backend resnet`;
   then `diagnose-wsi --backend {uni,resnet}` on one slide each and
   confirm the per-slide dx pickles exist in the right directory.

### Post-unification follow-ups (replaces the V2 list at the top)

In rough priority order:

1. **End-to-end smoke run** with the new CLI against a small slide
   set, one backend at a time. First use of the unified CLI — will
   shake out any remaining integration bugs (e.g., annotate paths, CSV
   helpers) that static import tests miss.
2. **Head-to-head UNI vs ResNet** on the same validation split. The
   whole point of keeping both backends alive. Produces the concrete
   numbers that justify the UNI rewrite (or don't).
3. **Fix the tech-debt items 1–3 above** — parametrize `annotate_png`
   / `annotate_svs` / `cardiac_utils` CSV helpers, then drop backend-
   specific paths from shared config. Unblocks turning on annotations
   for the ResNet backend.
4. **Per-slide confidence calibration** — carried over from the V1
   list; still relevant.
5. **Stain-jitter or Macenko normalization** / **multi-crop TTA at
   encode time** — carried over; UNI-backend-only tweaks.
6. **Periodic check for pathology-DINOv3 variants** (carried over).

---

## 2026-04-22 — Consolidated `Image-Data/` back into `data/`

Reversed the 2026-04-21 split. `WSI/`, `DeepHistoPath/`, and `Patches/`
moved from the sibling `../Image-Data/` tree back under the project's
local `data/` directory. `Cardiac-ACR-Resnet` is no longer sharing
these inputs, so the cross-project shared-directory mechanism has no
remaining consumer.

`cardiac_acr/config.py` changes:

- Removed `SHARED_DATA_DIR` and the `CARDIAC_ACR_SHARED_DATA_DIR` env
  override.
- `DEEP_HISTO_DIR`, `WSI_DIR`, and `PATCH_DIR` now resolve under
  `DATA_DIR` (`<PROJECT_ROOT>/data`) alongside `BACKEND_DIR` and the
  per-backend output dirs.
- Module docstring simplified — all inputs and outputs live under
  `DATA_DIR`.

No other code referenced `SHARED_DATA_DIR`; the variable was only
read inside `config.py`. Historical log entries above still describe
the shared-dir layout as it existed on 2026-04-21.

---

## 2026-04-22 — DataLoader-fed WSI classification (~4× GPU utilization)

`wsi.diagnose.classify_patches` was a serial loop: open PNG → PIL
decode → transform → `torch.stack` → `classifier.classify`. `nvidia-smi
dmon` during a live run showed **~25% SM util averaged** (bursting to
40% during the forward pass, dropping to 6–16% while the main thread
loaded the next 64 patches). The GPU was idle ~75% of the time.

**Change.** Replaced the manual loop with a `torch.utils.data.DataLoader`
fed by a tiny `_PatchFileDataset` that does PIL decode + transform in
worker processes. Configured `num_workers=8`, `pin_memory=True`,
`shuffle=False`. Added `tqdm` progress bar with batch rate and
accumulated-patch count in the postfix.

**Measured impact.** Post-change `nvidia-smi dmon` on the same
machine (RTX 2070 SUPER, fp16 autocast on ViT-H/14):

```
before:  sm 6–40% (mean ~25%), bursty
after:   sm 97–98% flat, zero idle gaps
```

~4× GPU-feeding efficiency. Per-slide classify time drops by roughly
the same factor. At 98% SM util we're at the Turing fp16 ceiling for
UNI2-h encode (~66 img/s in the training-side benchmark); further
wins require either hardware upgrade (5090 / H-series) or the streaming
inference refactor outlined in `PERFORMANCE_IDEAS.md` §1, which
would only marginally improve on the current state since IO is no
longer the bottleneck.

**Code.** `cardiac_acr/wsi/diagnose.py` — added `_PatchFileDataset`
and `_CLASSIFY_WORKERS = 8`; rewrote `classify_patches` to iterate a
DataLoader with a tqdm wrapper.

---

## 2026-04-22 — Quieted filter-module stdout

The preprocessing filter modules (`preprocessing/filter.py`,
`preprocessing/filter_patches.py`, and `utils/util.np_info`) printed
hundreds of lines per slide during the per-slide loop — every filter
step announced its shape/dtype/timing, and the overmask-retry logic
recursed with a debug `print` per step. During a diagnose run this
drowned out the `tqdm` classification bar and the per-slide progress
markers we actually care about.

**Change.** Added `VERBOSE = False` (plus `log(msg)` helper) to
`cardiac_acr/utils/util.py`. Routed all `print` calls in
`preprocessing/filter.py` and `preprocessing/filter_patches.py` —
np_info shape/time output, slide-processing banners, save-image
timings, task assignments, done markers, and the overmask-retry
prints — through `util.log()`. `np_info` itself early-returns when
not verbose. Flip `util.VERBOSE = True` to restore the original
chatter for debugging.

High-level progress (`\n=== slide N ===`, per-slide timing, classify
`tqdm` bar, thresholded kept-fraction, dx summary) is printed directly
from `wsi/diagnose.py` and is not gated by the flag — so the terminal
stays readable without losing the top-level signal.

---

## 2026-04-22 — Progress-bar polish and deterministic slide order

Follow-up polish on the diagnose pipeline's live UX.

**Single classify bar was doubling and bleeding across lines.** The
initial tqdm integration called `progress.set_postfix(patches=...)`
after every batch, which forces a second refresh per tick — the
terminal showed two bars per batch. Bars also rendered wider than the
actual terminal, so each `\r` landed mid-line and updates concatenated
instead of overwriting. Fix: dropped `set_postfix` (batch count is
already visible), added `dynamic_ncols=True` so tqdm re-queries
`$COLUMNS` on each refresh, and `mininterval=0.5` to cap the redraw
rate.

**Added outer slides bar.** `run()` now wraps the per-slide loop in a
top-level `tqdm(slides_to_process, desc="slides", position=0)`, so
the terminal shows two stacked bars: outer for total progress (with
slides/min ETA for the full run) and inner for the current slide's
classify batches (`position=1, leave=False` so each finished slide's
bar clears and the outer bar stays pinned). The per-slide
"done in Xs" summary uses `tqdm.write()` so it prints cleanly above
the live bars.

**Deterministic slide order.** `utils.get_test_slide_numbers()` used
to return slides in whatever `os.listdir` gave — filesystem
insertion/inode order, not numeric. On this machine the actual run
started `139, 260, 183, 161, …`, which makes "resume from slide N"
semantically awkward. Added `slide_list.sort()` so runs always
process in ascending slide number. Only affects iteration order;
the set of processed slides is unchanged.

**Code.**
- `cardiac_acr/wsi/diagnose.py` — tqdm kwargs updated on the
  classify bar; outer slide tqdm added in `run()`.
- `cardiac_acr/utils/cardiac_utils.py` — `slide_list.sort()` added
  before return in `get_test_slide_numbers`.

## 2026-04-23 — Fused tissue filter into the classify DataLoader

**Motivation.** After 2026-04-22's DataLoader switch, the per-slide
loop still ran three passes over every patch: `tileset_utils` wrote
~25k 224×224 PNGs; `filter_patches_multiprocess` read each one,
computed tissue %, and `os.remove`'d the rejects; then
`classify_patches` opened every survivor again. Non-tissue patches
were pure overhead — written, read, deleted; tissue patches were read
twice. (See `PERFORMANCE_IDEAS.md` item #3.)

**Change.** `_PatchFileDataset.__getitem__` now runs the same tissue
check the old pass ran (`filter_patches.apply_image_filters` +
`tissue_percent`) on the numpy RGB it already decoded, and returns a
`(path, None)` sentinel for patches under 50% tissue. A new
`_drop_empty_collate` is handed to the DataLoader; it filters the
sentinels out of each batch and stacks whatever's left, returning
`([], None)` for fully-empty batches. `classify_patches` skips those.

Because `util.VERBOSE` is already `False` (set in 2026-04-22's
"Quieted filter-module stdout" change), `apply_image_filters` runs
silently in the workers — no per-patch log spam across 8 processes.

**Semantics.**
- Same 50%-tissue cutoff as before; same `apply_image_filters` used as
  the mask source, so rejection decisions are identical.
- Rejected PNGs are no longer deleted from disk during the run. They
  get overwritten the next time `tileset_utils` rebuilds that slide's
  split-tile dir. No other code path depends on their absence.
- `classify_patches`'s end-of-slide log now reads
  `"kept X/Y tissue patches ..."` instead of the raw write count.
- `filter_patches_multiprocess` local helper deleted from
  `wsi/diagnose.py`. `preprocessing/filter_patches.py` itself is
  untouched — `apply_image_filters` and `tissue_percent` are imported
  directly by the dataset; nothing else imports the module now, but
  its public surface is preserved in case a future caller wants the
  standalone tissue-fraction dict.

**Code.**
- `cardiac_acr/wsi/diagnose.py` — dataset does tissue filter + sentinel
  return; `_drop_empty_collate` added; `DataLoader(collate_fn=...)` set;
  classify loop handles empty batches; `filter_patches_multiprocess`
  helper and its call from `run()` removed; module docstring updated.

**Expected impact.** One fewer per-slide pass through the patch dir
(removes the `filter_patches_multiprocess` cost entirely), one fewer
read of every tissue patch, no `os.remove` syscalls mid-run. On SSD
this is modest; on HDD/NFS it is more noticeable. The change also
shrinks the scope of the streaming refactor (`PERFORMANCE_IDEAS.md`
#1) since the tissue check is already colocated with the classify
dataset.

**Measured on slide 139 (2070 SUPER, SSD).** 39,925 patches / 344.1s
classify → 116 patches/s at the dataset level (tissue-rejected patches
never hit the GPU). Of those, 25,540 (64%) passed the 50% tissue
check, giving ~74 tissue-patches/s through UNI — modestly faster than
2026-04-22's ~66 img/s on the same hardware, with the separate filter
pass eliminated. At `PREDICTION_THRESHOLD=0.99`, 9,211 / 25,540 patches
(36%) cleared the top-softmax cutoff. Dx came out `2R` (driven by the
1R2 focus count from the segmentation pipeline).

## 2026-04-24 — Streaming WSI inference behind `--streaming` flag

**Motivation.** Yesterday's run on slide 139 spent 432.9s in preprocessing
— writing ~500 level-0 1120×1120 tile PNGs (~1.8 GB) to `TILE_DIR` and
~39,925 224×224 patch PNGs (~3.7 GB) to `SPLIT_TILE_DIR`, then reading
all of the surviving patches once from the classify DataLoader and
never touching them again. On a 13-minute per-slide total, the pre-loop
was more expensive than classify itself. See `PERFORMANCE_IDEAS.md`
item #1.

**Change.** Added an OpenSlide-backed streaming path to `wsi/diagnose.py`
as an additive, flag-gated alternative to the disk path. Both modes
coexist; the default is still disk-based. `--streaming` on the CLI
picks the new path (or `streaming=True` on `run()`).

New `_StreamingPatchDataset` alongside the existing `_PatchFileDataset`:

- Constructor takes `(slide_number, top_tiles, transform)`. Stores the
  SVS path only — no open handle at construction time, since OpenSlide
  handles don't survive DataLoader `fork()`.
- Eagerly walks each `TileSummary.top_tiles()` entry in 224-pixel
  strides starting at its level-0 bbox `(o_c_s, o_r_s)`, producing a
  flat list of `(tile_r, tile_c, patch_x, patch_y)` level-0 coords.
  Keeps `tile_r`/`tile_c` for human-readable synthetic filenames.
- `__getitem__` lazily opens the slide once per worker
  (`if self._slide is None: self._slide = slide.open_slide(path)`).
  Works with `num_workers=0` identically — no `worker_init_fn` needed.
- Reads always request a full 224×224 window at level 0. OpenSlide
  zero-pads past the slide edge (transparent → black on RGB convert),
  matching the legacy `tileset_utils.tiles_to_patches` paste-on-black
  behaviour for partial boundary patches.
- Reuses the exact same `apply_image_filters` + `tissue_percent` + 50%
  cutoff + `(name, None)` sentinel + `_drop_empty_collate` pipeline
  from 2026-04-23's `_PatchFileDataset`. The only difference is where
  the RGB bytes come from.
- Synthetic filename keys: `f"{slide:03d}-tile-r{r}-c{c}-x{x}-y{y}.png"`.
  Matches the `.*-x(\d+)-y(\d+).*\..*` regex in
  `utils.get_coords_from_name`, so `count_1r2`, `annotate_png`, and
  `annotate_svs` all work unchanged on the new pickles.

`run()` now takes `streaming=False` and wraps the
`tiles.multiprocess_filtered_images_to_tiles` and
`tileset_utils.process_tilesets_multiprocess` preprocessing calls in
`if not streaming:`. `slide.multiprocess_training_slides_to_images`
and `wsi_filter.multiprocess_apply_filters_to_images` stay
unconditional because `count_1r2` reads `PNG_SLIDE_DIR` +
`FILTERED_IMAGE_DIR`, and streaming's `score_tiles` call consumes the
filtered PNG. Those two produce just one PNG each (tens of MB total),
not the multi-GB chunk.

`classify_patches(streaming=False)` picks the dataset class; the
DataLoader, classify loop, predictions-dict serialization, and
end-of-slide log line are shared across both modes.

**Measured on slide 139 (A/B against yesterday's disk-mode baseline,
same hardware).**

| Metric | Disk (yesterday) | Streaming (today) |
|---|---|---|
| Preprocessing time | 432.9s | **6.8s** (-98%) |
| Classify time | 344.1s | 344.2s (identical) |
| Total per-slide time | 777s | **351s** (-55%) |
| Tissue patches kept | 25,540 / 39,925 | 25,540 / 39,925 (identical) |
| Threshold-pass @ 0.99 | 9,211 / 25,540 | 9,211 / 25,540 (identical) |
| Class counts (filtered) | `{1R1A: 199, 1R2: 14, Healing: 2949, Normal: 6025, Quilty: 24}` | identical |
| Slide dx | `2R` | `2R` |
| New bytes under `TILE_DIR`/`SPLIT_TILE_DIR` | ~5.5 GB | **0** |

**Patch-level agreement.** 25,540 / 25,540 coordinates match between
the two runs. 100% argmax-class agreement. Zero drift at tile
boundaries — the predicted <1% edge drift from OpenSlide reading real
neighbouring pixels instead of black padding did not show up in
practice, because the boundary patches either still pass/fail the
tissue check the same way, or they fall inside the filtered-tissue
region where the extra real pixels don't change argmax.

**Multi-slide reproducibility check (4 slides total).** Re-ran the
same A/B on three additional slides (111, 119, 135) against their
disk-mode baselines from 2026-04-22. Summary:

| Slide | Patches | Coord overlap | Argmax agreement | Max prob drift | Filtered counts | Dx |
|---|---|---|---|---|---|---|
| 111 | 27,344 | 100% | 100% | 0.0013 | identical | 1R2 |
| 119 | 28,118 | 100% | 100% | 0.0012 | identical | 1R2 |
| 135 | 28,883 | 100% | 100% | 0.0018 | identical | 2R |
| 139 | 25,540 | 100% | 100% | 0.0015 | identical | 2R |

109,885 patch pairs total across all four slides, 100% argmax agreement.
Max per-patch probability drift is ~0.002 — well below any decision
boundary at the 0.99 threshold. The drift (not exactly zero, unlike
139 alone where it was bit-identical) is consistent with a 1-LSB
difference in the decoded RGB: legacy reads a PIL-re-saved PNG,
streaming decodes the SVS's JPEG tile natively, and those two paths
round slightly differently in a handful of pixels. No functional
consequence — same patches kept, same class per patch, same filtered
pickle, same slide dx.

Confidence: the streaming path is safe to promote to default once the
flag-default flip is convenient.

**Code.**
- `cardiac_acr/wsi/diagnose.py` — `_StreamingPatchDataset` added
  alongside `_PatchFileDataset`; `classify_patches` takes
  `streaming=False`; `run()` gates the tile-save and tileset-split
  preprocessing calls; `main()` adds `--streaming` argparse flag;
  module docstring describes both modes.
- `README.md` — "Running the pipeline" section documents both CLI
  invocations.
- `preprocessing/tiles.py` — unchanged. The `save_top_tiles = True`
  hardcode at line 534 is irrelevant because disk-mode callers want
  saves anyway and streaming mode skips the call via the
  `if not streaming:` gate.

**Git workflow.** This change landed on `Cardiac-ACR-2026` master as
part of Phase 0's git setup (Phase 0 = pushing the post-#3 state onto
the existing GitHub repo as split commits). Streaming is additive;
disk mode is untouched. Before paper submission, a follow-up branch
will strip the disk path entirely (remove `_PatchFileDataset`, the
`--streaming` flag, the gated preprocessing calls, and
`preprocessing/tileset_utils.py`).

**Next candidates (from `PERFORMANCE_IDEAS.md`).** With #1 shipped, the
remaining sub-second-to-tens-of-seconds-per-slide wins are #2 (cache
UNI features per test slide so threshold/head sweeps don't re-encode)
and #6 (structured experiment logging). #4 (multi-slide pipelining)
and #7 (profile pre-loop) are lower-priority given the pre-loop is
now 6.8s.

## 2026-04-24 — Streaming becomes the default

Follow-up to the same-day streaming shipit. With the four-slide
reproducibility check clean (109,885 patch pairs, 100% argmax
agreement), flipped the default:

- `run(streaming=False)` → `run(streaming=True)` in
  `wsi/diagnose.py`; same for `classify_patches`.
- CLI changed from `--streaming` (store_true, default off) to
  `--streaming` / `--no-streaming` via
  `argparse.BooleanOptionalAction` with `default=True`. `python -m
  cardiac_acr.wsi.diagnose --backend uni` now runs streaming.
  `--no-streaming` reinstates the legacy disk-based pipeline for
  anyone who wants the 5 GB of intermediate PNGs (debugging, visual
  inspection of rejected patches, etc.).
- Module docstring + README updated; "Running the pipeline" now
  documents the legacy path as an explicit opt-in under
  `--no-streaming` instead of implying it's the default.

The disk path code stays intact. Eventual paper-prep branch removes
it along with `_PatchFileDataset`, the CLI flag, the conditional in
`run()`, and `preprocessing/tileset_utils.py`.

## 2026-04-24 — Disk path removed from master; preserved on `disk-mode` branch

Retrospective on the same-day flag-based design: with the four-slide
reproducibility check showing zero functional drift (100% argmax
agreement across 109,885 patch pairs; max prob drift 0.002), the
disk path is purely ballast on master — two parallel code paths to
read, one of them never exercised going forward, slated for bit-rot
the next time someone refactors `apply_image_filters` or the
transform.

User still has a legitimate use case for disk-materialized patches:
filter tuning benefits from being able to sort on-disk 224×224 PNGs
by tissue% and eyeball the marginal cases. Resolution: keep the
disk path fully alive on a dedicated `disk-mode` branch (pushed to
`origin/disk-mode` at `b609d3e`, the pre-cleanup state), and strip
it cleanly from master.

**Removed from master.**
- `_PatchFileDataset` class (disk-mode patch loader).
- `streaming` kwarg from `classify_patches` and `run`; both now
  single-path streaming.
- `if not streaming:` block in `run()` that called
  `tiles.multiprocess_filtered_images_to_tiles` and looped
  `tileset_utils.process_tilesets_multiprocess`.
- `--streaming/--no-streaming` CLI flag from `main()`.
- Imports: `from os import listdir`, `from cardiac_acr.preprocessing
  import tiles` (replaced by the already-present
  `from cardiac_acr.preprocessing.tiles import score_tiles`),
  `from cardiac_acr.preprocessing import tileset_utils`.
- `mode: streaming|disk-based` print line from `run()`.
- Whole file `cardiac_acr/preprocessing/tileset_utils.py` (no
  remaining callers anywhere in the package).

**Kept on master.** `preprocessing/tiles.py` is intact —
`score_tiles`, `TileSummary.top_tiles`, and `tile_to_pil_tile` are
still called. Some of its module-level functions are now unreachable
on master (`multiprocess_filtered_images_to_tiles`, `image_list_to_tiles`,
`summary_and_tiles`); pruning them is module-internal noise and can
wait.

**Recovery.** `git checkout disk-mode` restores the full two-path
state. To pull just one file back without switching branches,
`git show disk-mode:cardiac_acr/preprocessing/tileset_utils.py > ...`.

**Long-term replacement for on-disk debugging.** A `--dump-patches`
flag on the streaming dataset would let filter tuning happen from
master: run diagnose with the flag, the DataLoader worker writes each
224×224 patch (plus tissue%) to a debug dir as a side effect, and
you get the same visual affordance without keeping a parallel
codepath. Not implemented today; implement when filter tuning
actually needs it.

**Code.**
- `cardiac_acr/wsi/diagnose.py` — edits above; single-path streaming
  is now the entire public surface.
- `cardiac_acr/preprocessing/tileset_utils.py` — deleted.
- `README.md` — "Running the pipeline" is one command; notes
  `disk-mode` branch as the recovery path for intermediate-PNG
  debugging; `tileset_utils.py` line removed from the project-layout
  tree.

## 2026-04-24 — Streaming vs disk: consolidated performance comparison

All numbers below: 2070 SUPER (8 GB VRAM, Turing, fp16 autocast),
local SSD, UNI2-h + linear head, `PREDICTION_THRESHOLD=0.99`,
`_CLASSIFY_BATCH=64`, `_CLASSIFY_WORKERS=8`. Disk-mode numbers come
from the 2026-04-22 and 2026-04-23 runs; streaming numbers from the
four verification runs on 2026-04-24. Classify time is the dominant
term in both modes because the GPU pass is identical; the delta is
almost entirely in preprocessing.

### Per-slide runtime (slide 139, 39,925 candidate patches)

| Phase | Disk mode | Streaming | Δ |
|---|---|---|---|
| SVS → scaled PNG | kept in both; ~1–2s per slide | kept in both; ~1–2s per slide | 0 |
| Tissue-filter PNG | kept in both; ~2–3s per slide | kept in both; ~2–3s per slide | 0 |
| Tile scoring | written to disk as 1120×1120 PNGs (~500 × ~3.7 MB = ~1.8 GB) | in-memory, no writes | disk writes eliminated |
| Patch extraction | split into 224×224 PNGs (~39,925 × ~150 KB = ~6 GB before tissue filter) | none — patches streamed on demand | disk writes eliminated |
| **Total preprocessing** | **~432.9 s** | **~6.8 s** | **−98.4 %** |
| Classify (DataLoader → UNI → head) | 344.1 s | 343.8–344.2 s (3 runs) | ~0 |
| Threshold + diagnose | <1 s | <1 s | 0 |
| **Total per slide** | **~777 s (~13 min)** | **~351 s (~6 min)** | **−55 %** |

The preprocessing saving (−426 s per slide) is the entire wall-clock
win. Classify is wall-clock-identical because both paths feed the
same 25,540 tensors through the same model.

### Multi-slide runtime (slides 111, 119, 135 batched in one run)

Streaming preprocesses all three slides in parallel (SVS→PNG + filter
PNG are already multiprocess-per-slide calls) for **7.7 s total**,
then classify runs serially per slide:

| Slide | Candidate patches | Tissue patches kept | Classify time | Total |
|---|---|---|---|---|
| 111 | 43,625 | 27,344 (62.7 %) | 366.6 s | 367.6 s |
| 119 | 44,575 | 28,118 (63.1 %) | 358.2 s | 359.2 s |
| 135 | 42,150 | 28,883 (68.5 %) | 370.9 s | 371.8 s |
| **3-slide batch** | 130,350 | 84,345 (64.7 %) | 1,095.7 s | **~18.3 min** |

The equivalent disk-mode run (estimated from 2026-04-22 numbers) would
have spent an extra ~430 s per slide in preprocessing × 3 = **~21.5
min** just on intermediate-PNG materialization, for a total of ~40
min — more than 2× the streaming wall-clock.

### Storage impact (per slide)

| Artifact | Disk mode | Streaming |
|---|---|---|
| Scaled PNG (1/40× of SVS) | ~30–50 MB | ~30–50 MB |
| Filter PNG | ~20–40 MB | ~20–40 MB |
| Level-0 tile PNGs (`TILE_DIR/<slide>/`) | **~1.8 GB** (~500 × 1120×1120) | 0 |
| 224×224 patch PNGs (`SPLIT_TILE_DIR/<slide>/`) | **~3.5–4 GB** (~25–40 k × 150 KB) | 0 |
| Predictions pickle | ~5 MB | ~5 MB |
| **Intermediate writes** | **~5.5 GB / slide** | **~0 bytes** |

On a 200-slide test set, that's ~1.1 TB of intermediate PNGs the disk
path would have written (and read once, and then kept on disk until
manually cleaned). Streaming is zero.

### Classification correctness (streaming vs disk baseline)

Four-slide A/B, 109,885 total patch pairs. Disk baselines come from
the 2026-04-22 run (pre-#3 tissue fuse) and the 2026-04-23 slide 139
run (post-#3); both paths ran against the same UNI head checkpoint
and the same `apply_image_filters` + `tissue_percent` tissue check,
so candidate-patch selection is algorithmically identical.

| Slide | Patches | Coord overlap | Argmax agreement | Max per-patch prob drift | Dx disk | Dx stream |
|---|---|---|---|---|---|---|
| 111 | 27,344 | 100 % | 100 % | 0.0013 | 1R2 | 1R2 |
| 119 | 28,118 | 100 % | 100 % | 0.0012 | 1R2 | 1R2 |
| 135 | 28,883 | 100 % | 100 % | 0.0018 | 2R | 2R |
| 139 | 25,540 | 100 % | 100 % | 0.0015 | 2R | 2R |
| **total** | **109,885** | **100 %** | **100 %** | **≤ 0.002** | — | **all match** |

Filtered pickles are bit-identical at the class-count level for every
slide. The per-patch prob drift (~1e-3) is 1-LSB decode noise — the
disk path reads a PIL-re-saved PNG, streaming decodes the SVS's JPEG
tile natively, and those two round slightly differently in a handful
of pixels. The drift is three orders of magnitude below the 0.99
threshold, so no classification decision flips.

### Throughput (unchanged by mode choice)

Classify-only throughput is the same across both paths because the
GPU pass is the bottleneck and it sees the same 25,540 tensors either
way:

- Dataset-level throughput: **~116 patches/s** (includes tissue-rejected
  patches that short-circuit inside the DataLoader worker and never
  hit the GPU)
- Tissue-patches-through-UNI: **~74 patches/s** (steady-state after
  2026-04-23's tissue-filter fuse + 2026-04-22's DataLoader rewrite;
  hardware ceiling is UNI2-h ViT-H fp16 inference on Turing, ~97-98 %
  SM utilization)

Streaming does not move either number because the `read_region` +
tissue-filter + transform cost per patch is in the same ballpark as
the old PIL-open-PNG path, and both are hidden behind `num_workers=8`.

### Summary

Streaming wins on wall-clock by **~55 % per slide** (−7 min on a 13
min run), eliminates **~5.5 GB of intermediate disk writes per slide**,
produces **100 % identical classification decisions**, and leaves
classify throughput unchanged. The only cost is the loss of on-disk
intermediate PNGs as a debug affordance for filter tuning — mitigated
by keeping the full disk pipeline alive on the `disk-mode` branch
(pushed at `b609d3e`) and, long-term, by a `--dump-patches`
side-effect flag on streaming (not yet implemented).

---

## 2026-04-24 — D4 multi-view encoding for the UNI head

### Motivation

The UNI head plateaus at ~94 % validation accuracy; the prior end-to-end
ResNet-50 reached ~96 %. Leading suspect: ResNet trained under
`ColorJitter` + `RandomRotation(180)`, while the UNI pipeline trains the
head on **pre-encoded features with zero augmentation** — each patch is
seen exactly once in its canonical orientation. H&E pathology is
rotation- and reflection-invariant, so the dihedral group D4 (4
rotations × 2 flips = 8 views) is a "free" augmentation. Plan in
`docs/UNI_MULTIVIEW_PLAN.md`.

### Implementation

`backends/uni/encode_patches.py` — added `_D4_VIEWS` (the 8 D4 group
elements), `_apply_d4(img, k_rot, do_flip)` using
`torchvision.transforms.functional.rotate` + `hflip`, and
`_build_view_transforms(num_views)` returning a list of N deterministic
Compose pipelines. `_encode_split` now loops the encoder once per view,
appending features and labels into a single cache. `main()` passes
`uni_cfg.NUM_TRAIN_VIEWS` for Training and hardcoded `1` for Validation.
`backends/uni/config.py` — added `NUM_TRAIN_VIEWS = 8` and routed
`TRAINING_FEATURES_PATH` to `training_views8.pt` (suffixed for trivial
A/B against the legacy `training.pt`). `backends/uni/train.py` — one
extra print line surfacing the encoding count and view multiplier.

### Verification

Encode pass on slide-set training split (11,710 patches × 8 views):
- 1207.5 s wall-clock at 77.6 img/s steady state on a 2070 SUPER.
- Output: `training_views8.pt`, shape `(93680, 1536)`, 576 MB on disk.
- Validation: 35.1 s for 2,743 patches at 78.2 img/s, single canonical
  view. Schema unchanged.
- Per-class counts in the new cache are exactly 8× the prior numbers
  (`1R1A 2232 → 17856`, `Hemorrhage 147 → 1176`, etc.) — confirms the
  loop didn't drop or duplicate.

### A/B comparison (same `validation.pt`, 50 epochs MLP head)

| Metric | Baseline (1 view) | D4 (8 views) | Δ |
|---|---|---|---|
| Best val acc | 0.9344 | **0.9402** | +0.58 pp |
| Final eval accuracy | 0.9355 | 0.9391 | +0.36 pp |
| Macro AUROC | 0.9947 | 0.9952 | +0.0005 |
| 1R1A F1 | 0.9155 | 0.9173 | +0.0018 |
| 1R2 F1 | 0.8771 | 0.8828 | +0.0057 |
| Healing F1 | 0.9349 | 0.9411 | +0.0062 |
| Normal F1 | 0.9531 | 0.9556 | +0.0025 |
| Quilty F1 | 1.0000 | 1.0000 | 0 |
| Train loss @ epoch 50 | 0.018 | 0.0003 | — |
| Val loss @ epoch 50 | 0.205 | 0.367 | — |

Both runs trained on the freshly-encoded `validation.pt`, so the comparison
isolates the training cache change. Confusion-matrix shifts are small and
local: `1R2 → 1R1A` errors drop from 28 to 19, `Healing → Normal` from 55
to 50, while `1R1A → 1R2` rises 29 → 40. The hard pairs (1R1A↔1R2 and
Healing↔Normal) remain the bulk of errors in both runs — these are
clinically subtle distinctions that augmentation alone cannot resolve.

### Interpretation

D4 augmentation **helps marginally but does not close the gap to ResNet's
~96 %**. The +0.58 pp lift is real and uniform across non-trivial classes,
but the curve also tells a clear secondary story: the D4 model overfits
*harder*, not less. Train loss collapses to 0.0003 (effectively
memorizing); val loss climbs from 0.16 → 0.37 across epochs while val
accuracy oscillates around 0.93–0.94. The frozen UNI2-h backbone is
highly D4-equivariant by design, so encoding 8 symmetries produces
*nearly-redundant* feature vectors — the head gets 8× more samples to
memorize without 8× more independent information.

The implication: classic augmentation is not the binding constraint on
this pipeline. The remaining ~2 pp gap to ResNet is more likely about (a)
the frozen-features ceiling — UNI's representation, however good in
general, can't be tailored to the specific decision boundaries this
dataset needs without backbone-side gradient flow; (b) hard val examples
in the 1R1A↔1R2 and Healing↔Normal pairs that no amount of training-data
multiplication will help; (c) the Hemorrhage class (0 in val, weighted
13.3× in CE) introducing high-variance noise into the loss.

### What ships

D4 encoding is left **on as the default** (`NUM_TRAIN_VIEWS = 8`). The
+0.58 pp is small but free — encode is one-time, training takes seconds,
and the saved checkpoint is strictly better on every per-class F1. The
legacy `training.pt` stays on disk for fallback; can be deleted once
confidence is established. To reproduce the baseline:
`uni_cfg.NUM_TRAIN_VIEWS = 1; uni_cfg.TRAINING_FEATURES_PATH = ".../training.pt"`.

### Next levers (not yet attempted)

If closing the remaining gap matters, the productive directions are:

- **Test-time D4 averaging at WSI inference** — run the head on all 8
  views and average logits before threshold. Cheap (8× classify pass) and
  the literature suggests it adds another ~0.5–1 pp on top of train-time
  D4.
- **Unfreeze the last UNI block and fine-tune end-to-end** with per-batch
  augmentation. Highest ceiling, most expensive — reverts to the ResNet
  pipeline's compute profile.
- **Per-batch feature-space augmentation in `train.py`** (Gaussian noise,
  feature dropout) — adds *real* variance the frozen-features pipeline
  currently lacks, without touching the encoder.
- **Stochastic ColorJitter at encode time** as enumerable views. A
  degraded form of true train-time stochastic jitter, but adds genuine
  appearance variance the D4 group does not.

These are deferred until and unless the val-acc gap matters for a
specific downstream goal (paper threshold, reviewer comment).
