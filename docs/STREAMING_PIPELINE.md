# Streaming WSI Inference Pipeline

How `cardiac_acr.wsi.diagnose` extracts and classifies patches from a
slide on master (post-2026-04-24, streaming-only). The legacy disk
pipeline that materialized intermediate tile + patch PNGs is preserved
on the `disk-mode` branch.

## TL;DR

Two cheap ~MB PNGs per slide get written before the slide loop (the
scaled and filtered images, both at 1/40 resolution). The 1120×1120
tile grid is conceptual — we never materialize a tile image at level 0.
The 224×224 patches are read **directly** from the SVS via OpenSlide
`read_region` at level 0 inside DataLoader workers.

The general design has always been: **score on the cheap (1/40) image,
extract on the expensive (level-0) image**. Streaming deleted the
on-disk middle layer between "we know the level-0 patch coords" and
"the tensor is on the GPU."

## Per-run setup (once, before the slide loop)

These two steps run on every slide before the classify loop. They are
unchanged from the disk-mode pipeline.

1. **`slide.multiprocess_training_slides_to_images`** — SVS → scaled
   PNG, downsampled by `SCALE_FACTOR=40`. A ~50,000×40,000 SVS becomes
   a ~1,250×1,000 PNG (a few MB). Written to
   `data/DeepHistoPath/training_png/`.
2. **`wsi_filter.multiprocess_apply_filters_to_images`** — runs the
   tissue/pen-mark filter chain on the scaled PNG, writes the result
   to `data/DeepHistoPath/filter_png/`. Same 1/40 resolution.

Both are needed: `score_tiles` reads the filtered PNG, `count_1r2`
reads the scaled PNG. Combined cost is ~7s/slide.

Everything else is per-slide and in-memory.

## Per-slide flow

### 1. Tile scoring — `score_tiles(slide_number)`

Defined in `preprocessing/tiles.py`. Reads the **filtered PNG** from
disk (the 1/40-downsampled image written in pre-loop step 2), then
lays a grid on it sized `ROW_TILE_SIZE / SCALE_FACTOR` = 1120/40 = 28
pixels per tile. Scoring runs entirely on the small image — each
"tile" being scored is a 28×28 region of the ~1,250×1,000 PNG. Score
is HSV color factor (purple/pink hue) × saturation/value × tissue%.

Each `Tile` records both coordinate systems:
- `r, c` — small-image grid index
- `o_r_s, o_r_e, o_c_s, o_c_e` — **level-0** bounding box, multiplied
  back up by `SCALE_FACTOR=40`

`top_tiles()` returns tiles with score > 0.05. The level-0 coords are
what the next step uses to read pixels from the SVS.

No image data is materialized at level 0 — just coordinates and
scores. Scoring 28×28 tiles on a tiny PNG is ~1s/slide.

Historical note: this function is bit-identical between `master` and
`disk-mode`. Tile scoring has always run on the downsampled filtered
PNG; the streaming change is downstream of scoring.

### 2. Patch coordinate enumeration — `_StreamingPatchDataset.__init__`

Defined in `wsi/diagnose.py:94`. For each top tile, enumerates a 5×5
grid of 224×224 patch origins (1120 / 224 = 5 exactly). Result: a flat
list of `(tile_r, tile_c, x, y)` level-0 coordinates. Still no pixels.

### 3. Patch read + filter — `_StreamingPatchDataset.__getitem__`

Runs inside DataLoader workers (`num_workers=8`). Per call:

- Opens the SVS lazily — OpenSlide handles don't survive `fork()`, so
  each worker gets its own handle on first read.
- `self._slide.read_region((x, y), 0, (224, 224))` — reads the 224×224
  patch **directly** from the SVS at level 0. The 1120×1120 tile is
  never assembled in memory.
- Applies the in-memory tissue filter; <50% tissue returns a sentinel
  `(name, None)`.
- Otherwise transforms and returns a tensor.

### 4. Sentinel drop — `_drop_empty_collate`

Custom DataLoader collate function drops any `(name, None)` sentinels
before the batch reaches the GPU. Net effect: every patch is touched
exactly once across the worker pool, and rejected patches never burn
GPU time.

### 5. Classify

`classifier.classify(batch)` runs UNI2-h backbone + linear/MLP head,
softmax over 6 classes, saves a `{patch_name → 6-vector}` dict to
pickle.

### 6. Threshold + diagnose

- Threshold at `PREDICTION_THRESHOLD=0.99` (drop low-confidence
  patches).
- 1R2 focus count comes from the dedicated segmentation pipeline
  (`count_1r2`), not the patch classifier.
- Aggregate into a slide-level rejection grade: `0R`, `1R1A`, `1R2`,
  or `2R`.

## Why a tile grid at all?

The grid exists for **scoring**, not extraction. Scoring 28×28 cells
on the 1/40 filtered PNG (each cell = a 1120×1120 level-0 region) is
fast, and it lets us skip ~70% of a slide that's pure background or
pen marks. Without it, the 25k–40k candidate patches per slide would
balloon to hundreds of thousands.

The patch read itself bypasses the tile entirely —
`read_region((x, y), 0, (224, 224))` is a direct seek into the SVS's
JPEG-compressed pyramid at full resolution.

## What the disk path used to do after scoring

For reference, the legacy disk pipeline (still on `disk-mode`) did the
same scoring step, then added two more *extract* phases:

1. `multiprocess_filtered_images_to_tiles` → `tile_to_pil_tile` called
   `slide.read_region((o_c_s, o_r_s), 0, (1120, 1120))` for each top
   tile and wrote the result as a 1120×1120 PNG to `TILE_DIR`.
2. `tileset_utils.process_tilesets_multiprocess` cropped each
   1120×1120 PNG into 25 × 224×224 patch PNGs in `SPLIT_TILE_DIR`.

Both happened *after* scoring; neither fed back into it. Streaming
deletes both: `_StreamingPatchDataset` enumerates the same level-0
patch coordinates and reads them on demand inside DataLoader workers.

## Why "5×5 grid"

`PATCH_SIZE=224` × 5 = 1120 = `ROW_TILE_SIZE`. Chosen so a tile
partitions cleanly into patches with no overlap or remainder. If you
ever change one, change both — otherwise the patch enumeration in
`_StreamingPatchDataset.__init__` will produce ragged grids with
edge-padded reads.

## File / function pointers

| Component | File | Symbol |
|---|---|---|
| Pipeline entry | `cardiac_acr/wsi/diagnose.py` | `run()` |
| Per-slide classify | `cardiac_acr/wsi/diagnose.py` | `classify_patches()` |
| Streaming dataset | `cardiac_acr/wsi/diagnose.py:79` | `_StreamingPatchDataset` |
| Sentinel collate | `cardiac_acr/wsi/diagnose.py:141` | `_drop_empty_collate` |
| Tile scoring | `cardiac_acr/preprocessing/tiles.py` | `score_tiles`, `TileSummary.top_tiles` |
| Patch tissue filter | `cardiac_acr/preprocessing/filter_patches.py` | `apply_image_filters`, `tissue_percent` |
| 1R2 focus count | `cardiac_acr/wsi/count_1r2.py` | `main` |
| Patch / tile size | `cardiac_acr/config.py:47` | `PATCH_SIZE = 224` |
| Tile size | `cardiac_acr/preprocessing/tiles.py:37` | `ROW_TILE_SIZE = COL_TILE_SIZE = 1120` |

## Resolutions at each stage

| Stage | Reads from | Resolution | Output |
|---|---|---|---|
| Scaled PNG extract | SVS | 1/40 | one ~MB PNG/slide |
| Tissue filter | scaled PNG | 1/40 | one ~MB PNG/slide |
| `score_tiles` | filtered PNG | 1/40 (28×28 cells) | in-memory `TileSummary` |
| Patch read (`__getitem__`) | **SVS directly** | level 0 (224×224) | tensor → GPU |

## Performance reference (slide 139, 2070 SUPER)

| Stage | Time | Disk |
|---|---|---|
| Preprocessing (SVS → scaled PNG → filtered PNG) | ~7 s | 2 PNGs/slide |
| Classify (25,540 patches, UNI2-h fp16, batch 64) | ~344 s | 1 pickle |
| Total per slide | ~351 s | — |

Compare to the disk path (`disk-mode` branch): ~777 s/slide and
~5.5 GB of intermediate PNGs per slide. Full per-slide tables in
`DEVELOPMENT_LOG.md` under the 2026-04-24 "Streaming vs disk:
consolidated performance comparison" entry.
