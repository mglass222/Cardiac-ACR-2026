# Streaming WSI Inference Pipeline

How `cardiac_acr.wsi.diagnose` extracts and classifies patches from a
slide on master (post-2026-04-24, streaming-only). The legacy disk
pipeline that materialized intermediate tile + patch PNGs is preserved
on the `disk-mode` branch.

## TL;DR

The 1120×1120 tile grid is conceptual/logical only — we never
materialize a tile image. The 224×224 patches are read **directly**
from the SVS via OpenSlide `read_region` at level 0. The tile grid
exists for *scoring* (which regions are worth classifying), not for
*extraction*.

## Per-run setup (once, before the slide loop)

1. **`slide.multiprocess_training_slides_to_images`** — SVS → scaled
   PNG, downsampled by `SCALE_FACTOR=40`. One PNG per slide. This is
   the only "image extract" step.
2. **`wsi_filter.multiprocess_apply_filters_to_images`** — runs the
   tissue-vs-background filter chain on the scaled PNG, writes the
   filtered PNG. Used by tile scoring and `count_1r2`.

Everything else is per-slide and in-memory.

## Per-slide flow

### 1. Tile scoring — `score_tiles(slide_number)`

Defined in `preprocessing/tiles.py`. Overlays a 1120×1120 grid
(`ROW_TILE_SIZE = COL_TILE_SIZE = 1120`) on the filtered PNG and scores
each tile by HSV color/saturation/value + tissue%. Returns a
`TileSummary` whose `top_tiles()` yields tiles with score > 0.05, each
carrying its **level-0** bounding box (`o_r_s`, `o_r_e`, `o_c_s`,
`o_c_e`).

No image data is materialized — just coordinates and scores.

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

The 1120×1120 grid exists for **scoring**, not extraction. Scoring
1120-pixel regions on the downsampled filtered PNG is fast and lets us
skip ~70% of a slide that's pure background or pen marks. Without it,
the 25k–40k candidate patches per slide would balloon to hundreds of
thousands.

The patch read itself bypasses the tile entirely —
`read_region((x, y), 0, (224, 224))` is a direct seek into the SVS's
JPEG-compressed pyramid at full resolution.

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

## Performance reference (slide 139, 2070 SUPER)

| Stage | Time | Disk |
|---|---|---|
| Preprocessing (SVS → PNG → filtered PNG) | ~7 s | 2 PNGs/slide |
| Classify (25,540 patches, UNI2-h fp16, batch 64) | ~344 s | 1 pickle |
| Total per slide | ~351 s | — |

Compare to the disk path (`disk-mode` branch): ~777 s/slide and
~5.5 GB of intermediate PNGs per slide. Full per-slide tables in
`DEVELOPMENT_LOG.md` under the 2026-04-24 "Streaming vs disk:
consolidated performance comparison" entry.
