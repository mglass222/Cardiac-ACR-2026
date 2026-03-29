# Plan: Remove Unused Components from Cardiac-ACR-Opus46

## Context

The Cardiac-ACR codebase has accumulated dead code over time — unused files, commented-out code paths, orphaned functions, and unnecessary imports. A full code trace (documented in `CODE_ANALYSIS.md`) identified ~69 unused functions across active files, 7 entirely removable files, and numerous unused imports. This cleanup removes dead code to improve maintainability and clarity without affecting any active functionality.

## Reference

- Full trace analysis: `/Users/matthew/Documents/Code/Cardiac-ACR-Opus46/CODE_ANALYSIS.md`

---

## Step 1: Delete unused files (7 files)

Delete these files entirely — none are imported or referenced by active code:

| File | Reason |
|---|---|
| `cardiac_dirs.py` | Not imported by anything |
| `slide_backup.py` | Old copy of `slide.py`, not imported |
| `count_1r2_new.py` | Alternate version of `count_1r2.py`, not imported |
| `count_1r2_testing.py` | Test version of `count_1r2.py`, not imported |
| `segmentation_tools.py` | Not imported by anything |
| `filter_tiles.py` | Imported but no function ever calls into it |
| `Cardiac_ACR_Backend_V13_FINAL.ipynb` | Jupyter notebook duplicate of the `.py` entry point |

---

## Step 2: Clean up `Cardiac_ACR_Backend_V13_FINAL.py`

### 2a: Fix the commented-out line
- Uncomment `slides_to_process = utils.get_test_slide_numbers()` (line 385)
- Remove the hardcoded `slides_to_process = [111]` (line 387)

### 2b: Remove unused imports
- `import filter_tiles` (line 20)
- `import xml.etree.ElementTree as ET` (line 33)
- `import shutil` (line 35)
- `import multiprocessing` (line 36)
- `from datetime import datetime` (line 47)
- `import re` (line 48)
- `import cv2` (line 50)
- `import matplotlib.pyplot as plt` (line 51) and `plt.rcParams` line (line 81)
- `ImageOps` from the PIL import (line 44) — keep `Image`
- `isdir` and `join` from `os.path` import (line 32) — keep `isfile`

### 2c: Remove unused module-level variables
- `TISSUE_PERCENT_DIR = cg.TISSUE_PERCENT_DIR` (line 76)
- `FONT_PATH = cg.FONT_PATH` (line 79)

### 2d: Remove Jupyter notebook cell markers
- Strip all `# In[X]:` comments throughout the file

---

## Step 3: Remove 12 unused functions from `cardiac_utils.py`

Remove these functions (none are called by active code):

- `get_filtered_image_path()`
- `get_files()` (also has bug: calls undefined `GetFiles()`)
- `get_training_slide_numbers()`
- `get_slide_info()` (also has bug: calls undefined `GetFiles()`)
- `get_patches_dir_from_slide_number()`
- `small_to_large_coords()`
- `clean_csv_files()`
- `make_top_slides_csv_file()`
- `filter_tiles_multiprocess()`
- `display_tissue_percent_patches()`
- `display_tissue_percent_tiles()`

Also remove any imports that become unused after these deletions (e.g., `import filter_tiles` if present).

---

## Step 4: Remove 6 unused functions from `slide.py`

- `get_tile_image_path_by_slide_row_col()`
- `show_slide()`
- `training_slide_range_to_images()`
- `singleprocess_training_slides_to_images()`
- `slide_stats()`
- `slide_info()`

---

## Step 5: Remove 30 unused functions from `filter.py`

These filter functions are never called by `apply_image_filters()` or anything else:

- `filter_rgb_to_grayscale()`, `filter_complement()`, `filter_hysteresis_threshold()`, `filter_otsu_threshold()`, `filter_local_otsu_threshold()`, `filter_entropy()`, `filter_canny()`
- `filter_remove_small_holes()`, `filter_contrast_stretch()`, `filter_histogram_equalization()`, `filter_adaptive_equalization()`, `filter_local_equalization()`
- `filter_rgb_to_hed()`, `filter_rgb_to_hsv()`, `filter_hsv_to_h()`, `filter_hsv_to_s()`, `filter_hsv_to_v()`, `filter_hed_to_hematoxylin()`, `filter_hed_to_eosin()`
- `filter_binary_fill_holes()`, `filter_binary_erosion()`, `filter_binary_dilation()`, `filter_binary_opening()`, `filter_binary_closing()`
- `filter_kmeans_segmentation()`, `filter_rag_threshold()`, `filter_threshold()`
- `uint8_to_bool()`, `np_to_pil()` (duplicate of `util.np_to_pil`)
- `save_display()`, `save_filtered_image()`
- `singleprocess_apply_filters_to_images()`

After removal, also clean up any imports that are only used by the removed functions (e.g., `skimage` modules like `morphology`, `filters`, `feature`, `exposure`, `color`, `segmentation`, `future.graph` if no remaining function uses them).

---

## Step 6: Remove 2 unused functions from `filter_patches.py`

- `apply_filters_to_image_list()`
- `singleprocess_apply_filters_to_images()`

---

## Step 7: Remove 19 unused functions from `tiles.py`

- `tile_to_np_tile()`, `singleprocess_filtered_images_to_tiles()`
- `image_row()`, `generate_tiled_html_result()`
- Histogram functions: `np_hsv_hue_histogram()`, `np_histogram()`, `np_hsv_saturation_histogram()`, `np_hsv_value_histogram()`, `np_rgb_channel_histogram()`, `np_rgb_r_histogram()`, `np_rgb_g_histogram()`, `np_rgb_b_histogram()`, `pil_hue_histogram()`
- Display functions: `display_image_with_hsv_hue_histogram()`, `display_image()`, `display_image_with_hsv_histograms()`, `display_image_with_rgb_histograms()`, `display_image_with_rgb_and_hsv_histograms()`, `display_tile()`
- `np_text()`, `hsv_purple_vs_pink_average_factor()`, `dynamic_tiles()`, `dynamic_tile()`

After removal, clean up imports only used by removed functions (e.g., `matplotlib` if no remaining code uses it).

---

## Step 8: Remove 1 unused function from `util.py`

- `display_img()`

---

## Step 9: Update `CODE_ANALYSIS.md`

Update the analysis file to reflect the completed cleanup.

---

## Verification

After all changes, verify no active code is broken:

1. **Import check**: Run `python -c "import Cardiac_ACR_Backend_V13_FINAL"` (or equivalent) to verify no `ImportError` or `NameError` at import time
2. **Grep sanity check**: For each removed function, grep the remaining codebase to confirm no references exist
3. **No new files created**: This is purely a deletion/cleanup task — no new files should be introduced
