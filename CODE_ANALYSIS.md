# Cardiac-ACR Code Analysis

## Date: 2026-03-28 (updated after cleanup)

## Entry Point
`cardiac_acr_diagnose_wsi.py`

---

## Active Execution Flow

### Module-level (runs on import):
1. Imports all project modules (triggers module-level code in each)
2. `check_filesystem()` — ensures directories exist
3. `utils.initialize_gpu()` — sets up CUDA/CPU device
4. Loads PyTorch model (`resnet50_ft`)
5. Calls `main()`

### main() call chain:
```
main()
├── utils.get_test_slide_numbers()  ← determines which slides to process
├── slide.multiprocess_training_slides_to_images()
│   └── training_slide_list_to_images() → training_slide_to_image()
│       └── slide_to_scaled_pil_image() → open_slide(), get_training_slide_path()
│           save via: get_training_image_path(), save_thumbnail(), get_training_thumbnail_path()
│
├── filter.multiprocess_apply_filters_to_images()
│   └── apply_filters_to_image_list() → apply_filters_to_image()
│       └── apply_image_filters()
│           ├── filter_green_channel()
│           ├── filter_grays()
│           ├── filter_red_pen() → filter_red()
│           ├── filter_green_pen() → filter_green()
│           ├── filter_blue_pen() → filter_blue()
│           ├── filter_black_pen() → filter_black()
│           ├── filter_remove_small_objects()
│           └── util.mask_rgb()
│
├── tiles.multiprocess_filtered_images_to_tiles()
│   └── image_list_to_tiles() → summary_and_tiles()
│       ├── score_tiles() → get_num_tiles(), get_tile_indices(),
│       │   score_tile() → hsv_purple_pink_factor(), hsv_saturation_and_value_factor(),
│       │   tissue_quantity(), tissue_quantity_factor(), slide.small_to_large_mapping()
│       ├── save_tile_data()
│       ├── generate_tile_summaries() → create_summary_pil_img(), tile_border_color(), tile_border()
│       ├── generate_top_tile_summaries() → faded_tile_border_color(), add_tile_stats_to_top_tile_summary()
│       └── Tile.save_tile() → save_display_tile() → tile_to_pil_tile()
│
├── For each slide:
│   ├── tileset_utils.process_tilesets_multiprocess()
│   │   └── process_tiles() → tiles_to_patches() → utils.get_patchname(), utils.pad_image_number()
│   ├── filter_patches_multiprocess() [local function]
│   │   └── filter_patches.multiprocess_apply_filters_to_images()
│   │       └── apply_filters_to_image_list_multiprocess() → apply_filters_to_image()
│   │           └── apply_image_filters() → filter_green_channel(), filter_grays(), util.mask_rgb()
│   ├── classify_patches_batch()
│   │   └── Model_Predict_batch()
│   ├── threshold_predictions()
│   │   └── utils.model_prediction_dict_to_csv()
│   ├── diagnose()
│   │   ├── count_1r2.main()
│   │   │   ├── annotate_1r2() → utils.get_png_slide_path(), get_png_slide_name(),
│   │   │   │   parse_dimensions_from_image_filename(), get_coords_from_name(), large_to_small_coords()
│   │   │   ├── segment_image() → remove_small(), enlarge_boxes(), analyze_boxes(),
│   │   │   │   combine_boxes(), check_overlap(), remove_duplicates(), filter_boxes(),
│   │   │   │   get_coords(), utils.make_directory()
│   │   │   └── analyze_segments() → calculate_area()
│   │   └── utils.slide_dx_to_csv()
│   ├── annotate_png.main()
│   │   └── annotate_png() → utils.get_png_slide_path(), get_png_slide_name(),
│   │       parse_dimensions_from_image_filename(), get_coords_from_name(),
│   │       large_to_small_coords(), make_directory()
│   │   └── get_color()
│   └── annotate_svs.main()
│       └── annotate_slide() → get_extracted_slide_name(), initilialize_xml_file(),
│           load_diagnoses() → random_sample(), update_xml_file() → initialize_annotation_type(),
│           add_region() → get_coords() → get_coords_from_name() [internal version],
│           pretty_print(), utils.make_directory()
│
└── display_results()
```

---

## Cleanup Completed (2026-03-28)

### Files deleted (7 total):
- `cardiac_dirs.py` — not imported by any file
- `slide_backup.py` — old copy of slide.py, not imported
- `count_1r2_new.py` — alternate version of count_1r2.py, not imported
- `count_1r2_testing.py` — test version of count_1r2.py, not imported
- `segmentation_tools.py` — not imported by any file
- `filter_tiles.py` — imported but no function ever called into it
- `Cardiac_ACR_Backend_V13_FINAL.ipynb` — Jupyter notebook duplicate

### Functions removed:
- **cardiac_utils.py**: 11 unused functions removed (get_filtered_image_path, get_files, get_training_slide_numbers, get_slide_info, get_patches_dir_from_slide_number, small_to_large_coords, clean_csv_files, make_top_slides_csv_file, filter_tiles_multiprocess, display_tissue_percent_patches, display_tissue_percent_tiles)
- **slide.py**: 6 unused functions removed (get_tile_image_path_by_slide_row_col, show_slide, training_slide_range_to_images, singleprocess_training_slides_to_images, slide_stats, slide_info)
- **filter.py**: 32 unused functions removed (all unused filter variants, save_display, save_filtered_image, singleprocess version)
- **filter_patches.py**: 2 unused functions removed (apply_filters_to_image_list, singleprocess_apply_filters_to_images)
- **tiles.py**: 23 unused functions removed (histogram, display, HTML generation, and other unused functions)
- **util.py**: 1 unused function removed (display_img)

### Other cleanup:
- Removed unused imports across all files
- Removed Jupyter `# In[X]:` cell markers from main file
- Removed unused module-level variables (TISSUE_PERCENT_DIR, FONT_PATH) from main file
- Uncommented `slides_to_process = utils.get_test_slide_numbers()` and removed hardcoded `[111]`
- Fixed dead reference to removed `training_slide_range_to_images` in `multiprocess_training_slides_to_images`

## Cross-Platform Path Refactor (2026-03-28)

### Problem
All paths were hardcoded Windows paths (`D:\Cardiac_ACR\`, `E:\Cardiac_ACR\`, `C:\Windows\Fonts\`) with `\\` separators, making the code Windows-only.

### Solution
- **`cardiac_globals.py`**: Rewrote to derive all paths from `PROJECT_ROOT` using `os.path.join()`. Data lives under `data/` in the project folder. Font lives under `fonts/`.
- **`OPENSLIDE_BIN_PATH`**: Now read from `OPENSLIDE_BIN_PATH` environment variable (empty by default; only needed on Windows).
- **`count_1r2.py`**: Replaced 8 hardcoded paths with `cg.*` references from cardiac_globals.
- **All files**: Replaced every `"\\"` concatenation with `os.path.join()`. Fixed `split("\\")[5]` filename extraction with `os.path.basename()`.
- **`slide.py` / `import_openslide.py`**: OpenSlide import is now platform-aware (uses `add_dll_directory` only on Windows when path is set).
- **`tiles.py`**: Font paths now reference `cg.FONT_PATH` instead of hardcoded Windows path.
- **Main file**: Removed `sys.path.insert()` hack (unnecessary when all files are in same directory).

### Data directory structure
```
data/
├── WSI/Test/                              (input slides)
├── WSI/Training/
├── WSI/TEST_SLIDE_ANNOTATIONS/Weighted_Loss/
├── DeepHistoPath/training_png/
├── DeepHistoPath/tile_data/
├── DeepHistoPath/tiles_png/
├── DeepHistoPath/filter_png/
├── DeepHistoPath/tiles_png_split/
├── Backend/Saved_Databases/Weighted_Loss/
├── Backend/Annotated_Test_Slides/Weighted_Loss/
├── Backend/Slide_Dx/Weighted_Loss/
├── Backend/Test_Slide_Predictions/Weighted_Loss/
├── Backend/Count_1R2/
│   ├── ROI-1R2-Only/
│   ├── ROI-Filtered-PNG/
│   ├── Annotated_1R2/
│   └── Segmented/ (Bounding_Boxes/, Combined_Boxes/)
└── Saved_Models/Weighted_Loss/
fonts/
    arial.ttf                              (user must place font file here)
```

### Setup on Windows
Set the environment variable for OpenSlide:
```
set OPENSLIDE_BIN_PATH=C:\Openslide_4003\bin
```

## Notes
- `annotate_svs.py` has its own internal `get_coords_from_name()` that duplicates the one in `cardiac_utils.py`
- All remaining code is actively used in the execution path
- Place `.svs` slide files in `data/WSI/Test/` and trained model in `data/Saved_Models/Weighted_Loss/`
- Place `arial.ttf` (or compatible font) in `fonts/`
