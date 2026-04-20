# WSI Mainline Code Trace

Scope: start-to-finish trace of the executable path from `Code/cardiac_acr_diagnose_wsi.py`.

This is a static trace based on the current repository state. It is meant to support dead-code removal by showing what the program actually walks through when the script is executed.

## 1. Process Startup

Entry file: `Code/cardiac_acr_diagnose_wsi.py`

Import and startup order:

1. Project modules import:
   - `cardiac_globals`
   - `filter`
   - `slide`
   - `tiles`
   - `tileset_utils`
   - `filter_patches`
   - `cardiac_utils`
   - `count_1r2`
   - `annotate_svs`
   - `annotate_png`
   - `import_openslide`
2. `import_openslide` runs import-time setup so `openslide` can be imported on Windows when `OPENSLIDE_BIN_PATH` is set.
3. Global constants are copied from `cardiac_globals`.
4. `check_filesystem()` creates:
   - `MODEL_DIR`
   - `SAVED_DATABASE_DIR`
   - `SLIDE_DX_DIR`
5. GPU/model setup runs before `main()`:
   - `device = utils.initialize_gpu()`
   - `model = torch.load(MODEL_DIR + "resnet50_ft")`
   - `model = model.to(device)`
   - `model.eval()`
6. If run as a script, `main()` executes.

## 2. Top-Level Main Flow

Main function: `cardiac_acr_diagnose_wsi.main()`

### Phase A: discover slides

1. `utils.get_test_slide_numbers()`
2. Reads `cg.TEST_SLIDE_DIR`
3. Filters for `.svs`
4. Converts filenames like `001.svs` to integers like `1`
5. Returns `slides_to_process`

### Phase B: whole-slide preprocessing for all slides

This runs once across the full slide list before the per-slide loop.

1. `slide.multiprocess_training_slides_to_images(image_num_list=slides_to_process)`
2. `filter.multiprocess_apply_filters_to_images(image_num_list=slides_to_process)`
3. `tiles.multiprocess_filtered_images_to_tiles(save_top_tiles=False, image_num_list=slides_to_process)`

### Phase C: per-slide loop

For each `slide_number` in `slides_to_process`:

1. `tileset_utils.process_tilesets_multiprocess(slide_number)`
2. `filter_patches_multiprocess(slide_number)`
3. `classify_patches_batch(slide_number)`
4. `threshold_predictions(slide_number)`
5. `diagnose(slide_number)`
6. `annotate_png.main(slide_number)`
7. `annotate_svs.main(slide_number)`

### Phase D: final reporting

1. `display_results(slides_to_process)`
2. Loads the saved slide diagnosis pickle
3. Drops diagnoses for any slide no longer in `slides_to_process`
4. Prints each diagnosis

## 3. Detailed Call Trace

### A. Slide image extraction

Call chain:

1. `slide.multiprocess_training_slides_to_images(...)`
2. worker processes call `slide.training_slide_list_to_images(sublist)`
3. each slide calls `slide.training_slide_to_image(slide_number)`
4. `slide.training_slide_to_image()` calls `slide.slide_to_scaled_pil_image(slide_number)`
5. `slide.slide_to_scaled_pil_image()` calls:
   - `slide.get_training_slide_path(slide_number)`
   - `slide.open_slide(filename)`
6. `openslide` reads the WSI:
   - `slide.dimensions`
   - `slide.get_best_level_for_downsample(SCALE_FACTOR)`
   - `slide.read_region((0, 0), level, slide.level_dimensions[level])`
7. PIL converts and resizes the image
8. `slide.training_slide_to_image()` then calls:
   - `slide.get_training_image_path(...)`
   - `slide.get_training_thumbnail_path(...)`
   - `slide.save_thumbnail(...)`
9. Output:
   - scaled PNG in `training_png`
   - thumbnail JPG in `training_thumbnail_jpg`

### B. Whole-slide tissue filtering

Call chain:

1. `filter.multiprocess_apply_filters_to_images(...)`
2. worker processes call `filter.apply_filters_to_image_list(sublist, save, display)`
3. each slide calls `filter.apply_filters_to_image(slide_num, save=True, display=False)`
4. `filter.apply_filters_to_image()` loads the scaled slide:
   - `slide.get_training_image_path(slide_num)`
   - `slide.open_image_np(img_path)`
   - `slide.open_image()`
   - `util.pil_to_np_rgb()`
5. `filter.apply_image_filters(np_orig, ...)` builds masks in sequence:
   - `filter_green_channel()`
   - `filter_grays()`
   - `filter_red_pen()` -> repeated `filter_red()`
   - `filter_green_pen()` -> repeated `filter_green()`
   - `filter_blue_pen()` -> repeated `filter_blue()`
   - `filter_black_pen()` -> `filter_black()`
   - `filter_remove_small_objects()`
   - `util.mask_rgb()` after each major mask stage
6. The filtered image is saved via:
   - `slide.get_filter_image_result(slide_num)`
   - `util.np_to_pil()`
   - `slide.get_filter_thumbnail_result(slide_num)`
   - `slide.save_thumbnail(...)`
7. Output:
   - filtered slide PNG
   - filtered slide thumbnail JPG

### C. Whole-slide tile scoring and top-tile extraction

Call chain:

1. `tiles.multiprocess_filtered_images_to_tiles(save_top_tiles=False, image_num_list=slides_to_process)`
2. worker processes call `tiles.image_list_to_tiles(sublist, ...)`
3. each slide calls `tiles.summary_and_tiles(slide_num, ...)`
4. `tiles.summary_and_tiles()`:
   - loads filtered slide via `slide.get_filter_image_result()` and `slide.open_image_np()`
   - calls `tiles.score_tiles(slide_num, np_img)`
5. `tiles.score_tiles()`:
   - gets dimensions from `slide.parse_dimensions_from_image_filename(...)`
   - computes tile grid using `tiles.get_num_tiles()` and `tiles.get_tile_indices()`
   - computes total tissue with `filter.tissue_percent(np_img)`
   - creates `TileSummary`
   - for each tile:
     - computes tile tissue with `filter.tissue_percent(np_tile)`
     - classifies quantity via `tiles.tissue_quantity()`
     - maps scaled coordinates back to original via `slide.small_to_large_mapping()`
     - scores tile via `tiles.score_tile(np_tile, t_p, slide_num, r, c)`
6. `tiles.score_tile()`:
   - `tiles.hsv_purple_pink_factor(np_tile)`
   - `tiles.hsv_saturation_and_value_factor(np_tile)`
   - `tiles.tissue_quantity()`
   - `tiles.tissue_quantity_factor()`
   - computes final score
7. `tiles.hsv_purple_pink_factor()` calls:
   - `tiles.rgb_to_hues()`
   - `tiles.hsv_purple_deviation()`
   - `tiles.hsv_pink_deviation()`
8. `tiles.rgb_to_hues()` currently expects:
   - `filter.filter_rgb_to_hsv()`
   - `filter.filter_hsv_to_h()`
9. `tiles.hsv_saturation_and_value_factor()` currently expects:
   - `filter.filter_rgb_to_hsv()`
   - `filter.filter_hsv_to_s()`
   - `filter.filter_hsv_to_v()`
10. After scoring, `tiles.summary_and_tiles()` also calls:
   - `tiles.generate_tile_summaries(...)`
   - `tiles.generate_top_tile_summaries(...)`
11. `tiles.generate_tile_summaries(...)` and `tiles.generate_top_tile_summaries(...)` call:
   - `tiles.create_summary_pil_img()`
   - `slide.get_training_image_path()`
   - `slide.open_image_np()`
   - `tiles.tile_border_color()`
   - `tiles.tile_border()`
   - `tiles.summary_title()`
   - `tiles.summary_stats()`
   - `tiles.add_tile_stats_to_top_tile_summary()`
   - `tiles.np_tile_stat_img()`
   - `tiles.pil_text()`
12. Even though the caller passes `save_top_tiles=False`, `tiles.summary_and_tiles()` overrides that locally and still saves top tiles.
13. Saving top tiles happens through:
   - `tile_sum.top_tiles()`
   - each `Tile.save_tile()`
   - `tiles.save_display_tile(tile, save=True, display=False)`
   - `tiles.tile_to_pil_tile(tile)`
   - `slide.get_training_slide_path()`
   - `slide.open_slide()`
   - `openslide.read_region(...)`
   - `slide.get_tile_image_path(tile)`
14. Output:
   - top tile PNGs in `tiles_png/<slide>/...`
   - optional summary objects in memory

Important note:

- The current repository is missing the HSV helper functions that `tiles.py` calls from `filter.py`. That means the traced tile-scoring path is conceptually on the mainline, but it appears broken in the current code state.

### D. Split top tiles into 224x224 patches

Call chain:

1. `tileset_utils.process_tilesets_multiprocess(slide_number)`
2. pads slide number with `utils.pad_image_number()`
3. calls `tileset_utils.process_tiles(slide_num)`
4. `process_tiles()`:
   - uses tile images from `cg.TILE_DIR/<slide>`
   - creates output dir via `utils.make_directory()`
   - creates multiprocessing tasks
5. worker processes call `tileset_utils.tiles_to_patches(tile_list, slide_num)`
6. each tile image:
   - opens with `PIL.Image.open`
   - splits into `cg.ANNOTATION_SIZE` patches
   - pads edge patches with black when needed
   - names patches with `utils.get_patchname(...)`
   - saves patch PNGs into `cg.SPLIT_TILE_DIR/<slide>`

### E. Patch-level tissue filtering

Call chain:

1. `cardiac_acr_diagnose_wsi.filter_patches_multiprocess(slide_number)`
2. calls `filter_patches.multiprocess_apply_filters_to_images(slide_number, save=False)`
3. worker processes call `filter_patches.apply_filters_to_image_list_multiprocess(...)`
4. each patch calls `filter_patches.apply_filters_to_image(image, save_dir, save=False, ...)`
5. `filter_patches.apply_filters_to_image()`:
   - opens patch via PIL
   - converts to NumPy
   - calls `filter_patches.apply_image_filters(np_img, ...)`
6. `filter_patches.apply_image_filters()` is a simplified patch filter:
   - `filter_green_channel()`
   - `filter_grays()`
   - combines masks
   - `util.mask_rgb()`
7. The returned filtered patch image is scored with:
   - `filter_patches.tissue_percent()`
   - `filter_patches.mask_percent()`
8. Back in `cardiac_acr_diagnose_wsi.filter_patches_multiprocess()`:
   - if tissue percent `< 50`
   - patch file is deleted from `cg.SPLIT_TILE_DIR/<slide>`

### F. Batch patch classification

Call chain:

1. `classify_patches_batch(slide_number)`
2. loads patch file paths from `cg.SPLIT_TILE_DIR/<slide>`
3. builds batches of 200 PIL images
4. each batch calls `Model_Predict_batch(batch, model)`
5. `Model_Predict_batch()`:
   - builds torchvision transform pipeline:
     - `Resize(INPUT_SIZE)`
     - `CenterCrop(INPUT_SIZE)`
     - `ToTensor()`
     - `Normalize(mean, std)`
   - stacks tensors with `torch.stack`
   - moves batch to `device`
   - runs model under `torch.no_grad()`
6. back in `classify_patches_batch()`:
   - applies `torch.nn.functional.softmax`
   - moves to CPU
   - converts to NumPy
   - stores `{patch_path: class_probs}`
7. output:
   - `model_predictions_dict_<slide>.pickle`

### G. Probability thresholding

Call chain:

1. `threshold_predictions(slide_number)`
2. loads `model_predictions_dict_<slide>.pickle`
3. keeps only predictions where any class probability exceeds `PREDICTION_THRESHOLD`
4. saves:
   - `model_predictions_dict_<slide>_filtered.pickle`
5. exports CSV via:
   - `utils.model_prediction_dict_to_csv(slide_number)`
   - `utils.make_directory(cg.TEST_SLIDE_PREDICTIONS_DIR)`

### H. Slide diagnosis

Call chain:

1. `diagnose(slide_number)`
2. loads `model_predictions_dict_<slide>_filtered.pickle`
3. loads or initializes slide diagnosis dictionary
4. counts predicted classes with `np.argmax`
5. delegates 1R2 cluster counting to:
   - `count_1r2.main(slide_number)`

#### H1. `count_1r2.main(slide_number)`

Call chain:

1. `annotate_1r2(slide_number)`
2. `segment_image(slide_number)`
3. `analyze_segments(slide_number)`
4. returns `max_1r2`

#### H2. `count_1r2.annotate_1r2(slide_number)`

1. loads filtered prediction pickle
2. finds slide PNG via:
   - `utils.get_png_slide_path()`
   - `utils.get_png_slide_name()`
3. parses slide dimensions via `utils.parse_dimensions_from_image_filename()`
4. for each patch whose class is `1R2`:
   - `utils.get_coords_from_name()`
   - `utils.large_to_small_coords()`
   - draw a small red box
5. saves `<slide>_1r2.png`

#### H3. `count_1r2.segment_image(slide_number)`

1. loads the 1R2-only overlay image with OpenCV
2. finds the matching filtered slide image in `cg.FILTERED_IMAGE_DIR`
3. applies image processing:
   - grayscale
   - Gaussian blur
   - Canny edge detection
   - dilation
   - contour detection
4. removes small contours via `remove_small()`
5. enlarges contour boxes via `enlarge_boxes()`
6. combines overlapping boxes via:
   - `analyze_boxes()`
   - `check_overlap()`
   - `combine_boxes()`
   - `remove_duplicates()`
   - `filter_boxes()`
   - `calculate_area()`
   - `get_coords()`
7. extracts ROI images and saves:
   - `ROI_1R2`
   - `ROI_FILTER`
8. writes combined-box visualizations

#### H4. `count_1r2.analyze_segments(slide_number)`

1. loads each ROI image from `ROI_1R2/<slide>`
2. converts to grayscale NumPy
3. runs:
   - Otsu thresholding
   - hole filling
   - binary dilation
   - connected-component labeling
4. collects `num_1r2` for each ROI
5. returns `max(segment_list)`

#### H5. Back in `diagnose(slide_number)`

1. receives `_1R2_count`
2. derives slide diagnosis string:
   - `0R`
   - `1R1A`
   - `1R2`
   - `2R`
3. saves diagnosis pickle
4. exports diagnosis CSV via `utils.slide_dx_to_csv(...)`

### I. Annotated PNG generation

Call chain:

1. `annotate_png.main(slide_number)`
2. `annotate_png.annotate_png(slide_number)`
3. loads filtered prediction pickle
4. gets slide PNG path/name via:
   - `utils.get_png_slide_path()`
   - `utils.get_png_slide_name()`
5. parses dimensions via `utils.parse_dimensions_from_image_filename()`
6. for each filtered patch:
   - computes class with `np.argmax`
   - maps class to RGBA via `get_color()`
   - gets patch coords via `utils.get_coords_from_name()`
   - converts to scaled coords via `utils.large_to_small_coords()`
   - draws a color-coded box
7. ensures save directory exists with `utils.make_directory()`
8. writes annotated PNG

### J. XML annotation generation

Call chain:

1. `annotate_svs.main(slide_number)`
2. `annotate_svs.annotate_slide(slide_number)`
3. calls `get_extracted_slide_name(slide_number)` but does not use the result
4. ensures XML output directory exists with `utils.make_directory()`
5. initializes blank XML via `initilialize_xml_file(xmlfilename)`
6. parses XML into ElementTree
7. loads diagnosis dictionaries via `load_diagnoses(slide_number)`
8. `load_diagnoses()`:
   - loads filtered prediction pickle
   - partitions patches by `np.argmax`
   - calls `random_sample()` for each class-specific dict
9. for each class dict:
   - `update_xml_file(root, current_dict, annotation_id, region_id)`
10. `update_xml_file()`:
   - finds or creates annotation type via `initialize_annotation_type()`
   - appends each patch via `add_region()`
11. `add_region()`:
   - computes patch box via `get_coords(patchname)`
   - `get_coords()` calls `get_coords_from_name()`
   - writes four rectangle vertices into XML
12. XML is written and reformatted with `pretty_print(xmlfilename)`

## 4. Output Artifacts Produced Along the Path

Mainline outputs include:

- scaled slide PNGs
- filtered slide PNGs
- top tile PNGs
- split patch PNGs
- patch prediction pickle
- filtered patch prediction pickle
- prediction CSV
- slide diagnosis pickle
- slide diagnosis CSV
- annotated PNG overlays
- XML annotation files
- 1R2 intermediate ROI and segmentation images

## 5. Mainline Issues Found While Tracing

These are not cleanup decisions by themselves, but they matter before removing code:

1. `tiles.py` depends on `filter.filter_rgb_to_hsv`, `filter.filter_hsv_to_h`, `filter.filter_hsv_to_s`, and `filter.filter_hsv_to_v`, but those functions are not present in the current `Code/filter.py`.
2. `tiles.summary_and_tiles()` overrides its own `save_top_tiles` argument and always enables top-tile saving.
3. `annotate_svs.annotate_slide()` calls `get_extracted_slide_name()` without using the return value.
4. `count_1r2.pad_image()` is defined but is not reached on the traced path.
5. In `diagnose()`, the `class_count` updates for Hemorrhage and Normal appear to target the wrong keys:
   - Hemorrhage branch updates `"Normal"`
   - Normal branch updates `"Quilty"`

## 6. Practical Cleanup Order

If the next step is code removal, the safest order is:

1. Fix or resolve the missing HSV helper dependency first.
2. Preserve every function listed in the mainline inventory until the runtime path is validated.
3. Start pruning with functions explicitly noted as not reached on the mainline.
4. After that, inspect modules for imports and helpers that are only used by already-dead code paths.
