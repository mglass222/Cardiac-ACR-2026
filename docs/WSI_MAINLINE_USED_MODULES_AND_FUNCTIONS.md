# WSI Mainline: Used Modules and Functions

Scope: static trace of the executable path rooted at `Code/cardiac_acr_diagnose_wsi.py`.

This file focuses on project-local modules and functions that are actually reached by the mainline program, plus a short list of startup side effects and cleanup observations that matter for dead-code removal.

## Entry Script

### `Code/cardiac_acr_diagnose_wsi.py`

Used startup/global flow:

- `check_filesystem()` at line 59
- `utils.initialize_gpu()` at line 368
- `torch.load(...)` at line 371
- `model.to(device)` at line 374
- `model.eval()` at line 377
- `main()` at line 379 when run as `__main__`

Used functions defined in this file:

- `check_filesystem()` line 59
- `filter_patches_multiprocess(slide_number)` line 69
- `classify_patches_batch(slide_number)` line 81
- `Model_Predict_batch(batch, model)` line 150
- `threshold_predictions(slide_number)` line 180
- `diagnose(slide_number)` line 211
- `display_results(slides_to_process)` line 302
- `main()` line 323

Used imported local modules from this file:

- `cardiac_globals` as configuration/constants
- `cardiac_utils`
- `slide`
- `filter`
- `tiles`
- `tileset_utils`
- `filter_patches`
- `count_1r2`
- `annotate_png`
- `annotate_svs`
- `import_openslide` for import-time OpenSlide setup side effect

## Project Modules Reached by Mainline

### `Code/import_openslide.py`

Used at import time only:

- module top-level import logic loads `openslide`
- uses `cg.OPENSLIDE_BIN_PATH`
- enables Windows DLL path setup when needed

### `Code/cardiac_globals.py`

Used as a constants module. Mainline-relevant constants include:

- `ANNOTATION_SIZE`
- `SCALE_FACTOR`
- `PREDICTION_THRESHOLD`
- `BASE_DIR`
- `TEST_SLIDE_DIR`
- `PNG_SLIDE_DIR`
- `TILE_DATA_DIR`
- `TILE_DIR`
- `SPLIT_TILE_DIR`
- `MODEL_DIR`
- `SAVED_DATABASE_DIR`
- `SLIDE_DX_DIR`
- `FILTERED_IMAGE_DIR`
- `TEST_SLIDE_PREDICTIONS_DIR`
- `ANNOTATED_PNG_DIR`
- `TEST_SLIDE_ANNOTATIONS_DIR`
- `ROI_1R2_DIR`
- `ROI_FILTER_DIR`
- `ANNOTATED_1R2_DIR`
- `SEGMENTED_DIR`
- `BOUNDING_BOXES_DIR`
- `COMBINED_BOXES_DIR`
- `_1R2_DILATION_ITERS`
- `FONT_PATH`

### `Code/cardiac_utils.py`

Functions reached from mainline:

- `initialize_gpu()` line 21
- `get_test_slide_numbers()` line 38
- `model_prediction_dict_to_csv(slide_number)` line 51
- `slide_dx_to_csv(slide_dx_dict, filename)` line 84
- `get_png_slide_path(slide_number)` line 109
- `get_png_slide_name(slide_number)` line 124
- `get_coords_from_name(name)` line 135
- `parse_dimensions_from_image_filename(filename)` line 145
- `large_to_small_coords(...)` line 157
- `pad_image_number(number)` line 171
- `get_patchname(tile_name, slide_num, x_start, y_start)` line 179
- `make_directory(directory)` line 14

### `Code/slide.py`

Functions reached from mainline:

- `multiprocess_training_slides_to_images(image_num_list=None)` line 763
- `training_slide_list_to_images(image_num_list)` line 745
- `training_slide_to_image(slide_number)` line 653
- `slide_to_scaled_pil_image(slide_number)` line 673
- `get_training_slide_path(slide_number)` line 151
- `open_slide(filename)` line 103
- `get_training_image_path(...)` line 188
- `get_training_thumbnail_path(...)` line 217
- `save_thumbnail(...)` line 713
- `open_image_np(filename)` line 136
- `open_image(filename)` line 122
- `get_filter_image_result(slide_number)` line 570
- `get_filter_thumbnail_result(slide_number)` line 592
- `parse_dimensions_from_image_filename(filename)` line 614
- `small_to_large_mapping(small_pixel, large_dimensions)` line 635
- `get_tile_image_path(tile)` line 169
- `get_tile_summary_image_path(slide_number)` line 317
- `get_tile_summary_thumbnail_path(slide_number)` line 336
- `get_tile_summary_on_original_image_path(slide_number)` line 355
- `get_tile_summary_on_original_thumbnail_path(slide_number)` line 374
- `get_top_tiles_image_path(slide_number)` line 491
- `get_top_tiles_thumbnail_path(slide_number)` line 510
- `get_top_tiles_on_original_image_path(slide_number)` line 394
- `get_top_tiles_on_original_thumbnail_path(slide_number)` line 413
- `get_tile_data_path(slide_number)` line 551
- `get_tile_summary_image_filename(slide_number, thumbnail=False)` line 433
- `get_top_tiles_image_filename(slide_number, thumbnail=False)` line 462
- `get_tile_data_filename(slide_number)` line 528

### `Code/filter.py`

Functions reached from mainline:

- `multiprocess_apply_filters_to_images(...)` line 714
- `apply_filters_to_image_list(image_num_list, save, display)` line 674
- `apply_filters_to_image(slide_num, save=True, display=False)` line 483
- `apply_image_filters(np_img, ...)` line 437
- `filter_green_channel(...)` line 98
- `filter_grays(...)` line 405
- `filter_red_pen(rgb, output_type="bool")` line 171
- `filter_red(...)` line 137
- `filter_green_pen(rgb, output_type="bool")` line 238
- `filter_green(...)` line 202
- `filter_blue_pen(rgb, output_type="bool")` line 309
- `filter_blue(...)` line 275
- `filter_black_pen(rgb, output_type="bool")` line 378
- `filter_black(...)` line 344
- `filter_remove_small_objects(...)` line 59
- `tissue_percent(np_img)` line 46
- `mask_percent(np_img)` line 28

Referenced by `tiles.py` but missing from the current file:

- `filter_rgb_to_hsv`
- `filter_hsv_to_h`
- `filter_hsv_to_s`
- `filter_hsv_to_v`

These names are on the mainline scoring path and should be resolved before cleanup work.

### `Code/tiles.py`

Functions/classes reached from mainline:

- `multiprocess_filtered_images_to_tiles(...)` line 845
- `image_list_to_tiles(...)` line 806
- `summary_and_tiles(...)` line 508
- `score_tiles(...)` line 625
- `get_num_tiles(...)` line 79
- `get_tile_indices(...)` line 99
- `score_tile(...)` line 730
- `tissue_quantity_factor(amount)` line 765
- `tissue_quantity(tissue_percentage)` line 786
- `generate_tile_summaries(...)` line 153
- `generate_top_tile_summaries(...)` line 212
- `create_summary_pil_img(...)` line 125
- `tile_border_color(tissue_percentage)` line 330
- `tile_border(...)` line 413
- `summary_title(tile_summary)` line 372
- `summary_stats(tile_summary)` line 385
- `add_tile_stats_to_top_tile_summary(...)` line 291
- `np_tile_stat_img(tiles)` line 306
- `pil_text(...)` line 917
- `rgb_to_hues(rgb)` line 964
- `hsv_saturation_and_value_factor(rgb)` line 979
- `hsv_purple_deviation(hsv_hues)` line 1012
- `hsv_pink_deviation(hsv_hues)` line 1026
- `hsv_purple_pink_factor(rgb)` line 1040
- `TileSummary` class line 1067
- `TileSummary.tiles_by_score()` line 1139
- `TileSummary.top_tiles()` line 1149
- `Tile` class line 1209
- `Tile.save_tile()` line 1256
- `save_display_tile(tile, save=True, display=False)` line 601
- `tile_to_pil_tile(tile)` line 579

Conditionally skipped by current mainline parameters:

- `save_tile_data(...)` line 548 is not reached because `save_data=False`
- summary image save helpers are not reached because `save_summary=False`

Behavior note:

- `summary_and_tiles()` forcibly resets `save_top_tiles = True` inside the function, so top-tile extraction still runs even though the caller passes `save_top_tiles=False`.

### `Code/tileset_utils.py`

Functions reached from mainline:

- `process_tilesets_multiprocess(slide_num)` line 33
- `process_tiles(slide_num)` line 49
- `tiles_to_patches(tile_list, slide_num)` line 109

### `Code/filter_patches.py`

Functions reached from mainline:

- `multiprocess_apply_filters_to_images(folder, save=False, ...)` line 248
- `apply_filters_to_image_list_multiprocess(...)` line 220
- `apply_filters_to_image(image, save_dir, save, display=False)` line 178
- `apply_image_filters(np_img, save=True, display=False)` line 148
- `filter_green_channel(...)` line 71
- `filter_grays(...)` line 114
- `tissue_percent(np_img)` line 55
- `mask_percent(np_img)` line 35

### `Code/count_1r2.py`

Functions reached from mainline:

- `main(slide_number)` line 477
- `annotate_1r2(slide_number)` line 225
- `segment_image(slide_number)` line 306
- `analyze_segments(slide_number)` line 423
- `remove_small(cnts)` line 153
- `enlarge_boxes(x, y, w, h, offset, image_dims)` line 190
- `analyze_boxes(bounding_boxes)` line 53
- `check_overlap(box1, box2)` line 120
- `combine_boxes(box1, box2)` line 31
- `remove_duplicates(combined_boxes)` line 90
- `filter_boxes(boxes)` line 99
- `calculate_area(box)` line 204
- `get_coords(box)` line 214

Imported but not reached on the current mainline:

- `pad_image(...)` line 167

### `Code/annotate_png.py`

Functions reached from mainline:

- `main(slide_number)` line 129
- `annotate_png(slide_number)` line 16
- `get_color(value)` line 97

### `Code/annotate_svs.py`

Functions reached from mainline:

- `main(slide_number)` line 343
- `annotate_slide(slide_number)` line 23
- `initilialize_xml_file(xmlfilename)` line 66
- `load_diagnoses(slide_number)` line 77
- `random_sample(dx_dict, num_samples)` line 115
- `update_xml_file(root, current_dict, annotation_id, region_id)` line 144
- `initialize_annotation_type(root, dx, color, annotation_id)` line 172
- `add_region(annotation, dx, region_id, patchname)` line 233
- `get_coords(patchname)` line 288
- `get_coords_from_name(name)` line 332
- `pretty_print(xmlfilename)` line 55
- `get_extracted_slide_name(slide_number)` line 320

Behavior note:

- `get_extracted_slide_name(slide_number)` is called inside `annotate_slide()` but its return value is never used.

### `Code/util.py`

Functions/classes reached from mainline:

- `pil_to_np_rgb(pil_img)` line 25
- `np_to_pil(np_img)` line 43
- `np_info(np_arr, name=None, elapsed=None)` line 60
- `mask_rgb(rgb, mask)` line 87
- `Time` class line 104

## External Libraries Touched on Mainline

High-value external dependencies on the execution path:

- `torch`
- `torchvision.transforms`
- `openslide`
- `PIL`
- `numpy`
- `cv2`
- `skimage.filters`
- `skimage.measure`
- `scipy.ndimage`
- `xml.etree.ElementTree`

## Cleanup Notes

These are useful when the next step is dead-code removal:

- The mainline path definitely uses `slide`, `filter`, `tiles`, `tileset_utils`, `filter_patches`, `cardiac_utils`, `count_1r2`, `annotate_png`, `annotate_svs`, `util`, `cardiac_globals`, and `import_openslide`.
- `cardiac_acr_diagnose_wsi.py` appears to carry unused imports:
  - `torchvision`
  - `optim`
  - `cuda`
  - `datasets`
  - `models`
- `tiles.py` currently depends on four HSV helper functions that are not present in `filter.py`. Treat those as unresolved runtime dependencies before deleting anything around tile scoring.
- `tiles.summary_and_tiles()` ignores the caller's `save_top_tiles=False` by hard-setting `save_top_tiles = True`.
- `count_1r2.pad_image()` is not on the traced mainline path.
- `annotate_svs.get_extracted_slide_name()` is called, but its return value is unused.
