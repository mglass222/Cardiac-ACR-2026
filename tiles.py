# ------------------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ------------------------------------------------------------------------

import math
import multiprocessing
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from enum import Enum

import util
import filter
import slide
from util import Time

TISSUE_HIGH_THRESH = 80
TISSUE_LOW_THRESH = 10

# ROW_TILE_SIZE = 1024
# COL_TILE_SIZE = 1024

# for 224 x 224 patches
ROW_TILE_SIZE = 1120
COL_TILE_SIZE = 1120



NUM_TOP_TILES = 50

DISPLAY_TILE_SUMMARY_LABELS = False
TILE_LABEL_TEXT_SIZE = 10
LABEL_ALL_TILES_IN_TOP_TILE_SUMMARY = False
BORDER_ALL_TILES_IN_TOP_TILE_SUMMARY = False

TILE_BORDER_SIZE = 2  # The size of the colored rectangular border around summary tiles.

HIGH_COLOR = (0, 255, 0)
MEDIUM_COLOR = (255, 255, 0)
LOW_COLOR = (255, 165, 0)
NONE_COLOR = (255, 0, 0)

FADED_THRESH_COLOR = (128, 255, 128)
FADED_MEDIUM_COLOR = (255, 255, 128)
FADED_LOW_COLOR = (255, 210, 128)
FADED_NONE_COLOR = (255, 128, 128)

# FONT_PATH = "/Library/Fonts/Arial Bold.ttf"
# SUMMARY_TITLE_FONT_PATH = "/Library/Fonts/Courier New Bold.ttf"

FONT_PATH = "C:\\Windows\\Fonts\\Arial.ttf"
SUMMARY_TITLE_FONT_PATH = "C:\\Windows\\Fonts\\Arial.ttf"



SUMMARY_TITLE_TEXT_COLOR = (0, 0, 0)
SUMMARY_TITLE_TEXT_SIZE = 14
SUMMARY_TILE_TEXT_COLOR = (255, 255, 255)
TILE_TEXT_COLOR = (0, 0, 0)
TILE_TEXT_SIZE = 16
TILE_TEXT_BACKGROUND_COLOR = (255, 255, 255)
TILE_TEXT_W_BORDER = 5
TILE_TEXT_H_BORDER = 4

HSV_PURPLE = 270
HSV_PINK = 330


def get_num_tiles(rows, cols, row_tile_size, col_tile_size):
  """
  Obtain the number of vertical and horizontal tiles that an image can be divided into given a row tile size and
  a column tile size.

  Args:
    rows: Number of rows.
    cols: Number of columns.
    row_tile_size: Number of pixels in a tile row.
    col_tile_size: Number of pixels in a tile column.

  Returns:
    Tuple consisting of the number of vertical tiles and the number of horizontal tiles that the image can be divided
    into given the row tile size and the column tile size.
  """
  num_row_tiles = math.ceil(rows / row_tile_size)
  num_col_tiles = math.ceil(cols / col_tile_size)
  return num_row_tiles, num_col_tiles


def get_tile_indices(rows, cols, row_tile_size, col_tile_size):
  """
  Obtain a list of tile coordinates (starting row, ending row, starting column, ending column, row number, column number).

  Args:
    rows: Number of rows.
    cols: Number of columns.
    row_tile_size: Number of pixels in a tile row.
    col_tile_size: Number of pixels in a tile column.

  Returns:
    List of tuples representing tile coordinates consisting of starting row, ending row,
    starting column, ending column, row number, column number.
  """
  indices = list()
  num_row_tiles, num_col_tiles = get_num_tiles(rows, cols, row_tile_size, col_tile_size)
  for r in range(0, num_row_tiles):
    start_r = r * row_tile_size
    end_r = ((r + 1) * row_tile_size) if (r < num_row_tiles - 1) else rows
    for c in range(0, num_col_tiles):
      start_c = c * col_tile_size
      end_c = ((c + 1) * col_tile_size) if (c < num_col_tiles - 1) else cols
      indices.append((start_r, end_r, start_c, end_c, r + 1, c + 1))
  return indices


def create_summary_pil_img(np_img, title_area_height, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles):
  """
  Create a PIL summary image including top title area and right side and bottom padding.

  Args:
    np_img: Image as a NumPy array.
    title_area_height: Height of the title area at the top of the summary image.
    row_tile_size: The tile size in rows.
    col_tile_size: The tile size in columns.
    num_row_tiles: The number of row tiles.
    num_col_tiles: The number of column tiles.

  Returns:
    Summary image as a PIL image. This image contains the image data specified by the np_img input and also has
    potentially a top title area and right side and bottom padding.
  """
  r = row_tile_size * num_row_tiles + title_area_height
  c = col_tile_size * num_col_tiles
  summary_img = np.zeros([r, c, np_img.shape[2]], dtype=np.uint8)
  # add gray edges so that tile text does not get cut off
  summary_img.fill(120)
  # color title area white
  summary_img[0:title_area_height, 0:summary_img.shape[1]].fill(255)
  summary_img[title_area_height:np_img.shape[0] + title_area_height, 0:np_img.shape[1]] = np_img
  summary = util.np_to_pil(summary_img)
  return summary


def generate_tile_summaries(tile_sum, np_img, display=True, save_summary=False):
  """
  Generate summary images/thumbnails showing a 'heatmap' representation of the tissue segmentation of all tiles.

  Args:
    tile_sum: TileSummary object.
    np_img: Image as a NumPy array.
    display: If True, display tile summary to screen.
    save_summary: If True, save tile summary images.
  """
  z = 300  # height of area at top of summary slide
  slide_num = tile_sum.slide_num
  rows = tile_sum.scaled_h
  cols = tile_sum.scaled_w
  row_tile_size = tile_sum.scaled_tile_h
  col_tile_size = tile_sum.scaled_tile_w
  num_row_tiles, num_col_tiles = get_num_tiles(rows, cols, row_tile_size, col_tile_size)
  summary = create_summary_pil_img(np_img, z, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles)
  draw = ImageDraw.Draw(summary)

  original_img_path = slide.get_training_image_path(slide_num)
  np_orig = slide.open_image_np(original_img_path)
  summary_orig = create_summary_pil_img(np_orig, z, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles)
  draw_orig = ImageDraw.Draw(summary_orig)

  for t in tile_sum.tiles:
    border_color = tile_border_color(t.tissue_percentage)
    tile_border(draw, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color)
    tile_border(draw_orig, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color)

  summary_txt = summary_title(tile_sum) + "\n" + summary_stats(tile_sum)

  summary_font = ImageFont.truetype(SUMMARY_TITLE_FONT_PATH, size=SUMMARY_TITLE_TEXT_SIZE,encoding="unic")

  
  draw.text((5, 5), summary_txt, SUMMARY_TITLE_TEXT_COLOR, font=summary_font)
  draw_orig.text((5, 5), summary_txt, SUMMARY_TITLE_TEXT_COLOR, font=summary_font)

  if DISPLAY_TILE_SUMMARY_LABELS:
    count = 0
    for t in tile_sum.tiles:
      count += 1
      label = "R%d\nC%d" % (t.r, t.c)
      font = ImageFont.truetype(FONT_PATH, size=TILE_LABEL_TEXT_SIZE)
      # drop shadow behind text
      draw.text(((t.c_s + 3), (t.r_s + 3 + z)), label, (0, 0, 0), font=font)
      draw_orig.text(((t.c_s + 3), (t.r_s + 3 + z)), label, (0, 0, 0), font=font)

      draw.text(((t.c_s + 2), (t.r_s + 2 + z)), label, SUMMARY_TILE_TEXT_COLOR, font=font)
      draw_orig.text(((t.c_s + 2), (t.r_s + 2 + z)), label, SUMMARY_TILE_TEXT_COLOR, font=font)

  if display:
    summary.show()
    summary_orig.show()
  if save_summary:
    save_tile_summary_image(summary, slide_num)
    save_tile_summary_on_original_image(summary_orig, slide_num)


def generate_top_tile_summaries(tile_sum, np_img, display=True, save_summary=False, show_top_stats=True,
                                label_all_tiles=LABEL_ALL_TILES_IN_TOP_TILE_SUMMARY,
                                border_all_tiles=BORDER_ALL_TILES_IN_TOP_TILE_SUMMARY):
  """
  Generate summary images/thumbnails showing the top tiles ranked by score.

  Args:
    tile_sum: TileSummary object.
    np_img: Image as a NumPy array.
    display: If True, display top tiles to screen.
    save_summary: If True, save top tiles images.
    show_top_stats: If True, append top tile score stats to image.
    label_all_tiles: If True, label all tiles. If False, label only top tiles.
  """
  z = 300  # height of area at top of summary slide
  slide_num = tile_sum.slide_num
  rows = tile_sum.scaled_h
  cols = tile_sum.scaled_w
  row_tile_size = tile_sum.scaled_tile_h
  col_tile_size = tile_sum.scaled_tile_w
  num_row_tiles, num_col_tiles = get_num_tiles(rows, cols, row_tile_size, col_tile_size)
  summary = create_summary_pil_img(np_img, z, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles)
  draw = ImageDraw.Draw(summary)

  original_img_path = slide.get_training_image_path(slide_num)
  np_orig = slide.open_image_np(original_img_path)
  summary_orig = create_summary_pil_img(np_orig, z, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles)
  draw_orig = ImageDraw.Draw(summary_orig)

  if border_all_tiles:
    for t in tile_sum.tiles:
      border_color = faded_tile_border_color(t.tissue_percentage)
      tile_border(draw, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color, border_size=1)
      tile_border(draw_orig, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color, border_size=1)

  tbs = TILE_BORDER_SIZE
  top_tiles = tile_sum.top_tiles()
  for t in top_tiles:
    border_color = tile_border_color(t.tissue_percentage)
    tile_border(draw, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color)
    tile_border(draw_orig, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color)
    if border_all_tiles:
      tile_border(draw, t.r_s + z + tbs, t.r_e + z - tbs, t.c_s + tbs, t.c_e - tbs, (0, 0, 0))
      tile_border(draw_orig, t.r_s + z + tbs, t.r_e + z - tbs, t.c_s + tbs, t.c_e - tbs, (0, 0, 0))

  summary_title = "Slide %03d Top Tile Summary:" % slide_num
  summary_txt = summary_title + "\n" + summary_stats(tile_sum)

  summary_font = ImageFont.truetype(SUMMARY_TITLE_FONT_PATH, size=SUMMARY_TITLE_TEXT_SIZE)
  draw.text((5, 5), summary_txt, SUMMARY_TITLE_TEXT_COLOR, font=summary_font)
  draw_orig.text((5, 5), summary_txt, SUMMARY_TITLE_TEXT_COLOR, font=summary_font)

  tiles_to_label = tile_sum.tiles if label_all_tiles else top_tiles
  h_offset = TILE_BORDER_SIZE + 2
  v_offset = TILE_BORDER_SIZE
  h_ds_offset = TILE_BORDER_SIZE + 3
  v_ds_offset = TILE_BORDER_SIZE + 1
  for t in tiles_to_label:
    label = "R%d\nC%d" % (t.r, t.c)
    font = ImageFont.truetype(FONT_PATH, size=TILE_LABEL_TEXT_SIZE)
    # drop shadow behind text
    draw.text(((t.c_s + h_ds_offset), (t.r_s + v_ds_offset + z)), label, (0, 0, 0), font=font)
    draw_orig.text(((t.c_s + h_ds_offset), (t.r_s + v_ds_offset + z)), label, (0, 0, 0), font=font)

    draw.text(((t.c_s + h_offset), (t.r_s + v_offset + z)), label, SUMMARY_TILE_TEXT_COLOR, font=font)
    draw_orig.text(((t.c_s + h_offset), (t.r_s + v_offset + z)), label, SUMMARY_TILE_TEXT_COLOR, font=font)

  if show_top_stats:
    summary = add_tile_stats_to_top_tile_summary(summary, top_tiles, z)
    summary_orig = add_tile_stats_to_top_tile_summary(summary_orig, top_tiles, z)

  if display:
    summary.show()
    summary_orig.show()
  if save_summary:
    save_top_tiles_image(summary, slide_num)
    save_top_tiles_on_original_image(summary_orig, slide_num)


def add_tile_stats_to_top_tile_summary(pil_img, tiles, z):
  np_sum = util.pil_to_np_rgb(pil_img)
  sum_r, sum_c, sum_ch = np_sum.shape
  np_stats = np_tile_stat_img(tiles)
  st_r, st_c, _ = np_stats.shape
  combo_c = sum_c + st_c
  combo_r = max(sum_r, st_r + z)
  combo = np.zeros([combo_r, combo_c, sum_ch], dtype=np.uint8)
  combo.fill(255)
  combo[0:sum_r, 0:sum_c] = np_sum
  combo[z:st_r + z, sum_c:sum_c + st_c] = np_stats
  result = util.np_to_pil(combo)
  return result


def np_tile_stat_img(tiles):
  """
  Generate tile scoring statistics for a list of tiles and return the result as a NumPy array image.

  Args:
    tiles: List of tiles (such as top tiles)

  Returns:
    Tile scoring statistics converted into an NumPy array image.
  """
  tt = sorted(tiles, key=lambda t: (t.r, t.c), reverse=False)
  tile_stats = "Tile Score Statistics:\n"
  count = 0
  for t in tt:
    if count > 0:
      tile_stats += "\n"
    count += 1
    tup = (t.r, t.c, t.rank, t.tissue_percentage, t.color_factor, t.s_and_v_factor, t.quantity_factor, t.score)
    tile_stats += "R%03d C%03d #%003d TP:%6.2f%% CF:%4.0f SVF:%4.2f QF:%4.2f S:%0.4f" % tup
  pil_stats = pil_text(tile_stats, font_path=SUMMARY_TITLE_FONT_PATH, font_size=14)
  np_stats = util.pil_to_np_rgb(pil_stats)
  return np_stats


def tile_border_color(tissue_percentage):
  """
  Obtain the corresponding tile border color for a particular tile tissue percentage.

  Args:
    tissue_percentage: The tile tissue percentage

  Returns:
    The tile border color corresponding to the tile tissue percentage.
  """
  if tissue_percentage >= TISSUE_HIGH_THRESH:
    border_color = HIGH_COLOR
  elif (tissue_percentage >= TISSUE_LOW_THRESH) and (tissue_percentage < TISSUE_HIGH_THRESH):
    border_color = MEDIUM_COLOR
  elif (tissue_percentage > 0) and (tissue_percentage < TISSUE_LOW_THRESH):
    border_color = LOW_COLOR
  else:
    border_color = NONE_COLOR
  return border_color


def faded_tile_border_color(tissue_percentage):
  """
  Obtain the corresponding faded tile border color for a particular tile tissue percentage.

  Args:
    tissue_percentage: The tile tissue percentage

  Returns:
    The faded tile border color corresponding to the tile tissue percentage.
  """
  if tissue_percentage >= TISSUE_HIGH_THRESH:
    border_color = FADED_THRESH_COLOR
  elif (tissue_percentage >= TISSUE_LOW_THRESH) and (tissue_percentage < TISSUE_HIGH_THRESH):
    border_color = FADED_MEDIUM_COLOR
  elif (tissue_percentage > 0) and (tissue_percentage < TISSUE_LOW_THRESH):
    border_color = FADED_LOW_COLOR
  else:
    border_color = FADED_NONE_COLOR
  return border_color


def summary_title(tile_summary):
  """
  Obtain tile summary title.

  Args:
    tile_summary: TileSummary object.

  Returns:
     The tile summary title.
  """
  return "Slide %03d Tile Summary:" % tile_summary.slide_num


def summary_stats(tile_summary):
  """
  Obtain various stats about the slide tiles.

  Args:
    tile_summary: TileSummary object.

  Returns:
     Various stats about the slide tiles as a string.
  """
  return "Original Dimensions: %dx%d\n" % (tile_summary.orig_w, tile_summary.orig_h) + \
         "Original Tile Size: %dx%d\n" % (tile_summary.orig_tile_w, tile_summary.orig_tile_h) + \
         "Scale Factor: 1/%dx\n" % tile_summary.scale_factor + \
         "Scaled Dimensions: %dx%d\n" % (tile_summary.scaled_w, tile_summary.scaled_h) + \
         "Scaled Tile Size: %dx%d\n" % (tile_summary.scaled_tile_w, tile_summary.scaled_tile_w) + \
         "Total Mask: %3.2f%%, Total Tissue: %3.2f%%\n" % (
           tile_summary.mask_percentage(), tile_summary.tissue_percentage) + \
         "Tiles: %dx%d = %d\n" % (tile_summary.num_col_tiles, tile_summary.num_row_tiles, tile_summary.count) + \
         " %5d (%5.2f%%) tiles >=%d%% tissue\n" % (
           tile_summary.high, tile_summary.high / tile_summary.count * 100, TISSUE_HIGH_THRESH) + \
         " %5d (%5.2f%%) tiles >=%d%% and <%d%% tissue\n" % (
           tile_summary.medium, tile_summary.medium / tile_summary.count * 100, TISSUE_LOW_THRESH,
           TISSUE_HIGH_THRESH) + \
         " %5d (%5.2f%%) tiles >0%% and <%d%% tissue\n" % (
           tile_summary.low, tile_summary.low / tile_summary.count * 100, TISSUE_LOW_THRESH) + \
         " %5d (%5.2f%%) tiles =0%% tissue" % (tile_summary.none, tile_summary.none / tile_summary.count * 100)


def tile_border(draw, r_s, r_e, c_s, c_e, color, border_size=TILE_BORDER_SIZE):
  """
  Draw a border around a tile with width TILE_BORDER_SIZE.

  Args:
    draw: Draw object for drawing on PIL image.
    r_s: Row starting pixel.
    r_e: Row ending pixel.
    c_s: Column starting pixel.
    c_e: Column ending pixel.
    color: Color of the border.
    border_size: Width of tile border in pixels.
  """
  for x in range(0, border_size):
    draw.rectangle([(c_s + x, r_s + x), (c_e - 1 - x, r_e - 1 - x)], outline=color)


def save_tile_summary_image(pil_img, slide_num):
  """
  Save a tile summary image and thumbnail to the file system.

  Args:
    pil_img: Image as a PIL Image.
    slide_num: The slide number.
  """
  t = Time()
  filepath = slide.get_tile_summary_image_path(slide_num)
  pil_img.save(filepath)
  print("%-20s | Time: %-14s  Name: %s" % ("Save Tile Sum", str(t.elapsed()), filepath))

  t = Time()
  thumbnail_filepath = slide.get_tile_summary_thumbnail_path(slide_num)
  slide.save_thumbnail(pil_img, slide.THUMBNAIL_SIZE, thumbnail_filepath)
  print("%-20s | Time: %-14s  Name: %s" % ("Save Tile Sum Thumb", str(t.elapsed()), thumbnail_filepath))


def save_top_tiles_image(pil_img, slide_num):
  """
  Save a top tiles image and thumbnail to the file system.

  Args:
    pil_img: Image as a PIL Image.
    slide_num: The slide number.
  """
  t = Time()
  filepath = slide.get_top_tiles_image_path(slide_num)
  pil_img.save(filepath)
  print("%-20s | Time: %-14s  Name: %s" % ("Save Top Tiles Image", str(t.elapsed()), filepath))

  t = Time()
  thumbnail_filepath = slide.get_top_tiles_thumbnail_path(slide_num)
  slide.save_thumbnail(pil_img, slide.THUMBNAIL_SIZE, thumbnail_filepath)
  print("%-20s | Time: %-14s  Name: %s" % ("Save Top Tiles Thumb", str(t.elapsed()), thumbnail_filepath))


def save_tile_summary_on_original_image(pil_img, slide_num):
  """
  Save a tile summary on original image and thumbnail to the file system.

  Args:
    pil_img: Image as a PIL Image.
    slide_num: The slide number.
  """
  t = Time()
  filepath = slide.get_tile_summary_on_original_image_path(slide_num)
  pil_img.save(filepath)
  print("%-20s | Time: %-14s  Name: %s" % ("Save Tile Sum Orig", str(t.elapsed()), filepath))

  t = Time()
  thumbnail_filepath = slide.get_tile_summary_on_original_thumbnail_path(slide_num)
  slide.save_thumbnail(pil_img, slide.THUMBNAIL_SIZE, thumbnail_filepath)
  print(
    "%-20s | Time: %-14s  Name: %s" % ("Save Tile Sum Orig T", str(t.elapsed()), thumbnail_filepath))


def save_top_tiles_on_original_image(pil_img, slide_num):
  """
  Save a top tiles on original image and thumbnail to the file system.

  Args:
    pil_img: Image as a PIL Image.
    slide_num: The slide number.
  """
  t = Time()
  filepath = slide.get_top_tiles_on_original_image_path(slide_num)
  pil_img.save(filepath)
  print("%-20s | Time: %-14s  Name: %s" % ("Save Top Orig", str(t.elapsed()), filepath))

  t = Time()
  thumbnail_filepath = slide.get_top_tiles_on_original_thumbnail_path(slide_num)
  slide.save_thumbnail(pil_img, slide.THUMBNAIL_SIZE, thumbnail_filepath)
  print(
    "%-20s | Time: %-14s  Name: %s" % ("Save Top Orig Thumb", str(t.elapsed()), thumbnail_filepath))


def summary_and_tiles(slide_num, display=True, save_summary=False, save_data=True, save_top_tiles=True):
  """
  Generate tile summary and top tiles for slide.

  Args:
    slide_num: The slide number.
    display: If True, display tile summary to screen.
    save_summary: If True, save tile summary images.
    save_data: If True, save tile data to csv file.
    save_top_tiles: If True, save top tiles to files.

  """
    
  ################# SAVES TOP TILES PNG ####################
  ##########################################################
  ##########################################################
  ##########################################################
  
  save_top_tiles = 	True
  
  ##########################################################
  ##########################################################
  ##########################################################
  ##########################################################
  
  
  img_path = slide.get_filter_image_result(slide_num)
  np_img = slide.open_image_np(img_path)

  tile_sum = score_tiles(slide_num, np_img)
  if save_data:
    save_tile_data(tile_sum)
  generate_tile_summaries(tile_sum, np_img, display=display, save_summary=save_summary)
  generate_top_tile_summaries(tile_sum, np_img, display=display, save_summary=save_summary)
  if save_top_tiles:
    for tile in tile_sum.top_tiles():
      tile.save_tile()
  return tile_sum


def save_tile_data(tile_summary):
  """
  Save tile data to csv file.

  Args
    tile_summary: TimeSummary object.
  """

  time = Time()

  csv = summary_title(tile_summary) + "\n" + summary_stats(tile_summary)

  csv += "\n\n\nTile Num,Row,Column,Tissue %,Tissue Quantity,Col Start,Row Start,Col End,Row End,Col Size,Row Size," + \
         "Original Col Start,Original Row Start,Original Col End,Original Row End,Original Col Size,Original Row Size," + \
         "Color Factor,S and V Factor,Quantity Factor,Score\n"

  for t in tile_summary.tiles:
    line = "%d,%d,%d,%4.2f,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%4.0f,%4.2f,%4.2f,%0.4f\n" % (
      t.tile_num, t.r, t.c, t.tissue_percentage, t.tissue_quantity().name, t.c_s, t.r_s, t.c_e, t.r_e, t.c_e - t.c_s,
      t.r_e - t.r_s, t.o_c_s, t.o_r_s, t.o_c_e, t.o_r_e, t.o_c_e - t.o_c_s, t.o_r_e - t.o_r_s, t.color_factor,
      t.s_and_v_factor, t.quantity_factor, t.score)
    csv += line

  data_path = slide.get_tile_data_path(tile_summary.slide_num)
  csv_file = open(data_path, "w")
  csv_file.write(csv)
  csv_file.close()

  print("%-20s | Time: %-14s  Name: %s" % ("Save Tile Data", str(time.elapsed()), data_path))


def tile_to_pil_tile(tile):
  """
  Convert tile information into the corresponding tile as a PIL image read from the whole-slide image file.

  Args:
    tile: Tile object.

  Return:
    Tile as a PIL image.
  """
  t = tile
  slide_filepath = slide.get_training_slide_path(t.slide_num)
  s = slide.open_slide(slide_filepath)

  x, y = t.o_c_s, t.o_r_s
  w, h = t.o_c_e - t.o_c_s, t.o_r_e - t.o_r_s
  tile_region = s.read_region((x, y), 0, (w, h))
  # RGBA to RGB
  pil_img = tile_region.convert("RGB")
  return pil_img


def save_display_tile(tile, save=True, display=False):
  """
  Save and/or display a tile image.

  Args:
    tile: Tile object.
    save: If True, save tile image.
    display: If True, dispaly tile image.
  """
  tile_pil_img = tile_to_pil_tile(tile)

  if save:
    t = Time()
    img_path = slide.get_tile_image_path(tile)
    dir = os.path.dirname(img_path)
    if not os.path.exists(dir):
      os.makedirs(dir)
    tile_pil_img.save(img_path)
    print("%-20s | Time: %-14s  Name: %s" % ("Save Tile", str(t.elapsed()), img_path))

  if display:
    tile_pil_img.show()


def score_tiles(slide_num, np_img=None, dimensions=None, small_tile_in_tile=False):
  """
  Score all tiles for a slide and return the results in a TileSummary object.

  Args:
    slide_num: The slide number.
    np_img: Optional image as a NumPy array.
    dimensions: Optional tuple consisting of (original width, original height, new width, new height). Used for dynamic
      tile retrieval.
    small_tile_in_tile: If True, include the small NumPy image in the Tile objects.

  Returns:
    TileSummary object which includes a list of Tile objects containing information about each tile.
  """
  if dimensions is None:
    img_path = slide.get_filter_image_result(slide_num)
    o_w, o_h, w, h = slide.parse_dimensions_from_image_filename(img_path)
  else:
    o_w, o_h, w, h = dimensions

  if np_img is None:
    np_img = slide.open_image_np(img_path)

  ###################################################################
  ###################################################################
  ###################################################################  

  row_tile_size = round(ROW_TILE_SIZE / slide.SCALE_FACTOR)  # use round?
  col_tile_size = round(COL_TILE_SIZE / slide.SCALE_FACTOR)  # use round?

  # row_tile_size = int(ROW_TILE_SIZE / slide.SCALE_FACTOR)  # use round?
  # col_tile_size = int(COL_TILE_SIZE / slide.SCALE_FACTOR)  # use round?

  ####################################################################
  ####################################################################
  ####################################################################
  


  num_row_tiles, num_col_tiles = get_num_tiles(h, w, row_tile_size, col_tile_size)

  tile_sum = TileSummary(slide_num=slide_num,
                         orig_w=o_w,
                         orig_h=o_h,
                         orig_tile_w=COL_TILE_SIZE,
                         orig_tile_h=ROW_TILE_SIZE,
                         scaled_w=w,
                         scaled_h=h,
                         scaled_tile_w=col_tile_size,
                         scaled_tile_h=row_tile_size,
                         tissue_percentage=filter.tissue_percent(np_img),
                         num_col_tiles=num_col_tiles,
                         num_row_tiles=num_row_tiles)

  count = 0
  high = 0
  medium = 0
  low = 0
  none = 0
  tile_indices = get_tile_indices(h, w, row_tile_size, col_tile_size)
  for t in tile_indices:
    count += 1  # tile_num
    r_s, r_e, c_s, c_e, r, c = t
    np_tile = np_img[r_s:r_e, c_s:c_e]
    t_p = filter.tissue_percent(np_tile)
    amount = tissue_quantity(t_p)
    if amount == TissueQuantity.HIGH:
      high += 1
    elif amount == TissueQuantity.MEDIUM:
      medium += 1
    elif amount == TissueQuantity.LOW:
      low += 1
    elif amount == TissueQuantity.NONE:
      none += 1
    o_c_s, o_r_s = slide.small_to_large_mapping((c_s, r_s), (o_w, o_h))
    o_c_e, o_r_e = slide.small_to_large_mapping((c_e, r_e), (o_w, o_h))

    # pixel adjustment in case tile dimension too large (for example, 1025 instead of 1024)
    if (o_c_e - o_c_s) > COL_TILE_SIZE:
      o_c_e -= 1
    if (o_r_e - o_r_s) > ROW_TILE_SIZE:
      o_r_e -= 1

    score, color_factor, s_and_v_factor, quantity_factor = score_tile(np_tile, t_p, slide_num, r, c)

    np_scaled_tile = np_tile if small_tile_in_tile else None
    tile = Tile(tile_sum, slide_num, np_scaled_tile, count, r, c, r_s, r_e, c_s, c_e, o_r_s, o_r_e, o_c_s,
                o_c_e, t_p, color_factor, s_and_v_factor, quantity_factor, score)
    tile_sum.tiles.append(tile)

  tile_sum.count = count
  tile_sum.high = high
  tile_sum.medium = medium
  tile_sum.low = low
  tile_sum.none = none

  tiles_by_score = tile_sum.tiles_by_score()
  rank = 0
  for t in tiles_by_score:
    rank += 1
    t.rank = rank

  return tile_sum


def score_tile(np_tile, tissue_percent, slide_num, row, col):
  """
  Score tile based on tissue percentage, color factor, saturation/value factor, and tissue quantity factor.

  Args:
    np_tile: Tile as NumPy array.
    tissue_percent: The percentage of the tile judged to be tissue.
    slide_num: Slide number.
    row: Tile row.
    col: Tile column.

  Returns tuple consisting of score, color factor, saturation/value factor, and tissue quantity factor.
  """
  color_factor = hsv_purple_pink_factor(np_tile)
  s_and_v_factor = hsv_saturation_and_value_factor(np_tile)
  amount = tissue_quantity(tissue_percent)
  quantity_factor = tissue_quantity_factor(amount)
  
  combined_factor = color_factor * s_and_v_factor * quantity_factor
  
  
  ##################### TILE SCORE EQUATION ############################
  ######################################################################

  score = (tissue_percent ** 2) * np.log(1 + combined_factor) / 1000.0

  ######################################################################
  ######################################################################

  # scale score to between 0 and 1

  score = 1.0 - (10.0 / (10.0 + score))
  return score, color_factor, s_and_v_factor, quantity_factor


def tissue_quantity_factor(amount):
  """
  Obtain a scoring factor based on the quantity of tissue in a tile.

  Args:
    amount: Tissue amount as a TissueQuantity enum value.

  Returns:
    Scoring factor based on the tile tissue quantity.
  """
  if amount == TissueQuantity.HIGH:
    quantity_factor = 1.0
  elif amount == TissueQuantity.MEDIUM:
    quantity_factor = 0.2
  elif amount == TissueQuantity.LOW:
    quantity_factor = 0.1
  else:
    quantity_factor = 0.0
  return quantity_factor


def tissue_quantity(tissue_percentage):
  """
  Obtain TissueQuantity enum member (HIGH, MEDIUM, LOW, or NONE) for corresponding tissue percentage.

  Args:
    tissue_percentage: The tile tissue percentage.

  Returns:
    TissueQuantity enum member (HIGH, MEDIUM, LOW, or NONE).
  """
  if tissue_percentage >= TISSUE_HIGH_THRESH:
    return TissueQuantity.HIGH
  elif (tissue_percentage >= TISSUE_LOW_THRESH) and (tissue_percentage < TISSUE_HIGH_THRESH):
    return TissueQuantity.MEDIUM
  elif (tissue_percentage > 0) and (tissue_percentage < TISSUE_LOW_THRESH):
    return TissueQuantity.LOW
  else:
    return TissueQuantity.NONE


def image_list_to_tiles(image_num_list, display=False, save_summary=True, save_data=True, save_top_tiles=True):
  """
  Generate tile summaries and tiles for a list of images.

  Args:
    image_num_list: List of image numbers.
    display: If True, display tile summary images to screen.
    save_summary: If True, save tile summary images.
    save_data: If True, save tile data to csv file.
    save_top_tiles: If True, save top tiles to files.
  """
  tile_summaries_dict = dict()
  for slide_num in image_num_list:
    tile_summary = summary_and_tiles(slide_num, display, save_summary, save_data, save_top_tiles)
    tile_summaries_dict[slide_num] = tile_summary
  return image_num_list, tile_summaries_dict


def image_range_to_tiles(start_ind, end_ind, display=False, save_summary=True, save_data=True, save_top_tiles=False):
  """
  Generate tile summaries and tiles for a range of images.

  Args:
    start_ind: Starting index (inclusive).
    end_ind: Ending index (inclusive).
    display: If True, display tile summary images to screen.
    save_summary: If True, save tile summary images.
    save_data: If True, save tile data to csv file.
    save_top_tiles: If True, save top tiles to files.
  """
  image_num_list = list()
  tile_summaries_dict = dict()
  for slide_num in range(start_ind, end_ind + 1):
    tile_summary = summary_and_tiles(slide_num, display, save_summary, save_data, save_top_tiles)
    image_num_list.append(slide_num)
    tile_summaries_dict[slide_num] = tile_summary
  return image_num_list, tile_summaries_dict


def multiprocess_filtered_images_to_tiles(display=False, save_summary=False, save_data=False, save_top_tiles=True,
                                          html=False, image_num_list=None):
  """
  Generate tile summaries and tiles for all training images using multiple processes (one process per core).

  Args:
    display: If True, display images to screen (multiprocessed display not recommended).
    save_summary: If True, save tile summary images.
    save_data: If True, save tile data to csv file.
    save_top_tiles: If True, save top tiles to files.
    html: If True, generate HTML page to display tiled images.
    image_num_list: Optionally specify a list of image slide numbers.
  """
  timer = Time()
  print("Generating tile summaries (multiprocess)\n")

  if save_summary and not os.path.exists(slide.TILE_SUMMARY_DIR):
    os.makedirs(slide.TILE_SUMMARY_DIR)

  # how many processes to use
  num_processes = multiprocessing.cpu_count()
  pool = multiprocessing.Pool(num_processes)

  if image_num_list is not None:
    num_train_images = len(image_num_list)
  else:
    num_train_images = slide.get_num_training_slides()
  if num_processes > num_train_images:
    num_processes = num_train_images
  images_per_process = num_train_images / num_processes

  print("Number of processes: " + str(num_processes))
  print("Number of training images: " + str(num_train_images))

  tasks = []
  for num_process in range(1, num_processes + 1):
    start_index = (num_process - 1) * images_per_process + 1
    end_index = num_process * images_per_process
    start_index = int(start_index)
    end_index = int(end_index)
    if image_num_list is not None:
      sublist = image_num_list[start_index - 1:end_index]
      tasks.append((sublist, display, save_summary, save_data, save_top_tiles))
      print("Task #" + str(num_process) + ": Process slides " + str(sublist))
    else:
      tasks.append((start_index, end_index, display, save_summary, save_data, save_top_tiles))
      if start_index == end_index:
        print("Task #" + str(num_process) + ": Process slide " + str(start_index))
      else:
        print("Task #" + str(num_process) + ": Process slides " + str(start_index) + " to " + str(end_index))

  # start tasks
  results = []
  for t in tasks:
    if image_num_list is not None:
      results.append(pool.apply_async(image_list_to_tiles, t))
    else:
      results.append(pool.apply_async(image_range_to_tiles, t))

  slide_nums = list()
  tile_summaries_dict = dict()
  for result in results:
    image_nums, tile_summaries = result.get()
    slide_nums.extend(image_nums)
    tile_summaries_dict.update(tile_summaries)
    print("Done tiling slides: %s" % image_nums)

  print("Time to generate tile previews (multiprocess): %s\n" % str(timer.elapsed()))

  pool.close()
  pool.join()

def pil_text(text, w_border=TILE_TEXT_W_BORDER, h_border=TILE_TEXT_H_BORDER, font_path=FONT_PATH,
              font_size=TILE_TEXT_SIZE, text_color=TILE_TEXT_COLOR, background=TILE_TEXT_BACKGROUND_COLOR):
    
  """
  Obtain a PIL image representation of text.

  Args:
    text: The text to convert to an image.
    w_border: Tile text width border (left and right).
    h_border: Tile text height border (top and bottom).
    font_path: Path to font.
    font_size: Size of font.
    text_color: Tile text color.
    background: Tile background color.

  Returns:
    PIL image representing the text.
  """

  # font = ImageFont.load_default()  
  # x, y = ImageDraw.Draw(Image.new("RGB", (1, 1), background)).textsize(text, font)
  # image = Image.new("RGB", (x + 2 * w_border, y + 2 * h_border), background)
  # draw = ImageDraw.Draw(image)
  # draw.text((w_border, h_border), text, text_color, font=font)
  # return image


  # Load the custom font
  font = ImageFont.truetype(font_path, font_size)
  
  # Create a temporary image to measure the text size
  temp_image = Image.new("RGB", (1, 1), background)
  draw = ImageDraw.Draw(temp_image)
  bbox = draw.textbbox((0, 0), text, font=font)
  x, y = bbox[2] - bbox[0], bbox[3] - bbox[1]
  
  # Create the actual image with the calculated size and additional border
  image = Image.new("RGB", (x + 2 * w_border, y + 2 * h_border), background)
  draw = ImageDraw.Draw(image)
  
  # Draw the text on the image at the specified position
  draw.text((w_border, h_border), text, text_color, font=font)
  
  return image


    
def rgb_to_hues(rgb):
  """
  Convert RGB NumPy array to 1-dimensional array of hue values (HSV H values in degrees).

  Args:
    rgb: RGB image as a NumPy array

  Returns:
    1-dimensional array of hue values in degrees
  """
  hsv = filter.filter_rgb_to_hsv(rgb, display_np_info=False)
  h = filter.filter_hsv_to_h(hsv, display_np_info=False)
  return h


def hsv_saturation_and_value_factor(rgb):
  """
  Function to reduce scores of tiles with narrow HSV saturations and values since saturation and value standard
  deviations should be relatively broad if the tile contains significant tissue.

  Example of a blurred tile that should not be ranked as a top tile:
    ../data/tiles_png/006/TUPAC-TR-006-tile-r58-c3-x2048-y58369-w1024-h1024.png

  Args:
    rgb: RGB image as a NumPy array

  Returns:
    Saturation and value factor, where 1 is no effect and less than 1 means the standard deviations of saturation and
    value are relatively small.
  """
  hsv = filter.filter_rgb_to_hsv(rgb, display_np_info=False)
  s = filter.filter_hsv_to_s(hsv)
  v = filter.filter_hsv_to_v(hsv)
  s_std = np.std(s)
  v_std = np.std(v)
  if s_std < 0.05 and v_std < 0.05:
    factor = 0.4
  elif s_std < 0.05:
    factor = 0.7
  elif v_std < 0.05:
    factor = 0.7
  else:
    factor = 1

  factor = factor ** 2
  return factor


def hsv_purple_deviation(hsv_hues):
  """
  Obtain the deviation from the HSV hue for purple.

  Args:
    hsv_hues: NumPy array of HSV hue values.

  Returns:
    The HSV purple deviation.
  """
  purple_deviation = np.sqrt(np.mean(np.abs(hsv_hues - HSV_PURPLE) ** 2))
  return purple_deviation


def hsv_pink_deviation(hsv_hues):
  """
  Obtain the deviation from the HSV hue for pink.

  Args:
    hsv_hues: NumPy array of HSV hue values.

  Returns:
    The HSV pink deviation.
  """
  pink_deviation = np.sqrt(np.mean(np.abs(hsv_hues - HSV_PINK) ** 2))
  return pink_deviation


def hsv_purple_pink_factor(rgb):
  """
  Compute scoring factor based on purple and pink HSV hue deviations and degree to which a narrowed hue color range
  average is purple versus pink.

  Args:
    rgb: Image an NumPy array.

  Returns:
    Factor that favors purple (hematoxylin stained) tissue over pink (eosin stained) tissue.
  """
  hues = rgb_to_hues(rgb)
  hues = hues[hues >= 260]  # exclude hues under 260
  hues = hues[hues <= 340]  # exclude hues over 340
  if len(hues) == 0:
    return 0  # if no hues between 260 and 340, then not purple or pink
  pu_dev = hsv_purple_deviation(hues)
  pi_dev = hsv_pink_deviation(hues)
  avg_factor = (340 - np.average(hues)) ** 2

  if pu_dev == 0:  # avoid divide by zero if tile has no tissue
    return 0

  factor = pi_dev / pu_dev * avg_factor
  return factor


class TileSummary:
  """
  Class for tile summary information.
  """

  slide_num = None
  orig_w = None
  orig_h = None
  orig_tile_w = None
  orig_tile_h = None
  scale_factor = slide.SCALE_FACTOR
  scaled_w = None
  scaled_h = None
  scaled_tile_w = None
  scaled_tile_h = None
  mask_percentage = None
  num_row_tiles = None
  num_col_tiles = None

  count = 0
  high = 0
  medium = 0
  low = 0
  none = 0

  def __init__(self, slide_num, orig_w, orig_h, orig_tile_w, orig_tile_h, scaled_w, scaled_h, scaled_tile_w,
               scaled_tile_h, tissue_percentage, num_col_tiles, num_row_tiles):
    self.slide_num = slide_num
    self.orig_w = orig_w
    self.orig_h = orig_h
    self.orig_tile_w = orig_tile_w
    self.orig_tile_h = orig_tile_h
    self.scaled_w = scaled_w
    self.scaled_h = scaled_h
    self.scaled_tile_w = scaled_tile_w
    self.scaled_tile_h = scaled_tile_h
    self.tissue_percentage = tissue_percentage
    self.num_col_tiles = num_col_tiles
    self.num_row_tiles = num_row_tiles
    self.tiles = []

  def __str__(self):
    return summary_title(self) + "\n" + summary_stats(self)

  def mask_percentage(self):
    """
    Obtain the percentage of the slide that is masked.

    Returns:
       The amount of the slide that is masked as a percentage.
    """
    return 100 - self.tissue_percentage

  def num_tiles(self):
    """
    Retrieve the total number of tiles.

    Returns:
      The total number of tiles (number of rows * number of columns).
    """
    return self.num_row_tiles * self.num_col_tiles

  def tiles_by_tissue_percentage(self):
    """
    Retrieve the tiles ranked by tissue percentage.

    Returns:
       List of the tiles ranked by tissue percentage.
    """
    sorted_list = sorted(self.tiles, key=lambda t: t.tissue_percentage, reverse=True)
    return sorted_list

  def tiles_by_score(self):
    """
    Retrieve the tiles ranked by score.

    Returns:
       List of the tiles ranked by score.
    """
    sorted_list = sorted(self.tiles, key=lambda t: t.score, reverse=True)
    return sorted_list

  def top_tiles(self):
    """
    Retrieve the top-scoring tiles.

    Returns:
       List of the top-scoring tiles.
    """
    sorted_tiles = self.tiles_by_score()
    
    # top_tiles = sorted_tiles[:NUM_TOP_TILES]
    



    ######## ADJUST TILE SCORE REQUIREMENT FOR TOP TILES #########
    ##############################################################
    ##############################################################
    ##############################################################
    ##############################################################
    ##############################################################
    ##############################################################

    
    top_tiles = [tile for tile in sorted_tiles if tile.score > 0.05]

    
    ##############################################################
    ##############################################################
    ##############################################################
    ##############################################################
    ##############################################################
    ##############################################################
    ##############################################################


    
    return top_tiles

  def get_tile(self, row, col):
    """
    Retrieve tile by row and column.

    Args:
      row: The row
      col: The column

    Returns:
      Corresponding Tile object.
    """
    tile_index = (row - 1) * self.num_col_tiles + (col - 1)
    tile = self.tiles[tile_index]
    return tile

  def display_summaries(self):
    """
    Display summary images.
    """
    summary_and_tiles(self.slide_num, display=True, save_summary=False, save_data=False, save_top_tiles=False)


class Tile:
  """
  Class for information about a tile.
  """

  def __init__(self, tile_summary, slide_num, np_scaled_tile, tile_num, r, c, r_s, r_e, c_s, c_e, o_r_s, o_r_e, o_c_s,
               o_c_e, t_p, color_factor, s_and_v_factor, quantity_factor, score):
    self.tile_summary = tile_summary
    self.slide_num = slide_num
    self.np_scaled_tile = np_scaled_tile
    self.tile_num = tile_num
    self.r = r
    self.c = c
    self.r_s = r_s
    self.r_e = r_e
    self.c_s = c_s
    self.c_e = c_e
    self.o_r_s = o_r_s
    self.o_r_e = o_r_e
    self.o_c_s = o_c_s
    self.o_c_e = o_c_e
    self.tissue_percentage = t_p
    self.color_factor = color_factor
    self.s_and_v_factor = s_and_v_factor
    self.quantity_factor = quantity_factor
    self.score = score

  def __str__(self):
    return "[Tile #%d, Row #%d, Column #%d, Tissue %4.2f%%, Score %0.4f]" % (
      self.tile_num, self.r, self.c, self.tissue_percentage, self.score)

  def __repr__(self):
    return "\n" + self.__str__()

  def mask_percentage(self):
    return 100 - self.tissue_percentage

  def tissue_quantity(self):
    return tissue_quantity(self.tissue_percentage)

  def get_pil_tile(self):
    return tile_to_pil_tile(self)

  def get_np_tile(self):
    pil_img = tile_to_pil_tile(self)
    return util.pil_to_np_rgb(pil_img)

  def save_tile(self):
    save_display_tile(self, save=True, display=False)

  def display_tile(self):
    save_display_tile(self, save=False, display=True)

  def display_with_histograms(self):
    raise NotImplementedError("Histogram display functions have been removed")

  def get_np_scaled_tile(self):
    return self.np_scaled_tile

  def get_pil_scaled_tile(self):
    return util.np_to_pil(self.np_scaled_tile)


class TissueQuantity(Enum):
  NONE = 0
  LOW = 1
  MEDIUM = 2
  HIGH = 3


if __name__ == "__main__":
   multiprocess_filtered_images_to_tiles()
