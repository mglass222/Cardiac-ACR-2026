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
import skimage.morphology as sk_morphology

import slide
import util
from util import Time


def mask_percent(np_img):
  """
  Determine the percentage of a NumPy array that is masked (how many of the values are 0 values).

  Args:
    np_img: Image as a NumPy array.

  Returns:
    The percentage of the NumPy array that is masked.
  """
  if (len(np_img.shape) == 3) and (np_img.shape[2] == 3):
    np_sum = np_img[:, :, 0] + np_img[:, :, 1] + np_img[:, :, 2]
    mask_percentage = 100 - np.count_nonzero(np_sum) / np_sum.size * 100
  else:
    mask_percentage = 100 - np.count_nonzero(np_img) / np_img.size * 100
  return mask_percentage


def tissue_percent(np_img):
  """
  Determine the percentage of a NumPy array that is tissue (not masked).

  Args:
    np_img: Image as a NumPy array.

  Returns:
    The percentage of the NumPy array that is tissue.
  """
  return 100 - mask_percent(np_img)


def filter_remove_small_objects(np_img, min_size=64, avoid_overmask=True, overmask_thresh=95, output_type="uint8"):
  """
  Filter image to remove small objects (connected components) less than a particular minimum size. If avoid_overmask
  is True, this function can recursively call itself with progressively smaller minimum size objects to remove to
  reduce the amount of masking that this filter performs.

  Args:
    np_img: Image as a NumPy array of type bool.
    min_size: Minimum size of small object to remove.
    avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
    overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8).
  """
  t = Time()

  rem_sm = np_img.astype(bool)  # make sure mask is boolean
  rem_sm = sk_morphology.remove_small_objects(rem_sm, min_size=min_size)
  mask_percentage = mask_percent(rem_sm)
  if (mask_percentage >= overmask_thresh) and (min_size >= 1) and (avoid_overmask is True):
    new_min_size = min_size / 2
    print("Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Small Objs size %d, so try %d" % (
      mask_percentage, overmask_thresh, min_size, new_min_size))
    rem_sm = filter_remove_small_objects(np_img, new_min_size, avoid_overmask, overmask_thresh, output_type)
  np_img = rem_sm

  if output_type == "bool":
    pass
  elif output_type == "float":
    np_img = np_img.astype(float)
  else:
    np_img = np_img.astype("uint8") * 255

  util.np_info(np_img, "Remove Small Objs", t.elapsed())
  return np_img


def filter_green_channel(np_img, green_thresh=200, avoid_overmask=True, overmask_thresh=90, output_type="bool"):
  """
  Create a mask to filter out pixels with a green channel value greater than a particular threshold, since hematoxylin
  and eosin are purplish and pinkish, which do not have much green to them.

  Args:
    np_img: RGB image as a NumPy array.
    green_thresh: Green channel threshold value (0 to 255). If value is greater than green_thresh, mask out pixel.
    avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
    overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array representing a mask where pixels above a particular green channel threshold have been masked out.
  """
  t = Time()

  g = np_img[:, :, 1]
  gr_ch_mask = (g < green_thresh) & (g > 0)
  mask_percentage = mask_percent(gr_ch_mask)
  if (mask_percentage >= overmask_thresh) and (green_thresh < 255) and (avoid_overmask is True):
    new_green_thresh = math.ceil((255 - green_thresh) / 2 + green_thresh)
    print(
      "Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Green Channel green_thresh=%d, so try %d" % (
        mask_percentage, overmask_thresh, green_thresh, new_green_thresh))
    gr_ch_mask = filter_green_channel(np_img, new_green_thresh, avoid_overmask, overmask_thresh, output_type)
  np_img = gr_ch_mask

  if output_type == "bool":
    pass
  elif output_type == "float":
    np_img = np_img.astype(float)
  else:
    np_img = np_img.astype("uint8") * 255

  util.np_info(np_img, "Filter Green Channel", t.elapsed())
  return np_img


def filter_red(rgb, red_lower_thresh, green_upper_thresh, blue_upper_thresh, output_type="bool",
               display_np_info=False):
  """
  Create a mask to filter out reddish colors, where the mask is based on a pixel being above a
  red channel threshold value, below a green channel threshold value, and below a blue channel threshold value.

  Args:
    rgb: RGB image as a NumPy array.
    red_lower_thresh: Red channel lower threshold value.
    green_upper_thresh: Green channel upper threshold value.
    blue_upper_thresh: Blue channel upper threshold value.
    output_type: Type of array to return (bool, float, or uint8).
    display_np_info: If True, display NumPy array info and filter time.

  Returns:
    NumPy array representing the mask.
  """
  if display_np_info:
    t = Time()
  r = rgb[:, :, 0] > red_lower_thresh
  g = rgb[:, :, 1] < green_upper_thresh
  b = rgb[:, :, 2] < blue_upper_thresh
  result = ~(r & g & b)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  if display_np_info:
    util.np_info(result, "Filter Red", t.elapsed())
  return result


def filter_red_pen(rgb, output_type="bool"):
  """
  Create a mask to filter out red pen marks from a slide.

  Args:
    rgb: RGB image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array representing the mask.
  """
  t = Time()
  result = filter_red(rgb, red_lower_thresh=150, green_upper_thresh=80, blue_upper_thresh=90) & \
           filter_red(rgb, red_lower_thresh=110, green_upper_thresh=20, blue_upper_thresh=30) & \
           filter_red(rgb, red_lower_thresh=185, green_upper_thresh=65, blue_upper_thresh=105) & \
           filter_red(rgb, red_lower_thresh=195, green_upper_thresh=85, blue_upper_thresh=125) & \
           filter_red(rgb, red_lower_thresh=220, green_upper_thresh=115, blue_upper_thresh=145) & \
           filter_red(rgb, red_lower_thresh=125, green_upper_thresh=40, blue_upper_thresh=70) & \
           filter_red(rgb, red_lower_thresh=200, green_upper_thresh=120, blue_upper_thresh=150) & \
           filter_red(rgb, red_lower_thresh=100, green_upper_thresh=50, blue_upper_thresh=65) & \
           filter_red(rgb, red_lower_thresh=85, green_upper_thresh=25, blue_upper_thresh=45)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  util.np_info(result, "Filter Red Pen", t.elapsed())
  return result


def filter_green(rgb, red_upper_thresh, green_lower_thresh, blue_lower_thresh, output_type="bool",
                 display_np_info=False):
  """
  Create a mask to filter out greenish colors, where the mask is based on a pixel being below a
  red channel threshold value, above a green channel threshold value, and above a blue channel threshold value.
  Note that for the green ink, the green and blue channels tend to track together, so we use a blue channel
  lower threshold value rather than a blue channel upper threshold value.

  Args:
    rgb: RGB image as a NumPy array.
    red_upper_thresh: Red channel upper threshold value.
    green_lower_thresh: Green channel lower threshold value.
    blue_lower_thresh: Blue channel lower threshold value.
    output_type: Type of array to return (bool, float, or uint8).
    display_np_info: If True, display NumPy array info and filter time.

  Returns:
    NumPy array representing the mask.
  """
  if display_np_info:
    t = Time()
  r = rgb[:, :, 0] < red_upper_thresh
  g = rgb[:, :, 1] > green_lower_thresh
  b = rgb[:, :, 2] > blue_lower_thresh
  result = ~(r & g & b)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  if display_np_info:
    util.np_info(result, "Filter Green", t.elapsed())
  return result


def filter_green_pen(rgb, output_type="bool"):
  """
  Create a mask to filter out green pen marks from a slide.

  Args:
    rgb: RGB image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array representing the mask.
  """
  t = Time()
  result = filter_green(rgb, red_upper_thresh=150, green_lower_thresh=160, blue_lower_thresh=140) & \
           filter_green(rgb, red_upper_thresh=70, green_lower_thresh=110, blue_lower_thresh=110) & \
           filter_green(rgb, red_upper_thresh=45, green_lower_thresh=115, blue_lower_thresh=100) & \
           filter_green(rgb, red_upper_thresh=30, green_lower_thresh=75, blue_lower_thresh=60) & \
           filter_green(rgb, red_upper_thresh=195, green_lower_thresh=220, blue_lower_thresh=210) & \
           filter_green(rgb, red_upper_thresh=225, green_lower_thresh=230, blue_lower_thresh=225) & \
           filter_green(rgb, red_upper_thresh=170, green_lower_thresh=210, blue_lower_thresh=200) & \
           filter_green(rgb, red_upper_thresh=20, green_lower_thresh=30, blue_lower_thresh=20) & \
           filter_green(rgb, red_upper_thresh=50, green_lower_thresh=60, blue_lower_thresh=40) & \
           filter_green(rgb, red_upper_thresh=30, green_lower_thresh=50, blue_lower_thresh=35) & \
           filter_green(rgb, red_upper_thresh=65, green_lower_thresh=70, blue_lower_thresh=60) & \
           filter_green(rgb, red_upper_thresh=100, green_lower_thresh=110, blue_lower_thresh=105) & \
           filter_green(rgb, red_upper_thresh=165, green_lower_thresh=180, blue_lower_thresh=180) & \
           filter_green(rgb, red_upper_thresh=140, green_lower_thresh=140, blue_lower_thresh=150) & \
           filter_green(rgb, red_upper_thresh=185, green_lower_thresh=195, blue_lower_thresh=195)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  util.np_info(result, "Filter Green Pen", t.elapsed())
  return result


def filter_blue(rgb, red_upper_thresh, green_upper_thresh, blue_lower_thresh, output_type="bool",
                display_np_info=False):
  """
  Create a mask to filter out blueish colors, where the mask is based on a pixel being below a
  red channel threshold value, below a green channel threshold value, and above a blue channel threshold value.

  Args:
    rgb: RGB image as a NumPy array.
    red_upper_thresh: Red channel upper threshold value.
    green_upper_thresh: Green channel upper threshold value.
    blue_lower_thresh: Blue channel lower threshold value.
    output_type: Type of array to return (bool, float, or uint8).
    display_np_info: If True, display NumPy array info and filter time.

  Returns:
    NumPy array representing the mask.
  """
  if display_np_info:
    t = Time()
  r = rgb[:, :, 0] < red_upper_thresh
  g = rgb[:, :, 1] < green_upper_thresh
  b = rgb[:, :, 2] > blue_lower_thresh
  result = ~(r & g & b)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  if display_np_info:
    util.np_info(result, "Filter Blue", t.elapsed())
  return result


def filter_blue_pen(rgb, output_type="bool"):
  """
  Create a mask to filter out blue pen marks from a slide.

  Args:
    rgb: RGB image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array representing the mask.
  """
  t = Time()
  result = filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=120, blue_lower_thresh=190) & \
           filter_blue(rgb, red_upper_thresh=120, green_upper_thresh=170, blue_lower_thresh=200) & \
           filter_blue(rgb, red_upper_thresh=175, green_upper_thresh=210, blue_lower_thresh=230) & \
           filter_blue(rgb, red_upper_thresh=145, green_upper_thresh=180, blue_lower_thresh=210) & \
           filter_blue(rgb, red_upper_thresh=37, green_upper_thresh=95, blue_lower_thresh=160) & \
           filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=65, blue_lower_thresh=130) & \
           filter_blue(rgb, red_upper_thresh=130, green_upper_thresh=155, blue_lower_thresh=180) & \
           filter_blue(rgb, red_upper_thresh=40, green_upper_thresh=35, blue_lower_thresh=85) & \
           filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=20, blue_lower_thresh=65) & \
           filter_blue(rgb, red_upper_thresh=90, green_upper_thresh=90, blue_lower_thresh=140) & \
           filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=60, blue_lower_thresh=120) & \
           filter_blue(rgb, red_upper_thresh=110, green_upper_thresh=110, blue_lower_thresh=175)

  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  util.np_info(result, "Filter Blue Pen", t.elapsed())
  return result


def filter_black(rgb, red_upper_thresh, green_upper_thresh, blue_lower_thresh, output_type="bool", display_np_info=False):
  """
  Create a mask to filter out blackish colors, where the mask is based on a pixel being below a
  red channel threshold value, below a green channel threshold value, and below a blue channel threshold value.

  Args:
    rgb: RGB image as a NumPy array.
    red_upper_thresh: Red channel upper threshold value.
    green_upper_thresh: Green channel upper threshold value.
    blue_lower_thresh: Blue channel upper threshold value.
    output_type: Type of array to return (bool, float, or uint8).
    display_np_info: If True, display NumPy array info and filter time.

  Returns:
    NumPy array representing the mask.
  """

  if display_np_info:
    t = Time()
  r = rgb[:, :, 0] < red_upper_thresh
  g = rgb[:, :, 1] < green_upper_thresh
  b = rgb[:, :, 2] < blue_lower_thresh
  result = ~(r & g & b)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  if display_np_info:
    util.np_info(result, "Filter Black", t.elapsed())
  return result


def filter_black_pen(rgb, output_type="bool"):
  """
  Create a mask to filter out black pen marks from a slide.

  Args:
    rgb: RGB image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array representing the mask.
  """
  t = Time()

  result = filter_black(rgb, red_upper_thresh=60, green_upper_thresh=60, blue_lower_thresh=60)

  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255

  util.np_info(result, "Filter Black Pen", t.elapsed())

  return result


def filter_grays(rgb, tolerance=15, output_type="bool"):
  """
  Create a mask to filter out pixels where the red, green, and blue channel values are similar.

  Args:
    np_img: RGB image as a NumPy array.
    tolerance: Tolerance value to determine how similar the values must be in order to be filtered out
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array representing a mask where pixels with similar red, green, and blue values have been masked out.
  """
  t = Time()
  (h, w, c) = rgb.shape

  rgb = rgb.astype(int)
  rg_diff = abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance
  rb_diff = abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance
  gb_diff = abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance
  result = ~(rg_diff & rb_diff & gb_diff)

  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255

  util.np_info(result, "Filter Grays", t.elapsed())
  return result


def apply_image_filters(np_img, slide_num=None, info=None, save=False, display=False):
  """
  Apply filters to image as NumPy array and optionally save and/or display filtered images.

  Args:
    np_img: Image as NumPy array.
    slide_num: The slide number (used for saving/displaying).
    info: Dictionary of slide information (used for HTML display).
    save: If True, save image.
    display: If True, display image.

  Returns:
    Resulting filtered image as a NumPy array.
  """
  rgb = np_img

  mask_not_green = filter_green_channel(rgb)
  rgb_not_green = util.mask_rgb(rgb, mask_not_green)

  mask_not_gray = filter_grays(rgb)
  rgb_not_gray = util.mask_rgb(rgb, mask_not_gray)

  mask_no_red_pen = filter_red_pen(rgb)
  rgb_no_red_pen = util.mask_rgb(rgb, mask_no_red_pen)

  mask_no_green_pen = filter_green_pen(rgb)
  rgb_no_green_pen = util.mask_rgb(rgb, mask_no_green_pen)

  mask_no_blue_pen = filter_blue_pen(rgb)
  rgb_no_blue_pen = util.mask_rgb(rgb, mask_no_blue_pen)

  mask_no_black_pen = filter_black_pen(rgb)
  rgb_no_black_pen = util.mask_rgb(rgb, mask_no_black_pen)

  mask_gray_green_pens =  mask_not_green & mask_not_gray & mask_no_red_pen & mask_no_green_pen & mask_no_blue_pen & mask_no_black_pen
  rgb_gray_green_pens = util.mask_rgb(rgb, mask_gray_green_pens)

  mask_remove_small = filter_remove_small_objects(mask_gray_green_pens, min_size=64, output_type="bool")

  rgb_remove_small = util.mask_rgb(rgb, mask_remove_small)

  img = rgb_remove_small

  return img


def apply_filters_to_image(slide_num, save=True, display=False):
  """
  Apply a set of filters to an image and optionally save and/or display filtered images.

  Args:
    slide_num: The slide number.
    save: If True, save filtered images.
    display: If True, display filtered images to screen.

  Returns:
    Tuple consisting of 1) the resulting filtered image as a NumPy array, and 2) dictionary of image information
    (used for HTML page generation).
  """
  t = Time()
  print("Processing slide #%d" % slide_num)

  info = dict()

  if save and not os.path.exists(slide.FILTER_DIR):
    os.makedirs(slide.FILTER_DIR)
  img_path = slide.get_training_image_path(slide_num)
  np_orig = slide.open_image_np(img_path)
  filtered_np_img = apply_image_filters(np_orig, slide_num, info, save=save, display=display)

  if save:
    t1 = Time()
    result_path = slide.get_filter_image_result(slide_num)
    pil_img = util.np_to_pil(filtered_np_img)
    pil_img.save(result_path)
    print("%-20s | Time: %-14s  Name: %s" % ("Save Image", str(t1.elapsed()), result_path))

    t1 = Time()
    thumbnail_path = slide.get_filter_thumbnail_result(slide_num)
    slide.save_thumbnail(pil_img, slide.THUMBNAIL_SIZE, thumbnail_path)
    print("%-20s | Time: %-14s  Name: %s" % ("Save Thumbnail", str(t1.elapsed()), thumbnail_path))

  print("Slide #%03d processing time: %s\n" % (slide_num, str(t.elapsed())))

  return filtered_np_img, info


def mask_percentage_text(mask_percentage):
  """
  Generate a formatted string representing the percentage that an image is masked.

  Args:
    mask_percentage: The mask percentage.

  Returns:
    The mask percentage formatted as a string.
  """
  return "%3.2f%%" % mask_percentage


def image_cell(slide_num, filter_num, display_text, file_text):
  """
  Generate HTML for viewing a processed image.

  Args:
    slide_num: The slide number.
    filter_num: The filter number.
    display_text: Filter display name.
    file_text: Filter name for file.

  Returns:
    HTML for a table cell for viewing a filtered image.
  """
  filt_img = slide.get_filter_image_path(slide_num, filter_num, file_text)
  filt_thumb = slide.get_filter_thumbnail_path(slide_num, filter_num, file_text)
  img_name = slide.get_filter_image_filename(slide_num, filter_num, file_text)
  return "      <td>\n" + \
         "        <a target=\"_blank\" href=\"%s\">%s<br/>\n" % (filt_img, display_text) + \
         "          <img src=\"%s\" />\n" % (filt_thumb) + \
         "        </a>\n" + \
         "      </td>\n"


def html_header(page_title):
  """
  Generate an HTML header for previewing images.

  Returns:
    HTML header for viewing images.
  """
  html = "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Strict//EN\" " + \
         "\"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd\">\n" + \
         "<html xmlns=\"http://www.w3.org/1999/xhtml\" lang=\"en\" xml:lang=\"en\">\n" + \
         "  <head>\n" + \
         "    <title>%s</title>\n" % page_title + \
         "    <style type=\"text/css\">\n" + \
         "     img { border: 2px solid black; }\n" + \
         "     td { border: 2px solid black; }\n" + \
         "    </style>\n" + \
         "  </head>\n" + \
         "  <body>\n"
  return html


def html_footer():
  """
  Generate an HTML footer for previewing images.

  Returns:
    HTML footer for viewing images.
  """
  html = "</body>\n" + \
         "</html>\n"
  return html


def generate_filter_html_result(html_page_info):
  """
  Generate HTML to view the filtered images. If slide.FILTER_PAGINATE is True, the results will be paginated.

  Args:
    html_page_info: Dictionary of image information.
  """
  if not slide.FILTER_PAGINATE:
    html = ""
    html += html_header("Filtered Images")
    html += "  <table>\n"

    row = 0
    for key in sorted(html_page_info):
      value = html_page_info[key]
      current_row = value[0]
      if current_row > row:
        html += "    <tr>\n"
        row = current_row
      html += image_cell(value[0], value[1], value[2], value[3])
      next_key = key + 1
      if next_key not in html_page_info:
        html += "    </tr>\n"

    html += "  </table>\n"
    html += html_footer()
    text_file = open(os.path.join(slide.FILTER_HTML_DIR, "filters.html"), "w")
    text_file.write(html)
    text_file.close()
  else:
    slide_nums = set()
    for key in html_page_info:
      slide_num = math.floor(key / 1000)
      slide_nums.add(slide_num)
    slide_nums = sorted(list(slide_nums))
    total_len = len(slide_nums)
    page_size = slide.FILTER_PAGINATION_SIZE
    num_pages = math.ceil(total_len / page_size)

    for page_num in range(1, num_pages + 1):
      start_index = (page_num - 1) * page_size
      end_index = (page_num * page_size) if (page_num < num_pages) else total_len
      page_slide_nums = slide_nums[start_index:end_index]

      html = ""
      html += html_header("Filtered Images, Page %d" % page_num)

      html += "  <div style=\"font-size: 20px\">"
      if page_num > 1:
        if page_num == 2:
          html += "<a href=\"filters.html\">&lt;</a> "
        else:
          html += "<a href=\"filters-%d.html\">&lt;</a> " % (page_num - 1)
      html += "Page %d" % page_num
      if page_num < num_pages:
        html += " <a href=\"filters-%d.html\">&gt;</a> " % (page_num + 1)
      html += "</div>\n"

      html += "  <table>\n"
      for slide_num in page_slide_nums:
        html += "  <tr>\n"
        filter_num = 1

        lookup_key = slide_num * 1000 + filter_num
        while lookup_key in html_page_info:
          value = html_page_info[lookup_key]
          html += image_cell(value[0], value[1], value[2], value[3])
          lookup_key += 1
        html += "  </tr>\n"

      html += "  </table>\n"

      html += html_footer()
      if page_num == 1:
        text_file = open(os.path.join(slide.FILTER_HTML_DIR, "filters.html"), "w")
      else:
        text_file = open(os.path.join(slide.FILTER_HTML_DIR, "filters-%d.html" % page_num), "w")
      text_file.write(html)
      text_file.close()


def apply_filters_to_image_list(image_num_list, save, display):
  """
  Apply filters to a list of images.

  Args:
    image_num_list: List of image numbers.
    save: If True, save filtered images.
    display: If True, display filtered images to screen.

  Returns:
    Tuple consisting of 1) a list of image numbers, and 2) a dictionary of image filter information.
  """
  html_page_info = dict()
  for slide_num in image_num_list:
    _, info = apply_filters_to_image(slide_num, save, display=display)
    html_page_info.update(info)
  return image_num_list, html_page_info


def apply_filters_to_image_range(start_ind, end_ind, save, display):
  """
  Apply filters to a range of images.

  Args:
    start_ind: Starting index (inclusive).
    end_ind: Ending index (inclusive).
    save: If True, save filtered images.
    display: If True, display filtered images to screen.

  Returns:
    Tuple consisting of 1) staring index of slides converted to images, 2) ending index of slides converted to images,
    and 3) a dictionary of image filter information.
  """
  html_page_info = dict()
  for slide_num in range(start_ind, end_ind + 1):
    _, info = apply_filters_to_image(slide_num, save=save, display=display)
    html_page_info.update(info)
  return start_ind, end_ind, html_page_info


def multiprocess_apply_filters_to_images(save=True, display=False, html=True, image_num_list=None):
  """
  Apply a set of filters to all training images using multiple processes (one process per core).

  Args:
    save: If True, save filtered images.
    display: If True, display filtered images to screen (multiprocessed display not recommended).
    html: If True, generate HTML page to display filtered images.
    image_num_list: Optionally specify a list of image slide numbers.
  """
  timer = Time()
  print("Applying filters to images (multiprocess)\n")

  if save and not os.path.exists(slide.FILTER_DIR):
    os.makedirs(slide.FILTER_DIR)

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
      tasks.append((sublist, save, display))
      print("Task #" + str(num_process) + ": Process slides " + str(sublist))
    else:
      tasks.append((start_index, end_index, save, display))
      if start_index == end_index:
        print("Task #" + str(num_process) + ": Process slide " + str(start_index))
      else:
        print("Task #" + str(num_process) + ": Process slides " + str(start_index) + " to " + str(end_index))

  # start tasks
  results = []
  for t in tasks:
    if image_num_list is not None:
      results.append(pool.apply_async(apply_filters_to_image_list, t))
    else:
      results.append(pool.apply_async(apply_filters_to_image_range, t))

  html_page_info = dict()
  for result in results:
    if image_num_list is not None:
      (image_nums, html_page_info_res) = result.get()
      html_page_info.update(html_page_info_res)
      print("Done filtering slides: %s" % image_nums)
    else:
      (start_ind, end_ind, html_page_info_res) = result.get()
      html_page_info.update(html_page_info_res)
      if (start_ind == end_ind):
        print("Done filtering slide %d" % start_ind)
      else:
        print("Done filtering slides %d through %d" % (start_ind, end_ind))

  if html:
    generate_filter_html_result(html_page_info)

  print("Time to apply filters to all images (multiprocess): %s\n" % str(timer.elapsed()))

  pool.close()
  pool.join()

if __name__ == "__main__":
  multiprocess_apply_filters_to_images()
