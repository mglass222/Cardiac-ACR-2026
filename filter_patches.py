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


import slide
import util
from util import Time

from PIL import Image

import cardiac_globals as cg 
import cardiac_utils as utils 



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
    print("Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Green Channel green_thresh=%d, so try %d" % \
    (mask_percentage, overmask_thresh, green_thresh, new_green_thresh))
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




def filter_grays(rgb, tolerance=15, output_type="bool"):

  """
  Create a mask to filter out pixels where the red, green, and blue channel values are similar.

  Args:
    np_img: RGB image as a NumPy array.
    tolerance: Tolerance value to determine how similar the values must be in order to be filtered out (ORIG == 15)
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



def apply_image_filters(np_img, save=True, display=False):
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

  mask_not_gray = filter_grays(rgb)

  mask_green_gray = mask_not_gray & mask_not_green

  rgb_gray_green = util.mask_rgb(rgb, mask_green_gray)

  img = rgb_gray_green

  return img



def apply_filters_to_image(image, save_dir, save, display=False):
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
  # print("Processing slide #%d" % slide_num)

  # info = dict()

  image_path = image
  image_name = image_path.split(".")[0].split("\\")[5]
  image_ext = image_path.split(".")[1]

  result_path = save_dir + "\\" + image_name + "_filtered" + "." + image_ext

  image = Image.open(image_path)
  np_orig = np.asarray(image)
  filtered_np_img = apply_image_filters(np_orig, save, display=display)

  if save:
  	pil_img = util.np_to_pil(filtered_np_img)
  	pil_img.save(result_path)

    # t1 = Time()
    # thumbnail_path = slide.get_filter_thumbnail_result(slide_num)
    # slide.save_thumbnail(pil_img, slide.THUMBNAIL_SIZE, thumbnail_path)
    # print("%-20s | Time: %-14s  Name: %s" % ("Save Thumbnail", str(t1.elapsed()), thumbnail_path))

  # print("Slide #%03d processing time: %s\n" % (slide_num, str(t.elapsed())))

  return filtered_np_img


def apply_filters_to_image_list_multiprocess(image_list, save_dir, save, display):
  """
  Apply filters to a list of images.

  Args:
    image_num_list: List of image numbers.
    save: If True, save filtered images.
    display: If True, display filtered images to screen.

  Returns:
    Tuple consisting of 1) a list of image numbers, and 2) a dictionary of image filter information.
  """
  # html_page_info = dict()

  tissue_percent_dict = {}

  for image in image_list:

    filtered_np_image = apply_filters_to_image(image, save_dir, save, display=display)

    tissue_percentage = tissue_percent(filtered_np_image)

    tissue_percent_dict.update({image:tissue_percentage})
   
  return tissue_percent_dict 



def multiprocess_apply_filters_to_images(folder, save=False, display=False, html=False, image_num_list=None):
  """
  Apply a set of filters to all training images using multiple processes (one process per core).

  Args:
    save: If True, save filtered images.
    display: If True, display filtered images to screen (multiprocessed display not recommended).
    html: If True, generate HTML page to display filtered images.
    image_num_list: Optionally specify a list of image slide numbers.
  """

  timer = Time()

  print("Applying filters to Patches (multiprocess)\n")

  split_tiles_dir = cg.SPLIT_TILE_DIR + str(folder) + "\\"

  save_dir = cg.SPLIT_TILE_DIR + str(folder) + "_filtered" + "\\"

  image_list = [split_tiles_dir + image for image in os.listdir(split_tiles_dir)]  

  num_images = len(image_list)

  tissue_percent_dict = {}

  if save and not os.path.exists(save_dir):
    os.makedirs(save_dir)

  # how many processes to use
  num_processes = multiprocessing.cpu_count()
  pool = multiprocessing.Pool(num_processes)


  if num_processes > num_images:
    num_processes = num_images

  images_per_process = num_images / num_processes

  print("Number of processes: " + str(num_processes))
  print("Number of images: " + str(num_images))


  tasks = []
  for num_process in range(1, num_processes + 1):
    start_index = (num_process - 1) * images_per_process + 1
    end_index = num_process * images_per_process
    start_index = int(start_index)
    end_index = int(end_index)

    if image_list is not None:
      sublist = image_list[start_index - 1:end_index]
      tasks.append((sublist, save_dir, save, display))
      print("Task # " + str(num_process) + " process " + str(len(sublist)) + " images")

  # start tasks
  results = []
  for t in tasks:
    if image_list is not None:
      results.append(pool.apply_async(apply_filters_to_image_list_multiprocess, t))


  for result in results:
    if image_list is not None:

      dict_update = result.get()

      tissue_percent_dict.update(dict_update)

      print("Done filtering images")

  pool.close()
  pool.join()

  return tissue_percent_dict
