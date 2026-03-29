#!/usr/bin/env python
# coding: utf-8

import os
from os import listdir
from os.path import isfile

import pickle
import re

import cardiac_globals as cg


def make_directory(directory):
    if os.path.exists(directory):
        pass
    else:
        os.makedirs(directory)


def initialize_gpu():

    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("Training on CPU")

    num_devices = torch.cuda.device_count()
    for i in range(num_devices):
        print(f"GPU {i} = ", torch.cuda.get_device_name(i))

    return device


def get_test_slide_numbers():

    slides = [file for file in listdir(cg.TEST_SLIDE_DIR) if file.split(".")[1] == "svs"]
    slide_list = []
    # get rid of the .svs and make integer
    for i in range(len(slides)):
        item = slides[i]
        item = int(item.split(".")[0])
        slide_list.append(item)

    return slide_list


def model_prediction_dict_to_csv(slide_number):

    """ e.g. "model_predictions_dict_001_filtered.pickle" """

    import csv

    filename = "model_predictions_dict_" + str(slide_number) + "_filtered.pickle"


    with open(cg.SAVED_DATABASE_DIR + filename, 'rb') as handle:
        mydict = pickle.load(handle)

    # get rid of the ".pickle"
    filename = filename.replace(filename.split(".")[1], "")
    filename = filename.replace(".", "")

    outfile = cg.TEST_SLIDE_PREDICTIONS_DIR + filename + ".csv"

    make_directory(cg.TEST_SLIDE_PREDICTIONS_DIR)

    # delete old file is exists
    if isfile(outfile):
        print("Removing old prediction dict")
        os.remove(outfile)
    else:
        print("No old prediction dict found")

    with open(outfile, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in mydict.items():
            writer.writerow([key, value])


def slide_dx_to_csv(slide_dx_dict, filename):

    import csv

    try:
        # get rid of the ".pickle"
        filename = filename.split(".")[0] + ".csv"

        outfile = cg.SLIDE_DX_DIR + filename

        # delete old file is exists
        if isfile(outfile): os.remove(outfile)

        with open(outfile, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in slide_dx_dict.items():
                writer.writerow([key, value])

        print(f"Successfully wrote {outfile}")

    except:

        print(f"Error writing {outfile}")


def get_png_slide_path(slide_number):

    files = listdir(cg.PNG_SLIDE_DIR)

    png_slide_path = None

    for file in files:
        if str(file.split("-")[0]) == str(slide_number):
            png_slide = file
            png_slide_path = cg.PNG_SLIDE_DIR + png_slide


    return png_slide_path


def get_png_slide_name(slide_number):

    files = listdir(cg.PNG_SLIDE_DIR)

    for file in files:
        if str(file.split("-")[0]) == str(slide_number):
            png_slide_name = file

    return png_slide_name


def get_coords_from_name(name):

    # Get coordinates from filename
    m = re.match(".*-x([\d]*)-y([\d]*).*\..*", name)
    x_coord = int(m.group(1))
    y_coord = int(m.group(2))

    return x_coord, y_coord


def parse_dimensions_from_image_filename(filename):

    # Get coordinates from filename
    m = re.match(".*-([\d]*)x([\d]*)-([\d]*)x([\d]*).*\..*", filename)
    large_w = int(m.group(1))
    large_h = int(m.group(2))
    small_w = int(m.group(3))
    small_h = int(m.group(4))

    return large_w, large_h, small_w, small_h


def large_to_small_coords(large_w, large_h, small_w, small_h, large_x, large_y):

    """
    Converts from coordinates of original large image to scaled down image.
    Dont need to include SCALE_FACTOR in calculations since it cancels out in the equations
    and all you need are the ratios of the length and width to calculate
    """

    small_x = round(large_x * (small_w/large_w))
    small_y = round(large_y * (small_h/large_h))

    return small_x, small_y


def pad_image_number(number):

    if len(str(number)) < 3:
        number = str(number).zfill(3)

    return number


def get_patchname(tile_name, slide_num, x_start, y_start):

    """
    Get the X and Y coordinates of a tile from the tilename and calculate patch coordinates
    """

    # Get large_x and large_y coords from the tile we are processing
    m = re.match(".*-r([\d]*)-c([\d]*)-x([\d]*)-y([\d]*).*\..*", tile_name)

    large_x = int(m.group(3)) + x_start
    large_y = int(m.group(4)) + y_start

    patch_name = tile_name.replace(tile_name.split("-")[4], "x" + str(large_x))
    patch_name = patch_name.replace(tile_name.split("-")[5], "y" + str(large_y))

    return patch_name
