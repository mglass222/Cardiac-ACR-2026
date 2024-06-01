#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))  


# In[2]:


# import project modules ###

import sys  
py_path = "D:\\Cardiac_ACR\\Py_Files\\"
sys.path.insert(0, py_path)

# import openslide
import cardiac_globals as cg
import filter
import slide
import tiles
import tileset_utils
import filter_patches
import filter_tiles
import cardiac_utils as utils
import count_1r2
import count_1r2_testing
import annotate_svs
import annotate_png

import import_openslide


# Import System Modules, etc.
import os
from os import listdir
from os.path import isfile, isdir, join
import xml.etree.ElementTree as ET
import pickle
import shutil
import multiprocessing

import torch
import torchvision
from torch import optim, cuda
from torchvision import datasets, models, transforms

import numpy as np
from PIL import Image, ImageOps
import time

from datetime import datetime
import re

import cv2
import matplotlib.pyplot as plt

# size that nn is expecting
INPUT_SIZE = cg.ANNOTATION_SIZE

# how much to scale down original image (default = 40)
SCALE_FACTOR = cg.SCALE_FACTOR

# minimum probability required for model predictions to count
PREDICTION_THRESHOLD = cg.PREDICTION_THRESHOLD

# ### Main Source / Output Folders ###
BASE_DIR = cg.BASE_DIR
TEST_SLIDE_DIR = cg.TEST_SLIDE_DIR
PNG_SLIDE_DIR = cg.PNG_SLIDE_DIR
TILE_DATA_DIR = cg.TILE_DATA_DIR
TILE_DIR = cg.TILE_DIR
SPLIT_TILE_DIR = cg.SPLIT_TILE_DIR
MODEL_DIR = cg.MODEL_DIR
SAVED_DATABASE_DIR = cg.SAVED_DATABASE_DIR
SLIDE_DX_DIR = cg.SLIDE_DX_DIR
FILTERED_IMAGE_DIR = cg.FILTERED_IMAGE_DIR


### Directories to display tissue percent on tiles and patches ###
TISSUE_PERCENT_DIR = cg.TISSUE_PERCENT_DIR

## Font path for displaying tissue percentage on images ###
FONT_PATH = cg.FONT_PATH

plt.rcParams['figure.figsize'] = [20, 15]


# In[3]:


def check_filesystem():

    for path in [MODEL_DIR,SAVED_DATABASE_DIR,SLIDE_DX_DIR]:
        if not os.path.exists(path):
            os.makedirs(path)
            
    for path in [MODEL_DIR,SAVED_DATABASE_DIR,SLIDE_DX_DIR]:
        print(path)


# In[4]:


def filter_patches_multiprocess(slide_number):

    # permanently delete patches that dont contain at least 50 % tissue (mostly white space)

    tissue_percent_dict = filter_patches.multiprocess_apply_filters_to_images(slide_number, save=False)
       
    for key, value in tissue_percent_dict.items():
        if int(value) < 50:
            if isfile(key):
                os.remove(key)


# In[5]:


def classify_patches_batch(slide_number):

    """ Utilized batch method """

    t = time.time()
    batch_size = 200
    patch_dir = SPLIT_TILE_DIR + str(slide_number) + "\\"
    patches = [patch_dir + patch for patch in listdir(patch_dir)]
    
    num_patches = len(patches)
    num_batches = int(num_patches / batch_size)
    last_batch = num_patches - batch_size * num_batches

    print("Number of patches in the directory = ", num_patches)
    print("Total patches to process  = ", last_batch + batch_size * num_batches)
    print("batch_size * num_batches = ", batch_size * num_batches)
    print("last batch size = ", num_patches - batch_size * num_batches)

    model_predictions_dict = {}
    print(f"Classify {num_patches} patches")
    count = 0

    # num_batches + 1 to get the last batch. Range goes from 1 to num_batches -1
    for i in range(num_batches + 1):
        batch = []
        batch_patches = []

        # changebatch_size for last batch
        if i == num_batches and last_batch == 0:
            print("Last batch size == 0. Breaking...")
            break

        elif i == num_batches:
            batch_size = last_batch
#             print("LAST BATCH")
            print("Batch_size = ", batch_size)

        for j in range(batch_size):

            # get the patchname
            patch = patches[count]
            # save_patchname
            batch_patches.append(patch)
            image = Image.open(patch)
            # append image to batch
            batch.append(image)
            #update iterator
            count += 1

#         if (i % 10 == 0):
#             print(f"Processed {i} out of {num_batches} batches")

        # Classify images in the batch
        preds = Model_Predict_batch(batch, model)
        # Get class probabilities with softmax
        preds = torch.nn.functional.softmax(preds, dim = 1)
        # send from GPU to CPU for the next step
        preds = preds.cpu()
        # current batch
        preds = preds.detach().numpy()
        # update the model_prediction_dict. Update with preds (current batch predictions)
        for ii in range(len(batch_patches)):
            model_predictions_dict.update({batch_patches[ii]: preds[ii]})

    # Save model_prediction_dict
    save_name = "model_predictions_dict_" + str(slide_number) + ".pickle"

    with open(SAVED_DATABASE_DIR + save_name, 'wb') as handle:
        pickle.dump(model_predictions_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Done with predictions. this took {time.time() - t} seconds")


# In[6]:


def Model_Predict_batch(batch, model):

    from torchvision import transforms

    # Imagenet Standards
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # must also normalize the test data
    data_transforms = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)])

    batch_t = []
    for i in range(len(batch)):
        img = batch[i]
        img = data_transforms(img)
        batch_t.append(img)

    batch_tensor = torch.stack(batch_t)
    model.batch_size = len(batch_tensor)
    batch_tensor = batch_tensor.to(device)
    
    #disable autograd engine
    with torch.no_grad():
        predictions = model(batch_tensor)
        
    return predictions


# In[7]:


def threshold_predictions(slide_number):

    filename = "model_predictions_dict_" + str(slide_number) + ".pickle"

    with open(SAVED_DATABASE_DIR + filename, 'rb') as handle:
        model_predictions_dict = pickle.load(handle)

    filtered_dict = {}
    probabilities_dict = {}

    for key,value in model_predictions_dict.items():
        # get rid of predictions below threshold
        pred_mask = np.asarray(value) > PREDICTION_THRESHOLD
        if True in pred_mask:
            # get the class prediction 
            # class_prediction = np.argmax(value)
            # filtered_dict.update({key:class_prediction})
            
            filtered_dict.update({key:value})


    print("Thresholding Predictions...")
    print("Number of original predictions = ", len(model_predictions_dict))
    print("Number of filtered predictions = ", len(filtered_dict))

    save_name = "model_predictions_dict_" + str(slide_number) + "_filtered.pickle"

    # Save predictions
    with open(SAVED_DATABASE_DIR + save_name, 'wb') as handle:
        pickle.dump(filtered_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Make CSV file with model predictions
    utils.model_prediction_dict_to_csv(slide_number)


# In[8]:


def diagnose(slide_number):
    
    filename = "model_predictions_dict_" + str(slide_number) + "_filtered.pickle"

    # Load predictions
    with open(SAVED_DATABASE_DIR + filename, 'rb') as handle:
        filtered_predictions = pickle.load(handle)

    # slide diagnosis file (one file for all slides)
    open_name = "slide_dx_dict_" + str(int(PREDICTION_THRESHOLD*100)) + "% CONFIDENCE.pickle"

    if os.path.exists(SLIDE_DX_DIR + open_name):
        with open(SLIDE_DX_DIR + open_name, 'rb') as handle:
             slide_dx_dict = pickle.load(handle)
    else:
        slide_dx_dict = {}
    
    _1R1A_count = 0
    _1R2_count = 0
    healing_count = 0
    hemorrhage_count = 0
    normal_count = 0
    quilty_count = 0

    class_count = {"1R1A":0, "1R2":0, "Healing":0, "Hemorrhage":0, "Normal":0, "Quilty":0}

    for k, v in filtered_predictions.items():
        
        # get the class prediction 
        value = np.argmax(v)
    
        # 1R1A
        if value == 0:
            _1R1A_count += 1
            class_count.update({"1R1A":_1R1A_count})
        # 1R2
        elif value == 1:
            pass
        # Healing
        elif value == 2:
            healing_count += 1
            class_count.update({"Healing":healing_count})
        # Hemorrhage
        elif value == 3:
            hemorrhage_count += 1
            class_count.update({"Normal":normal_count})
        # Normal  
        elif value == 4:
            normal_count += 1
            class_count.update({"Quilty":quilty_count})
        # Quilty
        elif value == 5:
            quilty_count += 1
            class_count.update({"Quilty":quilty_count})
            
            
    # call special 1r2 count function
    _1R2_count = count_1r2.main(slide_number)

    
    class_count.update({"1R2":_1R2_count})
    print("class_counts = ", class_count)
    
    ########## Diagnose the slide #########
    dx = ""
     
    if _1R1A_count == 0  and _1R2_count == 0:
        dx = dx + "0R"
    elif _1R1A_count > 0 and _1R2_count == 0:
        dx = dx + "1R1A"
    elif _1R2_count > 0:
        if _1R2_count < 2:
            dx = dx + "1R2"
        else :
            dx = dx + "2R"
    ########################################
    
    print(f"Slide Diagnosis = {dx}")

    slide_dx_dict.update({slide_number:dx})


    save_name = "slide_dx_dict_" + str(int(PREDICTION_THRESHOLD*100)) + "% confidence.pickle"

    # Save diagnoses
    with open(SLIDE_DX_DIR + save_name, 'wb') as handle:
        pickle.dump(slide_dx_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    utils.slide_dx_to_csv(slide_dx_dict, save_name)


# In[9]:


def display_results(slides_to_process):

    open_name = open_name = "slide_dx_dict_" + str(int(PREDICTION_THRESHOLD*100)) + "% confidence.pickle"
    
    with open(SLIDE_DX_DIR + open_name, 'rb') as handle:
        slide_dx_dict = pickle.load(handle)
    
    ###### get rid of entries from deleted slides ######
    slide_dx_dict_copy = slide_dx_dict.copy()
    
    for key, value in slide_dx_dict.items():
        if key not in slides_to_process:
            del  slide_dx_dict_copy[key]
            
    slide_dx_dict = slide_dx_dict_copy
    #######################################################

    for key, value in slide_dx_dict.items():    
        # print(key, value)
        print(value)


# In[10]:


def main():
    
    t = time.time()

    # slides_to_process = utils.get_test_slide_numbers()

    slides_to_process = [111]

    print(slides_to_process)

    """ leave these out of the main program loop for multiprocessing at the slide level """
    slide.multiprocess_training_slides_to_images(image_num_list=slides_to_process) 
    filter.multiprocess_apply_filters_to_images(image_num_list=slides_to_process)
    tiles.multiprocess_filtered_images_to_tiles(save_top_tiles=False, image_num_list=slides_to_process)

    
    print(f"\nTime to extract training image, apply filters, and make tiles {time.time()-t}")


    for folder in slides_to_process:

        loop_timer = time.time()
        slide_number = folder
        

        print(f"\nStarting slide number {slide_number}")

        tileset_utils.process_tilesets_multiprocess(slide_number)
        filter_patches_multiprocess(slide_number)
        # classify_patches_batch(slide_number)
        # threshold_predictions(slide_number)
        # diagnose(slide_number)
        # annotate_png.main(slide_number)
        # annotate_svs.main(slide_number)

        print(f"\nDone processing slide {slide_number}. processing time: {time.time() - loop_timer}\n")

    print(f"Total processing time for all slides: {time.time() - t}")
    display_results(slides_to_process)

    


# In[11]:


# check that all directories exists
check_filesystem()

print("Confidence threshold = ", PREDICTION_THRESHOLD)

# Set up GPU for training
device = utils.initialize_gpu()

# Load trained model
model = torch.load(MODEL_DIR + "resnet50_ft")

#Send the model to GPU(s)
model = model.to(device)

# Set model to evaluation mode
model.eval()

if __name__ == "__main__": main()
    


# In[ ]:




