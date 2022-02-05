
import random
import glob
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import shutil
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from PIL import Image
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import time
from shapely.geometry import Polygon

ROOT_DIR = os.path.abspath("./")
DIR_TO_SAVE = "./TestMarkerResults/"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils




def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax






def predict(model, image_path, labels, max_score = 0.9):
    
    '''
    
        Predicting image from mrcnn model
        
        Input:
            model       : Mrcnn Model
            image_path. : Path of the image
            labels.     : Labels you used in training
            
    '''
    # Reading Image
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    
    # Detecting Model
    results = model.detect([image], verbose=0)
    
    # Post Processing Results
    r = results[0]
    classes = ["background"]
    classes += labels
    
    scores    = []
    mask      = []
    class_ids = []
    rois = []
    
    if r["scores"].shape[0] > 1:
    
        for index, i in enumerate(r["scores"]):
            if i >= max_score:
                
                scores.append(r["scores"][index])
                class_ids.append(r["class_ids"][index])
                mask.append(r["masks"][:,:,index])
                rois.append(r["rois"][index])
        
        
        
        
        if len(scores) > 0:
            r = dict({"scores" : np.asarray(scores), "class_ids" : np.asarray(class_ids), "masks" : np.asarray(mask), "rois" : np.asarray(rois)} )   
            r['masks'] = np.rollaxis((r['masks']), 0, 3)
        else:
            return image , dict({}), 0
    
    
    
    out = display_instances(image,r['rois'],r['masks'],r['class_ids'],classes,r['scores'])

    return out, r



def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image


def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

        
        
    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )
    return image
